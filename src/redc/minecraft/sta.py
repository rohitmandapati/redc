"""Static timing analysis of a :class:`MinecraftPhysicalDesign`.

The timing graph is built from the SAME :class:`~redc.minecraft.connectivity.CompiledWorld`
the simulator runs on, and every edge delay comes from
:mod:`redc.minecraft.timing` -- the definitions the simulator's behaviours
schedule with.  Nothing here reads backend net ids; annotations only label
the reported paths.

Nodes (one per element of the compiled world):

* ``net``    -- a dust network (zero delay inside, Java updates dust within one tick);
* ``source`` -- a repeater front, torch, lever, redstone block, abstract
  output pin or externally driven port bit;
* ``strong`` / ``power`` -- a conductor block's strong power (what dust reads)
  and total power (what mechanisms read);
* ``reader`` -- a device input (repeater back, torch attachment, lamp, abstract
  input pin, observed output bit).

Edges and their (min, max) delays:

* source -> net / strong, strong -> net / power, net -> power, net / power /
  source -> reader: 0;
* repeater reader -> repeater source: the configured delay; the MAX becomes
  :func:`~redc.minecraft.timing.repeater_settle_bound_gt` when the input may
  glitch (pulse extension);
* torch reader -> torch source: :data:`~redc.minecraft.timing.TORCH_DELAY_GT`;
* abstract input pin -> abstract output pin: the declared arc (min, max);
* a register's q is NOT connected to its d / clk: q is a launch point.

A node "may glitch" (change more than once per cycle) if a predecessor may,
or if two or more of its predecessors can change at all.  Register outputs
and input ports change at most once per cycle.

Analyses are separate forward propagations (latest = max, earliest = min,
ties broken toward the lowest node id, so equal builds report equal paths):

* clock: from the clock port at 0 -> every register clock pin;
* reset: from the reset port at 0 -> every register reset pin;
* data from registers: launched at ``clock arrival + clk->Q`` (min / max);
* data from input ports: launched at 0 (the test bench adds an input offset).

A clock that reaches data, data that reaches a clock pin, or a reset that
reaches data are structural timing errors, never silently analysed.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..parser import CompileError
from .connectivity import REF_BLOCK, REF_DUST, REF_SOURCE, CompiledWorld, compile_world
from .design import MinecraftPhysicalDesign
from .timing import (
    LAMP_OFF_DELAY_GT,
    LAMP_ON_DELAY_GT,
    TIMING_MODEL,
    TORCH_DELAY_GT,
    SequentialTiming,
    repeater_settle_bound_gt,
)
from .units import ticks_record

EDGE_WIRE = "wire"
EDGE_REPEATER = "repeater"
EDGE_TORCH = "torch"
EDGE_ARC = "cell_arc"
EDGE_LAMP = "lamp"


class TimingError(CompileError):
    """A structural timing problem (``code`` is a ``timing/...`` failure code)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class Edge:
    pred: int
    min_gt: int
    max_gt: int
    kind: str
    #: Repeater edges: the configured delay (the max depends on glitching).
    setting_rt: int = 0


@dataclass
class Arrival:
    """The result of one propagation."""

    earliest: list[int | None]
    latest: list[int | None]
    glitchy: list[bool]
    via_earliest: list[int | None]
    via_latest: list[int | None]
    #: Edge delay used on the latest / earliest incoming edge.
    used_latest: list[int]
    used_earliest: list[int]

    def reached(self, node: int) -> bool:
        return self.latest[node] is not None


@dataclass(frozen=True)
class Register:
    """A sequential component as STA sees it."""

    name: str
    timing: SequentialTiming
    data: int
    clock: int
    reset: int | None
    q: int
    labels: dict[str, Any]


@dataclass(frozen=True)
class Endpoint:
    """A data observation point: a register's data pin or an output bit."""

    node: int
    kind: str  # "register" | "output"
    name: str
    register: Register | None = None
    #: Added to the arrival at the node (a lamp's on / off delay).
    extra_min_gt: int = 0
    extra_max_gt: int = 0


@dataclass
class TimingGraph:
    """Nodes + incoming edges of a compiled world (see the module docstring)."""

    world: CompiledWorld
    kinds: list[str]
    labels: list[str]
    preds: list[list[Edge]]
    order: list[int]
    registers: list[Register]
    endpoints: list[Endpoint]
    clock_source: int | None
    reset_source: int | None
    input_sources: list[tuple[str, int, int]] = field(default_factory=list)
    _succ: list[list[int]] = field(default_factory=list)

    # -- node numbering -------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.kinds)

    def describe(self, node: int) -> dict[str, Any]:
        """A JSON record of one node for path reports (with annotations)."""
        w = self.world
        kind = self.kinds[node]
        record: dict[str, Any] = {"node": node, "kind": kind, "label": self.labels[node]}
        nets: set[Any] = set()
        annotations = w.design.annotations
        if kind == "net":
            network = node - self.base_net
            coords = [w.dust_coords[d] for d in w.components[network]]
            record["dust_blocks"] = len(coords)
            record["first"] = list(coords[0])
            record["last"] = list(coords[-1])
            nets.update(annotations[c] for c in coords if c in annotations)
        elif kind == "source":
            source = w.sources[node - self.base_source]
            device = w.devices[source.device]
            record["coord"] = list(source.coord)
            record["device"] = device.kind
            if device.block is not None and device.block.id == "minecraft:repeater":
                record["delay_rt"] = int(device.block.get("delay", 1))
            if device.component is not None:
                record["component"] = device.component.name
                record["pin"] = source.pin
                record["labels"] = dict(device.component.labels)
            if source.coord in annotations:
                nets.add(annotations[source.coord])
        elif kind == "reader":
            reader = w.readers[node - self.base_reader]
            device = w.devices[reader.device]
            record["coord"] = list(reader.coord)
            record["device"] = device.kind
            if device.component is not None:
                record["component"] = device.component.name
                record["pin"] = reader.pin
                record["labels"] = dict(device.component.labels)
            if reader.coord in annotations:
                nets.add(annotations[reader.coord])
        else:
            block = w.blocks[node - (self.base_strong if kind == "strong" else self.base_power)]
            record["coord"] = list(block.coord)
        if nets:
            record["nets"] = sorted(nets, key=str)
        return record

    base_net: int = 0
    base_source: int = 0
    base_strong: int = 0
    base_power: int = 0
    base_reader: int = 0

    # -- propagation ------------------------------------------------------------

    def propagate(self, launches: Mapping[int, tuple[int, int]]) -> Arrival:
        """Forward min/max arrival from ``launches`` (node -> (earliest, latest))."""
        n = self.size
        earliest: list[int | None] = [None] * n
        latest: list[int | None] = [None] * n
        glitchy = [False] * n
        via_e: list[int | None] = [None] * n
        via_l: list[int | None] = [None] * n
        used_e = [0] * n
        used_l = [0] * n
        for node, (lo, hi) in launches.items():
            earliest[node], latest[node] = lo, hi
        for node in self.order:
            if node in launches:
                continue
            best_e: int | None = None
            best_l: int | None = None
            active = 0
            any_glitch = False
            for edge in self.preds[node]:
                p = edge.pred
                lp = latest[p]
                if lp is None:
                    continue
                ep = earliest[p]
                assert ep is not None
                active += 1
                any_glitch = any_glitch or glitchy[p]
                hi = edge.max_gt
                if edge.kind == EDGE_REPEATER:
                    hi = repeater_settle_bound_gt(edge.setting_rt, single_transition=not glitchy[p])
                cand_l = lp + hi
                cand_e = ep + edge.min_gt
                if best_l is None or cand_l > best_l:
                    best_l, via_l[node], used_l[node] = cand_l, p, hi
                if best_e is None or cand_e < best_e:
                    best_e, via_e[node], used_e[node] = cand_e, p, edge.min_gt
            if best_l is not None:
                latest[node], earliest[node] = best_l, best_e
                glitchy[node] = any_glitch or active >= 2
        return Arrival(earliest, latest, glitchy, via_e, via_l, used_l, used_e)

    def path(self, arrival: Arrival, node: int, *, latest: bool = True) -> list[dict[str, Any]]:
        """The worst (latest or earliest) path ending at ``node``, launch first."""
        steps = []
        cursor: int | None = node
        seen = set()
        while cursor is not None and cursor not in seen:
            seen.add(cursor)
            record = self.describe(cursor)
            at = arrival.latest[cursor] if latest else arrival.earliest[cursor]
            record["arrival"] = ticks_record(at)
            record["edge_delay"] = ticks_record(arrival.used_latest[cursor] if latest else arrival.used_earliest[cursor])
            steps.append(record)
            cursor = arrival.via_latest[cursor] if latest else arrival.via_earliest[cursor]
        steps.reverse()
        if steps:
            steps[0]["edge_delay"] = ticks_record(0)
        return steps

    def reachable(self, start: int) -> set[int]:
        if not self._succ:
            succ: list[list[int]] = [[] for _ in range(self.size)]
            for node, edges in enumerate(self.preds):
                for e in edges:
                    succ[e.pred].append(node)
            self._succ = succ
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in self._succ[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen


def build_timing_graph(world: CompiledWorld | MinecraftPhysicalDesign) -> TimingGraph:
    """The timing graph of a compiled world (or of a design, compiled first)."""
    if isinstance(world, MinecraftPhysicalDesign):
        world = compile_world(world)
    w = world
    base_net = 0
    base_source = base_net + len(w.components)
    base_strong = base_source + len(w.sources)
    base_power = base_strong + len(w.blocks)
    base_reader = base_power + len(w.blocks)
    size = base_reader + len(w.readers)
    kinds = ["net"] * len(w.components) + ["source"] * len(w.sources) + ["strong"] * len(w.blocks)
    kinds += ["power"] * len(w.blocks) + ["reader"] * len(w.readers)
    labels: list[str] = []
    for c in w.components:
        labels.append(f"dust network at {list(w.dust_coords[c[0]])} ({len(c)} blocks)")
    for s in w.sources:
        dev = w.devices[s.device]
        labels.append(f"{dev.label}" + (f" pin {s.pin}" if s.pin else ""))
    for b in w.blocks:
        labels.append(f"strong power of block {list(b.coord)}")
    for b in w.blocks:
        labels.append(f"power of block {list(b.coord)}")
    for r in w.readers:
        dev = w.devices[r.device]
        labels.append(f"input of {dev.label}" + (f" pin {r.pin}" if r.pin else ""))
    preds: list[list[Edge]] = [[] for _ in range(size)]

    def wire(dst: int, src: int) -> None:
        preds[dst].append(Edge(src, 0, 0, EDGE_WIRE))

    for s in w.sources:
        for d in s.dust_out:
            wire(base_net + w.dust_component[d], base_source + s.index)
        for target in s.strong_out:
            wire(base_strong + target, base_source + s.index)
    for power in w.blocks:
        wire(base_power + power.index, base_strong + power.index)
        for d in power.dust_out:
            wire(base_net + w.dust_component[d], base_strong + power.index)
        for d in sorted({w.dust_component[d] for d in power.weak_in}):
            wire(base_power + power.index, base_net + d)
    for r in w.readers:
        for kind, index in r.refs:
            if kind == REF_DUST:
                wire(base_reader + r.index, base_net + w.dust_component[index])
            elif kind == REF_BLOCK:
                wire(base_reader + r.index, base_power + index)
            elif kind == REF_SOURCE:
                wire(base_reader + r.index, base_source + index)

    registers: list[Register] = []
    endpoints: list[Endpoint] = []
    for device in w.devices:
        if device.behavior is not None and device.block is not None and device.kind in ("repeater", "torch"):
            reader = base_reader + device.readers[0]
            source = base_source + device.sources[0]
            if device.kind == "repeater":
                setting = int(device.block.get("delay", 1))
                lo = device.static["delay_gt"]
                preds[source].append(Edge(reader, lo, lo, EDGE_REPEATER, setting))
            else:
                preds[source].append(Edge(reader, TORCH_DELAY_GT, TORCH_DELAY_GT, EDGE_TORCH))
        elif device.kind == "abstract":
            comp = device.component
            assert comp is not None
            reader_of = {w.readers[r].pin: base_reader + r for r in device.readers}
            source_of = {w.sources[s].pin: base_source + s for s in device.sources}
            if comp.function == "combinational":
                for arc in comp.timing.arcs:
                    preds[source_of[arc.to_pin]].append(Edge(reader_of[arc.from_pin], arc.min_gt, arc.max_gt, EDGE_ARC))
            else:
                seq = comp.timing.sequential
                assert seq is not None
                reg = Register(
                    comp.name, seq, reader_of[seq.data_pin], reader_of[seq.clock_pin],
                    reader_of[seq.reset_pin] if seq.reset_pin else None, source_of[seq.q_pin], dict(comp.labels),
                )  # fmt: skip
                registers.append(reg)
                endpoints.append(Endpoint(reg.data, "register", comp.name, reg))
    clock_source = reset_source = None
    input_sources: list[tuple[str, int, int]] = []
    for port in w.design.ports:
        for bit in range(port.width):
            device = w.devices[w.port_devices[(port.name, bit)]]
            if port.direction == "in":
                if not device.sources:
                    continue  # an open (unused) input bit
                node = base_source + device.sources[0]
                if port.role == "clock":
                    clock_source = node
                elif port.role == "reset":
                    reset_source = node
                else:
                    input_sources.append((port.name, bit, node))
            elif device.kind == "lamp":
                endpoints.append(
                    Endpoint(base_reader + device.readers[0], "output", f"{port.name}[{bit}]", None,
                             LAMP_ON_DELAY_GT, LAMP_OFF_DELAY_GT)
                )  # fmt: skip
            else:
                endpoints.append(Endpoint(base_reader + device.readers[0], "output", f"{port.name}[{bit}]"))
    for edges in preds:
        edges.sort(key=lambda e: (e.pred, e.kind))
    # Topological order (Kahn, lowest node first).
    indegree = [len(p) for p in preds]
    succ: list[list[int]] = [[] for _ in range(size)]
    for node, edges in enumerate(preds):
        for e in edges:
            succ[e.pred].append(node)
    ready = [node for node in range(size) if indegree[node] == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for nxt in succ[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(ready, nxt)
    if len(order) != size:
        stuck = [labels[n] for n in range(size) if indegree[n] > 0][:6]
        raise TimingError("timing/combinational_loop", f"the circuit has a loop without a register: {'; '.join(stuck)}")
    graph = TimingGraph(w, kinds, labels, preds, order, registers, endpoints, clock_source, reset_source, input_sources)
    graph._succ = succ
    graph.base_net, graph.base_source, graph.base_strong = base_net, base_source, base_strong
    graph.base_power, graph.base_reader = base_power, base_reader
    return graph


@dataclass
class TimingAnalysis:
    """Clock, reset and data arrivals of one design (no period chosen yet)."""

    graph: TimingGraph
    fingerprint: str
    clock: Arrival | None
    reset: Arrival | None
    from_registers: Arrival
    from_inputs: Arrival

    @property
    def registers(self) -> list[Register]:
        return self.graph.registers

    @property
    def sequential(self) -> bool:
        return bool(self.graph.registers)

    def clock_arrival(self, reg: Register) -> tuple[int, int] | None:
        if self.clock is None or not self.clock.reached(reg.clock):
            return None
        lo, hi = self.clock.earliest[reg.clock], self.clock.latest[reg.clock]
        assert lo is not None and hi is not None
        return lo, hi


def analyze(design: MinecraftPhysicalDesign, *, world: CompiledWorld | None = None) -> TimingAnalysis:
    """Clock, reset and data propagation (see the module docstring).  Raises
    :class:`TimingError` for structural problems."""
    graph = build_timing_graph(world if world is not None else compile_world(design))
    registers = graph.registers
    clock = reset = None
    if registers:
        if graph.clock_source is None:
            raise TimingError("timing/clock_unreached", "the design has registers but no clock port")
        clock = graph.propagate({graph.clock_source: (0, 0)})
        unreached = [r.name for r in registers if not clock.reached(r.clock)]
        if unreached:
            raise TimingError("timing/clock_unreached", f"the clock never reaches {', '.join(unreached[:8])}")
        leaks = [e.name for e in graph.endpoints if clock.reached(e.node)]
        leaks += [r.name for r in registers if r.reset is not None and clock.reached(r.reset)]
        if leaks:
            raise TimingError("timing/clock_leak", f"the clock network reaches data / reset inputs: {', '.join(leaks[:8])}")
        glitchy = [r.name for r in registers if clock.glitchy[r.clock]]
        if glitchy:
            raise TimingError("timing/clock_glitch", f"the clock may glitch at {', '.join(glitchy[:8])}")
        if graph.reset_source is not None:
            reset = graph.propagate({graph.reset_source: (0, 0)})
            leaks = [e.name for e in graph.endpoints if reset.reached(e.node)]
            leaks += [r.name for r in registers if reset.reached(r.clock)]
            if leaks:
                raise TimingError("timing/reset_leak", f"the reset network reaches data / clock inputs: {', '.join(leaks[:8])}")
    launches: dict[int, tuple[int, int]] = {}
    for reg in registers:
        assert clock is not None
        lo, hi = clock.earliest[reg.clock], clock.latest[reg.clock]
        assert lo is not None and hi is not None
        launches[reg.q] = (lo + reg.timing.clk_to_q_min_gt, hi + reg.timing.clk_to_q_max_gt)
    from_registers = graph.propagate(launches)
    from_inputs = graph.propagate({node: (0, 0) for _n, _b, node in graph.input_sources})
    for arrival, what in ((from_registers, "register outputs"), (from_inputs, "data inputs")):
        bad = [r.name for r in registers if arrival.reached(r.clock)]
        if bad:
            raise TimingError("timing/data_reaches_clock", f"{what} reach the clock pin of {', '.join(bad[:8])}")
        bad = [r.name for r in registers if r.reset is not None and arrival.reached(r.reset)]
        if bad:
            raise TimingError(
                "timing/data_reaches_reset",
                f"{what} reach the asynchronous reset of {', '.join(bad[:8])} (reset is not data)",
            )
    return TimingAnalysis(graph, design.fingerprint(), clock, reset, from_registers, from_inputs)


def model_record() -> dict[str, Any]:
    return {"model": TIMING_MODEL, "units": "game ticks (gt); rt = redstone ticks = 2 gt"}


def latest_of(arrival: Arrival, node: int) -> int | None:
    return arrival.latest[node]


def earliest_of(arrival: Arrival, node: int) -> int | None:
    return arrival.earliest[node]


def sorted_endpoints(endpoints: Sequence[Endpoint]) -> list[Endpoint]:
    return sorted(endpoints, key=lambda e: (e.kind, e.name, e.node))


__all__ = [
    "Arrival",
    "Edge",
    "Endpoint",
    "Register",
    "TimingAnalysis",
    "TimingError",
    "TimingGraph",
    "analyze",
    "build_timing_graph",
    "model_record",
    "sorted_endpoints",
]
