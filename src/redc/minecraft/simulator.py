"""``RedstoneSimulator``: an event-driven, deterministic redstone simulator
(``redc-redstone-sim-v1``) for :class:`~redc.minecraft.design.MinecraftPhysicalDesign`.

Nothing is simulated per tick or per block.  Time only advances to the next
scheduled event:

    priority queue of (game tick, phase, sequence number, event)

* phase 0 -- external stimulus (port changes from the test bench);
* phase 1 -- scheduled device activity (repeater / torch / lamp ticks,
  abstract component evaluations), in the order it was scheduled.

Processing one event changes some source (a repeater turns on, a lever
flips, an abstract output rises).  Its effects then settle INSTANTLY within the
same game tick, in a fixed order that is a DAG by construction:

    changed sources -> strong block power -> dust networks (recomputed whole,
    by strength-descending propagation) -> weak block power -> readers

and a reader whose input changed notifies its device, which may schedule a
FUTURE event (every device delay is >= 1 game tick, so there are no
zero-delay loops).  Sets are always visited in sorted id order and ids are
sorted coordinates, so identical design + stimulus give an identical event
sequence -- no hash-order dependence anywhere.

What is modelled is documented in :mod:`redc.minecraft.behaviors`; what is
NOT (tick priorities, update order inside one tick, comparators, pistons,
...) is listed in :data:`~redc.minecraft.behaviors.UNSUPPORTED_MECHANICS`
and reported with every result.  RedC's technology cells are meant to avoid
all of it; the simulator never silently approximates it.

Usage::

    sim = RedstoneSimulator(design)
    sim.set_input("a", 1)
    sim.run_until_stable()
    sim.read_output("y")

    sim.schedule_input("clk", 1, at_tick=20)   # game ticks
    sim.run_until(100)                          # every event before tick 100
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Mapping
from typing import Any

from .abstract import ADAPTERS, CORNERS, AbstractAdapter, Capture, TimingViolation
from .behaviors import (
    BURNOUT_TOGGLES,
    BURNOUT_WINDOW_GT,
    UNSUPPORTED_MECHANICS,
    Diagnostic,
)
from .blocks import Coord
from .connectivity import (
    REF_BLOCK,
    REF_DUST,
    CompiledWorld,
    SimulationError,
    compile_world,
)
from .design import MinecraftPhysicalDesign

SIMULATION_MODEL = "redc-redstone-sim-v1"

PHASE_INPUT = 0
PHASE_DEVICE = 1

_EV_INPUT = 0
_EV_TICK = 1
_EV_ABSTRACT = 2

#: Trace levels: ``off``; ``events`` (devices, ports, registers); ``full`` (+ every dust change).
TRACE_LEVELS = ("off", "events", "full")


class RedstoneSimulator:
    """Simulates one design (see the module docstring)."""

    def __init__(
        self,
        design: MinecraftPhysicalDesign,
        *,
        trace: str = "off",
        corner: str = "max",
        world: CompiledWorld | None = None,
        max_events: int = 50_000_000,
    ) -> None:
        if trace not in TRACE_LEVELS:
            raise ValueError(f"trace must be one of {TRACE_LEVELS}")
        if corner not in CORNERS:
            raise ValueError(f"corner must be one of {CORNERS}")
        self.design = design
        self.world = world if world is not None else compile_world(design)
        self.trace_level = trace
        self.corner = corner
        self.max_events = max_events
        w = self.world
        self.devices = w.devices
        self.time = 0
        self.dust_power = [0] * len(w.dust_coords)
        self.source_out = [0] * len(w.sources)
        self.block_strong = [0] * len(w.blocks)
        self.block_weak = [0] * len(w.blocks)
        self.reader_value = [0] * len(w.readers)
        self.device_state = [False] * len(w.devices)
        self.events: list[dict[str, Any]] = []
        self.warnings: list[Diagnostic] = []
        self.violations: list[TimingViolation] = []
        self.captures: list[Capture] = []
        self.stats: dict[str, int] = {
            "events": 0,
            "input_events": 0,
            "device_ticks": 0,
            "abstract_events": 0,
            "dust_network_updates": 0,
            "dust_changes": 0,
            "peak_queue": 0,
        }
        self._queue: list[tuple[int, int, int, int, Any]] = []
        self._seq = 0
        self._pending_tick: set[int] = set()
        self._dirty_sources: list[int] = []
        self._toggles: dict[int, deque[int]] = {}
        self._adapters: dict[int, AbstractAdapter] = {}
        #: Last observed bit of every observer / probe device.
        self._observed: dict[int, int] = {}
        for device in w.devices:
            if device.kind == "abstract":
                assert device.component is not None
                self._adapters[device.index] = ADAPTERS[device.component.function](self, device.index, device.component)
        self._power_on()

    # -- information --------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self.design.mode

    @property
    def pending_events(self) -> int:
        return len(self._queue)

    def is_stable(self) -> bool:
        """No event is scheduled: nothing will ever change without new stimulus."""
        return not self._queue

    def next_event_time(self) -> int | None:
        return self._queue[0][0] if self._queue else None

    def report(self) -> dict[str, Any]:
        return {
            "model": SIMULATION_MODEL,
            "mode": self.mode,
            "minecraft_version": self.design.minecraft_version,
            "corner": self.corner,
            "time_gt": self.time,
            "world": self.world.stats(),
            "stats": dict(self.stats),
            "warnings": [w.to_dict() for w in self.warnings[:50]],
            "violations": [v.to_dict() for v in self.violations[:50]],
            "unsupported": list(UNSUPPORTED_MECHANICS),
        }

    # -- device API (used by behaviours and abstract adapters) ---------------------

    def device_on(self, device: int) -> bool:
        return self.device_state[device]

    def device_input(self, device: int) -> int:
        return self.reader_value[self.devices[device].readers[0]]

    def set_device_output(self, device: int, on: bool) -> None:
        self.device_state[device] = on
        for source in self.devices[device].sources:
            self.set_source(source, on)

    def set_source(self, source: int, on: bool) -> None:
        strength = self.world.sources[source].on_strength if on else 0
        if self.source_out[source] != strength:
            self.source_out[source] = strength
            self._dirty_sources.append(source)

    def has_pending_tick(self, device: int) -> bool:
        return device in self._pending_tick

    def schedule_tick(self, device: int, delay_gt: int) -> None:
        if delay_gt < 1:
            raise SimulationError("simulation/zero_delay", f"device {self.devices[device].label} scheduled a zero-delay tick")
        self._pending_tick.add(device)
        self._push(self.time + delay_gt, PHASE_DEVICE, _EV_TICK, device)

    def schedule_abstract(self, device: int, delay_gt: int, payload: Any) -> None:
        self._push(self.time + delay_gt, PHASE_DEVICE, _EV_ABSTRACT, (device, payload))

    def record(self, type: str, **data: Any) -> None:
        if self.trace_level == "off":
            return
        coord = data.get("coord")
        if coord is not None:
            data["coord"] = list(coord)
        self.events.append({"t": self.time, "type": type, **data})

    def warn(self, code: str, message: str, coord: Coord | None = None) -> None:
        self.warnings.append(Diagnostic(code, f"tick {self.time}: {message}", coord))
        self.record("warning", code=code, message=message)

    def violation(self, violation: TimingViolation) -> None:
        self.violations.append(violation)
        self.record("timing_violation", **violation.to_dict())

    def capture(self, capture: Capture) -> None:
        self.captures.append(capture)
        self.record("register_capture", component=capture.component, value=capture.value, q_at=capture.q_at_gt)

    def note_torch_toggle(self, device: int) -> None:
        recent = self._toggles.setdefault(device, deque())
        recent.append(self.time)
        while recent and recent[0] <= self.time - BURNOUT_WINDOW_GT:
            recent.popleft()
        if len(recent) >= BURNOUT_TOGGLES:
            raise SimulationError(
                "simulation/unstable",
                f"{self.devices[device].label} toggled {len(recent)} times within {BURNOUT_WINDOW_GT} game ticks: "
                "Java would burn it out (torch burnout is not modelled)",
            )

    # -- stimulus ----------------------------------------------------------------

    def set_input(self, name: str, value: int) -> None:
        """Change input port ``name`` now (takes effect when the simulation runs)."""
        self.schedule_input(name, value, at_tick=self.time)

    def schedule_input(self, name: str, value: int, *, at_tick: int) -> None:
        """Change input port ``name`` to ``value`` at game tick ``at_tick``."""
        port = self.design.port(name)
        if port.direction != "in":
            raise SimulationError("simulation/bad_binding", f"port {name!r} is an output")
        if at_tick < self.time:
            raise SimulationError("simulation/bad_stimulus", f"cannot schedule {name} at tick {at_tick} < now {self.time}")
        bits = value & ((1 << port.width) - 1)
        self._push(at_tick, PHASE_INPUT, _EV_INPUT, (name, bits))

    # -- running -----------------------------------------------------------------

    def step(self) -> int | None:
        """Process every event of the next scheduled game tick; returns that
        tick (``None`` if nothing is scheduled)."""
        if not self._queue:
            return None
        now = self._queue[0][0]
        self.time = now
        queue = self._queue
        while queue and queue[0][0] == now:
            _t, _phase, _seq, kind, payload = heapq.heappop(queue)
            self.stats["events"] += 1
            if self.stats["events"] > self.max_events:
                raise SimulationError("simulation/unstable", f"more than {self.max_events} events")
            if kind == _EV_TICK:
                self.stats["device_ticks"] += 1
                self._pending_tick.discard(payload)
                device = self.devices[payload]
                assert device.behavior is not None
                device.behavior.on_tick(self, payload)
            elif kind == _EV_ABSTRACT:
                self.stats["abstract_events"] += 1
                index, data = payload
                self._adapters[index].on_event(data)
            else:
                self.stats["input_events"] += 1
                self._apply_input(*payload)
            self._settle()
        return now

    def run_until(self, tick: int) -> None:
        """Process every event scheduled strictly BEFORE game tick ``tick``,
        then stand at ``tick`` (its own events have not happened yet)."""
        while self._queue and self._queue[0][0] < tick:
            self.step()
        self.time = max(self.time, tick)

    def run_until_stable(self, *, max_ticks: int = 100_000) -> int:
        """Run until no event is scheduled; returns the time of the last
        event.  Raises ``simulation/unstable`` if activity continues beyond
        ``max_ticks`` game ticks from now (e.g. an oscillator)."""
        deadline = self.time + max_ticks
        last = self.time
        while self._queue:
            if self._queue[0][0] > deadline:
                pending = sorted({self.devices[p].label for _t, _ph, _s, k, p in self._queue if k == _EV_TICK})[:5]
                raise SimulationError(
                    "simulation/unstable",
                    f"still active after {max_ticks} game ticks ({len(self._queue)} pending events"
                    + (f", e.g. {'; '.join(pending)}" if pending else "")
                    + ")",
                )
            stepped = self.step()
            assert stepped is not None
            last = stepped
        self.record("stable")
        return last

    def evaluate(self, *, max_ticks: int = 100_000, **inputs: int) -> dict[str, int]:
        """Combinational convenience: apply ``inputs`` now, settle, read every output."""
        for name, value in inputs.items():
            self.set_input(name, value)
        self.run_until_stable(max_ticks=max_ticks)
        return self.read_outputs()

    # -- observation -------------------------------------------------------------

    def _bit(self, device: int) -> int:
        dev = self.devices[device]
        if dev.kind in ("observer", "probe"):
            return 1 if self.reader_value[dev.readers[0]] >= dev.static["threshold"] else 0
        return 1 if self.device_state[device] else 0

    def read_output(self, name: str) -> int:
        """Current value of port ``name`` (signed ports are read two's complement)."""
        port = self.design.port(name)
        raw = 0
        for index in range(port.width):
            raw |= self._bit(self.world.port_devices[(name, index)]) << index
        if port.signed and raw >> (port.width - 1):
            raw -= 1 << port.width
        return raw

    def read_outputs(self) -> dict[str, int]:
        return {p.name: self.read_output(p.name) for p in self.design.outputs}

    def probe(self, name: str) -> int:
        return self._bit(self.world.probe_devices[name])

    def probes(self) -> dict[str, int]:
        return {name: self._bit(device) for name, device in self.world.probe_devices.items()}

    def dust_at(self, coord: Coord) -> int:
        """Signal strength of the dust at ``coord``."""
        return self.dust_power[self.world.dust_index[coord]]

    def device_at(self, coord: Coord) -> bool:
        """On-state of the block device (repeater, torch, lamp, lever) at ``coord``."""
        for device in self.devices:
            if device.coord == coord and device.behavior is not None:
                return self.device_state[device.index]
        raise KeyError(coord)

    def snapshot(self) -> tuple[Any, ...]:
        """The complete dynamic state (for determinism checks)."""
        return (
            self.time,
            tuple(self.dust_power),
            tuple(self.source_out),
            tuple(self.block_strong),
            tuple(self.block_weak),
            tuple(self.reader_value),
            tuple(self.device_state),
            tuple(sorted((t, ph, s, k, repr(p)) for t, ph, s, k, p in self._queue)),
        )

    # -- internals ----------------------------------------------------------------

    def _push(self, time: int, phase: int, kind: int, payload: Any) -> None:
        heapq.heappush(self._queue, (time, phase, self._seq, kind, payload))
        self._seq += 1
        self.stats["peak_queue"] = max(self.stats["peak_queue"], len(self._queue))

    def _power_on(self) -> None:
        """Cold start at tick 0: everything off, then constants turn on and
        every device looks at its inputs once."""
        for device in self.devices:
            if device.behavior is not None and device.block is not None and device.behavior.initially_on(device.block):
                self.set_device_output(device.index, True)
        for adapter in self._adapters.values():
            adapter.power_on()
        self._settle(force_all=True)

    def _apply_input(self, name: str, bits: int) -> None:
        port = self.design.port(name)
        for index in range(port.width):
            device = self.world.port_devices[(name, index)]
            on = bool((bits >> index) & 1)
            if self.device_state[device] != on:
                self.set_device_output(device, on)
                self.record("input_changed", port=name, bit=index, value=int(on))
                if port.role == "clock":
                    self.record("clock_edge", port=name, edge="rise" if on else "fall")

    def _read(self, reader: int) -> int:
        best = 0
        for kind, index in self.world.readers[reader].refs:
            if kind == REF_DUST:
                value = self.dust_power[index]
            elif kind == REF_BLOCK:
                value = max(self.block_strong[index], self.block_weak[index])
            else:
                value = self.source_out[index]
            best = max(best, value)
        return best

    def _recompute(self, component: int) -> list[int]:
        """Settle one dust network; returns the dust ids whose power changed."""
        w = self.world
        members = w.components[component]
        source_out, block_strong = self.source_out, self.block_strong
        level: dict[int, int] = {}
        buckets: list[list[int]] = [[] for _ in range(16)]
        for d in members:
            best = 0
            for s in w.dust_sources[d]:
                best = max(best, source_out[s])
            for b in w.dust_blocks[d]:
                best = max(best, block_strong[b])
            level[d] = best
            if best > 1:
                buckets[best].append(d)
        links_out = w.dust_links_out
        for strength in range(15, 1, -1):
            for d in buckets[strength]:
                if level[d] != strength:
                    continue
                weaker = strength - 1
                for e in links_out[d]:
                    if level[e] < weaker:
                        level[e] = weaker
                        buckets[weaker].append(e)
        changed = [d for d in members if level[d] != self.dust_power[d]]
        for d in changed:
            self.dust_power[d] = level[d]
        self.stats["dust_network_updates"] += 1
        self.stats["dust_changes"] += len(changed)
        if changed and self.trace_level == "full":
            self.events.append(
                {
                    "t": self.time,
                    "type": "wire_strength_changed",
                    "changes": [[*w.dust_coords[d], level[d]] for d in changed],
                }
            )
        return changed

    def _settle(self, *, force_all: bool = False) -> None:
        """Propagate every pending source change through this game tick."""
        w = self.world
        while self._dirty_sources or force_all:
            dirty_blocks: set[int] = set()
            dirty_networks: set[int] = set()
            dirty_readers: set[int] = set()
            if force_all:
                dirty_blocks.update(range(len(w.blocks)))
                dirty_networks.update(range(len(w.components)))
            for s in self._dirty_sources:
                source = w.sources[s]
                dirty_blocks.update(source.strong_out)
                for d in source.dust_out:
                    dirty_networks.add(w.dust_component[d])
                dirty_readers.update(source.readers_out)
            self._dirty_sources.clear()
            for b in sorted(dirty_blocks):
                block = w.blocks[b]
                strong = max((self.source_out[s] for s in block.strong_in), default=0)
                if strong != self.block_strong[b]:
                    self.block_strong[b] = strong
                    for d in block.dust_out:
                        dirty_networks.add(w.dust_component[d])
                    dirty_readers.update(block.readers_out)
            weak_blocks: set[int] = set(range(len(w.blocks))) if force_all else set()
            for c in sorted(dirty_networks):
                for d in self._recompute(c):
                    weak_blocks.update(w.dust_weak_out[d])
                    dirty_readers.update(w.dust_readers[d])
            for b in sorted(weak_blocks):
                block = w.blocks[b]
                weak = max((self.dust_power[d] for d in block.weak_in), default=0)
                if weak != self.block_weak[b]:
                    self.block_weak[b] = weak
                    dirty_readers.update(block.readers_out)
            if force_all:
                # Power-on: every device evaluates its input once.
                for r in range(len(w.readers)):
                    self.reader_value[r] = self._read(r)
                for r in range(len(w.readers)):
                    self._notify(r, self.reader_value[r])
                force_all = False
                continue
            for r in sorted(dirty_readers):
                value = self._read(r)
                if value != self.reader_value[r]:
                    self.reader_value[r] = value
                    self._notify(r, value)

    def _notify(self, reader: int, value: int) -> None:
        r = self.world.readers[reader]
        device = self.devices[r.device]
        if device.kind == "abstract":
            assert r.pin is not None
            self._adapters[device.index].on_input(r.pin, value)
        elif device.behavior is not None:
            device.behavior.on_input(self, device.index, value)
        elif device.kind in ("observer", "probe"):
            bit = 1 if value >= device.static["threshold"] else 0
            if 0 < value < device.static["threshold"]:
                self.warn("simulation/weak_input", f"{device.label} sees strength {value}", device.coord)
            if self._observed.get(device.index, 0) != bit:
                self._observed[device.index] = bit
                self.record("output_changed" if device.kind == "observer" else "probe_changed",
                            target=device.label, value=bit)  # fmt: skip


def simulate_combinational(
    design: MinecraftPhysicalDesign, inputs: Mapping[str, int], *, max_ticks: int = 100_000
) -> dict[str, int]:
    """One fresh simulation: apply ``inputs`` at tick 0, settle, read the outputs."""
    return RedstoneSimulator(design).evaluate(max_ticks=max_ticks, **inputs)


__all__ = [
    "PHASE_DEVICE",
    "PHASE_INPUT",
    "SIMULATION_MODEL",
    "TRACE_LEVELS",
    "RedstoneSimulator",
    "simulate_combinational",
]
