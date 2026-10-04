"""Primitive place-and-route at block resolution: placement, redstone-aware A*,
route trees, negotiated congestion, electrical legalization and retries.

End to end, every successful design is re-checked by :func:`assert_legal_design`,
an INDEPENDENT test-local re-derivation of the conservative redstone model
(:mod:`redc.physical_primitive.redstone`, rules 1-6) from the final artifacts
alone -- placed cells, route trees and realized routes -- before the backend's
own :func:`verify_design` is asked to agree.  Unit tests drive
:func:`search_branch`, :func:`validate_branch`, :func:`legalize_route` and the
:class:`NegotiatedRedstoneRouter` on hand-built :class:`BlockGrid` s, pin sites
and :class:`RouteTree` s.  Conversely, legal designs are mutated one rule at a
time (a flipped repeater, a missing support, a short, ...) and
:func:`verify_design` must REPORT each broken rule rather than pass or crash.

Designs come from ``compile_source`` and use pad ports (no peripherals) unless a
test says otherwise.  Everything is deterministic: there is no randomness here.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import pairwise

import pytest

from redc import CompileError, Graph, compile_source
from redc.physical_primitive import (
    DEFAULT_INTERFACE_POLICY,
    GATE_KINDS,
    PHYSICAL_SCHEMA,
    PRIMITIVE_TECHNOLOGY,
    BitTerminal,
    InterfacePolicy,
    PadInterfacePolicy,
    PrimitiveKind,
    PrimitiveNetlist,
    PrimitivePhysicalNetlist,
    PrimitivePnRConfig,
    PrimitivePnRResult,
    PrimitiveTechnologyLibrary,
    PrimitiveTraceRecorder,
    Provenance,
    map_primitives_to_minecraft,
    place_and_route_graph,
    place_and_route_primitive,
    synthesize_to_primitives,
)
from redc.physical_primitive.geometry import Bounds, Coord, Direction, below
from redc.physical_primitive.grid import BlockGrid, PinSite
from redc.physical_primitive.pnr import (
    LegalizationFailure,
    NegotiatedRedstoneRouter,
    NetState,
    PlacementError,
    PrimitivePnRError,
    RealizedRoute,
    RouteBranch,
    RouteRequest,
    RouteTree,
    TraceLevel,
    legalize_route,
    placement_columns,
    search_branch,
    validate_branch,
    verify_design,
)
from redc.physical_primitive.redstone import (
    MAX_SIGNAL_STRENGTH,
    MIN_SIGNAL_Y,
    REPEATER_DELAY_TICKS,
    ElementKind,
    clearance_of,
    is_move,
    neighborhood,
    support_of,
)

EAST, SOUTH, WEST, NORTH = Direction.EAST, Direction.SOUTH, Direction.WEST, Direction.NORTH
PADS = PadInterfacePolicy()

SOURCES = {
    "not": "bool main(bool a) { return !a; }",
    "and": "bool main(bool a, bool b) { return a && b; }",
    "half_adder": "uint2 main(bool a, bool b) { return (uint2)a + (uint2)b; }",
    "full_adder": "uint2 main(bool a, bool b, bool c) { return (uint2)a + (uint2)b + (uint2)c; }",
    "uint4_add": "uint4 main(uint4 a, uint4 b) { return a + b; }",
    # `s` drives NOT(s) and four mux ANDs: one five-sink net.
    "fanout_mux": "uint4 main(bool s, uint4 a) { return s ? a : ~a; }",
    # four unrelated AND gates: eight parallel input wires, four output wires.
    "parallel_and": "uint4 main(uint4 a, uint4 b) { return a & b; }",
    # operands swapped against the pad order: the two input wires must cross.
    "swapped_and": "bool main(bool a, bool b) { return b && a; }",
    "swapped_xor": "uint2 main(uint2 a, uint2 b) { return b ^ a; }",
    # registers, an enable mux, the global clock and reset.
    "counter": "uint2 main(uint2 n) { uint2 x = 0; for (uint2 i = 0; i < n; i++) { x = x + 1; } return x; }",
    # `result : uint8` becomes ONE display peripheral under the default interface.
    "passthrough8": "uint8 main(uint8 a) { return a; }",
    # two `const` IR nodes, each feeding one AND per bit
    "masked_or": "uint4 main(uint4 a, uint4 b) { return (a & 5) | (b & 10); }",
}


@dataclass(frozen=True)
class Run:
    graph: Graph
    netlist: PrimitiveNetlist
    mapped: PrimitivePhysicalNetlist
    result: PrimitivePnRResult
    interface: InterfacePolicy
    library: PrimitiveTechnologyLibrary


def run_pnr(
    name: str,
    *,
    level: str = "basic",
    interface: InterfacePolicy = PADS,
    library: PrimitiveTechnologyLibrary = PRIMITIVE_TECHNOLOGY,
    **config,
) -> Run:
    graph = compile_source(SOURCES[name])
    netlist, mapped, result = place_and_route_graph(
        graph, PrimitivePnRConfig(**config), interface=interface, library=library,
        trace=PrimitiveTraceRecorder(level),
    )  # fmt: skip
    return Run(graph, netlist, mapped, result, interface, library)


@functools.cache
def routed(name: str, **config) -> Run:
    """``run_pnr`` memoized per (design, config): P&R results are read-only here."""
    return run_pnr(name, **config)


# -- the independent legality checker ------------------------------------------------


def step(a: Coord, b: Coord) -> tuple[int, int]:
    """The horizontal unit step from ``a`` to ``b``."""
    return (b[0] - a[0], b[2] - a[2])


def heading(direction: Direction) -> tuple[int, int]:
    return (direction.vector[0], direction.vector[2])


def tree_parents(tree: RouteTree) -> dict[Coord, Coord | None]:
    """Rebuild a tree's parent map from its ORDERED branch paths, checking that
    each branch starts on the tree built so far, steps legally and only adds
    new blocks (so the union really is a tree rooted at the driver pin)."""
    parent: dict[Coord, Coord | None] = {tree.root: None}
    for branch in tree.branches:
        assert branch.path[0] in parent, f"net {tree.net}: branch to {branch.goal} starts off the tree"
        for a, b in pairwise(branch.path):
            assert is_move(a, b), f"net {tree.net}: illegal step {a} -> {b}"
            assert b not in parent, f"net {tree.net}: branch re-enters the tree at {b}"
            parent[b] = a
    return parent


def children_of(parent: dict[Coord, Coord | None]) -> dict[Coord, list[Coord]]:
    children: dict[Coord, list[Coord]] = {cell: [] for cell in parent}
    for cell, up in parent.items():
        if up is not None:
            children[up].append(cell)
    return children


def depth_of(tree: RouteTree) -> dict[Coord, int]:
    """Steps from the root to every block of ``tree``."""
    depth: dict[Coord, int] = {}
    for cell in tree.cells:
        up = tree.parent[cell]
        depth[cell] = 0 if up is None else depth[up] + 1
    return depth


def signal_levels(root: Coord, children, kinds: dict[Coord, ElementKind], drive: int) -> dict[Coord, int]:
    """Signal strength of every dust block (a repeater: its INPUT strength),
    walked breadth-first from the root -- independent of construction order."""
    level = {root: drive}
    queue = deque([root])
    while queue:
        cell = queue.popleft()
        for child in children[cell]:
            if kinds[cell] is ElementKind.REPEATER:
                level[child] = MAX_SIGNAL_STRENGTH if level[cell] >= 1 else 0
            elif kinds[child] is ElementKind.REPEATER:
                level[child] = level[cell]
            else:
                level[child] = max(0, level[cell] - 1)
            queue.append(child)
    return level


def assert_repeater_legal(cell: Coord, element, parent, children, pin_blocks) -> None:
    """Rule 5: a straight, level segment -- parent behind, exactly one child in
    front, same y, never a pin -- facing the child (its output side)."""
    up, kids = parent[cell], children[cell]
    assert cell not in pin_blocks, f"repeater on pin block {cell}"
    assert up is not None and len(kids) == 1, f"repeater {cell} is not on a single-file segment"
    (down,) = kids
    assert up[1] == cell[1] == down[1], f"repeater {cell} is not level"
    assert step(up, cell) == step(cell, down), f"repeater {cell} sits on a turn"
    assert element.facing is not None and element.facing.vector == (down[0] - cell[0], 0, down[2] - cell[2])


def assert_legal_placement(mapped: PrimitivePhysicalNetlist, max_y: int):
    """No voxel collisions, no body in a keep-out, pins free and resting on their cell."""
    body: dict[Coord, int] = {}
    keepout: dict[Coord, set[int]] = defaultdict(set)
    pins: dict[Coord, tuple[int, str]] = {}
    for inst in mapped.instances:
        assert inst.is_placed and inst.orientation is not None, f"instance {inst.id} is unplaced"
        placed = inst.placed
        assert placed.origin == inst.origin
        for cell in placed.occupied:
            assert cell not in body, f"instances {body[cell]} and {inst.id} collide at {cell}"
            assert 0 <= cell[1] <= max_y, f"instance {inst.id} voxel {cell} outside the height limit"
            body[cell] = inst.id
        for cell in placed.keepout:
            keepout[cell].add(inst.id)
        for name, pin in placed.pins.items():
            assert pin.position not in pins, f"pins {pins[pin.position]} and {(inst.id, name)} coincide"
            pins[pin.position] = (inst.id, name)
            assert below(pin.position) in placed.occupied, f"pin {(inst.id, name)} does not rest on its cell"
    for cell, owner in body.items():
        assert not keepout.get(cell), f"instance {owner} sits in the keep-out of {keepout[cell]} at {cell}"
    for cell, ref in pins.items():
        assert cell not in body and not keepout.get(cell), f"pin {ref} at {cell} is covered"
    return body, dict(keepout), pins


def assert_legal_design(run: Run) -> None:
    """Re-derive EVERY rule of the redstone model from the final artifacts."""
    result, mapped = run.result, run.mapped
    assert result.success, result.failure
    max_y = result.geometry.max_y
    routes, realized, grid = result.routes, result.realized, result.grid
    assert grid is not None and result.routed is not None and result.legalized is not None
    body, keepout, pins = assert_legal_placement(mapped, max_y)
    assert grid.body == body and set(grid.keepout) == set(keepout) and set(grid.pins) == set(pins)
    sites = {(inst.id, name): pin for inst in mapped.instances for name, pin in inst.placed.pins.items()}
    assert sorted(routes) == sorted(realized) == [net.id for net in mapped.nets]

    signal: dict[Coord, int] = {}
    support: dict[Coord, int] = {}
    clearance: dict[Coord, set[int]] = defaultdict(set)
    parents: dict[int, dict[Coord, Coord | None]] = {}
    for net in mapped.nets:
        tree, route = routes[net.id], realized[net.id]
        driver = sites[net.driver]
        sinks = {sites[s].position: s for s in net.sinks}
        # -- the tree connects exactly this net's pins ---------------------------
        assert tree.net == route.net == net.id and route.tree == tree
        assert tree.root == driver.position, f"net {net.id} does not start at its driver pin"
        parent = parents[net.id] = tree_parents(tree)
        assert parent == tree.parent
        children = children_of(parent)
        assert len(tree.branches) == net.fanout
        assert sorted(b.goal for b in tree.branches) == sorted(sinks), f"net {net.id} misses a sink pin"
        for branch in tree.branches:
            assert (branch.sink.instance, branch.sink.pin) == tuple(sinks[branch.goal])
            assert branch.sink.cell == branch.goal
        touched = {pins[c] for c in parent if c in pins}
        assert touched == {tuple(net.driver), *(tuple(s) for s in net.sinks)}, f"net {net.id} uses a foreign pin"
        # -- pins are left / entered only along their facing ---------------------
        assert children[tree.root], f"net {net.id} never leaves its driver"
        for child in children[tree.root]:
            assert step(tree.root, child) == heading(driver.facing), f"net {net.id} leaves its driver sideways"
        for goal, sink in sinks.items():
            facing = heading(sites[sink].facing)
            assert step(parent[goal], goal) == (-facing[0], -facing[1]), f"net {net.id} enters {sink} sideways"
            assert children[goal] == [], f"net {net.id} continues past sink pin {sink}"
        # -- realized elements, supports and clearances ----------------------------
        kinds = {e.coord: e.kind for e in route.elements}
        assert len(route.elements) == len(kinds) == len(parent) and set(kinds) == set(parent)
        assert set(kinds.values()) <= {ElementKind.DUST, ElementKind.REPEATER}
        assert all(e.parent == parent[e.coord] for e in route.elements)
        pin_blocks = {tree.root, *sinks}
        assert sorted(route.supports) == sorted(support_of(c) for c in parent if c not in pin_blocks)
        staircases = [(up, c) for c, up in parent.items() if up is not None and up[1] != c[1]]
        assert set(route.clearances) == {clearance_of(up, c) for up, c in staircases}
        for cell in parent:
            assert cell not in signal, f"nets {signal[cell]} and {net.id} share signal block {cell}"
            signal[cell] = net.id
        for cell in route.supports:
            assert cell not in support, f"nets {support[cell]} and {net.id} share support {cell}"
            support[cell] = net.id
        for cell in route.clearances:
            clearance[cell].add(net.id)
        # -- electrical: repeaters and signal strength, recomputed -------------------
        elements = {e.coord: e for e in route.elements}
        for element in route.repeaters:
            assert_repeater_legal(element.coord, element, parent, children, pin_blocks)
        drive = driver.strength
        if drive == 0:  # a constant-0 anchor: never powered, never repeated
            assert not route.powered and not route.repeaters
            continue
        assert route.powered
        level = signal_levels(tree.root, children, kinds, drive)
        assert set(level) == set(parent)
        for cell, strength in level.items():
            assert strength >= 1, f"net {net.id} block {cell} ({kinds[cell].value}) is unpowered"
            assert elements[cell].strength == strength, f"net {net.id} records a wrong strength at {cell}"
        for goal, sink in sinks.items():
            assert level[goal] >= sites[sink].strength, f"net {net.id} reaches {sink} too weak"
        reports = {(r.sink.instance, r.sink.pin): r for r in route.sinks}
        assert sorted(reports) == sorted(tuple(s) for s in net.sinks)
        for goal, sink in sinks.items():
            report, hops, cursor, repeaters = reports[tuple(sink)], 0, parent[goal], 0
            while cursor is not None:
                hops += 1
                repeaters += kinds[cursor] is ElementKind.REPEATER
                cursor = parent[cursor]
            assert (report.strength, report.required) == (level[goal], sites[sink].strength)
            assert (report.distance, report.repeaters) == (hops, repeaters)
            assert report.delay_ticks == repeaters * REPEATER_DELAY_TICKS

    # -- global block legality: one owner per block, rules 1-3 and 6 -------------------
    for cell, net_id in signal.items():
        assert MIN_SIGNAL_Y <= cell[1] <= max_y, f"net {net_id} signal {cell} outside the height limit"
        assert cell not in support, f"block {cell} is both signal of {net_id} and support of {support[cell]}"
        if cell in pins:
            assert below(cell) in body, f"pin block {cell} does not rest on its cell"
        else:
            assert cell not in body and cell not in keepout, f"net {net_id} signal {cell} inside a component"
            assert support.get(support_of(cell)) == net_id, f"net {net_id} signal {cell} has no own support"
    for cell, net_id in support.items():
        assert cell not in body and cell not in keepout and cell not in pins, f"support {cell} inside a component"
    for cell, nets in clearance.items():
        blockers = (cell in signal, cell in support, cell in body, cell in pins)
        assert not any(blockers), f"staircase clearance {cell} of nets {sorted(nets)} is not air"
    for cell, net_id in signal.items():
        for nb in neighborhood(cell):
            other = signal.get(nb)
            if other is None:
                continue
            assert other == net_id, f"nets {net_id} and {other} short at {cell} / {nb}"
            parent = parents[net_id]
            assert parent[nb] == cell or parent[cell] == nb, f"net {net_id} touches itself at {cell} / {nb}"

    # -- the router's working state agrees, as does the backend's own verifier --------------
    assert grid.conflicts() == []
    for net in mapped.nets:
        claims = grid.claims[net.id]
        assert claims.signals == dict.fromkeys(parents[net.id], 1), f"net {net.id} signal claims are off"
        assert claims.supports == dict.fromkeys(realized[net.id].supports, 1)
        assert set(claims.clearances) == set(realized[net.id].clearances)
        request = result.routed.requests[net.id]
        assert request.driver.cell == routes[net.id].root
        assert {s.cell for s in request.sinks} == {b.goal for b in routes[net.id].branches}
    assert verify_design(mapped, result.routed.requests, realized, max_y=max_y) == []
    assert result.violations == []
    assert_only_placement_changed(run)
    assert_metrics_consistent(run, body, signal, support, clearance)
    assert_design_file(run)


def assert_only_placement_changed(run: Run) -> None:
    """P&R decides origins and orientations only: the logical netlist and the
    mapped connectivity equal a fresh synthesis of the same graph."""
    fresh = synthesize_to_primitives(run.graph, interface=run.interface)
    assert run.netlist.to_dict() == fresh.to_dict()
    mapped = run.mapped.to_dict()
    for record in mapped["instances"]:
        assert record["origin"] is not None and record["orientation"] is not None
        record["origin"] = record["orientation"] = None
    assert mapped == map_primitives_to_minecraft(fresh, library=run.library).to_dict()


def assert_metrics_consistent(run: Run, body, signal, support, clearance) -> None:
    result, mapped = run.result, run.mapped
    m = result.metrics
    json.dumps(m)  # plain JSON, no wall-clock objects
    routes, realized = result.routes, result.realized
    routing = m["routing"]
    dust = sum(len(r.dust) for r in realized.values())
    repeaters = sum(len(r.repeaters) for r in realized.values())
    assert routing["routed_nets"] == len(mapped.nets) == m["logical"]["one_bit_nets"]
    assert routing["dust_blocks"] == dust and routing["repeaters"] == repeaters
    assert dust + repeaters == len(signal) == routing["total_routed_length"] == sum(t.length for t in routes.values())
    assert routing["max_routed_length"] == max(t.length for t in routes.values())
    assert routing["total_branch_steps"] == sum(len(b.path) - 1 for t in routes.values() for b in t.branches)
    assert routing["total_branch_steps"] == len(signal) - len(routes)  # branches share, never overlap
    assert routing["vertical_steps"] == sum(
        1 for t in routes.values() for c, up in t.parent.items() if up is not None and up[1] != c[1]
    )
    assert routing["support_blocks"] == len(support)
    assert routing["clearance_blocks"] == len(clearance)
    assert routing["iterations"] == result.routed.iterations >= 1
    assert routing["rip_ups"] == result.routed.rip_ups >= 0
    assert m["legalization"]["repeaters"] == repeaters
    assert m["legalization"]["rounds"] == result.legalized.rounds >= 1
    powered = [s.strength for r in realized.values() if r.powered for s in r.sinks]
    assert m["legalization"]["min_sink_strength"] == min(powered, default=None)
    assert m["placement"]["component_count"] == m["techmap"]["components"] == len(mapped.instances)
    assert m["placement"]["occupied_component_voxels"] == len(body) == m["techmap"]["component_voxels"]
    assert m["placement"]["keepout_voxels"] == len(result.grid.keepout)
    assert m["final"]["blocks"] == len(body) + len(signal) + len(support)
    bounds = Bounds.of([*body, *signal, *support])
    assert bounds is not None and m["final"]["bounds"] == bounds.to_dict() == result.design_bounds.to_dict()
    assert m["final"]["volume"] == bounds.volume
    assert m["pnr"] == {"attempts": result.attempts, "attempt": result.geometry.attempt}
    assert result.attempts == result.geometry.attempt + 1
    depth, ticks = timing_of(run)
    assert m["logical"]["logic_depth"] == m["timing"]["logic_depth"] == depth
    assert m["timing"]["critical_path_ticks"] == ticks
    assert m["timing"]["state_elements"] == m["logical"]["register_bits"]


def timing_of(run: Run) -> tuple[int, int]:
    """``(logic depth, critical path ticks)`` re-derived from the netlist and
    the realized routes: gates count one level each (register bits, inputs and
    constants are sources); a path's ticks are cell latencies plus the repeater
    delay of every route it rides, up to an output, register or peripheral pin."""
    logical, cells = run.mapped.logical, run.mapped.instances
    driver = {(s.instance, s.pin): net.driver.instance for net in logical.nets for s in net.sinks}
    reports = [r for route in run.result.realized.values() for r in route.sinks]
    hop = {(r.sink.instance, r.sink.pin): r.delay_ticks for r in reports}

    @functools.cache
    def depth(inst: int) -> int:
        prim = logical.instances[inst]
        return 1 + max(depth(driver[(inst, pin)]) for pin in prim.inputs) if prim.kind in GATE_KINDS else 0

    @functools.cache
    def settled(inst: int) -> int:
        prim, latency = logical.instances[inst], cells[inst].cell.latency or 0
        if prim.kind not in GATE_KINDS:
            return latency
        return latency + max(settled(driver[(inst, pin)]) + hop[(inst, pin)] for pin in prim.inputs)

    ends = (PrimitiveKind.OUTPUT_BIT, PrimitiveKind.REGISTER_BIT, PrimitiveKind.PERIPHERAL)
    observed = [(i.id, pin) for i in logical.instances if i.kind in ends for pin in i.inputs]
    gates = [i.id for i in logical.instances if i.kind in GATE_KINDS]
    return (
        max((depth(g) for g in gates), default=0),
        max((settled(driver[end]) + hop[end] for end in observed), default=0),
    )


def assert_design_file(run: Run) -> None:
    """The final ``redc.physical-primitive.v1`` file states exactly the artifacts."""
    result, mapped = run.result, run.mapped
    design = json.loads(json.dumps(result.to_design_dict()))
    assert design["schema"] == PHYSICAL_SCHEMA and design["coordinate_system"]["units"] == "blocks"
    assert [i["id"] for i in design["instances"]] == [i.id for i in mapped.instances]
    for record, inst in zip(design["instances"], mapped.instances):
        assert record["origin"] == list(inst.origin)
        assert sorted(tuple(v["coord"]) for v in record["voxels"]) == sorted(inst.placed.occupied)
        assert {p["name"]: tuple(p["position"]) for p in record["pins"]} == {
            name: pin.position for name, pin in inst.placed.pins.items()
        }
    assert [n["id"] for n in design["nets"]] == [n.id for n in mapped.nets]
    for record in design["nets"]:
        route = result.realized[record["id"]]
        assert record["route"] == json.loads(json.dumps(route.tree.to_dict()))
        assert record["realized"] == json.loads(json.dumps(route.to_dict()))
    assert design["metrics"] == json.loads(json.dumps(result.metrics))


# -- end to end ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_every_design_routes_legally(name: str) -> None:
    run = routed(name)
    assert run.result.success, run.result.failure
    assert_legal_design(run)


@pytest.mark.parametrize("name", ["not", "and"])
def test_single_gates_route_one_branch_per_pin(name: str) -> None:
    run = routed(name)
    (gate,) = [i for i in run.mapped.instances if i.kind.is_gate]
    nets = run.mapped.nets
    assert len(nets) == len(gate.cell.pins)  # one net per gate pin, nothing else
    for net in nets:
        (branch,) = run.result.routes[net.id].branches
        assert branch.start == run.result.routes[net.id].root
        assert branch.path[0] == run.mapped.instances[net.driver.instance].placed.pins[net.driver.pin].position
    assert run.result.attempts == 1


def test_fanout_net_is_one_tree_sharing_a_trunk() -> None:
    run = routed("fanout_mux")
    result, mapped = run.result, run.mapped
    (select,) = mapped.logical.port("in_s").bits
    (net,) = [n for n in mapped.nets if n.driver == select]  # ONE net, however many sinks
    assert net.fanout == 5  # NOT(s) plus one AND per result bit
    tree = result.routes[net.id]
    assert len(tree.branches) == net.fanout
    assert any(b.start != tree.root for b in tree.branches)  # branches attach to the trunk
    # Each branch adds only new blocks: the union is a tree, never parallel copies.
    assert tree.length == 1 + sum(len(b.path) - 1 for b in tree.branches)
    depth = depth_of(tree)
    assert sum(depth[b.goal] for b in tree.branches) > tree.length - 1  # the trunk is shared
    # Every block of the one tree is claimed exactly once, by this net only.
    grid = result.grid
    for cell in tree.cells:
        assert grid.signal[cell] == {net.id: 1}
    for cell in result.realized[net.id].supports:
        assert grid.support[cell] == {net.id: 1}


def test_unrelated_parallel_wires_never_short() -> None:
    run = routed("parallel_and")
    result, mapped = run.result, run.mapped
    assert len(mapped.nets) == 12  # 8 input wires + 4 output wires, all one bit
    blocks = {n.id: result.routes[n.id].cell_set for n in mapped.nets}
    for a, mine in blocks.items():
        hood = {nb for cell in mine for nb in neighborhood(cell)}
        supports = set(result.realized[a].supports)
        for b, theirs in blocks.items():
            if a == b:
                continue
            assert not mine & theirs
            assert not hood & theirs, f"nets {a} and {b} are electrically adjacent"
            assert not supports & set(result.realized[b].supports)
            assert not supports & theirs
    # Bit i of `a` and of `b` reach AND gate i, which drives output pad i.
    logical = mapped.logical
    for gate in (i for i in mapped.instances if i.kind is PrimitiveKind.AND):
        bit = logical.instances[gate.id].provenance.bit
        for pin, port in (("a", "in_a"), ("b", "in_b")):
            assert logical.driver_of(BitTerminal(gate.id, pin)) == logical.port(port).bits[bit]
        (out,) = [n for n in mapped.nets if n.driver.instance == gate.id]
        assert out.sinks == (logical.port("result").bits[bit],)


@pytest.mark.parametrize("name", ["swapped_and", "swapped_xor", "half_adder"])
def test_crossing_routes_stack_two_blocks_apart(name: str) -> None:
    run = routed(name)
    result = run.result
    columns: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for net_id, tree in result.routes.items():
        for x, y, z in tree.cells:
            columns[(x, z)].append((y, net_id))
    crossings = {xz: users for xz, users in columns.items() if len({n for _, n in users}) > 1}
    assert crossings, "the swapped operands must cross somewhere"
    for (x, z), users in crossings.items():
        for (y1, n1), (y2, n2) in pairwise(sorted(users)):
            if n1 != n2:
                assert abs(y1 - y2) >= 2, f"nets {n1} and {n2} cross at {(x, z)} only {abs(y1 - y2)} apart"
                upper = max((y1, n1), (y2, n2))
                # The upper route's support sits directly above the lower dust.
                assert support_of((x, upper[0], z)) in result.realized[upper[1]].supports
    assert max(c[1] for t in result.routes.values() for c in t.cells) >= 3
    # Climbing to the upper level takes staircases whose clearance stays air.
    occupied = set(result.grid.body) | set(result.grid.pins) | set(result.grid.signal) | set(result.grid.support)
    staircases = [
        (up, c) for t in result.routes.values() for c, up in t.parent.items() if up is not None and up[1] != c[1]
    ]
    assert staircases
    for up, cell in staircases:
        lower = up if up[1] < cell[1] else cell
        clear = clearance_of(up, cell)
        assert clear == (lower[0], lower[1] + 1, lower[2])  # directly above the LOWER dust
        assert clear not in occupied, f"staircase {up} -> {cell} clearance {clear} is not air"
    assert_legal_design(run)


@pytest.mark.parametrize("name", ["not", "and", "half_adder"])
def test_long_routes_get_repeaters_facing_downstream(name: str) -> None:
    run = routed(name, channel_width=20)
    result = run.result
    assert result.success, result.failure
    long_nets = [
        n
        for n, t in result.routes.items()
        if result.realized[n].powered and max(depth_of(t).values()) > MAX_SIGNAL_STRENGTH
    ]
    assert long_nets, "the wide channel must force a powered route past 15 blocks"
    for net_id in long_nets:
        route = result.realized[net_id]
        assert route.repeaters, f"net {net_id} needs a repeater"
        tree = route.tree
        children = children_of(tree.parent)
        kinds = {e.coord: e.kind for e in route.elements}
        for element in route.repeaters:
            assert_repeater_legal(element.coord, element, tree.parent, children, tree.pin_cells)
            assert element.delay >= 0 and 1 <= element.strength <= MAX_SIGNAL_STRENGTH
        level = signal_levels(tree.root, children, kinds, result.routed.requests[net_id].driver.strength)
        assert min(level.values()) >= 1
        # Without the repeaters the far end of the route would be dark.
        bare = signal_levels(tree.root, children, dict.fromkeys(kinds, ElementKind.DUST), MAX_SIGNAL_STRENGTH)
        assert min(bare.values()) == 0
        # Delay counts the repeaters passed on the way to each sink.
        assert max(s.delay_ticks for s in route.sinks) >= REPEATER_DELAY_TICKS
    events = [e for e in result.trace.events if e["type"] == "repeater_inserted"]
    assert len(events) == result.metrics["routing"]["repeaters"] > 0
    assert_legal_design(run)


def test_tight_channels_are_negotiated_with_rip_ups() -> None:
    run = routed("half_adder", channel_width=1, component_spacing=0)
    result = run.result
    assert result.success, result.failure
    events = result.trace.events
    ends = [e for e in events if e["type"] == "routing_iteration_end"]
    assert ends[0]["conflicts"] > 0, "the first pass must collide in one-block channels"
    assert ends[-1]["conflicts"] == 0
    assert result.metrics["routing"]["rip_ups"] > 0
    assert result.metrics["routing"]["iterations"] == len(ends) > 1
    rip_ups = [e for e in events if e["type"] == "net_rip_up"]
    assert len(rip_ups) == result.metrics["routing"]["rip_ups"]
    assert {e["reason"] for e in rip_ups} <= {"congestion", "legalization"}
    snapshots = [e for e in events if e["type"] == "congestion_snapshot"]
    assert snapshots[0]["conflicts"] and snapshots[-1]["conflicts"] == []
    # A ripped-up net is committed again afterwards, somewhere else.
    first, last = {}, {}
    for e in events:
        if e["type"] == "net_route_committed":
            first.setdefault(e["net"], e["cells"])
            last[e["net"]] = e["cells"]
    assert any(first[n] != last[n] for n in first)
    assert result.grid.conflicts() == []
    assert_legal_design(run)


def test_registers_share_one_clock_tree_and_one_reset_tree() -> None:
    run = routed("counter")
    result, mapped = run.result, run.mapped
    registers = [i for i in mapped.instances if i.kind is PrimitiveKind.REGISTER_BIT]
    assert len(registers) >= 2
    controls = (("clock", PrimitiveKind.CLOCK_SOURCE, "clk"), ("reset", PrimitiveKind.RESET_SOURCE, "rst"))
    for role, source, pin in controls:
        (net,) = [n for n in mapped.nets if n.role == role]
        assert mapped.instances[net.driver.instance].kind is source
        assert sorted(tuple(s) for s in net.sinks) == sorted((r.id, pin) for r in registers)
        tree = result.routes[net.id]
        assert len(tree.branches) == len(registers)
        assert {b.goal for b in tree.branches} == {r.placed.pins[pin].position for r in registers}
        assert tree.root == mapped.instances[net.driver.instance].placed.pins[pin].position
        # No other net reaches a register's clock or reset pin.
        others = {tuple(s) for n in mapped.nets if n.id != net.id for s in n.sinks}
        assert not others & {(r.id, pin) for r in registers}
    assert_legal_design(run)


def test_pnr_changes_only_placement() -> None:
    graph = compile_source(SOURCES["uint4_add"])
    netlist = synthesize_to_primitives(graph, interface=PADS)
    mapped = map_primitives_to_minecraft(netlist)
    before, mapped_before = netlist.to_dict(), mapped.to_dict()
    assert all(i["origin"] is None and i["orientation"] is None for i in mapped_before["instances"])
    result = place_and_route_primitive(mapped, PrimitivePnRConfig())
    assert result.success, result.failure
    assert netlist.to_dict() == before
    after = mapped.to_dict()
    for record, inst in zip(after["instances"], mapped.instances):
        assert record["origin"] is not None
        assert record["orientation"] in {o.name for o in inst.cell.orientations}
        record["origin"] = record["orientation"] = None
    assert after == mapped_before
    mapped.validate()


@pytest.mark.parametrize(("name", "level"), [("half_adder", "detailed"), ("uint4_add", "basic")])
def test_pnr_is_deterministic(name: str, level: str) -> None:
    first, second = run_pnr(name, level=level), run_pnr(name, level=level)
    assert first.result.success and second.result.success
    assert first.result.to_design_dict() == second.result.to_design_dict()
    assert first.result.trace.to_json() == second.result.trace.to_json()


# -- search_branch / validate_branch on hand-built grids -----------------------------


def site(cell: Coord, facing: Direction, instance: int, *, net: int | None = 1, out: bool = False) -> PinSite:
    """A pin endpoint: an output (drive 15) or an input (needs strength 1)."""
    return PinSite(cell, instance, "y" if out else "a", "out" if out else "in", facing, 15 if out else 1, net)


def grid_with(*sites: PinSite, walls=(), max_y: int = 6, claim: bool = True) -> BlockGrid:
    """A grid holding each pin on a one-block base (and optional wall bodies);
    pins are claimed by their nets, as the router does before routing."""
    grid = BlockGrid(max_y=max_y)
    for s in sites:
        grid.place(s.instance, [below(s.cell)], [], [s])
        if claim and s.net is not None:
            grid.claim_signal(s.net, s.cell)
    if walls:
        grid.place(99, walls, [], [])
    return grid


ROOT = site((0, 1, 0), EAST, 0, out=True)
GOAL = site((6, 1, 0), WEST, 1)
BOX = Bounds((-4, 0, -5), (10, 5, 5))


def net_state(root: PinSite, *others: PinSite, tree=()) -> NetState:
    return NetState(root.net, root, frozenset([root.cell, *(o.cell for o in others)]), {root.cell, *tree}, set(), {})


def search(grid: BlockGrid, root: PinSite = ROOT, goal: PinSite = GOAL, *, bounds: Bounds = BOX,
           state: NetState | None = None, max_expansions: int = 20_000, **kwargs):  # fmt: skip
    state = state or net_state(root, goal)
    result = search_branch(grid, state, sorted(state.tree), goal, bounds=bounds, present_factor=0.5,
                           vertical_cost=1.0, max_expansions=max_expansions, **kwargs)  # fmt: skip
    return state, result


def assert_path_legal(grid: BlockGrid, state: NetState, path, goal: PinSite) -> None:
    assert all(is_move(a, b) for a, b in pairwise(path))
    assert path[0] in state.tree and path[-1] == goal.cell
    for cell in path[1:-1]:
        assert cell not in grid.body and cell not in grid.keepout and cell not in grid.pins
        assert support_of(cell) not in grid.body and support_of(cell) not in grid.pins
    for a, b in pairwise(path):
        clear = clearance_of(a, b)
        assert clear is None or (clear not in grid.body and clear not in path)
    assert validate_branch(state, path, goal.cell) is None


def test_search_runs_straight_between_facing_pins() -> None:
    grid = grid_with(ROOT, GOAL)
    state, result = search(grid)
    assert result.found and result.reason is None
    assert result.path == tuple((x, 1, 0) for x in range(7))
    assert 0 < result.expansions < 50
    assert_path_legal(grid, state, result.path, GOAL)


def test_search_detours_around_a_full_height_body() -> None:
    wall = [(3, y, z) for y in range(7) for z in range(-2, 3)]
    grid = grid_with(ROOT, GOAL, walls=wall)
    state, result = search(grid)
    assert result.found
    path = result.path
    assert not set(path) & set(wall)
    assert len(path) - 1 == 6 + 2 * 3  # around the wall end, nothing longer
    assert {c[1] for c in path} == {1}  # climbing only costs more here
    assert any(c[0] == 3 and abs(c[2]) == 3 for c in path)
    assert result.blocked.get("component", 0) + result.blocked.get("support_blocked", 0) > 0
    assert_path_legal(grid, state, path, GOAL)


def test_search_climbs_a_staircase_over_a_low_wall() -> None:
    wall = [(3, y, z) for y in (0, 1) for z in range(-5, 6)]  # spans the whole box in z
    grid = grid_with(ROOT, GOAL, walls=wall)
    state, result = search(grid)
    assert result.found
    path = result.path
    assert max(c[1] for c in path) >= 3  # its dust must rest on a support above the wall
    climbs = [(a, b) for a, b in pairwise(path) if a[1] != b[1]]
    assert climbs and all(abs(a[1] - b[1]) == 1 for a, b in climbs)
    for a, b in climbs:  # every staircase clearance is air (here: not a wall block)
        assert clearance_of(a, b) not in grid.body
    assert_path_legal(grid, state, path, GOAL)


def test_search_enters_and_leaves_pins_only_along_their_facing() -> None:
    root = site((0, 1, 0), WEST, 0, out=True)  # leaves AWAY from the goal
    goal = site((6, 1, 0), NORTH, 1)  # entered moving south, from (6, 1, -1)
    grid = grid_with(root, goal)
    state, result = search(grid, root, goal)
    assert result.found
    path = result.path
    assert path[1] == (-1, 1, 0)
    assert path[-2] == (6, 1, -1)
    assert result.blocked.get("pin_facing", 0) > 0
    assert_path_legal(grid, state, path, goal)


def test_search_failure_reasons() -> None:
    grid = grid_with(ROOT, GOAL)
    outside = site((40, 1, 0), WEST, 2)
    _, result = search(grid_with(ROOT, outside), ROOT, outside)
    assert (result.path, result.reason, result.expansions) == (None, "goal_out_of_bounds", 0)
    _, result = search(grid, max_expansions=3)
    assert (result.path, result.reason, result.expansions) == (None, "expansion_limit", 3)
    # A wall whose only gap lies outside the search box: unreachable, and the
    # search never expands a block outside the box.
    wall = [(3, y, z) for y in range(7) for z in range(-8, 9) if z != 7]
    tight = Bounds((-2, 0, -4), (8, 5, 4))
    expanded = []
    _, result = search(grid_with(ROOT, GOAL, walls=wall), bounds=tight,
                       on_expand=lambda cell, g, h, frontier: expanded.append(cell))  # fmt: skip
    assert result.path is None and result.reason == "unreachable"
    assert expanded and all(tight.contains(c) for c in expanded)
    assert result.expansions == len(expanded)
    _, result = search(grid_with(ROOT, GOAL, walls=wall), bounds=BOX.expand(0, 3))
    assert result.found and (3, 1, 7) in result.path  # the gap, once inside the box


def test_search_avoids_blocks_and_its_own_other_pins() -> None:
    grid = grid_with(ROOT, GOAL)
    state, result = search(grid, avoid=frozenset({(3, 1, 0)}))
    assert result.found and (3, 1, 0) not in result.path
    assert result.blocked.get("avoid", 0) > 0
    # Another sink of the SAME net sits next to the straight line: rule 1 keeps
    # the branch out of its neighbourhood (only parent/child may touch).
    other = site((3, 1, 1), WEST, 2)
    grid = grid_with(ROOT, GOAL, other)
    state, result = search(grid, state=net_state(ROOT, GOAL, other))
    assert result.found
    assert all(other.cell not in neighborhood(c) for c in result.path)
    assert result.blocked.get("own_adjacency", 0) > 0
    assert_path_legal(grid, state, result.path, GOAL)


def test_search_keeps_out_of_reserved_approach_corridors() -> None:
    grid = grid_with(ROOT, GOAL)
    for reserved in ({(3, 1, 0)}, {(3, 0, 0)}):  # a signal block, then a support block, on the straight line
        state, result = search(grid, reserved=frozenset(reserved))
        assert result.found and result.blocked.get("reserved_approach", 0) > 0
        path = result.path
        supports = {support_of(c) for c in path[1:-1]}
        clearances = {clearance_of(a, b) for a, b in pairwise(path)} - {None}
        assert not (set(path) | supports | clearances) & reserved
        assert_path_legal(grid, state, path, GOAL)


def test_history_steers_the_search_unless_it_is_ignored() -> None:
    grid = grid_with(ROOT, GOAL)
    hot = [(x, 1, 0) for x in range(1, 6)]
    grid.add_history(hot, 10.0)
    state, result = search(grid)
    assert result.found and not {(2, 1, 0), (3, 1, 0), (4, 1, 0)} & set(result.path)
    assert_path_legal(grid, state, result.path, GOAL)
    _, ignored = search(grid, history_weight=0.0)
    assert ignored.path == tuple((x, 1, 0) for x in range(7))


def test_weighted_search_does_not_flood_a_congested_region() -> None:
    """When every step costs more than the heuristic admits (here: uniform
    history), optimal A* floods the box; weighted A* stays directed."""
    root, goal = site((0, 1, 0), EAST, 0, out=True), site((30, 1, 0), WEST, 1)
    box = Bounds((-4, 0, -12), (34, 5, 12))
    grid = grid_with(root, goal)
    grid.add_history([(x, y, z) for x in range(-4, 35) for y in range(6) for z in range(-12, 13)], 1.0)
    _, optimal = search(grid, root, goal, bounds=box, max_expansions=10**6)
    state, weighted = search(grid, root, goal, bounds=box, weight=3.0, max_expansions=10**6)
    assert optimal.found and weighted.found
    assert weighted.expansions * 4 < optimal.expansions
    assert len(weighted.path) >= len(optimal.path) == 31
    assert_path_legal(grid, state, weighted.path, goal)


def test_search_returns_the_goal_when_it_is_already_on_the_tree() -> None:
    state = net_state(ROOT, GOAL, tree=[GOAL.cell])
    result = search_branch(grid_with(ROOT, GOAL), state, [ROOT.cell], GOAL, bounds=BOX,
                           present_factor=0.5, vertical_cost=1.0, max_expansions=10)  # fmt: skip
    assert result.path == (GOAL.cell,) and result.expansions == 0


@pytest.mark.parametrize(
    ("path", "tree", "reason", "offenders"),
    [
        # a U-turn: the second and fifth blocks touch without being path neighbours
        ([(0, 1, 0), (1, 1, 0), (2, 1, 0), (2, 1, 1), (1, 1, 1)], (), "path_touches_itself", {(1, 1, 0), (1, 1, 1)}),
        # back next to the tree block it started from (only that one is not a NEW block)
        ([(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)], (), "path_touches_itself", {(0, 1, 1)}),
        # beside an existing tree block that is not the branch start
        ([(0, 1, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2), (2, 1, 2)], [(0, 1, 2)], "path_touches_own_net", {(1, 1, 2)}),
        # climbing, then turning back over the start: that dust would rest on dust
        ([(0, 1, 0), (1, 2, 0), (0, 2, 0), (0, 3, 1)], (), "support_on_signal", {(0, 2, 0)}),
        # a plain staircase is fine
        ([(0, 1, 0), (1, 2, 0), (2, 2, 0)], (), None, None),
    ],
)
def test_validate_branch(path, tree, reason, offenders) -> None:
    goal = site(path[-1], WEST, 1)
    state = net_state(ROOT, goal, tree=tree)
    found = validate_branch(state, path, goal.cell)
    if reason is None:
        assert found is None
    else:
        assert found is not None and found[0] == reason
        assert found[1] in offenders and found[1] in path[1:]  # a NEW block the re-search can avoid


def test_validate_branch_rejects_supports_and_clearances_of_its_own_tree() -> None:
    goal = site((2, 2, 0), WEST, 1)
    state = net_state(ROOT, goal)
    state.supports.add((0, 2, 0))  # another branch's support above the start block
    assert validate_branch(state, [(0, 1, 0), (1, 2, 0), (2, 2, 0)], goal.cell) == ("clearance_blocked", (1, 2, 0))
    goal = site((2, 1, 0), WEST, 1)
    state = net_state(ROOT, goal)
    state.clearances[(1, 0, 0)] = 1  # must stay air, so no support may go there
    assert validate_branch(state, [(0, 1, 0), (1, 1, 0), (2, 1, 0)], goal.cell) == ("support_in_clearance", (1, 1, 0))


# -- legalize_route on hand-built trees ------------------------------------------------


def tree_of(*paths, drive: int = MAX_SIGNAL_STRENGTH, net: int = 5) -> tuple[RouteTree, RouteRequest]:
    """A route tree whose first path starts at the driver pin and whose later
    paths start on earlier ones; each path ends on its own sink pin."""
    root = paths[0][0]
    driver = PinSite(root, 0, "y", "out", EAST, drive, net)
    sinks = tuple(PinSite(p[-1], k + 1, "a", "in", WEST, 1, net) for k, p in enumerate(paths))
    tree = RouteTree(net, driver, root, tuple(RouteBranch(s, tuple(p)) for s, p in zip(sinks, paths)))
    return tree, RouteRequest(net, "data", driver, sinks)


def realize(tree: RouteTree, request: RouteRequest) -> RealizedRoute:
    route = legalize_route(tree, request)
    assert isinstance(route, RealizedRoute), route
    children = children_of(tree.parent)
    kinds = {e.coord: e.kind for e in route.elements}
    for element in route.repeaters:
        assert_repeater_legal(element.coord, element, tree.parent, children, tree.pin_cells)
    if route.powered:
        level = signal_levels(tree.root, children, kinds, request.driver.strength)
        assert min(level.values()) >= 1
        assert {e.coord: e.strength for e in route.elements} == level
    return route


def straight(steps: int, y: int = 1, z: int = 0) -> list[Coord]:
    return [(x, y, z) for x in range(steps + 1)]


@pytest.mark.parametrize(("steps", "repeaters"), [(14, 0), (15, 1), (20, 1), (30, 1), (31, 2), (46, 2), (47, 3)])
def test_straight_runs_get_one_repeater_per_fifteen_blocks(steps: int, repeaters: int) -> None:
    tree, request = tree_of(straight(steps))
    route = realize(tree, request)
    assert len(route.repeaters) == repeaters
    assert all(e.facing is EAST for e in route.repeaters)
    (sink,) = route.sinks
    assert sink.strength >= 1 and sink.repeaters == repeaters and sink.distance == steps
    assert sink.delay_ticks == repeaters * REPEATER_DELAY_TICKS


def test_a_twenty_block_run_is_refreshed_where_its_dust_would_die() -> None:
    tree, request = tree_of(straight(20))
    route = realize(tree, request)
    (repeater,) = route.repeaters
    # greedy, as far downstream as possible: the block that would be at strength 0
    assert repeater.coord == (15, 1, 0) and repeater.strength == 1 and repeater.parent == (14, 1, 0)
    assert repeater.facing is EAST and repeater.delay == 0
    (sink,) = route.sinks
    assert sink.strength == MAX_SIGNAL_STRENGTH - (20 - 16) and sink.delay_ticks == REPEATER_DELAY_TICKS


def zigzag(steps: int) -> list[Coord]:
    """Turns at every block: no straight segment anywhere."""
    path = [(0, 1, 0)]
    for k in range(steps):
        x, y, z = path[-1]
        path.append((x + 1, y, z) if k % 2 == 0 else (x, y, z + 1))
    return path


def staircase(steps: int) -> list[Coord]:
    """Straight in plan but up and down at every block: never level."""
    return [(x, 1 + x % 2, 0) for x in range(steps + 1)]


@pytest.mark.parametrize("shape", [zigzag, staircase], ids=["zigzag", "staircase"])
def test_runs_without_a_straight_level_segment_cannot_be_legalized(shape) -> None:
    path = shape(20)
    tree, request = tree_of(path)
    failure = legalize_route(tree, request)
    assert isinstance(failure, LegalizationFailure)
    assert failure.net == 5 and "no repeater site" in failure.reason
    assert failure.coord == path[MAX_SIGNAL_STRENGTH]  # the first block that would be dark
    assert failure.cells == tuple(reversed(path[: MAX_SIGNAL_STRENGTH + 1]))  # weak block back to the root
    assert failure.sink is None
    assert failure.to_dict()["coord"] == list(path[MAX_SIGNAL_STRENGTH])
    # Fifteen steps short of the sink, the same shape needs no repeater at all.
    tree, request = tree_of(shape(14))
    assert realize(tree, request).repeaters == ()


def test_one_trunk_repeater_serves_two_branches() -> None:
    trunk = straight(18)
    north = [*trunk, (18, 1, -1), (18, 1, -2)]
    south = [(18, 1, 0), (18, 1, 1), (18, 1, 2)]
    tree, request = tree_of(north, south)
    route = realize(tree, request)
    (repeater,) = route.repeaters
    assert repeater.coord in trunk and repeater.coord == (15, 1, 0)
    assert [s.repeaters for s in route.sinks] == [1, 1]
    assert all(s.strength >= 1 for s in route.sinks)


def test_branches_past_the_split_get_their_own_repeaters() -> None:
    east = straight(25)
    south = [(10, 1, z) for z in range(16)]
    tree, request = tree_of(east, south)
    route = realize(tree, request)
    assert sorted(e.coord for e in route.repeaters) == [(10, 1, 5), (15, 1, 0)]
    assert {e.coord: e.facing for e in route.repeaters} == {(15, 1, 0): EAST, (10, 1, 5): SOUTH}
    assert [s.repeaters for s in route.sinks] == [1, 1]


def test_an_unpowered_constant_net_needs_no_repeaters() -> None:
    tree, request = tree_of(straight(40), drive=0)
    route = legalize_route(tree, request)
    assert isinstance(route, RealizedRoute)
    assert not route.powered and route.repeaters == ()
    assert {e.strength for e in route.elements} == {0}


# -- the negotiated router on a hand-built bottleneck --------------------------------


def bottleneck() -> tuple[BlockGrid, list[RouteRequest], Bounds]:
    """Two nets whose straight lines share a one-block gap; a far gap exists."""
    a_out, a_in = site((0, 1, -1), EAST, 0, net=1, out=True), site((10, 1, -1), WEST, 1, net=1)
    b_out, b_in = site((0, 1, 1), EAST, 2, net=2, out=True), site((10, 1, 1), WEST, 3, net=2)
    wall = [(5, y, z) for y in (0, 1) for z in range(-10, 11) if z not in (0, 8)]
    grid = grid_with(a_out, a_in, b_out, b_in, walls=wall, claim=False)
    requests = [RouteRequest(1, "data", a_out, (a_in,)), RouteRequest(2, "data", b_out, (b_in,))]
    return grid, requests, Bounds((-1, 1, -10), (11, 1, 10))  # one routing layer


def test_negotiation_resolves_a_shared_gap_with_rip_ups() -> None:
    grid, requests, bounds = bottleneck()
    trace = PrimitiveTraceRecorder("basic")
    outcome = NegotiatedRedstoneRouter(grid, requests, bounds=bounds, config=PrimitivePnRConfig(), trace=trace).run()
    assert outcome.success and outcome.failure is None
    assert outcome.rip_ups > 0 and outcome.iterations > 1
    assert grid.conflicts() == []
    through_far = [n for n, tree in outcome.routes.items() if (5, 1, 8) in tree.cell_set]
    through_near = [n for n, tree in outcome.routes.items() if (5, 1, 0) in tree.cell_set]
    assert len(through_far) == len(through_near) == 1
    for request in requests:
        tree = outcome.routes[request.net]
        assert tree.root == request.driver.cell and tree.branches[0].goal == request.sinks[0].cell
        assert set(grid.net_signals(request.net)) == tree.cell_set
    types = [e["type"] for e in trace.events]
    assert types.count("net_rip_up") == outcome.rip_ups
    snapshots = [e for e in trace.events if e["type"] == "congestion_snapshot"]
    assert {n for c in snapshots[0]["conflicts"] for n in c["nets"]} == {1, 2}
    assert snapshots[-1]["conflicts"] == []
    assert types[-1] == "routing_complete"


def test_negotiation_gives_up_with_a_congestion_failure() -> None:
    grid, requests, bounds = bottleneck()
    outcome = NegotiatedRedstoneRouter(
        grid, requests, bounds=bounds, config=PrimitivePnRConfig(max_routing_iterations=0)
    ).run()
    assert not outcome.success
    assert outcome.failure.reason == "congestion" and outcome.failure.conflicts
    assert outcome.conflicts == grid.conflicts() != []
    assert "did not converge" in outcome.failure.message


def test_an_enclosed_sink_is_unroutable_and_releases_its_partial_tree() -> None:
    out, sink = site((0, 1, 0), EAST, 0, out=True), site((6, 1, 0), WEST, 1)
    box = [(x, y, z) for x in (4, 5, 6, 7) for y in range(4) for z in (-1, 0, 1) if (x, y, z) != sink.cell]
    box.remove((6, 0, 0))  # the pin's own base is placed with the pin
    grid = grid_with(out, sink, walls=box, claim=False)
    trace = PrimitiveTraceRecorder("basic")
    outcome = NegotiatedRedstoneRouter(grid, [RouteRequest(1, "data", out, (sink,))],
                                       bounds=Bounds((-2, 1, -3), (9, 3, 3)), config=PrimitivePnRConfig(),
                                       trace=trace).run()  # fmt: skip
    assert not outcome.success
    assert (outcome.failure.reason, outcome.failure.search, outcome.failure.net) == ("unroutable", "unreachable", 1)
    assert set(grid.net_signals(1)) == {out.cell, sink.cell}  # only the reserved pins stay claimed
    failed = [e for e in trace.events if e["type"] == "branch_route_failed"]
    assert failed and failed[0]["goal"] == list(sink.cell)
    assert trace.events[-1]["type"] == "routing_failed"


# -- placement --------------------------------------------------------------------------


def mapped_design(name: str) -> PrimitivePhysicalNetlist:
    return map_primitives_to_minecraft(synthesize_to_primitives(compile_source(SOURCES[name]), interface=PADS))


@pytest.mark.parametrize("name", ["uint4_add", "counter", "fanout_mux"])
def test_placement_columns_follow_signal_flow(name: str) -> None:
    mapped = mapped_design(name)
    columns = placement_columns(mapped)
    assert sorted(columns) == [i.id for i in mapped.instances]
    kind = {i.id: i.kind for i in mapped.instances}
    last = max(columns.values())
    sources = {PrimitiveKind.INPUT_BIT, PrimitiveKind.CLOCK_SOURCE, PrimitiveKind.RESET_SOURCE}
    for inst_id, column in columns.items():
        if kind[inst_id] in sources:
            assert column == 0
        elif kind[inst_id] is PrimitiveKind.OUTPUT_BIT:
            assert column == last
        else:
            assert 0 < column < last
    cut = []
    for net in mapped.nets:
        for sink in net.sinks:
            if kind[net.driver.instance] is PrimitiveKind.REGISTER_BIT:
                cut.append(columns[sink.instance] - columns[net.driver.instance])
            else:  # signal flows toward higher columns
                assert columns[sink.instance] > columns[net.driver.instance], (net.id, sink)
    if name == "counter":
        # Register outputs are cut, so the sequential loop gets an order at all:
        # some consumer of a register's q sits at or before that register.
        assert cut and min(cut) <= 0


@pytest.mark.parametrize("name", ["counter", "masked_or"])
def test_constants_sit_one_super_column_before_their_first_consumer(name: str) -> None:
    """``const`` IR nodes are placed as late as possible: in the super-column
    (block rank) just before their earliest consumer's, so their wires stay short."""
    run = routed(name)
    result, mapped = run.result, run.mapped
    events = result.trace.events
    begin = next(e for e in events if e["type"] == "placement_begin")
    super_column = {b["block"]: b["super_column"] for b in begin["blocks"]}
    block = {e["instance"]: e["block"] for e in events if e["type"] == "component_placed"}
    columns = result.placed.columns
    consumers = defaultdict(list)
    for net in mapped.nets:
        consumers[net.driver.instance].extend(s.instance for s in net.sinks)
    logical = mapped.logical
    constants = [
        i.id
        for i in mapped.instances
        if i.kind in (PrimitiveKind.CONST0, PrimitiveKind.CONST1)
        and logical.ir_nodes[logical.instances[i.id].provenance.ir_node].op == "const"
    ]
    assert constants
    for const in constants:
        first = min(super_column[block[c]] for c in consumers[const])
        assert super_column[block[const]] == first - 1, const
        assert columns[const] < min(columns[c] for c in consumers[const])


def test_a_combinational_loop_cannot_be_placed() -> None:
    netlist = PrimitiveNetlist()
    netlist.add_group("design", "root", None, None)
    first = netlist.add_instance(PrimitiveKind.NOT, Provenance(None, "ring", group=0))
    second = netlist.add_instance(PrimitiveKind.NOT, Provenance(None, "ring", group=0))
    netlist.add_net(BitTerminal(first.id, "y"), [BitTerminal(second.id, "a")])
    netlist.add_net(BitTerminal(second.id, "y"), [BitTerminal(first.id, "a")])
    mapped = map_primitives_to_minecraft(netlist)
    with pytest.raises(PlacementError, match="combinational loop"):
        placement_columns(mapped)


# -- failures and retries ------------------------------------------------------------------


def test_an_impossible_search_budget_fails_cleanly() -> None:
    run = run_pnr("half_adder", max_astar_expansions=1, expansions_per_block=0, max_pnr_attempts=2)
    result = run.result
    assert not result.success
    assert result.attempts == 2 and result.geometry.attempt == 1
    assert result.failure.stage == "routing" and result.failure.reason == "unroutable"
    assert result.failure.attempts == 2 and result.failure.net is not None
    with pytest.raises(PrimitivePnRError, match="after 2 attempt"):
        result.raise_for_failure()
    with pytest.raises(PrimitivePnRError):
        result.to_design_dict()
    trace = result.trace.to_dict()
    assert trace["final"]["success"] is False and trace["final"]["failure"]["stage"] == "routing"
    ends = [e for e in trace["events"] if e["type"] == "pnr_attempt_end"]
    assert [e["status"] for e in ends] == ["failed", "failed"]
    json.dumps(trace)


#: Free staircases and optimal A* route one half-adder net as an up-and-down
#: staircase: no straight level block anywhere for a repeater to sit on.
STAIRCASES = {"vertical_cost": 0.0, "channel_width": 8, "astar_weight": 1.0}


def test_a_net_legalization_rejects_is_rerouted_and_renegotiated() -> None:
    run = run_pnr("half_adder", **STAIRCASES)
    result = run.result
    events = result.trace.events
    failed = [e for e in events if e["type"] == "legalization_failed"]
    assert failed, "the scenario no longer produces a route without a repeater site"
    assert {e["round"] for e in failed} == {0}
    ripped = [e for e in events if e["type"] == "net_rip_up" and e["reason"] == "legalization"]
    assert sorted(e["net"] for e in ripped) == sorted(e["net"] for e in failed)
    after = events[events.index(ripped[0]) :]
    assert any(e["type"] == "net_route_committed" and e["net"] == ripped[0]["net"] for e in after)
    rounds = [e for e in events if e["type"] == "legalization_complete"]
    assert rounds[0]["failures"] == len(failed) and rounds[-1]["failures"] == 0
    assert result.success and result.metrics["legalization"]["rounds"] == len(rounds) >= 2
    assert_legal_design(run)
    # Without repair rounds the same rejection is final.
    once = run_pnr("half_adder", max_legalization_rounds=0, max_pnr_attempts=1, **STAIRCASES).result
    assert not once.success
    assert (once.failure.stage, once.failure.reason) == ("legalization", "signal_strength")
    assert once.failure.net == failed[0]["net"] and "no repeater site" in once.failure.message
    assert once.trace.final["failure"]["stage"] == "legalization"
    with pytest.raises(PrimitivePnRError, match="no repeater site"):
        once.raise_for_failure()


def test_attempt_geometry_grows_deterministically() -> None:
    config = PrimitivePnRConfig(component_spacing=2, channel_width=4, routing_margin=5, max_y=6)
    growth = {
        "component_spacing": (2, config.retry_spacing_growth),
        "channel_width": (4, config.retry_channel_growth),
        "routing_margin": (5, config.retry_margin_growth),
        "max_y": (6, config.retry_height_growth),
    }
    assert all(step > 0 for _, step in growth.values())  # by default every retry spreads out
    for attempt in range(3):
        geometry = config.attempt_geometry(attempt)
        assert geometry == config.attempt_geometry(attempt) and geometry.attempt == attempt
        for name, (start, step) in growth.items():
            assert getattr(geometry, name) == start + attempt * step, name
        assert geometry.to_dict() == {"attempt": attempt, **{n: getattr(geometry, n) for n in growth}}
    assert PrimitivePnRConfig(max_y=383).attempt_geometry(5).max_y == 383  # the world height caps growth


@pytest.mark.parametrize(("name", "iterations"), [("swapped_xor", 1), ("and", 0)])
def test_a_retry_with_wider_geometry_succeeds_where_the_first_attempt_failed(name: str, iterations: int) -> None:
    config = {"channel_width": 0, "component_spacing": 0, "max_routing_iterations": iterations}
    run = run_pnr(name, **config)
    result = run.result
    assert result.success, result.failure
    assert result.attempts == 2
    assert result.geometry == PrimitivePnRConfig(**config).attempt_geometry(1)
    events = result.trace.events
    begins = [e for e in events if e["type"] == "pnr_attempt_begin"]
    ends = [e for e in events if e["type"] == "pnr_attempt_end"]
    assert [b["attempt"] for b in begins] == [0, 1]
    assert [e["status"] for e in ends] == ["failed", "success"]
    assert ends[0]["failure"]["stage"] == "routing"
    assert begins[1]["channel_width"] > begins[0]["channel_width"]
    assert begins[1]["max_y"] > begins[0]["max_y"]
    bounds = [e["search_bounds"] for e in events if e["type"] == "routing_begin"]
    assert bounds[1]["volume"] > bounds[0]["volume"]
    # The second attempt re-placed every component from scratch.
    placed = [e for e in events if e["type"] == "component_placed"]
    assert len(placed) == 2 * len(run.mapped.instances)
    assert_legal_design(run)


def test_a_cell_taller_than_the_route_height_fails_placement_then_retries_taller() -> None:
    # The 8-bit display placeholder stands five blocks tall; max_y=3 cannot hold it.
    run = run_pnr("passthrough8", interface=DEFAULT_INTERFACE_POLICY, max_y=3)
    result = run.result
    assert result.success, result.failure
    assert result.attempts == 2 and result.geometry.max_y == 3 + PrimitivePnRConfig().retry_height_growth
    events = result.trace.events
    assert [e["attempt"] for e in events if e["type"] == "placement_failed"] == [0]
    ends = [e for e in events if e["type"] == "pnr_attempt_end"]
    assert [e["status"] for e in ends] == ["failed", "success"]
    assert ends[0]["failure"]["stage"] == "placement"
    assert_legal_design(run)
    once = run_pnr("passthrough8", interface=DEFAULT_INTERFACE_POLICY, max_y=3, max_pnr_attempts=1).result
    assert not once.success and once.failure.stage == "placement" and once.attempts == 1
    assert once.routes == {} and once.realized == {}
    with pytest.raises(PrimitivePnRError, match="after 1 attempt"):
        once.to_design_dict()


# -- peripherals -----------------------------------------------------------------------


def test_a_display_peripheral_takes_eight_separately_routed_bits() -> None:
    run = run_pnr("passthrough8", interface=DEFAULT_INTERFACE_POLICY)
    result, mapped = run.result, run.mapped
    assert result.success, result.failure
    (display,) = [i for i in mapped.instances if i.kind is PrimitiveKind.PERIPHERAL]
    feeding = [n for n in mapped.nets if any(s.instance == display.id for s in n.sinks)]
    assert len(feeding) == 8  # one one-bit net per display pin, never a bus
    assert sorted(s.pin for n in feeding for s in n.sinks) == sorted(f"b{k}" for k in range(8))
    for net in feeding:
        (branch,) = result.routes[net.id].branches
        assert branch.goal == display.placed.pins[net.sinks[0].pin].position
    assert_legal_design(run)


# -- the backend verifier must reject (and report) every broken design -----------------------


def verdict(run: Run, *, requests=None, realized=None, max_y: int | None = None) -> set[str]:
    """The violation kinds :func:`verify_design` reports for mutated artifacts."""
    result = run.result
    violations = verify_design(
        run.mapped,
        result.routed.requests if requests is None else requests,
        result.realized if realized is None else realized,
        max_y=result.geometry.max_y if max_y is None else max_y,
    )
    return {v.kind for v in violations}


def with_route(run: Run, route: RealizedRoute) -> dict[int, RealizedRoute]:
    return {**run.result.realized, route.net: route}


def with_request(run: Run, request: RouteRequest) -> dict[int, RouteRequest]:
    return {**run.result.routed.requests, request.net: request}


def replace_elements(route: RealizedRoute, change) -> RealizedRoute:
    return dataclasses.replace(route, elements=tuple(change(e) for e in route.elements))


def grown(run: Run, net: int, path) -> RealizedRoute:
    """``net``'s route plus one extra branch along ``path`` (starting on its
    tree), realized and supported -- legal in itself, so the verifier must
    object to what it touches."""
    tree = run.result.routes[net]
    end = PinSite(path[-1], 10_000, "x", "in", WEST, 1, net)
    tree = RouteTree(tree.net, tree.driver, tree.root, (*tree.branches, RouteBranch(end, tuple(path))))
    route = legalize_route(tree, run.result.routed.requests[net])
    assert isinstance(route, RealizedRoute)
    return dataclasses.replace(route, supports=(*route.supports, support_of(path[-1])))


def extension(run: Run, wanted) -> tuple[int, Coord, Coord]:
    """The first ``(net, tree block, neighbour)`` (deterministic order) for which
    ``wanted(net, block, neighbour, owner)`` holds; ``owner`` maps signal blocks to nets."""
    result = run.result
    owner = {c: n for n, t in result.routes.items() for c in t.cells}
    for net in sorted(result.routes):
        tree = result.routes[net]
        for cell in tree.cells:
            if cell in tree.pin_cells:
                continue
            for nb in neighborhood(cell):
                free = nb[1] >= MIN_SIGNAL_Y and nb not in result.grid.body and nb not in result.grid.pins
                if free and wanted(net, cell, nb, owner):
                    return net, cell, nb
    raise AssertionError("the design offers no such block")


def first_with(run: Run, has) -> RealizedRoute:
    return next(r for _, r in sorted(run.result.realized.items()) if has(r))


def at(coord: Coord, **changes):
    """An element mapper changing only the element at ``coord``."""
    return lambda e: dataclasses.replace(e, **changes) if e.coord == coord else e


def flip_first_repeater(run: Run):
    route = first_with(run, lambda r: r.repeaters)
    repeater = route.repeaters[0]
    return {"realized": with_route(run, replace_elements(route, at(repeater.coord, facing=repeater.facing.opposite)))}


def strip_repeaters(run: Run):
    route = first_with(run, lambda r: r.repeaters)
    dust = replace_elements(route, lambda e: dataclasses.replace(e, kind=ElementKind.DUST, facing=None))
    return {"realized": with_route(run, dust)}


def drop_support(run: Run):
    route = first_with(run, lambda r: r.supports)
    return {"realized": with_route(run, dataclasses.replace(route, supports=route.supports[1:]))}


def misrecord_strength(run: Run):
    route = first_with(run, lambda r: r.powered)
    element = route.elements[1]
    return {"realized": with_route(run, replace_elements(route, at(element.coord, strength=element.strength + 1)))}


def drop_branch(run: Run):
    route = first_with(run, lambda r: len(r.tree.branches) > 1)
    tree = route.tree
    pruned = RouteTree(tree.net, tree.driver, tree.root, tree.branches[:1])
    return {"realized": with_route(run, legalize_route(pruned, run.result.routed.requests[tree.net]))}


def drop_route(run: Run):
    net = min(run.result.realized)
    return {"realized": {n: r for n, r in run.result.realized.items() if n != net}}


def skip_a_step(run: Run):
    route = first_with(run, lambda r: len(r.tree.branches[0].path) > 4)
    tree = route.tree
    first, *rest = tree.branches
    jumpy = RouteBranch(first.sink, first.path[:2] + first.path[3:])
    tree = RouteTree(tree.net, tree.driver, tree.root, (jumpy, *rest))
    return {"realized": with_route(run, legalize_route(tree, run.result.routed.requests[tree.net]))}


def turn_sink(run: Run):
    request = run.result.routed.requests[min(run.result.routed.requests)]
    sink, *others = request.sinks
    turned = dataclasses.replace(sink, facing=sink.facing.rotated(1))
    return {"requests": with_request(run, dataclasses.replace(request, sinks=(turned, *others)))}


def turn_driver(run: Run):
    request = run.result.routed.requests[min(run.result.routed.requests)]
    driver = dataclasses.replace(request.driver, facing=request.driver.facing.rotated(1))
    return {"requests": with_request(run, dataclasses.replace(request, driver=driver))}


def demand_full_strength(run: Run):
    route = first_with(run, lambda r: r.powered and any(s.strength < MAX_SIGNAL_STRENGTH for s in r.sinks))
    request = run.result.routed.requests[route.net]
    sinks = tuple(dataclasses.replace(s, strength=MAX_SIGNAL_STRENGTH) for s in request.sinks)
    return {"requests": with_request(run, dataclasses.replace(request, sinks=sinks))}


def move_driver(run: Run):
    request = run.result.routed.requests[min(run.result.routed.requests)]
    x, y, z = request.driver.cell
    driver = dataclasses.replace(request.driver, cell=(x, y, z - 1))
    return {"requests": with_request(run, dataclasses.replace(request, driver=driver))}


def beside_another_net(run: Run) -> tuple[int, Coord, Coord]:
    """A free block next to ``net``'s tree that touches ANOTHER net (rule 1 keeps
    every other net at least one such block away)."""

    def wanted(n, c, nb, owner):
        near = {owner[x] for x in neighborhood(nb) if x in owner}
        return nb not in owner and nb not in run.result.grid.keepout and bool(near - {n})

    return extension(run, wanted)


def grow_into_another_net(run: Run):
    net, cell, nb = beside_another_net(run)
    owner = {c: n for n, t in run.result.routes.items() for c in t.cells}
    theirs = min(x for x in neighborhood(nb) if owner.get(x, net) != net)
    return {"realized": with_route(run, grown(run, net, (cell, nb, theirs)))}


def grow_beside_another_net(run: Run):
    net, cell, nb = beside_another_net(run)
    return {"realized": with_route(run, grown(run, net, (cell, nb)))}


def grow_beside_itself(run: Run):
    def wanted(n, c, nb, owner):
        near = [x for x in neighborhood(nb) if x in owner and x != c]
        free = nb not in owner and nb not in run.result.grid.keepout
        return free and bool(near) and all(owner[x] == n for x in near)

    net, cell, nb = extension(run, wanted)
    return {"realized": with_route(run, grown(run, net, (cell, nb)))}


def grow_into_a_keepout(run: Run):
    net, cell, nb = extension(run, lambda n, c, nb, owner: nb in run.result.grid.keepout and nb not in owner)
    return {"realized": with_route(run, grown(run, net, (cell, nb)))}


def fill_a_clearance(run: Run):
    route = first_with(run, lambda r: len(r.elements) > 2)
    block = route.elements[1].coord  # a dust block is never a staircase's air
    return {"realized": with_route(run, dataclasses.replace(route, clearances=(*route.clearances, block)))}


def forget_a_clearance(run: Run):
    route = first_with(run, lambda r: r.clearances)
    return {"realized": with_route(run, dataclasses.replace(route, clearances=route.clearances[1:]))}


def repeat_on_the_root(run: Run):
    route = first_with(run, lambda r: r.powered)
    root = replace_elements(route, at(route.tree.root, kind=ElementKind.REPEATER, facing=EAST))
    return {"realized": with_route(run, root)}


def misdirect_an_element(run: Run):
    route = first_with(run, lambda r: len(r.elements) > 3)
    element = route.elements[3]
    return {"realized": with_route(run, replace_elements(route, at(element.coord, parent=element.coord)))}


def clear_inside_a_body(run: Run):
    route = first_with(run, lambda r: r.powered)
    body = min(run.result.grid.body)
    return {"realized": with_route(run, dataclasses.replace(route, clearances=(*route.clearances, body)))}


def support_inside_a_body(run: Run):
    route = first_with(run, lambda r: r.powered)
    body = min(run.result.grid.body)
    return {"realized": with_route(run, dataclasses.replace(route, supports=(*route.supports, body)))}


def lower_the_ceiling(run: Run):
    return {"max_y": max(c[1] for t in run.result.routes.values() for c in t.cells) - 1}


def drop_an_element(run: Run):
    route = first_with(run, lambda r: len(r.elements) > 3)
    return {"realized": with_route(run, dataclasses.replace(route, elements=route.elements[:2] + route.elements[3:]))}


def detach_a_branch(run: Run):
    route = first_with(run, lambda r: len(r.tree.branches) > 1)
    tree = route.tree
    *kept, last = tree.branches
    loose = RouteTree(tree.net, tree.driver, tree.root, (*kept, RouteBranch(last.sink, last.path[1:])))
    return {"realized": with_route(run, dataclasses.replace(route, tree=loose))}


def solidify_an_element(run: Run):
    route = first_with(run, lambda r: len(r.elements) > 2)
    solid = replace_elements(route, at(route.elements[1].coord, kind=ElementKind.SUPPORT))
    return {"realized": with_route(run, solid)}


MUTATIONS = [
    (flip_first_repeater, "repeater"),
    (strip_repeaters, "weak_signal"),
    (drop_support, "unsupported"),
    (misrecord_strength, "strength_record"),
    (drop_branch, "unreached_sink"),
    (drop_route, "unrouted"),
    (skip_a_step, "illegal_step"),
    (turn_sink, "pin_facing"),
    (turn_driver, "pin_facing"),
    (demand_full_strength, "weak_sink"),
    (move_driver, "root"),
    (grow_into_another_net, "shared_signal"),
    (grow_beside_another_net, "short"),
    (grow_beside_itself, "self_loop"),
    (grow_into_a_keepout, "signal_in_component"),
    (fill_a_clearance, "clearance_blocked"),
    (forget_a_clearance, "clearance_missing"),
    (repeat_on_the_root, "repeater"),
    (misdirect_an_element, "direction"),
    (clear_inside_a_body, "clearance_blocked"),
    (support_inside_a_body, "support_in_component"),
    (lower_the_ceiling, "height"),
    # malformed realized routes must be REPORTED, not crash the verifier
    (drop_an_element, "elements"),
    (detach_a_branch, "detached_branch"),
    # a signal block realized as a solid block is no longer a wire
    (solidify_an_element, None),
]


@pytest.mark.parametrize(("mutate", "kind"), MUTATIONS, ids=[m.__name__ for m, _ in MUTATIONS])
def test_verify_design_reports_every_broken_rule(mutate, kind) -> None:
    run = routed("half_adder", channel_width=20)  # repeaters, staircases, fanout and crossings
    assert verdict(run) == set()
    kinds = verdict(run, **mutate(run))
    if kind is None:
        assert kinds, "the broken design passed verification"
    else:
        assert kind in kinds, f"expected a {kind!r} violation, got {sorted(kinds)}"


def test_verify_design_reports_component_collisions() -> None:
    run = run_pnr("and")  # a private run: this test moves instances around
    mapped = run.mapped
    pad, other = mapped.instances[0], mapped.instances[1]
    home = pad.origin
    pad.place(other.origin, other.orientation)  # onto the other input pad
    assert "voxel_collision" in verdict(run)
    gate = next(i for i in mapped.instances if i.kind is PrimitiveKind.AND)
    bodies = {c for i in mapped.instances if i is not pad for c in i.placed.occupied}
    shape = pad.cell.oriented(pad.orientation)
    gx, gy, gz = gate.origin
    spot = next(
        origin
        for origin in ((gx + dx, gy, gz + dz) for dx in range(-5, 6) for dz in range(-5, 6))
        if shape.translate(origin).occupied & gate.placed.keepout and not shape.translate(origin).occupied & bodies
    )
    pad.place(spot, pad.orientation)  # into the gate's keep-out, touching no voxel
    assert "keepout_violation" in verdict(run)
    pad.place(home, pad.orientation)
    assert verdict(run) == set()


def test_verify_design_rejects_a_route_that_continues_past_a_sink_pin() -> None:
    """Rule 6: a sink pin is ENTERED along its facing and never left -- dust
    that turned away on an input pin would no longer point into the cell.
    Cells without side keep-outs (legal: keep-outs are a per-cell choice)
    leave room for such a route."""
    cells = [dataclasses.replace(cell, keepout=frozenset()) for cell in PRIMITIVE_TECHNOLOGY.values()]
    run = run_pnr("not", library=PrimitiveTechnologyLibrary("bare", cells))
    assert all(not inst.cell.keepout for inst in run.mapped.instances)
    assert_legal_design(run)  # keep-out-free cells still route legally
    gate = next(i for i in run.mapped.instances if i.kind is PrimitiveKind.NOT)
    net = next(n for n in run.mapped.nets if BitTerminal(gate.id, "a") in n.sinks)
    pin = gate.placed.pins["a"].position
    owner = {c for t in run.result.routes.values() for c in t.cells}
    side = next(
        nb for nb in ((pin[0], pin[1], pin[2] + 1), (pin[0], pin[1], pin[2] - 1))
        if nb not in owner and nb not in run.result.grid.body
        and not any(x in owner for x in neighborhood(nb) if x != pin)
    )  # fmt: skip
    kinds = verdict(run, realized=with_route(run, grown(run, net.id, (pin, side))))
    assert "pin_facing" in kinds, f"a route leaving input pin {pin} sideways passed: {sorted(kinds)}"


# -- the router's repair entry point -------------------------------------------------------


def test_repair_reroutes_rejected_nets_away_from_the_penalized_blocks() -> None:
    grid, requests, bounds = bottleneck()
    trace = PrimitiveTraceRecorder("basic")
    router = NegotiatedRedstoneRouter(grid, requests, bounds=bounds, config=PrimitivePnRConfig(), trace=trace)
    first = router.run()
    assert first.success
    (near,) = [n for n, tree in first.routes.items() if (5, 1, 0) in tree.cell_set]
    penalized = list(first.routes[near].cells)
    outcome = router.repair([near], reason="legalization", penalize=penalized)
    assert outcome.success and grid.conflicts() == []
    assert outcome.rip_ups > first.rip_ups and outcome.iterations > first.iterations
    assert all(grid.history.get(c, 0.0) > 0 for c in penalized)
    ripped = [e for e in trace.events if e["type"] == "net_rip_up" and e["reason"] == "legalization"]
    assert [e["net"] for e in ripped] == [near]
    assert ripped[0]["cells"] == [list(c) for c in penalized]
    for request in requests:
        tree = outcome.routes[request.net]
        assert tree.root == request.driver.cell and tree.branches[0].goal == request.sinks[0].cell


# -- configuration ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"max_y": 2},
        {"max_y": 384},
        {"channel_width": -1},
        {"component_spacing": -1},
        {"max_pnr_attempts": 0},
        {"max_astar_expansions": 0},
        {"present_factor_growth": 0.5},
        {"vertical_cost": -1.0},
        {"trace_level": "loud"},
    ],
    ids=lambda bad: next(iter(bad)),
)
def test_config_rejects_impossible_values(bad) -> None:
    with pytest.raises(CompileError):
        PrimitivePnRConfig(**bad)


def test_config_is_plain_json_in_blocks() -> None:
    config = PrimitivePnRConfig(trace_level="search")
    assert config.trace_level is TraceLevel.SEARCH
    record = json.loads(json.dumps(config.to_dict()))
    assert record["units"] == "blocks" and record["trace_level"] == "search"
    assert record["max_y"] == config.max_y and record["channel_width"] == config.channel_width
