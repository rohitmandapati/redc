"""First-pass 3D place-and-route: placement, A*, trees, negotiation, retries."""

import itertools
from pathlib import Path

import pytest

from redc import BOOL, CompileError, IRType, compile_source
from redc.physical import (
    LIBRARY,
    PERIPHERALS,
    CellKind,
    ClockSource,
    Face,
    Grid,
    InputPad,
    OperationSignature,
    OutputPad,
    PeripheralDirection,
    PhysicalNetlist,
    Register,
    ResetSource,
    lower_to_physical,
    simulate,
)
from redc.physical.cells.peripheral import LEVER, TWO_DIGIT_SEVEN_SEGMENT
from redc.physical.pnr import (
    Bounds,
    NegotiatedRouter,
    PnRConfig,
    PnRError,
    RouteEndpoint,
    RouteRequest,
    TraceLevel,
    TraceRecorder,
    astar,
    check_placement,
    escape_cell,
    place_and_route,
    place_instance,
    placement_layers,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
U8 = IRType(8)
TOPS = {
    "array_mux.redc": "main",
    "dot_product.redc": "main",
    "popcount.redc": "main",
    "priority_encoder.redc": "main",
    "uint8_add.redc": "add",
    "uint8_fib.redc": "fib",
}
(LEVER_CELL,) = PERIPHERALS.find(LEVER, BOOL, PeripheralDirection.INPUT)
(DISPLAY,) = PERIPHERALS.find(TWO_DIGIT_SEVEN_SEGMENT, U8, PeripheralDirection.OUTPUT)
(REGISTER,) = LIBRARY.cells(OperationSignature.of("register", U8, U8, BOOL))
(SUB,) = LIBRARY.cells(OperationSignature.of("sub", U8, U8, U8))


def mapped(example: str):
    graph = compile_source((EXAMPLES / example).read_text(), top=TOPS[example])
    return graph, lower_to_physical(graph)


def owned_cells(grid: Grid, owner: int) -> dict[tuple[int, int, int], CellKind]:
    return {(x, y, z): k for x, y, z, o, k in grid.occupied_cells() if o == owner}


# -- 54. component dimensions are real -----------------------------------------


@pytest.mark.parametrize("component", [DISPLAY, LEVER_CELL, REGISTER, SUB], ids=lambda c: c.name)
def test_placed_footprint_is_the_full_yaml_box(component) -> None:
    nl = PhysicalNetlist()
    inst = nl.add(component)
    grid = Grid()
    escapes: dict = {}
    origin = (3, 2, -4)
    assert check_placement(component, origin, grid, escapes) is None
    place_instance(grid, inst, origin, escapes)
    cells = owned_cells(grid, inst.id)
    dx, dy, dz = component.dim
    assert len(cells) == dx * dy * dz  # every cell of the prism, one owner
    pins = {p.absolute(origin) for p in component.ports}
    assert {c for c, k in cells.items() if k == CellKind.PIN} == pins
    assert all(k == CellKind.COMPONENT for c, k in cells.items() if c not in pins)
    assert inst.origin == origin
    # Escapes are recorded for routing but NOT written into the grid.
    for port in component.ports:
        assert escapes[escape_cell(port, origin)] == (inst.id, port.name)
        assert grid.is_free(*escape_cell(port, origin))


def test_display_occupies_its_6x5x2_placeholder_box() -> None:
    assert DISPLAY.dim == (6, 5, 2)
    _graph, nl = mapped("uint8_add.redc")
    result = place_and_route(nl)
    assert result.success
    (display,) = [i for i in nl.instances.values() if i.component is DISPLAY]
    cells = owned_cells(result.grid, display.id)
    assert len(cells) == 60
    box = Bounds.of(cells)
    assert box is not None and box.dims == (6, 5, 2)
    assert box.lo == display.origin


def test_check_placement_explains_conflicts() -> None:
    nl = PhysicalNetlist()
    first = nl.add(SUB)
    grid = Grid()
    escapes: dict = {}
    place_instance(grid, first, (0, 1, 0), escapes)
    assert "overlaps" in check_placement(SUB, (0, 1, 2), grid, escapes)
    # Directly west: the new body would sit on the first cell's WEST pin escapes.
    assert "would cover the routing escape of instance 0 pin 'a'" in check_placement(
        SUB, (-1, 1, 0), grid, escapes
    )
    # Further west and shifted in z the body is clear, but its EAST output
    # escape would land on the first cell's input escape.
    assert "already the escape of instance 0 pin 'a'" in check_placement(
        SUB, (-2, 1, -2), grid, escapes
    )
    assert "outside the grid" in check_placement(SUB, (5, 30, 0), grid, escapes)
    # A BOTTOM-pinned register at y=0 would escape below the grid.
    assert "outside the grid" in check_placement(REGISTER, (20, 0, 0), grid, escapes)


# -- 55/56. legality of every successful placement -----------------------------


def assert_legal_placement(result) -> None:
    grid, nl = result.grid, result.netlist
    seen: dict[tuple[int, int, int], int] = {}
    for inst in nl.instances.values():
        assert inst.origin is not None
        for cell in inst.component.footprint_cells(inst.origin):
            assert cell not in seen, f"instances {seen.get(cell)} and {inst.id} overlap"
            seen[cell] = inst.id
            assert 0 <= cell[1] < grid.height
    for inst in nl.instances.values():
        for port in inst.component.ports:
            out = inst.terminal(port.name).outward
            assert 0 <= out[1] < grid.height, f"{inst.id}.{port.name} escapes out of the grid"
            assert out not in seen, f"{inst.id}.{port.name} escape is inside instance {seen.get(out)}"


@pytest.mark.parametrize("example", sorted(TOPS))
def test_examples_place_without_overlap(example: str) -> None:
    _graph, nl = mapped(example)
    result = place_and_route(nl)
    assert result.success, result.failure
    assert_legal_placement(result)


def test_bottom_pins_lift_registers_above_y0() -> None:
    _graph, nl = mapped("uint8_fib.redc")
    result = place_and_route(nl, PnRConfig(base_y=0))
    assert result.success
    registers = [i for i in nl.instances.values() if isinstance(i.component, Register)]
    assert registers
    for reg in registers:
        assert {p.face for p in reg.component.ports} >= {Face.BOTTOM}
        assert reg.origin[1] >= 1
        assert reg.terminal("clk").outward[1] >= 0
        assert reg.terminal("rst").outward[1] >= 0
    assert_legal_placement(result)


def test_layers_put_sources_first_and_outputs_last() -> None:
    _graph, nl = mapped("uint8_fib.redc")
    layers = placement_layers(nl)
    by_label = {i.label: i for i in nl.instances.values()}
    last = max(layers.values())
    assert layers[by_label["start"].id] == 0  # lever on the west
    assert layers[by_label["in_n"].id] == 0
    assert layers[by_label["clk"].id] == 0 and layers[by_label["rst"].id] == 0
    assert layers[by_label["result"].id] == last  # display on the east
    assert layers[by_label["done"].id] == last
    assert place_and_route(nl).success
    assert by_label["start"].origin[0] < by_label["result"].origin[0]


# -- 57/58. A* ----------------------------------------------------------------


def wall_grid(gap_z: int) -> Grid:
    grid = Grid(height=1)
    for z in range(-4, 5):
        if z != gap_z:
            grid.place(2, 0, z, owner=99, kind=CellKind.BLOCKED)
    return grid


def test_astar_finds_the_gap_deterministically() -> None:
    bounds = Bounds((-1, 0, -4), (5, 0, 4))
    paths = []
    for _ in range(2):
        grid = wall_grid(gap_z=2)
        result = astar(grid, [(0, 0, 0)], (4, 0, 0), net=1, bounds=bounds,
                       present_factor=0.5, max_expansions=10_000)  # fmt: skip
        assert result.found
        paths.append(result.path)
    path = paths[0]
    assert paths[0] == paths[1]
    assert path[0] == (0, 0, 0) and path[-1] == (4, 0, 0)
    assert (2, 0, 2) in path  # through the gap
    assert len(path) - 1 == 4 + 2 * 2  # shortest detour
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) == 1 for a, b in itertools.pairwise(path))
    grid = wall_grid(gap_z=2)
    assert all(grid.is_routable(*c) for c in path)


def test_astar_uses_the_third_dimension() -> None:
    grid = Grid(height=3)
    for z in range(-3, 4):  # a full-depth wall on the y=0 plane only
        grid.place(2, 0, z, owner=99, kind=CellKind.BLOCKED)
    bounds = Bounds((-1, 0, -3), (5, 2, 3))
    result = astar(grid, [(0, 0, 0)], (4, 0, 0), net=1, bounds=bounds,
                   present_factor=0.5, max_expansions=10_000)  # fmt: skip
    assert result.found
    assert any(c[1] > 0 for c in result.path)
    assert len(result.path) - 1 == 4 + 2  # up, over, down


def test_astar_respects_bounds_and_expansion_limits() -> None:
    grid = wall_grid(gap_z=4)
    tight = Bounds((-1, 0, -3), (5, 0, 3))  # the gap at z=4 is outside
    result = astar(grid, [(0, 0, 0)], (4, 0, 0), net=1, bounds=tight,
                   present_factor=0.5, max_expansions=10_000)  # fmt: skip
    assert not result.found and result.reason == "unreachable"
    assert result.expansions <= 7 * 1 * 7  # never searched beyond the box
    limited = astar(Grid(height=1), [(0, 0, 0)], (30, 0, 0), net=1,
                    bounds=Bounds((-50, 0, -50), (50, 0, 50)),
                    present_factor=0.5, max_expansions=5)  # fmt: skip
    assert limited.reason == "expansion_limit" and limited.expansions == 5
    outside = astar(Grid(height=1), [(0, 0, 0)], (9, 0, 0), net=1, bounds=tight,
                    present_factor=0.5, max_expansions=10)  # fmt: skip
    assert outside.reason == "goal_out_of_bounds"


# -- 59. fanout tree ----------------------------------------------------------


def endpoint(instance: int, port: str, coord) -> RouteEndpoint:
    return RouteEndpoint(instance, port, coord)


def test_fanout_net_is_one_tree_with_shared_trunk() -> None:
    grid = Grid(height=1)
    request = RouteRequest(
        7,
        endpoint(0, "out", (0, 0, 0)),
        (
            endpoint(1, "a", (10, 0, 2)),
            endpoint(2, "a", (10, 0, -2)),
            endpoint(3, "a", (12, 0, 0)),
        ),
    )
    router = NegotiatedRouter(grid, [request], bounds=Bounds((-2, 0, -6), (14, 0, 6)), config=PnRConfig())
    outcome = router.run()
    assert outcome.success
    (routed,) = outcome.routes.values()
    assert routed.net == 7 and len(routed.branches) == 3
    assert routed.root == (0, 0, 0)
    assert {b.goal for b in routed.branches} == {(10, 0, 2), (10, 0, -2), (12, 0, 0)}
    # Branches attach to the existing tree instead of restarting at the driver.
    assert sum(len(b.path) for b in routed.branches) > routed.length
    assert any(b.start != routed.root for b in routed.branches)
    # The shared trunk is one net: occupancy is exactly 1 on every tree cell.
    assert all(grid.occupancy_at(*c) == 1 for c in routed.cells)
    assert grid.route_of(7) == routed.cell_set
    assert grid.max_occupancy() == 1


# -- 60. congestion / rip-up ----------------------------------------------------


def bottleneck() -> tuple[Grid, list[RouteRequest], Bounds]:
    grid = Grid(height=1)
    for z in range(-10, 11):
        if z not in (0, 8):  # a near gap and a far one
            grid.place(5, 0, z, owner=99, kind=CellKind.BLOCKED)
    requests = [
        RouteRequest(1, endpoint(0, "out", (0, 0, -1)), (endpoint(1, "a", (10, 0, -1)),)),
        RouteRequest(2, endpoint(2, "out", (0, 0, 1)), (endpoint(3, "a", (10, 0, 1)),)),
    ]
    return grid, requests, Bounds((-1, 0, -10), (11, 0, 10))


def test_negotiation_resolves_a_shared_bottleneck() -> None:
    grid, requests, bounds = bottleneck()
    trace = TraceRecorder(TraceLevel.BASIC)
    outcome = NegotiatedRouter(grid, requests, bounds=bounds, config=PnRConfig(), trace=trace).run()
    assert outcome.success
    assert grid.overused() == []
    assert outcome.iterations > 1
    # Exactly one net takes the far gap.
    through_far = [n for n, r in outcome.routes.items() if (5, 0, 8) in r.cell_set]
    assert len(through_far) == 1
    types = [e["type"] for e in trace.events]
    assert "net_rip_up" in types
    snapshots = [e for e in trace.events if e["type"] == "congestion_snapshot"]
    assert snapshots[0]["cells"] and snapshots[-1]["cells"] == []
    assert {n for c in snapshots[0]["cells"] for n in c["nets"]} == {1, 2}
    # Some net's committed route changed after being ripped up.
    first, last = {}, {}
    for e in trace.events:
        if e["type"] == "net_route_committed":
            first.setdefault(e["net"], e["cells"])
            last[e["net"]] = e["cells"]
    assert any(first[n] != last[n] for n in first)
    rip = types.index("net_rip_up")
    assert "net_route_committed" in types[rip:]


def test_negotiation_is_deterministic() -> None:
    def run():
        grid, requests, bounds = bottleneck()
        trace = TraceRecorder(TraceLevel.DETAILED)
        NegotiatedRouter(grid, requests, bounds=bounds, config=PnRConfig(), trace=trace).run()
        return trace.events

    assert run() == run()


# -- 61. routing failure ----------------------------------------------------------


def test_impossible_route_fails_cleanly() -> None:
    grid = Grid(height=3)
    goal = (6, 1, 0)
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        grid.place(goal[0] + dx, goal[1] + dy, goal[2] + dz, owner=99, kind=CellKind.BLOCKED)
    request = RouteRequest(1, endpoint(0, "out", (0, 1, 0)), (endpoint(1, "a", goal),))
    trace = TraceRecorder()
    outcome = NegotiatedRouter(
        grid, [request], bounds=Bounds((-3, 0, -3), (9, 2, 3)), config=PnRConfig(), trace=trace
    ).run()
    assert not outcome.success
    assert outcome.failure.reason == "unroutable"
    assert outcome.failure.search == "unreachable"
    assert outcome.failure.net == 1
    assert grid.route_of(1) == frozenset()  # the partial tree was released
    failed = [e for e in trace.events if e["type"] == "branch_route_failed"]
    assert failed and failed[0]["goal"] == list(goal)
    assert trace.events[-1]["type"] == "routing_failed"


def test_failed_pnr_still_has_a_complete_trace() -> None:
    _graph, nl = mapped("uint8_add.redc")
    result = place_and_route(nl, PnRConfig(max_astar_expansions=1, max_pnr_attempts=2))
    assert not result.success
    assert result.failure.reason == "unroutable" and result.failure.attempts == 2
    trace = result.trace.to_dict()
    assert trace["final"]["success"] is False
    assert trace["final"]["failure"]["reason"] == "unroutable"
    ends = [e for e in trace["events"] if e["type"] == "pnr_attempt_end"]
    assert [e["status"] for e in ends] == ["failed", "failed"]
    with pytest.raises(PnRError, match="after 2 attempt"):
        result.raise_for_failure()
    with pytest.raises(PnRError):
        result.to_physical_dict()


def test_placement_failure_is_reported_per_attempt() -> None:
    _graph, nl = mapped("uint8_fib.redc")  # registers need y >= 1 for BOTTOM pins
    result = place_and_route(nl, PnRConfig(grid_height=1, base_y=0, max_pnr_attempts=2))
    assert not result.success
    assert result.failure.stage == "placement"
    assert "does not fit" in result.failure.message
    failed = [e for e in result.trace.events if e["type"] == "placement_failed"]
    assert [e["attempt"] for e in failed] == [0, 1]


# -- 62. retry / design expansion -----------------------------------------------


def crossing_netlist() -> PhysicalNetlist:
    """Two nets that must cross: impossible in 2D at capacity 1 unless a route
    can go around the outside of the design."""
    nl = PhysicalNetlist()
    a = nl.add(InputPad.of("a", U8), label="a")
    b = nl.add(InputPad.of("b", U8), label="b")
    sub = nl.add(SUB, label="sub")
    out = nl.add(OutputPad.of("out", U8), label="out")
    nl.connect(a.terminal("out"), sub.terminal("b"))
    nl.connect(b.terminal("out"), sub.terminal("a"))
    nl.connect(sub.terminal("out"), out.terminal("in"))
    return nl


def test_retry_expands_the_design_until_it_routes() -> None:
    config = PnRConfig(
        grid_height=1, base_y=0, component_spacing=1, layer_gap=3, routing_margin=0,
        retry_spacing_growth=1, max_routing_iterations=8, max_pnr_attempts=3,
    )  # fmt: skip
    result = place_and_route(crossing_netlist(), config)
    assert result.success and result.attempts == 2
    events = result.trace.events
    begins = [e for e in events if e["type"] == "pnr_attempt_begin"]
    ends = [e for e in events if e["type"] == "pnr_attempt_end"]
    assert [b["attempt"] for b in begins] == [0, 1]
    assert [e["status"] for e in ends] == ["failed", "success"]
    assert ends[0]["failure"]["reason"] == "congestion"
    assert begins[1]["routing_margin"] > begins[0]["routing_margin"]
    bounds = [e["search_bounds"] for e in events if e["type"] == "routing_begin"]
    assert bounds[1]["volume"] > bounds[0]["volume"]
    assert any(c[0] < 0 or c[2] < bounds[0]["min"][2] for r in result.routes.values() for c in r.cells)
    # The second attempt re-placed every component from scratch.
    placed = [e for e in events if e["type"] == "component_placed"]
    assert len(placed) == 2 * len(result.netlist.instances)


# -- 63. real programs end to end -------------------------------------------------


@pytest.mark.parametrize("example", ["uint8_add.redc", "uint8_fib.redc"])
def test_real_program_end_to_end(example: str) -> None:
    graph, nl = mapped(example)
    before = nl.to_dict()
    result = place_and_route(nl)
    assert result.success
    nl.validate(complete=True)
    assert all(inst.origin is not None for inst in nl.instances.values())
    assert_legal_placement(result)
    grid = result.grid
    assert grid.overused() == []
    assert set(result.routes) == set(nl.nets)
    for net in nl.nets.values():
        routed = result.routes[net.id]
        assert routed.root == net.driver.outward
        assert {s.outward for s in net.sinks} <= routed.cell_set
        for cell in routed.cells:
            assert grid.kind_at(*cell) == CellKind.WIRE and grid.owner_at(*cell) == net.id
        assert grid.route_of(net.id) == routed.cell_set
    # P&R changed geometry only: connectivity, types and instances are intact.
    after = nl.to_dict()
    for entry in after["instances"]:
        entry["origin"] = None
    assert after == before
    if example == "uint8_fib.redc":
        regs = [i for i in nl.instances.values() if isinstance(i.component, Register)]
        for cls, pin in ((ClockSource, "clk"), (ResetSource, "rst")):
            (net,) = [n for n in nl.nets.values() if isinstance(n.driver.instance.component, cls)]
            assert {s.instance.id for s in net.sinks} == {r.id for r in regs}
            assert all(s.port.name == pin for s in net.sinks)
            assert len(result.routes[net.id].branches) == len(regs)
        # Functional simulation still agrees with the IR.
        ir_state, nl_state = graph.reset_state(), simulate.reset_state(nl)
        for start in [1] + [0] * 12 + [1] + [0] * 6:
            ir_out, ir_state = graph.step(ir_state, in_n=8, start=start)
            nl_out, nl_state = simulate.step(nl, nl_state, {"in_n": 8, "start": start})
            assert nl_out == ir_out


def test_metrics_are_consistent() -> None:
    _graph, nl = mapped("uint8_fib.redc")
    result = place_and_route(nl)
    m = result.metrics
    assert m["placement"]["component_count"] == len(nl.instances)
    assert m["placement"]["component_cells"] == sum(i.component.volume for i in nl.instances.values())
    assert m["routing"]["routed_nets"] == m["routing"]["net_count"] == len(nl.nets)
    assert m["routing"]["wire_cells"] == sum(r.length for r in result.routes.values())
    assert m["routing"]["overused_cells"] == 0 and m["routing"]["max_occupancy"] == 1
    assert m["routing"]["total_fanout"] == sum(n.fanout for n in nl.nets.values())
    wires = sum(1 for *_, k in result.grid.occupied_cells() if k == CellKind.WIRE)
    assert wires == m["final"]["wire_volume"]
    assert m["final"]["volume"] >= m["placement"]["bounds"]["volume"]


def test_config_validation() -> None:
    with pytest.raises(CompileError):
        PnRConfig(grid_height=0)
    with pytest.raises(CompileError):
        PnRConfig(component_spacing=-1)
    with pytest.raises(CompileError, match="trace level"):
        PnRConfig(trace_level="loud")
    assert PnRConfig(trace_level="search").trace_level is TraceLevel.SEARCH
    geometry = PnRConfig(component_spacing=2, retry_spacing_growth=3).attempt_geometry(2)
    assert geometry.component_spacing == 8
