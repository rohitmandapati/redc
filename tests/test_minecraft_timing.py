"""Static timing analysis and synchronous closure on hand-built worlds, with
the redstone simulator as the dynamic cross-check (no RedC compilation).

Every world is laid out as straight dust lines (one per net, two blocks apart
so no two nets ever touch) joined by abstract components whose pins sit on the
lines -- so every arrival below can be derived by hand."""

from __future__ import annotations

import pytest

from redc.minecraft import (
    AbstractComponent,
    ComponentPin,
    ComponentTiming,
    MinecraftPhysicalDesign,
    Port,
    PortBit,
    RedstoneSimulator,
)
from redc.minecraft.blocks import redstone_wire, repeater, stone
from redc.minecraft.closure import ClockConstraints, close_timing
from redc.minecraft.sta import TimingError, analyze
from redc.minecraft.testbench import ClockSchedule, run_clocked, transaction_inputs
from redc.minecraft.timing import (
    MIN_PULSE_GT,
    CombinationalArc,
    SequentialTiming,
    repeater_delay_gt,
    repeater_settle_bound_gt,
    uniform_arcs,
)
from redc.minecraft.units import rt_to_gt

Coord = tuple[int, int, int]
CTQ, SETUP, HOLD = 4, 2, 2


class World:
    """A tiny world builder: straight dust lines, repeaters, abstract parts."""

    def __init__(self) -> None:
        self.blocks: dict = {}
        self.components: list[AbstractComponent] = []
        self.ports: list[Port] = []

    def line(self, z: int, length: int = 24, repeaters: dict[int, int] | None = None) -> None:
        """Dust along x = 0..length-1 at row z; ``repeaters``: x -> delay (rt), facing east."""
        for x in range(length):
            self.blocks[(x, 0, z)] = stone()
            delay = (repeaters or {}).get(x)
            self.blocks[(x, 1, z)] = repeater("east", delay) if delay else redstone_wire()

    def gate(self, name: str, table: str, ins: list[tuple[str, Coord]], out: Coord, delay: int, lo: int | None = None) -> None:
        pins = tuple(ComponentPin(n, "in", c, 1) for n, c in ins) + (ComponentPin("y", "out", out, 15),)
        arcs = tuple(CombinationalArc(n, "y", delay if lo is None else lo, delay) for n, _ in ins)
        self.components.append(AbstractComponent(name, "combinational", pins, ComponentTiming(arcs), (("y", table),)))

    def reg(self, name: str, d: Coord, clk: Coord, rst: Coord, q: Coord, init: int = 0) -> None:
        seq = SequentialTiming("clk", "d", "q", "rst", CTQ, CTQ, SETUP, HOLD, 2, 2)
        pins = (ComponentPin("d", "in", d, 1), ComponentPin("clk", "in", clk, 1), ComponentPin("rst", "in", rst, 1),
                ComponentPin("q", "out", q, 15))  # fmt: skip
        self.components.append(AbstractComponent(name, "dff", pins, ComponentTiming(sequential=seq), init=init))

    def port(self, name: str, direction: str, coord: Coord, role: str = "data") -> None:
        kind = "source" if direction == "in" else "dust"
        self.ports.append(Port(name, direction, (PortBit(kind, coord),), role=role))

    def design(self) -> MinecraftPhysicalDesign:
        return MinecraftPhysicalDesign(dict(self.blocks), tuple(self.components), tuple(self.ports), source_backend="test")


# -- combinational ----------------------------------------------------------------------------


def comb_world(repeater_rt: int = 2, arc: int = 4) -> MinecraftPhysicalDesign:
    w = World()
    w.line(0, 12, {5: repeater_rt})
    w.line(2, 12)
    w.port("a", "in", (0, 1, 0))
    w.gate("buf", "01", [("a", (11, 1, 0))], (0, 1, 2), arc)
    w.port("y", "out", (11, 1, 2))
    return w.design()


def test_combinational_critical_path_is_the_sum_of_physical_delays_and_matches_simulation() -> None:
    design = comb_world(repeater_rt=2, arc=4)
    closure = close_timing(analyze(design))
    report = closure.report["combinational"]
    assert report["settle"]["gt"] == repeater_delay_gt(2) + 4 == 8
    kinds = [s["kind"] for s in report["critical_path"]["steps"]]
    assert kinds[0] == "source" and kinds[-1] == "reader"
    assert any(s.get("delay_rt") == 2 for s in report["critical_path"]["steps"])  # the repeater is on the path
    sim = RedstoneSimulator(design)
    sim.run_until_stable()
    sim.set_input("a", 1)
    start = sim.time
    assert sim.run_until_stable() - start == 8 and sim.read_output("y") == 1


def test_fanout_sink_delays_are_independent() -> None:
    w = World()
    w.line(0, 20, {4: 1, 12: 3})
    w.line(2, 4)
    w.line(4, 4)
    w.port("a", "in", (0, 1, 0))
    w.gate("near", "01", [("a", (6, 1, 0))], (0, 1, 2), 2)  # after the first repeater only
    w.gate("far", "01", [("a", (16, 1, 0))], (0, 1, 4), 2)  # after both
    w.port("near_y", "out", (3, 1, 2))
    w.port("far_y", "out", (3, 1, 4))
    rows = {r.endpoint.name: r for r in close_timing(analyze(w.design())).endpoints}
    assert rows["near_y[0]"].in_max == rt_to_gt(1) + 2
    assert rows["far_y[0]"].in_max == rt_to_gt(1 + 3) + 2


def test_a_glitching_input_makes_a_repeater_count_its_pulse_extension() -> None:
    """XOR of a signal with its delayed copy glitches; a delay-4 repeater after
    it can stretch the glitch, so STA's latest arrival grows by the bound."""
    w = World()
    w.line(0, 8, {4: 2})
    w.line(2, 10, {5: 4})
    w.port("a", "in", (0, 1, 0))
    w.gate("xor", "0110", [("a", (2, 1, 0)), ("b", (7, 1, 0))], (0, 1, 2), 2)
    w.port("y", "out", (9, 1, 2))
    graph_rows = close_timing(analyze(w.design())).endpoints
    (row,) = graph_rows
    # a -> xor: 0 / 4 gt (two arrivals: the XOR may glitch) -> delay-4 repeater may stretch it.
    assert row.in_max == 4 + 2 + repeater_settle_bound_gt(4, single_transition=False)
    assert repeater_settle_bound_gt(4, single_transition=False) == 2 * rt_to_gt(4) - MIN_PULSE_GT
    # The simulator never settles later than STA's bound.
    sim = RedstoneSimulator(w.design())
    sim.run_until_stable()
    sim.set_input("a", 1)
    start = sim.time
    assert sim.run_until_stable() - start <= row.in_max


# -- sequential --------------------------------------------------------------------------------


def pipeline_world(*, clock_repeater_r2: int = 0, trunk_repeater: int = 0, logic: int = 4) -> MinecraftPhysicalDesign:
    """r1 -> NOT (``logic`` gt) -> r2 -> r1 (a 2-cycle toggling loop), one clock
    line (optionally a repeater before r2's tap, or on the shared trunk), one
    reset line, output = r2.q."""
    w = World()
    clock_repeaters = {}
    if trunk_repeater:
        clock_repeaters[2] = trunk_repeater
    if clock_repeater_r2:
        clock_repeaters[10] = clock_repeater_r2
    w.line(20, 18, clock_repeaters)  # clock: r1 taps x=6, r2 taps x=14
    w.line(24, 18)  # reset
    w.line(30, 8)  # r1.q -> NOT
    w.line(32, 8)  # NOT -> r2.d
    w.line(34, 8)  # r2.q -> r1.d, output
    w.port("clk", "in", (0, 1, 20), "clock")
    w.port("rst", "in", (0, 1, 24), "reset")
    w.reg("r1", d=(7, 1, 34), clk=(6, 1, 20), rst=(6, 1, 24), q=(0, 1, 30), init=1)
    w.gate("inv", "10", [("a", (7, 1, 30))], (0, 1, 32), logic)
    w.reg("r2", d=(7, 1, 32), clk=(14, 1, 20), rst=(14, 1, 24), q=(0, 1, 34), init=0)
    w.port("out", "out", (7, 1, 34))
    return w.design()


def rows_of(closure):
    return {r.endpoint.name: r for r in closure.endpoints}


def test_register_to_gate_to_register_setup_and_hold_by_hand() -> None:
    closure = close_timing(analyze(pipeline_world()))
    rows = rows_of(closure)
    assert rows["r2"].data_max == CTQ + 4  # clk(r1)=0 + clk->Q + NOT
    assert rows["r1"].data_max == CTQ  # r2.q straight back
    # P >= D_max + setup - clk(capture): 8 + 2 = 10, already whole rt.
    assert closure.required_period_gt == 10
    assert closure.period_gt == 10 + rt_to_gt(1)  # default margin 1 rt
    assert rows["r2"].setup_slack == closure.period_gt + 0 - SETUP - 8
    assert rows["r2"].hold_slack == (CTQ + 4) - 0 - HOLD
    assert closure.passed


def test_clock_to_q_is_counted_exactly_once() -> None:
    closure = close_timing(analyze(pipeline_world(logic=6)))
    path = closure.report["setup"]["critical_path"]
    assert path["capture_register"] == "r2" and path["launch_register"] == "r1"
    assert path["clk_to_q"]["gt"] == CTQ
    steps = path["steps"]
    assert steps[0]["arrival"]["gt"] == CTQ  # the launch q: clock arrival 0 + clk->Q
    assert steps[-1]["arrival"]["gt"] == path["data_arrival"]["gt"] == CTQ + 6
    assert sum(s["edge_delay"]["gt"] for s in steps) == 6  # the path itself adds only the logic


def test_asymmetric_clock_routes_are_measured_and_reported_as_skew() -> None:
    closure = close_timing(analyze(pipeline_world(clock_repeater_r2=3)))
    clock = closure.report["clock"]
    assert clock["arrival_min"]["gt"] == 0 and clock["arrival_max"]["gt"] == rt_to_gt(3)
    assert clock["skew"]["gt"] == rt_to_gt(3)
    arrivals = {s["register"]: s["arrival"]["gt"] for s in clock["sinks"]}
    assert arrivals == {"r1": 0, "r2": 6}
    assert [c for c, _ in closure.failures] == ["timing/clock_skew"]
    allowed = close_timing(analyze(pipeline_world(clock_repeater_r2=3)), ClockConstraints(max_skew_rt=3))
    assert allowed.passed
    rows = rows_of(allowed)
    # Setup uses capture arrival: r1 -> r2 gains the skew, r2 -> r1 pays it.
    assert rows["r2"].setup_slack == allowed.period_gt + 6 - SETUP - 8
    assert rows["r1"].setup_slack == allowed.period_gt + 0 - SETUP - (6 + CTQ)
    assert rows["r2"].hold_slack == 8 - 6 - HOLD == 0


def test_shared_clock_trunk_delays_every_downstream_register() -> None:
    clock = close_timing(analyze(pipeline_world(trunk_repeater=2))).report["clock"]
    assert {s["register"]: s["arrival"]["gt"] for s in clock["sinks"]} == {"r1": 4, "r2": 4}
    assert clock["skew"]["gt"] == 0


def test_setup_violation_with_a_user_period_and_a_longer_period_fixes_it() -> None:
    analysis = analyze(pipeline_world(logic=10))
    required_rt = close_timing(analysis).required_period_gt // 2
    tight = close_timing(analysis, ClockConstraints(period_rt=required_rt - 1))
    assert not tight.passed and tight.failures[0][0] == "timing/setup"
    assert tight.period_gt == rt_to_gt(required_rt - 1)  # never silently increased
    assert tight.report["setup"]["worst_slack"]["gt"] < 0
    relaxed = close_timing(analysis, ClockConstraints(period_rt=required_rt + 5))
    assert relaxed.passed and relaxed.period_gt == rt_to_gt(required_rt + 5)


def test_hold_violation_and_a_longer_period_does_not_fix_it() -> None:
    analysis = analyze(pipeline_world(clock_repeater_r2=4))  # capture clock 8 gt late; data min 8 gt
    for period in (None, 40, 200):
        closure = close_timing(analysis, ClockConstraints(period_rt=period, max_skew_rt=4))
        codes = [c for c, _ in closure.failures]
        assert "timing/hold" in codes, period
        assert rows_of(closure)["r2"].hold_slack == 8 - 8 - HOLD
        assert closure.report["hold"]["critical_path"]["capture_register"] == "r2"


def test_auto_period_is_the_minimum_safe_period_plus_margin() -> None:
    analysis = analyze(pipeline_world(logic=7))
    for margin in (0, 1, 3):
        closure = close_timing(analysis, ClockConstraints(margin_rt=margin))
        assert closure.period_gt == closure.required_period_gt + rt_to_gt(margin)
        assert closure.required_period_gt % 2 == 0
    exact = close_timing(analysis, ClockConstraints(period_rt=close_timing(analysis).required_period_gt // 2))
    assert exact.passed
    shorter = close_timing(analysis, ClockConstraints(period_rt=exact.period_gt // 2 - 1))
    assert not shorter.passed


def test_reset_and_clock_pins_are_classified_apart_from_data() -> None:
    analysis = analyze(pipeline_world())
    endpoints = {e.name: e.kind for e in analysis.graph.endpoints}
    assert endpoints == {"r1": "register", "r2": "register", "out[0]": "output"}
    regs = {r.name: r for r in analysis.registers}
    assert analysis.reset is not None and analysis.clock is not None
    assert all(analysis.reset.reached(r.reset) and not analysis.reset.reached(r.data) for r in regs.values())
    assert all(analysis.clock.reached(r.clock) and not analysis.clock.reached(r.data) for r in regs.values())
    # Data reaching a clock pin (a gated clock) or a reset pin is a structural
    # error, never analysed as data.
    for target in ("clock", "reset"):
        w = World()
        w.line(20)  # clock
        w.line(24)  # reset
        w.line(30, 8)  # ra.q
        w.line(36, 8)  # gate output
        w.port("clk", "in", (0, 1, 20), "clock")
        w.port("rst", "in", (0, 1, 24), "reset")
        w.reg("ra", d=(5, 1, 30), clk=(6, 1, 20), rst=(6, 1, 24), q=(0, 1, 30))
        source = (10, 1, 20) if target == "clock" else (10, 1, 24)
        w.gate("gate", "0001", [("a", source), ("b", (7, 1, 30))], (0, 1, 36), 2)
        if target == "clock":
            w.reg("rb", d=(3, 1, 30), clk=(7, 1, 36), rst=(14, 1, 24), q=(0, 1, 40))
        else:
            w.reg("rb", d=(3, 1, 30), clk=(14, 1, 20), rst=(7, 1, 36), q=(0, 1, 40))
        w.line(40, 4)
        with pytest.raises(TimingError) as info:
            analyze(w.design())
        assert info.value.code == ("timing/data_reaches_clock" if target == "clock" else "timing/data_reaches_reset")


def test_equal_critical_paths_are_reported_deterministically() -> None:
    def build() -> MinecraftPhysicalDesign:
        w = World()
        w.line(0, 8)
        w.line(2, 8)
        w.line(4, 8)
        w.port("a", "in", (0, 1, 0))
        w.port("b", "in", (0, 1, 2))
        w.gate("and", "0001", [("a", (7, 1, 0)), ("b", (7, 1, 2))], (0, 1, 4), 4)
        w.port("y", "out", (7, 1, 4))
        return w.design()

    reports = [close_timing(analyze(build())).report for _ in range(3)]
    assert reports[0] == reports[1] == reports[2]
    first = reports[0]["combinational"]["critical_path"]["steps"][0]
    assert first["coord"] == [0, 1, 0]  # ties go to the lowest node: input a


def test_stale_timing_is_detectable_after_a_route_changes() -> None:
    design = pipeline_world()
    closure = close_timing(analyze(design))
    changed = pipeline_world(trunk_repeater=1)
    assert closure.analysis.fingerprint == design.fingerprint() != changed.fingerprint()
    assert close_timing(analyze(changed)).report["world_fingerprint"] == changed.fingerprint()


# -- dynamic validation -------------------------------------------------------------------------


def toggling_reference(cycles: int) -> list[tuple[int, int]]:
    """(r1, r2) after reset and every edge: r1' = r2, r2' = NOT r1."""
    r1, r2, states = 1, 0, []
    for _ in range(cycles):
        states.append((r1, r2))
        r1, r2 = r2, 1 - r1
    return states


@pytest.mark.parametrize("world", [{}, {"clock_repeater_r2": 2}, {"trunk_repeater": 3}])
def test_simulation_at_the_closed_period_follows_one_logical_cycle_per_clock(world: dict) -> None:
    design = pipeline_world(**world)
    closure = close_timing(analyze(design), ClockConstraints(max_skew_rt=4))
    assert closure.passed
    run = run_clocked(design, ClockSchedule.from_closure(closure), transaction_inputs({}, 8, has_start=False))
    assert not run.violations and not run.warnings
    assert [obs.outputs["out"] for obs in run.observations] == [r2 for _r1, r2 in toggling_reference(8)]
    assert run.capture_errors(closure.clock_arrivals()) == []
    assert run.stats["cycles"] == 8


def test_simulation_detects_a_deliberately_too_fast_clock() -> None:
    """Data r1 -> r2 needs CTQ + 20 = 24 gt (+ 2 setup).  At a 24 gt period it
    lands in the very tick of the next edge: inside the (setup, hold) window
    -- stimulus runs before device events, so it is reported as the data
    changing right after the edge (hold).  At 20 gt it lands
    after the next edge, outside setup AND hold -- silently a two-cycle path,
    caught as a logical mismatch against the one-cycle reference."""
    design = pipeline_world(logic=20)
    closure = close_timing(analyze(design))
    assert closure.required_period_gt == 26
    reference = [r2 for _r1, r2 in toggling_reference(8)]
    on_edge = run_clocked(design, ClockSchedule.from_closure(closure, period_gt=24), transaction_inputs({}, 8, has_start=False))
    assert {v.kind for v in on_edge.violations} & {"setup", "hold"}
    late = run_clocked(design, ClockSchedule.from_closure(closure, period_gt=20), transaction_inputs({}, 8, has_start=False))
    assert [obs.outputs["out"] for obs in late.observations] != reference
    safe = run_clocked(design, ClockSchedule.from_closure(closure), transaction_inputs({}, 8, has_start=False))
    assert not safe.violations and [obs.outputs["out"] for obs in safe.observations] == reference


def test_uniform_arcs_helper() -> None:
    arcs = uniform_arcs(["a", "b"], ["y"], 6)
    assert [(a.from_pin, a.to_pin, a.min_gt, a.max_gt) for a in arcs] == [("a", "y", 6, 6), ("b", "y", 6, 6)]
