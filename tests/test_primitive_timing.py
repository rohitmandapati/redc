"""The physical-primitive backend's timing closure and physical simulation:

    Graph  ==  PrimitiveSimulator  ==  RedstoneSimulator (materialized P&R output)

for combinational designs (exhaustively where small, randomized otherwise),
and logical-cycle-by-cycle agreement for sequential designs run at the
physically closed clock period -- plus clock balancing, user periods, stale
reports and the block-accurate reference library."""

from __future__ import annotations

import dataclasses
import functools
import itertools
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from redc import compile_source
from redc.ir import Graph
from redc.minecraft import RedstoneSimulator
from redc.minecraft.closure import ClockConstraints, close_timing
from redc.minecraft.sta import analyze
from redc.minecraft.testbench import ClockSchedule, run_clocked, transaction_inputs
from redc.minecraft.timing import repeater_delay_gt
from redc.parser import CompileError
from redc.physical_primitive import (
    PadInterfacePolicy,
    PrimitivePnRConfig,
    PrimitiveSimulator,
    PrimitiveTraceRecorder,
    place_and_route_graph,
)
from redc.physical_primitive.materialize import materialize_design, register_probe
from redc.physical_primitive.pnr.clock import sink_arrivals
from redc.physical_primitive.pnr.timing import validate_by_simulation
from redc.physical_primitive.redstone import ElementKind
from redc.physical_primitive.technology.structures import reference_library

PADS = PadInterfacePolicy()
COUNTER = "uint2 main(uint2 n) { uint2 x = 0; for (uint2 i = 0; i < n; i++) { x = x + 1; } return x; }"
SMALL = {
    "not": "bool main(bool a) { return !a; }",
    "and": "bool main(bool a, bool b) { return a & b; }",
    "xor": "bool main(bool a, bool b) { return a ^ b; }",
    "mux": "uint2 main(bool s, uint2 a, uint2 b) { return s ? a : b; }",
    "half_adder": "uint2 main(bool a, bool b) { return (uint2)a + (uint2)b; }",
    "add3": "uint3 main(uint3 a, uint3 b) { return a + b; }",
    "compare": "bool main(uint3 a, uint3 b) { return a < b; }",
    "signed_neg": "int3 main(int3 a) { return -a; }",
}


@functools.cache
def pnr(source: str, top: str = "main", **config):
    graph = compile_source(source, top=top)
    recorder = PrimitiveTraceRecorder("basic")
    netlist, mapped, result = place_and_route_graph(
        graph, PrimitivePnRConfig(**config), interface=PADS, trace=recorder
    )
    return graph, netlist, mapped, result, recorder


def all_inputs(graph: Graph):
    ports = [(p["name"], graph.nodes[p["node"]]["type"]) for p in graph.inputs]
    ranges = []
    for _name, typ in ports:
        width, signed = typ["width"], typ["signed"]
        ranges.append(range(-(1 << (width - 1)), 1 << (width - 1)) if signed else range(1 << width))
    for values in itertools.product(*ranges):
        yield dict(zip((n for n, _ in ports), values))


# -- combinational: three-way differential ----------------------------------------------------


@pytest.mark.parametrize("name", sorted(SMALL))
def test_small_designs_agree_exhaustively_in_all_three_simulators(name: str) -> None:
    graph, netlist, _mapped, result, _rec = pnr(SMALL[name])
    assert result.success, result.failure
    world = result.timing.world
    assert world.mode == "abstract-components"
    primitive = PrimitiveSimulator(netlist)
    redstone = RedstoneSimulator(world)
    settle = result.timing.closure.report["combinational"]["settle"]["gt"]
    for vector in all_inputs(graph):
        expected = graph.evaluate(**vector)
        assert primitive.evaluate(**vector) == expected
        start = redstone.time
        assert redstone.evaluate(**vector) == expected, vector
        assert redstone.time - start <= settle  # never slower than static timing says


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.tuples(st.integers(0, 255), st.integers(0, 255)), min_size=1, max_size=6))
def test_uint8_adder_agrees_on_random_input_sequences(vectors: list[tuple[int, int]]) -> None:
    graph, _netlist, _mapped, result, _rec = pnr(Path("examples/uint8_add.redc").read_text(), top="add")
    assert result.success
    sim = RedstoneSimulator(result.timing.world)
    for a, b in vectors:  # each vector starts from the previous one's settled state
        assert sim.evaluate(in_a=a, in_b=b) == graph.evaluate(in_a=a, in_b=b)


def test_validation_catches_a_world_that_computes_the_wrong_function() -> None:
    _graph, _netlist, mapped, result, _rec = pnr(SMALL["and"])
    world = result.timing.world
    broken = dataclasses.replace(
        world,
        components=tuple(
            dataclasses.replace(c, truth_tables=(("y", "0111"),)) if dict(c.labels).get("kind") == "and" else c
            for c in world.components
        ),
    )
    from redc.minecraft.connectivity import compile_world

    report = validate_by_simulation(mapped, broken, compile_world(broken), result.timing.closure, PrimitivePnRConfig())
    assert not report["validated"]
    assert report["failures"][0]["code"] == "simulation/logic_mismatch"


# -- sequential: logical cycles ------------------------------------------------------------------


def test_counter_matches_graph_state_after_every_physical_clock_edge() -> None:
    graph, netlist, _mapped, result, _rec = pnr(COUNTER)
    assert result.success, result.failure
    closure = result.timing.closure
    primitive = PrimitiveSimulator(netlist)
    for n in range(4):
        cycles = transaction_inputs({"in_n": n}, 8, has_start=True)
        run = run_clocked(result.timing.world, ClockSchedule.from_closure(closure), cycles)
        assert not run.violations and not run.warnings
        assert run.capture_errors(closure.clock_arrivals()) == []
        state = graph.reset_state()
        for obs, stimulus in zip(run.observations, cycles):
            outputs, next_state = graph.step(state, **stimulus)
            assert obs.outputs == outputs, (n, obs.cycle)
            bits = {int(name[3:]): value for name, value in obs.probes.items() if name.startswith("reg")}
            assert primitive.state_to_ir(bits) == state, (n, obs.cycle)  # physical register state == IR state
            state = next_state


def test_counter_done_asserts_on_the_same_logical_cycle_as_graph_run() -> None:
    graph, _netlist, _mapped, result, _rec = pnr(COUNTER)
    closure = result.timing.closure
    for n in range(4):
        cycles = transaction_inputs({"in_n": n}, 12, has_start=True)
        run = run_clocked(result.timing.world, ClockSchedule.from_closure(closure), cycles, stop_when_done=True)
        assert run.done_cycle is not None
        reference = graph.reset_state()
        for cycle, stimulus in enumerate(cycles):
            outputs, reference = graph.step(reference, **stimulus)
            if cycle > 0 and outputs["done"]:
                break
        assert run.done_cycle == cycle
        assert run.observations[-1].outputs["result"] == graph.run(in_n=n)["result"]


def test_clock_balancing_is_physically_realized_in_the_world() -> None:
    _graph, _netlist, _mapped, result, rec = pnr(COUNTER)
    balance = result.timing.balance
    assert balance is not None and balance.success and balance.skew_before_rt > 0 == balance.skew_after_rt
    route = result.realized[balance.net]
    world = result.timing.world
    for change in balance.changes:
        coord = tuple(change["coord"])
        element = next(e for e in route.elements if e.coord == coord)
        assert element.kind is ElementKind.REPEATER and element.setting == change["to_rt"]
        block = world.blocks[coord]
        assert block.id == "minecraft:repeater" and block.get("delay") == change["to_rt"]
    # Every register sees the same arrival -- statically ...
    request = result.routed.requests[balance.net]
    arrivals = {a.arrival_rt for a in sink_arrivals(route, request)}
    assert len(arrivals) == 1
    clock = result.timing.closure.report["clock"]
    assert clock["skew"]["gt"] == 0 and clock["arrival_max"]["rt"] == arrivals.pop()
    # ... and dynamically: one clock edge reaches every register clk pin in the same tick.
    sim = RedstoneSimulator(world, trace="events")
    sim.run_until_stable()
    start = sim.time
    sim.set_input("clk", 1)
    sim.run_until_stable()
    edges = [e["t"] - start for e in sim.events if e["type"] == "register_clock_edge"]
    assert len(edges) == len(result.timing.closure.analysis.registers) and len(set(edges)) == 1
    assert edges[0] == clock["arrival_max"]["gt"]
    # The trace replaces the clock route with the balanced one.
    balanced = [e for e in rec.events if e["type"] == "clock_balanced"]
    assert balanced and balanced[-1]["realized"] == route.to_dict()


def test_balanced_repeaters_only_change_the_clock_net() -> None:
    _graph, _netlist, _mapped, result, _rec = pnr(COUNTER)
    clock = result.timing.balance.net
    for net, route in result.realized.items():
        for element in route.repeaters:
            assert element.setting in (1, 2, 3, 4)
            if net != clock:
                assert element.setting == 1
    assert repeater_delay_gt(4) == 8


def test_clock_taps_give_every_register_an_exclusive_straight_entry() -> None:
    _graph, _netlist, _mapped, result, _rec = pnr(COUNTER)
    net = result.timing.balance.net
    request = result.routed.requests[net]
    assert request.taps and len(request.taps) == len(request.sinks)
    tree = result.routes[net]
    for sink in request.sinks:
        tap = request.tap(sink)
        fx, _, fz = sink.facing.vector
        assert tap == tuple((sink.cell[0] + fx * k, sink.cell[1], sink.cell[2] + fz * k) for k in range(1, len(tap) + 1))
        cursor = sink.cell
        for block in tap:  # the route enters the pin straight down its tap
            assert tree.parent[cursor] == block
            cursor = block


def test_a_user_period_is_used_exactly_or_fails_without_being_raised() -> None:
    _graph, _netlist, _mapped, auto, _rec = pnr(COUNTER)
    required_rt = auto.timing.closure.required_period_gt // 2
    _g, _n, _m, generous, _r = pnr(COUNTER, clock_period_rt=required_rt + 7)
    assert generous.success and generous.timing.closure.report["clock"]["period"]["rt"] == required_rt + 7
    assert generous.timing.closure.report["clock"]["mode"] == "user"
    _g, _n, _m, tight, rec = pnr(COUNTER, clock_period_rt=required_rt - 2)
    assert not tight.success
    assert tight.failure.code == "timing/setup" and not tight.failure.retryable
    assert tight.attempts == 1  # a deterministic timing failure is not retried
    assert tight.timing.closure.period_gt == 2 * (required_rt - 2)
    assert any(e["type"] == "timing_failed" for e in rec.events)
    assert rec.final["timing"]["closure"]["passed"] is False


def test_stale_timing_is_refused_when_the_design_changed_after_analysis() -> None:
    _graph, _netlist, _mapped, result, _rec = pnr(COUNTER)
    clock = result.timing.balance.net
    unbalanced = dataclasses.replace(
        result.realized[clock],
        elements=tuple(
            dataclasses.replace(e, setting=1) if e.kind is ElementKind.REPEATER else e
            for e in result.realized[clock].elements
        ),
    )
    stale = dataclasses.replace(result)
    stale.legalized = dataclasses.replace(result.legalized, realized={**result.realized, clock: unbalanced})
    with pytest.raises(CompileError, match="stale timing"):
        stale.to_design_dict()
    assert result.to_design_dict()["timing"]["world_fingerprint"] == result.timing.world.fingerprint()


def test_the_design_file_carries_the_world_timing_and_simulation_reports() -> None:
    _graph, _netlist, mapped, result, _rec = pnr(COUNTER)
    design = result.to_design_dict()
    assert design["simulation"]["validated"] is True
    assert design["simulation"]["mode"] == design["world"]["mode"] == "abstract-components"
    assert design["timing"]["closure"]["passed"] is True
    assert design["timing"]["clock"]["skew"]["gt"] == 0
    assert design["clock_tree"]["skew_after"]["gt"] == 0
    rebuilt = materialize_design(mapped, result.realized, source=result.trace.source)
    assert design["world"] == rebuilt.to_dict()
    assert all(register_probe(r) in {p["name"] for p in design["world"]["probes"]}
               for r in (i.id for i in mapped.instances if i.kind.value == "register_bit"))  # fmt: skip


def test_identical_builds_report_identical_timing_and_critical_paths() -> None:
    first = pnr.__wrapped__(COUNTER)[3]
    second = pnr.__wrapped__(COUNTER)[3]
    assert first.timing.closure.report == second.timing.closure.report
    assert first.timing.simulation == second.timing.simulation


def test_sta_and_route_level_delays_agree_on_the_clock_tree() -> None:
    """Route records (repeater settings summed) and STA (geometry -> timing
    graph) compute every clock arrival independently -- they must agree."""
    _graph, _netlist, _mapped, result, _rec = pnr(COUNTER)
    net = result.timing.balance.net
    route_level = {a.instance: 2 * a.arrival_rt for a in sink_arrivals(result.realized[net], result.routed.requests[net])}
    sta = {dict(s["labels"])["instance"]: s["arrival"]["gt"] for s in result.timing.closure.report["clock"]["sinks"]}
    assert route_level == sta


# -- block-accurate reference cells ----------------------------------------------------------------


@pytest.mark.parametrize(
    "source", [SMALL["not"], "bool main(bool a, bool b) { return a | b; }", "bool main(bool a, bool b) { return !(a | b); }"]
)
def test_reference_cells_give_a_block_accurate_end_to_end_simulation(source: str) -> None:
    graph = compile_source(source)
    _netlist, _mapped, result = place_and_route_graph(graph, PrimitivePnRConfig(), interface=PADS, library=reference_library())
    assert result.success, result.failure
    world = result.timing.world
    assert world.mode == "block-accurate" and not world.components
    assert result.timing.simulation["mode"] == "block-accurate" and result.timing.simulation["validated"]
    sim = RedstoneSimulator(world)
    for vector in all_inputs(graph):
        assert sim.evaluate(**vector) == graph.evaluate(**vector)


# -- uint8_fib ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fib():
    return pnr(Path("examples/uint8_fib.redc").read_text(), top="fib")


def test_uint8_fib_closes_timing_with_an_automatic_clock(fib) -> None:
    _graph, _netlist, _mapped, result, _rec = fib
    assert result.success, result.failure
    report = result.timing.closure.report
    assert report["clock"]["mode"] == "auto" and report["clock"]["skew"]["gt"] == 0
    assert report["setup"]["worst_slack"]["gt"] >= 0 and report["hold"]["worst_slack"]["gt"] >= 0
    assert report["closure"]["passed"] and result.timing.simulation["validated"]


@pytest.mark.parametrize("n", [2, 7])
def test_uint8_fib_finishes_on_the_same_logical_cycle_as_the_primitive_simulator(fib, n: int) -> None:
    graph, netlist, _mapped, result, _rec = fib
    closure = result.timing.closure
    cycles = transaction_inputs({"in_n": n}, 16, has_start=True)
    run = run_clocked(result.timing.world, ClockSchedule.from_closure(closure), cycles, stop_when_done=True)
    primitive = PrimitiveSimulator(netlist)
    state = primitive.reset_state()
    done = None
    for obs in run.observations:
        outputs, nxt = primitive.step(state, **cycles[obs.cycle])
        assert obs.outputs == outputs
        assert all(obs.probes[register_probe(i)] == bit for i, bit in state.items() if register_probe(i) in obs.probes)
        if obs.cycle > 0 and outputs["done"]:
            done = obs.cycle
            break
        state = nxt
    assert done is not None and run.done_cycle == done
    assert run.observations[-1].outputs["result"] == graph.run(in_n=n)["result"] == primitive.run(in_n=n)["result"]
    assert not run.violations and run.capture_errors(closure.clock_arrivals()) == []


def test_uint8_fib_too_fast_clock_is_caught_by_the_simulator(fib) -> None:
    graph, _netlist, _mapped, result, _rec = fib
    closure = result.timing.closure
    fast = close_timing(analyze(result.timing.world), ClockConstraints(period_rt=closure.required_period_gt // 4))
    assert not fast.passed and fast.failures[0][0] == "timing/setup"
    schedule = ClockSchedule.from_closure(closure, period_gt=closure.required_period_gt // 2)
    run = run_clocked(result.timing.world, schedule, transaction_inputs({"in_n": 9}, 12, has_start=True))
    state = graph.reset_state()
    mismatch = bool(run.violations)
    for obs in run.observations:
        outputs, state = graph.step(state, **transaction_inputs({"in_n": 9}, 12, has_start=True)[obs.cycle])
        mismatch = mismatch or obs.outputs != outputs
    assert mismatch
