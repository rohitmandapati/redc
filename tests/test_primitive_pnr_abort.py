"""Early abort of nonviable P&R attempts: divergence, stagnation, effort backstop.

The rules only decide WHEN an attempt is given up; the legality criteria are
untouched, and an aborted attempt is retried with wider spacing as usual.
"""

from __future__ import annotations

import pytest

from redc import compile_source
from redc.physical_primitive import PadInterfacePolicy, place_and_route_graph
from redc.physical_primitive.geometry import Bounds, Direction, below
from redc.physical_primitive.grid import BlockGrid, Conflict, PinSite
from redc.physical_primitive.pnr import (
    NegotiatedRedstoneRouter,
    PrimitivePnRConfig,
    PrimitiveTraceRecorder,
    RouteRequest,
    RouteTree,
)

FULL_ADDER = "uint2 main(bool a, bool b, bool c) { return (uint2)a + (uint2)b + (uint2)c; }"


def scripted_router(counts: list[int], **config) -> tuple[NegotiatedRedstoneRouter, PrimitiveTraceRecorder]:
    """A router whose grid reports ``counts[i]`` conflicts after iteration i
    and whose rerouting is a no-op, so only the abort decision is exercised."""
    sites = [
        PinSite((0, 1, 0), 0, "y", "out", Direction.EAST, 15, 1),
        PinSite((8, 1, 0), 1, "a", "in", Direction.WEST, 1, 1),
    ]
    grid = BlockGrid(max_y=6)
    for s in sites:
        grid.place(s.instance, [below(s.cell)], [], [s])
    request = RouteRequest(1, "data", sites[0], (sites[1],))
    trace = PrimitiveTraceRecorder("basic")
    router = NegotiatedRedstoneRouter(
        grid, [request], bounds=Bounds((-2, 1, -3), (10, 4, 3)), config=PrimitivePnRConfig(**config), trace=trace
    )
    script = iter(counts)

    def conflicts() -> list[Conflict]:
        n = next(script, 0)
        return [Conflict("adjacent_signals", ((0, 1, i), (0, 1, i + 1)), (1, 2)) for i in range(n)]

    grid.conflicts = conflicts  # type: ignore[method-assign]
    router.route_net = lambda req: RouteTree(req.net, req.driver, req.driver.cell, ())  # type: ignore[method-assign]
    router.routes[1] = RouteTree(1, sites[0], sites[0].cell, ())
    return router, trace


def aborts(trace: PrimitiveTraceRecorder) -> list[dict]:
    return [e for e in trace.events if e["type"] == "routing_aborted"]


def test_a_conflict_spike_far_above_the_best_seen_aborts_as_diverged() -> None:
    router, trace = scripted_router([600, 80, 9, 6, 7, 300, 2000])
    outcome = router._negotiate(1)
    assert not outcome.success and outcome.failure is not None
    assert outcome.failure.reason == "diverged"
    assert outcome.failure.iteration == 5  # 300 > 4 * 6 + 50: the first spike
    (event,) = aborts(trace)
    assert event["reason"] == "diverged" and "300 conflicts after reaching 6" in event["detail"]
    assert trace.events[-1]["type"] == "routing_failed"
    assert trace.events[-1]["failure"]["reason"] == "diverged"


def test_small_fluctuations_do_not_count_as_divergence() -> None:
    # 2 -> 40 is a 20x jump but stays within the absolute margin of 50.
    router, _trace = scripted_router([90, 10, 2, 40, 3, 1, 0])
    outcome = router._negotiate(1)
    assert outcome.success


def test_no_new_best_for_n_iterations_aborts_as_stagnated() -> None:
    router, trace = scripted_router([50, 9, 9, 10, 9, 11, 9, 12], abort_stagnation_iterations=3)
    outcome = router._negotiate(1)
    assert outcome.failure is not None and outcome.failure.reason == "stagnated"
    assert outcome.failure.iteration == 4  # best 9 set at iteration 1; 3 iterations without beating it
    assert aborts(trace)[0]["reason"] == "stagnated"


@pytest.mark.parametrize("rule", ["abort_divergence_factor", "abort_stagnation_iterations"])
def test_each_rule_can_be_disabled(rule: str) -> None:
    router, trace = scripted_router([50, 9, 9, 10, 9, 300, 0], **{rule: 0, "abort_stagnation_iterations": 0})
    outcome = router._negotiate(1)
    if rule == "abort_divergence_factor":
        assert outcome.success and not aborts(trace)
    else:
        assert outcome.failure is not None and outcome.failure.reason == "diverged"


def test_attempt_effort_backstop_aborts_every_attempt_and_keeps_a_full_trace() -> None:
    trace = PrimitiveTraceRecorder("basic")
    config = PrimitivePnRConfig(attempt_effort_per_net=1, attempt_effort_min=1, max_pnr_attempts=2)
    _nl, _m, result = place_and_route_graph(
        compile_source(FULL_ADDER), config, interface=PadInterfacePolicy(), trace=trace
    )
    assert not result.success
    assert result.failure is not None and result.failure.reason == "effort_exhausted"
    assert result.attempts == 2
    reasons = [e["reason"] for e in aborts(trace)]
    assert reasons == ["effort_exhausted", "effort_exhausted"]  # one per attempt, none silently dropped
    ends = [e for e in trace.events if e["type"] == "pnr_attempt_end"]
    assert [e["status"] for e in ends] == ["failed", "failed"]
    assert all(e["failure"]["reason"] == "effort_exhausted" for e in ends)
    assert trace.final is not None and trace.final["success"] is False
    # The next attempt still spread out, as the normal retry policy says.
    begins = [e for e in trace.events if e["type"] == "pnr_attempt_begin"]
    assert begins[1]["channel_width"] > begins[0]["channel_width"]


def test_abort_rules_do_not_disturb_designs_that_converge() -> None:
    for source in (FULL_ADDER, "uint4 main(uint4 a, uint4 b) { return a + b; }"):
        _nl, _m, result = place_and_route_graph(
            compile_source(source), PrimitivePnRConfig(), interface=PadInterfacePolicy()
        )
        assert result.success and result.attempts == 1 and result.violations == []
        effort = result.metrics["routing"]["effort"]
        assert effort["expansions"] == sum(i["expansions"] for i in effort["iterations"])
        assert effort["searches"] >= len(result.routes)
