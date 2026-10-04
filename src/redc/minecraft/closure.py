"""Synchronous timing closure: choose (or check) the physical clock period.

One logical clock cycle is one physical clock period ``P``.  Physical delay
decides how LONG ``P`` must be; it never changes what happens on which cycle.

Edge arithmetic (all game ticks; the clock source's rising edge ``k`` is at
``E_k = E_1 + (k - 1) * P``; ``clk(r)`` is the physical clock arrival at
register ``r``'s clock pin, exact because the clock tree never glitches):

* a register ``L`` launches at ``E_k + clk(L) + clk->Q`` (min / max);
* data inputs are changed by the test bench at ``E_k + input_offset``;
* capture register ``C`` samples at ``E_{k+1} + clk(C)``.

Setup, for every capture register ``C`` (``D_max`` = latest data arrival
relative to ``E_k``, launch clock arrival and clk->Q INCLUDED, counted once)::

    setup_slack(C) = P + clk(C) - setup(C) - D_max(C)            >= 0

which is the textbook ``P + clk_capture - clk_launch - setup - path_max``
with ``D_max = clk_launch + clk->Q_max + path_max``.

Hold (``D_min`` = earliest data change relative to ``E_k``)::

    hold_slack(C) = D_min(C) - clk(C) - hold(C)                   >= 0

``P`` does not appear: a slower clock NEVER fixes a hold violation, and
automatic period selection never looks at hold.

Outputs are sampled at ``E_{k+1}`` before that tick's events, so they must
have settled one game tick earlier (:data:`OUTPUT_SAMPLE_SETUP_GT`).  The
input offset is the smallest whole number of redstone ticks that keeps every
input -> register path hold-safe.

The automatic period is the smallest whole number of redstone ticks meeting
every setup constraint, both clock phases at least the longest repeater delay
in the clock tree (so no clock pulse is ever distorted), and both phases at
least one redstone tick -- plus ``margin_rt``.  A user-specified period is
checked, never silently increased.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..parser import CompileError
from .sta import EDGE_REPEATER, Endpoint, TimingAnalysis
from .timing import MIN_PULSE_GT, TIMING_MODEL
from .units import gt_to_rt, gt_to_rt_ceil, round_up_to_rt, rt_to_gt, ticks_record

OUTPUT_SAMPLE_SETUP_GT = 1
CLOSURE_MODEL = "redc-sync-closure-v1"


@dataclass(frozen=True)
class ClockConstraints:
    """``period_rt=None`` selects the period automatically."""

    period_rt: int | None = None
    margin_rt: int = 1
    max_skew_rt: int = 0

    def __post_init__(self) -> None:
        if self.period_rt is not None and self.period_rt < 2:
            raise CompileError("a clock period needs at least 2 redstone ticks (two phases of >= 1 rt)")
        if self.margin_rt < 0 or self.max_skew_rt < 0:
            raise CompileError("clock margin and maximum skew must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_rt": self.period_rt,
            "mode": "auto" if self.period_rt is None else "user",
            "margin_rt": self.margin_rt,
            "max_skew_rt": self.max_skew_rt,
        }


@dataclass
class EndpointTiming:
    endpoint: Endpoint
    capture_clock_gt: int
    setup_gt: int
    hold_gt: int | None
    reg_max: int | None
    reg_min: int | None
    in_max: int | None
    in_min: int | None
    data_max: int | None = None
    setup_slack: int | None = None
    hold_slack: int | None = None
    #: Which launch set decides setup: "registers" or "inputs".
    worst_from: str | None = None


@dataclass
class TimingClosure:
    analysis: TimingAnalysis
    constraints: ClockConstraints
    period_gt: int | None
    high_gt: int | None
    input_offset_gt: int
    recovery_gap_gt: int
    required_period_gt: int | None
    endpoints: list[EndpointTiming]
    failures: list[tuple[str, str]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def sequential(self) -> bool:
        return self.analysis.sequential

    def clock_arrivals(self) -> dict[str, int]:
        result = {}
        for reg in self.analysis.registers:
            arrival = self.analysis.clock_arrival(reg)
            assert arrival is not None
            result[reg.name] = arrival[0]
        return result


def close_timing(analysis: TimingAnalysis, constraints: ClockConstraints | None = None) -> TimingClosure:
    """Period selection / checking, setup and hold (see the module docstring)."""
    constraints = constraints or ClockConstraints()
    graph = analysis.graph
    regs = analysis.registers
    failures: list[tuple[str, str]] = []
    rows: list[EndpointTiming] = []
    clock_of: dict[str, int] = {}
    for reg in regs:
        arrival = analysis.clock_arrival(reg)
        assert arrival is not None and arrival[0] == arrival[1]
        clock_of[reg.name] = arrival[0]
    fr, fi = analysis.from_registers, analysis.from_inputs
    for endpoint in sorted(graph.endpoints, key=lambda e: (e.kind != "register", e.name, e.node)):
        node = endpoint.node

        def shift(value: int | None, extra: int) -> int | None:
            return None if value is None else value + extra

        if endpoint.register is not None:
            seq = endpoint.register.timing
            capture, setup, hold = clock_of[endpoint.register.name], seq.setup_gt, seq.hold_gt
        else:
            capture, setup, hold = 0, OUTPUT_SAMPLE_SETUP_GT, None
        rows.append(
            EndpointTiming(
                endpoint, capture, setup, hold,
                shift(fr.latest[node], endpoint.extra_max_gt), shift(fr.earliest[node], endpoint.extra_min_gt),
                shift(fi.latest[node], endpoint.extra_max_gt), shift(fi.earliest[node], endpoint.extra_min_gt),
            )
        )  # fmt: skip

    # Input offset: the earliest input change must not reach a register before
    # that register's hold window after the SAME edge closes.
    offset = 0
    for row in rows:
        if row.hold_gt is not None and row.in_min is not None:
            offset = max(offset, row.capture_clock_gt + row.hold_gt - row.in_min)
    offset = round_up_to_rt(offset)
    for row in rows:
        candidates = []
        if row.reg_max is not None:
            candidates.append((row.reg_max, "registers"))
        if row.in_max is not None:
            candidates.append((offset + row.in_max, "inputs"))
        if candidates:
            row.data_max, row.worst_from = max(candidates)
        lows = [v for v in (row.reg_min, None if row.in_min is None else offset + row.in_min) if v is not None]
        if row.hold_gt is not None and lows:
            row.hold_slack = min(lows) - row.capture_clock_gt - row.hold_gt

    clock_repeaters = 0
    if analysis.clock is not None:
        for node in range(graph.size):
            if analysis.clock.reached(node):
                for edge in graph.preds[node]:
                    if edge.kind == EDGE_REPEATER and analysis.clock.reached(edge.pred):
                        clock_repeaters = max(clock_repeaters, edge.min_gt)

    recovery_gap = 0
    if analysis.reset is not None:
        for reg in regs:
            if reg.reset is not None and analysis.reset.reached(reg.reset):
                late = analysis.reset.latest[reg.reset]
                assert late is not None
                recovery_gap = max(recovery_gap, late + reg.timing.recovery_gt - clock_of[reg.name])
    recovery_gap = round_up_to_rt(max(recovery_gap, MIN_PULSE_GT))

    period = high = required = None
    if regs:
        need = 2 * max(MIN_PULSE_GT, clock_repeaters)
        for row in rows:
            if row.data_max is not None:
                need = max(need, row.data_max + row.setup_gt - row.capture_clock_gt)
        required = round_up_to_rt(need)
        if constraints.period_rt is None:
            period = required + rt_to_gt(constraints.margin_rt)
        else:
            period = rt_to_gt(constraints.period_rt)
            if period < required:
                message = (
                    f"the requested clock period of {constraints.period_rt} rt is shorter than the "
                    f"{gt_to_rt_ceil(required)} rt setup closure needs (it is never increased silently)"
                )
                failures.append(("timing/setup", message))
        high = rt_to_gt(gt_to_rt(period) // 2)
        low = period - high
        if min(high, low) < max(MIN_PULSE_GT, clock_repeaters):
            message = f"clock phases {high}/{low} gt are shorter than the clock tree's {clock_repeaters} gt repeaters"
            failures.append(("timing/clock_pulse", message))
        for row in rows:
            if row.data_max is not None:
                row.setup_slack = period + row.capture_clock_gt - row.setup_gt - row.data_max
        bad = [r for r in rows if r.setup_slack is not None and r.setup_slack < 0]
        if bad and not any(code == "timing/setup" for code, _ in failures):
            failures.append(("timing/setup", f"{len(bad)} endpoint(s) miss setup at {period} gt"))
        holds = [r for r in rows if r.hold_slack is not None and r.hold_slack < 0]
        if holds:
            worst = min(holds, key=lambda r: (r.hold_slack, r.endpoint.name))
            message = (
                f"{len(holds)} register(s) miss hold; worst {worst.endpoint.name} by "
                f"{-(worst.hold_slack or 0)} gt (a longer clock period cannot fix this)"
            )
            failures.append(("timing/hold", message))
        if offset >= period:
            failures.append(
                ("timing/input_offset", f"inputs must change {offset} gt after the edge (hold), beyond the {period} gt period")
            )
        skew = (max(clock_of.values()) - min(clock_of.values())) if clock_of else 0
        if skew > rt_to_gt(constraints.max_skew_rt):
            failures.append(
                ("timing/clock_skew", f"clock skew {skew} gt exceeds the allowed {constraints.max_skew_rt} rt")
            )
    closure = TimingClosure(analysis, constraints, period, high, offset, recovery_gap, required, rows, failures)
    closure.report = timing_report(closure)
    return closure


def _path_record(closure: TimingClosure, row: EndpointTiming, *, setup: bool) -> dict[str, Any]:
    analysis, graph = closure.analysis, closure.analysis.graph
    endpoint = row.endpoint
    record: dict[str, Any] = {"endpoint": endpoint.name, "endpoint_kind": endpoint.kind}
    if setup:
        source = analysis.from_registers if row.worst_from == "registers" else analysis.from_inputs
        steps = graph.path(source, endpoint.node, latest=True)
        record["launch"] = "register" if row.worst_from == "registers" else "input"
        record["data_arrival"] = ticks_record(row.data_max)
        record["required"] = ticks_record(
            None if closure.period_gt is None else closure.period_gt + row.capture_clock_gt - row.setup_gt
        )
        record["slack"] = ticks_record(row.setup_slack)
        if row.worst_from == "inputs":
            record["input_offset"] = ticks_record(closure.input_offset_gt)
    else:
        from_regs = row.reg_min is not None and (
            row.in_min is None or row.reg_min <= closure.input_offset_gt + row.in_min
        )
        source = analysis.from_registers if from_regs else analysis.from_inputs
        steps = graph.path(source, endpoint.node, latest=False)
        record["launch"] = "register" if from_regs else "input"
        record["data_earliest"] = ticks_record(row.reg_min if from_regs else closure.input_offset_gt + (row.in_min or 0))
        record["required"] = ticks_record(row.capture_clock_gt + (row.hold_gt or 0))
        record["slack"] = ticks_record(row.hold_slack)
    record["steps"] = steps
    launch_reg = None
    if record["launch"] == "register" and steps:
        first = steps[0]["node"]
        launch_reg = next((r for r in analysis.registers if r.q == first), None)
    clock = analysis.clock
    if launch_reg is not None and clock is not None:
        record["launch_register"] = launch_reg.name
        record["launch_register_labels"] = launch_reg.labels
        record["launch_clock_arrival"] = ticks_record(clock.latest[launch_reg.clock])
        record["launch_clock_path"] = graph.path(clock, launch_reg.clock)
        record["clk_to_q"] = ticks_record(
            launch_reg.timing.clk_to_q_max_gt if setup else launch_reg.timing.clk_to_q_min_gt
        )
    if endpoint.register is not None and clock is not None:
        record["capture_register"] = endpoint.register.name
        record["capture_register_labels"] = endpoint.register.labels
        record["capture_clock_arrival"] = ticks_record(row.capture_clock_gt)
        record["capture_clock_path"] = graph.path(clock, endpoint.register.clock)
        record["setup" if setup else "hold"] = ticks_record(row.setup_gt if setup else row.hold_gt)
    return record


def timing_report(closure: TimingClosure) -> dict[str, Any]:
    """The JSON ``timing`` block of a design (see docs/minecraft-timing.md)."""
    analysis = closure.analysis
    rows = closure.endpoints
    report: dict[str, Any] = {
        "model": TIMING_MODEL,
        "closure_model": CLOSURE_MODEL,
        "units": "every duration is {gt: game ticks, rt: redstone ticks}; 1 rt = 2 gt",
        "world_fingerprint": analysis.fingerprint,
        "sequential": closure.sequential,
        "constraints": closure.constraints.to_dict(),
    }
    if closure.sequential:
        arrivals = closure.clock_arrivals()
        sinks = [
            {"register": reg.name, "labels": reg.labels, "arrival": ticks_record(arrivals[reg.name])}
            for reg in sorted(analysis.registers, key=lambda r: r.name)
        ]
        low = min(arrivals.values())
        high = max(arrivals.values())
        assert closure.period_gt is not None and closure.high_gt is not None
        report["clock"] = {
            "period": ticks_record(closure.period_gt),
            "high": ticks_record(closure.high_gt),
            "low": ticks_record(closure.period_gt - closure.high_gt),
            "mode": "auto" if closure.constraints.period_rt is None else "user",
            "required_period": ticks_record(closure.required_period_gt),
            "margin": ticks_record(rt_to_gt(closure.constraints.margin_rt)),
            "arrival_min": ticks_record(low),
            "arrival_max": ticks_record(high),
            "skew": ticks_record(high - low),
            "sinks": sinks,
        }
        report["input_offset"] = ticks_record(closure.input_offset_gt)
        report["reset_recovery_gap"] = ticks_record(closure.recovery_gap_gt)
        setup_rows = [r for r in rows if r.setup_slack is not None]
        hold_rows = [r for r in rows if r.hold_slack is not None]
        worst_setup = min(setup_rows, key=lambda r: (r.setup_slack, r.endpoint.kind, r.endpoint.name), default=None)
        worst_hold = min(hold_rows, key=lambda r: (r.hold_slack, r.endpoint.name), default=None)
        report["setup"] = {
            "worst_slack": None if worst_setup is None else ticks_record(worst_setup.setup_slack),
            "critical_path": None if worst_setup is None else _path_record(closure, worst_setup, setup=True),
            "violations": [
                {"endpoint": r.endpoint.name, "slack": ticks_record(r.setup_slack)}
                for r in sorted(setup_rows, key=lambda r: (r.setup_slack, r.endpoint.name))
                if (r.setup_slack or 0) < 0
            ][:100],
            "endpoints": len(setup_rows),
        }
        report["hold"] = {
            "worst_slack": None if worst_hold is None else ticks_record(worst_hold.hold_slack),
            "critical_path": None if worst_hold is None else _path_record(closure, worst_hold, setup=False),
            "violations": [
                {"endpoint": r.endpoint.name, "slack": ticks_record(r.hold_slack)}
                for r in sorted(hold_rows, key=lambda r: (r.hold_slack, r.endpoint.name))
                if (r.hold_slack or 0) < 0
            ][:100],
            "endpoints": len(hold_rows),
        }
    else:
        outputs = [r for r in rows if r.in_max is not None]
        worst = max(outputs, key=lambda r: (r.in_max, r.endpoint.name), default=None)
        report["combinational"] = {
            "settle": ticks_record(None if worst is None else worst.in_max),
            "earliest_change": ticks_record(min((r.in_min for r in outputs if r.in_min is not None), default=None)),
            "critical_path": None
            if worst is None
            else {
                "endpoint": worst.endpoint.name,
                "arrival": ticks_record(worst.in_max),
                "steps": analysis.graph.path(analysis.from_inputs, worst.endpoint.node, latest=True),
            },
        }
    report["closure"] = {
        "passed": closure.passed,
        "failures": [{"code": code, "message": message} for code, message in closure.failures],
    }
    return report


__all__ = [
    "CLOSURE_MODEL",
    "OUTPUT_SAMPLE_SETUP_GT",
    "ClockConstraints",
    "EndpointTiming",
    "TimingClosure",
    "close_timing",
    "timing_report",
]
