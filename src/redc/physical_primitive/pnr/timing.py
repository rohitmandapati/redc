"""Timing closure and physical simulation validation of a legalized primitive design.

    legalized routes
      -> clock-tree analysis + physical balancing   (:mod:`.clock`)
      -> independent re-verification of the changed clock route
      -> materialize a MinecraftPhysicalDesign      (:mod:`redc.physical_primitive.materialize`)
      -> static timing analysis + closure          (:mod:`redc.minecraft.sta`, :mod:`redc.minecraft.closure`)
      -> redstone simulation at the chosen clock    (:mod:`redc.minecraft.testbench`)
         compared, logical cycle by logical cycle, with the PrimitiveSimulator

A design only succeeds if every step passes.  Failures carry explicit codes
(``timing/clock_balance``, ``timing/setup``, ``timing/hold``,
``timing/clock_skew``, ``simulation/logic_mismatch``,
``simulation/cycle_mismatch``, ``simulation/timing_violation``, ...); only
``timing/clock_balance`` is worth another, roomier P&R attempt.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...minecraft.closure import ClockConstraints, TimingClosure, close_timing
from ...minecraft.connectivity import CompiledWorld, SimulationError, compile_world
from ...minecraft.design import MinecraftPhysicalDesign
from ...minecraft.playback import record_playback
from ...minecraft.simulator import SIMULATION_MODEL, RedstoneSimulator
from ...minecraft.sta import TimingError, analyze
from ...minecraft.testbench import ClockSchedule, run_clocked, transaction_inputs
from ...minecraft.units import ticks_record
from ..materialize import component_name, materialize_design, register_probe
from ..netlist import PrimitiveKind
from ..physical import PrimitivePhysicalNetlist
from ..simulate import PrimitiveSimulator
from .clock import ClockBalance, balance_clock_route
from .config import PrimitivePnRConfig
from .legalize import RealizedRoute
from .routing import RouteRequest
from .trace import PrimitiveTraceRecorder
from .verify import Violation, verify_design

#: Failures a fresh, roomier P&R attempt may fix.
RETRYABLE = frozenset({"timing/clock_balance"})


@dataclass
class TimingStage:
    """Everything the timing / simulation stage produced (of one attempt)."""

    realized: dict[int, RealizedRoute]
    balance: ClockBalance | None = None
    world: MinecraftPhysicalDesign | None = None
    closure: TimingClosure | None = None
    simulation: dict[str, Any] | None = None
    #: A short recorded simulation for the viewer (trace only; never verified against).
    playback: dict[str, Any] | None = None
    failure: tuple[str, str] | None = None
    violations: list[Violation] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.failure is None

    @property
    def retryable(self) -> bool:
        return self.failure is not None and self.failure[0] in RETRYABLE


def clock_constraints(config: PrimitivePnRConfig) -> ClockConstraints:
    return ClockConstraints(config.clock_period_rt, config.clock_margin_rt, config.max_clock_skew_rt)


def run_timing_stage(
    mapped: PrimitivePhysicalNetlist,
    requests: Mapping[int, RouteRequest],
    realized: Mapping[int, RealizedRoute],
    config: PrimitivePnRConfig,
    trace: PrimitiveTraceRecorder | None,
    *,
    max_y: int,
) -> TimingStage:
    """Balance the clock, close timing and validate by simulation (see the module docstring)."""
    stage = TimingStage(dict(realized))

    def emit(phase: str, type_: str, **data: Any) -> None:
        if trace is not None:
            trace.emit(phase, type_, **data)

    clock_nets = [n for n, r in requests.items() if r.role == "clock"]
    if clock_nets:
        (net,) = clock_nets
        balance = balance_clock_route(stage.realized[net], requests[net], max_skew_rt=config.max_clock_skew_rt)
        stage.balance = balance
        emit(
            "timing", "clock_tree_analyzed", net=net, sinks=[a.to_dict() for a in balance.before],
            skew=ticks_record(2 * balance.skew_before_rt),
        )  # fmt: skip
        if not balance.success:
            emit("timing", "clock_balance_failed", net=net, message=balance.message)
            stage.failure = ("timing/clock_balance", balance.message)
            return stage
        if balance.changes:
            stage.realized[net] = balance.route
            emit("timing", "clock_balanced", **balance.to_dict(), realized=balance.route.to_dict())
            # The changed clock route is re-verified like any other route.
            stage.violations = verify_design(mapped, requests, stage.realized, max_y=max_y)
            if stage.violations:
                first = stage.violations[0]
                stage.failure = ("timing/clock_balance", f"the balanced clock route is illegal: {first.message}")
                return stage
    try:
        stage.world = materialize_design(mapped, stage.realized, source=trace.source if trace else None)
        world = compile_world(stage.world)
        analysis = analyze(stage.world, world=world)
    except SimulationError as error:
        stage.failure = (error.code, str(error))
        return stage
    except TimingError as error:
        stage.failure = (error.code, str(error))
        return stage
    stage.closure = closure = close_timing(analysis, clock_constraints(config))
    report = closure.report
    emit("timing", "timing_analyzed", **_timing_summary(report))
    if not closure.passed:
        code, message = closure.failures[0]
        emit("timing", "timing_failed", failures=report["closure"]["failures"])
        stage.failure = (code, message)
        return stage
    emit("timing", "timing_closed", **_timing_summary(report))
    if config.simulation_validation == "off":
        stage.simulation = {"model": SIMULATION_MODEL, "mode": stage.world.mode, "validated": False, "skipped": True}
        return stage
    stage.simulation = validate_by_simulation(mapped, stage.world, world, closure, config)
    sim = stage.simulation
    emit("simulation", "simulation_validated" if sim["validated"] else "simulation_failed", **_sim_summary(sim))
    if not sim["validated"]:
        failure = sim["failures"][0]
        stage.failure = (failure["code"], failure["message"])
        return stage
    # The final block never depends on the trace level, so this is recorded
    # at every level (it is small and deterministic).
    try:
        stage.playback = record_playback(stage.world, closure, world=world)
    except SimulationError:  # a visualization aid only: never fails the design
        stage.playback = None
    return stage


def _timing_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"sequential": report["sequential"], "passed": report["closure"]["passed"]}
    if report["sequential"]:
        clock = report["clock"]
        summary.update(
            period=clock["period"], mode=clock["mode"], skew=clock["skew"], arrival_max=clock["arrival_max"],
            worst_setup_slack=report["setup"]["worst_slack"], worst_hold_slack=report["hold"]["worst_slack"],
        )  # fmt: skip
    else:
        summary["settle"] = report["combinational"]["settle"]
    return summary


def _sim_summary(sim: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("mode", "validated", "vectors", "transactions_run", "cycles", "sim_events", "simulated_ticks_gt")
    return {k: sim[k] for k in keys if k in sim} | {"failures": sim.get("failures", [])[:5]}


def _random_value(rng: random.Random, width: int, signed: bool) -> int:
    raw = rng.getrandbits(width)
    if signed and raw >> (width - 1):
        raw -= 1 << width
    return raw


def validate_by_simulation(
    mapped: PrimitivePhysicalNetlist,
    design: MinecraftPhysicalDesign,
    world: CompiledWorld,
    closure: TimingClosure,
    config: PrimitivePnRConfig,
) -> dict[str, Any]:
    """Drive the materialized design in the redstone simulator and compare it
    with the PrimitiveSimulator (outputs per vector; outputs, register state
    and ``done`` per logical cycle)."""
    reference = PrimitiveSimulator(mapped.logical)
    rng = random.Random(config.validation_seed)
    failures: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "model": SIMULATION_MODEL,
        "mode": design.mode,
        "minecraft_version": design.minecraft_version,
        "world_fingerprint": design.fingerprint(),
        "corner": "max",
    }
    data_inputs = [p for p in design.inputs if p.role == "data" and not (closure.sequential and p.name == "start")]

    def fail(code: str, message: str) -> None:
        if len(failures) < 50:
            failures.append({"code": code, "message": message})

    try:
        if not closure.sequential:
            vectors = _combinational_vectors(data_inputs, config.validation_vectors, rng)
            sim = RedstoneSimulator(design, world=world)
            settle = closure.report["combinational"]["settle"]["gt"]
            worst = 0
            for vector in vectors:
                applied = sim.time
                got = sim.evaluate(**vector)
                worst = max(worst, sim.time - applied)
                want = reference.evaluate(**vector)
                if got != want:
                    fail("simulation/logic_mismatch", f"inputs {vector}: redstone {got} != primitive {want}")
            if settle is not None and worst > settle:
                fail("simulation/timing_mismatch", f"outputs settled after {worst} gt > STA bound {settle} gt")
            if sim.warnings:
                fail("simulation/weak_signal", sim.warnings[0].message)
            record.update(vectors=len(vectors), observed_settle=ticks_record(worst), **_stats(sim.stats, sim.time))
        else:
            schedule = ClockSchedule.from_closure(closure)
            arrivals = closure.clock_arrivals()
            has_start = any(p.name == "start" for p in design.inputs)
            has_done = any(p.name == "done" for p in design.outputs)
            totals = {"sim_events": 0, "simulated_ticks_gt": 0, "peak_queue": 0, "cycles": 0}
            transactions = []
            for index in range(config.validation_transactions):
                values = {p.name: _random_value(rng, p.width, p.signed) for p in data_inputs}
                if index == 0:
                    values = {p.name: 0 for p in data_inputs}
                cycles = transaction_inputs(values, config.validation_cycles, has_start=has_start)
                run = run_clocked(design, schedule, cycles, stop_when_done=has_done, world=world)
                state = reference.reset_state()
                ref_done = None
                for obs in run.observations:
                    outputs, nxt = reference.step(state, **cycles[obs.cycle])
                    if obs.outputs != outputs:
                        fail("simulation/logic_mismatch",
                             f"transaction {values} cycle {obs.cycle}: redstone outputs {obs.outputs} != {outputs}")  # fmt: skip
                    for inst_id, bit in state.items():
                        name = register_probe(inst_id)
                        if name in obs.probes and obs.probes[name] != bit:
                            fail("simulation/logic_mismatch",
                                 f"transaction {values} cycle {obs.cycle}: register bit {inst_id} is "
                                 f"{obs.probes[name]}, primitive simulator says {bit}")  # fmt: skip
                    if ref_done is None and obs.cycle > 0 and outputs.get("done"):
                        ref_done = obs.cycle
                    state = nxt
                if has_done and run.done_cycle != ref_done:
                    fail("simulation/cycle_mismatch",
                         f"transaction {values}: done on cycle {run.done_cycle}, primitive simulator {ref_done}")  # fmt: skip
                expected = {component_name(i.id): arrivals[component_name(i.id)] for i in mapped.instances
                            if i.kind is PrimitiveKind.REGISTER_BIT and component_name(i.id) in arrivals}  # fmt: skip
                for error in run.capture_errors(expected):
                    fail("simulation/cycle_mismatch", f"unexpected register capture: {error}")
                for violation in run.violations[:5]:
                    fail("simulation/timing_violation", f"{violation.kind} at {violation.component}: {violation.detail}")
                for warning in run.warnings[:3]:
                    fail("simulation/weak_signal", warning.message)
                transactions.append(
                    {"inputs": values, "cycles": len(run.observations), "done_cycle": run.done_cycle,
                     "reference_done_cycle": ref_done, "sim_events": run.stats["events"]}
                )  # fmt: skip
                totals["sim_events"] += run.stats["events"]
                totals["simulated_ticks_gt"] += run.stats["simulated_ticks_gt"]
                totals["peak_queue"] = max(totals["peak_queue"], run.stats["peak_queue"])
                totals["cycles"] += len(run.observations)
            record.update(schedule=schedule.to_dict(), transactions=transactions, **totals)
            record["transactions_run"] = len(transactions)
    except SimulationError as error:
        fail(error.code, str(error))
    # Wall-clock time stays out of the (deterministic) record; see ``runtime_s`` on the stage.
    record["failures"] = failures
    record["validated"] = not failures
    record["world"] = world.stats()
    return record


def _stats(stats: Mapping[str, int], ticks: int) -> dict[str, Any]:
    return {
        "sim_events": stats["events"],
        "peak_queue": stats["peak_queue"],
        "simulated_ticks_gt": ticks,
    }


def _combinational_vectors(ports: list[Any], count: int, rng: random.Random) -> list[dict[str, int]]:
    total = sum(p.width for p in ports)
    if total <= 12 and 1 << total <= max(count, 1):
        vectors = []
        for code in range(1 << total):
            vector, shift = {}, 0
            for p in ports:
                raw = (code >> shift) & ((1 << p.width) - 1)
                shift += p.width
                vector[p.name] = raw - (1 << p.width) if p.signed and raw >> (p.width - 1) else raw
            vectors.append(vector)
        return vectors
    vectors = [{p.name: 0 for p in ports}, {p.name: -1 if p.signed else (1 << p.width) - 1 for p in ports}]
    while len(vectors) < count:
        vectors.append({p.name: _random_value(rng, p.width, p.signed) for p in ports})
    return vectors[: max(count, 2)]


__all__ = ["RETRYABLE", "TimingStage", "clock_constraints", "run_timing_stage", "validate_by_simulation"]
