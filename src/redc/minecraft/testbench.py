"""Dynamic clocked validation: drive a design's physical clock in the redstone
simulator at a chosen period and observe one logical cycle per period.

This is a VALIDATION layer on top of static timing analysis, never a
replacement for it.  The protocol (all game ticks):

1. tick 0: the clock is low, reset is asserted and the cycle-0 inputs are
   applied; the simulator runs until the circuit is stable (every register
   holds its reset value, all logic has settled);
2. reset is released at the next whole redstone tick ``R``; the first clock
   edge is at ``E_1 = R + recovery_gap`` (the closure's reset recovery gap);
3. edge ``k`` is at ``E_k = E_1 + (k - 1) * P``; the clock is high for the
   closure's high phase;
4. the inputs of cycle ``k >= 1`` change at ``E_k + input_offset``;
5. cycle ``k`` is OBSERVED at ``E_{k+1}``, before that tick's events: the
   outputs ``O_k`` and every probe (e.g. the register state ``S_k``).

Observation ``k`` therefore corresponds exactly to ``Graph.step`` number
``k``: state before the edge, outputs of that cycle.  The run records every
register capture, so a caller can check that each register captured exactly
once per edge, at the expected time.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .abstract import Capture, TimingViolation
from .behaviors import Diagnostic
from .closure import TimingClosure
from .connectivity import CompiledWorld, SimulationError
from .design import MinecraftPhysicalDesign
from .simulator import RedstoneSimulator
from .units import round_up_to_rt, rt_to_gt


@dataclass(frozen=True)
class ClockSchedule:
    period_gt: int
    high_gt: int
    input_offset_gt: int
    recovery_gap_gt: int

    def __post_init__(self) -> None:
        if self.period_gt < 2 or not 0 < self.high_gt < self.period_gt:
            raise SimulationError("simulation/bad_stimulus", "a clock needs a period >= 2 gt and 0 < high < period")
        if self.input_offset_gt < 0 or self.input_offset_gt >= self.period_gt:
            raise SimulationError("simulation/bad_stimulus", "the input offset must be inside one period")

    @classmethod
    def from_closure(cls, closure: TimingClosure, *, period_gt: int | None = None) -> ClockSchedule:
        """The closure's schedule (optionally at another period, e.g. a
        deliberately too-fast one: the phases scale, the offsets stay)."""
        assert closure.period_gt is not None and closure.high_gt is not None
        period = closure.period_gt if period_gt is None else period_gt
        high = closure.high_gt if period_gt is None else max(rt_to_gt(1), rt_to_gt(period // rt_to_gt(1) // 2))
        offset = min(closure.input_offset_gt, max(0, period - rt_to_gt(1)))
        return cls(period, high, offset, closure.recovery_gap_gt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_gt": self.period_gt,
            "high_gt": self.high_gt,
            "input_offset_gt": self.input_offset_gt,
            "recovery_gap_gt": self.recovery_gap_gt,
        }


@dataclass(frozen=True)
class CycleObservation:
    cycle: int
    sample_gt: int
    outputs: dict[str, int]
    probes: dict[str, int]


@dataclass
class ClockedRun:
    schedule: ClockSchedule
    reset_release_gt: int
    edges_gt: list[int]
    observations: list[CycleObservation]
    captures: list[Capture]
    violations: list[TimingViolation]
    warnings: list[Diagnostic]
    stats: dict[str, Any]
    report: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)
    done_cycle: int | None = None
    runtime_s: float = 0.0

    def captures_by_register(self) -> dict[str, list[Capture]]:
        result: dict[str, list[Capture]] = {}
        for capture in self.captures:
            result.setdefault(capture.component, []).append(capture)
        return result

    def capture_errors(self, clock_arrival_gt: Mapping[str, int]) -> list[str]:
        """Every register must capture exactly once per clock edge, exactly at
        ``edge + its physical clock arrival`` -- no missing, extra or early
        captures (an extra one would be an unintended intermediate cycle)."""
        errors = []
        by_reg = self.captures_by_register()
        for name, arrival in sorted(clock_arrival_gt.items()):
            want = [edge + arrival for edge in self.edges_gt]
            got = [c.edge_gt for c in by_reg.get(name, [])]
            if got != want:
                errors.append(f"{name}: captured at {got[:6]}..., expected {want[:6]}...")
        extra = sorted(set(by_reg) - set(clock_arrival_gt))
        if extra:
            errors.append(f"unexpected capturing components {extra[:6]}")
        return errors


def transaction_inputs(inputs: Mapping[str, int], cycles: int, *, has_start: bool) -> list[dict[str, int]]:
    """Per-cycle stimulus of one ``Graph.run`` transaction: ``start`` pulses on
    cycle 0, every data input is held."""
    stimulus = []
    for cycle in range(cycles):
        values = dict(inputs)
        if has_start:
            values["start"] = 1 if cycle == 0 else 0
        stimulus.append(values)
    return stimulus


def run_clocked(
    design: MinecraftPhysicalDesign,
    schedule: ClockSchedule,
    cycles: Sequence[Mapping[str, int]],
    *,
    stop_when_done: bool = False,
    world: CompiledWorld | None = None,
    trace: str = "off",
    max_reset_ticks: int = 200_000,
) -> ClockedRun:
    """Run ``len(cycles)`` logical cycles (see the module docstring)."""
    clock = design.role_port("clock")
    if clock is None:
        raise SimulationError("simulation/bad_binding", "the design has no clock port")
    reset = design.role_port("reset")
    started = time.perf_counter()
    sim = RedstoneSimulator(design, world=world, trace=trace)
    if reset is not None:
        sim.set_input(reset.name, 1)
    for name, value in (cycles[0] if cycles else {}).items():
        sim.set_input(name, value)
    sim.run_until_stable(max_ticks=max_reset_ticks)
    release = round_up_to_rt(sim.time) + rt_to_gt(1)
    if reset is not None:
        sim.schedule_input(reset.name, 0, at_tick=release)
    first = release + schedule.recovery_gap_gt
    edges: list[int] = []
    observations: list[CycleObservation] = []
    done_cycle = None
    for k in range(len(cycles)):
        edge = first + k * schedule.period_gt
        sim.run_until(edge)
        outputs = sim.read_outputs()
        observations.append(CycleObservation(k, edge, outputs, sim.probes()))
        if stop_when_done and k > 0 and outputs.get("done"):
            done_cycle = k
            break
        if k + 1 == len(cycles):
            break
        sim.schedule_input(clock.name, 1, at_tick=edge)
        sim.schedule_input(clock.name, 0, at_tick=edge + schedule.high_gt)
        edges.append(edge)
        for name, value in cycles[k + 1].items():
            if cycles[k].get(name) != value:
                sim.schedule_input(name, value, at_tick=edge + schedule.input_offset_gt)
    #: Wall-clock time is kept OUT of the stats (artifacts are deterministic).
    elapsed = time.perf_counter() - started
    simulated = max(sim.time, 1)
    stats = {
        **sim.stats,
        "simulated_ticks_gt": sim.time,
        "cycles": len(observations),
        "events_per_simulated_tick": round(sim.stats["events"] / simulated, 3),
    }
    return ClockedRun(
        schedule, release, edges, observations, list(sim.captures), list(sim.violations), list(sim.warnings),
        stats, sim.report(), sim.events, done_cycle, elapsed,
    )  # fmt: skip


__all__ = [
    "ClockSchedule",
    "ClockedRun",
    "CycleObservation",
    "run_clocked",
    "transaction_inputs",
]
