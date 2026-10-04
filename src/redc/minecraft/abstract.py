"""ABSTRACT component simulation: declared Boolean / timing behaviour for cells
that have no block-level implementation yet.

This is NOT Minecraft simulation.  Routes, repeaters and every real block
around an abstract component are simulated block-accurately, but the
component itself behaves exactly as its declaration says.  A design that
contains one is always reported as ``abstract-components`` mode.

Semantics (shared with static timing analysis through
:mod:`redc.minecraft.timing`):

* **combinational**: transport delay per arc.  The output at time ``T`` is the
  truth-table function of each input's value at ``T - delay(input -> out)``
  (the simulation corner picks the arc's ``max`` or ``min``).  Every input
  change at ``t`` schedules a re-evaluation at ``t + delay`` for each output it
  reaches.  Glitches propagate (conservative: a register downstream still has
  to see stable data in its setup/hold window).  Cold start: inputs and
  outputs are 0 before time 0 and every output is first evaluated at its
  largest arc delay.
* **dff**: rising-edge register with asynchronous active-high reset.  A rising
  clock-pin edge at ``e`` (reset low) samples the data pin and drives q at
  ``e + clk_to_q``.  The data pin must not change in ``(e - setup, e + hold)``
  and a reset release must precede the edge by ``recovery``; violations are
  recorded (dynamic setup / hold / recovery checks) -- the sampled value is
  still the data pin's value at ``e``.  While reset is high, clock edges are
  ignored and q is forced to ``init`` ``reset_to_q`` after reset rises.  Before
  the first reset, q is 0 (cold).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .design import AbstractComponent

if TYPE_CHECKING:
    from .simulator import RedstoneSimulator

CORNERS = ("max", "min")


@dataclass(frozen=True)
class TimingViolation:
    """A setup / hold / recovery violation seen during simulation."""

    kind: str
    component: str
    time_gt: int
    edge_gt: int | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "component": self.component,
            "time_gt": self.time_gt,
            "edge_gt": self.edge_gt,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Capture:
    """One register capture: the clock-pin edge time and the value sampled."""

    component: str
    edge_gt: int
    value: int
    q_at_gt: int


class AbstractAdapter:
    """Base of the per-component runtime state."""

    def __init__(self, sim: RedstoneSimulator, device: int, component: AbstractComponent) -> None:
        self.sim = sim
        self.device = device
        self.component = component
        world = sim.world
        dev = world.devices[device]
        #: input pin name -> reader id, output pin name -> source id
        self.reader_of = {world.readers[r].pin: r for r in dev.readers}
        self.source_of = {world.sources[s].pin: s for s in dev.sources}
        self.threshold = {p.name: p.strength for p in component.pins if p.direction == "in"}
        self.value: dict[str, int] = {pin: 0 for pin in component.input_names}

    def input_bit(self, pin: str, strength: int) -> int:
        threshold = self.threshold[pin]
        if 0 < strength < threshold:
            self.sim.warn(
                "simulation/weak_input",
                f"{self.component.name} pin {pin!r} sees strength {strength} < required {threshold}",
                self.component.pin(pin).coord,
            )
        return 1 if strength >= threshold else 0

    def power_on(self) -> None:
        """Called once at time 0."""

    def on_input(self, pin: str, strength: int) -> None:
        raise NotImplementedError

    def on_event(self, payload: Any) -> None:
        raise NotImplementedError


class CombinationalAdapter(AbstractAdapter):
    def __init__(self, sim: RedstoneSimulator, device: int, component: AbstractComponent) -> None:
        super().__init__(sim, device, component)
        corner = sim.corner
        self.inputs = component.input_names
        self.tables = {o: component.table(o) for o in component.output_names}
        # (input, output) -> delay in the chosen corner
        self.delay = {
            (a.from_pin, a.to_pin): (a.max_gt if corner == "max" else a.min_gt) for a in component.timing.arcs
        }
        self.horizon = max(self.delay.values(), default=0)
        self.history: dict[str, tuple[list[int], list[int]]] = {pin: ([], []) for pin in self.inputs}
        self.output: dict[str, int] = {o: 0 for o in component.output_names}

    def _value_at(self, pin: str, time: int) -> int:
        times, values = self.history[pin]
        k = bisect_right(times, time)
        return values[k - 1] if k else 0

    def _evaluate(self, output: str) -> int:
        now = self.sim.time
        index = 0
        for i, pin in enumerate(self.inputs):
            if self._value_at(pin, now - self.delay[(pin, output)]):
                index |= 1 << i
        return 1 if self.tables[output][index] == "1" else 0

    def power_on(self) -> None:
        for output in self.component.output_names:
            delay = max((self.delay[(pin, output)] for pin in self.inputs), default=0)
            if not self.inputs:
                # A constant: its value holds from the start.
                self._apply(output, 1 if self.tables[output] == "1" else 0)
            else:
                self.sim.schedule_abstract(self.device, delay, output)

    def on_input(self, pin: str, strength: int) -> None:
        bit = self.input_bit(pin, strength)
        if bit == self.value[pin]:
            return
        self.value[pin] = bit
        now = self.sim.time
        times, values = self.history[pin]
        times.append(now)
        values.append(bit)
        if len(times) > 64:  # keep the window the delays can still look back into
            keep = max(0, bisect_right(times, now - self.horizon) - 1)
            del times[:keep], values[:keep]
        for output in self.component.output_names:
            self.sim.schedule_abstract(self.device, self.delay[(pin, output)], output)

    def on_event(self, payload: Any) -> None:
        output = str(payload)
        self._apply(output, self._evaluate(output))

    def _apply(self, output: str, value: int) -> None:
        if value == self.output[output]:
            return
        self.output[output] = value
        self.sim.set_source(self.source_of[output], bool(value))
        self.sim.record("component_output_changed", component=self.component.name, pin=output, value=value)


@dataclass
class _DffState:
    q: int = 0
    d: int = 0
    clk: int = 0
    rst: int = 0
    last_d_change: int | None = None
    last_edge: int | None = None
    reset_release: int | None = None
    captures: list[Capture] = field(default_factory=list)


class DffAdapter(AbstractAdapter):
    def __init__(self, sim: RedstoneSimulator, device: int, component: AbstractComponent) -> None:
        super().__init__(sim, device, component)
        seq = component.timing.sequential
        assert seq is not None
        self.seq = seq
        self.clk_to_q = seq.clk_to_q_max_gt if sim.corner == "max" else seq.clk_to_q_min_gt
        self.state = _DffState()

    def on_input(self, pin: str, strength: int) -> None:
        bit = self.input_bit(pin, strength)
        st, seq, now, name = self.state, self.seq, self.sim.time, self.component.name
        if pin == seq.data_pin:
            if bit == st.d:
                return
            st.d = bit
            st.last_d_change = now
            if st.last_edge is not None and now < st.last_edge + seq.hold_gt:
                self.sim.violation(
                    TimingViolation(
                        "hold", name, now, st.last_edge,
                        f"data changed {now - st.last_edge} gt after the edge (hold {seq.hold_gt} gt)",
                    )
                )  # fmt: skip
        elif pin == seq.clock_pin:
            if bit == st.clk:
                return
            st.clk = bit
            if not bit:
                return
            self.sim.record("register_clock_edge", component=name)
            if st.rst:
                self.sim.record("register_edge_ignored", component=name, reason="reset asserted")
                return
            if st.last_d_change is not None and st.last_d_change > now - seq.setup_gt:
                self.sim.violation(
                    TimingViolation(
                        "setup", name, now, now,
                        f"data changed {now - st.last_d_change} gt before the edge (setup {seq.setup_gt} gt)",
                    )
                )  # fmt: skip
            if st.reset_release is not None and now < st.reset_release + seq.recovery_gt:
                self.sim.violation(
                    TimingViolation(
                        "recovery", name, now, now,
                        f"reset released {now - st.reset_release} gt before the edge "
                        f"(recovery {seq.recovery_gt} gt)",
                    )
                )  # fmt: skip
            st.last_edge = now
            capture = Capture(name, now, st.d, now + self.clk_to_q)
            st.captures.append(capture)
            self.sim.capture(capture)
            self.sim.schedule_abstract(self.device, self.clk_to_q, ("capture", st.d))
        else:  # reset
            if bit == st.rst:
                return
            st.rst = bit
            if bit:
                self.sim.schedule_abstract(self.device, seq.reset_to_q_gt, ("reset", self.component.init))
            else:
                st.reset_release = now

    def on_event(self, payload: Any) -> None:
        cause, value = payload
        if cause == "capture" and self.state.rst:
            return  # an asynchronous reset overrides a capture still in flight
        if value == self.state.q:
            return
        self.state.q = value
        self.sim.set_source(self.source_of[self.seq.q_pin], bool(value))
        self.sim.record("component_output_changed", component=self.component.name, pin=self.seq.q_pin, value=value)


ADAPTERS: dict[str, type[AbstractAdapter]] = {
    "combinational": CombinationalAdapter,
    "dff": DffAdapter,
}

__all__ = [
    "ADAPTERS",
    "CORNERS",
    "AbstractAdapter",
    "Capture",
    "CombinationalAdapter",
    "DffAdapter",
    "TimingViolation",
]
