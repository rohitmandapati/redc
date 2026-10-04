"""The ONE timing model shared by the redstone simulator and static timing analysis.

``redc-redstone-timing-v1``.  Every delay below is defined exactly once; the
simulator's block behaviours schedule events with these functions and STA
builds timing-graph edges with the very same functions, so the two can never
disagree about, say, what a delay-3 repeater costs.

All durations are GAME TICKS (see :mod:`redc.minecraft.units`).  Every
modelled delay is a whole number of redstone ticks (an even number of game
ticks), so -- with stimuli on even ticks -- every event happens on an even
tick and the narrowest pulse a circuit can produce is :data:`MIN_PULSE_GT`.

Component timing (cells that are not materialized as blocks) is declared with
explicit arcs, never a single generic latency:

* :class:`CombinationalArc` -- ``from_pin -> to_pin`` with a min and max delay;
* :class:`SequentialTiming` -- clock / data / q / reset pins, clk->Q, setup,
  hold, reset->Q and reset recovery, all relative to the component's own pins.

Edge arithmetic (what "setup" and "hold" mean, used identically by the
simulator's abstract registers and STA): for a capture edge arriving at the
clock pin at time ``e``, the data pin must not change in the open interval
``(e - setup, e + hold)``: the last change before the edge may happen at
``e - setup`` at the latest, the first change after it at ``e + hold`` at the
earliest.  With ``setup >= 1`` and ``hold >= 1`` the edge tick itself is
always inside the window, so no capture ever depends on the order of two
events in the same tick.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .units import rt_to_gt, ticks_record

TIMING_MODEL = "redc-redstone-timing-v1"

#: Redstone dust changes in the same game tick as its source (Java: block
#: updates within one tick).
DUST_DELAY_GT = 0
#: A redstone torch toggles one redstone tick after its input changes.
TORCH_DELAY_GT = rt_to_gt(1)
#: A redstone lamp lights at once but turns off two redstone ticks later.
LAMP_ON_DELAY_GT = 0
LAMP_OFF_DELAY_GT = rt_to_gt(2)
#: The narrowest pulse a v1 circuit produces (every delay is whole rt).
MIN_PULSE_GT = rt_to_gt(1)
#: Configurable repeater delays, in redstone ticks.
REPEATER_DELAYS_RT: tuple[int, ...] = (1, 2, 3, 4)
MAX_REPEATER_DELAY_RT = REPEATER_DELAYS_RT[-1]


def repeater_delay_gt(setting_rt: int) -> int:
    """Propagation delay of a repeater set to ``setting_rt`` (1..4) redstone ticks."""
    if setting_rt not in REPEATER_DELAYS_RT:
        raise ValueError(f"repeater delay must be one of {REPEATER_DELAYS_RT} redstone ticks, got {setting_rt}")
    return rt_to_gt(setting_rt)


def repeater_settle_bound_gt(setting_rt: int, *, single_transition: bool) -> int:
    """Latest output transition of a repeater after its input's LAST transition.

    A repeater extends any input pulse shorter than its delay to the full
    delay (Java: a tick that turns it on while the input is already off
    schedules a second tick).  So for an input that may glitch, the output can
    still change up to ``2 * delay - MIN_PULSE`` after the input's last change;
    an input that changes at most once per cycle settles after ``delay``."""
    delay = repeater_delay_gt(setting_rt)
    return delay if single_transition else max(delay, 2 * delay - MIN_PULSE_GT)


@dataclass(frozen=True, slots=True)
class CombinationalArc:
    """A component's delay from one input pin to one output pin (game ticks)."""

    from_pin: str
    to_pin: str
    min_gt: int
    max_gt: int

    def __post_init__(self) -> None:
        if not 0 <= self.min_gt <= self.max_gt:
            raise ValueError(f"arc {self.from_pin}->{self.to_pin}: need 0 <= min <= max, got {self.min_gt}, {self.max_gt}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_pin": self.from_pin,
            "to_pin": self.to_pin,
            "min_delay": ticks_record(self.min_gt),
            "max_delay": ticks_record(self.max_gt),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CombinationalArc:
        return cls(str(data["from_pin"]), str(data["to_pin"]), int(data["min_delay"]["gt"]), int(data["max_delay"]["gt"]))


@dataclass(frozen=True, slots=True)
class SequentialTiming:
    """An edge-triggered (rising edge) state element with an asynchronous,
    active-high reset.  Times are relative to the component's own pins."""

    clock_pin: str
    data_pin: str
    q_pin: str
    reset_pin: str | None
    clk_to_q_min_gt: int
    clk_to_q_max_gt: int
    setup_gt: int
    hold_gt: int
    reset_to_q_gt: int = 0
    recovery_gt: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.clk_to_q_min_gt <= self.clk_to_q_max_gt:
            raise ValueError("clk->Q needs 0 <= min <= max")
        if self.setup_gt < 1 or self.hold_gt < 1:
            raise ValueError("setup and hold must each be at least one game tick (no same-tick races)")
        if self.reset_to_q_gt < 0 or self.recovery_gt < 0:
            raise ValueError("reset->Q and recovery must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "clock_pin": self.clock_pin,
            "data_pin": self.data_pin,
            "q_pin": self.q_pin,
            "reset_pin": self.reset_pin,
            "edge": "rising",
            "reset": "asynchronous, active high",
            "clk_to_q_min": ticks_record(self.clk_to_q_min_gt),
            "clk_to_q_max": ticks_record(self.clk_to_q_max_gt),
            "setup": ticks_record(self.setup_gt),
            "hold": ticks_record(self.hold_gt),
            "reset_to_q": ticks_record(self.reset_to_q_gt),
            "recovery": ticks_record(self.recovery_gt),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SequentialTiming:
        return cls(
            str(data["clock_pin"]),
            str(data["data_pin"]),
            str(data["q_pin"]),
            None if data.get("reset_pin") is None else str(data["reset_pin"]),
            int(data["clk_to_q_min"]["gt"]),
            int(data["clk_to_q_max"]["gt"]),
            int(data["setup"]["gt"]),
            int(data["hold"]["gt"]),
            int(data["reset_to_q"]["gt"]),
            int(data["recovery"]["gt"]),
        )


@dataclass(frozen=True)
class ComponentTiming:
    """All timing of one component.  ``source`` says where the numbers come
    from: ``declared`` (hand-entered, e.g. a placeholder estimate) or
    ``characterized`` (measured by simulating the component's blocks)."""

    arcs: tuple[CombinationalArc, ...] = ()
    sequential: SequentialTiming | None = None
    source: str = "declared"

    def __post_init__(self) -> None:
        if self.source not in ("declared", "characterized"):
            raise ValueError(f"timing source must be 'declared' or 'characterized', got {self.source!r}")
        pairs = [(a.from_pin, a.to_pin) for a in self.arcs]
        if len(set(pairs)) != len(pairs):
            raise ValueError(f"duplicate timing arcs {pairs}")

    def arcs_from(self, pin: str) -> tuple[CombinationalArc, ...]:
        return tuple(a for a in self.arcs if a.from_pin == pin)

    def arcs_to(self, pin: str) -> tuple[CombinationalArc, ...]:
        return tuple(a for a in self.arcs if a.to_pin == pin)

    @property
    def max_delay_gt(self) -> int | None:
        """Largest declared delay (``None`` if there is nothing to delay)."""
        values = [a.max_gt for a in self.arcs]
        if self.sequential is not None:
            values.append(self.sequential.clk_to_q_max_gt)
        return max(values) if values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": TIMING_MODEL,
            "source": self.source,
            "arcs": [a.to_dict() for a in self.arcs],
            "sequential": None if self.sequential is None else self.sequential.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ComponentTiming:
        seq = data.get("sequential")
        return cls(
            tuple(CombinationalArc.from_dict(a) for a in data.get("arcs", ())),
            None if seq is None else SequentialTiming.from_dict(seq),
            str(data.get("source", "declared")),
        )


def uniform_arcs(inputs: Iterable[str], outputs: Iterable[str], delay_gt: int) -> tuple[CombinationalArc, ...]:
    """Every input -> every output with the same fixed delay."""
    outs = tuple(outputs)
    return tuple(CombinationalArc(i, o, delay_gt, delay_gt) for i in inputs for o in outs)


@dataclass(frozen=True)
class TimingModelInfo:
    """The model's constants as one JSON record (embedded in reports)."""

    name: str = TIMING_MODEL
    constants: dict[str, int] = field(
        default_factory=lambda: {
            "dust_delay_gt": DUST_DELAY_GT,
            "torch_delay_gt": TORCH_DELAY_GT,
            "lamp_on_delay_gt": LAMP_ON_DELAY_GT,
            "lamp_off_delay_gt": LAMP_OFF_DELAY_GT,
            "min_pulse_gt": MIN_PULSE_GT,
            "repeater_delay_gt_per_rt": rt_to_gt(1),
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "units": "game ticks (1 redstone tick = 2 game ticks)", **self.constants}


__all__ = [
    "DUST_DELAY_GT",
    "LAMP_OFF_DELAY_GT",
    "LAMP_ON_DELAY_GT",
    "MAX_REPEATER_DELAY_RT",
    "MIN_PULSE_GT",
    "REPEATER_DELAYS_RT",
    "TIMING_MODEL",
    "TORCH_DELAY_GT",
    "CombinationalArc",
    "ComponentTiming",
    "SequentialTiming",
    "TimingModelInfo",
    "repeater_delay_gt",
    "repeater_settle_bound_gt",
    "uniform_arcs",
]
