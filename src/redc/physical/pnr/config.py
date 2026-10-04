"""Place-and-route configuration and trace verbosity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...parser import CompileError
from ...tracing import TraceLevel
from ..grid import MAX_HEIGHT

__all__ = ["AttemptGeometry", "PnRConfig", "TraceLevel"]


@dataclass(frozen=True)
class AttemptGeometry:
    """The spacing used by one P&R attempt (grows on every retry)."""

    attempt: int
    component_spacing: int
    layer_gap: int
    routing_margin: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "component_spacing": self.component_spacing,
            "layer_gap": self.layer_gap,
            "routing_margin": self.routing_margin,
        }


@dataclass(frozen=True)
class PnRConfig:
    """Every knob of the v1 placer and router.

    Distances are in abstract grid cells.  Defaults favour legality and generous
    routing room over compactness.  The algorithms are fully deterministic (no
    randomness is consumed), so equal inputs and config give equal results.
    """

    #: Grid height in cells (``0 <= y < grid_height``).
    grid_height: int = MAX_HEIGHT
    #: Lowest origin y for components.  Leaves room to route underneath; cells
    #: with BOTTOM pins are always lifted so their escapes stay at ``y >= 0``.
    base_y: int = 1
    #: Free cells between neighbouring components inside one placement layer
    #: (along z).
    component_spacing: int = 3
    #: Free cells between consecutive placement layers (along x).
    layer_gap: int = 4
    #: How far beyond the placed design (in x and z) routes may search.
    routing_margin: int = 4
    #: Negotiated-congestion iterations allowed after the initial routing pass.
    max_routing_iterations: int = 40
    #: Node-expansion limit for a single A* branch search.
    max_astar_expansions: int = 200_000
    #: Present-congestion factor of the first routing pass ...
    present_factor_initial: float = 0.5
    #: ... multiplied by this after every negotiation iteration.
    present_factor_growth: float = 1.6
    #: History added per unit of overuse at the end of each iteration.
    history_increment: float = 1.0
    #: Outer place-and-route attempts before giving up.
    max_pnr_attempts: int = 4
    #: Added to spacing, layer gap and routing margin on every retry.
    retry_spacing_growth: int = 2
    #: Replay-trace verbosity.
    trace_level: TraceLevel = TraceLevel.BASIC
    #: Emit a state ``keyframe`` event every N routing iterations (0 = never).
    keyframe_interval: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.grid_height <= MAX_HEIGHT:
            raise CompileError(f"grid_height must be between 1 and {MAX_HEIGHT}")
        if not 0 <= self.base_y < self.grid_height:
            raise CompileError("base_y must lie inside the grid")
        for name in ("component_spacing", "layer_gap", "routing_margin",
                     "max_routing_iterations", "retry_spacing_growth",
                     "keyframe_interval"):  # fmt: skip
            if getattr(self, name) < 0:
                raise CompileError(f"{name} must be non-negative")
        if self.max_pnr_attempts < 1 or self.max_astar_expansions < 1:
            raise CompileError("max_pnr_attempts and max_astar_expansions must be positive")
        if self.present_factor_initial < 0 or self.present_factor_growth < 1:
            raise CompileError("present factor must start >= 0 and grow by >= 1x")
        if self.history_increment < 0:
            raise CompileError("history_increment must be non-negative")
        object.__setattr__(self, "trace_level", TraceLevel.parse(self.trace_level))

    def attempt_geometry(self, attempt: int) -> AttemptGeometry:
        """Deterministic spacing for ``attempt``: each retry spreads farther."""
        growth = attempt * self.retry_spacing_growth
        return AttemptGeometry(
            attempt=attempt,
            component_spacing=self.component_spacing + growth,
            layer_gap=self.layer_gap + growth,
            routing_margin=self.routing_margin + growth,
        )

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "grid_height": self.grid_height,
            "base_y": self.base_y,
            "component_spacing": self.component_spacing,
            "layer_gap": self.layer_gap,
            "routing_margin": self.routing_margin,
            "max_routing_iterations": self.max_routing_iterations,
            "max_astar_expansions": self.max_astar_expansions,
            "present_factor_initial": self.present_factor_initial,
            "present_factor_growth": self.present_factor_growth,
            "history_increment": self.history_increment,
            "max_pnr_attempts": self.max_pnr_attempts,
            "retry_spacing_growth": self.retry_spacing_growth,
            "trace_level": self.trace_level.label,
            "keyframe_interval": self.keyframe_interval,
        }
