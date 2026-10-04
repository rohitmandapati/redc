"""Primitive place-and-route configuration.

Every distance is in BLOCKS (one coordinate = one Minecraft block).  Defaults
favour legality and routing room over compactness; every algorithm is
deterministic, so equal inputs and config give byte-identical results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...parser import CompileError
from ...tracing import TraceLevel

#: Minecraft's build limit spans 384 blocks; the design must fit inside it.
WORLD_HEIGHT = 384


@dataclass(frozen=True)
class AttemptGeometry:
    """The spacing one P&R attempt uses (it grows on every retry)."""

    attempt: int
    component_spacing: int
    channel_width: int
    routing_margin: int
    max_y: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "component_spacing": self.component_spacing,
            "channel_width": self.channel_width,
            "routing_margin": self.routing_margin,
            "max_y": self.max_y,
        }


@dataclass(frozen=True)
class PrimitivePnRConfig:
    """Every knob of the primitive placer, router and legalizer."""

    #: Highest block y a route may use (gates sit on y=0..1; crossings stack
    #: routes two blocks apart above them).
    max_y: int = 9
    #: Free blocks between neighbouring cells inside one placement column (z),
    #: beyond each cell's own keep-out.
    component_spacing: int = 1
    #: Free blocks between placement columns (x): the routing channels.
    channel_width: int = 6
    #: How far beyond the placed design (x and z) routes may search.
    routing_margin: int = 6
    #: Negotiated-congestion iterations after the initial routing pass.
    max_routing_iterations: int = 30
    #: Node-expansion budget for one branch search: at least this many, or
    #: ``expansions_per_block`` per block of distance to the goal if larger.
    max_astar_expansions: int = 200_000
    expansions_per_block: int = 400
    #: Weighted A* factor (``f = g + w * h``; 1.0 = optimal A*, larger = faster,
    #: slightly longer routes -- like VPR's ``astar_fac``).
    astar_weight: float = 1.3
    #: A* weight of a "greedy" re-search after a branch exhausts its budget.
    greedy_astar_weight: float = 3.0
    #: Total expansions one branch may spend over all its retries, as a
    #: multiple of its single-search budget (a hopeless branch fails fast).
    branch_effort: int = 4
    #: Early abort of a nonviable attempt (0 disables each rule).  They only
    #: decide when to give up and retry wider; legality is never relaxed.
    #: Diverged: conflicts > factor * fewest-so-far + margin.
    abort_divergence_factor: float = 4.0
    abort_divergence_margin: int = 50
    #: Stagnated: no new fewest-conflicts for this many iterations.
    abort_stagnation_iterations: int = 8
    #: Effort exhausted: total A* expansions of the attempt above
    #: max(attempt_effort_min, attempt_effort_per_net * nets).
    attempt_effort_per_net: int = 60_000
    attempt_effort_min: int = 5_000_000
    #: Present-congestion factor of the first pass, multiplied each iteration.
    present_factor_initial: float = 0.5
    present_factor_growth: float = 1.6
    #: History added to every block involved in a conflict, per iteration.
    history_increment: float = 1.0
    #: Extra cost of one climb/descent step (keeps routes flat unless needed).
    vertical_cost: float = 1.0
    #: Re-searches allowed when a found path violates intra-net rules; the
    #: re-searches check moves against ``search_lookback`` blocks of their
    #: own path.
    max_path_retries: int = 8
    search_lookback: int = 8
    #: Re-routes of one net with a failing sink moved first (a net can wall in
    #: its own sink).
    max_sink_order_restarts: int = 2
    #: Rounds of reroute-and-renegotiate for nets electrical legalization rejects.
    max_legalization_rounds: int = 4
    #: Outer place-and-route attempts before giving up.
    max_pnr_attempts: int = 3
    #: Growth per retry: spacing, channel width, margin and route height.
    retry_spacing_growth: int = 1
    retry_channel_growth: int = 3
    retry_margin_growth: int = 3
    retry_height_growth: int = 2
    #: Replay-trace verbosity.
    trace_level: TraceLevel = TraceLevel.BASIC
    #: Emit a full ``keyframe`` every N routing iterations (0 = never).
    keyframe_interval: int = 0
    #: SEARCH-level cap on ``route_transition_blocked`` events per branch.
    max_blocked_events_per_branch: int = 64

    def __post_init__(self) -> None:
        if not 3 <= self.max_y < WORLD_HEIGHT:
            raise CompileError(f"max_y must be between 3 and {WORLD_HEIGHT - 1}")
        for name in (
            "component_spacing", "channel_width", "routing_margin", "max_routing_iterations",
            "max_path_retries", "search_lookback", "branch_effort", "max_sink_order_restarts",
            "max_legalization_rounds", "abort_divergence_margin", "abort_stagnation_iterations",
            "attempt_effort_per_net", "attempt_effort_min", "retry_spacing_growth",
            "retry_channel_growth", "retry_margin_growth", "retry_height_growth",
            "keyframe_interval", "max_blocked_events_per_branch",
        ):  # fmt: skip
            if getattr(self, name) < 0:
                raise CompileError(f"{name} must be non-negative")
        if self.max_pnr_attempts < 1 or self.max_astar_expansions < 1:
            raise CompileError("max_pnr_attempts and max_astar_expansions must be positive")
        if self.expansions_per_block < 0 or self.astar_weight < 1 or self.greedy_astar_weight < 1:
            raise CompileError("expansions_per_block must be >= 0 and the A* weights >= 1")
        if self.present_factor_initial < 0 or self.present_factor_growth < 1:
            raise CompileError("present factor must start >= 0 and grow by >= 1x")
        if self.abort_divergence_factor < 0:
            raise CompileError("abort_divergence_factor must be non-negative")
        if self.history_increment < 0 or self.vertical_cost < 0:
            raise CompileError("history_increment and vertical_cost must be non-negative")
        object.__setattr__(self, "trace_level", TraceLevel.parse(self.trace_level))

    def attempt_geometry(self, attempt: int) -> AttemptGeometry:
        """Deterministic spacing for ``attempt``: each retry spreads farther."""
        return AttemptGeometry(
            attempt=attempt,
            component_spacing=self.component_spacing + attempt * self.retry_spacing_growth,
            channel_width=self.channel_width + attempt * self.retry_channel_growth,
            routing_margin=self.routing_margin + attempt * self.retry_margin_growth,
            max_y=min(WORLD_HEIGHT - 1, self.max_y + attempt * self.retry_height_growth),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "units": "blocks",
            "max_y": self.max_y,
            "component_spacing": self.component_spacing,
            "channel_width": self.channel_width,
            "routing_margin": self.routing_margin,
            "max_routing_iterations": self.max_routing_iterations,
            "max_astar_expansions": self.max_astar_expansions,
            "expansions_per_block": self.expansions_per_block,
            "astar_weight": self.astar_weight,
            "greedy_astar_weight": self.greedy_astar_weight,
            "branch_effort": self.branch_effort,
            "abort_divergence_factor": self.abort_divergence_factor,
            "abort_divergence_margin": self.abort_divergence_margin,
            "abort_stagnation_iterations": self.abort_stagnation_iterations,
            "attempt_effort_per_net": self.attempt_effort_per_net,
            "attempt_effort_min": self.attempt_effort_min,
            "present_factor_initial": self.present_factor_initial,
            "present_factor_growth": self.present_factor_growth,
            "history_increment": self.history_increment,
            "vertical_cost": self.vertical_cost,
            "max_path_retries": self.max_path_retries,
            "search_lookback": self.search_lookback,
            "max_sink_order_restarts": self.max_sink_order_restarts,
            "max_legalization_rounds": self.max_legalization_rounds,
            "max_pnr_attempts": self.max_pnr_attempts,
            "retry_spacing_growth": self.retry_spacing_growth,
            "retry_channel_growth": self.retry_channel_growth,
            "retry_margin_growth": self.retry_margin_growth,
            "retry_height_growth": self.retry_height_growth,
            "trace_level": self.trace_level.label,
            "keyframe_interval": self.keyframe_interval,
            "max_blocked_events_per_branch": self.max_blocked_events_per_branch,
        }


__all__ = ["WORLD_HEIGHT", "AttemptGeometry", "PrimitivePnRConfig", "TraceLevel"]
