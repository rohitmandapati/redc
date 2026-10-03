"""Routing trees and the negotiated-congestion (PathFinder) router.

**Trees.**  A net is routed as an incrementally grown tree, never as independent
driver-to-sink paths: the tree starts at the driver's escape cell, and each sink
(nearest to the driver first; ties by instance id, then pin name) is connected
by a multi-source A* from *every* cell already in the tree, so branches share
trunks naturally.  A :class:`RoutedNet` keeps each :class:`RouteBranch`'s
ORDERED path -- the canonical replay geometry -- and derives the cell union.

**Negotiation.**  The working routing lives in the grid's occupancy/history
arrays, where nets may temporarily share cells:

1. route every net (high fanout first, then larger bounding box, then id);
2. if no cell is overused, stop -- the routing is legal;
3. otherwise add history to the overused cells, raise the present-congestion
   factor, rip up every net that touches an overused cell and reroute it (same
   deterministic order) against the new costs;
4. repeat up to ``max_routing_iterations`` times.

Temporary sharing is allowed, its price rises every iteration, and history
remembers repeat offenders, so nets are pushed apart until each cell holds at
most one.  Clock and reset are routed like any other (high-fanout) tree in v1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from ..grid import Grid
from .config import PnRConfig, TraceLevel
from .geometry import Bounds, Coord, coord_list, manhattan
from .search import astar
from .trace import TraceRecorder


@dataclass(frozen=True)
class RouteEndpoint:
    """One terminal's attachment point: the pin and its escape cell."""

    instance: int
    port: str
    coord: Coord

    def ref(self) -> dict[str, Any]:
        return {"instance": self.instance, "port": self.port}

    def to_dict(self) -> dict[str, Any]:
        return {**self.ref(), "coord": coord_list(self.coord)}


@dataclass(frozen=True)
class RouteRequest:
    """What one net must connect: its driver escape to every sink escape."""

    net: int
    driver: RouteEndpoint
    sinks: tuple[RouteEndpoint, ...]

    @property
    def fanout(self) -> int:
        return len(self.sinks)

    @property
    def span(self) -> int:
        """Half-perimeter of the endpoints' bounding box (routing-length estimate)."""
        box = Bounds.of([self.driver.coord, *(s.coord for s in self.sinks)])
        return 0 if box is None else sum(d - 1 for d in box.dims)


@dataclass(frozen=True)
class RouteBranch:
    """One sink's connection: an ordered, six-connected path from a cell already
    on the tree (``start``) to the sink's escape cell (``goal``)."""

    sink: RouteEndpoint
    path: tuple[Coord, ...]

    @property
    def start(self) -> Coord:
        return self.path[0]

    @property
    def goal(self) -> Coord:
        return self.path[-1]

    @property
    def length(self) -> int:
        """Steps along the path (0 when the sink escape was already on the tree)."""
        return len(self.path) - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "sink": self.sink.ref(),
            "start": coord_list(self.start),
            "goal": coord_list(self.goal),
            "path": [coord_list(c) for c in self.path],
        }


@dataclass(frozen=True)
class RoutedNet:
    """A net's routing tree: its root (the driver escape) and ordered branches."""

    net: int
    root: Coord
    branches: tuple[RouteBranch, ...]

    @cached_property
    def cells(self) -> tuple[Coord, ...]:
        """Every cell of the tree, each once, in construction order."""
        seen = {self.root}
        ordered = [self.root]
        for branch in self.branches:
            for cell in branch.path:
                if cell not in seen:
                    seen.add(cell)
                    ordered.append(cell)
        return tuple(ordered)

    @cached_property
    def cell_set(self) -> frozenset[Coord]:
        return frozenset(self.cells)

    @property
    def length(self) -> int:
        """Wire cells used (the union of all branches)."""
        return len(self.cells)

    @property
    def steps(self) -> int:
        """Total branch path length."""
        return sum(branch.length for branch in self.branches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "net": self.net,
            "root": coord_list(self.root),
            "branches": [branch.to_dict() for branch in self.branches],
            "cells": [coord_list(c) for c in self.cells],
            "length": self.length,
            "fanout": len(self.branches),
        }


@dataclass(frozen=True)
class RoutingFailure:
    """Why routing gave up: ``unroutable`` (a branch search failed) or
    ``congestion`` (overuse remained after the last iteration)."""

    reason: str
    iteration: int
    net: int | None = None
    sink: RouteEndpoint | None = None
    search: str | None = None
    expansions: int = 0
    overused: tuple[Coord, ...] = ()

    @property
    def message(self) -> str:
        if self.reason == "congestion":
            return (
                f"routing did not converge: {len(self.overused)} cells still overused "
                f"after iteration {self.iteration}"
            )
        where = f" to instance {self.sink.instance} pin {self.sink.port!r}" if self.sink else ""
        return (
            f"net {self.net}{where} is unroutable in iteration {self.iteration} "
            f"({self.search}, {self.expansions} expansions)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "message": self.message,
            "iteration": self.iteration,
            "net": self.net,
            "sink": self.sink.to_dict() if self.sink else None,
            "search": self.search,
            "expansions": self.expansions,
            "overused": [coord_list(c) for c in self.overused],
        }


@dataclass
class RoutingOutcome:
    success: bool
    routes: dict[int, RoutedNet]
    iterations: int
    present_factor: float
    failure: RoutingFailure | None = None
    overused: list[Coord] = field(default_factory=list)


class NegotiatedRouter:
    """Routes a set of :class:`RouteRequest` on ``grid`` (see module docstring).

    ``escape_owner`` maps pin-escape cells to the net that may use them (``-1``
    for unconnected pins); a net never routes through another net's escape.
    ``design_bounds`` (the placed bodies) and ``placement`` (instance origins)
    only feed the trace."""

    def __init__(
        self,
        grid: Grid,
        requests: Sequence[RouteRequest],
        *,
        bounds: Bounds,
        config: PnRConfig,
        trace: TraceRecorder | None = None,
        escape_owner: Mapping[Coord, int] | None = None,
        design_bounds: Bounds | None = None,
        placement: list[dict[str, Any]] | None = None,
    ) -> None:
        self.grid = grid
        self.bounds = bounds
        self.config = config
        self.trace = trace
        self.escape_owner = dict(escape_owner or {})
        self.order = sorted(requests, key=lambda r: (-r.fanout, -r.span, r.net))
        self.routes: dict[int, RoutedNet] = {}
        self.present_factor = config.present_factor_initial
        self._design_bounds = design_bounds
        self._last_bounds = design_bounds
        self._placement = placement or []

    # -- one net -------------------------------------------------------------

    def route_net(self, request: RouteRequest, iteration: int) -> RoutedNet | RoutingFailure:
        """Build ``request``'s tree in the grid's working state."""
        trace, net = self.trace, request.net
        if trace:
            trace.emit(
                "routing",
                "net_route_begin",
                level=TraceLevel.DETAILED,
                net=net,
                iteration=iteration,
                driver=request.driver.to_dict(),
                sinks=[s.to_dict() for s in request.sinks],
                fanout=request.fanout,
            )
        root = request.driver.coord
        if not self.bounds.contains(root) or not self.grid.is_routable(*root):
            return self._fail_branch(request, iteration, None, "root_blocked", 0)
        self.grid.claim(net, *root)
        tree = [root]
        on_tree = {root}
        sinks = sorted(
            request.sinks, key=lambda s: (manhattan(root, s.coord), s.instance, s.port)
        )
        forbidden = self._forbidden_for(net)
        branches: list[RouteBranch] = []
        for sink in sinks:
            if trace:
                trace.emit(
                    "routing",
                    "branch_route_begin",
                    level=TraceLevel.DETAILED,
                    net=net,
                    iteration=iteration,
                    sink=sink.ref(),
                    goal=coord_list(sink.coord),
                )
            hook = None
            if trace and trace.wants(TraceLevel.SEARCH):
                hook = self._search_hook(net, iteration, sink)
            result = astar(
                self.grid,
                tree,
                sink.coord,
                net=net,
                bounds=self.bounds,
                present_factor=self.present_factor,
                max_expansions=self.config.max_astar_expansions,
                forbidden=forbidden,
                on_expand=hook,
            )
            if result.path is None:
                self.grid.rip_up_net(net)  # never leave a half-built tree claimed
                return self._fail_branch(
                    request, iteration, sink, result.reason or "unreachable", result.expansions
                )
            for cell in result.path:
                self.grid.claim(net, *cell)
                if cell not in on_tree:
                    on_tree.add(cell)
                    tree.append(cell)
            branch = RouteBranch(sink, result.path)
            branches.append(branch)
            if trace:
                trace.emit(
                    "routing",
                    "branch_route_found",
                    level=TraceLevel.DETAILED,
                    net=net,
                    iteration=iteration,
                    sink=sink.ref(),
                    start=coord_list(branch.start),
                    goal=coord_list(branch.goal),
                    path=[coord_list(c) for c in branch.path],
                    length=branch.length,
                    expansions=result.expansions,
                )
        routed = RoutedNet(net, root, tuple(branches))
        if trace:
            trace.emit("routing", "net_route_committed", iteration=iteration, **routed.to_dict())
        return routed

    def _forbidden_for(self, net: int):
        owner = self.escape_owner
        if not owner:
            return None
        return lambda cell: owner.get(cell, net) != net

    def _search_hook(self, net: int, iteration: int, sink: RouteEndpoint):
        trace = self.trace
        assert trace is not None
        ref = sink.ref()

        def hook(cell: Coord, g: float, h: float, frontier: int) -> None:
            trace.emit(
                "routing",
                "route_search_expand",
                level=TraceLevel.SEARCH,
                net=net,
                iteration=iteration,
                sink=ref,
                coord=coord_list(cell),
                g=round(g, 6),
                h=h,
                f=round(g + h, 6),
                frontier=frontier,
            )

        return hook

    def _fail_branch(
        self,
        request: RouteRequest,
        iteration: int,
        sink: RouteEndpoint | None,
        reason: str,
        expansions: int,
    ) -> RoutingFailure:
        failure = RoutingFailure(
            reason="unroutable",
            iteration=iteration,
            net=request.net,
            sink=sink,
            search=reason,
            expansions=expansions,
        )
        if self.trace:
            self.trace.emit(
                "routing",
                "branch_route_failed",
                net=request.net,
                iteration=iteration,
                sink=sink.ref() if sink else None,
                goal=coord_list(sink.coord) if sink else coord_list(request.driver.coord),
                reason=reason,
                expansions=expansions,
                message=failure.message,
            )
        return failure

    # -- negotiation -----------------------------------------------------------

    def run(self) -> RoutingOutcome:
        trace, config = self.trace, self.config
        if trace:
            trace.emit(
                "routing",
                "routing_begin",
                nets=len(self.order),
                order=[r.net for r in self.order],
                search_bounds=self.bounds.to_dict(),
                present_factor=self.present_factor,
            )
        iteration = 0
        self._begin_iteration(iteration, [r.net for r in self.order])
        for request in self.order:
            result = self.route_net(request, iteration)
            if isinstance(result, RoutingFailure):
                return self._finish(False, iteration, result)
            self.routes[request.net] = result
        rerouted = changed = len(self.order)
        while True:
            overused = sorted(self.grid.overused())
            if overused:
                self.grid.add_history(increment=config.history_increment)
            self._end_iteration(iteration, rerouted, changed, overused)
            if not overused:
                return self._finish(True, iteration, None)
            if iteration >= config.max_routing_iterations:
                failure = RoutingFailure(
                    reason="congestion", iteration=iteration, overused=tuple(overused)
                )
                return self._finish(False, iteration, failure, overused)
            iteration += 1
            self.present_factor *= config.present_factor_growth
            hot = set(overused)
            congested = [r for r in self.order if self.routes[r.net].cell_set & hot]
            self._begin_iteration(iteration, [r.net for r in congested])
            rerouted, changed = len(congested), 0
            for request in congested:
                old = self.routes.pop(request.net)
                self.grid.rip_up_net(request.net)
                if trace:
                    trace.emit(
                        "routing",
                        "net_rip_up",
                        net=request.net,
                        iteration=iteration,
                        reason="congestion",
                        cells=[coord_list(c) for c in old.cells],
                        length=old.length,
                    )
                result = self.route_net(request, iteration)
                if isinstance(result, RoutingFailure):
                    return self._finish(False, iteration, result)
                self.routes[request.net] = result
                if result.cells != old.cells:
                    changed += 1

    def _begin_iteration(self, iteration: int, nets: list[int]) -> None:
        if self.trace:
            self.trace.emit(
                "routing",
                "routing_iteration_begin",
                iteration=iteration,
                present_factor=self.present_factor,
                nets=nets,
            )

    def _end_iteration(
        self, iteration: int, rerouted: int, changed: int, overused: list[Coord]
    ) -> None:
        trace = self.trace
        if not trace:
            return
        stats = self.grid.congestion_stats()
        trace.emit(
            "routing",
            "routing_iteration_end",
            iteration=iteration,
            present_factor=self.present_factor,
            rerouted=rerouted,
            changed=changed,
            overused=len(overused),
            max_occupancy=stats["max_occupancy"],
            routed_cells=sum(r.length for r in self.routes.values()),
            history_total=stats["history_total"],
            history_max=stats["history_max"],
        )
        trace.emit(
            "congestion",
            "congestion_snapshot",
            iteration=iteration,
            present_factor=self.present_factor,
            stats=stats,
            cells=self.congestion_cells(overused),
        )
        bounds = self._design_bounds
        for routed in self.routes.values():
            box = Bounds.of(routed.cells)
            if box is not None:
                bounds = box.union(bounds)
        if bounds is not None and bounds != self._last_bounds:
            trace.emit("routing", "design_bounds_changed", **bounds.to_dict())
            self._last_bounds = bounds
        interval = self.config.keyframe_interval
        if interval and iteration % interval == 0:
            trace.emit(
                "routing",
                "keyframe",
                iteration=iteration,
                placement=self._placement,
                routes=[self.routes[n].to_dict() for n in sorted(self.routes)],
            )

    def congestion_cells(self, overused: Sequence[Coord]) -> list[dict[str, Any]]:
        """Overused cells with occupancy, history and the nets sharing them."""
        cells = []
        for cell in overused:
            nets = sorted(n for n, r in self.routes.items() if cell in r.cell_set)
            cells.append(
                {
                    "coord": coord_list(cell),
                    "occupancy": self.grid.occupancy_at(*cell),
                    "history": self.grid.history_at(*cell),
                    "nets": nets,
                }
            )
        return cells

    def _finish(
        self,
        success: bool,
        iteration: int,
        failure: RoutingFailure | None,
        overused: list[Coord] | None = None,
    ) -> RoutingOutcome:
        if self.trace:
            self.trace.emit(
                "routing",
                "routing_complete" if success else "routing_failed",
                iteration=iteration,
                iterations=iteration + 1,
                routed_nets=len(self.routes),
                failure=failure.to_dict() if failure else None,
            )
        return RoutingOutcome(
            success=success,
            routes=dict(self.routes),
            iterations=iteration + 1,
            present_factor=self.present_factor,
            failure=failure,
            overused=list(overused or []),
        )
