"""Single-bit routing trees and the negotiated-congestion redstone router.

Every routed net is ONE Boolean signal.  A ``uint8`` value feeding another
operation is eight independent nets that may take eight different paths; any
logical grouping stays metadata and is never shared route occupancy.

**Trees.**  A net is a rooted, directed tree, never independent driver-to-sink
paths: it starts at the driver's pin block and each sink (nearest first; ties
by instance id, then pin name) is connected by a multi-source A*
(:func:`~redc.physical_primitive.pnr.search.search_branch`) from every block
already on the tree, so branches share trunks.  A :class:`RouteTree` keeps each
:class:`RouteBranch`'s ORDERED path; the parent map (signal direction from the
root), supports and staircase clearances are derived from those paths.

**Negotiation (PathFinder).**  Nets may temporarily share or crowd blocks:

1. route every net (highest fanout first, then widest span, then id);
2. if :meth:`BlockGrid.conflicts` is empty, stop -- the routing is legal;
3. else add history to every conflicting block, raise the present-congestion
   factor, rip up every net involved in a conflict and reroute it in the same
   deterministic order;
4. repeat up to ``max_routing_iterations`` times.

Pin endpoint blocks are claimed by their nets before anything is routed and
are never ripped up, so every route can see (and avoid crowding) every pin.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property, partial
from itertools import pairwise
from typing import Any

from ...tracing import TraceLevel
from ..geometry import Bounds, Coord, coord_list, manhattan
from ..grid import BlockGrid, Conflict, PinSite
from ..physical import PrimitivePhysicalNetlist
from ..redstone import clearance_of, support_of
from .config import PrimitivePnRConfig
from .search import NetState, search_branch, validate_branch
from .trace import PrimitiveTraceRecorder


@dataclass(frozen=True)
class RouteRequest:
    """What one net must connect: its driver pin block to every sink pin block."""

    net: int
    role: str
    driver: PinSite
    sinks: tuple[PinSite, ...]

    @property
    def fanout(self) -> int:
        return len(self.sinks)

    @property
    def span(self) -> int:
        box = Bounds.of([self.driver.cell, *(s.cell for s in self.sinks)])
        return 0 if box is None else sum(d - 1 for d in box.dims)

    @property
    def pins(self) -> frozenset[Coord]:
        return frozenset([self.driver.cell, *(s.cell for s in self.sinks)])


@dataclass(frozen=True)
class RouteBranch:
    """One sink's connection: an ordered path from a block already on the tree
    (``start``) to the sink's pin block (``goal``); consecutive blocks are one
    legal redstone step apart."""

    sink: PinSite
    path: tuple[Coord, ...]

    @property
    def start(self) -> Coord:
        return self.path[0]

    @property
    def goal(self) -> Coord:
        return self.path[-1]

    @property
    def length(self) -> int:
        return len(self.path) - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "sink": self.sink.ref(),
            "start": coord_list(self.start),
            "goal": coord_list(self.goal),
            "path": [coord_list(c) for c in self.path],
        }


@dataclass(frozen=True)
class RouteTree:
    """A net's geometric routing tree: root (driver pin block) + ordered branches."""

    net: int
    driver: PinSite
    root: Coord
    branches: tuple[RouteBranch, ...]

    @cached_property
    def parent(self) -> dict[Coord, Coord | None]:
        """Signal direction: every block's upstream neighbour (root -> None)."""
        parent: dict[Coord, Coord | None] = {self.root: None}
        for branch in self.branches:
            for a, b in zip(branch.path, branch.path[1:]):
                parent.setdefault(b, a)
        return parent

    @cached_property
    def children(self) -> dict[Coord, list[Coord]]:
        children: dict[Coord, list[Coord]] = {cell: [] for cell in self.parent}
        for cell, up in self.parent.items():
            if up is not None:
                children.setdefault(up, []).append(cell)
        return children

    @cached_property
    def cells(self) -> tuple[Coord, ...]:
        """Every signal block, each once, in construction (root-first) order."""
        return tuple(self.parent)

    @cached_property
    def cell_set(self) -> frozenset[Coord]:
        return frozenset(self.parent)

    @cached_property
    def pin_cells(self) -> frozenset[Coord]:
        return frozenset([self.root, *(b.goal for b in self.branches)])

    @cached_property
    def supports(self) -> tuple[Coord, ...]:
        """Solid blocks this route must place (pins rest on their cell's base)."""
        return tuple(support_of(c) for c in self.cells if c not in self.pin_cells)

    @cached_property
    def clearances(self) -> tuple[Coord, ...]:
        """Blocks that must stay air so every staircase step connects."""
        seen: dict[Coord, None] = {}
        for cell, up in self.parent.items():
            if up is not None:
                clear = clearance_of(up, cell)
                if clear is not None:
                    seen.setdefault(clear, None)
        return tuple(seen)

    @property
    def length(self) -> int:
        """Signal blocks used (the union of all branches)."""
        return len(self.parent)

    @property
    def steps(self) -> int:
        return sum(b.length for b in self.branches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "net": self.net,
            "driver": self.driver.ref(),
            "root": coord_list(self.root),
            "branches": [b.to_dict() for b in self.branches],
            "cells": [coord_list(c) for c in self.cells],
            "length": self.length,
            "fanout": len(self.branches),
        }


@dataclass(frozen=True)
class RoutingFailure:
    """Why routing gave up: ``unroutable`` (a branch search failed) or
    ``congestion`` (conflicts remained after the last iteration)."""

    reason: str
    iteration: int
    net: int | None = None
    sink: PinSite | None = None
    search: str | None = None
    expansions: int = 0
    conflicts: tuple[Conflict, ...] = ()

    @property
    def message(self) -> str:
        if self.reason == "congestion":
            return (
                f"routing did not converge: {len(self.conflicts)} redstone conflicts remain after "
                f"iteration {self.iteration}"
            )
        where = f" to primitive {self.sink.instance} pin {self.sink.pin!r}" if self.sink else ""
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
            "sink": None if self.sink is None else {**self.sink.ref(), "coord": coord_list(self.sink.cell)},
            "search": self.search,
            "expansions": self.expansions,
            "conflicts": [c.to_dict() for c in self.conflicts[:200]],
        }


@dataclass
class RoutingOutcome:
    success: bool
    routes: dict[int, RouteTree]
    iterations: int
    present_factor: float
    rip_ups: int
    failure: RoutingFailure | None = None
    conflicts: list[Conflict] = field(default_factory=list)


#: Branch-search modes, escalated when a branch exhausts its expansion budget.
NORMAL, GREEDY, IGNORE_CONGESTION = 0, 1, 2
SEARCH_MODES = ("normal", "greedy", "ignore_congestion")


def approach_corridor(site: PinSite) -> tuple[Coord, Coord]:
    """The two blocks a route reaches a pin through: the level block in front
    of it and the block above that (the top of a one-step staircase)."""
    x, y, z = site.cell
    fx, _, fz = site.facing.vector
    return ((x + fx, y, z + fz), (x + fx, y + 1, z + fz))


def route_requests(mapped: PrimitivePhysicalNetlist, grid: BlockGrid) -> list[RouteRequest]:
    """One :class:`RouteRequest` per net (id order) from the placed pin sites."""
    sites = {(site.instance, site.pin): site for site in grid.pins.values()}
    requests = []
    for net in mapped.nets:
        driver = sites[(net.driver.instance, net.driver.pin)]
        sinks = tuple(sites[(s.instance, s.pin)] for s in net.sinks)
        requests.append(RouteRequest(net.id, net.role, driver, sinks))
    return requests


class NegotiatedRedstoneRouter:
    """Routes :class:`RouteRequest` s on ``grid`` (see the module docstring)."""

    def __init__(
        self,
        grid: BlockGrid,
        requests: Sequence[RouteRequest],
        *,
        bounds: Bounds,
        config: PrimitivePnRConfig,
        trace: PrimitiveTraceRecorder | None = None,
        keyframe_state: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.grid = grid
        #: Extra state (placement, realized routes) a ``keyframe`` must carry
        #: so a replay can restart from it; supplied by the P&R driver.
        self.keyframe_state = keyframe_state
        self.bounds = bounds
        self.config = config
        self.trace = trace
        self.requests = {r.net: r for r in requests}
        self.order = sorted(requests, key=lambda r: (-r.fanout, -r.span, r.net))
        self.routes: dict[int, RouteTree] = {}
        self.present_factor = config.present_factor_initial
        self.iteration = 0
        self.rip_ups = 0
        #: net -> search mode it needed this iteration (reset every iteration).
        self._relax: dict[int, int] = {}
        for request in self.order:
            for site in (request.driver, *request.sinks):
                grid.claim_signal(request.net, site.cell)

    # -- one net ---------------------------------------------------------------

    def _rip_up(self, net: int, reason: str) -> None:
        old = self.routes.pop(net, None)
        self.grid.release(net, keep=self.requests[net].pins)
        self.rip_ups += 1
        if self.trace and old is not None:
            self.trace.emit(
                "routing",
                "net_rip_up",
                net=net,
                iteration=self.iteration,
                reason=reason,
                cells=[coord_list(c) for c in old.cells],
                length=old.length,
            )

    def route_net(self, request: RouteRequest) -> RouteTree | RoutingFailure:
        """Build ``request``'s tree in the grid's working state."""
        trace, net = self.trace, request.net
        iteration = self.iteration
        detailed = trace is not None and trace.wants(TraceLevel.DETAILED)
        if detailed:
            assert trace is not None
            trace.emit(
                "routing", "net_route_begin", level=TraceLevel.DETAILED, net=net, iteration=iteration,
                driver={**request.driver.ref(), "coord": coord_list(request.driver.cell)},
                sinks=[{**s.ref(), "coord": coord_list(s.cell)} for s in request.sinks],
                fanout=request.fanout,
            )  # fmt: skip
        root = request.driver.cell
        order = sorted(request.sinks, key=lambda s: (manhattan(root, s.cell), s.instance, s.pin))
        restarts = 0
        while True:
            outcome = self._grow_tree(request, order, detailed)
            if not isinstance(outcome, tuple):
                break
            failed, failure = outcome
            if restarts >= self.config.max_sink_order_restarts or len(order) < 2 or order[0] == failed:
                return failure
            # The net may have walled in its own sink: route that sink first.
            restarts += 1
            order = [failed] + [s for s in order if s != failed]
            if trace:
                trace.emit(
                    "routing", "net_route_restarted", net=net, iteration=iteration,
                    first_sink=failed.ref(), restart=restarts,
                )  # fmt: skip
        tree = outcome
        if trace:
            trace.emit("routing", "net_route_committed", iteration=iteration, **tree.to_dict())
        return tree

    def _grow_tree(
        self, request: RouteRequest, order: list[PinSite], detailed: bool
    ) -> RouteTree | tuple[PinSite, RoutingFailure]:
        """Connect the sinks in ``order``; on failure release the net's claims
        and return the failing sink with the failure."""
        trace, net, grid, iteration = self.trace, request.net, self.grid, self.iteration
        root = request.driver.cell
        state = NetState(net, request.driver, request.pins, {root}, set(), {})
        tree_order: list[Coord] = [root]
        branches: list[RouteBranch] = []
        for index, sink in enumerate(order):
            if detailed:
                assert trace is not None
                trace.emit(
                    "routing", "branch_route_begin", level=TraceLevel.DETAILED, net=net,
                    iteration=iteration, sink=sink.ref(), goal=coord_list(sink.cell),
                )  # fmt: skip
            reserved = frozenset(c for later in order[index + 1 :] for c in approach_corridor(later))
            path = self._search(request, state, tree_order, sink, reserved)
            if isinstance(path, RoutingFailure):
                grid.release(net, keep=request.pins)
                return sink, path
            self._commit(net, state, tree_order, path, sink)
            branch = RouteBranch(sink, path)
            branches.append(branch)
            if detailed:
                assert trace is not None
                trace.emit(
                    "routing", "branch_route_found", level=TraceLevel.DETAILED, net=net,
                    iteration=iteration, sink=sink.ref(), start=coord_list(branch.start),
                    goal=coord_list(branch.goal), path=[coord_list(c) for c in path],
                    length=branch.length,
                )  # fmt: skip
        return RouteTree(net, request.driver, root, tuple(branches))

    def _search(
        self,
        request: RouteRequest,
        state: NetState,
        tree_order: list[Coord],
        sink: PinSite,
        reserved: frozenset[Coord] = frozenset(),
    ) -> tuple[Coord, ...] | RoutingFailure:
        trace, config = self.trace, self.config
        net, iteration = request.net, self.iteration
        avoid: set[Coord] = set()
        sources = [c for c in tree_order if c not in state.pins or c == request.driver.cell]
        expand_hook = blocked_hook = None
        if trace and trace.wants(TraceLevel.SEARCH):
            expand_hook, blocked_hook = self._search_hooks(net, iteration, sink)
        # Long branches get a proportionally larger expansion budget.
        distance = min(manhattan(c, sink.cell) for c in sources)
        budget = max(config.max_astar_expansions, config.expansions_per_block * distance)
        # A net that already needed a relaxed search this iteration starts there.
        mode = self._relax.get(net, NORMAL)
        # Every search of this branch -- retries, relaxations, look-back
        # fallbacks -- draws on one effort budget, so a hopeless branch fails
        # fast and the attempt loop can spread the placement instead.
        effort, spent = budget * max(1, config.branch_effort), 0
        for attempt in range(config.max_path_retries + 3):
            if spent >= effort:
                return self._branch_failed(net, sink, "effort_limit", spent)
            greedy, ignore = mode >= GREEDY, mode >= IGNORE_CONGESTION
            search = partial(
                search_branch, self.grid, state, sources, sink,
                bounds=self.bounds, present_factor=0.0 if ignore else self.present_factor,
                vertical_cost=config.vertical_cost,
                max_expansions=min(budget * (2 if greedy else 1), effort - spent),
                weight=max(config.astar_weight, config.greedy_astar_weight) if greedy else config.astar_weight,
                history_weight=0.0 if ignore else 1.0, avoid=frozenset(avoid), reserved=reserved,
                on_expand=expand_hook, on_blocked=blocked_hook,
            )  # fmt: skip
            # After a rejected path, look back along the path's own blocks.  A*
            # keeps one parent per block, so a path-dependent rule can prune a
            # block's only stored approach: the look-back is an accelerator
            # only, and a look-back search that finds nothing is redone
            # without it.
            lookback = config.search_lookback if avoid else 0
            result = search(lookback=lookback)
            spent += result.expansions
            if result.path is None and lookback and spent < effort:
                result = search(lookback=0, max_expansions=min(budget * (2 if greedy else 1), effort - spent))
                spent += result.expansions
            if trace and trace.wants(TraceLevel.DETAILED):
                trace.emit(
                    "routing", "branch_search_stats", level=TraceLevel.DETAILED, net=net,
                    iteration=iteration, sink=sink.ref(), attempt=attempt, mode=SEARCH_MODES[mode],
                    expansions=result.expansions, found=result.found,
                    blocked=dict(sorted(result.blocked.items())),
                )  # fmt: skip
            if result.path is None and result.reason == "expansion_limit" and mode < IGNORE_CONGESTION:
                # Late in negotiation, history inflates costs far beyond the
                # distance heuristic and A* floods its budget.  First search
                # greedily (higher A* weight) while still pricing congestion;
                # only as a last resort ignore congestion and let later
                # iterations push the temporary overlap apart.
                mode += 1
                self._relax[net] = max(self._relax.get(net, NORMAL), mode)
                if trace:
                    trace.emit(
                        "routing", "branch_search_relaxed", net=net, iteration=iteration,
                        sink=sink.ref(), mode=SEARCH_MODES[mode], expansions=result.expansions,
                    )  # fmt: skip
                continue
            if result.path is None:
                return self._branch_failed(net, sink, result.reason or "unreachable", result.expansions)
            violation = validate_branch(state, result.path, sink.cell)
            if violation is None:
                return result.path
            reason, cell = violation
            avoid.add(cell)
            if trace and trace.wants(TraceLevel.DETAILED):
                trace.emit(
                    "routing", "branch_path_rejected", level=TraceLevel.DETAILED, net=net,
                    iteration=iteration, sink=sink.ref(), reason=reason, coord=coord_list(cell),
                    path=[coord_list(c) for c in result.path],
                )  # fmt: skip
        return self._branch_failed(net, sink, "intra_net_conflict", spent)

    def _branch_failed(self, net: int, sink: PinSite, reason: str, expansions: int) -> RoutingFailure:
        failure = RoutingFailure("unroutable", self.iteration, net, sink, reason, expansions)
        if self.trace:
            self.trace.emit(
                "routing", "branch_route_failed", net=net, iteration=self.iteration, sink=sink.ref(),
                goal=coord_list(sink.cell), reason=reason, expansions=expansions, message=failure.message,
            )  # fmt: skip
        return failure

    def _commit(
        self, net: int, state: NetState, tree_order: list[Coord], path: Sequence[Coord], sink: PinSite
    ) -> None:
        grid = self.grid
        for a, b in pairwise(path):
            if b not in state.pins:
                grid.claim_signal(net, b)
                sup = support_of(b)
                grid.claim_support(net, sup)
                state.supports.add(sup)
            clear = clearance_of(a, b)
            if clear is not None:
                grid.claim_clearance(net, clear)
                state.clearances[clear] = state.clearances.get(clear, 0) + 1
            if b not in state.tree:
                state.tree.add(b)
                tree_order.append(b)

    def _search_hooks(self, net: int, iteration: int, sink: PinSite):
        trace = self.trace
        assert trace is not None
        ref = sink.ref()
        cap = self.config.max_blocked_events_per_branch
        count = [0]

        def on_expand(cell: Coord, g: float, h: float, frontier: int) -> None:
            trace.emit(
                "routing", "route_search_expand", level=TraceLevel.SEARCH, net=net,
                iteration=iteration, sink=ref, coord=coord_list(cell), g=round(g, 6), h=round(h, 6),
                f=round(g + h, 6), frontier=frontier,
            )  # fmt: skip

        def on_blocked(a: Coord, b: Coord, reason: str) -> None:
            if count[0] >= cap:
                return
            count[0] += 1
            trace.emit(
                "routing", "route_transition_blocked", level=TraceLevel.SEARCH, net=net,
                iteration=iteration, sink=ref, to=coord_list(b), reason=reason, **{"from": coord_list(a)},
            )  # fmt: skip

        return on_expand, on_blocked

    # -- negotiation -----------------------------------------------------------

    def run(self) -> RoutingOutcome:
        """Initial routing of every net, then negotiation to a conflict-free state."""
        if self.trace:
            self.trace.emit(
                "routing",
                "routing_begin",
                nets=len(self.order),
                order=[r.net for r in self.order],
                search_bounds=self.bounds.to_dict(),
                present_factor=self.present_factor,
            )
        self._begin_iteration([r.net for r in self.order])
        for request in self.order:
            result = self.route_net(request)
            if isinstance(result, RoutingFailure):
                return self._finish(False, result)
            self.routes[request.net] = result
        return self._negotiate(len(self.order))

    def repair(self, nets: Iterable[int], *, reason: str, penalize: Iterable[Coord] = ()) -> RoutingOutcome:
        """Rip up ``nets`` (e.g. rejected by electrical legalization), add history
        on ``penalize`` so they look elsewhere, reroute them, then renegotiate."""
        nets = sorted(set(nets))
        self.grid.add_history(penalize, self.config.history_increment)
        self.iteration += 1
        self.present_factor *= self.config.present_factor_growth
        self._begin_iteration(nets)
        for net in nets:
            self._rip_up(net, reason)
        for request in self.order:
            if request.net in nets:
                result = self.route_net(request)
                if isinstance(result, RoutingFailure):
                    return self._finish(False, result)
                self.routes[request.net] = result
        return self._negotiate(len(nets))

    def _negotiate(self, rerouted: int) -> RoutingOutcome:
        config = self.config
        changed = rerouted
        start = self.iteration
        while True:
            conflicts = self.grid.conflicts()
            hot = sorted({c for conflict in conflicts for c in conflict.cells})
            if conflicts:
                self.grid.add_history(hot, config.history_increment)
            self._end_iteration(rerouted, changed, conflicts, hot)
            if not conflicts:
                return self._finish(True, None)
            if self.iteration - start >= config.max_routing_iterations:
                failure = RoutingFailure("congestion", self.iteration, conflicts=tuple(conflicts))
                return self._finish(False, failure, conflicts)
            self.iteration += 1
            self.present_factor *= config.present_factor_growth
            involved = {net for conflict in conflicts for net in conflict.nets}
            congested = [r for r in self.order if r.net in involved]
            self._begin_iteration([r.net for r in congested])
            rerouted, changed = len(congested), 0
            for request in congested:
                old = self.routes.get(request.net)
                self._rip_up(request.net, "congestion")
                result = self.route_net(request)
                if isinstance(result, RoutingFailure):
                    return self._finish(False, result)
                self.routes[request.net] = result
                if old is None or result.cells != old.cells:
                    changed += 1

    def _begin_iteration(self, nets: list[int]) -> None:
        self._relax = {}
        if self.trace:
            self.trace.emit(
                "routing",
                "routing_iteration_begin",
                iteration=self.iteration,
                present_factor=self.present_factor,
                nets=nets,
            )

    def _end_iteration(self, rerouted: int, changed: int, conflicts: list[Conflict], hot: list[Coord]) -> None:
        trace = self.trace
        if not trace:
            return
        grid = self.grid
        trace.emit(
            "routing",
            "routing_iteration_end",
            iteration=self.iteration,
            present_factor=self.present_factor,
            rerouted=rerouted,
            changed=changed,
            conflicts=len(conflicts),
            conflict_cells=len(hot),
            routed_cells=sum(r.length for r in self.routes.values()),
            history_total=float(sum(grid.history.values())),
        )
        snapshot_cells = [
            {
                "coord": coord_list(c),
                "history": grid.history.get(c, 0.0),
                "nets": sorted(set(grid.signal.get(c, {})) | set(grid.support.get(c, {}))),
            }
            for c in hot[:2000]
        ]
        trace.emit(
            "congestion",
            "congestion_snapshot",
            iteration=self.iteration,
            present_factor=self.present_factor,
            conflicts=[c.to_dict() for c in conflicts[:2000]],
            cells=snapshot_cells,
            history_cells=len(grid.history),
        )
        if trace.wants(TraceLevel.DETAILED):
            for conflict in conflicts[:500]:
                trace.emit(
                    "congestion", "physical_conflict", level=TraceLevel.DETAILED,
                    iteration=self.iteration, **conflict.to_dict(),
                )  # fmt: skip
        interval = self.config.keyframe_interval
        if interval and self.iteration % interval == 0:
            extra = self.keyframe_state() if self.keyframe_state is not None else {}
            trace.emit(
                "routing",
                "keyframe",
                iteration=self.iteration,
                routes=[self.routes[n].to_dict() for n in sorted(self.routes)],
                congestion=snapshot_cells,
                **extra,
            )

    def _finish(
        self, success: bool, failure: RoutingFailure | None, conflicts: list[Conflict] | None = None
    ) -> RoutingOutcome:
        if self.trace:
            self.trace.emit(
                "routing",
                "routing_complete" if success else "routing_failed",
                iteration=self.iteration,
                iterations=self.iteration + 1,
                routed_nets=len(self.routes),
                rip_ups=self.rip_ups,
                failure=failure.to_dict() if failure else None,
            )
        return RoutingOutcome(
            success=success,
            routes=dict(self.routes),
            iterations=self.iteration + 1,
            present_factor=self.present_factor,
            rip_ups=self.rip_ups,
            failure=failure,
            conflicts=list(conflicts or []),
        )


__all__ = [
    "NegotiatedRedstoneRouter",
    "RouteBranch",
    "RouteRequest",
    "RouteTree",
    "RoutingFailure",
    "RoutingOutcome",
    "approach_corridor",
    "route_requests",
]
