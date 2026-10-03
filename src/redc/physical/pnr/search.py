"""Deterministic six-connected A* over the routing grid.

The search grows from a SET of source cells (the net's existing routing tree,
all at cost 0) to one goal cell (the next sink's escape), so a fanout branch
attaches wherever the tree is cheapest to reach.  Step costs come from
:meth:`Grid.routing_cost` with the routing net's id (PathFinder congestion), the
heuristic is Manhattan distance -- admissible because every step costs at least
:data:`~redc.physical.grid.BASE_COST` -- and the search is confined to a
bounding box and capped in node expansions so it can never run away into the
unbounded horizontal plane.

Ties are broken by ``(f, h, insertion sequence)``: prefer the node closer to
the goal, then the one discovered first.  Neighbour order is the grid's fixed
move order, so equal inputs always give the identical path.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..grid import Grid
from .geometry import Bounds, Coord, manhattan

#: ``(cell, g, h, frontier_size) -> None`` -- called on every node expansion.
ExpandHook = Callable[[Coord, float, float, int], None]


@dataclass(frozen=True)
class SearchResult:
    """A found path (first cell on the existing tree, last cell the goal), or
    ``path is None`` with a ``reason``: ``goal_out_of_bounds``,
    ``goal_blocked``, ``expansion_limit`` or ``unreachable``."""

    path: tuple[Coord, ...] | None
    expansions: int
    reason: str | None = None

    @property
    def found(self) -> bool:
        return self.path is not None


def astar(
    grid: Grid,
    sources: Sequence[Coord],
    goal: Coord,
    *,
    net: int,
    bounds: Bounds,
    present_factor: float,
    max_expansions: int,
    forbidden: Callable[[Coord], bool] | None = None,
    on_expand: ExpandHook | None = None,
) -> SearchResult:
    """Cheapest path from any of ``sources`` to ``goal`` for ``net``.

    ``forbidden(cell)`` excludes extra cells (other nets' pin escapes); it is
    never consulted for the goal itself."""
    if not bounds.contains(goal):
        return SearchResult(None, 0, "goal_out_of_bounds")
    if not grid.is_routable(*goal):
        return SearchResult(None, 0, "goal_blocked")
    if goal in sources:
        return SearchResult((goal,), 0)

    best: dict[Coord, float] = {}
    parent: dict[Coord, Coord | None] = {}
    heap: list[tuple[float, int, int, float, Coord]] = []
    seq = 0
    for source in sources:
        if source in best:
            continue
        best[source] = 0.0
        parent[source] = None
        h = manhattan(source, goal)
        heap.append((float(h), h, seq, 0.0, source))
        seq += 1
    heapq.heapify(heap)

    closed: set[Coord] = set()
    expansions = 0
    while heap:
        _f, h, _seq, g, cell = heapq.heappop(heap)
        if cell in closed or g > best[cell]:
            continue
        closed.add(cell)
        expansions += 1
        if on_expand is not None:
            on_expand(cell, g, float(h), len(heap))
        if cell == goal:
            path = [cell]
            previous = parent[cell]
            while previous is not None:
                path.append(previous)
                previous = parent[previous]
            path.reverse()
            return SearchResult(tuple(path), expansions)
        if expansions >= max_expansions:
            return SearchResult(None, expansions, "expansion_limit")
        for neighbour in grid.neighbors(*cell):
            if neighbour in closed or not bounds.contains(neighbour):
                continue
            if forbidden is not None and neighbour != goal and forbidden(neighbour):
                continue
            step = grid.routing_cost(*neighbour, present_factor=present_factor, net=net)
            if math.isinf(step):
                continue
            ng = g + step
            if ng < best.get(neighbour, math.inf):
                best[neighbour] = ng
                parent[neighbour] = cell
                nh = manhattan(neighbour, goal)
                heapq.heappush(heap, (ng + nh, nh, seq, ng, neighbour))
                seq += 1
    return SearchResult(None, expansions, "unreachable")
