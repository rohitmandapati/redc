"""Deterministic redstone-aware A* for one branch of one single-bit net.

The search grows from a SET of source blocks (the net's existing routing tree,
all at cost 0) to the next sink's pin block, so fanout branches attach wherever
the tree is cheapest to reach.  A state is the block a signal (dust) would
occupy; the moves are :data:`~redc.physical_primitive.redstone.MOVES` -- four
level steps and eight one-up/one-down staircase steps.  There is no vertical
move.

Hard rules (a move is impossible):

* the block, its support and (for a staircase) its clearance must respect
  component bodies, keep-outs and pins (:mod:`redc.physical_primitive.redstone`
  rules 2, 3, 6);
* pins are entered / left only along their facing;
* intra-net consistency: never reuse the net's own signal, support or
  clearance blocks inconsistently, and -- rule 1 -- a new block may touch the
  net's existing signal blocks (tree + its other pins) only through the block
  it extends from, so the realized electrical graph stays exactly the tree.

Soft costs (negotiated congestion): sharing a block, support or clearance
with another net, or sitting in another net's signal neighbourhood, multiplies
the step cost by ``1 + present_factor * overlap``; accumulated ``history`` adds
to the base cost.  The heuristic ``max(H, Y) + vertical_cost * Y`` (``H`` the
horizontal and ``Y`` the vertical distance) is admissible.  Ties break by
``(f, h, insertion order)`` and neighbours are tried continuing straight first,
which biases routes toward straight runs (good repeater sites).

Rules that depend on the path's own earlier blocks are not Markovian; they are
re-checked by :func:`validate_branch`, and the router re-searches with the
offending blocks avoided.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from ..geometry import Bounds, Coord
from ..grid import STATIC_BODY, STATIC_PIN, BlockGrid, PinSite
from ..redstone import (
    MIN_SIGNAL_Y,
    MOVES,
    SIGNAL_NEIGHBORHOOD,
    clearance_of,
    support_of,
)

#: ``(block, g, h, frontier_size) -> None``, called on every node expansion.
ExpandHook = Callable[[Coord, float, float, int], None]
#: ``(from, to, reason) -> None``, called on a hard-blocked move (SEARCH traces).
BlockedHook = Callable[[Coord, Coord, str], None]

_NEIGHBORHOOD = frozenset(SIGNAL_NEIGHBORHOOD)

#: MOVES reordered so the moves continuing a given horizontal heading come first.
def _straight_first(heading: tuple[int, int]) -> tuple[Coord, ...]:
    return tuple(sorted(MOVES, key=lambda move: (move[0], move[2]) != heading))


_STRAIGHT_FIRST: dict[tuple[int, int], tuple[Coord, ...]] = {
    (m[0], m[2]): _straight_first((m[0], m[2])) for m in MOVES
}


@dataclass(frozen=True)
class SearchResult:
    """A path (first block on the existing tree, last block the goal pin), or
    ``path is None`` with ``reason``: ``goal_out_of_bounds``,
    ``expansion_limit`` or ``unreachable``."""

    path: tuple[Coord, ...] | None
    expansions: int
    reason: str | None = None
    blocked: dict[str, int] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.path is not None


@dataclass
class NetState:
    """What the router knows about the net being routed."""

    net: int
    root: PinSite
    pins: frozenset[Coord]  # every pin block of this net (driver + all sinks)
    tree: set[Coord]  # signal blocks connected so far (root included)
    supports: set[Coord]  # supports this net placed
    clearances: dict[Coord, int]  # clearances this net requires (refcount)


def search_branch(
    grid: BlockGrid,
    state: NetState,
    sources: Sequence[Coord],
    goal: PinSite,
    *,
    bounds: Bounds,
    present_factor: float,
    vertical_cost: float,
    max_expansions: int,
    weight: float = 1.0,
    history_weight: float = 1.0,
    avoid: frozenset[Coord] = frozenset(),
    reserved: frozenset[Coord] = frozenset(),
    lookback: int = 0,
    goal_link: Coord | None = None,
    on_expand: ExpandHook | None = None,
    on_blocked: BlockedHook | None = None,
) -> SearchResult:
    """Cheapest legal path from any of ``sources`` to the ``goal`` pin block.

    ``weight`` > 1 runs *weighted* A* (``f = g + weight * h``, VPR's
    ``astar_fac``): no longer guaranteed optimal, but it stops long branches
    from flooding the search volume when every column crossing costs a climb.
    ``history_weight`` scales the accumulated-congestion term (0 = ignore it).
    ``reserved`` blocks may hold none of this branch's signals, supports or
    clearances: the approach corridors of the net's still-unreached sinks, so
    a trunk can never wall them off from their own net.
    ``lookback`` > 0 also checks every move against that many blocks of the
    path's own parent chain (supports, clearances, self-contact), catching
    most of the non-Markovian rules before :func:`validate_branch` has to.
    ``goal_link``: the goal is a clock-tap entry whose tap continues to this
    (already reserved) block -- the one own signal the goal may touch."""
    gx, gy, gz = goal_cell = goal.cell
    if not bounds.contains(goal_cell):
        return SearchResult(None, 0, "goal_out_of_bounds")
    if goal_cell in state.tree:
        return SearchResult((goal_cell,), 0)
    net = state.net
    static, used = grid.static, grid.used
    signal, support, clearance = grid.signal, grid.support, grid.clearance
    adjacent, history = grid.adjacent, grid.history
    tree, own_supports, own_clearances = state.tree, state.supports, state.clearances
    own_pins = state.pins
    root_cell = state.root.cell
    root_step = (state.root.facing.vector[0], state.root.facing.vector[2])
    goal_step = (-goal.facing.vector[0], -goal.facing.vector[2])
    lo_x, lo_y, lo_z = bounds.lo
    hi_x, hi_y, hi_z = bounds.hi
    lo_y = max(lo_y, MIN_SIGNAL_Y)
    blocked: dict[str, int] = {}

    def block(a: Coord, b: Coord, why: str) -> None:
        blocked[why] = blocked.get(why, 0) + 1
        if on_blocked is not None:
            on_blocked(a, b, why)

    goal_hood = frozenset((gx + dx, gy + dy, gz + dz) for dx, dy, dz in SIGNAL_NEIGHBORHOOD)
    air_blockers = STATIC_BODY | STATIC_PIN

    best: dict[Coord, float] = {}
    parent: dict[Coord, Coord | None] = {}
    heap: list[tuple[float, float, int, float, Coord]] = []
    seq = 0

    def heuristic(cell: Coord) -> float:
        horizontal = abs(gx - cell[0]) + abs(gz - cell[2])
        vertical = abs(gy - cell[1])
        return max(horizontal, vertical) + vertical_cost * vertical

    for source in sources:
        if source in best:
            continue
        best[source] = 0.0
        parent[source] = None
        h = heuristic(source)
        heap.append((weight * h, h, seq, 0.0, source))
        seq += 1
    heapq.heapify(heap)
    closed: set[Coord] = set()
    expansions = 0
    last_step: dict[Coord, Coord] = {}
    while heap:
        _f, h, _s, g, cell = heapq.heappop(heap)
        if cell in closed or g > best[cell]:
            continue
        closed.add(cell)
        expansions += 1
        if on_expand is not None:
            on_expand(cell, g, h, len(heap))
        if cell == goal_cell:
            path = [cell]
            prev = parent[cell]
            while prev is not None:
                path.append(prev)
                prev = parent[prev]
            path.reverse()
            return SearchResult(tuple(path), expansions, None, blocked)
        if expansions >= max_expansions:
            return SearchResult(None, expansions, "expansion_limit", blocked)
        px, py, pz = cell
        from_tree = cell in tree
        if lookback:
            chain: list[Coord] = []
            walk: Coord | None = cell
            while walk is not None and walk not in tree and len(chain) < lookback:
                chain.append(walk)
                walk = parent[walk]
            path_cells = set(chain)
            path_supports = {(c[0], c[1] - 1, c[2]) for c in chain if c not in own_pins}
            path_clear = set()
            for later, earlier in pairwise(chain):
                step_clear = clearance_of(earlier, later)
                if step_clear is not None:
                    path_clear.add(step_clear)
            if walk is not None and chain:  # the step leaving the tree
                step_clear = clearance_of(walk, chain[-1])
                if step_clear is not None:
                    path_clear.add(step_clear)
            path_hood = {(c[0] + dx, c[1] + dy, c[2] + dz) for c in chain[1:] for dx, dy, dz in SIGNAL_NEIGHBORHOOD}
        straight = last_step.get(cell)
        moves = MOVES if straight is None else _STRAIGHT_FIRST[(straight[0], straight[2])]
        for dx, dy, dz in moves:
            nxt = (px + dx, py + dy, pz + dz)
            nx, ny, nz = nxt
            if nxt in closed:
                continue
            if not (lo_x <= nx <= hi_x and lo_y <= ny <= hi_y and lo_z <= nz <= hi_z):
                continue
            if nxt in avoid:
                block(cell, nxt, "avoid")
                continue
            if nxt in reserved:
                block(cell, nxt, "reserved_approach")
                continue
            if nxt in tree or nxt in own_supports or nxt in own_clearances:
                block(cell, nxt, "own_route")
                continue
            if cell == root_cell and (dx, dz) != root_step:
                block(cell, nxt, "pin_facing")
                continue
            is_goal = nxt == goal_cell
            sup = (nx, ny - 1, nz)
            if is_goal:
                if (dx, dz) != goal_step:
                    block(cell, nxt, "pin_facing")
                    continue
            else:
                mask = static.get(nxt)
                if mask:
                    block(cell, nxt, "pin" if mask & STATIC_PIN else "component")
                    continue
                if static.get(sup):
                    block(cell, nxt, "support_blocked")
                    continue
                if sup in reserved:
                    block(cell, nxt, "reserved_approach")
                    continue
                if sup in tree or sup in own_clearances:
                    block(cell, nxt, "own_route")
                    continue
            clear: Coord | None = None
            if dy:
                clear = (px, py + 1, pz) if dy > 0 else (nx, ny + 1, nz)
                if static.get(clear, 0) & air_blockers or clear in tree or clear in own_supports or clear in reserved:
                    block(cell, nxt, "clearance_blocked")
                    continue
            if lookback and (
                nxt in path_hood
                or nxt in path_clear
                or (not is_goal and (sup in path_cells or sup in path_clear))
                or (clear is not None and (clear in path_cells or clear in path_supports))
            ):
                block(cell, nxt, "own_path")
                continue
            # Rule 1 (Markovian part): touch own signal blocks only via `cell`.
            own = adjacent.get(nxt)
            touching = own.get(net, 0) if own else 0
            if not is_goal and nxt in goal_hood or is_goal and goal_link is not None:
                touching -= 1
            if touching != (1 if from_tree else 0):
                block(cell, nxt, "own_adjacency")
                continue
            # -- soft costs (blocks no route touches need no lookups) -----------
            overlap = 0
            if used.get(nxt):
                users = signal.get(nxt)
                if users:
                    overlap += len(users) - (1 if net in users else 0)
                users = support.get(nxt)
                if users:
                    overlap += len(users)
                users = clearance.get(nxt)
                if users:
                    overlap += len(users) - (1 if net in users else 0)
            if not is_goal and used.get(sup):
                users = signal.get(sup)
                if users:
                    overlap += len(users)
                users = support.get(sup)
                if users:
                    overlap += len(users) - (1 if net in users else 0)
                users = clearance.get(sup)
                if users:
                    overlap += len(users) - (1 if net in users else 0)
            if clear is not None and used.get(clear):
                users = signal.get(clear)
                if users:
                    overlap += len(users)
                users = support.get(clear)
                if users:
                    overlap += len(users) - (1 if net in users else 0)
            if own:
                overlap += len(own) - (1 if net in own else 0)
            base = 1.0 + (vertical_cost if dy else 0.0) + history_weight * history.get(nxt, 0.0)
            if not is_goal:
                base += history_weight * history.get(sup, 0.0)
            step = base * (1.0 + present_factor * overlap)
            ng = g + step
            if ng < best.get(nxt, math.inf):
                best[nxt] = ng
                parent[nxt] = cell
                last_step[nxt] = (dx, dy, dz)
                vertical = abs(gy - ny)
                nh = max(abs(gx - nx) + abs(gz - nz), vertical) + vertical_cost * vertical
                heapq.heappush(heap, (ng + weight * nh, nh, seq, ng, nxt))
                seq += 1
    return SearchResult(None, expansions, "unreachable", blocked)


def validate_branch(
    state: NetState, path: Sequence[Coord], goal: Coord, goal_link: Coord | None = None
) -> tuple[str, Coord] | None:
    """Re-check the non-Markovian intra-net rules on a found path.

    Returns ``(reason, offending NEW path block)`` -- a block the re-search can
    avoid -- or ``None``.  Checks that every new block touches only its path
    neighbours (rule 1, including the path touching itself and the goal being
    touched by anything but its predecessor), that no new support is a signal
    or a clearance of the path or the tree, and that no new clearance is a
    signal or support."""
    if len(path) < 2:
        return None
    own_signals = state.tree | (state.pins - {goal})
    index = {cell: i for i, cell in enumerate(path)}
    for i, cell in enumerate(path):
        if i == 0:
            continue
        x, y, z = cell
        for dx, dy, dz in SIGNAL_NEIGHBORHOOD:
            nb = (x + dx, y + dy, z + dz)
            j = index.get(nb)
            if j is not None:
                if abs(j - i) != 1:
                    return ("path_touches_itself", cell)
                continue
            if nb in own_signals and not (cell == goal and nb == goal_link):
                return ("path_touches_own_net", cell)
    signals = set(path)
    steps = list(pairwise(path))
    clearances: dict[Coord, Coord] = {}  # clearance block -> the later block of its step
    for a, b in steps:
        clear = clearance_of(a, b)
        if clear is not None:
            clearances.setdefault(clear, b)
    supports = {support_of(c): c for c in path[1:] if c != goal}
    for sup, owner in supports.items():
        if sup in signals:
            return ("support_on_signal", owner)
        if sup in clearances or sup in state.clearances:
            return ("support_in_clearance", owner)
    for clear, owner in clearances.items():
        if clear in signals or clear in state.tree or clear in state.supports or clear in supports:
            return ("clearance_blocked", owner)
    return None


__all__ = ["BlockedHook", "ExpandHook", "NetState", "SearchResult", "search_branch", "validate_branch"]
