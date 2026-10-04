"""Redstone electrical legalization: geometric route tree -> realized route.

    BitNet --routing--> RouteTree (geometry) --legalization--> RealizedRoute
                                                  dust | repeater | dust | ...

Geometric routing only decides WHERE a net's signal blocks go.  Legalization
decides WHAT each block is, honouring Minecraft's finite signal strength:

* the driver's cell delivers ``strength`` (usually 15) into the root pin dust;
* every further dust block along the DIRECTED tree (root -> sinks) is one
  weaker; a repeater instead re-drives the block in front of it to 15 and
  delays by its configured setting (legalization inserts
  :data:`~redc.physical_primitive.redstone.REPEATER_DELAY_TICKS`; clock-tree
  balancing may raise it to 4 or add repeaters -- see :func:`realize_route`);
* every dust block must stay at strength >= 1 and every sink pin at or above
  the sink cell's required strength.

Repeaters are physical ROUTE elements, never netlist components.  They are
inserted greedily, as far downstream as possible, on eligible blocks: a
straight, level segment (parent behind, exactly one child in front, same y)
that is not a pin.  Because the strength recomputation walks the real tree, a
repeater on a shared trunk refreshes every downstream branch, and branches
that need their own repeaters get them separately.  (Rule 1 of the redstone
model guarantees the electrical graph IS the tree, so no repeater can ever be
bypassed or feed itself.)  A net whose driver never powers its pin (a
constant-0 anchor) needs no repeaters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...minecraft.timing import repeater_delay_gt
from ...minecraft.units import gt_to_rt
from ..geometry import Coord, Direction, coord_list
from ..grid import PinSite
from ..redstone import MAX_SIGNAL_STRENGTH, REPEATER_DELAY_TICKS, ElementKind
from .routing import RouteRequest, RouteTree


@dataclass(frozen=True, slots=True)
class RouteElement:
    """One realized signal block.  ``facing`` is a repeater's OUTPUT direction
    (the way the signal leaves it).  Careful: Minecraft's own
    ``minecraft:repeater[facing=...]`` blockstate names the INPUT side -- the
    opposite -- which is exported separately as ``blockstate_facing``.
    ``strength`` is a dust block's signal strength (a repeater's INPUT strength)
    when the driver is on; ``delay`` the redstone ticks of repeater delay
    between root and here; ``setting`` a repeater's own delay (1..4 rt)."""

    coord: Coord
    kind: ElementKind
    parent: Coord | None
    strength: int
    delay: int
    facing: Direction | None = None
    setting: int | None = None

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "coord": coord_list(self.coord),
            "kind": self.kind.value,
            "parent": None if self.parent is None else coord_list(self.parent),
            "strength": self.strength,
            "delay": self.delay,
        }
        if self.facing is not None:
            record["facing"] = self.facing.label
            record["blockstate_facing"] = self.facing.opposite.label
        if self.setting is not None:
            record["repeater_delay"] = self.setting
        return record


@dataclass(frozen=True, slots=True)
class SinkReport:
    """Electrical result at one sink pin."""

    sink: PinSite
    strength: int
    required: int
    repeaters: int
    delay_ticks: int
    distance: int

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.sink.ref(),
            "coord": coord_list(self.sink.cell),
            "strength": self.strength,
            "required": self.required,
            "repeaters": self.repeaters,
            "delay_ticks": self.delay_ticks,
            "distance": self.distance,
        }


@dataclass(frozen=True)
class RealizedRoute:
    """A legal, buildable route: every signal block typed, plus the solid
    supports to place and the air clearances to keep."""

    net: int
    tree: RouteTree
    elements: tuple[RouteElement, ...]
    supports: tuple[Coord, ...]
    clearances: tuple[Coord, ...]
    sinks: tuple[SinkReport, ...]
    powered: bool

    @property
    def repeaters(self) -> tuple[RouteElement, ...]:
        return tuple(e for e in self.elements if e.kind is ElementKind.REPEATER)

    @property
    def dust(self) -> tuple[RouteElement, ...]:
        return tuple(e for e in self.elements if e.kind is ElementKind.DUST)

    def to_dict(self) -> dict[str, Any]:
        return {
            "net": self.net,
            "powered": self.powered,
            "elements": [e.to_dict() for e in self.elements],
            "repeaters": [e.to_dict() for e in self.repeaters],
            "supports": [coord_list(c) for c in self.supports],
            "clearances": [coord_list(c) for c in self.clearances],
            "sinks": [s.to_dict() for s in self.sinks],
            "min_strength": min((e.strength for e in self.dust), default=0) if self.powered else 0,
            "max_delay_ticks": max((s.delay_ticks for s in self.sinks), default=0),
        }


@dataclass(frozen=True)
class LegalizationFailure:
    """Why a net could not be made electrically legal."""

    net: int
    reason: str
    coord: Coord
    sink: PinSite | None = None
    #: The failing branch, weak block first, back to the root.
    cells: tuple[Coord, ...] = field(default_factory=tuple)

    @property
    def message(self) -> str:
        return f"net {self.net}: {self.reason} at {coord_list(self.coord)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "net": self.net,
            "reason": self.reason,
            "message": self.message,
            "coord": coord_list(self.coord),
            "sink": None if self.sink is None else self.sink.ref(),
        }


def repeater_eligible(tree: RouteTree, cell: Coord) -> bool:
    """A straight, level, non-pin segment block with exactly one child."""
    if cell in tree.pin_cells:
        return False
    up = tree.parent.get(cell)
    kids = tree.children.get(cell, ())
    if up is None or len(kids) != 1:
        return False
    down = kids[0]
    if not up[1] == cell[1] == down[1]:
        return False
    return (cell[0] - up[0], cell[2] - up[2]) == (down[0] - cell[0], down[2] - cell[2])


def signal_levels(tree: RouteTree, drive: int, repeaters: set[Coord]) -> dict[Coord, int]:
    """The level of every block when the driver is on.

    A dust block's level is its signal strength; a repeater's level is its
    INPUT strength.  Dust after dust loses one; anything after a powered
    repeater is driven at 15; a repeater after dust receives that dust's full
    strength (it is powered iff that is >= 1)."""
    level: dict[Coord, int] = {}
    for cell in tree.cells:  # root-first construction order is a topological order
        up = tree.parent[cell]
        if up is None:
            level[cell] = drive
        elif up in repeaters:
            level[cell] = MAX_SIGNAL_STRENGTH if level[up] >= 1 else 0
        elif cell in repeaters:
            level[cell] = level[up]
        else:
            level[cell] = max(0, level[up] - 1)
    return level


def legalize_route(tree: RouteTree, request: RouteRequest) -> RealizedRoute | LegalizationFailure:
    """Type every block of ``tree`` and insert the repeaters it needs."""
    drive = request.driver.strength
    required = {s.cell: s.strength for s in request.sinks}
    repeaters: set[Coord] = set()
    powered = drive > 0
    while powered:
        strength = signal_levels(tree, drive, repeaters)
        weak = next(
            (c for c in tree.cells if c not in repeaters and strength[c] < required.get(c, 1)),
            None,
        )
        if weak is None:
            break
        # Walk upstream from the weak block to the latest eligible repeater site
        # whose input (the dust behind it) is still powered.
        site = None
        cursor: Coord | None = weak
        while cursor is not None and cursor not in repeaters:
            up = tree.parent[cursor]
            if up is not None and repeater_eligible(tree, cursor) and strength[up] >= 1:
                site = cursor
                break
            cursor = up
        if site is None:
            sink = next((s for s in request.sinks if s.cell == weak), None)
            upstream: list[Coord] = []
            walk: Coord | None = weak
            while walk is not None:
                upstream.append(walk)
                walk = tree.parent[walk]
            return LegalizationFailure(
                tree.net, "signal too weak and no repeater site upstream", weak, sink, tuple(upstream)
            )
        repeaters.add(site)
    return realize_route(tree, request, {c: REPEATER_DELAY_TICKS for c in repeaters})


def realize_route(tree: RouteTree, request: RouteRequest, repeaters: Mapping[Coord, int]) -> RealizedRoute:
    """Type every block of ``tree`` given the repeater sites and their delay
    settings (redstone ticks).  Strength and per-sink delay are recomputed
    from scratch; the caller guarantees every site is eligible."""
    drive = request.driver.strength
    powered = drive > 0
    sites = set(repeaters)
    strength = signal_levels(tree, drive, sites) if powered else {c: 0 for c in tree.cells}
    delay: dict[Coord, int] = {}
    elements: list[RouteElement] = []
    for cell in tree.cells:
        up = tree.parent[cell]
        if up is None:
            delay[cell] = 0
        else:
            delay[cell] = delay[up] + (gt_to_rt(repeater_delay_gt(repeaters[up])) if up in repeaters else 0)
        if cell in repeaters:
            (child,) = tree.children[cell]
            facing = Direction.of((child[0] - cell[0], 0, child[2] - cell[2]))
            elements.append(
                RouteElement(cell, ElementKind.REPEATER, up, strength[cell], delay[cell], facing, repeaters[cell])
            )
        else:
            elements.append(RouteElement(cell, ElementKind.DUST, up, strength[cell], delay[cell]))
    sinks = []
    for branch in tree.branches:
        goal = branch.goal
        count, cursor, distance = 0, tree.parent[goal], 0
        while cursor is not None:
            distance += 1
            count += cursor in repeaters
            cursor = tree.parent[cursor]
        sinks.append(SinkReport(branch.sink, strength[goal], branch.sink.strength, count, delay[goal], distance))
    return RealizedRoute(
        net=tree.net,
        tree=tree,
        elements=tuple(elements),
        supports=tree.supports,
        clearances=tree.clearances,
        sinks=tuple(sinks),
        powered=powered,
    )


__all__ = [
    "LegalizationFailure",
    "RealizedRoute",
    "RouteElement",
    "SinkReport",
    "legalize_route",
    "realize_route",
    "repeater_eligible",
]
