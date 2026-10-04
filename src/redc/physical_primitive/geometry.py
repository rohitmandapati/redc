"""Block-resolution geometry for the ``physical-primitive`` backend.

ONE coordinate is ONE Minecraft block position.  ``(x, y, z)`` with x pointing
east, y up and z south (Minecraft's own axes); block ``(x, y, z)`` is the unit
cube ``[x, x+1) x [y, y+1) x [z, z+1)``.  There is no tile, sub-chunk or
abstract routing-cube scale anywhere in this backend.

Technology cells are described in *local* coordinates with their signal flow
along local +x (:attr:`Direction.EAST`); an :class:`Orientation` rotates them
about the vertical axis in quarter turns before they are translated to their
placed origin.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

Coord = tuple[int, int, int]


def add(a: Coord, b: Coord) -> Coord:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Coord, b: Coord) -> Coord:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def coord_list(cell: Coord) -> list[int]:
    """A coordinate as the JSON ``[x, y, z]`` array."""
    return [int(cell[0]), int(cell[1]), int(cell[2])]


def below(cell: Coord) -> Coord:
    return (cell[0], cell[1] - 1, cell[2])


def above(cell: Coord) -> Coord:
    return (cell[0], cell[1] + 1, cell[2])


class Direction(Enum):
    """A horizontal direction (redstone signal never travels diagonally in a
    plane).  Values are unit vectors."""

    EAST = (1, 0, 0)
    SOUTH = (0, 0, 1)
    WEST = (-1, 0, 0)
    NORTH = (0, 0, -1)

    @property
    def vector(self) -> Coord:
        return self.value

    @property
    def opposite(self) -> Direction:
        return _OPPOSITE[self]

    @property
    def label(self) -> str:
        return self.name.lower()

    def rotated(self, quarter_turns: int) -> Direction:
        """This direction turned clockwise (seen from above) ``quarter_turns`` times."""
        return CLOCKWISE[(CLOCKWISE.index(self) + quarter_turns) % 4]

    @classmethod
    def of(cls, vector: Coord) -> Direction:
        """The direction of a horizontal unit step (y component ignored)."""
        return _BY_VECTOR[(vector[0], 0, vector[2])]

    @classmethod
    def parse(cls, name: str) -> Direction:
        return cls[name.upper()]


#: Clockwise order seen from above (x east, z south): east -> south -> west -> north.
CLOCKWISE: tuple[Direction, ...] = (Direction.EAST, Direction.SOUTH, Direction.WEST, Direction.NORTH)
_OPPOSITE = {
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
    Direction.SOUTH: Direction.NORTH,
    Direction.NORTH: Direction.SOUTH,
}
_BY_VECTOR = {d.value: d for d in Direction}


@dataclass(frozen=True, slots=True)
class Orientation:
    """A rotation about +y by ``quarter_turns`` clockwise quarter turns.

    Named after where local :attr:`Direction.EAST` (a cell's signal-flow
    direction) points after rotation: ``east`` (identity), ``south``, ``west``,
    ``north``."""

    quarter_turns: int

    def __post_init__(self) -> None:
        if not 0 <= self.quarter_turns < 4:
            raise ValueError(f"quarter_turns must be 0..3, got {self.quarter_turns}")

    def apply(self, cell: Coord) -> Coord:
        x, y, z = cell
        for _ in range(self.quarter_turns):
            x, z = -z, x
        return (x, y, z)

    def direction(self, direction: Direction) -> Direction:
        return direction.rotated(self.quarter_turns)

    @property
    def name(self) -> str:
        return Direction.EAST.rotated(self.quarter_turns).label

    @classmethod
    def parse(cls, name: str) -> Orientation:
        return cls(CLOCKWISE.index(Direction.parse(name)))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "quarter_turns": self.quarter_turns}


IDENTITY = Orientation(0)
ORIENTATIONS: tuple[Orientation, ...] = tuple(Orientation(q) for q in range(4))


@dataclass(frozen=True, slots=True)
class Bounds:
    """An inclusive axis-aligned box of blocks: ``lo[i] <= c[i] <= hi[i]``."""

    lo: Coord
    hi: Coord

    @classmethod
    def of(cls, cells: Iterable[Coord]) -> Bounds | None:
        """The tightest box around ``cells`` (``None`` if there are none)."""
        it = iter(cells)
        first = next(it, None)
        if first is None:
            return None
        lx, ly, lz = first
        hx, hy, hz = first
        for x, y, z in it:
            if x < lx:
                lx = x
            elif x > hx:
                hx = x
            if y < ly:
                ly = y
            elif y > hy:
                hy = y
            if z < lz:
                lz = z
            elif z > hz:
                hz = z
        return cls((lx, ly, lz), (hx, hy, hz))

    def union(self, other: Bounds | None) -> Bounds:
        if other is None:
            return self
        return Bounds(
            (min(self.lo[0], other.lo[0]), min(self.lo[1], other.lo[1]), min(self.lo[2], other.lo[2])),
            (max(self.hi[0], other.hi[0]), max(self.hi[1], other.hi[1]), max(self.hi[2], other.hi[2])),
        )

    def expand(self, dx: int, dz: int, y_range: tuple[int, int] | None = None) -> Bounds:
        """Grow by ``dx`` / ``dz`` horizontally; optionally replace the y span."""
        lo_y, hi_y = y_range if y_range is not None else (self.lo[1], self.hi[1])
        return Bounds(
            (self.lo[0] - dx, lo_y, self.lo[2] - dz),
            (self.hi[0] + dx, hi_y, self.hi[2] + dz),
        )

    def translate(self, offset: Coord) -> Bounds:
        return Bounds(add(self.lo, offset), add(self.hi, offset))

    def contains(self, cell: Coord) -> bool:
        return (
            self.lo[0] <= cell[0] <= self.hi[0]
            and self.lo[1] <= cell[1] <= self.hi[1]
            and self.lo[2] <= cell[2] <= self.hi[2]
        )

    @property
    def dims(self) -> tuple[int, int, int]:
        return (
            self.hi[0] - self.lo[0] + 1,
            self.hi[1] - self.lo[1] + 1,
            self.hi[2] - self.lo[2] + 1,
        )

    @property
    def volume(self) -> int:
        dx, dy, dz = self.dims
        return dx * dy * dz

    def to_dict(self) -> dict[str, Any]:
        return {
            "min": coord_list(self.lo),
            "max": coord_list(self.hi),
            "dims": list(self.dims),
            "volume": self.volume,
        }


__all__ = [
    "CLOCKWISE",
    "IDENTITY",
    "ORIENTATIONS",
    "Bounds",
    "Coord",
    "Direction",
    "Orientation",
    "above",
    "add",
    "below",
    "coord_list",
    "manhattan",
    "sub",
]
