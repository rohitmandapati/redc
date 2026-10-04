"""Grid-cell geometry shared by placement, routing and the replay trace.

Coordinates are abstract grid cells ``(x, y, z)`` -- x east, y up, z south,
matching :class:`~redc.physical.components.Face` -- never Minecraft blocks.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from ..components import Component, Port

Coord = tuple[int, int, int]


def coord_list(cell: Coord) -> list[int]:
    """A coordinate as the trace's plain ``[x, y, z]`` array."""
    return [int(cell[0]), int(cell[1]), int(cell[2])]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


@dataclass(frozen=True)
class Bounds:
    """An inclusive axis-aligned box of cells: ``lo[i] <= c[i] <= hi[i]``."""

    lo: Coord
    hi: Coord

    @classmethod
    def of(cls, cells: Iterable[Coord]) -> Bounds | None:
        """The tightest box around ``cells`` (``None`` if there are none)."""
        it = iter(cells)
        first = next(it, None)
        if first is None:
            return None
        lo, hi = list(first), list(first)
        for cell in it:
            for i in range(3):
                if cell[i] < lo[i]:
                    lo[i] = cell[i]
                elif cell[i] > hi[i]:
                    hi[i] = cell[i]
        return cls((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))

    @classmethod
    def box(cls, origin: Coord, dim: tuple[int, int, int]) -> Bounds:
        """The cells of a ``dim``-sized prism whose low corner is ``origin``."""
        ox, oy, oz = origin
        dx, dy, dz = dim
        return cls(origin, (ox + dx - 1, oy + dy - 1, oz + dz - 1))

    def union(self, other: Bounds | None) -> Bounds:
        if other is None:
            return self
        return Bounds(
            (min(self.lo[0], other.lo[0]), min(self.lo[1], other.lo[1]), min(self.lo[2], other.lo[2])),
            (max(self.hi[0], other.hi[0]), max(self.hi[1], other.hi[1]), max(self.hi[2], other.hi[2])),
        )

    def expand_horizontal(self, margin: int, height: int) -> Bounds:
        """Grow by ``margin`` in x and z, and span the full grid height in y."""
        return Bounds(
            (self.lo[0] - margin, 0, self.lo[2] - margin),
            (self.hi[0] + margin, height - 1, self.hi[2] + margin),
        )

    def contains(self, cell: Coord) -> bool:
        return (
            self.lo[0] <= cell[0] <= self.hi[0]
            and self.lo[1] <= cell[1] <= self.hi[1]
            and self.lo[2] <= cell[2] <= self.hi[2]
        )

    def intersects(self, other: Bounds) -> bool:
        return all(self.lo[i] <= other.hi[i] and other.lo[i] <= self.hi[i] for i in range(3))

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


def footprint(component: Component, origin: Coord) -> Iterator[Coord]:
    """Every cell the component body occupies at ``origin`` (its exact dim)."""
    yield from component.footprint_cells(origin)


def pin_cell(port: Port, origin: Coord) -> Coord:
    return port.absolute(origin)


def escape_cell(port: Port, origin: Coord) -> Coord:
    """The routing cell just outside ``port``'s face -- where its wire attaches
    (equal to :attr:`~redc.physical.netlist.Terminal.outward` once placed)."""
    px, py, pz = port.absolute(origin)
    nx, ny, nz = port.face.normal
    return (px + nx, py + ny, pz + nz)


def pin_cells(component: Component, origin: Coord) -> set[Coord]:
    """Absolute cells holding at least one pin (several ports may share one)."""
    return {pin_cell(port, origin) for port in component.ports}
