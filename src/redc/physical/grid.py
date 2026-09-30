"""Occupancy grid for component placement and net routing.

The grid is the abstract environment that placement and routing operate on.  It
is deliberately agnostic to in-game sizing: coordinates are *cells*, and the
mapping from a cell to a region of Minecraft blocks lives elsewhere.  A cell is
whatever cube the physical backend later decides to fill (nominally 16x16x16
blocks), but the grid neither knows nor cares.

Storage is asymmetric, mirroring a real Minecraft world:

* ``y`` is a *bounded* vertical axis, ``0 <= y < height`` with ``height <= 24``.
  A cube stack 24 tall at 16 blocks per cube is 384 blocks, the world height.
  Writes outside the range are a physical impossibility and raise.
* ``x`` and ``z`` are *unbounded* horizontal axes.  The backing arrays grow
  transparently as writes land outside the current extent, and negative
  coordinates are supported via an origin offset.  Callers only ever see virtual
  coordinates; the capacity and origin are private implementation details.

Per-cell data is kept as a structure-of-arrays so numpy stays vectorizable and
the routing cost fields can be added later without disturbing occupancy:

* ``owner`` -- id of the component or net occupying the cell (``EMPTY`` if free).
* ``kind``  -- what *sort* of thing occupies it (see :class:`CellKind`).

All mutation goes through :meth:`place`, :meth:`reserve` and :meth:`rip_up` so
that every change to the environment is a single call that can later be logged as
an event for the place-and-route replay trace.
"""

from __future__ import annotations

from enum import IntEnum
from collections.abc import Iterator

import numpy as np

from ..parser import CompileError

#: Sentinel stored in ``owner`` for an unoccupied cell.
EMPTY = -1

#: Hard ceiling on the vertical axis: 24 cubes * 16 blocks = 384 = world height.
MAX_HEIGHT = 24

#: Default horizontal caps -- effectively unbounded.  A compiler flag can lower
#: these to constrain how wide/deep a generated circuit is allowed to grow.
DEFAULT_MAX_HORIZONTAL = 1 << 30


class CellKind(IntEnum):
    """What occupies a cell.  Stored as ``uint8`` in the grid."""

    FREE = 0
    COMPONENT = 1  # body of a placed component
    PIN = 2        # a component's connection point, where nets attach
    WIRE = 3       # a routed net segment
    RESERVED = 4   # claimed but not yet filled (keep-out / halo)
    BLOCKED = 5    # permanently unusable (obstacle)


class Grid:
    """A vertically bounded 3D occupancy grid, horizontally unbounded by default.

    ``max_x`` / ``max_z`` optionally cap how wide and deep the *used* region may
    grow (a span, so negative coordinates still work -- the design may sit
    anywhere, it just may not exceed the cap in extent).  Both default humongous,
    so the grid is effectively unbounded unless a caller -- e.g. a compiler flag
    -- lowers them.
    """

    def __init__(
        self,
        height: int = MAX_HEIGHT,
        max_x: int = DEFAULT_MAX_HORIZONTAL,
        max_z: int = DEFAULT_MAX_HORIZONTAL,
    ) -> None:
        if not 1 <= height <= MAX_HEIGHT:
            raise CompileError(f"grid height must be between 1 and {MAX_HEIGHT}")
        if max_x < 1 or max_z < 1:
            raise CompileError("grid max_x and max_z must be at least 1")
        self.height = height
        self.max_x = max_x
        self.max_z = max_z

        # Horizontal origin: virtual (x, z) == physical index (x - ox, z - oz).
        self._ox = 0
        self._oz = 0
        # Current horizontal capacity.  Starts empty; the first write sizes it.
        self._nx = 0
        self._nz = 0
        # Bounding box of cells actually written, used to enforce the span caps.
        self._used_min_x: int | None = None
        self._used_max_x: int | None = None
        self._used_min_z: int | None = None
        self._used_max_z: int | None = None

        self._owner = self._new_owner(0, 0)
        self._kind = self._new_kind(0, 0)

    # -- allocation helpers ------------------------------------------------

    def _new_owner(self, nx: int, nz: int) -> np.ndarray:
        return np.full((nx, self.height, nz), EMPTY, dtype=np.int32)

    def _new_kind(self, nx: int, nz: int) -> np.ndarray:
        return np.full((nx, self.height, nz), CellKind.FREE, dtype=np.uint8)

    def _check_y(self, y: int) -> None:
        if not 0 <= y < self.height:
            raise CompileError(
                f"y={y} is out of bounds; grid height is {self.height} (0..{self.height - 1})"
            )

    def _check_horizontal(self, x: int, z: int) -> None:
        """Reject a write whose cell would push the used region past a cap."""
        lo_x = x if self._used_min_x is None else min(self._used_min_x, x)
        hi_x = x if self._used_max_x is None else max(self._used_max_x, x)
        if hi_x - lo_x + 1 > self.max_x:
            raise CompileError(
                f"x={x} would make the grid {hi_x - lo_x + 1} cells wide, "
                f"exceeding max_x={self.max_x}"
            )
        lo_z = z if self._used_min_z is None else min(self._used_min_z, z)
        hi_z = z if self._used_max_z is None else max(self._used_max_z, z)
        if hi_z - lo_z + 1 > self.max_z:
            raise CompileError(
                f"z={z} would make the grid {hi_z - lo_z + 1} cells deep, "
                f"exceeding max_z={self.max_z}"
            )

    def _extend_used(self, x: int, z: int) -> None:
        self._used_min_x = x if self._used_min_x is None else min(self._used_min_x, x)
        self._used_max_x = x if self._used_max_x is None else max(self._used_max_x, x)
        self._used_min_z = z if self._used_min_z is None else min(self._used_min_z, z)
        self._used_max_z = z if self._used_max_z is None else max(self._used_max_z, z)

    def _ensure(self, x: int, z: int) -> None:
        """Grow the horizontal extent so virtual ``(x, z)`` is addressable."""
        if self._nx == 0:
            # First ever write establishes the origin on a 1x1 box.
            self._ox, self._oz = x, z
            self._nx = self._nz = 1
            self._owner = self._new_owner(1, 1)
            self._kind = self._new_kind(1, 1)
            return

        px, pz = x - self._ox, z - self._oz
        if 0 <= px < self._nx and 0 <= pz < self._nz:
            return

        # Union of the current box with the new point, then amortise growth by
        # at least doubling the deficient axis so repeated outward writes stay
        # cheap overall.
        lo_x = min(self._ox, x)
        lo_z = min(self._oz, z)
        hi_x = max(self._ox + self._nx - 1, x)
        hi_z = max(self._oz + self._nz - 1, z)

        need_x = hi_x - lo_x + 1
        need_z = hi_z - lo_z + 1
        new_nx = max(need_x, 2 * self._nx)
        new_nz = max(need_z, 2 * self._nz)

        # Anchor the new box so the whole current box still fits inside it.
        new_ox = min(lo_x, hi_x - new_nx + 1)
        new_oz = min(lo_z, hi_z - new_nz + 1)

        owner = self._new_owner(new_nx, new_nz)
        kind = self._new_kind(new_nx, new_nz)

        # Copy the old contents into their new location.
        dx = self._ox - new_ox
        dz = self._oz - new_oz
        owner[dx : dx + self._nx, :, dz : dz + self._nz] = self._owner
        kind[dx : dx + self._nx, :, dz : dz + self._nz] = self._kind

        self._owner, self._kind = owner, kind
        self._ox, self._oz = new_ox, new_oz
        self._nx, self._nz = new_nx, new_nz

    def _index(self, x: int, y: int, z: int) -> tuple[int, int, int] | None:
        """Virtual -> physical index, or ``None`` if never allocated (so empty)."""
        self._check_y(y)
        px, pz = x - self._ox, z - self._oz
        if 0 <= px < self._nx and 0 <= pz < self._nz:
            return px, y, pz
        return None

    # -- reads -------------------------------------------------------------

    def owner_at(self, x: int, y: int, z: int) -> int:
        idx = self._index(x, y, z)
        return EMPTY if idx is None else int(self._owner[idx])

    def kind_at(self, x: int, y: int, z: int) -> CellKind:
        idx = self._index(x, y, z)
        return CellKind.FREE if idx is None else CellKind(int(self._kind[idx]))

    def is_free(self, x: int, y: int, z: int) -> bool:
        return self.owner_at(x, y, z) == EMPTY

    # -- mutations ---------------------------------------------------------

    def _write(self, x: int, y: int, z: int, owner: int, kind: CellKind) -> None:
        self._check_y(y)
        self._check_horizontal(x, z)
        self._ensure(x, z)
        px, pz = x - self._ox, z - self._oz
        self._owner[px, y, pz] = owner
        self._kind[px, y, pz] = kind
        self._extend_used(x, z)

    def place(
        self, x: int, y: int, z: int, owner: int, kind: CellKind = CellKind.COMPONENT
    ) -> None:
        """Occupy a cell for ``owner``.  Raises if the cell is already taken."""
        if not self.is_free(x, y, z):
            existing = self.owner_at(x, y, z)
            raise CompileError(f"cell ({x}, {y}, {z}) already occupied by owner {existing}")
        self._write(x, y, z, owner, kind)

    def reserve(self, x: int, y: int, z: int, owner: int) -> None:
        """Claim a cell as keep-out without filling it."""
        self.place(x, y, z, owner, CellKind.RESERVED)

    def rip_up(self, x: int, y: int, z: int) -> None:
        """Free a cell.  The core primitive negotiated-congestion routing needs."""
        self._write(x, y, z, EMPTY, CellKind.FREE)

    # -- introspection / trace --------------------------------------------

    def occupied_cells(self) -> Iterator[tuple[int, int, int, int, CellKind]]:
        """Yield ``(x, y, z, owner, kind)`` for every non-free cell."""
        if self._nx == 0:
            return
        xs, ys, zs = np.nonzero(self._owner != EMPTY)
        for px, y, pz in zip(xs.tolist(), ys.tolist(), zs.tolist()):
            yield (
                px + self._ox,
                y,
                pz + self._oz,
                int(self._owner[px, y, pz]),
                CellKind(int(self._kind[px, y, pz])),
            )

    def to_dict(self) -> dict:
        """Serialisable snapshot for the place-and-route replay trace."""
        return {
            "height": self.height,
            "cells": [
                {"x": x, "y": y, "z": z, "owner": owner, "kind": int(kind)}
                for x, y, z, owner, kind in self.occupied_cells()
            ],
        }
