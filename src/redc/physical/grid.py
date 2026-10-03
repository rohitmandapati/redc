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

Per-cell data is kept as a structure-of-arrays so numpy stays vectorizable.  The
grid serves two regimes with different rules:

*Placement* is strict -- one owner per cell, :meth:`place` raises on conflict:

* ``owner`` -- id of the component (or committed net) in the cell (``EMPTY`` if
  free).
* ``kind``  -- what *sort* of thing occupies it (see :class:`CellKind`).

*Routing* is negotiated -- nets may temporarily *share* cells and are driven
apart by rising cost.  It rides two more arrays plus a per-net index:

* ``occupancy`` -- how many nets currently route through the cell (present
  congestion); a legal routing has at most :data:`CAPACITY` per cell.
* ``history``   -- congestion accumulated across iterations; the permanent
  memory that makes negotiated-congestion converge instead of oscillate.

A bus of any width is one net occupying one path of cells (width lives on the
net, not the grid) -- so these arrays stay scalar and width-agnostic.

Placement mutation goes through :meth:`place`/:meth:`reserve`/:meth:`rip_up`;
routing through :meth:`claim`/:meth:`rip_up_net`.  Every change is a single call
that can be logged as an event for the place-and-route replay trace.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from enum import IntEnum

import numpy as np

from ..parser import CompileError

#: Sentinel stored in ``owner`` for an unoccupied cell.
EMPTY = -1

#: Hard ceiling on the vertical axis: 24 cubes * 16 blocks = 384 = world height.
MAX_HEIGHT = 24

#: Default horizontal caps -- effectively unbounded.  A compiler flag can lower
#: these to constrain how wide/deep a generated circuit is allowed to grow.
DEFAULT_MAX_HORIZONTAL = 1 << 30

#: Negotiated-congestion routing knobs.  A routable cell costs ``BASE_COST``; a
#: cell shared by more than ``CAPACITY`` nets is *overused* and penalised.
#: ``PRESENT_FACTOR`` (raised by the router across iterations) scales the
#: present-overuse penalty; ``HISTORY_INCREMENT`` scales how fast the permanent
#: history term accrues.
BASE_COST = 1.0
CAPACITY = 1
PRESENT_FACTOR = 0.5
HISTORY_INCREMENT = 1.0

#: Raw ``CellKind`` values a wire may pass through (see :meth:`Grid.is_routable`).
_ROUTABLE = (0, 3)  # FREE, WIRE

#: The six axis-aligned moves a wire may take (redstone has no diagonals).
_MOVES = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


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
        # Routing layer: present occupancy + accumulated history, and the per-net
        # cell index that makes whole-net rip-up O(net length).
        self._history = self._new_history(0, 0)
        self._occupancy = self._new_occupancy(0, 0)
        self._net_cells: dict[int, set[tuple[int, int, int]]] = {}

    # -- allocation helpers ------------------------------------------------

    def _new_owner(self, nx: int, nz: int) -> np.ndarray:
        return np.full((nx, self.height, nz), EMPTY, dtype=np.int32)

    def _new_kind(self, nx: int, nz: int) -> np.ndarray:
        return np.full((nx, self.height, nz), CellKind.FREE, dtype=np.uint8)

    def _new_history(self, nx: int, nz: int) -> np.ndarray:
        return np.zeros((nx, self.height, nz), dtype=np.float32)

    def _new_occupancy(self, nx: int, nz: int) -> np.ndarray:
        return np.zeros((nx, self.height, nz), dtype=np.int32)

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

    def fits_span(self, x: int, z: int) -> bool:
        """Whether using column ``(x, z)`` keeps the used region within the
        ``max_x`` / ``max_z`` span caps.  Never raises."""
        return self._horizontal_ok(x, z)

    def _horizontal_ok(self, x: int, z: int) -> bool:
        """Whether using ``(x, z)`` stays within the span caps (predicate form of
        :meth:`_check_horizontal`, for neighbour enumeration -- never raises)."""
        lo_x = x if self._used_min_x is None else min(self._used_min_x, x)
        hi_x = x if self._used_max_x is None else max(self._used_max_x, x)
        if hi_x - lo_x + 1 > self.max_x:
            return False
        lo_z = z if self._used_min_z is None else min(self._used_min_z, z)
        hi_z = z if self._used_max_z is None else max(self._used_max_z, z)
        return hi_z - lo_z + 1 <= self.max_z

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
            self._history = self._new_history(1, 1)
            self._occupancy = self._new_occupancy(1, 1)
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
        # Grow only a deficient axis.  Doubling an axis that already fits would
        # make repeated outward writes along the other axis blow up
        # exponentially in memory.
        new_nx = self._nx if 0 <= px < self._nx else max(need_x, 2 * self._nx)
        new_nz = self._nz if 0 <= pz < self._nz else max(need_z, 2 * self._nz)

        # Anchor the new box so the whole current box still fits inside it and
        # the spare room lies on the side that grew: growing toward +x keeps the
        # low edge, growing toward -x keeps the high edge.  (Centring the spare
        # room the other way would re-trigger growth on every further step.)
        new_ox = hi_x - new_nx + 1 if px < 0 else self._ox
        new_oz = hi_z - new_nz + 1 if pz < 0 else self._oz

        owner = self._new_owner(new_nx, new_nz)
        kind = self._new_kind(new_nx, new_nz)
        history = self._new_history(new_nx, new_nz)
        occupancy = self._new_occupancy(new_nx, new_nz)

        # Copy the old contents into their new location.
        dx = self._ox - new_ox
        dz = self._oz - new_oz
        window = (slice(dx, dx + self._nx), slice(None), slice(dz, dz + self._nz))
        owner[window] = self._owner
        kind[window] = self._kind
        history[window] = self._history
        occupancy[window] = self._occupancy

        self._owner, self._kind = owner, kind
        self._history, self._occupancy = history, occupancy
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

    # -- routing: moves, cost, negotiated congestion -----------------------

    def is_routable(self, x: int, y: int, z: int) -> bool:
        """Whether a net may pass through this cell.  Component bodies, pins and
        permanent obstacles block routing; free space and existing wire segments
        are fair game -- wires *may* be shared, and resolving that overuse is
        exactly what negotiated-congestion routing does."""
        if not 0 <= y < self.height:
            return False
        px, pz = x - self._ox, z - self._oz
        if 0 <= px < self._nx and 0 <= pz < self._nz:
            return int(self._kind[px, y, pz]) in _ROUTABLE
        return True  # never-allocated space is free

    def neighbors(self, x: int, y: int, z: int) -> Iterator[tuple[int, int, int]]:
        """The routable six-connected cells adjacent to ``(x, y, z)`` -- the legal
        one-step wire moves out of it.  A neighbour is yielded only if it is in
        bounds, within the span caps, and :meth:`is_routable`."""
        for dx, dy, dz in _MOVES:
            nx, ny, nz = x + dx, y + dy, z + dz
            if self._horizontal_ok(nx, nz) and self.is_routable(nx, ny, nz):
                yield (nx, ny, nz)

    def routing_cost(
        self,
        x: int,
        y: int,
        z: int,
        *,
        present_factor: float = PRESENT_FACTOR,
        net: int | None = None,
    ) -> float:
        """Cost of routing a net through this cell.

        Combines a base cost, the accumulated *history* congestion, and the
        *present* overuse (nets sharing the cell beyond :data:`CAPACITY`).
        Returns ``inf`` for cells a net may not use.  ``present_factor`` is
        raised by the router across iterations so temporary sharing is squeezed
        out.

        Without ``net`` the overuse is the cell's current state.  With ``net``
        it is the overuse that WOULD result if that net used the cell (PathFinder
        semantics): a cell held by one other net is already priced as shared,
        while a cell ``net`` itself already holds (another branch of its own
        tree) is not charged again."""
        if not self.is_routable(x, y, z):
            return math.inf
        idx = self._index(x, y, z)
        if idx is None:
            return BASE_COST  # pristine free space: base cost only
        occupancy = int(self._occupancy[idx])
        if net is not None and (x, y, z) not in self._net_cells.get(net, ()):
            occupancy += 1
        overuse = max(0, occupancy - CAPACITY)
        return (BASE_COST + float(self._history[idx])) * (1.0 + present_factor * overuse)

    def occupancy_at(self, x: int, y: int, z: int) -> int:
        """How many nets currently route through the cell."""
        idx = self._index(x, y, z)
        return 0 if idx is None else int(self._occupancy[idx])

    def history_at(self, x: int, y: int, z: int) -> float:
        """The accumulated historical congestion on the cell."""
        idx = self._index(x, y, z)
        return 0.0 if idx is None else float(self._history[idx])

    def route_of(self, net: int) -> frozenset[tuple[int, int, int]]:
        """The set of cells ``net`` currently occupies."""
        return frozenset(self._net_cells.get(net, ()))

    def claim(self, net: int, x: int, y: int, z: int) -> None:
        """Route ``net`` through a cell (negotiated-congestion working state).

        Unlike :meth:`place`, this *allows overuse*: several nets may claim the
        same cell at once, and the rising congestion cost is what later drives
        them apart.  The claim is recorded per net so :meth:`rip_up_net` can undo
        the whole route in one call.

        Claims are idempotent per net: the branches of one fanout tree that share
        a trunk occupy it as ONE net, so re-claiming a cell the net already holds
        does not raise its occupancy again."""
        self._check_y(y)
        self._check_horizontal(x, z)
        if not self.is_routable(x, y, z):
            raise CompileError(
                f"cannot route net {net} through blocked cell ({x}, {y}, {z})"
            )
        cells = self._net_cells.setdefault(net, set())
        if (x, y, z) in cells:
            return
        self._ensure(x, z)
        px, pz = x - self._ox, z - self._oz
        self._occupancy[px, y, pz] += 1
        cells.add((x, y, z))
        self._extend_used(x, z)

    def rip_up_net(self, net: int) -> None:
        """Free every cell ``net`` routes through -- the whole-net counterpart to
        :meth:`rip_up`, run each iteration before a net is rerouted against the
        updated congestion costs.  Each cell is released exactly once (claims are
        per-net sets).  Only working routes should be ripped up: cells already
        made permanent by :meth:`commit_routes` keep their ``WIRE`` marking."""
        for x, y, z in self._net_cells.pop(net, set()):
            idx = self._index(x, y, z)
            if idx is not None:
                self._occupancy[idx] = max(0, int(self._occupancy[idx]) - 1)

    def overused(self) -> list[tuple[int, int, int]]:
        """Every cell claimed by more nets than :data:`CAPACITY` -- the congestion
        that must reach zero before a routing is legal."""
        if self._nx == 0:
            return []
        xs, ys, zs = np.nonzero(self._occupancy > CAPACITY)
        return [
            (int(px) + self._ox, int(y), int(pz) + self._oz)
            for px, y, pz in zip(xs.tolist(), ys.tolist(), zs.tolist())
        ]

    def add_history(self, *, increment: float = HISTORY_INCREMENT) -> None:
        """Accumulate historical congestion on every overused cell -- called once
        per router iteration.  This permanent memory is what makes negotiated-
        congestion converge instead of oscillating between the same two routes."""
        if self._nx == 0:
            return
        overuse = np.maximum(0, self._occupancy - CAPACITY).astype(np.float32)
        self._history += increment * overuse

    def max_occupancy(self) -> int:
        """The most nets currently sharing any one cell (0 if none)."""
        if self._nx == 0:
            return 0
        return int(self._occupancy.max(initial=0))

    def congestion_stats(self) -> dict[str, float | int]:
        """Grid-wide congestion summary: overused cell count, peak occupancy, and
        how much history has accumulated (and over how many cells)."""
        if self._nx == 0:
            return {
                "overused": 0,
                "max_occupancy": 0,
                "history_cells": 0,
                "history_total": 0.0,
                "history_max": 0.0,
            }
        return {
            "overused": int(np.count_nonzero(self._occupancy > CAPACITY)),
            "max_occupancy": int(self._occupancy.max(initial=0)),
            "history_cells": int(np.count_nonzero(self._history > 0)),
            "history_total": float(self._history.sum(dtype=np.float64)),
            "history_max": float(self._history.max(initial=0.0)),
        }

    def commit_routes(self) -> int:
        """Finalize the working routing: mark every claimed cell as permanent
        ``WIRE`` owned by its net, so :meth:`to_dict` shows the routed layout.

        Only a legal (capacity-respecting) routing may be committed; raises if
        any cell is overused or a claimed cell belongs to something else.  The
        per-net claim index is kept, so :meth:`route_of` still answers
        afterwards.  Returns the number of cells committed."""
        overused = self.overused()
        if overused:
            raise CompileError(f"cannot commit routes: {len(overused)} cells are overused")
        committed = 0
        for net in sorted(self._net_cells):
            for x, y, z in sorted(self._net_cells[net]):
                owner = self.owner_at(x, y, z)
                own_wire = owner == net and self.kind_at(x, y, z) == CellKind.WIRE
                if owner != EMPTY and not own_wire:
                    raise CompileError(
                        f"cannot commit net {net}: cell ({x}, {y}, {z}) is owned by {owner}"
                    )
                self._write(x, y, z, net, CellKind.WIRE)
                committed += 1
        return committed

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
