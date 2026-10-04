"""Sparse block-resolution occupancy for primitive placement and routing.

One coordinate is one Minecraft block.  Bit blasting makes designs huge, so
nothing here is a dense world-sized array: every layer is a dict keyed by the
blocks actually used, and memory grows with the design, not its bounding box.

*Placement* is strict (one owner per block, checked before writing):

* ``body``    -- block -> instance whose structure occupies it;
* ``keepout`` -- block -> first instance keeping routes out of it (keep-outs of
  different instances may overlap; bodies may never sit in a keep-out);
* ``pins``    -- endpoint block -> :class:`PinSite` (owning instance, pin, net).

*Routing* is negotiated (PathFinder): nets may temporarily share or crowd
blocks, and rising cost drives them apart.  Per block it records which nets
use it as a SIGNAL (dust/repeater), a SUPPORT (solid block under a signal) or a
CLEARANCE (must stay air for a staircase), each with a per-net reference
count, plus an electrical-adjacency index (which nets have a signal in the
block's 12-block signal neighbourhood) and accumulated ``history``.

:meth:`BlockGrid.conflicts` lists every violation of the redstone model
(:mod:`redc.physical_primitive.redstone`) between different nets; a legal
routing has none.  Intra-net legality is enforced by the router itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..parser import CompileError
from .geometry import Bounds, Coord, Direction, coord_list
from .redstone import SIGNAL_NEIGHBORHOOD

HISTORY_INCREMENT = 1.0

#: Bits of :attr:`BlockGrid.static`: what placement put in a block.
STATIC_BODY = 1
STATIC_KEEPOUT = 2
STATIC_PIN = 4


@dataclass(frozen=True, slots=True)
class PinSite:
    """A placed pin endpoint block, reserved for its net (``None`` = unconnected)."""

    cell: Coord
    instance: int
    pin: str
    direction: str
    facing: Direction
    strength: int
    net: int | None

    def ref(self) -> dict[str, Any]:
        return {"instance": self.instance, "pin": self.pin}


@dataclass(frozen=True, slots=True)
class Conflict:
    """One cross-net violation.  ``kind`` is ``shared_signal``,
    ``shared_support``, ``signal_on_support``, ``clearance_blocked`` or
    ``adjacent_signals``."""

    kind: str
    cells: tuple[Coord, ...]
    nets: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "cells": [coord_list(c) for c in self.cells], "nets": list(self.nets)}


@dataclass
class NetClaims:
    """Everything one net currently claims (so rip-up is O(route size))."""

    signals: dict[Coord, int] = field(default_factory=dict)
    supports: dict[Coord, int] = field(default_factory=dict)
    clearances: dict[Coord, int] = field(default_factory=dict)


def _inc(table: dict[Coord, dict[int, int]], cell: Coord, net: int) -> None:
    users = table.get(cell)
    if users is None:
        table[cell] = {net: 1}
    else:
        users[net] = users.get(net, 0) + 1


def _dec(table: dict[Coord, dict[int, int]], cell: Coord, net: int, count: int = 1) -> None:
    users = table[cell]
    left = users[net] - count
    if left > 0:
        users[net] = left
    else:
        del users[net]
        if not users:
            del table[cell]


class BlockGrid:
    """Sparse block grid (see module docstring).  ``0 <= y <= max_y``."""

    def __init__(self, *, max_y: int) -> None:
        if max_y < 2:
            raise CompileError("the primitive grid needs max_y >= 2")
        self.max_y = max_y
        self.body: dict[Coord, int] = {}
        self.keepout: dict[Coord, int] = {}
        self.pins: dict[Coord, PinSite] = {}
        self.approaches: dict[Coord, PinSite] = {}
        self.signal: dict[Coord, dict[int, int]] = {}
        self.support: dict[Coord, dict[int, int]] = {}
        self.clearance: dict[Coord, dict[int, int]] = {}
        self.adjacent: dict[Coord, dict[int, int]] = {}
        self.history: dict[Coord, float] = {}
        self.claims: dict[int, NetClaims] = {}
        #: Per block: STATIC_* bits (one lookup instead of three in the router).
        self.static: dict[Coord, int] = {}
        #: Per block: total routing claims of every kind and net (0 = untouched).
        self.used: dict[Coord, int] = {}
        self._bounds: Bounds | None = None

    # -- placement -----------------------------------------------------------

    def in_height(self, cell: Coord) -> bool:
        return 0 <= cell[1] <= self.max_y

    @property
    def placement_bounds(self) -> Bounds | None:
        """Bounding box of every body, keep-out and pin block placed so far."""
        return self._bounds

    def check_placement(
        self, occupied: Iterable[Coord], keepout: Iterable[Coord], pins: Iterable[tuple[Coord, Coord]]
    ) -> str | None:
        """Why a cell with these absolute blocks cannot be placed (``None`` = ok).

        ``pins`` are ``(endpoint, approach)`` pairs.  Bodies may not touch
        bodies, keep-outs, pin endpoints or approach blocks; keep-outs may not
        cover bodies, pins or approaches; a new pin may not be in another pin's
        12-block signal neighbourhood (their dust would short) nor have its
        approach blocked."""
        occupied = list(occupied)
        keepout = list(keepout)
        pins = list(pins)
        for cell in occupied:
            if not self.in_height(cell):
                return f"block {coord_list(cell)} is outside the height limit 0..{self.max_y}"
            if cell in self.body:
                return f"overlaps instance {self.body[cell]} at {coord_list(cell)}"
            if cell in self.keepout:
                return f"enters the keep-out of instance {self.keepout[cell]} at {coord_list(cell)}"
            if cell in self.pins or cell in self.approaches:
                site = self.pins.get(cell) or self.approaches[cell]
                return f"covers pin {site.pin!r} of instance {site.instance} at {coord_list(cell)}"
        for cell in keepout:
            if cell in self.body:
                return f"keep-out covers instance {self.body[cell]} at {coord_list(cell)}"
            if cell in self.pins or cell in self.approaches:
                site = self.pins.get(cell) or self.approaches[cell]
                return f"keep-out covers pin {site.pin!r} of instance {site.instance} at {coord_list(cell)}"
        for endpoint, approach in pins:
            if not self.in_height(endpoint) or not self.in_height(approach):
                return f"pin at {coord_list(endpoint)} is outside the height limit"
            for cell in (endpoint, approach):
                if cell in self.body or cell in self.keepout:
                    return f"pin block {coord_list(cell)} is occupied or kept out"
            if endpoint in self.pins:
                return f"pin {coord_list(endpoint)} collides with another pin"
            if endpoint in self.approaches:
                return f"pin {coord_list(endpoint)} sits on another pin's approach"
            if approach in self.pins or approach in self.approaches:
                return f"pin approach {coord_list(approach)} is another pin's endpoint or approach"
            # The endpoint dust and the approach dust both carry this pin's net:
            # neither may touch another pin's endpoint or approach block.
            for cell in (endpoint, approach):
                x, y, z = cell
                for dx, dy, dz in SIGNAL_NEIGHBORHOOD:
                    nb = (x + dx, y + dy, z + dz)
                    other = self.pins.get(nb) or self.approaches.get(nb)
                    if other is not None:
                        return (
                            f"pin block {coord_list(cell)} would touch pin {other.pin!r} of instance "
                            f"{other.instance} near {coord_list(other.cell)}"
                        )
        return None

    def place(
        self,
        instance: int,
        occupied: Iterable[Coord],
        keepout: Iterable[Coord],
        pins: Iterable[PinSite],
    ) -> None:
        """Write a placed cell.  The caller must have run :meth:`check_placement`."""
        touched: list[Coord] = []
        for cell in occupied:
            if cell in self.body:
                raise CompileError(f"block {coord_list(cell)} is already occupied")
            self.body[cell] = instance
            self.static[cell] = self.static.get(cell, 0) | STATIC_BODY
            touched.append(cell)
        for cell in keepout:
            self.keepout.setdefault(cell, instance)
            self.static[cell] = self.static.get(cell, 0) | STATIC_KEEPOUT
            touched.append(cell)
        for site in pins:
            self.pins[site.cell] = site
            self.static[site.cell] = self.static.get(site.cell, 0) | STATIC_PIN
            approach = (
                site.cell[0] + site.facing.vector[0],
                site.cell[1],
                site.cell[2] + site.facing.vector[2],
            )
            self.approaches.setdefault(approach, site)
            touched.append(site.cell)
        box = Bounds.of(touched)
        if box is not None:
            self._bounds = box.union(self._bounds)

    def is_blocked(self, cell: Coord) -> bool:
        """A component body or keep-out (no route signal/support may go here)."""
        return cell in self.body or cell in self.keepout

    # -- routing claims --------------------------------------------------------

    def _claims(self, net: int) -> NetClaims:
        claims = self.claims.get(net)
        if claims is None:
            claims = self.claims[net] = NetClaims()
        return claims

    def claim_signal(self, net: int, cell: Coord) -> None:
        claims = self._claims(net)
        claims.signals[cell] = claims.signals.get(cell, 0) + 1
        _inc(self.signal, cell, net)
        self.used[cell] = self.used.get(cell, 0) + 1
        if claims.signals[cell] == 1:
            x, y, z = cell
            for dx, dy, dz in SIGNAL_NEIGHBORHOOD:
                _inc(self.adjacent, (x + dx, y + dy, z + dz), net)

    def claim_support(self, net: int, cell: Coord) -> None:
        claims = self._claims(net)
        claims.supports[cell] = claims.supports.get(cell, 0) + 1
        _inc(self.support, cell, net)
        self.used[cell] = self.used.get(cell, 0) + 1

    def claim_clearance(self, net: int, cell: Coord) -> None:
        claims = self._claims(net)
        claims.clearances[cell] = claims.clearances.get(cell, 0) + 1
        _inc(self.clearance, cell, net)
        self.used[cell] = self.used.get(cell, 0) + 1

    def release(self, net: int, *, keep: Iterable[Coord] = ()) -> None:
        """Rip up every claim of ``net`` except signal claims on ``keep``
        (its pin endpoints, which stay reserved)."""
        claims = self.claims.pop(net, None)
        if claims is None:
            return
        kept = NetClaims()
        keep_set = set(keep)
        for cell, count in claims.signals.items():
            if cell in keep_set:
                kept.signals[cell] = 1
                if count > 1:
                    _dec(self.signal, cell, net, count - 1)
                    self._unuse(cell, count - 1)
                continue
            _dec(self.signal, cell, net, count)
            self._unuse(cell, count)
            x, y, z = cell
            for dx, dy, dz in SIGNAL_NEIGHBORHOOD:
                _dec(self.adjacent, (x + dx, y + dy, z + dz), net)
        for cell, count in claims.supports.items():
            _dec(self.support, cell, net, count)
            self._unuse(cell, count)
        for cell, count in claims.clearances.items():
            _dec(self.clearance, cell, net, count)
            self._unuse(cell, count)
        if kept.signals:
            self.claims[net] = kept

    def _unuse(self, cell: Coord, count: int) -> None:
        left = self.used[cell] - count
        if left > 0:
            self.used[cell] = left
        else:
            del self.used[cell]

    def net_signals(self, net: int) -> dict[Coord, int]:
        claims = self.claims.get(net)
        return {} if claims is None else claims.signals

    # -- congestion ------------------------------------------------------------

    def conflicts(self) -> list[Conflict]:
        """Every cross-net violation, deterministically ordered."""
        found: list[Conflict] = []
        signal, support, clearance = self.signal, self.support, self.clearance
        for cell in sorted(signal):
            nets = signal[cell]
            if len(nets) > 1:
                found.append(Conflict("shared_signal", (cell,), tuple(sorted(nets))))
            # A block is dust/repeater OR solid OR air -- never two of them.
            other = support.get(cell)
            if other:
                found.append(Conflict("signal_on_support", (cell,), tuple(sorted(set(other) | set(nets)))))
            blocked = clearance.get(cell)
            if blocked:
                found.append(Conflict("clearance_blocked", (cell,), tuple(sorted(set(blocked) | set(nets)))))
            x, y, z = cell
            for dx, dy, dz in SIGNAL_NEIGHBORHOOD:
                nb = (x + dx, y + dy, z + dz)
                if nb <= cell:
                    continue  # each unordered pair once
                others = signal.get(nb)
                if not others:
                    continue
                mixed = set(nets) | set(others)
                if len(mixed) > 1:
                    found.append(Conflict("adjacent_signals", (cell, nb), tuple(sorted(mixed))))
        for cell in sorted(support):
            nets = support[cell]
            if len(nets) > 1:
                found.append(Conflict("shared_support", (cell,), tuple(sorted(nets))))
            blocked = clearance.get(cell)
            if blocked:
                found.append(Conflict("clearance_blocked", (cell,), tuple(sorted(set(blocked) | set(nets)))))
        return found

    def add_history(self, cells: Iterable[Coord], increment: float = HISTORY_INCREMENT) -> int:
        """Accumulate history on each distinct block; returns how many."""
        distinct = set(cells)
        for cell in distinct:
            self.history[cell] = self.history.get(cell, 0.0) + increment
        return len(distinct)

    def congestion_stats(self) -> dict[str, float | int]:
        conflicts = self.conflicts()
        hot = {c for conflict in conflicts for c in conflict.cells}
        return {
            "conflicts": len(conflicts),
            "conflict_cells": len(hot),
            "history_cells": len(self.history),
            "history_total": float(sum(self.history.values())),
            "history_max": float(max(self.history.values(), default=0.0)),
        }

    def occupied_blocks(self) -> Iterator[tuple[Coord, str, int]]:
        """``(block, layer, owner)`` for every body, support and signal claim."""
        for cell, inst in self.body.items():
            yield cell, "body", inst
        for cell, nets in self.signal.items():
            for net in nets:
                yield cell, "signal", net
        for cell, nets in self.support.items():
            for net in nets:
                yield cell, "support", net


__all__ = [
    "HISTORY_INCREMENT",
    "STATIC_BODY",
    "STATIC_KEEPOUT",
    "STATIC_PIN",
    "BlockGrid",
    "Conflict",
    "NetClaims",
    "PinSite",
]
