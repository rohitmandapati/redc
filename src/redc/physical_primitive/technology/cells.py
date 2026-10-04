"""Primitive Minecraft technology cells: what ONE one-bit primitive looks like
in blocks.

A :class:`PrimitiveCell` is a definition (like a standard cell), described in
LOCAL block coordinates with its signal flowing along local +x:

* ``voxels`` -- the SPARSE set of blocks the structure occupies, each with a
  display role (``"base"``, ``"body"``, ...).  Never assumed to be a solid box.
* ``keepout`` -- blocks where no route signal or support block may go, so
  routing can never weakly power or touch the cell's internals.  A route's
  vertical-transition clearance (air) may still pass through.
* ``pins`` -- :class:`PrimitivePin` endpoint blocks.  The route's endpoint dust
  sits ON the pin block (resting on the cell's own base voxel below it) and
  enters or leaves along the pin's ``facing``.  Output pins state the signal
  ``strength`` they deliver there; input pins the minimum they need.
* ``orientations`` -- the quarter turns placement may choose from.
* ``latency`` -- propagation delay in redstone ticks (``None`` = unknown).
  Latency is NOT state: only REGISTER_BIT cells are stateful.
* ``structure`` -- a future NBT/structure reference; ``None`` today.
* ``placeholder`` -- True for every cell until a real in-game circuit with
  verified electrical behaviour replaces it.

:meth:`PrimitiveCell.oriented` rotates a cell once (cached); placement then
only translates.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ...parser import CompileError
from ..geometry import (
    ORIENTATIONS,
    Bounds,
    Coord,
    Direction,
    Orientation,
    add,
    below,
    coord_list,
)
from ..netlist import PIN_INTERFACE, PeripheralDirection, PrimitiveKind
from ..redstone import MAX_SIGNAL_STRENGTH, MIN_SIGNAL_Y, SIGNAL_NEIGHBORHOOD


@dataclass(frozen=True, slots=True)
class PrimitivePin:
    """One one-bit pin endpoint (local coordinates).

    ``direction`` is ``"in"`` or ``"out"``.  ``position`` is the block where the
    route's endpoint dust sits; ``facing`` points away from the cell, along the
    one horizontal direction the route may enter (input) or leave (output).
    ``strength``: for outputs, the signal strength the cell drives into the
    endpoint dust (0 = never powered, e.g. a constant-0 anchor); for inputs, the
    minimum strength the cell needs there."""

    name: str
    direction: str
    position: Coord
    facing: Direction
    strength: int = MAX_SIGNAL_STRENGTH

    @property
    def approach(self) -> Coord:
        """The level block just outside the pin, along its facing."""
        return add(self.position, self.facing.vector)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "position": coord_list(self.position),
            "facing": self.facing.label,
            "strength": self.strength,
        }


@dataclass(frozen=True, slots=True)
class OrientedPin:
    """A pin after rotation: position relative to the placed origin."""

    name: str
    direction: str
    position: Coord
    facing: Direction
    strength: int

    @property
    def approach(self) -> Coord:
        return add(self.position, self.facing.vector)


@dataclass(frozen=True)
class OrientedCell:
    """A cell rotated by ``orientation`` (still relative to the origin)."""

    cell: PrimitiveCell
    orientation: Orientation
    voxels: tuple[tuple[Coord, str], ...]
    occupied: frozenset[Coord]
    keepout: frozenset[Coord]
    pins: dict[str, OrientedPin]
    bounds: Bounds

    def translate(self, origin: Coord) -> PlacedCell:
        ox, oy, oz = origin
        return PlacedCell(
            oriented=self,
            origin=origin,
            occupied=frozenset((x + ox, y + oy, z + oz) for x, y, z in self.occupied),
            keepout=frozenset((x + ox, y + oy, z + oz) for x, y, z in self.keepout),
            pins={
                name: OrientedPin(p.name, p.direction, add(p.position, origin), p.facing, p.strength)
                for name, p in self.pins.items()
            },
            bounds=self.bounds.translate(origin),
        )


@dataclass(frozen=True)
class PlacedCell:
    """A cell at an absolute origin: every block in world coordinates."""

    oriented: OrientedCell
    origin: Coord
    occupied: frozenset[Coord]
    keepout: frozenset[Coord]
    pins: dict[str, OrientedPin]
    bounds: Bounds

    @property
    def orientation(self) -> Orientation:
        return self.oriented.orientation

    def voxels(self) -> list[tuple[Coord, str]]:
        return [(add(c, self.origin), role) for c, role in self.oriented.voxels]


@dataclass(frozen=True)
class PrimitiveCell:
    """A technology definition realizing one primitive kind (see module docstring)."""

    name: str
    kind: PrimitiveKind
    voxels: tuple[tuple[Coord, str], ...]
    keepout: frozenset[Coord]
    pins: tuple[PrimitivePin, ...]
    latency: int | None
    orientations: tuple[Orientation, ...] = ORIENTATIONS
    structure: str | None = None
    placeholder: bool = True
    #: ``(kind, direction, width)`` of a PERIPHERAL cell, else ``None``.
    peripheral: tuple[str, PeripheralDirection, int] | None = None
    description: str = ""
    _oriented: dict[int, OrientedCell] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        where = f"technology cell {self.name!r}"
        if not self.name:
            raise CompileError("a technology cell needs a name")
        if not isinstance(self.kind, PrimitiveKind):
            raise CompileError(f"{where}: kind must be a PrimitiveKind")
        if self.latency is not None and self.latency < 0:
            raise CompileError(f"{where}: latency must be non-negative")
        occupied = [c for c, _ in self.voxels]
        if len(set(occupied)) != len(occupied):
            raise CompileError(f"{where}: duplicate voxels")
        occ = frozenset(occupied)
        if not occ:
            raise CompileError(f"{where}: a cell occupies at least one voxel")
        if any(c[1] < 0 for c in occ):
            raise CompileError(f"{where}: voxels must have local y >= 0")
        if occ & self.keepout:
            raise CompileError(f"{where}: keep-out overlaps occupied voxels {sorted(occ & self.keepout)[:4]}")
        if not self.orientations:
            raise CompileError(f"{where}: needs at least one allowed orientation")
        names = [p.name for p in self.pins]
        if len(set(names)) != len(names):
            raise CompileError(f"{where}: duplicate pin names {names}")
        for pin in self.pins:
            self._check_pin(pin, occ, where)
        # Every pin's endpoint AND approach carry that pin's net: no block of
        # one pin may equal or touch a block of another (the grid applies the
        # same rule between cells).
        for i, first in enumerate(self.pins):
            for second in self.pins[i + 1 :]:
                for a in (first.position, first.approach):
                    for b in (second.position, second.approach):
                        d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
                        if a == b or d in SIGNAL_NEIGHBORHOOD:
                            raise CompileError(
                                f"{where}: pin blocks {list(a)} and {list(b)} could short each other "
                                f"(pins {first.name!r} and {second.name!r})"
                            )
        self._check_interface(where)

    def _check_pin(self, pin: PrimitivePin, occ: frozenset[Coord], where: str) -> None:
        if pin.direction not in ("in", "out"):
            raise CompileError(f"{where}: pin {pin.name!r} direction must be 'in' or 'out'")
        if pin.position[1] < MIN_SIGNAL_Y:
            raise CompileError(f"{where}: pin {pin.name!r} must be at local y >= {MIN_SIGNAL_Y}")
        if pin.position in occ or pin.position in self.keepout:
            raise CompileError(f"{where}: pin {pin.name!r} block is occupied or kept out")
        if below(pin.position) not in occ:
            raise CompileError(f"{where}: pin {pin.name!r} must rest on one of the cell's own voxels")
        if pin.approach in occ or pin.approach in self.keepout:
            raise CompileError(f"{where}: pin {pin.name!r} approach block {list(pin.approach)} is blocked")
        low = 0 if pin.direction == "out" else 1
        if not low <= pin.strength <= MAX_SIGNAL_STRENGTH:
            raise CompileError(f"{where}: pin {pin.name!r} strength {pin.strength} out of range")

    def _check_interface(self, where: str) -> None:
        ins = tuple(p.name for p in self.pins if p.direction == "in")
        outs = tuple(p.name for p in self.pins if p.direction == "out")
        if self.kind is PrimitiveKind.PERIPHERAL:
            if self.peripheral is None:
                raise CompileError(f"{where}: a peripheral cell needs (kind, direction, width)")
            kind, direction, width = self.peripheral
            if not kind or not isinstance(direction, PeripheralDirection):
                raise CompileError(f"{where}: peripheral spec needs a kind and a PeripheralDirection")
            if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 64:
                raise CompileError(f"{where}: peripheral width must be 1..64, got {width!r}")
            pins = tuple(f"b{i}" for i in range(width))
            want = (pins, ()) if direction is PeripheralDirection.OUTPUT else ((), pins)
            if (ins, outs) != want:
                raise CompileError(f"{where}: peripheral pins must be {want}, got {(ins, outs)}")
            return
        if self.peripheral is not None:
            raise CompileError(f"{where}: only PERIPHERAL cells carry a peripheral spec")
        if (ins, outs) != PIN_INTERFACE[self.kind]:
            raise CompileError(
                f"{where}: {self.kind.value} pins must be {PIN_INTERFACE[self.kind]}, got {(ins, outs)}"
            )

    # -- geometry --------------------------------------------------------------

    @property
    def occupied(self) -> frozenset[Coord]:
        return frozenset(c for c, _ in self.voxels)

    def pin(self, name: str) -> PrimitivePin:
        for pin in self.pins:
            if pin.name == name:
                return pin
        raise CompileError(f"technology cell {self.name!r} has no pin {name!r}")

    def oriented(self, orientation: Orientation) -> OrientedCell:
        """This cell rotated by ``orientation`` (computed once, then cached)."""
        cached = self._oriented.get(orientation.quarter_turns)
        if cached is not None:
            return cached
        if orientation not in self.orientations:
            raise CompileError(f"technology cell {self.name!r} does not allow orientation {orientation.name}")
        rot = orientation.apply
        voxels = tuple((rot(c), role) for c, role in self.voxels)
        occupied = frozenset(c for c, _ in voxels)
        keepout = frozenset(rot(c) for c in self.keepout)
        pins = {
            p.name: OrientedPin(p.name, p.direction, rot(p.position), orientation.direction(p.facing), p.strength)
            for p in self.pins
        }
        bounds = Bounds.of([*occupied, *keepout, *(p.position for p in pins.values())])
        assert bounds is not None
        oriented = OrientedCell(self, orientation, voxels, occupied, keepout, pins, bounds)
        self._oriented[orientation.quarter_turns] = oriented
        return oriented

    def to_dict(self) -> dict[str, Any]:
        """Self-contained JSON definition (a consumer needs no Python/YAML)."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "placeholder": self.placeholder,
            "structure": self.structure,
            "latency": self.latency,
            "stateful": self.kind is PrimitiveKind.REGISTER_BIT,
            "description": self.description,
            "orientations": [o.name for o in self.orientations],
            "peripheral": (
                None
                if self.peripheral is None
                else {"kind": self.peripheral[0], "direction": self.peripheral[1].value, "width": self.peripheral[2]}
            ),
            "voxels": [{"coord": coord_list(c), "role": role} for c, role in self.voxels],
            "keepout": [coord_list(c) for c in sorted(self.keepout)],
            "pins": [p.to_dict() for p in self.pins],
        }


def placeholder_cell(
    name: str,
    kind: PrimitiveKind,
    *,
    body: Iterable[Coord],
    pins: Iterable[PrimitivePin],
    latency: int | None,
    description: str,
    body_role: str = "body",
    peripheral: tuple[str, PeripheralDirection, int] | None = None,
) -> PrimitiveCell:
    """Build a deterministic PLACEHOLDER cell from a body and pins.

    Adds a ``base`` voxel under every body column and every pin (pins rest on
    the cell), and derives the keep-out: every horizontal neighbour of a body
    voxel at its own level, the layer directly above the body, and both sides
    of every pin -- excluding pin blocks and their approach blocks."""
    body_set = set(body)
    pin_list = tuple(pins)
    if any(c[1] < MIN_SIGNAL_Y for c in body_set):
        raise CompileError(f"{name}: placeholder body voxels sit at y >= {MIN_SIGNAL_Y}")
    columns = {(x, z) for x, _y, z in body_set} | {(p.position[0], p.position[2]) for p in pin_list}
    base = {(x, 0, z) for x, z in columns}
    occupied = base | body_set
    reserved = {p.position for p in pin_list} | {p.approach for p in pin_list}
    keepout: set[Coord] = set()
    for x, y, z in body_set:
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            keepout.add((x + dx, y, z + dz))
        keepout.add((x, y + 1, z))
    for pin in pin_list:
        fx, _, fz = pin.facing.vector
        for side in ((fz, 0, fx), (-fz, 0, -fx)):  # the two perpendicular neighbours
            keepout.add(add(pin.position, side))
    keepout -= occupied
    keepout -= reserved
    voxels = tuple(sorted(((c, "base") for c in base), key=lambda v: v[0])) + tuple(
        sorted(((c, body_role) for c in body_set), key=lambda v: v[0])
    )
    return PrimitiveCell(
        name=name,
        kind=kind,
        voxels=voxels,
        keepout=frozenset(keepout),
        pins=pin_list,
        latency=latency,
        description=description,
        peripheral=peripheral,
    )


__all__ = [
    "OrientedCell",
    "OrientedPin",
    "PlacedCell",
    "PrimitiveCell",
    "PrimitivePin",
    "placeholder_cell",
]
