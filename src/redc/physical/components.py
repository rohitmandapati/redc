"""Component library: the placeable cells the IR is mapped onto.

A :class:`Component` is a *definition* (like a standard cell in an ASIC flow),
not a placed instance.  It knows its own shape and timing but nothing about
*where* it sits -- placement supplies an origin later and resolves each port's
local ``offset`` into an absolute grid coordinate.

Every component carries:

* ``latency`` -- combinational delay in *ticks*.  ``0`` means purely
  combinational; ``> 0`` means the cell holds a result across ticks.  This is the
  physical-layer reflection of RedC's combinational-first principle: state is
  explicit, never inferred.
* ``dim`` -- the cell's footprint in occupancy-grid cells, ``(dx, dy, dz)``,
  with ``y`` up to match :mod:`redc.physical.grid`.
* ``inputs`` / ``outputs`` -- typed :class:`Port` pins.  Each pin names the
  ``face`` it lives on and its ``offset`` within the footprint, so the router
  knows both where a net must reach and which direction it enters from.

This module defines only the abstract data model: the base :class:`Component` and
the three families :class:`Operation`, :class:`PrimitiveGate` and :class:`Wiring`.
Concrete cells are subclasses that pin down every field, named by convention, e.g.
``uint8_add_a-0-1-1_b-0-1-0_out-0-0-0`` -- datatype, op, then each pin's offset.
There are enough such variants (differing pin offsets, datatypes) for the router
to choose a layout that fits its situation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..ir import IRType
from ..parser import CompileError


class Face(Enum):
    """One of the six faces of a component box, valued by its outward normal.

    The normal is a unit vector in grid space ``(x, y, z)`` with ``y`` up, so a
    pin's outward direction -- the way its wire leaves the cell -- is just
    ``face.normal``.
    """

    EAST = (1, 0, 0)
    WEST = (-1, 0, 0)
    TOP = (0, 1, 0)
    BOTTOM = (0, -1, 0)
    SOUTH = (0, 0, 1)
    NORTH = (0, 0, -1)

    @property
    def normal(self) -> tuple[int, int, int]:
        return self.value

    @property
    def axis(self) -> int:
        """Index of the axis this face is perpendicular to (0=x, 1=y, 2=z)."""
        return next(i for i, c in enumerate(self.value) if c != 0)


class PortDir(Enum):
    IN = "in"
    OUT = "out"


@dataclass(frozen=True, slots=True)
class Port:
    """A typed connection point on one face of a component.

    ``offset`` is the pin's cell position *local* to the component footprint
    (``0 <= offset[i] < dim[i]``); the pin also lies on ``face``, meaning it sits
    against that side of the box.
    """

    name: str
    dtype: IRType
    face: Face
    offset: tuple[int, int, int]
    direction: PortDir

    def absolute(self, origin: tuple[int, int, int]) -> tuple[int, int, int]:
        """Resolve this pin to an absolute grid cell given the component origin."""
        ox, oy, oz = origin
        lx, ly, lz = self.offset
        return (ox + lx, oy + ly, oz + lz)


@dataclass(frozen=True)
class Component:
    """A placeable cell definition with a footprint, timing, and typed pins."""

    name: str
    latency: int
    dim: tuple[int, int, int]
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    #: Filename of the NBT structure realising this cell, resolved at emission
    #: time.  ``None`` for cells that carry no physical implementation yet.
    nbt: str | None = None

    def __post_init__(self) -> None:
        if self.latency < 0:
            raise CompileError(f"{self.name}: latency must be non-negative")
        if any(d <= 0 for d in self.dim):
            raise CompileError(f"{self.name}: dimensions must be positive, got {self.dim}")
        for port in self.ports:
            self._check_port(port)

    def _check_port(self, port: Port) -> None:
        # The pin must sit inside the footprint...
        for o, d in zip(port.offset, self.dim):
            if not 0 <= o < d:
                raise CompileError(
                    f"{self.name}: pin {port.name!r} offset {port.offset} outside footprint {self.dim}"
                )
        # ...and actually touch the face it claims to live on.
        axis = port.face.axis
        on_high = port.face.normal[axis] > 0
        want = self.dim[axis] - 1 if on_high else 0
        if port.offset[axis] != want:
            raise CompileError(
                f"{self.name}: pin {port.name!r} on {port.face.name} must have "
                f"offset[{axis}] == {want}, got {port.offset[axis]}"
            )

    @property
    def ports(self) -> tuple[Port, ...]:
        return self.inputs + self.outputs

    @property
    def is_combinational(self) -> bool:
        return self.latency == 0

    @property
    def volume(self) -> int:
        dx, dy, dz = self.dim
        return dx * dy * dz

    def footprint_cells(self, origin: tuple[int, int, int]):
        """Yield every absolute grid cell the body occupies at ``origin``."""
        ox, oy, oz = origin
        dx, dy, dz = self.dim
        for x in range(ox, ox + dx):
            for y in range(oy, oy + dy):
                for z in range(oz, oz + dz):
                    yield (x, y, z)


# --------------------------------------------------------------------------
# Component families.  These are still abstract: concrete cells subclass one of
# them and fix every field (see the module docstring's naming convention).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Operation(Component):
    """A multi-bit datapath operator (add, sub, mux, shift, compare, ...)."""

    op: str = ""


@dataclass(frozen=True)
class PrimitiveGate(Component):
    """A boolean gate (and, or, not, xor, nand, nor, xnor)."""

    op: str = ""


@dataclass(frozen=True)
class Wiring(Component):
    """A signal-carrying passthrough: a buffer/repeater the router may insert."""

@dataclass(frozen=True)
class TypeCast(Component):
    """Takes a data type and casts it to another size, e.g. 32-bit -> 8-hex
        or uint8 -> uint4 (truncate and drop msb)"""
    
@dataclass(frozen=True)
class Register(Component):
    """Stores a specified amount of bits to keep state and has a 'done' signal"""