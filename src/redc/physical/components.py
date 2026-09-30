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

This module defines the data model *and* each cell's functional ``behavior``: the
base :class:`Component`; the datapath families :class:`Operation`,
:class:`PrimitiveGate`, :class:`Wiring`, :class:`TypeCast` and :class:`Register`;
and the :class:`Boundary` cells (:class:`InputPad`, :class:`OutputPad`,
:class:`Constant`, :class:`ClockSource`) that join the fabric to the outside
world.  A :class:`Clock` describes the single domain every register shares.
Concrete cells are subclasses that pin down every field, named by convention, e.g.
``uint8_add_a-0-1-1_b-0-1-0_out-0-0-0`` -- datatype, op, then each pin's offset.
There are enough such variants (differing pin offsets, datatypes) for the router
to choose a layout that fits its situation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from ..ir import BOOL, IRType, _calculate
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

    @property
    def width(self) -> int:
        """The widest datapath any pin carries (control pins are one bit)."""
        return max((port.dtype.width for port in self.ports), default=0)

    @property
    def is_stateful(self) -> bool:
        """True for cells that hold state across clock ticks (registers)."""
        return False

    @property
    def is_source(self) -> bool:
        """True for cells that originate a value with no fabric input."""
        return False

    @property
    def is_sink(self) -> bool:
        """True for cells that consume a value and drive nothing onward."""
        return False

    def input_port(self, name: str) -> Port | None:
        return next((p for p in self.inputs if p.name == name), None)

    def output_port(self, name: str) -> Port | None:
        return next((p for p in self.outputs if p.name == name), None)

    def index_keys(self) -> frozenset[tuple[str, int]]:
        """``(op-or-kind, width)`` keys the cell
        :class:`~redc.physical.cells.Library` files this cell under, so tech-map
        can resolve an IR node to candidate variants.  Empty for cells not
        chosen by operation (wiring, boundary)."""
        return frozenset()

    def behavior(self, inputs: Mapping[str, int]) -> dict[str, int]:
        """Pure functional model: given a value on each input pin (raw unsigned
        bits), return the value on each output pin.

        This is what lets a *placed* netlist be simulated with no Minecraft --
        propagate values cell-by-cell and check the outputs match the IR's own
        evaluation, which is how placement/routing get verified before any
        redstone exists.  Combinational cells implement this; stateful cells
        (:class:`Register`) use :meth:`~Register.read`/:meth:`~Register.step`;
        externally-driven sources (input pads, the clock) are fed by the
        simulator harness and never reach here."""
        raise NotImplementedError(f"{type(self).__name__} has no pure behavior")

    def _args(self, inputs: Mapping[str, int]) -> list[int]:
        """Input-pin values in declared order, for feeding an op evaluator."""
        try:
            return [inputs[port.name] for port in self.inputs]
        except KeyError as exc:
            raise CompileError(
                f"{self.name}: missing value for input pin {exc.args[0]!r}"
            ) from exc


# --------------------------------------------------------------------------
# Datapath families.  Concrete cells subclass one of these and fix every field
# (see the module docstring's naming convention); each supplies its functional
# model via :meth:`Component.behavior`.
# --------------------------------------------------------------------------

#: Operations whose result is a bool regardless of operand width.  They are
#: indexed by *operand* width -- what the datapath actually carries -- not by
#: their 1-bit output, so tech-map can find a comparator for an 8-bit compare.
COMPARISONS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


@dataclass(frozen=True)
class Operation(Component):
    """A multi-bit datapath operator (add, sub, mux, shift, compare, ...).

    ``behavior`` defers to the IR's own operation semantics, so a placed adder
    computes exactly what the ``add`` node it was mapped from would.  Input pins
    MUST be declared in the operation's canonical argument order (mux = select,
    then the two operands; shifts = value, then amount)."""

    op: str = ""

    def behavior(self, inputs: Mapping[str, int]) -> dict[str, int]:
        out = self.outputs[0]
        value = _calculate(
            self.op, self._args(inputs), out.dtype, [p.dtype for p in self.inputs]
        )
        return {out.name: value}

    def index_keys(self) -> frozenset[tuple[str, int]]:
        width = (
            self.inputs[0].dtype.width
            if self.op in COMPARISONS
            else self.outputs[0].dtype.width
        )
        return frozenset({(self.op, width)})


@dataclass(frozen=True)
class PrimitiveGate(Component):
    """A boolean gate (and, or, not, xor, nand, nor, xnor), applied bitwise."""

    op: str = ""

    def behavior(self, inputs: Mapping[str, int]) -> dict[str, int]:
        out = self.outputs[0]
        args = self._args(inputs)
        op = self.op
        if op == "and":
            value = args[0] & args[1]
        elif op == "or":
            value = args[0] | args[1]
        elif op == "xor":
            value = args[0] ^ args[1]
        elif op == "nand":
            value = ~(args[0] & args[1])
        elif op == "nor":
            value = ~(args[0] | args[1])
        elif op == "xnor":
            value = ~(args[0] ^ args[1])
        elif op in {"not", "inv"}:
            value = ~args[0]
        else:
            raise CompileError(f"{self.name}: unknown gate op {op!r}")
        return {out.name: value & out.dtype.mask}

    def index_keys(self) -> frozenset[tuple[str, int]]:
        return frozenset({(self.op, self.outputs[0].dtype.width)})


@dataclass(frozen=True)
class Wiring(Component):
    """A signal-carrying passthrough: a buffer/repeater the router may insert.

    Functionally the identity (output == input); its reason to exist is timing
    and reach -- a nonzero ``latency`` is the matched delay / repeater the router
    inserts to close timing (see the timing model)."""

    def behavior(self, inputs: Mapping[str, int]) -> dict[str, int]:
        out = self.outputs[0]
        return {out.name: self._args(inputs)[0] & out.dtype.mask}


@dataclass(frozen=True)
class TypeCast(Component):
    """Resize/reinterpret a value from one datatype to another: e.g. uint8 ->
    uint4 truncates the high bits, and a widening cast sign- or zero-extends per
    the source type.  Indexed by its ``(source_width, result_width)`` pair."""

    def behavior(self, inputs: Mapping[str, int]) -> dict[str, int]:
        src, dst = self.inputs[0], self.outputs[0]
        value = _calculate("cast", [inputs[src.name]], dst.dtype, [src.dtype])
        return {dst.name: value}

    @property
    def source_width(self) -> int:
        return self.inputs[0].dtype.width

    @property
    def result_width(self) -> int:
        return self.outputs[0].dtype.width


@dataclass(frozen=True)
class Register(Component):
    """A state element: the physical realization of an IR ``register`` node.

    It exposes four pins -- ``next`` (the value to latch), ``enable`` (latch only
    when high), ``clk`` (the global clock net; the ONLY place the clock attaches,
    per the timing model) and ``out`` (the value currently held).  Being
    *stateful* it has no pure ``behavior``; the simulator uses :meth:`initial`,
    :meth:`read` and :meth:`step`, mirroring how the IR evaluates registers over
    ticks (:meth:`redc.ir.Graph.run`).  ``init`` is the reset value."""

    init: int = 0

    NEXT = "next"
    ENABLE = "enable"
    CLOCK = "clk"
    OUT = "out"

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (self.NEXT, self.ENABLE, self.CLOCK):
            if self.input_port(name) is None:
                raise CompileError(f"{self.name}: register missing input pin {name!r}")
        if self.output_port(self.OUT) is None:
            raise CompileError(f"{self.name}: register missing output pin {self.OUT!r}")
        for name in (self.ENABLE, self.CLOCK):
            if self.input_port(name).dtype.width != 1:
                raise CompileError(f"{self.name}: register {name!r} pin must be one bit")

    @property
    def is_stateful(self) -> bool:
        return True

    @property
    def data_width(self) -> int:
        return self.output_port(self.OUT).dtype.width

    def index_keys(self) -> frozenset[tuple[str, int]]:
        return frozenset({("register", self.data_width)})

    def initial(self) -> int:
        """The reset value the register loads (its IR ``init``)."""
        return self.output_port(self.OUT).dtype.bits(self.init)

    def read(self, state: int) -> dict[str, int]:
        """The combinational view: the ``out`` pin exposes the stored value."""
        return {self.OUT: self.output_port(self.OUT).dtype.bits(state)}

    def step(self, state: int, inputs: Mapping[str, int]) -> int:
        """The next stored value on a clock edge: latch ``next`` iff enabled."""
        mask = self.output_port(self.OUT).dtype.mask
        return (inputs[self.NEXT] & mask) if inputs[self.ENABLE] else (state & mask)


# --------------------------------------------------------------------------
# Clock domain + boundary/source cells.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Clock:
    """The single, global clock domain every :class:`Register` shares.

    ``period`` stays ``None`` until static timing analysis measures the placed
    critical path and sets it (the timing model closes timing post-route, in the
    compiler).  A combinational design has no registers and thus no clock."""

    name: str = "clk"
    period: int | None = None
    edge: str = "posedge"
    reset: str = "rst"


@dataclass(frozen=True)
class Boundary(Component):
    """Base for cells at the edge of the design -- the pins where signals enter
    or leave the fabric, hardwired constants, and the clock generator.

    Unlike the datapath families these are *parametric per instance* (a constant
    carries a value, a pad carries a signal name), so the tech-mapper builds them
    on demand through the ``of(...)`` constructors rather than enumerating them in
    a YAML library."""


@dataclass(frozen=True)
class InputPad(Boundary):
    """An IR ``input``: an external signal entering the fabric (this includes the
    handshake ``start``).  A pure source with one output pin; the simulator
    drives its value from the stimulus, so it has no ``behavior``."""

    @classmethod
    def of(
        cls, name: str, dtype: IRType, *, face: Face = Face.EAST, latency: int = 0
    ) -> InputPad:
        return cls(
            name=name,
            latency=latency,
            dim=(1, 1, 1),
            inputs=(),
            outputs=(Port("out", dtype, face, (0, 0, 0), PortDir.OUT),),
        )

    @property
    def is_source(self) -> bool:
        return True


@dataclass(frozen=True)
class OutputPad(Boundary):
    """An IR ``output``: a signal leaving the fabric (this includes the handshake
    ``done``).  A sink with one input pin and nothing downstream."""

    @classmethod
    def of(
        cls, name: str, dtype: IRType, *, face: Face = Face.WEST, latency: int = 0
    ) -> OutputPad:
        return cls(
            name=name,
            latency=latency,
            dim=(1, 1, 1),
            inputs=(Port("in", dtype, face, (0, 0, 0), PortDir.IN),),
            outputs=(),
        )

    @property
    def is_sink(self) -> bool:
        return True

    def behavior(self, inputs: Mapping[str, int]) -> dict[str, int]:
        return {}


@dataclass(frozen=True)
class Constant(Boundary):
    """A hardwired constant: an IR ``const`` realized as a fixed power source.
    A pure source, so its ``behavior`` needs no inputs."""

    value: int = 0

    @classmethod
    def of(cls, value: int, dtype: IRType, *, face: Face = Face.EAST) -> Constant:
        return cls(
            name=f"const_{dtype.width}b_{dtype.bits(value)}",
            latency=0,
            dim=(1, 1, 1),
            inputs=(),
            outputs=(Port("out", dtype, face, (0, 0, 0), PortDir.OUT),),
            value=dtype.bits(value),
        )

    @property
    def is_source(self) -> bool:
        return True

    def behavior(self, inputs: Mapping[str, int]) -> dict[str, int]:
        out = self.outputs[0]
        return {out.name: out.dtype.bits(self.value)}


@dataclass(frozen=True)
class ClockSource(Boundary):
    """The generator that drives the clock net feeding every register's ``clk``
    pin.  One 1-bit output; ``period`` is set post-route (see :class:`Clock`).
    The simulator toggles it, so it has no ``behavior``."""

    period: int | None = None

    @classmethod
    def of(cls, *, period: int | None = None, face: Face = Face.EAST) -> ClockSource:
        return cls(
            name="clk_source",
            latency=0,
            dim=(1, 1, 1),
            inputs=(),
            outputs=(Port("clk", BOOL, face, (0, 0, 0), PortDir.OUT),),
            period=period,
        )

    @property
    def is_source(self) -> bool:
        return True