"""Component library: the placeable cells the IR is mapped onto.

A :class:`Component` is a *definition* (like a standard cell in an ASIC flow),
not a placed instance.  It knows its own shape and timing but nothing about
*where* it sits -- placement supplies an origin later and resolves each port's
local ``offset`` into an absolute grid coordinate.  Per-instance data (a
register's reset value) lives on :class:`~redc.physical.netlist.ComponentInstance`.

Every component carries:

* ``latency`` -- physical *propagation delay* in redstone ticks.  It says
  nothing about state: a NOT gate is combinational with ``latency == 1``, an
  adder is combinational with several ticks of delay, and a zero-tick adder is
  combinational with ``latency == 0``.
* :attr:`~Component.is_stateful` / :attr:`~Component.is_combinational` -- the
  independent "does this cell hold state?" property.  Only :class:`Register` is
  stateful, and only stateful cells attach to the global clock and reset.
* ``dim`` -- the cell's footprint in occupancy-grid cells, ``(dx, dy, dz)``,
  with ``y`` up to match :mod:`redc.physical.grid`.
* ``inputs`` / ``outputs`` -- typed :class:`Port` pins.  Each pin names the
  ``face`` it lives on and its ``offset`` within the footprint, so the router
  knows both where a net must reach and which direction it enters from.  Every
  pin type must be a supported Minecraft physical type
  (:mod:`redc.physical.signals`).  Input declaration order is the canonical IR
  operand order, so a mapper can pair ``zip(node args, operand_inputs)``.

Operation-implementing cells report their exact
:class:`~redc.physical.implementation.OperationSignature` via
:meth:`Component.signatures`; that is how the
:class:`~redc.physical.cells.Library` indexes them (full type identity, never
width alone).

This module defines the data model *and* each cell's functional ``behavior``: the
base :class:`Component`; the datapath families :class:`Operation`,
:class:`PrimitiveGate`, :class:`Wiring`, :class:`TypeCast` and :class:`Register`;
and the :class:`Boundary` cells (:class:`InputPad`, :class:`OutputPad`,
:class:`Constant`, :class:`ClockSource`, :class:`ResetSource`) that join the
fabric to the outside world.  A :class:`Clock` describes the single global
domain every register shares.  Concrete cells are named by convention, e.g.
``uint8_add_a-0-0-0_b-0-0-1_out-0-0-2`` -- datatype, op, then each pin's
offset -- and a signature may have several layout variants for the router to
choose from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from ..ir import BOOL, OPS, IRType, _calculate
from ..parser import CompileError
from .implementation import REGISTER_OP, OperationSignature
from .signals import require_supported_physical_type


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
    """A placeable cell definition with a footprint, timing, and typed pins.

    ``latency`` is propagation delay in ticks and is independent of
    :attr:`is_combinational` (see the module docstring)."""

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
        names = [port.name for port in self.ports]
        if len(set(names)) != len(names):
            raise CompileError(f"{self.name}: duplicate pin names {names}")
        for port in self.ports:
            self._check_port(port)

    def _check_port(self, port: Port) -> None:
        require_supported_physical_type(port.dtype, f"{self.name}: pin {port.name!r}")
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
    def is_stateful(self) -> bool:
        """True for cells that hold state across clock ticks (registers).  Only
        stateful cells need the global clock and reset."""
        return False

    @property
    def is_combinational(self) -> bool:
        """True for cells with no stored state, whatever their ``latency``."""
        return not self.is_stateful

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

    def port(self, name: str) -> Port:
        """The pin called ``name``; raises if the cell has none."""
        port = self.input_port(name) or self.output_port(name)
        if port is None:
            raise CompileError(f"{self.name}: no pin {name!r}")
        return port

    @property
    def operand_inputs(self) -> tuple[Port, ...]:
        """Input pins that carry IR operands, in IR argument order.  For most
        cells that is every input; a :class:`Register` excludes ``clk``/``rst``."""
        return self.inputs

    def signatures(self) -> tuple[OperationSignature, ...]:
        """Exact-typed operations this cell implements directly -- the keys the
        :class:`~redc.physical.cells.Library` files it under.  Empty for cells
        not chosen by operation (wiring, boundary)."""
        return ()

    def _signature(self, op: str) -> OperationSignature:
        if len(self.outputs) != 1:
            raise CompileError(f"{self.name}: an operation cell has exactly one output")
        try:
            return OperationSignature(
                op, self.outputs[0].dtype, tuple(p.dtype for p in self.operand_inputs)
            )
        except CompileError as exc:
            raise CompileError(f"{self.name}: {exc}") from exc

    def behavior(self, inputs: Mapping[str, int]) -> dict[str, int]:
        """Pure functional model: given a value on each input pin (raw unsigned
        bits), return the value on each output pin.

        This is what lets a *placed* netlist be simulated with no Minecraft --
        propagate values cell-by-cell and check the outputs match the IR's own
        evaluation, which is how placement/routing get verified before any
        redstone exists.  Combinational cells implement this; stateful cells
        (:class:`Register`) use :meth:`~Register.read`/:meth:`~Register.step`;
        externally-driven sources (input pads, clock, reset) are fed by the
        simulator harness and never reach here."""
        raise NotImplementedError(f"{type(self).__name__} has no pure behavior")

    def _args(self, inputs: Mapping[str, int]) -> list[int]:
        """Operand-pin values in declared order, for feeding an op evaluator."""
        try:
            return [inputs[port.name] for port in self.operand_inputs]
        except KeyError as exc:
            raise CompileError(
                f"{self.name}: missing value for input pin {exc.args[0]!r}"
            ) from exc


# --------------------------------------------------------------------------
# Datapath families.  Concrete cells fix every field (see the module
# docstring's naming convention); each supplies its functional model via
# :meth:`Component.behavior`, delegating arithmetic to the IR evaluator so there
# is exactly one definition of RedC semantics.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Operation(Component):
    """A multi-bit datapath operator (add, sub, mux, shift, compare, ...).

    ``behavior`` defers to the IR's own operation semantics, so a placed adder
    computes exactly what the ``add`` node it was mapped from would -- including
    signed comparison, division, modulo and arithmetic right shift.  Input pins
    MUST be declared in the operation's canonical argument order (mux = select,
    then the two operands; shifts = value, then the ``uint64`` amount); the
    signature check rejects anything the IR typing rules would."""

    op: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.op not in OPS or self.op == "cast":
            raise CompileError(f"{self.name}: {self.op!r} is not an IR datapath op")
        self._signature(self.op)

    def behavior(self, inputs: Mapping[str, int]) -> dict[str, int]:
        out = self.outputs[0]
        value = _calculate(
            self.op, self._args(inputs), out.dtype, [p.dtype for p in self.inputs]
        )
        return {out.name: value}

    def signatures(self) -> tuple[OperationSignature, ...]:
        return (self._signature(self.op),)


#: Gates computed as the bitwise complement of an IR operation.
_NEGATED_GATES = {"nand": "and", "nor": "or", "xnor": "xor"}


@dataclass(frozen=True)
class PrimitiveGate(Component):
    """A boolean gate (and, or, not, inv, xor, nand, nor, xnor), applied bitwise.

    Gates named after IR operations share the IR's semantics and typing (``not``
    is logical negation, so it only exists on ``bool``); ``nand``/``nor``/
    ``xnor`` are the complements of ``and``/``or``/``xor``."""

    op: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if _NEGATED_GATES.get(self.op, self.op) not in {"and", "or", "xor", "not", "inv"}:
            raise CompileError(f"{self.name}: unknown gate op {self.op!r}")
        self._signature(self.op)

    def behavior(self, inputs: Mapping[str, int]) -> dict[str, int]:
        out = self.outputs[0]
        base = _NEGATED_GATES.get(self.op, self.op)
        value = _calculate(
            base, self._args(inputs), out.dtype, [p.dtype for p in self.inputs]
        )
        if self.op in _NEGATED_GATES:
            value = ~value & out.dtype.mask
        return {out.name: value}

    def signatures(self) -> tuple[OperationSignature, ...]:
        return (self._signature(self.op),)


@dataclass(frozen=True)
class Wiring(Component):
    """A signal-carrying passthrough: a buffer/repeater the router may insert.

    Functionally the identity (output == input) and combinational; its reason to
    exist is timing and reach -- its nonzero ``latency`` is the matched delay /
    repeater the router inserts to close timing (see the timing model)."""

    def behavior(self, inputs: Mapping[str, int]) -> dict[str, int]:
        out = self.outputs[0]
        return {out.name: self._args(inputs)[0] & out.dtype.mask}


@dataclass(frozen=True)
class TypeCast(Component):
    """Convert a value from one datatype to another with exact IR ``cast``
    semantics: widening sign-extends a signed source and zero-extends an
    unsigned one, narrowing truncates to the destination width, ``-> bool`` is
    ``value != 0`` and ``bool ->`` yields 0 or 1.  A cast is not merely a wire
    width change, so it is identified by its FULL source and result types
    (``int8 -> uint16`` and ``uint8 -> uint16`` are different cells)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.inputs) != 1:
            raise CompileError(f"{self.name}: a cast has exactly one input")
        self._signature("cast")

    def behavior(self, inputs: Mapping[str, int]) -> dict[str, int]:
        src, dst = self.inputs[0], self.outputs[0]
        value = _calculate("cast", self._args(inputs), dst.dtype, [src.dtype])
        return {dst.name: value}

    def signatures(self) -> tuple[OperationSignature, ...]:
        return (self._signature("cast"),)

    @property
    def source_type(self) -> IRType:
        return self.inputs[0].dtype

    @property
    def result_type(self) -> IRType:
        return self.outputs[0].dtype


@dataclass(frozen=True)
class Register(Component):
    """A state element: the physical realization of an IR ``register`` node.

    A register *definition* describes how a register works; it does NOT own a
    reset value.  The value loaded on reset belongs to each instance
    (``ComponentInstance.init``), so registers with different initial values
    share one definition.

    Pins, in declaration order:

    * ``next``   -- value to latch (operand 0 of the IR node);
    * ``enable`` -- ``bool``; latch ``next`` on a clock edge only when high
      (operand 1 of the IR node);
    * ``clk``    -- ``bool``; the single global clock net, the ONLY place the
      clock attaches (per the timing model);
    * ``rst``    -- ``bool``; the single global reset net, active high and
      asynchronous like the SystemVerilog backend's ``posedge rst``;
    * ``out``    -- the value currently held.

    Reset has priority over enable: while ``rst`` is high the state is the
    instance's ``init``; otherwise on a clock edge the state becomes ``next`` if
    ``enable`` else holds.  Being stateful it has no pure ``behavior``; the
    simulator uses :meth:`reset_value`, :meth:`read` and :meth:`step`, mirroring
    :meth:`redc.ir.Graph.reset_state` / :meth:`redc.ir.Graph.step`."""

    NEXT = "next"
    ENABLE = "enable"
    CLOCK = "clk"
    RESET = "rst"
    OUT = "out"
    PINS = (NEXT, ENABLE, CLOCK, RESET)

    def __post_init__(self) -> None:
        super().__post_init__()
        if tuple(p.name for p in self.inputs) != self.PINS:
            raise CompileError(
                f"{self.name}: register inputs must be exactly {list(self.PINS)} "
                f"in that order, got {[p.name for p in self.inputs]}"
            )
        if tuple(p.name for p in self.outputs) != (self.OUT,):
            raise CompileError(f"{self.name}: register must have one output {self.OUT!r}")
        for name in (self.ENABLE, self.CLOCK, self.RESET):
            if self.port(name).dtype != BOOL:
                raise CompileError(f"{self.name}: register {name!r} pin must be bool")
        if self.port(self.NEXT).dtype != self.data_type:
            raise CompileError(f"{self.name}: register next/out types differ")
        self._signature(REGISTER_OP)

    @property
    def is_stateful(self) -> bool:
        return True

    @property
    def operand_inputs(self) -> tuple[Port, ...]:
        return (self.port(self.NEXT), self.port(self.ENABLE))

    @property
    def data_type(self) -> IRType:
        return self.port(self.OUT).dtype

    @property
    def data_width(self) -> int:
        return self.data_type.width

    def signatures(self) -> tuple[OperationSignature, ...]:
        return (self._signature(REGISTER_OP),)

    def check_init(self, init: int) -> int:
        """Validate an instance reset value against the data type and return its
        raw bits.  Accepts a raw bit pattern (``0 .. 2**w - 1``) or, for signed
        types, a negative two's-complement number (``-2**(w-1) .. -1``)."""
        typ = self.data_type
        low = -(1 << (typ.width - 1)) if typ.signed else 0
        if isinstance(init, bool) or not isinstance(init, int) or not low <= init <= typ.mask:
            raise CompileError(
                f"{self.name}: register init {init!r} does not fit {typ.name}"
            )
        return typ.bits(init)

    def reset_value(self, init: int) -> int:
        """The state held while ``rst`` is asserted: the instance's ``init``."""
        return self.check_init(init)

    def read(self, state: int) -> dict[str, int]:
        """The combinational view: the ``out`` pin exposes the stored value."""
        return {self.OUT: self.data_type.bits(state)}

    def step(self, state: int, inputs: Mapping[str, int], *, init: int) -> int:
        """The next stored value: ``init`` under reset (reset wins over enable),
        else on a clock edge latch ``next`` iff ``enable``, else hold."""
        if inputs.get(self.RESET, 0):
            return self.reset_value(init)
        mask = self.data_type.mask
        return (inputs[self.NEXT] & mask) if inputs[self.ENABLE] else (state & mask)


# --------------------------------------------------------------------------
# Clock domain + boundary/source cells.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Clock:
    """The single, global clock domain every :class:`Register` shares.

    v1 invariant: one sequential PhysicalNetlist == one clock domain == one
    :class:`ClockSource` whose single net reaches every register ``clk`` pin
    (and one :class:`ResetSource` whose single net reaches every ``rst`` pin).
    There are no per-register, per-FSM or per-region clocks: independent
    Minecraft clock generators drift out of phase (chunks load at different
    times), so all state shares one generator and skew is a later
    clock-routing/timing concern.

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
    or leave the fabric, hardwired constants, and the global clock and reset
    sources.

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
            name=f"const_{dtype.name}_{dtype.bits(value)}",
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
    """The one global clock generator; its single net feeds every register's
    ``clk`` pin.  One 1-bit output; ``period`` is set post-route (see
    :class:`Clock`).  The simulator toggles it, so it has no ``behavior``."""

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


@dataclass(frozen=True)
class ResetSource(Boundary):
    """The one global reset driver: a 1-bit, active-high source whose single net
    feeds every register's ``rst`` pin (the physical twin of the SystemVerilog
    ``rst`` port).  The simulator drives it, so it has no ``behavior``."""

    @classmethod
    def of(cls, *, face: Face = Face.EAST) -> ResetSource:
        return cls(
            name="rst_source",
            latency=0,
            dim=(1, 1, 1),
            inputs=(),
            outputs=(Port("rst", BOOL, face, (0, 0, 0), PortDir.OUT),),
        )

    @property
    def is_source(self) -> bool:
        return True
