"""Physical netlist: component instances wired together by nets.

The netlist is the circuit as *connectivity + geometry*, sitting between tech-map
(which produces it from the live IR) and place-and-route (which consumes it).  It
says *what connects to what* and, once placed, *where each instance sits* -- but
not how wires physically thread between them (routing's job).

* :class:`ComponentInstance` -- one placed-or-placeable use of a
  :class:`~redc.physical.components.Component` definition: the definition, a
  unique id, an ``origin`` (``None`` until placement assigns one), and any
  per-instance state such as a register's reset value ``init``.
* :class:`Terminal` -- a specific pin on a specific instance; the thing a net
  attaches to.  Resolves to an absolute grid cell once its instance is placed.
* :class:`Net` -- one logical signal: a single driver terminal fanning out to one
  or more sink terminals, all of exactly the same :class:`~redc.ir.IRType`.  A bus
  is *one* net even when it is 64 bits wide and will be routed as 16 hex lanes
  (:attr:`Net.layout`); abutment is a routing outcome, so the net is identical
  whether or not it ends up needing wire cells.
* :class:`PhysicalNetlist` -- the whole graph: instances + nets, a single id
  allocator (so instance ids and net ids never collide in the grid's ``owner``
  field), and :meth:`~PhysicalNetlist.validate` for the electrical rules.

Electrical rules (checked eagerly by :meth:`PhysicalNetlist.connect` and again,
from scratch, by :meth:`PhysicalNetlist.validate`):

* every terminal's instance is the very object stored in this netlist (identity,
  not just a matching id);
* an input pin is the sink of at most one net;
* an output pin drives at most one net -- fanout is ONE net with many sinks;
* driver and sink types match exactly (``int8`` never silently feeds ``uint8``;
  an explicit :class:`~redc.physical.components.TypeCast` must sit between);
* one global clock domain: at most one :class:`ClockSource` and one
  :class:`ResetSource`; a register's ``clk``/``rst`` may only be driven by them,
  and their nets may only reach register ``clk``/``rst`` pins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ir import IRType
from ..parser import CompileError
from .components import ClockSource, Component, Port, PortDir, Register, ResetSource
from .signals import PhysicalSignalLayout, signal_layout


@dataclass(eq=False)
class ComponentInstance:
    """One use of a component definition, with an optional placement origin.

    Identity is per-object (two instances are never "equal" even if they wrap the
    same definition), which is why ``eq=False``.  ``origin`` is mutable: placement
    assigns and may later adjust it.

    ``init`` is per-instance state: the reset value of a :class:`Register`
    instance (validated against its data type and stored as raw bits; defaults to
    0, like an IR register).  It must be ``None`` for every other component.
    """

    id: int
    component: Component
    origin: tuple[int, int, int] | None = None
    init: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.component, Register):
            self.init = self.component.check_init(0 if self.init is None else self.init)
        elif self.init is not None:
            raise CompileError(
                f"instance {self.id} ({self.component.name}): only registers carry "
                "an init value"
            )

    @property
    def is_placed(self) -> bool:
        return self.origin is not None

    def terminal(self, port_name: str) -> Terminal:
        """The :class:`Terminal` for pin ``port_name`` on this instance."""
        port = self.component.input_port(port_name) or self.component.output_port(
            port_name
        )
        if port is None:
            raise CompileError(
                f"instance {self.id} ({self.component.name}): no pin {port_name!r}"
            )
        return Terminal(self, port)

    def footprint_cells(self):
        """Absolute cells this instance's body occupies (requires placement)."""
        if self.origin is None:
            raise CompileError(f"instance {self.id} is not placed")
        return self.component.footprint_cells(self.origin)


@dataclass(frozen=True)
class Terminal:
    """A specific pin (:class:`~redc.physical.components.Port`) on a specific
    :class:`ComponentInstance` -- what a :class:`Net` connects to.  Equality and
    hashing follow the instance's object identity plus the port."""

    instance: ComponentInstance
    port: Port

    @property
    def dtype(self) -> IRType:
        return self.port.dtype

    @property
    def width(self) -> int:
        return self.port.dtype.width

    @property
    def cell(self) -> tuple[int, int, int]:
        """The absolute grid cell of this pin (requires the instance be placed)."""
        if self.instance.origin is None:
            raise CompileError(f"instance {self.instance.id} is not placed")
        return self.port.absolute(self.instance.origin)

    @property
    def outward(self) -> tuple[int, int, int]:
        """The cell just outside the pin's face -- where its wire first steps, and
        the cell a facing sink pin would occupy for a zero-length (abutted) net."""
        cx, cy, cz = self.cell
        nx, ny, nz = self.port.face.normal
        return (cx + nx, cy + ny, cz + nz)

    def describe(self) -> str:
        return f"pin {self.port.name!r} on instance {self.instance.id}"


@dataclass(frozen=True)
class Net:
    """One logical signal: a single driver fanning out to one or more sinks.

    Every sink must have exactly the driver's type (width AND signedness).  The
    net is grid-agnostic: a 32-bit bus is one net, not 32 (nor 8 nibble nets).
    """

    id: int
    driver: Terminal
    sinks: tuple[Terminal, ...]

    def __post_init__(self) -> None:
        if self.driver.port.direction is not PortDir.OUT:
            raise CompileError(
                f"net {self.id}: driver pin {self.driver.port.name!r} is not an output"
            )
        if not self.sinks:
            raise CompileError(f"net {self.id}: has no sinks")
        if len(set(self.sinks)) != len(self.sinks):
            raise CompileError(f"net {self.id}: lists the same sink pin twice")
        for sink in self.sinks:
            if sink.port.direction is not PortDir.IN:
                raise CompileError(
                    f"net {self.id}: sink pin {sink.port.name!r} is not an input"
                )
            if sink.dtype != self.driver.dtype:
                raise CompileError(
                    f"net {self.id}: type mismatch -- driver {self.driver.dtype.name} "
                    f"vs sink {sink.dtype.name} ({sink.describe()}); insert an "
                    "explicit cast"
                )

    @property
    def dtype(self) -> IRType:
        return self.driver.dtype

    @property
    def width(self) -> int:
        return self.driver.width

    @property
    def layout(self) -> PhysicalSignalLayout:
        """How this one logical net is to be realized physically (lanes)."""
        return signal_layout(self.dtype)

    @property
    def fanout(self) -> int:
        return len(self.sinks)


@dataclass
class PhysicalNetlist:
    """The circuit graph: component instances plus the nets wiring them."""

    instances: dict[int, ComponentInstance] = field(default_factory=dict)
    nets: dict[int, Net] = field(default_factory=dict)
    _next_id: int = 0
    # Incremental connectivity indexes for eager checks in connect().
    _driving: dict[Terminal, int] = field(default_factory=dict, repr=False)
    _driven: dict[Terminal, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for net in self.nets.values():
            self._driving[net.driver] = net.id
            for sink in net.sinks:
                self._driven[sink] = net.id

    def _fresh_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def add(
        self,
        component: Component,
        origin: tuple[int, int, int] | None = None,
        *,
        init: int | None = None,
    ) -> ComponentInstance:
        """Instantiate ``component`` into the netlist and return the instance.
        ``init`` is the reset value of a register instance."""
        inst = ComponentInstance(self._fresh_id(), component, origin, init)
        self.instances[inst.id] = inst
        return inst

    def connect(self, driver: Terminal, *sinks: Terminal) -> Net:
        """Wire ``driver`` to one or more ``sinks`` as a new net.

        Raises if the driver pin already drives a net (extend that net's fanout
        instead), a sink is already driven, or a terminal belongs to another
        netlist."""
        net = Net(self._fresh_id(), driver, tuple(sinks))
        self._check_net(net, self._driving, self._driven)
        self._driving[net.driver] = net.id
        for sink in net.sinks:
            self._driven[sink] = net.id
        self.nets[net.id] = net
        return net

    def owns(self, instance: ComponentInstance) -> bool:
        """Whether ``instance`` is the very object this netlist stores."""
        return self.instances.get(instance.id) is instance

    def _check_net(
        self, net: Net, driving: dict[Terminal, int], driven: dict[Terminal, int]
    ) -> None:
        for term in (net.driver, *net.sinks):
            if not self.owns(term.instance):
                raise CompileError(
                    f"net {net.id} references instance {term.instance.id} "
                    f"({term.instance.component.name}) that is not part of this netlist"
                )
        if net.driver in driving:
            raise CompileError(
                f"output {net.driver.describe()} already drives net "
                f"{driving[net.driver]}; net {net.id} would be a second net from the "
                "same driver (fanout must be one net with many sinks)"
            )
        for sink in net.sinks:
            if sink in driven:
                raise CompileError(
                    f"input {sink.describe()} is driven by nets {driven[sink]} "
                    f"and {net.id}"
                )
        _check_global_control(net)

    @property
    def is_sequential(self) -> bool:
        return any(inst.component.is_stateful for inst in self.instances.values())

    def validate(self, *, complete: bool = False) -> None:
        """Enforce the electrical rules (see the module docstring) from scratch.

        With ``complete=True`` the netlist must also be fully wired, as tech-map
        output will be: every input pin driven, and a sequential netlist has
        exactly one :class:`ClockSource` and one :class:`ResetSource` (a
        combinational one has neither), so every register shares one clock and
        one reset net."""
        for key, inst in self.instances.items():
            if key != inst.id:
                raise CompileError(f"instance {inst.id} is stored under id {key}")
        for key, net in self.nets.items():
            if key != net.id:
                raise CompileError(f"net {net.id} is stored under id {key}")
            if key in self.instances:
                raise CompileError(f"id {key} is used by both an instance and a net")

        driving: dict[Terminal, int] = {}
        driven: dict[Terminal, int] = {}
        for net in self.nets.values():
            self._check_net(net, driving, driven)
            driving[net.driver] = net.id
            for sink in net.sinks:
                driven[sink] = net.id

        clocks = [i for i in self.instances.values() if isinstance(i.component, ClockSource)]
        resets = [i for i in self.instances.values() if isinstance(i.component, ResetSource)]
        if len(clocks) > 1:
            raise CompileError(
                f"one global clock domain allows one ClockSource, found {len(clocks)}"
            )
        if len(resets) > 1:
            raise CompileError(
                f"one global reset allows one ResetSource, found {len(resets)}"
            )
        if not complete:
            return

        for inst in self.instances.values():
            for port in inst.component.inputs:
                if Terminal(inst, port) not in driven:
                    raise CompileError(
                        f"incomplete netlist: {Terminal(inst, port).describe()} "
                        f"({inst.component.name}) is undriven"
                    )
        if self.is_sequential:
            if not clocks or not resets:
                raise CompileError(
                    "a sequential netlist needs exactly one ClockSource and one "
                    "ResetSource"
                )
        elif clocks or resets:
            raise CompileError("a combinational netlist has no clock or reset source")

    def to_dict(self) -> dict[str, Any]:
        """Serialisable snapshot for the place-and-route replay trace / debugging."""
        instances: list[dict[str, Any]] = []
        for inst in self.instances.values():
            entry: dict[str, Any] = {
                "id": inst.id,
                "component": inst.component.name,
                "origin": list(inst.origin) if inst.origin is not None else None,
            }
            if inst.init is not None:
                entry["init"] = inst.init
            instances.append(entry)
        payload: dict[str, Any] = {
            "sequential": self.is_sequential,
            "instances": instances,
            "nets": [
                {
                    "id": net.id,
                    "type": net.dtype.to_dict(),
                    "width": net.width,
                    "layout": net.layout.to_dict(),
                    "driver": {
                        "instance": net.driver.instance.id,
                        "pin": net.driver.port.name,
                    },
                    "sinks": [
                        {"instance": s.instance.id, "pin": s.port.name}
                        for s in net.sinks
                    ],
                }
                for net in self.nets.values()
            ],
        }
        if self.is_sequential:
            payload["clock"] = {"name": "clk", "reset": "rst", "edge": "posedge"}
        return payload


def _check_global_control(net: Net) -> None:
    """Clock and reset are global control nets, never ordinary data."""
    source = net.driver.instance.component
    for pin, source_type, kind in (
        (Register.CLOCK, ClockSource, "clock"),
        (Register.RESET, ResetSource, "reset"),
    ):
        for sink in net.sinks:
            is_control_pin = (
                isinstance(sink.instance.component, Register) and sink.port.name == pin
            )
            if is_control_pin and not isinstance(source, source_type):
                raise CompileError(
                    f"net {net.id}: register {kind} {sink.describe()} must be driven "
                    f"by the global {source_type.__name__}"
                )
            if isinstance(source, source_type) and not is_control_pin:
                raise CompileError(
                    f"net {net.id}: the global {kind} may only reach register "
                    f"{pin!r} pins, not {sink.describe()}"
                )
