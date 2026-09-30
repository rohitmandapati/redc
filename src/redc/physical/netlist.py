"""Physical netlist: component instances wired together by nets.

The netlist is the circuit as *connectivity + geometry*, sitting between tech-map
(which produces it from the IR) and place-and-route (which consumes it).  It says
*what connects to what* and, once placed, *where each instance sits* -- but not
how wires physically thread between them (routing's job).

* :class:`ComponentInstance` -- one placed-or-placeable use of a
  :class:`~redc.physical.components.Component` definition: the definition, a
  unique id, and an ``origin`` (``None`` until placement assigns one).
* :class:`Terminal` -- a specific pin on a specific instance; the thing a net
  attaches to.  Resolves to an absolute grid cell once its instance is placed.
* :class:`Net` -- one signal: a single driver terminal fanning out to one or more
  sink terminals, carrying ``width`` bits.  A bus is *one* net (width lives here,
  not on the grid); abutment is a routing outcome, so the net is identical whether
  or not it ends up needing wire cells.
* :class:`PhysicalNetlist` -- the whole graph: instances + nets, a single id
  allocator (so instance ids and net ids never collide in the grid's ``owner``
  field), and :meth:`~PhysicalNetlist.validate` for the electrical rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..parser import CompileError
from .components import Component, Port, PortDir


@dataclass(eq=False)
class ComponentInstance:
    """One use of a component definition, with an optional placement origin.

    Identity is per-object (two instances are never "equal" even if they wrap the
    same definition), which is why ``eq=False``.  ``origin`` is mutable: placement
    assigns and may later adjust it.
    """

    id: int
    component: Component
    origin: tuple[int, int, int] | None = None

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
    :class:`ComponentInstance` -- what a :class:`Net` connects to."""

    instance: ComponentInstance
    port: Port

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


@dataclass(frozen=True)
class Net:
    """One signal: a single driver fanning out to one or more sinks.

    Width is the driver's width and every sink must match it.  The net is width-
    carrying but grid-agnostic: a 32-bit bus is one net, not 32.
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
        for sink in self.sinks:
            if sink.port.direction is not PortDir.IN:
                raise CompileError(
                    f"net {self.id}: sink pin {sink.port.name!r} is not an input"
                )
            if sink.width != self.driver.width:
                raise CompileError(
                    f"net {self.id}: width mismatch -- driver {self.driver.width} "
                    f"vs sink {sink.width}"
                )

    @property
    def width(self) -> int:
        return self.driver.width

    @property
    def fanout(self) -> int:
        return len(self.sinks)


@dataclass
class PhysicalNetlist:
    """The circuit graph: component instances plus the nets wiring them."""

    instances: dict[int, ComponentInstance] = field(default_factory=dict)
    nets: dict[int, Net] = field(default_factory=dict)
    _next_id: int = 0

    def _fresh_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def add(
        self, component: Component, origin: tuple[int, int, int] | None = None
    ) -> ComponentInstance:
        """Instantiate ``component`` into the netlist and return the instance."""
        inst = ComponentInstance(self._fresh_id(), component, origin)
        self.instances[inst.id] = inst
        return inst

    def connect(self, driver: Terminal, *sinks: Terminal) -> Net:
        """Wire ``driver`` to one or more ``sinks`` as a new net."""
        net = Net(self._fresh_id(), driver, tuple(sinks))
        self.nets[net.id] = net
        return net

    def validate(self) -> None:
        """Enforce the electrical rules: every terminal's instance belongs to this
        netlist, and no input pin is driven by more than one net."""
        driven: dict[tuple[int, str], int] = {}
        for net in self.nets.values():
            for term in (net.driver, *net.sinks):
                if term.instance.id not in self.instances:
                    raise CompileError(
                        f"net {net.id} references unknown instance {term.instance.id}"
                    )
            for sink in net.sinks:
                key = (sink.instance.id, sink.port.name)
                if key in driven:
                    raise CompileError(
                        f"input pin {sink.port.name!r} on instance {sink.instance.id} "
                        f"is driven by nets {driven[key]} and {net.id}"
                    )
                driven[key] = net.id

    def to_dict(self) -> dict:
        """Serialisable snapshot for the place-and-route replay trace / debugging."""
        return {
            "instances": [
                {
                    "id": inst.id,
                    "component": inst.component.name,
                    "origin": list(inst.origin) if inst.origin is not None else None,
                }
                for inst in self.instances.values()
            ],
            "nets": [
                {
                    "id": net.id,
                    "width": net.width,
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
