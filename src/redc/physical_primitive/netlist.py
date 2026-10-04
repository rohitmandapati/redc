"""The technology-neutral, one-bit primitive netlist.

Primitive synthesis (:mod:`redc.physical_primitive.synthesis`) lowers the
target-neutral IR :class:`~redc.ir.Graph` into this structure.  It is the
"what Boolean circuit computes this program?" answer, and knows NOTHING about
Minecraft: no blocks, dust, repeaters, voxels, orientations or coordinates.

Fundamental invariant: every ordinary runtime computation is a one-bit gate of
the canonical basis (:data:`GATE_KINDS` = AND, OR, XOR, NOT), every state
element is a one-bit :attr:`PrimitiveKind.REGISTER_BIT`, and every net carries
exactly one bit.  There is no width field anywhere: a pin *is* one bit, so a
multi-bit net or bus-routing resource cannot even be expressed.  Wide values
exist only as metadata (:class:`LogicalPort`, :class:`LogicalBus`) that names
which independent one-bit signals belong to one logical value; that metadata
never becomes a routing resource.

* :class:`PrimitiveInstance` -- one primitive (gate, register bit, constant,
  input/output bit, clock/reset source or peripheral) with its
  :class:`Provenance`.
* :class:`BitTerminal` -- one pin of one instance; a driver terminal doubles as
  the handle (:data:`Bit`) of the signal it drives.
* :class:`BitNet` -- one driver fanning out to one or more sinks (fanout is ONE
  net with many sinks, never several nets from one driver).
* :class:`HierarchyGroup` -- the synthesis hierarchy (IR node -> helper ->
  bit slice -> ...) so a viewer can highlight "every gate of IR node 42" or
  "adder bit slice 5".
* :class:`PrimitiveNetlist` -- all of the above, plus :meth:`validate`.

Ids are dense and per-kind: instances ``0..N-1``, nets ``0..M-1``, groups
``0..G-1`` (an instance id and a net id may be equal; records always say which
one they reference).  Bits are LSB first everywhere: ``bits[0]`` is the least
significant bit.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

from ..ir import OPS, IRType
from ..parser import CompileError

SCHEMA = "redc.primitive-netlist.v1"


class PrimitiveKind(Enum):
    """Every primitive the technology-neutral netlist may contain."""

    AND = "and"
    OR = "or"
    XOR = "xor"
    NOT = "not"
    REGISTER_BIT = "register_bit"
    CONST0 = "const0"
    CONST1 = "const1"
    INPUT_BIT = "input_bit"
    OUTPUT_BIT = "output_bit"
    CLOCK_SOURCE = "clock_source"
    RESET_SOURCE = "reset_source"
    PERIPHERAL = "peripheral"

    @property
    def is_gate(self) -> bool:
        """One of the canonical combinational basis gates."""
        return self in GATE_KINDS

    @property
    def category(self) -> str:
        """Display grouping: gate, state, constant, boundary, control, peripheral."""
        return _CATEGORY[self]


#: The canonical Boolean basis.  Synthesis must decompose EVERY operation into
#: these; nothing wider (adders, muxes, comparators, ...) may survive.
GATE_KINDS = frozenset({PrimitiveKind.AND, PrimitiveKind.OR, PrimitiveKind.XOR, PrimitiveKind.NOT})

#: Kinds that hold no computation: state, constants, boundaries, global control.
NON_GATE_KINDS = frozenset(PrimitiveKind) - GATE_KINDS

_CATEGORY = {
    PrimitiveKind.AND: "gate",
    PrimitiveKind.OR: "gate",
    PrimitiveKind.XOR: "gate",
    PrimitiveKind.NOT: "gate",
    PrimitiveKind.REGISTER_BIT: "state",
    PrimitiveKind.CONST0: "constant",
    PrimitiveKind.CONST1: "constant",
    PrimitiveKind.INPUT_BIT: "boundary",
    PrimitiveKind.OUTPUT_BIT: "boundary",
    PrimitiveKind.CLOCK_SOURCE: "control",
    PrimitiveKind.RESET_SOURCE: "control",
    PrimitiveKind.PERIPHERAL: "peripheral",
}

#: IR operation names that must never appear as a primitive (everything in
#: ``redc.ir.OPS`` except the four that coincide with basis gate names).
FORBIDDEN_PRIMITIVE_NAMES = frozenset(OPS) - {k.value for k in GATE_KINDS}

#: Fixed ``(inputs, outputs)`` pin names of every non-peripheral kind.  Every
#: pin is exactly one bit.  Input order is operand order.
PIN_INTERFACE: dict[PrimitiveKind, tuple[tuple[str, ...], tuple[str, ...]]] = {
    PrimitiveKind.AND: (("a", "b"), ("y",)),
    PrimitiveKind.OR: (("a", "b"), ("y",)),
    PrimitiveKind.XOR: (("a", "b"), ("y",)),
    PrimitiveKind.NOT: (("a",), ("y",)),
    PrimitiveKind.REGISTER_BIT: (("d", "clk", "rst"), ("q",)),
    PrimitiveKind.CONST0: ((), ("y",)),
    PrimitiveKind.CONST1: ((), ("y",)),
    PrimitiveKind.INPUT_BIT: ((), ("y",)),
    PrimitiveKind.OUTPUT_BIT: (("a",), ()),
    PrimitiveKind.CLOCK_SOURCE: ((), ("clk",)),
    PrimitiveKind.RESET_SOURCE: ((), ("rst",)),
}

#: Register-bit pin names.
REG_D, REG_CLK, REG_RST, REG_Q = "d", "clk", "rst", "q"


class BitTerminal(NamedTuple):
    """One one-bit pin of one primitive instance (what a :class:`BitNet`
    attaches to)."""

    instance: int
    pin: str

    def ref(self) -> dict[str, Any]:
        return {"instance": self.instance, "pin": self.pin}

    def describe(self) -> str:
        return f"pin {self.pin!r} of primitive {self.instance}"


#: A one-bit signal handle: the OUTPUT terminal that drives the signal.
Bit = BitTerminal

#: A logical bit-vector during synthesis: LSB first (``bits[0]`` is bit 0).
#: Allowed only as a synthesis-time value; it is never a routed object.
BitVector = tuple[BitTerminal, ...]


class PeripheralDirection(Enum):
    """Which way a peripheral moves bits across the design boundary."""

    INPUT = "input"  # device -> circuit (lever, button, ...): drives bits
    OUTPUT = "output"  # circuit -> device (lamp, display, ...): consumes bits


@dataclass(frozen=True, slots=True)
class PeripheralSpec:
    """A conceptual multi-bit boundary device with independent one-bit pins.

    ``kind`` is the interface name (``"lever"``, ``"2-dig-7-seg"``) as a future
    source annotation (``uint8<2-dig-7-seg> result``) would spell it -- NOT its
    Minecraft geometry, which the primitive technology library supplies.  Pins
    are ``b0 .. b{width-1}`` (LSB first): outputs of an INPUT device, inputs of
    an OUTPUT device.  The device stays ONE instance; its bits stay separate
    one-bit connections.
    """

    kind: str
    direction: PeripheralDirection
    width: int

    def __post_init__(self) -> None:
        if not self.kind:
            raise CompileError("a peripheral needs a kind")
        if not isinstance(self.direction, PeripheralDirection):
            raise CompileError(f"peripheral {self.kind!r}: direction must be a PeripheralDirection")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or not 1 <= self.width <= 64:
            raise CompileError(f"peripheral {self.kind!r}: width must be 1..64")

    @property
    def pins(self) -> tuple[str, ...]:
        return tuple(f"b{i}" for i in range(self.width))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "direction": self.direction.value, "width": self.width}


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a primitive came from.

    ``ir_node`` is the source IR node id (``None`` for global control such as
    the clock), ``role`` the generated role (``"propagate_xor"``,
    ``"carry_generate"``, ``"restoring_compare"``, ``"input_bit"``, ...),
    ``bit`` the logical bit index where one applies, ``group`` the innermost
    :class:`HierarchyGroup` id, ``port`` the logical port name of a boundary
    bit, and ``attrs`` any extra integer/string facts (``("iteration", 5)``).
    The IR op and result type are looked up from :attr:`PrimitiveNetlist.ir_nodes`.
    """

    ir_node: int | None
    role: str
    bit: int | None = None
    group: int | None = None
    port: str | None = None
    attrs: tuple[tuple[str, int | str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ir_node": self.ir_node,
            "role": self.role,
            "bit": self.bit,
            "group": self.group,
        }
        if self.port is not None:
            record["port"] = self.port
        if self.attrs:
            record["attrs"] = dict(self.attrs)
        return record


@dataclass(frozen=True, slots=True)
class HierarchyGroup:
    """One node of the synthesis hierarchy.

    ``kind`` is ``"ir_node"`` (the top group of one IR node), ``"slice"`` (a bit
    slice), ``"stage"`` (a shifter stage), ``"iteration"`` (a divider row),
    ``"helper"`` (an adder, comparator, mux, ...), ``"port"``, ``"register"`` or
    ``"control"``.  ``ir_node`` is inherited from the enclosing IR-node group.
    """

    id: int
    name: str
    kind: str
    parent: int | None
    ir_node: int | None
    attrs: tuple[tuple[str, int | str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "parent": self.parent,
            "ir_node": self.ir_node,
        }
        if self.attrs:
            record["attrs"] = dict(self.attrs)
        return record


@dataclass(frozen=True, slots=True)
class IRNodeInfo:
    """A compact copy of one live IR node, for provenance lookups."""

    id: int
    op: str
    type: IRType
    args: tuple[int, ...]
    attrs: tuple[tuple[str, int | str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": self.id,
            "op": self.op,
            "type": _type_record(self.type),
            "args": list(self.args),
        }
        record.update(dict(self.attrs))
        return record


@dataclass(frozen=True, slots=True)
class PrimitiveInstance:
    """One primitive.  ``init`` is the reset bit of a register bit (``None``
    otherwise); ``peripheral`` the device spec of a PERIPHERAL (``None``
    otherwise)."""

    id: int
    kind: PrimitiveKind
    provenance: Provenance
    init: bool | None = None
    peripheral: PeripheralSpec | None = None

    @property
    def inputs(self) -> tuple[str, ...]:
        if self.kind is PrimitiveKind.PERIPHERAL:
            assert self.peripheral is not None
            return self.peripheral.pins if self.peripheral.direction is PeripheralDirection.OUTPUT else ()
        return PIN_INTERFACE[self.kind][0]

    @property
    def outputs(self) -> tuple[str, ...]:
        if self.kind is PrimitiveKind.PERIPHERAL:
            assert self.peripheral is not None
            return self.peripheral.pins if self.peripheral.direction is PeripheralDirection.INPUT else ()
        return PIN_INTERFACE[self.kind][1]

    @property
    def is_gate(self) -> bool:
        return self.kind in GATE_KINDS

    @property
    def is_stateful(self) -> bool:
        return self.kind is PrimitiveKind.REGISTER_BIT

    def terminal(self, pin: str) -> BitTerminal:
        if pin not in self.inputs and pin not in self.outputs:
            raise CompileError(f"primitive {self.id} ({self.kind.value}) has no pin {pin!r}")
        return BitTerminal(self.id, pin)

    def output(self, pin: str | None = None) -> BitTerminal:
        """The (single, unless ``pin`` is given) output terminal."""
        outputs = self.outputs
        if pin is None:
            if len(outputs) != 1:
                raise CompileError(f"primitive {self.id} ({self.kind.value}) has {len(outputs)} outputs")
            pin = outputs[0]
        elif pin not in outputs:
            raise CompileError(f"primitive {self.id} ({self.kind.value}) has no output {pin!r}")
        return BitTerminal(self.id, pin)

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {"id": self.id, "kind": self.kind.value, **self.provenance.to_dict()}
        if self.init is not None:
            record["init"] = int(self.init)
        if self.peripheral is not None:
            record["peripheral"] = self.peripheral.to_dict()
        return record


@dataclass(frozen=True, slots=True)
class BitNet:
    """One one-bit signal: a single driver terminal fanning out to sinks.

    ``role`` is ``"clock"`` / ``"reset"`` for the global control nets, else
    ``"data"``.  ``width`` exists only so validation can assert it is 1."""

    id: int
    driver: BitTerminal
    sinks: tuple[BitTerminal, ...]
    role: str = "data"
    width: int = 1

    @property
    def fanout(self) -> int:
        return len(self.sinks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "width": self.width,
            "driver": self.driver.ref(),
            "sinks": [s.ref() for s in self.sinks],
            "fanout": self.fanout,
        }


@dataclass(frozen=True, slots=True)
class LogicalPort:
    """A module port as ONE logical value over independent one-bit signals.

    ``bits`` are LSB first: for an input port the driver terminals of each bit
    (an INPUT_BIT's ``y`` or a peripheral's ``b<i>`` output); for an output port
    the sink terminals that observe each bit (an OUTPUT_BIT's ``a`` or a
    peripheral's ``b<i>`` input).  ``realization`` is ``"pads"`` or the
    peripheral kind.  Pure grouping metadata -- never a routing resource."""

    name: str
    direction: str  # "input" | "output"
    type: IRType
    source_name: str | None
    ir_node: int
    bits: tuple[BitTerminal, ...]
    realization: str = "pads"

    @property
    def instances(self) -> tuple[int, ...]:
        seen: dict[int, None] = {}
        for bit in self.bits:
            seen.setdefault(bit.instance, None)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "type": _type_record(self.type),
            "source_name": self.source_name,
            "ir_node": self.ir_node,
            "realization": self.realization,
            "bits": [b.ref() for b in self.bits],
        }


@dataclass(frozen=True, slots=True)
class LogicalBus:
    """The result of one live IR node as an LSB-first vector of independent
    one-bit signals (driver terminals).  The same signal may appear in several
    buses (a signedness-only cast reuses its source bits) or several times in
    one bus (sign extension).  Metadata only: never routed as a unit."""

    ir_node: int
    op: str
    type: IRType
    bits: tuple[BitTerminal, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ir_node": self.ir_node,
            "op": self.op,
            "type": _type_record(self.type),
            "bits": [b.ref() for b in self.bits],
        }


def _type_record(typ: IRType) -> dict[str, Any]:
    return {"name": typ.name, "width": typ.width, "signed": typ.signed, "boolean": typ.boolean}


@dataclass
class PrimitiveNetlist:
    """The technology-neutral one-bit circuit.

    Built by :class:`~redc.physical_primitive.synthesis.builder.PrimitiveBuilder`
    (which is the only intended mutator); :meth:`validate` re-checks every rule
    from scratch.  ``graph_summary`` describes the source IR (live node count,
    ops by kind, sequential flag)."""

    instances: list[PrimitiveInstance] = field(default_factory=list)
    nets: list[BitNet] = field(default_factory=list)
    groups: list[HierarchyGroup] = field(default_factory=list)
    ports: list[LogicalPort] = field(default_factory=list)
    buses: dict[int, LogicalBus] = field(default_factory=dict)
    ir_nodes: dict[int, IRNodeInfo] = field(default_factory=dict)
    graph_summary: dict[str, Any] = field(default_factory=dict)
    # Connectivity indexes, maintained by add_net().
    _net_of_driver: dict[BitTerminal, int] = field(default_factory=dict, repr=False)
    _net_of_sink: dict[BitTerminal, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.nets and not self._net_of_driver:
            for net in self.nets:
                self._net_of_driver.setdefault(net.driver, net.id)
                for sink in net.sinks:
                    self._net_of_sink.setdefault(sink, net.id)

    # -- construction (used by the builder) --------------------------------

    def add_instance(
        self,
        kind: PrimitiveKind,
        provenance: Provenance,
        *,
        init: bool | None = None,
        peripheral: PeripheralSpec | None = None,
    ) -> PrimitiveInstance:
        instance = PrimitiveInstance(len(self.instances), kind, provenance, init, peripheral)
        _check_instance(instance, len(self.groups))
        self.instances.append(instance)
        return instance

    def add_group(
        self,
        name: str,
        kind: str,
        parent: int | None,
        ir_node: int | None,
        attrs: tuple[tuple[str, int | str], ...] = (),
    ) -> HierarchyGroup:
        if parent is not None and not 0 <= parent < len(self.groups):
            raise CompileError(f"group {name!r}: unknown parent group {parent}")
        group = HierarchyGroup(len(self.groups), name, kind, parent, ir_node, attrs)
        self.groups.append(group)
        return group

    def add_net(self, driver: BitTerminal, sinks: Iterable[BitTerminal], role: str = "data") -> BitNet:
        net = BitNet(len(self.nets), driver, tuple(sinks), role)
        self._check_net(net, self._net_of_driver, self._net_of_sink)
        self._net_of_driver[net.driver] = net.id
        for sink in net.sinks:
            self._net_of_sink[sink] = net.id
        self.nets.append(net)
        return net

    # -- queries -----------------------------------------------------------

    def instance(self, instance_id: int) -> PrimitiveInstance:
        if not 0 <= instance_id < len(self.instances):
            raise CompileError(f"unknown primitive instance {instance_id}")
        return self.instances[instance_id]

    def net_of_driver(self, bit: BitTerminal) -> BitNet | None:
        """The net ``bit`` drives (``None`` if the signal has no sinks)."""
        net_id = self._net_of_driver.get(bit)
        return None if net_id is None else self.nets[net_id]

    def driver_of(self, sink: BitTerminal) -> BitTerminal | None:
        """The signal driving input terminal ``sink`` (``None`` if undriven)."""
        net_id = self._net_of_sink.get(sink)
        return None if net_id is None else self.nets[net_id].driver

    @property
    def is_sequential(self) -> bool:
        return any(inst.kind is PrimitiveKind.REGISTER_BIT for inst in self.instances)

    def kind_counts(self) -> dict[str, int]:
        counts = Counter(inst.kind.value for inst in self.instances)
        return {kind.value: counts.get(kind.value, 0) for kind in PrimitiveKind}

    @property
    def gate_count(self) -> int:
        return sum(1 for inst in self.instances if inst.kind in GATE_KINDS)

    def fanout_distribution(self) -> dict[str, int]:
        """``{"1": nets with fanout 1, "2": ..., ...}`` (string keys for JSON)."""
        counts = Counter(net.fanout for net in self.nets)
        return {str(k): counts[k] for k in sorted(counts)}

    def group_path(self, group_id: int | None) -> tuple[str, ...]:
        """Names from the root group down to ``group_id``."""
        path: list[str] = []
        while group_id is not None:
            group = self.groups[group_id]
            path.append(group.name)
            group_id = group.parent
        return tuple(reversed(path))

    def descendants(self, group_id: int) -> set[int]:
        """``group_id`` and every group nested under it."""
        children: dict[int, list[int]] = {}
        for group in self.groups:
            if group.parent is not None:
                children.setdefault(group.parent, []).append(group.id)
        result, pending = set(), [group_id]
        while pending:
            current = pending.pop()
            result.add(current)
            pending.extend(children.get(current, ()))
        return result

    def instances_in_group(self, group_id: int) -> list[PrimitiveInstance]:
        """Every primitive whose provenance group is ``group_id`` or below."""
        groups = self.descendants(group_id)
        return [inst for inst in self.instances if inst.provenance.group in groups]

    def instances_of_ir_node(self, node_id: int) -> list[PrimitiveInstance]:
        return [inst for inst in self.instances if inst.provenance.ir_node == node_id]

    def bus_memberships(self) -> dict[BitTerminal, list[tuple[int, int]]]:
        """Signal -> every ``(ir_node, bit index)`` naming it (logical aliases)."""
        names: dict[BitTerminal, list[tuple[int, int]]] = {}
        for node_id in sorted(self.buses):
            for index, bit in enumerate(self.buses[node_id].bits):
                names.setdefault(bit, []).append((node_id, index))
        return names

    def port_memberships(self) -> dict[BitTerminal, list[tuple[str, int]]]:
        """Terminal -> every ``(port name, bit index)`` it carries."""
        names: dict[BitTerminal, list[tuple[str, int]]] = {}
        for port in self.ports:
            for index, bit in enumerate(port.bits):
                names.setdefault(bit, []).append((port.name, index))
        return names

    def port(self, name: str) -> LogicalPort:
        for port in self.ports:
            if port.name == name:
                return port
        raise CompileError(f"no logical port {name!r}")

    # -- validation ----------------------------------------------------------

    def _check_net(
        self,
        net: BitNet,
        driving: Mapping[BitTerminal, int],
        driven: Mapping[BitTerminal, int],
    ) -> None:
        if net.width != 1:
            raise CompileError(f"net {net.id}: every primitive net is one bit, got width {net.width}")
        if not net.sinks:
            raise CompileError(f"net {net.id}: has no sinks")
        if len(set(net.sinks)) != len(net.sinks):
            raise CompileError(f"net {net.id}: lists the same sink pin twice")
        driver = self._resolve(net.id, net.driver, output=True)
        if net.driver in driving:
            raise CompileError(
                f"net {net.id}: {net.driver.describe()} already drives net "
                f"{driving[net.driver]} (fanout must be one net with many sinks)"
            )
        sink_instances = []
        for sink in net.sinks:
            sink_instances.append(self._resolve(net.id, sink, output=False))
            if sink in driven:
                raise CompileError(
                    f"net {net.id}: input {sink.describe()} is already driven by net {driven[sink]}"
                )
        _check_global_control(net, driver, sink_instances)

    def _resolve(self, net_id: int, terminal: BitTerminal, *, output: bool) -> PrimitiveInstance:
        if not isinstance(terminal, BitTerminal):
            raise CompileError(f"net {net_id}: {terminal!r} is not a BitTerminal")
        if not 0 <= terminal.instance < len(self.instances):
            raise CompileError(
                f"net {net_id} references primitive {terminal.instance}, which is not part of this netlist"
            )
        instance = self.instances[terminal.instance]
        pins = instance.outputs if output else instance.inputs
        if terminal.pin not in pins:
            what = "output" if output else "input"
            raise CompileError(
                f"net {net_id}: primitive {instance.id} ({instance.kind.value}) has no {what} pin "
                f"{terminal.pin!r}"
            )
        return instance

    def validate(self, *, complete: bool = False) -> None:
        """Re-check every structural rule from scratch.

        Always: dense unique ids; every instance is a basis gate or a distinct
        boundary/state kind (no IR op such as ``add`` / ``mul`` survives); every
        net is one bit with one driver output pin and distinct sink input pins;
        an output pin drives at most one net; an input pin is driven by at most
        one net; every terminal names a real pin of a real instance; clock and
        reset nets only reach register ``clk`` / ``rst`` pins and those pins are
        only driven by the global sources; at most one clock and one reset
        source; ports and buses reference real terminals of the right
        direction; provenance groups exist.

        With ``complete=True`` additionally: every input pin is driven, and a
        sequential netlist has exactly one CLOCK_SOURCE and one RESET_SOURCE
        (a combinational one has neither)."""
        for index, inst in enumerate(self.instances):
            if inst.id != index:
                raise CompileError(f"primitive {inst.id} is stored at index {index}")
            _check_instance(inst, len(self.groups))
        for index, group in enumerate(self.groups):
            if group.id != index:
                raise CompileError(f"group {group.id} is stored at index {index}")
            if group.parent is not None and not 0 <= group.parent < index:
                raise CompileError(f"group {group.id}: parent {group.parent} must precede it")
        driving: dict[BitTerminal, int] = {}
        driven: dict[BitTerminal, int] = {}
        for index, net in enumerate(self.nets):
            if net.id != index:
                raise CompileError(f"net {net.id} is stored at index {index}")
            self._check_net(net, driving, driven)
            driving[net.driver] = net.id
            for sink in net.sinks:
                driven[sink] = net.id
        if driving != self._net_of_driver or driven != self._net_of_sink:
            raise CompileError("primitive netlist connectivity indexes are stale")

        clocks = [i for i in self.instances if i.kind is PrimitiveKind.CLOCK_SOURCE]
        resets = [i for i in self.instances if i.kind is PrimitiveKind.RESET_SOURCE]
        if len(clocks) > 1:
            raise CompileError(f"one global clock domain allows one CLOCK_SOURCE, found {len(clocks)}")
        if len(resets) > 1:
            raise CompileError(f"one global reset allows one RESET_SOURCE, found {len(resets)}")

        for port in self.ports:
            if port.direction not in ("input", "output"):
                raise CompileError(f"port {port.name!r}: bad direction {port.direction!r}")
            if len(port.bits) != port.type.width:
                raise CompileError(
                    f"port {port.name!r}: {len(port.bits)} bit terminals for a {port.type.name}"
                )
            for bit in port.bits:
                self._resolve(-1, bit, output=port.direction == "input")
        named = Counter((port.direction, port.name) for port in self.ports)
        for (direction, name), count in sorted(named.items()):
            if count > 1:
                raise CompileError(f"{count} {direction} ports are named {name!r}; port names must be unique")
        for node_id, bus in self.buses.items():
            if bus.ir_node != node_id or len(bus.bits) != bus.type.width:
                raise CompileError(f"bus of IR node {node_id} is malformed")
            for bit in bus.bits:
                self._resolve(-1, bit, output=True)

        if not complete:
            return
        for inst in self.instances:
            for pin in inst.inputs:
                if BitTerminal(inst.id, pin) not in driven:
                    raise CompileError(
                        f"incomplete primitive netlist: {BitTerminal(inst.id, pin).describe()} "
                        f"({inst.kind.value}) is undriven"
                    )
        if self.is_sequential:
            if not clocks or not resets:
                raise CompileError(
                    "a sequential primitive netlist needs exactly one CLOCK_SOURCE and one RESET_SOURCE"
                )
        elif clocks or resets:
            raise CompileError("a combinational primitive netlist has no clock or reset source")

    # -- serialization -------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Deterministic size metrics (no wall-clock data)."""
        kinds = self.kind_counts()
        return {
            "instances": len(self.instances),
            "gates": self.gate_count,
            "gates_by_kind": {k.value: kinds[k.value] for k in sorted(GATE_KINDS, key=lambda k: k.value)},
            "by_kind": kinds,
            "register_bits": kinds[PrimitiveKind.REGISTER_BIT.value],
            "constant_bits": kinds[PrimitiveKind.CONST0.value] + kinds[PrimitiveKind.CONST1.value],
            "nets": len(self.nets),
            "logical_ports": len(self.ports),
            "logical_buses": len(self.buses),
            "groups": len(self.groups),
            "fanout": self.fanout_distribution(),
            "max_fanout": max((n.fanout for n in self.nets), default=0),
        }

    def to_dict(self) -> dict[str, Any]:
        """The ``redc.primitive-netlist.v1`` JSON document."""
        return {
            "schema": SCHEMA,
            "backend": "physical-primitive",
            "stage": "primitive",
            "sequential": self.is_sequential,
            "bit_order": "lsb_first",
            "basis": sorted(k.value for k in GATE_KINDS),
            "ir": self.graph_summary,
            "summary": self.summary(),
            "ir_nodes": [self.ir_nodes[k].to_dict() for k in sorted(self.ir_nodes)],
            "groups": [g.to_dict() for g in self.groups],
            "ports": [p.to_dict() for p in self.ports],
            "buses": [self.buses[k].to_dict() for k in sorted(self.buses)],
            "instances": [i.to_dict() for i in self.instances],
            "nets": [n.to_dict() for n in self.nets],
        }

    def __iter__(self) -> Iterator[PrimitiveInstance]:
        return iter(self.instances)


def _check_instance(inst: PrimitiveInstance, group_count: int) -> None:
    if not isinstance(inst.kind, PrimitiveKind):
        raise CompileError(f"primitive {inst.id}: {inst.kind!r} is not a PrimitiveKind")
    if inst.kind.value in FORBIDDEN_PRIMITIVE_NAMES:  # pragma: no cover - enum guards this
        raise CompileError(f"primitive {inst.id}: IR operation {inst.kind.value!r} survived synthesis")
    if inst.kind is PrimitiveKind.REGISTER_BIT:
        if not isinstance(inst.init, bool):
            raise CompileError(f"register bit {inst.id}: init must be a bool, got {inst.init!r}")
    elif inst.init is not None:
        raise CompileError(f"primitive {inst.id} ({inst.kind.value}): only register bits carry init")
    if inst.kind is PrimitiveKind.PERIPHERAL:
        if not isinstance(inst.peripheral, PeripheralSpec):
            raise CompileError(f"peripheral {inst.id}: missing PeripheralSpec")
    elif inst.peripheral is not None:
        raise CompileError(f"primitive {inst.id} ({inst.kind.value}): only peripherals carry a spec")
    group = inst.provenance.group
    if group is not None and not 0 <= group < group_count:
        raise CompileError(f"primitive {inst.id}: unknown provenance group {group}")
    if not inst.provenance.role:
        raise CompileError(f"primitive {inst.id}: provenance needs a role")


def _check_global_control(
    net: BitNet, driver: PrimitiveInstance, sinks: list[PrimitiveInstance]
) -> None:
    """Clock and reset are global control nets, never ordinary data."""
    for pin, source_kind, label in (
        (REG_CLK, PrimitiveKind.CLOCK_SOURCE, "clock"),
        (REG_RST, PrimitiveKind.RESET_SOURCE, "reset"),
    ):
        for sink, inst in zip(net.sinks, sinks):
            is_control_pin = inst.kind is PrimitiveKind.REGISTER_BIT and sink.pin == pin
            if is_control_pin and driver.kind is not source_kind:
                raise CompileError(
                    f"net {net.id}: register {label} {sink.describe()} must be driven by the global "
                    f"{source_kind.value}"
                )
            if driver.kind is source_kind and not is_control_pin:
                raise CompileError(
                    f"net {net.id}: the global {label} may only reach register {pin!r} pins, not "
                    f"{sink.describe()}"
                )
    expected = {
        PrimitiveKind.CLOCK_SOURCE: "clock",
        PrimitiveKind.RESET_SOURCE: "reset",
    }.get(driver.kind, "data")
    if net.role != expected:
        raise CompileError(f"net {net.id}: role {net.role!r} but driven by {driver.kind.value}")


__all__ = [
    "FORBIDDEN_PRIMITIVE_NAMES",
    "GATE_KINDS",
    "NON_GATE_KINDS",
    "PIN_INTERFACE",
    "REG_CLK",
    "REG_D",
    "REG_Q",
    "REG_RST",
    "SCHEMA",
    "Bit",
    "BitNet",
    "BitTerminal",
    "BitVector",
    "HierarchyGroup",
    "IRNodeInfo",
    "LogicalBus",
    "LogicalPort",
    "PeripheralDirection",
    "PeripheralSpec",
    "PrimitiveInstance",
    "PrimitiveKind",
    "PrimitiveNetlist",
    "Provenance",
]
