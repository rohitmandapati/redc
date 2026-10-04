"""The primitive builder: the ONLY way synthesis creates primitives.

Synthesis recipes receive bit-vector handles and call this builder to emit
one-bit gates; they never touch netlist ids, net construction or anything
physical.  The builder

* allocates :class:`~redc.physical_primitive.netlist.PrimitiveInstance` ids;
* records each gate's operands as sinks of the driving signals and, at
  :meth:`PrimitiveBuilder.finish`, turns every driver with sinks into exactly
  ONE :class:`~redc.physical_primitive.netlist.BitNet` (fanout is one net);
* stamps every primitive with :class:`~redc.physical_primitive.netlist.Provenance`
  taken from the current scope stack -- the IR node being synthesized
  (:meth:`ir_node`) and nested helper groups (:meth:`scope`) such as an adder,
  bit slice 5 or divider iteration 3 -- plus the per-call ``role`` / ``bit``;
* hands out constants per IR node (:meth:`const`), so a constant source is
  reused within one operation (bounded fanout, clear provenance) instead of one
  global constant feeding the whole design.

It does no Boolean simplification: ``and_(x, const(True))`` emits an AND gate.
Constant propagation, structural hashing and friends are later, separate
``PrimitiveNetlist -> PrimitiveNetlist`` passes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from ...ir import IRType
from ...parser import CompileError
from ..netlist import (
    REG_CLK,
    REG_D,
    REG_RST,
    Bit,
    BitTerminal,
    PeripheralSpec,
    PrimitiveInstance,
    PrimitiveKind,
    PrimitiveNetlist,
    Provenance,
)

#: ``callback(instance)`` -- invoked for every primitive emitted (synthesis trace).
EmitHook = Callable[[PrimitiveInstance], None]

Attrs = tuple[tuple[str, int | str], ...]


class PrimitiveBuilder:
    """Emits one-bit primitives into a :class:`PrimitiveNetlist`."""

    def __init__(self, *, on_emit: EmitHook | None = None) -> None:
        self.netlist = PrimitiveNetlist()
        self.on_emit = on_emit
        self._sinks: dict[BitTerminal, list[BitTerminal]] = {}
        self._driven: set[BitTerminal] = set()
        root = self.netlist.add_group("design", "root", None, None)
        self._root = root.id
        # Scope stack: (group id, ir node id or None).
        self._stack: list[tuple[int, int | None]] = [(root.id, None)]
        # IR node -> its top-level group (constants live there).
        self._node_group: dict[int | None, int] = {None: root.id}
        self._consts: dict[tuple[int | None, bool], Bit] = {}
        self._clock: Bit | None = None
        self._reset: Bit | None = None
        self._finished = False

    # -- provenance scopes -------------------------------------------------

    @property
    def current_group(self) -> int:
        return self._stack[-1][0]

    @property
    def current_ir_node(self) -> int | None:
        return self._stack[-1][1]

    def group_of_ir_node(self, node_id: int) -> int | None:
        """The top-level hierarchy group of IR node ``node_id`` (if any yet)."""
        return self._node_group.get(node_id)

    @contextmanager
    def ir_node(self, node_id: int, op: str, typ: IRType | None = None) -> Iterator[int]:
        """Synthesize everything inside on behalf of IR node ``node_id``.

        Each IR node owns exactly one top-level group: re-entering a node (a
        register's next-state wiring happens in a later pass than its state
        bits) reuses it."""
        group_id = self._node_group.get(node_id)
        if group_id is None:
            attrs: Attrs = (("op", op),) + ((("type", typ.name),) if typ is not None else ())
            group_id = self.netlist.add_group(f"n{node_id}.{op}", "ir_node", self._root, node_id, attrs).id
            self._node_group[node_id] = group_id
        self._stack.append((group_id, node_id))
        try:
            yield group_id
        finally:
            self._stack.pop()

    @contextmanager
    def port(self, name: str, direction: str, ir_node: int | None) -> Iterator[int]:
        """Emit the boundary primitives of module port ``name``.  ``ir_node`` is
        the input node, or the node driving an output port."""
        attrs: Attrs = (("direction", direction), ("port", name))
        group = self.netlist.add_group(f"port.{name}", "port", self._root, ir_node, attrs)
        if direction == "input" and ir_node is not None:
            # An input port IS its IR node: the port group is the node's group.
            self._node_group.setdefault(ir_node, group.id)
        self._stack.append((group.id, ir_node))
        try:
            yield group.id
        finally:
            self._stack.pop()

    @contextmanager
    def scope(self, name: str, *, kind: str = "helper", **attrs: int | str) -> Iterator[int]:
        """A nested hierarchy group (``"adder"``, ``"bit5"``, ``"iter3"``, ...)."""
        group_id, node = self._stack[-1]
        group = self.netlist.add_group(name, kind, group_id, node, tuple(sorted(attrs.items())))
        self._stack.append((group.id, node))
        try:
            yield group.id
        finally:
            self._stack.pop()

    def _provenance(
        self, role: str, bit: int | None, port: str | None = None, attrs: Attrs = ()
    ) -> Provenance:
        group, node = self._stack[-1]
        return Provenance(node, role, bit, group, port, attrs)

    # -- emission ----------------------------------------------------------

    def _new(
        self,
        kind: PrimitiveKind,
        provenance: Provenance,
        *,
        init: bool | None = None,
        peripheral: PeripheralSpec | None = None,
    ) -> PrimitiveInstance:
        if self._finished:
            raise CompileError("primitive builder is finished; no further primitives may be emitted")
        instance = self.netlist.add_instance(kind, provenance, init=init, peripheral=peripheral)
        if self.on_emit is not None:
            self.on_emit(instance)
        return instance

    def _check_source(self, bit: Bit) -> None:
        if not isinstance(bit, BitTerminal):
            raise CompileError(f"{bit!r} is not a one-bit signal handle")
        instances = self.netlist.instances
        if not 0 <= bit.instance < len(instances) or bit.pin not in instances[bit.instance].outputs:
            raise CompileError(f"{bit!r} is not an output terminal of this netlist")

    def connect(self, source: Bit, sink: BitTerminal) -> None:
        """Wire signal ``source`` into input terminal ``sink`` (deferred wiring
        for register ``d``/``clk``/``rst`` and boundary sinks)."""
        self._check_source(source)
        instances = self.netlist.instances
        if not 0 <= sink.instance < len(instances) or sink.pin not in instances[sink.instance].inputs:
            raise CompileError(f"{sink!r} is not an input terminal of this netlist")
        if sink in self._driven:
            raise CompileError(f"{sink.describe()} is already driven")
        self._driven.add(sink)
        self._sinks.setdefault(source, []).append(sink)

    def _gate(self, kind: PrimitiveKind, operands: tuple[Bit, ...], role: str, bit: int | None) -> Bit:
        for operand in operands:
            self._check_source(operand)
        instance = self._new(kind, self._provenance(role, bit))
        sinks = self._sinks
        for pin, operand in zip(("a", "b"), operands):
            terminal = BitTerminal(instance.id, pin)
            self._driven.add(terminal)
            sinks.setdefault(operand, []).append(terminal)
        return BitTerminal(instance.id, "y")

    def and_(self, a: Bit, b: Bit, *, role: str = "and", bit: int | None = None) -> Bit:
        return self._gate(PrimitiveKind.AND, (a, b), role, bit)

    def or_(self, a: Bit, b: Bit, *, role: str = "or", bit: int | None = None) -> Bit:
        return self._gate(PrimitiveKind.OR, (a, b), role, bit)

    def xor(self, a: Bit, b: Bit, *, role: str = "xor", bit: int | None = None) -> Bit:
        return self._gate(PrimitiveKind.XOR, (a, b), role, bit)

    def not_(self, a: Bit, *, role: str = "not", bit: int | None = None) -> Bit:
        return self._gate(PrimitiveKind.NOT, (a,), role, bit)

    def const(self, value: bool) -> Bit:
        """The constant-``value`` signal of the current IR node (created once
        per node and value, in that node's top-level group)."""
        node = self.current_ir_node
        key = (node, bool(value))
        cached = self._consts.get(key)
        if cached is not None:
            return cached
        kind = PrimitiveKind.CONST1 if value else PrimitiveKind.CONST0
        group = self._node_group.get(node, self._root)
        instance = self._new(kind, Provenance(node, kind.value, None, group))
        bit = BitTerminal(instance.id, "y")
        self._consts[key] = bit
        return bit

    # -- state and boundaries ------------------------------------------------

    def register_bit(self, *, init: bool, bit: int, role: str = "register_bit") -> Bit:
        """A one-bit state element; returns its ``q`` signal.  Its ``d``,
        ``clk`` and ``rst`` inputs are wired later with :meth:`connect` (the
        register's next-state logic may not exist yet: IR back-edges)."""
        instance = self._new(
            PrimitiveKind.REGISTER_BIT, self._provenance(role, bit), init=bool(init)
        )
        return BitTerminal(instance.id, "q")

    @staticmethod
    def register_pins(q: Bit) -> tuple[BitTerminal, BitTerminal, BitTerminal]:
        """The ``(d, clk, rst)`` input terminals of the register owning ``q``."""
        return (
            BitTerminal(q.instance, REG_D),
            BitTerminal(q.instance, REG_CLK),
            BitTerminal(q.instance, REG_RST),
        )

    def input_bit(self, port: str, bit: int) -> Bit:
        instance = self._new(PrimitiveKind.INPUT_BIT, self._provenance("input_bit", bit, port))
        return BitTerminal(instance.id, "y")

    def output_bit(self, port: str, bit: int, source: Bit) -> BitTerminal:
        self._check_source(source)  # reject before emitting anything
        instance = self._new(PrimitiveKind.OUTPUT_BIT, self._provenance("output_bit", bit, port))
        sink = BitTerminal(instance.id, "a")
        self.connect(source, sink)
        return sink

    def peripheral(self, spec: PeripheralSpec, port: str) -> PrimitiveInstance:
        """ONE conceptual device with ``spec.width`` independent one-bit pins."""
        return self._new(
            PrimitiveKind.PERIPHERAL, self._provenance("peripheral", None, port), peripheral=spec
        )

    def _control(self, kind: PrimitiveKind, pin: str) -> Bit:
        group = self.netlist.add_group(kind.value, "control", self._root, None)
        instance = self._new(kind, Provenance(None, kind.value, None, group.id))
        return BitTerminal(instance.id, pin)

    def clock(self) -> Bit:
        """The single global clock signal (created on first use)."""
        if self._clock is None:
            self._clock = self._control(PrimitiveKind.CLOCK_SOURCE, "clk")
        return self._clock

    def reset(self) -> Bit:
        """The single global active-high reset signal (created on first use)."""
        if self._reset is None:
            self._reset = self._control(PrimitiveKind.RESET_SOURCE, "rst")
        return self._reset

    # -- finish --------------------------------------------------------------

    def finish(self, *, validate: bool = True) -> PrimitiveNetlist:
        """Create one net per driven signal (instance order, then pin order)
        and return the netlist, checked with ``validate(complete=True)``."""
        if self._finished:
            raise CompileError("primitive builder is already finished")
        self._finished = True
        netlist = self.netlist
        for instance in netlist.instances:
            for pin in instance.outputs:
                driver = BitTerminal(instance.id, pin)
                sinks = self._sinks.get(driver)
                if not sinks:
                    continue
                role = (
                    "clock"
                    if instance.kind is PrimitiveKind.CLOCK_SOURCE
                    else "reset"
                    if instance.kind is PrimitiveKind.RESET_SOURCE
                    else "data"
                )
                netlist.add_net(driver, sinks, role)
        if validate:
            netlist.validate(complete=True)
        return netlist


__all__ = ["EmitHook", "PrimitiveBuilder"]
