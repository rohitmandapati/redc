"""Primitive synthesis: validated IR :class:`~redc.ir.Graph` -> one-bit
:class:`~redc.physical_primitive.netlist.PrimitiveNetlist`.

This is the "what Boolean circuit computes this program?" phase.  It consumes
the very same validated graph the SystemVerilog and coarse physical backends
use, lowers only ``graph.live_nodes()`` (dead IR never becomes gates), and
knows nothing about Minecraft.

Register back-edges (``next`` / ``enable`` may point at later nodes) rule out a
naive single pass, so lowering runs in four passes:

* **Pass A -- declare.**  Every live ``input`` becomes ``width`` independent
  source bits (pads or one peripheral, per the :class:`InterfacePolicy`), and
  every live ``register`` becomes ``width`` one-bit REGISTER_BIT primitives
  whose ``q`` outputs are its current value -- so combinational logic can read
  state before the next-state logic exists.
* **Pass B -- combinational.**  ``const`` nodes are bit-blasted and every
  ordinary node is synthesized, in IR order, by the recipe its exact
  :class:`~redc.signature.OperationSignature` resolves to in the
  :class:`~redc.physical_primitive.synthesis.registry.SynthesisRegistry`.
* **Pass C -- outputs.**  Each output port observes its node's bits through
  output pads or one peripheral.
* **Pass D -- state wiring.**  Each register bit's ``d`` gets
  ``MUX(enable, next[i], q[i])`` (the IR's enable semantics expressed as
  ordinary gates, keeping the state primitive minimal), and its ``clk`` / ``rst``
  join the single global clock / reset nets.

The result is checked with ``validate(complete=True)``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Protocol

from ...ir import OPS, Graph, IRType
from ...parser import CompileError
from ...signature import OperationSignature
from ...tracing import TraceLevel
from ..netlist import (
    BitTerminal,
    BitVector,
    IRNodeInfo,
    LogicalBus,
    LogicalPort,
    PeripheralDirection,
    PeripheralSpec,
    PrimitiveInstance,
    PrimitiveNetlist,
)
from .builder import PrimitiveBuilder
from .interface import DEFAULT_INTERFACE_POLICY, InterfacePolicy
from .logic import constant_vector, mux_bit
from .registry import SynthesisRegistry


class SynthesisTrace(Protocol):
    """The part of a trace recorder synthesis reports to."""

    def wants(self, level: TraceLevel) -> bool: ...

    def emit(self, phase: str, type: str, *, level: TraceLevel = ..., **data: Any) -> None: ...


def default_registry() -> SynthesisRegistry:
    from .recipes import DEFAULT_SYNTHESIS

    return DEFAULT_SYNTHESIS


def synthesize_to_primitives(
    graph: Graph,
    *,
    registry: SynthesisRegistry | None = None,
    interface: InterfacePolicy = DEFAULT_INTERFACE_POLICY,
    trace: SynthesisTrace | None = None,
) -> PrimitiveNetlist:
    """Bit-blast every live IR node of ``graph`` into one-bit primitives."""
    graph.validate()
    _check_unique_port_names(graph)
    registry = registry or default_registry()
    live = graph.live_nodes()
    on_emit = None
    if trace is not None and trace.wants(TraceLevel.DETAILED):
        on_emit = _emit_hook(trace)
    b = PrimitiveBuilder(on_emit=on_emit)
    vectors: dict[int, BitVector] = {}
    ports: list[LogicalPort] = []
    registers: list[dict[str, Any]] = []
    input_ports = {port["node"]: port for port in graph.inputs}
    if trace is not None:
        trace.emit(
            "synthesis",
            "synthesis_begin",
            live_nodes=len(live),
            ops=dict(sorted(Counter(node["op"] for node in live).items())),
            sequential=any(node["op"] == "register" for node in live),
        )

    def report(node: dict[str, Any], phase: str, first: int, recipe: str | None = None) -> None:
        if trace is None or not trace.wants(TraceLevel.BASIC):
            return
        made = b.netlist.instances[first:]
        trace.emit(
            "synthesis",
            "ir_node_synthesized",
            ir_node=node["id"],
            op=node["op"],
            ir_type=IRType(**node["type"]).name,
            synthesis_pass=phase,
            recipe=recipe,
            group=b.group_of_ir_node(node["id"]),
            primitives=len(made),
            first_instance=first if made else None,
            last_instance=made[-1].id if made else None,
            by_kind=dict(sorted(Counter(i.kind.value for i in made).items())),
        )

    # -- Pass A: inputs and register state bits ------------------------------
    for node in live:
        node_id, op = node["id"], node["op"]
        typ = IRType(**node["type"])
        first = len(b.netlist.instances)
        if op == "input":
            port = input_ports[node_id]
            realization = interface.realize_input(port["name"], typ)
            with b.port(port["name"], "input", node_id):
                if realization.peripheral is None:
                    bits = tuple(b.input_bit(port["name"], i) for i in range(typ.width))
                else:
                    spec = PeripheralSpec(realization.peripheral, PeripheralDirection.INPUT, typ.width)
                    device = b.peripheral(spec, port["name"])
                    bits = tuple(device.output(pin) for pin in spec.pins)
            vectors[node_id] = bits
            ports.append(
                LogicalPort(
                    port["name"], "input", typ, port.get("source_name"), node_id, bits, realization.label
                )
            )
            report(node, "A", first)
        elif op == "register":
            with b.ir_node(node_id, op, typ):
                vectors[node_id] = tuple(
                    b.register_bit(init=bool((node["init"] >> i) & 1), bit=i) for i in range(typ.width)
                )
            registers.append(node)
            report(node, "A", first)

    # -- Pass B: constants and every ordinary operation, in IR order ----------
    for node in live:
        node_id, op = node["id"], node["op"]
        if op in ("input", "register"):
            continue
        typ = IRType(**node["type"])
        first = len(b.netlist.instances)
        if op == "const":
            with b.ir_node(node_id, op, typ):
                vectors[node_id] = constant_vector(b, node["value"], typ.width)
            report(node, "B", first)
            continue
        if op not in OPS:
            raise CompileError(f"IR node {node_id}: unknown operation {op!r}")
        signature = OperationSignature.from_ir_node(graph, node)
        entry = registry.resolve(signature)
        operands = tuple(vectors[arg] for arg in node["args"])
        with b.ir_node(node_id, op, typ):
            vectors[node_id] = entry.synthesize(b, signature, operands)
        report(node, "B", first, entry.name)

    # -- Pass C: output ports ----------------------------------------------------
    for port in graph.outputs:
        name, node_id = port["name"], port["node"]
        typ = IRType(**port["type"])
        source = vectors[node_id]
        realization = interface.realize_output(name, typ)
        with b.port(name, "output", node_id):
            if realization.peripheral is None:
                sinks = tuple(b.output_bit(name, i, bit) for i, bit in enumerate(source))
            else:
                spec = PeripheralSpec(realization.peripheral, PeripheralDirection.OUTPUT, typ.width)
                display: PrimitiveInstance = b.peripheral(spec, name)
                sinks = tuple(BitTerminal(display.id, pin) for pin in spec.pins)
                for bit, sink in zip(source, sinks):
                    b.connect(bit, sink)
        ports.append(LogicalPort(name, "output", typ, None, node_id, sinks, realization.label))

    # -- Pass D: register next-state (enable mux) and global clock / reset -----
    for node in registers:
        node_id = node["id"]
        typ = IRType(**node["type"])
        next_id, enable_id = node["args"]
        state, nxt, enable = vectors[node_id], vectors[next_id], vectors[enable_id][0]
        first = len(b.netlist.instances)
        with b.ir_node(node_id, "register", typ), b.scope("enable_mux", kind="helper"):
            nen = b.not_(enable, role="enable_mux_nsel")
            for i, (q, n) in enumerate(zip(state, nxt)):
                d = mux_bit(b, enable, n, q, nsel=nen, bit=i, role="enable_mux")
                d_pin, clk_pin, rst_pin = b.register_pins(q)
                b.connect(d, d_pin)
                b.connect(b.clock(), clk_pin)
                b.connect(b.reset(), rst_pin)
        report(node, "D", first)

    netlist = b.finish(validate=False)
    netlist.ports = ports
    for node in live:
        typ = IRType(**node["type"])
        netlist.ir_nodes[node["id"]] = IRNodeInfo(
            node["id"], node["op"], typ, tuple(node["args"]), _node_attrs(node)
        )
        netlist.buses[node["id"]] = LogicalBus(node["id"], node["op"], typ, vectors[node["id"]])
    netlist.graph_summary = {
        "live_nodes": len(live),
        "ops": dict(sorted(Counter(node["op"] for node in live).items())),
        "sequential": bool(registers),
        "inputs": [p["name"] for p in graph.inputs],
        "outputs": [p["name"] for p in graph.outputs],
    }
    netlist.validate(complete=True)
    if trace is not None:
        trace.emit("synthesis", "synthesis_complete", **netlist.summary())
    return netlist


def _check_unique_port_names(graph: Graph) -> None:
    """Ports are addressed by name (simulation stimulus, peripherals, the
    trace), so a repeated name would silently merge two ports."""
    for direction, ports in (("input", graph.inputs), ("output", graph.outputs)):
        names = Counter(port["name"] for port in ports)
        repeated = sorted(name for name, count in names.items() if count > 1)
        if repeated:
            raise CompileError(f"duplicate {direction} port names {repeated}: every port name must be unique")


def _node_attrs(node: dict[str, Any]) -> tuple[tuple[str, int | str], ...]:
    attrs: list[tuple[str, int | str]] = []
    for key in ("name", "value", "init"):
        if key in node:
            attrs.append((key, node[key]))
    return tuple(attrs)


def _emit_hook(trace: SynthesisTrace):
    def hook(instance: PrimitiveInstance) -> None:
        trace.emit(
            "synthesis",
            "primitive_emitted",
            level=TraceLevel.DETAILED,
            instance=instance.id,
            kind=instance.kind.value,
            **instance.provenance.to_dict(),
        )

    return hook


__all__ = ["SynthesisTrace", "default_registry", "synthesize_to_primitives"]
