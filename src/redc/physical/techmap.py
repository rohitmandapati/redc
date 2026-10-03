"""Technology mapping: lower a target-neutral IR :class:`~redc.ir.Graph` into an
unplaced :class:`~redc.physical.netlist.PhysicalNetlist`.

The mapper decides WHAT physical instances exist and HOW they connect, never
WHERE: every instance it creates has ``origin is None``.  It needs no NBT.

How the IR is shaped (this drives the algorithm):

* Connectivity is by node id: ``node["args"]`` lists producer ids in operand
  order, so one producer feeding many consumers is implicit fanout.  Fanout is
  therefore collected in a separate pass and emitted as ONE net per producer.
* Every node has exactly one result, so a producer is a single terminal.
* ``input`` / ``const`` nodes are leaves; ``graph.outputs`` are port records
  (name -> node id), possibly several pointing at one node.
* ``register`` nodes hold state; their ``args`` are ``[next, enable]`` back-edges
  and their ``init`` is the reset value.  The clock and reset are implicit in
  any graph with registers -- one global clock and one global reset.
* ``start`` / ``done`` are ordinary input / output ports; only the boundary
  policy (not the mapper) looks at port names.
* Only ``graph.live_nodes()`` are lowered: dead IR never becomes hardware.

Algorithm:

* Pass A -- instantiate.  For each live node: boundary policy for inputs,
  :class:`Constant` for constants, otherwise the node's exact
  :class:`OperationSignature` is looked up in the library and a variant chosen
  (registers get their per-instance ``init``).  Then one boundary component per
  output port, plus one :class:`ClockSource` and :class:`ResetSource` if any
  register exists.  Each node's result terminal is recorded as its driver, and
  each consumer pin -- paired with its operand by ``zip(args, operand_inputs)``,
  so declared pin order IS operand order -- is recorded as a sink of that node.
* Pass B -- connect.  For each producer in node order, one
  :meth:`PhysicalNetlist.connect` with all of its sinks; then one clock net and
  one reset net reaching every register.

The result is checked with ``validate(complete=True)`` before it is returned.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..ir import Graph, IRType
from ..parser import CompileError
from .boundary import DEFAULT_BOUNDARY_POLICY, BoundaryPolicy
from .cells import LIBRARY, Library
from .components import ClockSource, Component, Constant, Register, ResetSource
from .implementation import CompositeImplementation, OperationSignature
from .netlist import ComponentInstance, PhysicalNetlist, Terminal
from .signals import require_supported_physical_type

#: ``(node, signature, candidates) -> chosen cell``.  The seam where a later,
#: placement-aware selector replaces :func:`select_first_variant`.
VariantSelector = Callable[[dict[str, Any], OperationSignature, Sequence[Component]], Component]


def select_first_variant(
    node: dict[str, Any], signature: OperationSignature, candidates: Sequence[Component]
) -> Component:
    """v1 policy: the first candidate in the library's deterministic order."""
    return candidates[0]


def lower_to_physical(
    graph: Graph,
    *,
    library: Library = LIBRARY,
    boundary: BoundaryPolicy = DEFAULT_BOUNDARY_POLICY,
    select_variant: VariantSelector = select_first_variant,
) -> PhysicalNetlist:
    """Map ``graph`` onto physical cells and wire them into an unplaced netlist.

    Raises :class:`CompileError` if a type is not a supported Minecraft physical
    type or an operation has no direct cell in ``library``."""
    graph.validate()
    netlist = PhysicalNetlist()
    drivers: dict[int, Terminal] = {}  # IR node id -> its result terminal
    sinks: dict[int, list[Terminal]] = {}  # IR node id -> consumer terminals
    registers: list[ComponentInstance] = []
    live = graph.live_nodes()

    # -- Pass A: instantiate ------------------------------------------------
    for node in live:
        node_id, op = node["id"], node["op"]
        typ = IRType(**node["type"])
        if op == "input":
            require_supported_physical_type(typ, f"input port {node['name']!r}")
            component = _realize(
                boundary.realize_input(node["name"], typ), node["name"], typ, "input"
            )
            inst = netlist.add(component, label=node["name"])
        elif op == "const":
            require_supported_physical_type(typ, f"IR node {node_id} (const)")
            component = Constant.of(node["value"], typ)
            inst = netlist.add(component, label=f"n{node_id}")
        else:
            signature = OperationSignature.from_ir_node(graph, node)
            for t in (signature.result_type, *signature.operand_types):
                require_supported_physical_type(t, f"IR node {node_id} ({signature})")
            component = select_variant(
                node, signature, _candidates(library, node_id, signature)
            )
            if signature not in component.signatures():
                raise CompileError(
                    f"variant selector chose {component.name} for IR node {node_id}, "
                    f"which does not implement {signature}"
                )
            inst = netlist.add(
                component,
                init=node["init"] if component.is_stateful else None,
                label=f"n{node_id}",
            )
            # Declared operand-pin order is the IR operand order.
            for arg, port in zip(node["args"], component.operand_inputs, strict=True):
                sinks.setdefault(arg, []).append(Terminal(inst, port))
            if isinstance(component, Register):
                registers.append(inst)
        drivers[node_id] = Terminal(inst, component.outputs[0])

    for output in graph.outputs:
        name, typ = output["name"], IRType(**output["type"])
        require_supported_physical_type(typ, f"output port {name!r}")
        component = _realize(boundary.realize_output(name, typ), name, typ, "output")
        inst = netlist.add(component, label=name)
        sinks.setdefault(output["node"], []).append(Terminal(inst, component.inputs[0]))

    clock = reset = None
    if registers:
        clock = netlist.add(ClockSource.of(), label="clk")
        reset = netlist.add(ResetSource.of(), label="rst")

    # -- Pass B: one net per producer (fanout), then global clock / reset ---
    for node in live:
        consumers = sinks.get(node["id"])
        if consumers:  # an unused input port drives nothing
            netlist.connect(drivers[node["id"]], *consumers)
    if clock is not None and reset is not None:
        netlist.connect(
            Terminal(clock, clock.component.outputs[0]),
            *(r.terminal(Register.CLOCK) for r in registers),
        )
        netlist.connect(
            Terminal(reset, reset.component.outputs[0]),
            *(r.terminal(Register.RESET) for r in registers),
        )

    netlist.validate(complete=True)
    return netlist


def _candidates(
    library: Library, node_id: int, signature: OperationSignature
) -> tuple[Component, ...]:
    cells = library.cells(signature)
    if cells:
        return cells
    family = (
        "type_cast"
        if signature.op == "cast"
        else "register"
        if signature.op == "register"
        else "operation/primitive_gate"
    )
    detail = ""
    if any(isinstance(c, CompositeImplementation) for c in library.candidates(signature)):
        detail = " (only composite implementations exist; expansion is not supported yet)"
    raise CompileError(
        f"no physical cell variant for IR node {node_id}: {signature} "
        f"in the {family} library{detail}"
    )


def _realize(component: Component, name: str, typ: IRType, direction: str) -> Component:
    """Check that a boundary policy returned a usable port realization."""
    if direction == "input":
        ok = not component.inputs and len(component.outputs) == 1
        ports = component.outputs
    else:
        ok = not component.outputs and len(component.inputs) == 1
        ports = component.inputs
    if not ok or ports[0].dtype != typ:
        raise CompileError(
            f"boundary policy realized {direction} port {name!r} ({typ.name}) as "
            f"{component.name}, which does not carry exactly one {typ.name} {direction}"
        )
    return component
