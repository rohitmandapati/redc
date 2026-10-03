"""Cycle-level functional simulation of an unplaced PhysicalNetlist.

Propagates values through each component's ``behavior`` model -- no geometry, no
Minecraft -- so a tech-mapped netlist can be checked against the IR it came from
(:meth:`redc.ir.Graph.evaluate` / :meth:`redc.ir.Graph.step`) before placement.
Stimulus and results are keyed by each boundary instance's ``label`` (the module
port name tech-map records), so they line up with the IR's own port names.

Model: one call to :func:`step` settles all combinational logic for the current
register state (instances evaluated in a deterministic topological order;
register outputs break every cycle), then applies one clock edge.  A
combinational loop -- a cycle not through a register -- is a CompileError.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping

from ..parser import CompileError
from .components import ClockSource, Constant, Register, ResetSource
from .netlist import ComponentInstance, PhysicalNetlist, Terminal


def reset_state(netlist: PhysicalNetlist) -> dict[int, int]:
    """Register state after the global reset: instance id -> its ``init``."""
    return {
        inst.id: inst.init or 0
        for inst in netlist.instances.values()
        if inst.component.is_stateful
    }


def step(
    netlist: PhysicalNetlist,
    state: Mapping[int, int],
    inputs: Mapping[str, int],
    *,
    rst: int = 0,
) -> tuple[dict[str, int], dict[int, int]]:
    """One clock cycle: ``(outputs visible this cycle, next register state)``.

    ``inputs`` maps each source boundary label to its value; outputs map each
    sink boundary label to its (signed-interpreted) value, like ``Graph.step``.
    """
    driver_of = {sink: net.driver for net in netlist.nets.values() for sink in net.sinks}
    values: dict[Terminal, int] = {}

    def pin(inst: ComponentInstance, name: str) -> int:
        term = inst.terminal(name)
        driver = driver_of.get(term)
        if driver is None:
            raise CompileError(f"simulation: {term.describe()} is undriven")
        return values[driver]

    for inst in _topological(netlist):
        component = inst.component
        if isinstance(component, Register):
            out = component.read(state[inst.id])
        elif isinstance(component, ClockSource):
            out = {component.outputs[0].name: 0}  # edges are modelled by step()
        elif isinstance(component, ResetSource):
            out = {component.outputs[0].name: rst}
        elif isinstance(component, Constant):
            out = component.behavior({})
        elif component.is_source:  # InputPad / input Peripheral: the stimulus
            if inst.label is None or inst.label not in inputs:
                raise CompileError(f"simulation: missing input {inst.label!r}")
            port = component.outputs[0]
            out = {port.name: port.dtype.bits(inputs[inst.label])}
        else:
            out = component.behavior({p.name: pin(inst, p.name) for p in component.inputs})
        for name, value in out.items():
            values[inst.terminal(name)] = value

    outputs: dict[str, int] = {}
    next_state: dict[int, int] = {}
    for inst in netlist.instances.values():
        component = inst.component
        if component.is_sink and inst.label is not None:
            port = component.inputs[0]
            outputs[inst.label] = port.dtype.number(pin(inst, port.name))
        elif isinstance(component, Register):
            pins = {name: pin(inst, name) for name in (Register.NEXT, Register.ENABLE)}
            pins[Register.RESET] = pin(inst, Register.RESET)
            next_state[inst.id] = component.step(
                state[inst.id], pins, init=inst.init or 0
            )
    return outputs, next_state


def evaluate(netlist: PhysicalNetlist, inputs: Mapping[str, int]) -> dict[str, int]:
    """Settle a combinational netlist once and return its outputs."""
    if netlist.is_sequential:
        raise CompileError("sequential netlist holds state; use step()")
    return step(netlist, {}, inputs)[0]


def _topological(netlist: PhysicalNetlist) -> list[ComponentInstance]:
    """Instances ordered so every combinational input is computed first.  Edges
    into registers are excluded: a register's output depends only on state."""
    deps: dict[int, set[int]] = {i: set() for i in netlist.instances}
    users: dict[int, set[int]] = {i: set() for i in netlist.instances}
    for net in netlist.nets.values():
        for sink in net.sinks:
            if not sink.instance.component.is_stateful:
                deps[sink.instance.id].add(net.driver.instance.id)
                users[net.driver.instance.id].add(sink.instance.id)
    ready = [i for i, d in deps.items() if not d]
    heapq.heapify(ready)
    order: list[ComponentInstance] = []
    while ready:
        current = heapq.heappop(ready)
        order.append(netlist.instances[current])
        for user in users[current]:
            deps[user].discard(current)
            if not deps[user]:
                heapq.heappush(ready, user)
    if len(order) != len(netlist.instances):
        stuck = sorted(i for i, d in deps.items() if d)
        raise CompileError(f"simulation: combinational loop through instances {stuck}")
    return order
