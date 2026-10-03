"""Plain-JSON records describing a netlist's components, instances and nets.

Shared by the replay trace and the final physical-design file.  Everything here
is JSON-native (objects, arrays, numbers, strings, booleans, null) with stable
lowercase string enums, so browser JavaScript and a future Java Minecraft mod can
rebuild every box and pin without the YAML library or any Python object.
"""

from __future__ import annotations

from typing import Any

from ...ir import IRType
from ..components import (
    ClockSource,
    Component,
    Constant,
    InputPad,
    Operation,
    OutputPad,
    Peripheral,
    Port,
    PrimitiveGate,
    Register,
    ResetSource,
    TypeCast,
    Wiring,
)
from ..netlist import Net, PhysicalNetlist, Terminal
from ..signals import signal_layout
from .geometry import coord_list

#: (class, family, category) -- most specific first.  ``family`` names the cell
#: family; ``category`` groups families for display (compute / register /
#: routing / boundary / control / peripheral).
_FAMILIES: tuple[tuple[type[Component], str, str], ...] = (
    (Peripheral, "peripheral", "peripheral"),
    (Register, "register", "register"),
    (TypeCast, "type_cast", "compute"),
    (PrimitiveGate, "primitive_gate", "compute"),
    (Operation, "operation", "compute"),
    (Wiring, "wiring", "routing"),
    (InputPad, "input_pad", "boundary"),
    (OutputPad, "output_pad", "boundary"),
    (Constant, "constant", "boundary"),
    (ClockSource, "clock_source", "control"),
    (ResetSource, "reset_source", "control"),
)


def family_of(component: Component) -> tuple[str, str]:
    """``(family, category)`` strings for a component."""
    for cls, family, category in _FAMILIES:
        if isinstance(component, cls):
            return family, category
    return "component", "compute"


def type_record(typ: IRType) -> dict[str, Any]:
    return {"name": typ.name, "width": typ.width, "signed": typ.signed, "boolean": typ.boolean}


def port_record(port: Port) -> dict[str, Any]:
    return {
        "name": port.name,
        "direction": port.direction.value,
        "type": type_record(port.dtype),
        "layout": signal_layout(port.dtype).to_dict(),
        "face": port.face.name.lower(),
        "normal": list(port.face.normal),
        "offset": list(port.offset),
    }


def terminal_record(terminal: Terminal) -> dict[str, Any]:
    return {"instance": terminal.instance.id, "port": terminal.port.name}


class ComponentIds:
    """Stable string ids for the distinct component definitions of a netlist:
    the component name, suffixed ``#2``, ``#3``, ... only if two different
    definitions share a name."""

    def __init__(self, netlist: PhysicalNetlist) -> None:
        self._ids: dict[Component, str] = {}
        names: dict[str, int] = {}
        for inst in netlist.instances.values():
            component = inst.component
            if component in self._ids:
                continue
            count = names.get(component.name, 0) + 1
            names[component.name] = count
            self._ids[component] = component.name if count == 1 else f"{component.name}#{count}"

    def __getitem__(self, component: Component) -> str:
        return self._ids[component]

    def items(self):
        return self._ids.items()


def component_record(component: Component, def_id: str) -> dict[str, Any]:
    family, category = family_of(component)
    record: dict[str, Any] = {
        "id": def_id,
        "name": component.name,
        "family": family,
        "category": category,
        "op": getattr(component, "op", None) or None,
        "dim": list(component.dim),
        "latency": component.latency,
        "stateful": component.is_stateful,
        "combinational": component.is_combinational,
        "nbt": component.nbt,
        "signatures": [str(s) for s in component.signatures()],
        "ports": [port_record(p) for p in component.ports],
        "peripheral": None,
    }
    if isinstance(component, Peripheral) and component.direction is not None:
        record["peripheral"] = {"kind": component.kind, "direction": component.direction.value}
    if isinstance(component, Constant):
        record["value"] = component.value
    return record


def net_role(net: Net) -> str:
    """``clock`` / ``reset`` for the global control nets, else ``data``."""
    source = net.driver.instance.component
    if isinstance(source, ClockSource):
        return "clock"
    if isinstance(source, ResetSource):
        return "reset"
    return "data"


def design_records(netlist: PhysicalNetlist) -> dict[str, list[dict[str, Any]]]:
    """The design as the trace header carries it: unique component definitions,
    instances (with their current origin) and nets, all in id order."""
    ids = ComponentIds(netlist)
    instances = [
        {
            "id": inst.id,
            "component": ids[inst.component],
            "label": inst.label,
            "category": family_of(inst.component)[1],
            "init": inst.init,
            "origin": coord_list(inst.origin) if inst.origin is not None else None,
        }
        for inst in sorted(netlist.instances.values(), key=lambda i: i.id)
    ]
    nets = [
        {
            "id": net.id,
            "role": net_role(net),
            "type": type_record(net.dtype),
            "layout": net.layout.to_dict(),
            "fanout": net.fanout,
            "driver": terminal_record(net.driver),
            "sinks": [terminal_record(s) for s in net.sinks],
        }
        for net in sorted(netlist.nets.values(), key=lambda n: n.id)
    ]
    return {
        "components": [component_record(c, def_id) for c, def_id in ids.items()],
        "instances": instances,
        "nets": nets,
    }
