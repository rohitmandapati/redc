"""Plain-JSON records of the mapped primitive design, shared by the replay
trace header and the final ``redc.physical-primitive.v1`` design file.

Everything is JSON-native with lowercase string enums, so the browser viewer
or a future Minecraft mod can rebuild every cell, pin and net -- and explain
every gate's provenance -- without Python or the technology library.
"""

from __future__ import annotations

from typing import Any

from ..netlist import BitTerminal, PrimitiveNetlist
from ..physical import PrimitivePhysicalNetlist


def instance_record(mapped: PrimitivePhysicalNetlist, instance_id: int) -> dict[str, Any]:
    inst = mapped.instances[instance_id]
    prim = mapped.logical.instances[instance_id]
    prov = prim.provenance
    info = mapped.logical.ir_nodes.get(prov.ir_node) if prov.ir_node is not None else None
    record: dict[str, Any] = {
        "id": inst.id,
        "kind": inst.kind.value,
        "category": inst.kind.category,
        "cell": inst.cell.name,
        "realizes": list(inst.realizes),
        "ir_node": prov.ir_node,
        "ir_op": info.op if info else None,
        "ir_type": info.type.name if info else None,
        "role": prov.role,
        "bit": prov.bit,
        "group": prov.group,
        "port": prov.port,
        "init": None if prim.init is None else int(prim.init),
        "peripheral": None if prim.peripheral is None else prim.peripheral.to_dict(),
        "origin": None if inst.origin is None else list(inst.origin),
        "orientation": None if inst.orientation is None else inst.orientation.name,
    }
    if prov.attrs:
        record["attrs"] = dict(prov.attrs)
    return record


def net_records(mapped: PrimitivePhysicalNetlist) -> list[dict[str, Any]]:
    logical: PrimitiveNetlist = mapped.logical
    buses = logical.bus_memberships()
    ports = logical.port_memberships()
    records = []
    for net in mapped.nets:
        port_bits: list[tuple[str, int]] = list(ports.get(net.driver, ()))
        for sink in net.sinks:
            port_bits.extend(ports.get(sink, ()))
        records.append(
            {
                "id": net.id,
                "role": net.role,
                "width": 1,
                "fanout": net.fanout,
                "driver": net.driver.ref(),
                "sinks": [s.ref() for s in net.sinks],
                "logical": [{"ir_node": n, "bit": b} for n, b in buses.get(net.driver, ())],
                "ports": [{"port": p, "bit": b} for p, b in port_bits],
            }
        )
    return records


def design_records(mapped: PrimitivePhysicalNetlist) -> dict[str, Any]:
    """The design as the trace header and the final file describe it."""
    logical = mapped.logical
    return {
        "library": mapped.library,
        "cells": [cell.to_dict() for cell in mapped.cells_used().values()],
        "instances": [instance_record(mapped, inst.id) for inst in mapped.instances],
        "nets": net_records(mapped),
        "groups": [g.to_dict() for g in logical.groups],
        "ports": [p.to_dict() for p in logical.ports],
        "buses": [logical.buses[k].to_dict() for k in sorted(logical.buses)],
        "ir_nodes": [logical.ir_nodes[k].to_dict() for k in sorted(logical.ir_nodes)],
        "summary": logical.summary(),
    }


def terminal_ref(terminal: BitTerminal) -> dict[str, Any]:
    return terminal.ref()


__all__ = ["design_records", "instance_record", "net_records", "terminal_ref"]
