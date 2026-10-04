"""Materialize a legalized primitive design as a backend-neutral
:class:`~redc.minecraft.design.MinecraftPhysicalDesign`.

The primitive backend is ONE producer of the shared representation; the
redstone simulator and timing analysis never see a primitive kind, recipe or
netlist id -- only blocks, abstract components and ports:

* every realized route block becomes a real block: dust ->
  ``minecraft:redstone_wire``, a repeater -> ``minecraft:repeater`` with its
  blockstate facing (the INPUT side) and delay setting, every support ->
  ``minecraft:stone`` (:data:`~redc.physical_primitive.redstone.SUPPORT_MATERIAL`);
* a cell WITH a block structure (``cell.blocks``) is placed block by block,
  rotated with its orientation; a lever beside an input pin / a lamp beside
  an output pin is that port bit's binding;
* a cell WITHOUT one (every default placeholder) becomes an
  :class:`~redc.minecraft.design.AbstractComponent` whose pins are its pin
  endpoint blocks and whose voxels are ``redc:abstract_block`` markers:
  gates get their truth table and declared timing arcs, register bits a
  ``dff`` with sequential timing and the reset value; pads, clock and reset
  sources and peripherals become externally driven (``source``) / observed
  (``dust``) port bits.  The design is then simulated in
  ``abstract-components`` mode -- never called block-accurate.

Annotations (block -> net id, component -> instance id) are attached for
reports only.  Logic whose output is not routed anywhere (dead gates) is
listed in ``metadata.unobserved`` instead of being simulated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..minecraft.blocks import (
    LAMP_ID,
    LEVER_ID,
    MinecraftBlock,
    abstract_block,
    redstone_wire,
    repeater,
    stone,
)
from ..minecraft.design import (
    AbstractComponent,
    ComponentPin,
    MinecraftPhysicalDesign,
    Port,
    PortBit,
    Probe,
)
from ..parser import CompileError
from .geometry import Coord
from .netlist import GATE_KINDS, BitTerminal, PrimitiveKind
from .physical import MappedInstance, PrimitivePhysicalNetlist
from .pnr.legalize import RealizedRoute
from .redstone import SUPPORT_MATERIAL, ElementKind

SOURCE_BACKEND = "physical-primitive"

#: Truth tables over the pins in order (character k: input i = bit i of k).
GATE_TABLES: dict[PrimitiveKind, str] = {
    PrimitiveKind.AND: "0001",
    PrimitiveKind.OR: "0111",
    PrimitiveKind.XOR: "0110",
    PrimitiveKind.NOT: "10",
}
CONSTANT_TABLES: dict[PrimitiveKind, str] = {PrimitiveKind.CONST0: "0", PrimitiveKind.CONST1: "1"}
#: Kinds whose (abstract) realization is a test-bench-driven source.
DRIVEN_KINDS = frozenset({PrimitiveKind.INPUT_BIT, PrimitiveKind.CLOCK_SOURCE, PrimitiveKind.RESET_SOURCE})


def component_name(instance_id: int) -> str:
    return f"i{instance_id}"


def register_probe(instance_id: int) -> str:
    return f"reg{instance_id}"


def _neighbours(coord: Coord) -> list[Coord]:
    x, y, z = coord
    return [(x, y - 1, z), (x, y + 1, z), (x, y, z - 1), (x, y, z + 1), (x - 1, y, z), (x + 1, y, z)]


def materialize_design(
    mapped: PrimitivePhysicalNetlist,
    realized: Mapping[int, RealizedRoute],
    *,
    source: Mapping[str, Any] | None = None,
) -> MinecraftPhysicalDesign:
    """The shared Minecraft representation of a fully legalized design."""
    logical = mapped.logical
    blocks: dict[Coord, MinecraftBlock] = {}
    annotations: dict[Coord, Any] = {}
    owner: dict[Coord, str] = {}

    def put(coord: Coord, block: MinecraftBlock, what: str) -> None:
        if coord in blocks and blocks[coord] != block:
            raise CompileError(
                f"materialize: {what} puts {block} at {list(coord)}, already {blocks[coord]} from {owner[coord]}"
            )
        blocks[coord] = block
        owner[coord] = what

    routed_terminals: set[BitTerminal] = set()
    for net in mapped.nets:
        if net.id in realized:
            routed_terminals.add(net.driver)
            routed_terminals.update(net.sinks)

    # -- routes -------------------------------------------------------------------
    for net_id in sorted(realized):
        route = realized[net_id]
        what = f"net {net_id}"
        for element in route.elements:
            if element.kind is ElementKind.REPEATER:
                assert element.facing is not None and element.setting is not None
                put(element.coord, repeater(element.facing.label, element.setting), what)
            else:
                put(element.coord, redstone_wire(), what)
            annotations[element.coord] = net_id
        for support in route.supports:
            put(support, MinecraftBlock(SUPPORT_MATERIAL) if SUPPORT_MATERIAL != "minecraft:stone" else stone(), what)

    # -- cells ------------------------------------------------------------------
    components: list[AbstractComponent] = []
    unobserved: list[int] = []
    materialized: list[int] = []
    for inst in mapped.instances:
        placed = inst.placed
        what = f"instance {inst.id} ({inst.cell.name})"
        world_blocks = placed.world_blocks()
        if world_blocks is not None:
            materialized.append(inst.id)
            for coord, block in world_blocks:
                put(coord, block, what)
            continue
        for coord in sorted(placed.occupied):
            put(coord, abstract_block(), what)
        component = _abstract_component(mapped, inst, routed_terminals)
        if component is None:
            if inst.kind in GATE_KINDS or inst.kind is PrimitiveKind.REGISTER_BIT:
                unobserved.append(inst.id)
            continue
        components.append(component)

    ports = _ports(mapped, blocks, routed_terminals)
    probes = tuple(
        Probe(register_probe(inst.id), inst.placed.pins["q"].position)
        for inst in mapped.instances
        if inst.kind is PrimitiveKind.REGISTER_BIT and BitTerminal(inst.id, "q") in routed_terminals
    )
    metadata = {
        "backend": SOURCE_BACKEND,
        "library": mapped.library,
        "source": dict(source) if source else None,
        "instances": len(mapped.instances),
        "nets": len(mapped.nets),
        "abstract_cells": sorted({i.cell.name for i in mapped.instances if not i.cell.materialized}),
        "materialized_cells": sorted({i.cell.name for i in mapped.instances if i.cell.materialized}),
        "materialized_instances": len(materialized),
        "unobserved": unobserved,
        "ir": logical.graph_summary,
    }
    return MinecraftPhysicalDesign(
        blocks=blocks,
        components=tuple(components),
        ports=ports,
        probes=probes,
        annotations=annotations,
        source_backend=SOURCE_BACKEND,
        metadata=metadata,
    )


def _abstract_component(
    mapped: PrimitivePhysicalNetlist, inst: MappedInstance, routed: set[BitTerminal]
) -> AbstractComponent | None:
    """The declared black box of one non-materialized gate / register / constant."""
    kind = inst.kind
    cell = inst.cell
    prim = mapped.logical.instances[inst.id]
    if kind not in GATE_KINDS and kind not in CONSTANT_TABLES and kind is not PrimitiveKind.REGISTER_BIT:
        return None  # boundary / control / peripheral: realized as ports
    outputs = [p for p in cell.pins if p.direction == "out" and BitTerminal(inst.id, p.name) in routed]
    if not outputs:
        return None  # nothing observes it (dead logic)
    if cell.timing is None:
        raise CompileError(
            f"cell {cell.name!r} (instance {inst.id}) has unknown timing: it can be neither simulated nor timed"
        )
    placed = inst.placed
    pins = []
    for pin in cell.pins:
        if pin.direction == "out" and pin not in outputs:
            continue
        placed_pin = placed.pins[pin.name]
        pins.append(ComponentPin(pin.name, pin.direction, placed_pin.position, placed_pin.strength))
    labels: dict[str, Any] = {"instance": inst.id, "kind": kind.value, "cell": cell.name}
    prov = prim.provenance
    if prov.ir_node is not None:
        labels["ir_node"] = prov.ir_node
    if prov.bit is not None:
        labels["bit"] = prov.bit
    labels["role"] = prov.role
    voxels = tuple(sorted(placed.occupied))
    label_items = tuple(sorted(labels.items()))
    name = component_name(inst.id)
    if kind is PrimitiveKind.REGISTER_BIT:
        return AbstractComponent(
            name, "dff", tuple(pins), cell.timing, init=int(bool(prim.init)), voxels=voxels, labels=label_items
        )
    table = CONSTANT_TABLES.get(kind) or GATE_TABLES[kind]
    tables = tuple((p.name, table) for p in outputs)
    return AbstractComponent(name, "combinational", tuple(pins), cell.timing, tables, voxels=voxels, labels=label_items)


def _bound_block(blocks: Mapping[Coord, MinecraftBlock], pin_cell: Coord, block_id: str) -> Coord | None:
    found = [c for c in _neighbours(pin_cell) if c in blocks and blocks[c].id == block_id]
    return found[0] if len(found) == 1 else None


def _port_bit(
    mapped: PrimitivePhysicalNetlist,
    blocks: Mapping[Coord, MinecraftBlock],
    terminal: BitTerminal,
    routed: set[BitTerminal],
    direction: str,
) -> PortBit:
    inst = mapped.instances[terminal.instance]
    pin = inst.placed.pins[terminal.pin]
    if terminal not in routed:
        if direction == "in":
            return PortBit("open", pin.position)
        raise CompileError(f"materialize: output {terminal.describe()} is not routed")
    if inst.cell.materialized:
        block_id = LEVER_ID if direction == "in" else LAMP_ID
        coord = _bound_block(blocks, pin.position, block_id)
        if coord is None:
            raise CompileError(
                f"materialized cell {inst.cell.name!r} needs exactly one {block_id} beside pin {terminal.pin!r}"
            )
        return PortBit("lever" if direction == "in" else "lamp", coord)
    if direction == "in":
        return PortBit("source", pin.position, max(1, pin.strength))
    return PortBit("dust", pin.position, pin.strength)


def _ports(
    mapped: PrimitivePhysicalNetlist, blocks: Mapping[Coord, MinecraftBlock], routed: set[BitTerminal]
) -> tuple[Port, ...]:
    logical = mapped.logical
    ports: list[Port] = []
    taken = {p.name for p in logical.ports}
    for port in logical.ports:
        direction = "in" if port.direction == "input" else "out"
        bits = tuple(_port_bit(mapped, blocks, bit, routed, direction) for bit in port.bits)
        ports.append(Port(port.name, direction, bits, port.type.signed, "data"))
    for kind, pin, role, base in (
        (PrimitiveKind.CLOCK_SOURCE, "clk", "clock", "clk"),
        (PrimitiveKind.RESET_SOURCE, "rst", "reset", "rst"),
    ):
        sources = [i for i in mapped.instances if i.kind is kind]
        if not sources:
            continue
        (inst,) = sources
        name = base
        while name in taken:
            name = "_" + name
        taken.add(name)
        bit = _port_bit(mapped, blocks, BitTerminal(inst.id, pin), routed, "in")
        ports.append(Port(name, "in", (bit,), False, role))
    return tuple(ports)


__all__ = [
    "CONSTANT_TABLES",
    "GATE_TABLES",
    "SOURCE_BACKEND",
    "component_name",
    "materialize_design",
    "register_probe",
]
