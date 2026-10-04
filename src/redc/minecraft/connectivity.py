"""Static electrical connectivity of a :class:`MinecraftPhysicalDesign`,
derived from the BLOCK GEOMETRY ALONE.

:func:`compile_world` turns blocks, abstract components and ports into an
element graph used by BOTH the event-driven simulator and static timing
analysis -- one derivation, so the two can never disagree about what is
connected to what.  Backend metadata (net ids, annotations) is never read:
if two routes physically touch, they ARE connected here.

Elements (integer ids, assigned in sorted-coordinate order so everything is
deterministic):

* **dust** -- one per ``minecraft:redstone_wire``.  Power links follow Java's
  ``RedStoneWireBlock.calculateTargetStrength``: a dust block takes
  ``strength - 1`` from dust beside it on the same level; from dust one block
  up diagonally if the block beside it is a conductor and the block above
  itself is not; from dust one block down diagonally if the block beside it is
  not a conductor.  (So on a conductor support a staircase works both ways;
  on glass only upward.)  Its *shape* (which sides it points into) follows
  ``getConnectionState``: a side is connected to dust on the same level or a
  valid staircase, to a signal source, or to a repeater's front/back; a dust
  with no connection on one axis points along the other, an isolated dust
  points all four ways.  Dust weakly powers the block below it and every
  block it points into; weak power never reaches other dust.
* **sources** -- anything that emits power: repeater fronts, torches,
  levers, redstone blocks, abstract component outputs and externally driven
  ``source`` ports (both inject into the dust at their coordinate).
* **blocks** -- conductor blocks that something can power: STRONG power
  (from a repeater facing in, a torch below, an attached lever) reaches
  adjacent dust; total power (strong or weak) is what mechanisms read.
* **readers** -- inputs of devices: a repeater's back, a torch's attachment
  block, a lamp's neighbours, abstract input pins, observed output bits and
  probes.
* **devices** -- the stateful things the simulator schedules (repeaters,
  torches, lamps, levers, constants, abstract components, observers).

Dust networks: dust links are grouped into weakly connected *components*; a
change anywhere recomputes exactly one component (dust has no delay, so a
component settles inside one game tick).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..parser import CompileError
from .behaviors import (
    BEHAVIORS,
    READ_RAW,
    UNSUPPORTED,
    BlockBehavior,
    Diagnostic,
    WorldView,
)
from .blocks import (
    ALL_DIRECTIONS,
    HORIZONTAL_DIRECTIONS,
    LAMP_ID,
    LEVER_ID,
    WIRE_ID,
    Coord,
    MinecraftBlock,
    direction_between,
    offset,
)
from .design import AbstractComponent, MinecraftPhysicalDesign

REF_DUST = 0
REF_BLOCK = 1
REF_SOURCE = 2


class SimulationError(CompileError):
    """A design the simulator refuses (or a run that failed): ``code`` is a
    ``simulation/...`` failure code, ``diagnostics`` the details."""

    def __init__(self, code: str, message: str, diagnostics: tuple[Diagnostic, ...] = ()) -> None:
        self.code = code
        self.diagnostics = diagnostics
        super().__init__(f"{code}: {message}")


@dataclass
class Device:
    index: int
    kind: str
    coord: Coord
    label: str
    behavior: BlockBehavior | None = None
    block: MinecraftBlock | None = None
    component: AbstractComponent | None = None
    #: Owned sources: block devices have one; an abstract component one per output pin.
    sources: list[int] = field(default_factory=list)
    #: Readers feeding this device (block devices: one; components: one per input pin).
    readers: list[int] = field(default_factory=list)
    static: dict[str, Any] = field(default_factory=dict)


@dataclass
class Source:
    index: int
    device: int
    coord: Coord
    on_strength: int
    #: Output pin name for an abstract component source.
    pin: str | None = None
    #: How it reaches dust: ``emit`` (a neighbouring block) or ``inject`` (at its own coordinate).
    mode: str = "emit"
    dust_out: list[int] = field(default_factory=list)
    strong_out: list[int] = field(default_factory=list)
    readers_out: list[int] = field(default_factory=list)


@dataclass
class Reader:
    index: int
    device: int
    coord: Coord
    pin: str | None = None
    refs: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class PowerBlock:
    index: int
    coord: Coord
    strong_in: list[int] = field(default_factory=list)
    weak_in: list[int] = field(default_factory=list)
    dust_out: list[int] = field(default_factory=list)
    readers_out: list[int] = field(default_factory=list)


@dataclass
class CompiledWorld:
    """The element graph (see the module docstring)."""

    design: MinecraftPhysicalDesign
    view: WorldView
    dust_coords: list[Coord]
    dust_index: dict[Coord, int]
    dust_links_in: list[list[int]]
    dust_links_out: list[list[int]]
    dust_sources: list[list[int]]
    dust_blocks: list[list[int]]
    dust_weak_out: list[list[int]]
    dust_readers: list[list[int]]
    dust_shape: list[tuple[str, ...]]
    dust_component: list[int]
    components: list[list[int]]
    sources: list[Source]
    blocks: list[PowerBlock]
    block_index: dict[Coord, int]
    readers: list[Reader]
    devices: list[Device]
    #: (port name, bit) -> device index (lever / port source / observer / lamp).
    port_devices: dict[tuple[str, int], int]
    probe_devices: dict[str, int]
    component_devices: dict[str, int]
    diagnostics: list[Diagnostic]

    def stats(self) -> dict[str, int]:
        return {
            "blocks": len(self.design.blocks),
            "dust": len(self.dust_coords),
            "dust_networks": len(self.components),
            "sources": len(self.sources),
            "power_blocks": len(self.blocks),
            "readers": len(self.readers),
            "devices": len(self.devices),
        }


def dust_shape(view: WorldView, coord: Coord) -> tuple[str, ...]:
    """The horizontal sides a dust block points into (Java ``getConnectionState``)."""
    above = offset(coord, "up")
    open_above = not view.is_conductor(above)
    connected: dict[str, bool] = {}
    for direction in HORIZONTAL_DIRECTIONS:
        side = offset(coord, direction)
        link = False
        if open_above and view.is_sturdy_top(side) and view.is_wire(offset(side, "up")):
            link = True
        else:
            block = view.block(side)
            if block is not None and view.behavior(side).connects_sideways(block, direction) or not view.is_conductor(side) and view.is_wire(offset(side, "down")):
                link = True
        connected[direction] = link
    # No connection on one axis: the dust points both ways along the other
    # (one connection -> a straight line; none at all -> a "+" cross).
    no_ns = not connected["north"] and not connected["south"]
    no_ew = not connected["east"] and not connected["west"]
    final = dict(connected)
    if not connected["west"] and no_ns:
        final["west"] = True
    if not connected["east"] and no_ns:
        final["east"] = True
    if not connected["north"] and no_ew:
        final["north"] = True
    if not connected["south"] and no_ew:
        final["south"] = True
    return tuple(d for d in HORIZONTAL_DIRECTIONS if final[d])


def dust_power_links(view: WorldView, coord: Coord) -> list[Coord]:
    """Dust blocks ``coord`` takes ``strength - 1`` from (Java ``calculateTargetStrength``)."""
    links: list[Coord] = []
    above_conducts = view.is_conductor(offset(coord, "up"))
    for direction in HORIZONTAL_DIRECTIONS:
        side = offset(coord, direction)
        if view.is_wire(side):
            links.append(side)
        if view.is_conductor(side):
            if not above_conducts and view.is_wire(offset(side, "up")):
                links.append(offset(side, "up"))
        elif view.is_wire(offset(side, "down")):
            links.append(offset(side, "down"))
    return links


def dust_emits_toward(shape: tuple[str, ...], coord: Coord, target: Coord) -> bool:
    """Whether dust at ``coord`` gives a signal to its neighbour ``target``."""
    direction = direction_between(coord, target)
    return direction == "down" or direction in shape


def compile_world(design: MinecraftPhysicalDesign) -> CompiledWorld:
    """Derive the element graph of ``design`` (raises :class:`SimulationError`
    listing every problem if any block, state or binding is unsupported)."""
    blocks = design.blocks
    view = WorldView(blocks)
    diagnostics: list[Diagnostic] = []
    coords = sorted(blocks)
    for coord in coords:
        block = blocks[coord]
        behavior = BEHAVIORS.get(block.id, UNSUPPORTED)
        if behavior is UNSUPPORTED:
            diagnostics.append(
                Diagnostic(
                    "simulation/unsupported_block",
                    f"{block} at {list(coord)} is not in the supported redstone subset "
                    f"({', '.join(sorted(b for b in BEHAVIORS if b != 'minecraft:air'))})",
                    coord,
                )
            )
            continue
        diagnostics.extend(behavior.validate(view, coord, block))

    dust_coords = [c for c in coords if blocks[c].id == WIRE_ID]
    dust_index = {c: i for i, c in enumerate(dust_coords)}
    n_dust = len(dust_coords)
    dust_sources: list[list[int]] = [[] for _ in range(n_dust)]
    dust_readers: list[list[int]] = [[] for _ in range(n_dust)]

    devices: list[Device] = []
    sources: list[Source] = []
    readers: list[Reader] = []
    #: Reader -> list of (coordinate, mode) still to resolve.
    pending_inputs: list[list[tuple[Coord, str]]] = []
    block_devices: dict[Coord, int] = {}

    def new_device(kind: str, coord: Coord, label: str, **extra: Any) -> Device:
        device = Device(len(devices), kind, coord, label, **extra)
        devices.append(device)
        return device

    def new_source(device: Device, coord: Coord, strength: int, *, pin: str | None = None, mode: str = "emit") -> Source:
        source = Source(len(sources), device.index, coord, strength, pin, mode)
        sources.append(source)
        device.sources.append(source.index)
        return source

    def new_reader(device: Device, coord: Coord, inputs: list[tuple[Coord, str]], pin: str | None = None) -> Reader:
        reader = Reader(len(readers), device.index, coord, pin)
        readers.append(reader)
        pending_inputs.append(inputs)
        device.readers.append(reader.index)
        return reader

    def dust_at(coord: Coord, what: str) -> int | None:
        index = dust_index.get(coord)
        if index is None:
            diagnostics.append(
                Diagnostic(
                    "simulation/bad_binding",
                    f"{what} at {list(coord)} must sit on redstone dust, found {blocks.get(coord) or 'air'}",
                    coord,
                )
            )
        return index

    # -- block devices (sorted coordinates) -----------------------------------
    for coord in coords:
        block = blocks[coord]
        found = BEHAVIORS.get(block.id)
        if found is None or found.device_kind is None:
            continue
        device = new_device(
            found.device_kind, coord, f"{block} at {list(coord)}", behavior=found, block=block,
            static=found.static_state(block),
        )  # fmt: skip
        block_devices[coord] = device.index
        if found.source_kind is not None:
            new_source(device, coord, 15)
        inputs = found.inputs(block, coord)
        if inputs:
            new_reader(device, coord, inputs)

    # -- abstract components ----------------------------------------------------
    component_devices: dict[str, int] = {}
    for comp in design.components:
        anchor = comp.pins[0].coord if comp.pins else (comp.voxels[0] if comp.voxels else (0, 0, 0))
        device = new_device("abstract", anchor, f"abstract {comp.function} {comp.name}", component=comp)
        component_devices[comp.name] = device.index
        for pin in comp.pins:
            if pin.direction == "out":
                source = new_source(device, pin.coord, pin.strength, pin=pin.name, mode="inject")
                index = dust_at(pin.coord, f"output pin {pin.name!r} of {comp.name}")
                if index is not None:
                    dust_sources[index].append(source.index)
                    source.dust_out.append(index)
        for pin in comp.pins:
            if pin.direction == "in":
                new_reader(device, pin.coord, [(pin.coord, READ_RAW)], pin=pin.name)

    # -- ports and probes ------------------------------------------------------
    port_devices: dict[tuple[str, int], int] = {}
    for port in design.ports:
        for bit_index, bit in enumerate(port.bits):
            key = (port.name, bit_index)
            what = f"bit {bit_index} of port {port.name!r}"
            if bit.kind in ("lever", "lamp"):
                bound = block_devices.get(bit.coord)
                want = LEVER_ID if bit.kind == "lever" else LAMP_ID
                if bound is None or blocks[bit.coord].id != want:
                    diagnostics.append(Diagnostic("simulation/bad_binding", f"{what} must be a {want}", bit.coord))
                    continue
                port_devices[key] = bound
            elif bit.kind == "open":
                port_devices[key] = new_device("open", bit.coord, f"{what} (not connected)").index
            elif bit.kind == "source":
                device = new_device("port_source", bit.coord, f"{what} (externally driven source)")
                source = new_source(device, bit.coord, bit.strength, mode="inject")
                index = dust_at(bit.coord, what)
                if index is not None:
                    dust_sources[index].append(source.index)
                    source.dust_out.append(index)
                port_devices[key] = device.index
            else:  # observed dust
                device = new_device("observer", bit.coord, what, static={"threshold": bit.strength})
                new_reader(device, bit.coord, [(bit.coord, READ_RAW)])
                port_devices[key] = device.index
    probe_devices: dict[str, int] = {}
    for probe in design.probes:
        device = new_device("probe", probe.coord, f"probe {probe.name!r}", static={"threshold": probe.threshold})
        new_reader(device, probe.coord, [(probe.coord, READ_RAW)])
        probe_devices[probe.name] = device.index

    # -- block sources: emission into dust, strong power into conductors ---------
    strong_in: dict[Coord, list[int]] = {}
    for source in sources:
        device = devices[source.device]
        if source.mode != "emit":
            continue
        assert device.behavior is not None and device.block is not None
        for direction in ALL_DIRECTIONS:
            target = offset(source.coord, direction)
            if device.behavior.emits_toward(device.block, source.coord, target) and target in dust_index:
                dust_sources[dust_index[target]].append(source.index)
                source.dust_out.append(dust_index[target])
            if device.behavior.strongly_powers(device.block, source.coord, target) and view.is_conductor(target):
                strong_in.setdefault(target, []).append(source.index)

    # -- dust: shape, links, weak power ---------------------------------------
    dust_shape_list: list[tuple[str, ...]] = []
    dust_links_in: list[list[int]] = []
    weak_in: dict[Coord, list[int]] = {}
    for index, coord in enumerate(dust_coords):
        shape = dust_shape(view, coord)
        dust_shape_list.append(shape)
        dust_links_in.append(sorted({dust_index[c] for c in dust_power_links(view, coord)}))
        below = offset(coord, "down")
        if view.is_conductor(below):
            weak_in.setdefault(below, []).append(index)
        for direction in shape:
            side = offset(coord, direction)
            if view.is_conductor(side):
                weak_in.setdefault(side, []).append(index)
    dust_links_out: list[list[int]] = [[] for _ in range(n_dust)]
    for index, links in enumerate(dust_links_in):
        for other in links:
            dust_links_out[other].append(index)

    power_coords = sorted(set(strong_in) | set(weak_in))
    power_blocks = [PowerBlock(i, c, sorted(strong_in.get(c, ())), sorted(weak_in.get(c, ()))) for i, c in enumerate(power_coords)]
    block_index = {b.coord: b.index for b in power_blocks}
    for power in power_blocks:
        for s in power.strong_in:
            sources[s].strong_out.append(power.index)
    dust_blocks: list[list[int]] = [[] for _ in range(n_dust)]
    dust_weak_out: list[list[int]] = [[] for _ in range(n_dust)]
    for power in power_blocks:
        for d in power.weak_in:
            dust_weak_out[d].append(power.index)
        if power.strong_in:
            for direction in ALL_DIRECTIONS:
                neighbour = dust_index.get(offset(power.coord, direction))
                if neighbour is not None:
                    dust_blocks[neighbour].append(power.index)
                    power.dust_out.append(neighbour)

    # -- readers: resolve what each input coordinate gives -------------------
    for reader, inputs in zip(readers, pending_inputs):
        refs: list[tuple[int, int]] = []
        for coord, mode in inputs:
            if coord in dust_index:
                d = dust_index[coord]
                if mode == READ_RAW or dust_emits_toward(dust_shape_list[d], coord, reader.coord):
                    refs.append((REF_DUST, d))
                    dust_readers[d].append(reader.index)
            elif coord in block_index:
                refs.append((REF_BLOCK, block_index[coord]))
                power_blocks[block_index[coord]].readers_out.append(reader.index)
            elif coord in block_devices:
                device = devices[block_devices[coord]]
                assert device.behavior is not None and device.block is not None
                if device.sources and device.behavior.emits_toward(device.block, coord, reader.coord):
                    s = device.sources[0]
                    refs.append((REF_SOURCE, s))
                    sources[s].readers_out.append(reader.index)
        reader.refs = refs

    # -- dust networks ----------------------------------------------------------
    parent = list(range(n_dust))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for index, links in enumerate(dust_links_in):
        for other in links:
            a, b = find(index), find(other)
            if a != b:
                parent[max(a, b)] = min(a, b)
    roots: dict[int, int] = {}
    components: list[list[int]] = []
    dust_component = [0] * n_dust
    for index in range(n_dust):
        root = find(index)
        if root not in roots:
            roots[root] = len(components)
            components.append([])
        dust_component[index] = roots[root]
        components[roots[root]].append(index)

    errors = [d for d in diagnostics if d.code != "simulation/warning"]
    if errors:
        first = errors[0]
        more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
        raise SimulationError(first.code, first.message + more, tuple(errors))
    return CompiledWorld(
        design=design,
        view=view,
        dust_coords=dust_coords,
        dust_index=dust_index,
        dust_links_in=dust_links_in,
        dust_links_out=dust_links_out,
        dust_sources=[sorted(set(s)) for s in dust_sources],
        dust_blocks=dust_blocks,
        dust_weak_out=dust_weak_out,
        dust_readers=dust_readers,
        dust_shape=dust_shape_list,
        dust_component=dust_component,
        components=components,
        sources=sources,
        blocks=power_blocks,
        block_index=block_index,
        readers=readers,
        devices=devices,
        port_devices=port_devices,
        probe_devices=probe_devices,
        component_devices=component_devices,
        diagnostics=diagnostics,
    )


__all__ = [
    "REF_BLOCK",
    "REF_DUST",
    "REF_SOURCE",
    "CompiledWorld",
    "Device",
    "PowerBlock",
    "Reader",
    "SimulationError",
    "Source",
    "compile_world",
    "dust_emits_toward",
    "dust_power_links",
    "dust_shape",
]
