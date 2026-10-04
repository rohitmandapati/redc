"""Deterministic, hierarchical block-level placement of mapped primitives.

Placement decides ``instance -> origin + orientation`` and nothing else: it
never changes Boolean semantics or connectivity.  It works on arbitrary
primitive connectivity and uses provenance only as a locality hint:

1. **Blocks.**  Every primitive belongs to the block of the IR node that
   generated it (its gates, per-node constants, register bits and their enable
   muxes).  All external sources (input bits, input peripherals, clock, reset)
   form one ``inputs`` block and all output bits / output peripherals one
   ``outputs`` block.  Keeping a block together keeps gates of one IR operation
   -- and, through the ordering below, of one bit slice -- local, so a wide
   design does not degenerate into a few enormous columns of unrelated gates.
2. **Super-columns** order blocks along +x by IR depth: ``inputs`` first, then
   every operation one past its deepest operand (register outputs are sources,
   so sequential loops vanish; a register block comes after its next-state
   logic), constants one before their earliest consumer, ``outputs`` last.
3. **Local columns.**  Inside a block, primitives are levelized by their
   longest in-block path (edges out of register bits cut), giving the block's
   local columns.  Global column = (super-column, local column), so signal flow
   still runs along +x and every column is separated by ``channel_width`` free
   blocks: the routing channels.
4. **Ordering along z.**  Blocks of one super-column stack along z in the
   order of the barycentre of their already placed drivers.  Inside a block,
   each local column is ordered by its drivers' barycentre, ties by bit index
   then id, so sibling bit slices stay predictably ordered; the ``inputs``
   block interleaves ports bit by bit (``a[0], b[0], a[1], b[1], ...``).
5. **Packing.**  Each cell's footprint INCLUDING its keep-out is stacked along
   z with ``component_spacing`` extra free blocks.  A slot that
   :meth:`BlockGrid.check_placement` rejects is probed again one block further
   along z; every probe and rejection reason goes to the trace.

Cells keep their identity orientation (inputs face west, outputs east), which
matches the signal flow; the orientation is still recorded for every instance.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from ...parser import CompileError
from ...tracing import TraceLevel
from ..geometry import IDENTITY, Bounds, Coord, coord_list
from ..grid import BlockGrid, PinSite
from ..netlist import BitTerminal, PeripheralDirection, PrimitiveKind
from ..physical import MappedInstance, PrimitivePhysicalNetlist
from .config import AttemptGeometry
from .trace import PrimitiveTraceRecorder

#: Slots tried along z for one instance before placement gives up.
MAX_SLOT_PROBES = 4096

INPUTS: tuple = ("inputs",)
OUTPUTS: tuple = ("outputs",)
BlockKey = tuple

_SOURCES = frozenset(
    {PrimitiveKind.INPUT_BIT, PrimitiveKind.CLOCK_SOURCE, PrimitiveKind.RESET_SOURCE}
)
_CONSTANTS = frozenset({PrimitiveKind.CONST0, PrimitiveKind.CONST1})


class PlacementError(CompileError):
    """A design that cannot be placed under the current geometry."""


@dataclass
class PlacedPrimitiveDesign:
    """The placement stage artifact: every mapped instance has an origin and
    orientation, and the grid holds every body, keep-out and pin block."""

    mapped: PrimitivePhysicalNetlist
    grid: BlockGrid
    geometry: AttemptGeometry
    columns: dict[int, int]
    column_x: dict[int, int] = field(default_factory=dict)
    blocks: dict[int, BlockKey] = field(default_factory=dict)
    probes: int = 0

    @property
    def bounds(self) -> Bounds | None:
        """Bounding box of every placed body, keep-out and pin block."""
        return self.grid.placement_bounds


@dataclass(frozen=True)
class _Layout:
    blocks: dict[int, BlockKey]  # instance -> block
    rank: dict[BlockKey, int]  # block -> super-column
    level: dict[int, int]  # instance -> local column inside its block
    column: dict[int, int]  # instance -> global column

    @property
    def super_columns(self) -> list[int]:
        return sorted(set(self.rank.values()))


def block_of(mapped: PrimitivePhysicalNetlist, inst: MappedInstance) -> BlockKey:
    """The placement block of one instance (see the module docstring)."""
    kind = inst.kind
    prim = mapped.logical.instances[inst.id]
    if kind in _SOURCES:
        return INPUTS
    if kind is PrimitiveKind.OUTPUT_BIT:
        return OUTPUTS
    if kind is PrimitiveKind.PERIPHERAL:
        assert prim.peripheral is not None
        return INPUTS if prim.peripheral.direction is PeripheralDirection.INPUT else OUTPUTS
    node = prim.provenance.ir_node
    return INPUTS if node is None else ("node", node)


def _layout(mapped: PrimitivePhysicalNetlist) -> _Layout:
    logical = mapped.logical
    blocks = {inst.id: block_of(mapped, inst) for inst in mapped.instances}

    # -- IR-level depth of every node block -----------------------------------
    infos = logical.ir_nodes
    depth: dict[int, int] = {}
    for node_id in sorted(infos):
        info = infos[node_id]
        if info.op in ("input", "const", "register"):
            depth[node_id] = 0
        else:
            depth[node_id] = 1 + max((depth.get(a, 0) for a in info.args), default=0)
    for node_id in sorted(infos):
        info = infos[node_id]
        if info.op == "register":
            depth[node_id] = 1 + max((depth.get(a, 0) for a in info.args), default=0)
    consumers: dict[int, list[int]] = {}
    for node_id, info in infos.items():
        for arg in info.args:
            consumers.setdefault(arg, []).append(node_id)
    for node_id in sorted(infos):
        if infos[node_id].op == "const":
            uses = [depth[c] for c in consumers.get(node_id, ()) if infos[c].op != "const"]
            depth[node_id] = max(0, min(uses) - 1) if uses else 0
    rank: dict[BlockKey, int] = {}
    for block in set(blocks.values()):
        if block == INPUTS:
            rank[block] = 0
        elif block != OUTPUTS:
            rank[block] = depth.get(block[1], 0) + 1
    rank[OUTPUTS] = 1 + max(rank.values(), default=0)
    if OUTPUTS not in set(blocks.values()):
        del rank[OUTPUTS]

    # -- local levels inside each block -----------------------------------------
    preds: dict[int, set[int]] = {i: set() for i in blocks}
    for net in mapped.nets:
        driver = net.driver.instance
        if mapped.instances[driver].kind is PrimitiveKind.REGISTER_BIT:
            continue  # register outputs are sources: loops vanish
        for sink in net.sinks:
            if sink.instance != driver and blocks[sink.instance] == blocks[driver]:
                preds[sink.instance].add(driver)
    succs: dict[int, list[int]] = {i: [] for i in blocks}
    indegree = {i: len(p) for i, p in preds.items()}
    for i, p in preds.items():
        for d in p:
            succs[d].append(i)
    level: dict[int, int] = {}
    ready = [i for i, n in indegree.items() if n == 0]
    heapq.heapify(ready)
    while ready:
        current = heapq.heappop(ready)
        level[current] = 1 + max((level[p] for p in preds[current]), default=-1)
        for succ in succs[current]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                heapq.heappush(ready, succ)
    stuck = sorted(i for i in blocks if i not in level)
    if stuck:
        raise PlacementError(f"combinational loop through primitives {stuck[:20]}")

    # -- global columns -------------------------------------------------------------
    width: dict[int, int] = {}
    for i, block in blocks.items():
        r = rank[block]
        width[r] = max(width.get(r, 0), level[i] + 1)
    # A constant node's block feeds only later super-columns: right-align it
    # so its wires start next to its consumers instead of crossing channels.
    for i, block in blocks.items():
        if block[0] == "node" and infos[block[1]].op == "const":
            level[i] = width[rank[block]] - 1
    offset: dict[int, int] = {}
    total = 0
    for r in sorted(width):
        offset[r] = total
        total += width[r]
    column = {i: offset[rank[block]] + level[i] for i, block in blocks.items()}
    return _Layout(blocks, rank, level, column)


def placement_columns(mapped: PrimitivePhysicalNetlist) -> dict[int, int]:
    """Global column index of every instance: sources in column 0, outputs in
    the last column, signal flowing toward higher columns."""
    return _layout(mapped).column


def _source_key(mapped: PrimitivePhysicalNetlist, port_order: dict[str, int], inst: MappedInstance) -> tuple:
    prov = mapped.logical.instances[inst.id].provenance
    if inst.kind in (PrimitiveKind.CLOCK_SOURCE, PrimitiveKind.RESET_SOURCE):
        return (-1, -1, inst.id)
    bit = prov.bit if prov.bit is not None else 0
    return (bit, port_order.get(prov.port or "", len(port_order)), inst.id)


def place_design(
    mapped: PrimitivePhysicalNetlist,
    grid: BlockGrid,
    geometry: AttemptGeometry,
    trace: PrimitiveTraceRecorder | None = None,
) -> PlacedPrimitiveDesign:
    """Place every mapped instance (sets ``origin`` / ``orientation``)."""
    layout = _layout(mapped)
    placed = PlacedPrimitiveDesign(mapped, grid, geometry, layout.column, blocks=layout.blocks)
    net_of = mapped.net_of_terminal()
    drivers: dict[int, list[int]] = {inst.id: [] for inst in mapped.instances}
    for net in mapped.nets:
        for sink in net.sinks:
            if net.driver.instance != sink.instance:
                drivers[sink.instance].append(net.driver.instance)
    port_order = {port.name: index for index, port in enumerate(mapped.logical.ports)}
    detailed = trace is not None and trace.wants(TraceLevel.DETAILED)
    footprint = {inst.id: inst.cell.oriented(IDENTITY).bounds for inst in mapped.instances}

    # members[(rank, block, level)] and the width of every global column
    members: dict[tuple[int, BlockKey, int], list[MappedInstance]] = {}
    col_width: dict[int, int] = {}
    for inst in mapped.instances:
        block = layout.blocks[inst.id]
        member = (layout.rank[block], block, layout.level[inst.id])
        members.setdefault(member, []).append(inst)
        col = layout.column[inst.id]
        col_width[col] = max(col_width.get(col, 0), footprint[inst.id].dims[0])
    x = 0
    for col in sorted(col_width):
        placed.column_x[col] = x
        x += col_width[col] + geometry.channel_width
    if trace:
        by_column: dict[int, list[int]] = {}
        block_sizes: dict[BlockKey, int] = {}
        for inst in mapped.instances:
            by_column.setdefault(layout.column[inst.id], []).append(inst.id)
            block = layout.blocks[inst.id]
            block_sizes[block] = block_sizes.get(block, 0) + 1
        trace.emit(
            "placement",
            "placement_begin",
            attempt=geometry.attempt,
            component_spacing=geometry.component_spacing,
            channel_width=geometry.channel_width,
            columns=[{"column": c, "x": placed.column_x[c], "instances": by_column[c]} for c in sorted(by_column)],
            blocks=[
                {
                    "block": _block_label(b),
                    "super_column": layout.rank[b],
                    "instances": block_sizes.get(b, 0),
                }
                for b in sorted(layout.rank, key=lambda b: (layout.rank[b], _block_label(b)))
            ],
        )

    spacing = geometry.component_spacing
    block_gap = max(1, geometry.channel_width // 2)
    z_center: dict[int, float] = {}
    last_bounds: Bounds | None = None
    levels_of: dict[tuple[int, BlockKey], list[int]] = {}
    for (r, b, lvl) in members:
        levels_of.setdefault((r, b), []).append(lvl)
    insts_of: dict[BlockKey, list[int]] = {}
    for inst_id, block in layout.blocks.items():
        insts_of.setdefault(block, []).append(inst_id)

    def barycentre(inst_id: int) -> float | None:
        zs = [z_center[d] for d in drivers[inst_id] if d in z_center]
        return sum(zs) / len(zs) if zs else None

    def block_order(block: BlockKey) -> tuple:
        zs = [bc for bc in map(barycentre, insts_of[block]) if bc is not None]
        return (sum(zs) / len(zs) if zs else 0.0, _block_label(block))

    for rank in layout.super_columns:
        levels = {b: lvls for (r, b), lvls in levels_of.items() if r == rank}
        blocks_here = sorted(levels, key=block_order)
        heights = {
            b: max(
                sum(footprint[i.id].dims[2] for i in members[(rank, b, lvl)])
                + spacing * (len(members[(rank, b, lvl)]) - 1)
                for lvl in levels[b]
            )
            for b in blocks_here
        }
        cursor = -((sum(heights.values()) + block_gap * (len(blocks_here) - 1)) // 2)
        for block in blocks_here:
            block_end = cursor
            for lvl in sorted(levels[block]):
                insts = members[(rank, block, lvl)]
                if block == INPUTS:
                    insts = sorted(insts, key=lambda i: _source_key(mapped, port_order, i))
                else:

                    def key(inst: MappedInstance) -> tuple:
                        prov = mapped.logical.instances[inst.id].provenance
                        bc = barycentre(inst.id)
                        return (bc is None, bc or 0.0, prov.bit if prov.bit is not None else -1, inst.id)

                    insts = sorted(insts, key=key)
                z = cursor
                col_x = placed.column_x[layout.column[insts[0].id]]
                for inst in insts:
                    box = footprint[inst.id]
                    origin = _probe(inst, box, col_x, z, grid, net_of, trace, detailed, placed)
                    z = origin[2] + box.hi[2] + 1 + spacing
                    cell_bounds = inst.placed.bounds
                    z_center[inst.id] = (cell_bounds.lo[2] + cell_bounds.hi[2]) / 2
                    if trace:
                        trace.emit(
                            "placement",
                            "component_placed",
                            instance=inst.id,
                            column=layout.column[inst.id],
                            block=_block_label(block),
                            origin=coord_list(origin),
                            orientation=IDENTITY.name,
                            bounds=cell_bounds.to_dict(),
                        )
                        bounds = grid.placement_bounds
                        if bounds is not None and bounds != last_bounds:
                            trace.emit("placement", "design_bounds_changed", **bounds.to_dict())
                            last_bounds = bounds
                block_end = max(block_end, z)
            cursor = block_end + block_gap

    if trace:
        bounds = grid.placement_bounds
        trace.emit(
            "placement",
            "placement_complete",
            attempt=geometry.attempt,
            component_count=len(mapped.instances),
            component_voxels=len(grid.body),
            keepout_voxels=len(grid.keepout),
            probes=placed.probes,
            bounds=bounds.to_dict() if bounds else None,
        )
    return placed


def _block_label(block: BlockKey) -> str:
    return f"n{block[1]}" if block[0] == "node" else block[0]


def _probe(
    inst: MappedInstance,
    box: Bounds,
    x: int,
    z: int,
    grid: BlockGrid,
    net_of: dict[BitTerminal, int],
    trace: PrimitiveTraceRecorder | None,
    detailed: bool,
    placed: PlacedPrimitiveDesign,
) -> Coord:
    oriented = inst.cell.oriented(IDENTITY)
    for _ in range(MAX_SLOT_PROBES):
        origin = (x - box.lo[0], -box.lo[1], z - box.lo[2])
        candidate = oriented.translate(origin)
        placed.probes += 1
        if detailed:
            assert trace is not None
            trace.emit(
                "placement",
                "component_place_attempt",
                level=TraceLevel.DETAILED,
                instance=inst.id,
                origin=coord_list(origin),
                orientation=IDENTITY.name,
                voxels=[coord_list(c) for c in sorted(candidate.occupied)],
                keepout=len(candidate.keepout),
                bounds=candidate.bounds.to_dict(),
            )
        reason = grid.check_placement(
            candidate.occupied,
            candidate.keepout,
            [(p.position, p.approach) for p in candidate.pins.values()],
        )
        if reason is None:
            inst.place(origin, IDENTITY)
            grid.place(
                inst.id,
                candidate.occupied,
                candidate.keepout,
                [
                    PinSite(
                        p.position,
                        inst.id,
                        p.name,
                        p.direction,
                        p.facing,
                        p.strength,
                        net_of.get(BitTerminal(inst.id, p.name)),
                    )
                    for p in candidate.pins.values()
                ],
            )
            return origin
        if detailed:
            assert trace is not None
            trace.emit(
                "placement",
                "component_place_rejected",
                level=TraceLevel.DETAILED,
                instance=inst.id,
                origin=coord_list(origin),
                orientation=IDENTITY.name,
                reason=reason,
            )
        z += 1
    raise PlacementError(f"no legal slot for primitive {inst.id} ({inst.cell.name}) near x={x}")


__all__ = [
    "INPUTS",
    "MAX_SLOT_PROBES",
    "OUTPUTS",
    "PlacedPrimitiveDesign",
    "PlacementError",
    "block_of",
    "place_design",
    "placement_columns",
]
