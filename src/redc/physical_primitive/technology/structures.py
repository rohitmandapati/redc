"""Reference BLOCK-LEVEL implementations of some primitive cells.

These give a few placeholder footprints real Minecraft blocks, so a design
built only from them is simulated ``block-accurate`` -- end to end through
the compiler -- and their timing is MEASURED by simulation
(:func:`redc.minecraft.characterize.characterize_cell`), not declared.

Each structure keeps its placeholder cell's exact footprint (voxels, pins,
keep-out), so placement and routing are unchanged.  They avoid every
unmodelled mechanic (no quasi-connectivity, no same-tick races):

* ``NOT``  -- input dust -> stone block -> wall torch -> output dust (1 rt);
* ``OR``   -- each input through its own repeater (diode isolation, so one
  input can never drive the other's net) into a shared dust merge: output
  strength 13, 1 rt;
* ``INPUT_BIT`` / ``CLOCK_SOURCE`` / ``RESET_SOURCE`` / the 1-bit lever
  peripheral -- a floor lever beside the output pin (the port drives it);
* ``OUTPUT_BIT`` / the 8-bit display -- a redstone lamp the pin dust points into;
* ``CONST1`` -- a block of redstone; ``CONST0`` -- an inert stone anchor.

AND, XOR and the register bit have NO reference structure yet: a library
built with :func:`reference_library` still uses abstract placeholders for
them.  Every cell stays ``placeholder=True``: none is verified in game.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from ...minecraft.blocks import (
    MinecraftBlock,
    air,
    lever,
    redstone_block,
    redstone_lamp,
    redstone_wire,
    repeater,
    stone,
    wall_torch,
)
from ...minecraft.characterize import CellPin, characterize_cell
from ...parser import CompileError
from ..geometry import Coord
from ..netlist import PrimitiveKind
from .cells import PrimitiveCell
from .library import LIBRARY_NAME, PrimitiveTechnologyLibrary, placeholder_cells

REFERENCE_LIBRARY_NAME = "redc-primitive-reference-v1"

Structure = dict[Coord, MinecraftBlock]


def _not(cell: PrimitiveCell) -> Structure:
    return {(1, 1, 0): stone(), (2, 1, 0): wall_torch("east")}


def _or(cell: PrimitiveCell) -> Structure:
    return {
        (1, 1, 0): repeater("east", 1),
        (1, 1, 2): repeater("east", 1),
        (1, 1, 1): air(),
        (2, 1, 0): redstone_wire(),
        (2, 1, 1): redstone_wire(),
        (2, 1, 2): redstone_wire(),
    }


def _lever_beside(pin: str) -> Callable[[PrimitiveCell], Structure]:
    def build(cell: PrimitiveCell) -> Structure:
        p = cell.pin(pin)
        at = (p.position[0] - 1, p.position[1], p.position[2])  # the body block west of the east-facing pin
        return {at: lever("floor", "east")}

    return build


def _lamps(cell: PrimitiveCell) -> Structure:
    return {(p.position[0] + 1, p.position[1], p.position[2]): redstone_lamp() for p in cell.pins}


def _redstone_block(cell: PrimitiveCell) -> Structure:
    return {(0, 1, 0): redstone_block()}


def _anchor(cell: PrimitiveCell) -> Structure:
    return {(0, 1, 0): stone()}


#: Primitive kind (or peripheral kind) -> the body blocks; every other voxel is stone.
STRUCTURES: dict[object, Callable[[PrimitiveCell], Structure]] = {
    PrimitiveKind.NOT: _not,
    PrimitiveKind.OR: _or,
    PrimitiveKind.INPUT_BIT: _lever_beside("y"),
    PrimitiveKind.CLOCK_SOURCE: _lever_beside("clk"),
    PrimitiveKind.RESET_SOURCE: _lever_beside("rst"),
    PrimitiveKind.OUTPUT_BIT: _lamps,
    PrimitiveKind.CONST1: _redstone_block,
    PrimitiveKind.CONST0: _anchor,
    "lever": _lever_beside("b0"),
    "2-dig-7-seg": _lamps,
}


def structure_blocks(cell: PrimitiveCell) -> tuple[tuple[Coord, MinecraftBlock], ...] | None:
    """The reference block structure of ``cell`` (``None`` if it has none)."""
    key: object = cell.peripheral[0] if cell.peripheral is not None else cell.kind
    build = STRUCTURES.get(key)
    if build is None:
        return None
    body = build(cell)
    occupied = {c for c, _ in cell.voxels}
    if set(body) - occupied:
        raise CompileError(f"{cell.name}: reference structure outside the footprint {sorted(set(body) - occupied)}")
    return tuple((c, body.get(c, stone())) for c in sorted(occupied))


def cell_pins(cell: PrimitiveCell) -> list[CellPin]:
    return [CellPin(p.name, p.direction, p.position, p.facing.label, p.strength) for p in cell.pins]


GATE_TABLES = {PrimitiveKind.NOT: "10", PrimitiveKind.OR: "0111"}


def reference_cell(cell: PrimitiveCell, *, characterize: bool = True) -> PrimitiveCell:
    """``cell`` with its reference structure and -- for gates -- the timing
    MEASURED by simulating that structure (raises if the structure does not
    compute the cell's function or misses its promised output strength)."""
    blocks = structure_blocks(cell)
    if blocks is None:
        return cell
    timing = cell.timing
    if characterize and cell.kind in GATE_TABLES:
        result = characterize_cell(cell.name, dict(blocks), cell_pins(cell))
        out = next(p for p in cell.pins if p.direction == "out")
        problems = result.check({out.name: GATE_TABLES[cell.kind]}, None, {out.name: out.strength})
        if problems:
            raise CompileError(f"reference structure of {cell.name} fails characterization: {'; '.join(problems)}")
        timing = result.timing
    name = cell.name.replace("_placeholder", "_reference")
    return dataclasses.replace(
        cell,
        name=name,
        blocks=blocks,
        timing=timing,
        description=cell.description.replace("PLACEHOLDER", "REFERENCE STRUCTURE (simulator-verified, not in-game)"),
    )


def reference_library(*, characterize: bool = True) -> PrimitiveTechnologyLibrary:
    """The placeholder library with every available reference structure
    materialized (gates characterized by simulation)."""
    cells = [reference_cell(c, characterize=characterize) for c in placeholder_cells()]
    return PrimitiveTechnologyLibrary(REFERENCE_LIBRARY_NAME, cells)


__all__ = [
    "LIBRARY_NAME",
    "REFERENCE_LIBRARY_NAME",
    "STRUCTURES",
    "cell_pins",
    "reference_cell",
    "reference_library",
    "structure_blocks",
]
