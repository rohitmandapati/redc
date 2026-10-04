"""Simulation-derived characterization of block-level technology cells: truth
tables, transition timing, repeatability, and refusal of broken geometry even
when the cell's DECLARED function says it should work."""

from __future__ import annotations

import pytest

from redc.minecraft.blocks import MinecraftBlock, glass, repeater, stone, wall_torch
from redc.minecraft.characterize import CellPin, characterize_cell
from redc.minecraft.timing import CombinationalArc, ComponentTiming
from redc.minecraft.units import rt_to_gt
from redc.parser import CompileError
from redc.physical_primitive.netlist import PrimitiveKind
from redc.physical_primitive.technology.library import placeholder_cells
from redc.physical_primitive.technology.structures import (
    cell_pins,
    reference_cell,
    reference_library,
    structure_blocks,
)

Coord = tuple[int, int, int]


def cell_of(kind: PrimitiveKind):
    return next(c for c in placeholder_cells() if c.kind is kind and c.peripheral is None)


def blocks_of(kind: PrimitiveKind) -> dict[Coord, MinecraftBlock]:
    blocks = structure_blocks(cell_of(kind))
    assert blocks is not None
    return dict(blocks)


def test_torch_inverter_truth_table_timing_and_strength() -> None:
    result = characterize_cell("not", blocks_of(PrimitiveKind.NOT), cell_pins(cell_of(PrimitiveKind.NOT)))
    assert result.truth_tables == {"y": "10"}
    assert result.min_output_strength == {"y": 15}
    (arc,) = result.timing.arcs
    assert (arc.from_pin, arc.to_pin, arc.min_gt, arc.max_gt) == ("a", "y", rt_to_gt(1), rt_to_gt(1))
    assert result.timing.source == "characterized"
    edges = sorted((t.input_edge, t.output_edge, t.first_gt) for t in result.transitions)
    assert edges == [("fall", "rise", 2), ("rise", "fall", 2)]
    assert result.check({"y": "10"}, ComponentTiming((CombinationalArc("a", "y", 2, 2),)), {"y": 15}) == []


def test_diode_or_isolates_its_inputs_and_drives_strength_13() -> None:
    result = characterize_cell("or", blocks_of(PrimitiveKind.OR), cell_pins(cell_of(PrimitiveKind.OR)))
    assert result.truth_tables == {"y": "0111"}
    assert result.min_output_strength == {"y": 13}
    assert {(a.from_pin, a.min_gt, a.max_gt) for a in result.timing.arcs} == {("a", 2, 2), ("b", 2, 2)}
    # Only output changes are transitions: toggling a while b is high changes nothing.
    quiet = [t for t in result.transitions if t.input == "a" and dict(t.others) == {"b": 1}]
    assert quiet and all(t.output_edge is None and t.changes == 0 for t in quiet)


def test_characterization_is_repeatable() -> None:
    cell = cell_of(PrimitiveKind.OR)
    first = characterize_cell("or", blocks_of(PrimitiveKind.OR), cell_pins(cell)).to_dict()
    second = characterize_cell("or", blocks_of(PrimitiveKind.OR), cell_pins(cell)).to_dict()
    assert first == second


def test_declared_timing_must_be_conservative() -> None:
    result = characterize_cell("not", blocks_of(PrimitiveKind.NOT), cell_pins(cell_of(PrimitiveKind.NOT)))
    exact = ComponentTiming((CombinationalArc("a", "y", 2, 2),))
    assert result.check({"y": "10"}, exact) == []
    pessimistic = ComponentTiming((CombinationalArc("a", "y", 0, 8),))
    assert result.check({"y": "10"}, pessimistic) == []
    too_fast = ComponentTiming((CombinationalArc("a", "y", 1, 1),))
    assert any("declared [1, 1] gt but measured [2, 2]" in p for p in result.check({"y": "10"}, too_fast))


@pytest.mark.parametrize(
    "corruption",
    ["glass_block", "torch_replaced_by_stone", "missing_torch"],
)
def test_corrupted_inverter_geometry_fails_although_declared_as_not(corruption: str) -> None:
    blocks = blocks_of(PrimitiveKind.NOT)
    if corruption == "glass_block":
        blocks[(1, 1, 0)] = glass()  # not a conductor: the input can no longer turn the torch off
    elif corruption == "torch_replaced_by_stone":
        blocks[(2, 1, 0)] = stone()  # no torch: the output is never powered
    else:
        blocks[(2, 1, 0)] = MinecraftBlock("minecraft:air")
    result = characterize_cell("broken", blocks, cell_pins(cell_of(PrimitiveKind.NOT)))
    assert result.check({"y": "10"}) != []


def test_a_reversed_or_diode_breaks_the_cell() -> None:
    blocks = blocks_of(PrimitiveKind.OR)
    blocks[(1, 1, 2)] = repeater("west", 1)  # input b's diode now points back into b's own route
    result = characterize_cell("broken_or", blocks, cell_pins(cell_of(PrimitiveKind.OR)))
    assert result.truth_tables["y"] != "0111"
    assert result.check({"y": "0111"}) != []


def test_inputs_are_characterized_at_their_minimum_strength() -> None:
    """The harness drives an input at exactly the cell's required strength:
    a cell that only works at full strength is caught."""
    cell = cell_of(PrimitiveKind.NOT)
    weak = [CellPin(p.name, p.direction, p.coord, p.facing, 1 if p.direction == "in" else p.strength) for p in cell_pins(cell)]
    assert characterize_cell("not", blocks_of(PrimitiveKind.NOT), weak).truth_tables == {"y": "10"}


def test_reference_structures_replace_placeholders_without_changing_the_footprint() -> None:
    library = reference_library()
    for cell in placeholder_cells():
        ref = next(c for c in library.values() if c.kind is cell.kind and c.peripheral == cell.peripheral)
        assert ref.voxels == cell.voxels and ref.keepout == cell.keepout and ref.pins == cell.pins
        assert ref.placeholder is True  # simulator-verified, never claimed as in-game verified
        if ref.materialized:
            assert ref.name.endswith("_reference") and ref.blocks is not None
    materialized = {c.kind for c in library.values() if c.materialized}
    assert {PrimitiveKind.NOT, PrimitiveKind.OR, PrimitiveKind.INPUT_BIT, PrimitiveKind.OUTPUT_BIT} <= materialized
    assert PrimitiveKind.AND not in materialized and PrimitiveKind.REGISTER_BIT not in materialized
    not_cell = next(c for c in library.values() if c.kind is PrimitiveKind.NOT)
    assert not_cell.timing is not None and not_cell.timing.source == "characterized"


def test_a_reference_structure_that_fails_characterization_is_refused() -> None:
    import dataclasses

    from redc.physical_primitive.technology import structures

    cell = cell_of(PrimitiveKind.NOT)
    broken = {(1, 1, 0): glass(), (2, 1, 0): wall_torch("east")}
    original = structures.STRUCTURES[PrimitiveKind.NOT]
    structures.STRUCTURES[PrimitiveKind.NOT] = lambda _cell: broken
    try:
        with pytest.raises(CompileError, match="fails characterization"):
            reference_cell(dataclasses.replace(cell))
    finally:
        structures.STRUCTURES[PrimitiveKind.NOT] = original
