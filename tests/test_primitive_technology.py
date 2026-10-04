"""Primitive Minecraft technology: cell validation, block geometry and
orientations, the all-placeholder library and one-to-one technology mapping.

* :class:`PrimitiveCell` rejects every malformed definition (pins inside the
  structure, floating pins, blocked approaches, pins that could short, keep-outs
  over voxels, wrong pin interfaces, bad peripheral specs, ...);
* :class:`Orientation` is a proper clockwise quarter-turn rotation about +y and
  rotated / translated cells stay self-consistent;
* every cell of :data:`PRIMITIVE_TECHNOLOGY` is an honest PLACEHOLDER footprint
  following the library conventions in all four orientations;
* :func:`map_primitives_to_minecraft` realizes every primitive with exactly one
  cell, keeps ids / nets / provenance, assigns no coordinates, and
  :meth:`PrimitivePhysicalNetlist.validate` catches any tampering.

Cells for the validation tests are built by hand (``inverter`` / ``lever``)
so each case changes exactly one thing.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

import pytest

from redc import BOOL, CompileError, IRType, compile_source
from redc.physical_primitive import (
    DEFAULT_INTERFACE_POLICY,
    PRIMITIVE_TECHNOLOGY,
    MappedInstance,
    PadInterfacePolicy,
    PeripheralDirection,
    PeripheralSpec,
    PortRealization,
    PrimitiveCell,
    PrimitiveInstance,
    PrimitiveKind,
    PrimitiveNetlist,
    PrimitivePhysicalNetlist,
    PrimitivePin,
    PrimitiveTechnologyLibrary,
    PrimitiveTraceRecorder,
    Provenance,
    map_primitives_to_minecraft,
    place_and_route_graph,
    select_first_cell,
    synthesize_to_primitives,
)
from redc.physical_primitive.geometry import (
    CLOCKWISE,
    IDENTITY,
    ORIENTATIONS,
    Bounds,
    Coord,
    Direction,
    Orientation,
    add,
    below,
    manhattan,
    sub,
)
from redc.physical_primitive.netlist import PIN_INTERFACE
from redc.physical_primitive.pnr.records import design_records, instance_record
from redc.physical_primitive.redstone import MAX_SIGNAL_STRENGTH, SIGNAL_NEIGHBORHOOD
from redc.physical_primitive.synthesis.interface import PADS
from redc.physical_primitive.technology import (
    LIBRARY_NAME,
    OrientedCell,
    placeholder_cell,
    placeholder_cells,
)

K = PrimitiveKind
INPUT, OUTPUT = PeripheralDirection.INPUT, PeripheralDirection.OUTPUT
E, S, W, N = Direction.EAST, Direction.SOUTH, Direction.WEST, Direction.NORTH
PAD = PadInterfacePolicy()

INVERTER = "bool main(bool a) { return !a; }"
FULL_ADDER = "uint2 main(bool a, bool b, bool c) { return (uint2)a + (uint2)b + (uint2)c; }"
COUNTER = "uint2 main(uint2 n) { uint2 x = 0; for (uint2 i = 0; i < n; i++) { x = x + 1; } return x; }"
#: Sequential with a uint8 ``result``: under the default interface policy it
#: uses a lever (``start``) and the 2-digit display -- every PrimitiveKind.
EVERY_KIND = "uint8 main(uint2 n) { uint8 x = 0; for (uint2 i = 0; i < n; i++) { x = x + 1; } return x; }"

CELLS = tuple(PRIMITIVE_TECHNOLOGY.values())
GATES = (K.AND, K.OR, K.XOR, K.NOT)
NON_PERIPHERAL_KINDS = tuple(k for k in K if k is not K.PERIPHERAL)


# -- helpers -------------------------------------------------------------------

#: A minimal hand-built inverter: three base blocks in a row, one body block on
#: the middle one, input ``a`` resting on the west base, output ``y`` on the east.
BASE = (((0, 0, 0), "base"), ((1, 0, 0), "base"), ((2, 0, 0), "base"))
BODY = (((1, 1, 0), "body"),)
KEEPOUT = frozenset({(1, 2, 0), (1, 1, 1), (1, 1, -1)})
PIN_A = PrimitivePin("a", "in", (0, 1, 0), W, 1)
PIN_Y = PrimitivePin("y", "out", (2, 1, 0), E)

#: A minimal one-bit input peripheral: body on the west base, ``b0`` on the east.
LEVER_VOXELS = (((0, 0, 0), "base"), ((1, 0, 0), "base"), ((0, 1, 0), "lever"))
PIN_B0 = PrimitivePin("b0", "out", (1, 1, 0), E)


def inverter(**changes: Any) -> PrimitiveCell:
    fields: dict[str, Any] = {
        "name": "test_inverter",
        "kind": K.NOT,
        "voxels": BASE + BODY,
        "keepout": KEEPOUT,
        "pins": (PIN_A, PIN_Y),
        "latency": 1,
    }
    fields.update(changes)
    return PrimitiveCell(**fields)


def lever(**changes: Any) -> PrimitiveCell:
    fields: dict[str, Any] = {
        "name": "test_lever",
        "kind": K.PERIPHERAL,
        "voxels": LEVER_VOXELS,
        "keepout": frozenset({(0, 2, 0)}),
        "pins": (PIN_B0,),
        "latency": None,
        "peripheral": ("lever", INPUT, 1),
    }
    fields.update(changes)
    return PrimitiveCell(**fields)


def primitive(kind: PrimitiveKind, spec: PeripheralSpec | None = None) -> PrimitiveInstance:
    """A free-standing primitive (only kind / spec matter for candidate lookup)."""
    init = False if kind is K.REGISTER_BIT else None
    return PrimitiveInstance(0, kind, Provenance(None, kind.value), init, spec)


def synth(source: str, interface: Any = PAD) -> PrimitiveNetlist:
    return synthesize_to_primitives(compile_source(source), interface=interface)


def mapped_design(source: str = EVERY_KIND, interface: Any = DEFAULT_INTERFACE_POLICY) -> PrimitivePhysicalNetlist:
    return map_primitives_to_minecraft(synth(source, interface))


def first(mapped: PrimitivePhysicalNetlist, kind: PrimitiveKind) -> MappedInstance:
    return next(inst for inst in mapped.instances if inst.kind is kind)


def rebuilt(oriented: OrientedCell) -> PrimitiveCell:
    """A brand-new cell definition from rotated geometry: constructing it re-runs
    every validation rule in the rotated frame."""
    cell = oriented.cell
    return PrimitiveCell(
        name=f"{cell.name}@{oriented.orientation.name}",
        kind=cell.kind,
        voxels=oriented.voxels,
        keepout=oriented.keepout,
        pins=tuple(
            PrimitivePin(p.name, p.direction, p.position, p.facing, p.strength) for p in oriented.pins.values()
        ),
        latency=cell.latency,
        peripheral=cell.peripheral,
    )


def assert_plain_json(value: Any) -> None:
    if isinstance(value, dict):
        assert all(isinstance(k, str) for k in value)
        for v in value.values():
            assert_plain_json(v)
    elif isinstance(value, list):
        for v in value:
            assert_plain_json(v)
    else:
        assert value is None or type(value) in (str, int, float, bool), repr(value)


def cross(a: Coord, b: Coord) -> Coord:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


# -- PrimitiveCell validation ----------------------------------------------------


def test_hand_built_cells_are_valid() -> None:
    cell = inverter()
    assert cell.occupied == {(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)}
    assert cell.pin("y") is PIN_Y and cell.pin("a").approach == (-1, 1, 0)
    with pytest.raises(CompileError, match="has no pin 'q'"):
        cell.pin("q")
    assert lever().peripheral == ("lever", INPUT, 1)
    # Unknown latency and a never-powered output (constant 0) are both allowed.
    assert inverter(latency=None).latency is None
    assert inverter(pins=(PIN_A, replace(PIN_Y, strength=0))).pin("y").strength == 0


INVALID_CELLS = [
    ("no_name", {"name": ""}, "needs a name"),
    ("kind_not_an_enum", {"kind": "not"}, "kind must be a PrimitiveKind"),
    ("negative_latency", {"latency": -1}, "latency must be non-negative"),
    ("empty_voxels", {"voxels": ()}, "a cell occupies at least one voxel"),
    ("duplicate_voxels", {"voxels": BASE + BODY + (((1, 1, 0), "body"),)}, "duplicate voxels"),
    ("voxel_below_local_ground", {"voxels": BASE + BODY + (((1, -1, 0), "body"),)}, "local y >= 0"),
    ("keepout_overlaps_voxels", {"keepout": KEEPOUT | {(1, 1, 0)}}, "keep-out overlaps occupied voxels"),
    ("no_orientations", {"orientations": ()}, "at least one allowed orientation"),
    ("duplicate_pin_names", {"pins": (PIN_A, replace(PIN_Y, name="a"))}, "duplicate pin names"),
    ("pin_direction", {"pins": (replace(PIN_A, direction="inout"), PIN_Y)}, "direction must be 'in' or 'out'"),
    ("pin_below_signal_level", {"pins": (replace(PIN_A, position=(0, 0, 0)), PIN_Y)}, "local y >= 1"),
    ("pin_inside_voxels", {"pins": (replace(PIN_A, position=(1, 1, 0)), PIN_Y)}, "pin 'a' block is occupied"),
    ("pin_inside_keepout", {"pins": (replace(PIN_A, position=(1, 1, 1)), PIN_Y)}, "pin 'a' block is occupied"),
    (
        "pin_beside_the_cell",
        {"pins": (replace(PIN_A, position=(-1, 1, 0)), PIN_Y)},
        "pin 'a' must rest on one of the cell's own voxels",
    ),
    (
        "pin_on_air_above_its_column",
        {"pins": (replace(PIN_A, position=(0, 2, 0)), PIN_Y)},
        "pin 'a' must rest on one of the cell's own voxels",
    ),
    (
        "approach_into_the_body",
        {"pins": (PIN_A, replace(PIN_Y, facing=W))},
        "pin 'y' approach block [1, 1, 0] is blocked",
    ),
    (
        "approach_into_the_keepout",
        {"keepout": KEEPOUT | {(3, 1, 0)}},
        "pin 'y' approach block [3, 1, 0] is blocked",
    ),
    ("input_needs_some_strength", {"pins": (replace(PIN_A, strength=0), PIN_Y)}, "strength 0 out of range"),
    ("output_too_strong", {"pins": (PIN_A, replace(PIN_Y, strength=16))}, "strength 16 out of range"),
    (
        "pins_touch_level",
        {
            "voxels": BASE + BODY + (((0, 0, 1), "base"),),
            "keepout": frozenset({(1, 2, 0)}),
            "pins": (PIN_A, replace(PIN_Y, position=(0, 1, 1))),
        },
        "pin blocks [0, 1, 0] and [0, 1, 1] could short each other",
    ),
    (
        "pins_touch_one_up_diagonally",
        {"keepout": frozenset({(1, 1, 1), (1, 1, -1)}), "pins": (PIN_A, replace(PIN_Y, position=(1, 2, 0)))},
        "pin blocks [0, 1, 0] and [1, 2, 0] could short each other",
    ),
    (
        "pins_share_a_block",
        {"pins": (PIN_A, replace(PIN_Y, position=(0, 1, 0), facing=W))},
        "pin blocks [0, 1, 0] and [0, 1, 0] could short each other",
    ),
    ("wrong_pin_name", {"pins": (replace(PIN_A, name="x"), PIN_Y)}, "not pins must be"),
    (
        "swapped_pin_directions",
        {"pins": (replace(PIN_A, direction="out"), replace(PIN_Y, direction="in", strength=1))},
        "not pins must be",
    ),
    ("missing_pin", {"pins": (PIN_A,)}, "not pins must be"),
    ("pins_of_another_kind", {"kind": K.AND}, "and pins must be"),
    ("register_interface", {"kind": K.REGISTER_BIT}, "register_bit pins must be"),
    (
        "spec_on_a_non_peripheral",
        {"peripheral": ("lever", INPUT, 1)},
        "only PERIPHERAL cells carry a peripheral spec",
    ),
]


@pytest.mark.parametrize(
    ("changes", "message"), [case[1:] for case in INVALID_CELLS], ids=[case[0] for case in INVALID_CELLS]
)
def test_invalid_cells_are_rejected(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(CompileError, match=re.escape(message)):
        inverter(**changes)


#: Two pins two blocks apart (so their endpoints never touch) whose APPROACH
#: blocks clash: ``(voxels, clashing pins, control pins)`` -- the control turns
#: one pin away so the same cell is legal.
APPROACH_CLASHES = [
    (
        "two_pins_share_one_approach",
        (((0, 0, 0), "base"), ((2, 0, 0), "base"), ((1, 0, 1), "base"), ((1, 1, 1), "body")),
        (PrimitivePin("a", "in", (0, 1, 0), E, 1), PrimitivePin("y", "out", (2, 1, 0), W)),
        (PrimitivePin("a", "in", (0, 1, 0), W, 1), PrimitivePin("y", "out", (2, 1, 0), E)),
    ),
    (
        "approach_touches_the_other_pin",
        (((0, 0, 0), "base"), ((1, 0, 0), "base"), ((-1, 0, 1), "base"), ((1, 1, 0), "body")),
        (PrimitivePin("a", "in", (0, 1, 0), W, 1), PrimitivePin("y", "out", (-1, 1, 1), W)),
        (PrimitivePin("a", "in", (0, 1, 0), N, 1), PrimitivePin("y", "out", (-1, 1, 1), W)),
    ),
]


@pytest.mark.parametrize(
    ("voxels", "clashing", "control"),
    [case[1:] for case in APPROACH_CLASHES],
    ids=[case[0] for case in APPROACH_CLASHES],
)
def test_pin_approaches_may_not_clash(voxels, clashing, control) -> None:
    """A route enters / leaves a pin THROUGH its approach block, so that block
    carries the pin's net exactly like the pin block (``BlockGrid.check_placement``
    rejects this very geometry between two cells).  Within one cell an approach
    may therefore neither be nor touch another pin's endpoint or approach: the
    input and output nets of such an inverter can never both be routed."""
    fields: dict[str, Any] = {"name": "clash", "kind": K.NOT, "voxels": voxels, "keepout": frozenset(), "latency": 1}
    assert PrimitiveCell(**fields, pins=control).pin("y").direction == "out"
    with pytest.raises(CompileError):
        PrimitiveCell(**fields, pins=clashing)


INVALID_PERIPHERALS = [
    ("no_spec", {"peripheral": None}, "a peripheral cell needs (kind, direction, width)"),
    ("width_mismatch", {"peripheral": ("lever", INPUT, 2)}, "peripheral pins must be"),
    ("direction_mismatch", {"peripheral": ("lever", OUTPUT, 1)}, "peripheral pins must be"),
    ("pin_not_named_b0", {"pins": (replace(PIN_B0, name="y"),)}, "peripheral pins must be"),
    ("spec_on_a_pad", {"kind": K.INPUT_BIT}, "only PERIPHERAL cells carry a peripheral spec"),
]


@pytest.mark.parametrize(
    ("changes", "message"),
    [case[1:] for case in INVALID_PERIPHERALS],
    ids=[case[0] for case in INVALID_PERIPHERALS],
)
def test_peripheral_spec_must_match_the_pins(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(CompileError, match=re.escape(message)):
        lever(**changes)


def test_peripheral_spec_direction_is_type_checked() -> None:
    """Like ``kind``, the spec's direction must be a PeripheralDirection: the
    plain string ``"input"`` never matches any primitive's PeripheralSpec in
    library lookup and breaks :meth:`PrimitiveCell.to_dict`."""
    with pytest.raises(CompileError):
        lever(peripheral=("lever", "input", 1))


def test_peripheral_cells_follow_the_device_direction() -> None:
    # An OUTPUT device (lamp, display) consumes bits: its b<i> pins are inputs.
    lamp = PrimitiveCell(
        name="test_lamp",
        kind=K.PERIPHERAL,
        voxels=(((0, 0, 0), "base"), ((1, 0, 0), "base"), ((1, 1, 0), "lamp")),
        keepout=frozenset(),
        pins=(PrimitivePin("b0", "in", (0, 1, 0), W, 1),),
        latency=None,
        peripheral=("lamp", OUTPUT, 1),
    )
    assert lamp.to_dict()["peripheral"] == {"kind": "lamp", "direction": "output", "width": 1}


def test_placeholder_cell_derives_base_and_keepout() -> None:
    cell = placeholder_cell(
        "probe_inverter",
        K.NOT,
        body=[(1, 1, 0), (2, 1, 0)],
        pins=(PrimitivePin("a", "in", (0, 1, 0), W, 1), PrimitivePin("y", "out", (3, 1, 0), E)),
        latency=1,
        description="probe",
    )
    roles = dict(cell.voxels)
    assert {c for c, r in roles.items() if r == "base"} == {(x, 0, 0) for x in range(4)}
    assert {c for c, r in roles.items() if r == "body"} == {(1, 1, 0), (2, 1, 0)}
    # Body neighbours at their level, the layer above the body, both sides of
    # every pin -- never a pin block or an approach block.
    assert cell.keepout == {
        (1, 1, 1), (1, 1, -1), (2, 1, 1), (2, 1, -1), (1, 2, 0), (2, 2, 0),
        (0, 1, 1), (0, 1, -1), (3, 1, 1), (3, 1, -1),
    }  # fmt: skip
    assert cell.placeholder and cell.structure is None and cell.orientations == ORIENTATIONS
    with pytest.raises(CompileError, match="y >= 1"):
        placeholder_cell("sunken", K.NOT, body=[(1, 0, 0)], pins=(), latency=1, description="probe")


# -- geometry: orientations ------------------------------------------------------


def test_one_quarter_turn_maps_east_south_west_north() -> None:
    turn = Orientation(1)
    assert turn.apply(E.vector) == S.vector == (0, 0, 1)
    assert turn.apply(S.vector) == W.vector == (-1, 0, 0)
    assert turn.apply(W.vector) == N.vector == (0, 0, -1)
    assert turn.apply(N.vector) == E.vector == (1, 0, 0)
    assert CLOCKWISE == (E, S, W, N)
    # The documented formula for ``south``: (x, y, z) -> (-z, y, x).
    assert Orientation.parse("south").apply((2, 5, 3)) == (-3, 5, 2)


@pytest.mark.parametrize("turns", range(4))
def test_orientation_names_where_local_east_points(turns: int) -> None:
    orientation = Orientation(turns)
    assert orientation == ORIENTATIONS[turns]
    assert orientation.apply(E.vector) == CLOCKWISE[turns].vector
    assert orientation.name == CLOCKWISE[turns].label
    assert orientation.to_dict() == {"name": CLOCKWISE[turns].label, "quarter_turns": turns}
    assert Orientation.parse(orientation.name) == orientation
    assert Orientation.parse(orientation.name.upper()) == orientation


@pytest.mark.parametrize("orientation", ORIENTATIONS, ids=lambda o: o.name)
def test_orientation_is_a_proper_rotation_about_y(orientation: Orientation) -> None:
    rot = orientation.apply
    ex, ey, ez = rot((1, 0, 0)), rot((0, 1, 0)), rot((0, 0, 1))
    assert ey == (0, 1, 0)
    assert cross(ex, ey) == ez  # a rotation keeps the frame right-handed: never a mirror
    for cell in [(0, 0, 0), (3, -2, 7), (-5, 9, 1), (1, 1, -4)]:
        image = rot(cell)
        assert image[1] == cell[1]
        assert manhattan(image, (0, 0, 0)) == manhattan(cell, (0, 0, 0))
        assert image == tuple(cell[0] * ex[i] + cell[1] * ey[i] + cell[2] * ez[i] for i in range(3))


def test_four_quarter_turns_are_the_identity() -> None:
    cells = [(0, 0, 0), (3, -2, 7), (-5, 9, 1), (1, 1, -4)]
    turn = Orientation(1)
    for cell in cells:
        image = cell
        for _ in range(4):
            image = turn.apply(image)
        assert image == cell
        assert IDENTITY.apply(cell) == cell
    for a, b in itertools.product(range(4), repeat=2):
        composed = Orientation((a + b) % 4)
        assert all(Orientation(b).apply(Orientation(a).apply(c)) == composed.apply(c) for c in cells)


def test_direction_rotation_agrees_with_vector_rotation() -> None:
    assert E.rotated(1) is S and S.rotated(1) is W and W.rotated(1) is N and N.rotated(1) is E
    assert E.rotated(4) is E and E.rotated(-1) is N and E.rotated(6) is W
    for direction in Direction:
        assert direction.opposite is direction.rotated(2)
        assert Direction.of(direction.vector) is direction
        assert Direction.parse(direction.label) is direction
        for orientation in ORIENTATIONS:
            assert orientation.direction(direction) is direction.rotated(orientation.quarter_turns)
            assert Direction.of(orientation.apply(direction.vector)) is orientation.direction(direction)
    assert Direction.of((1, 1, 0)) is E  # the y component of a staircase step is ignored


@pytest.mark.parametrize("turns", [-1, 4, 7])
def test_orientation_rejects_out_of_range_turns(turns: int) -> None:
    with pytest.raises(ValueError, match="quarter_turns must be 0..3"):
        Orientation(turns)


def test_bounds_helpers() -> None:
    assert Bounds.of([]) is None
    box = Bounds.of([(1, 2, 3), (-1, 5, 0), (4, 0, 2)])
    assert box == Bounds((-1, 0, 0), (4, 5, 3))
    assert box.dims == (6, 6, 4) and box.volume == 144
    assert box.contains((0, 0, 0)) and box.contains((4, 5, 3)) and not box.contains((5, 0, 0))
    assert box.translate((1, 1, 1)) == Bounds((0, 1, 1), (5, 6, 4))
    assert box.union(None) is box
    assert box.union(Bounds((10, 10, 10), (11, 11, 11))) == Bounds((-1, 0, 0), (11, 11, 11))
    assert box.expand(2, 3) == Bounds((-3, 0, -3), (6, 5, 6))
    assert box.expand(1, 1, (1, 9)) == Bounds((-2, 1, -1), (5, 9, 4))
    assert box.to_dict() == {"min": [-1, 0, 0], "max": [4, 5, 3], "dims": [6, 6, 4], "volume": 144}


# -- oriented and placed cells --------------------------------------------------


def test_oriented_is_computed_once_and_cached() -> None:
    cell = PRIMITIVE_TECHNOLOGY["and_gate_2x3_placeholder"]
    for orientation in ORIENTATIONS:
        oriented = cell.oriented(orientation)
        assert cell.oriented(orientation) is oriented
        assert cell.oriented(Orientation.parse(orientation.name)) is oriented  # keyed by the turn
        assert oriented.cell is cell and oriented.orientation == orientation


def test_oriented_rejects_a_disallowed_orientation() -> None:
    fixed = inverter(orientations=(IDENTITY,))
    assert fixed.oriented(IDENTITY).orientation == IDENTITY
    with pytest.raises(CompileError, match="does not allow orientation south"):
        fixed.oriented(Orientation(1))
    assert fixed.to_dict()["orientations"] == ["east"]


def test_derived_cell_does_not_share_the_orientation_cache() -> None:
    """``dataclasses.replace`` is the natural way to derive a variant of a frozen
    cell; each cell must rotate ITS OWN geometry, never another cell's cached one."""
    wide = {
        "name": "test_inverter_wide",
        "voxels": (*BASE, ((3, 0, 0), "base"), *BODY),
        "pins": (PIN_A, replace(PIN_Y, position=(3, 1, 0))),
    }
    original = inverter()
    original.oriented(IDENTITY)  # the original is rotated first ...
    variant = replace(original, **wide)
    oriented = variant.oriented(IDENTITY)  # ... then its variant
    assert (oriented.cell.name, oriented.pins["y"].position) == ("test_inverter_wide", (3, 1, 0))
    # The other way round: rotating the variant first must not poison the original.
    original = inverter()
    replace(original, **wide).oriented(IDENTITY)
    oriented = original.oriented(IDENTITY)
    assert (oriented.cell.name, oriented.pins["y"].position) == ("test_inverter", (2, 1, 0))


@pytest.mark.parametrize("orientation", ORIENTATIONS, ids=lambda o: o.name)
@pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.name)
def test_rotated_cell_stays_consistent(cell: PrimitiveCell, orientation: Orientation) -> None:
    oriented = cell.oriented(orientation)
    rot = orientation.apply
    assert oriented.voxels == tuple((rot(c), role) for c, role in cell.voxels)
    assert oriented.occupied == {rot(c) for c in cell.occupied}
    assert oriented.keepout == {rot(c) for c in cell.keepout}
    assert not oriented.occupied & oriented.keepout
    assert list(oriented.pins) == [p.name for p in cell.pins]
    blocked = oriented.occupied | oriented.keepout
    for pin in cell.pins:
        rotated = oriented.pins[pin.name]
        assert (rotated.direction, rotated.strength) == (pin.direction, pin.strength)
        assert rotated.position == rot(pin.position)
        assert rotated.facing is orientation.direction(pin.facing)
        assert rotated.approach == rot(pin.approach)
        assert below(rotated.position) in oriented.occupied  # still rests on the cell
        assert rotated.position not in blocked
        assert rotated.approach not in blocked  # still reachable
    pins = [p.position for p in oriented.pins.values()]
    assert oriented.bounds == Bounds.of([*oriented.occupied, *oriented.keepout, *pins])
    # The rotated geometry passes every PrimitiveCell rule on its own.
    assert rebuilt(oriented).occupied == oriented.occupied


def test_placed_cell_translates_every_block() -> None:
    oriented = PRIMITIVE_TECHNOLOGY["xor_gate_2x3_placeholder"].oriented(Orientation(3))
    origin = (10, 2, -7)
    placed = oriented.translate(origin)
    assert placed.oriented is oriented and placed.origin == origin and placed.orientation == Orientation(3)
    assert placed.occupied == {add(c, origin) for c in oriented.occupied}
    assert placed.keepout == {add(c, origin) for c in oriented.keepout}
    assert placed.voxels() == [(add(c, origin), role) for c, role in oriented.voxels]
    assert placed.bounds == oriented.bounds.translate(origin)
    assert list(placed.pins) == list(oriented.pins)
    for name, pin in oriented.pins.items():
        moved = placed.pins[name]
        assert moved.position == add(pin.position, origin)
        assert moved.approach == add(pin.approach, origin)
        assert (moved.facing, moved.direction, moved.strength) == (pin.facing, pin.direction, pin.strength)
        assert sub(moved.approach, moved.position) == pin.facing.vector


def test_mapped_instance_place_and_unplace() -> None:
    cell = PRIMITIVE_TECHNOLOGY["not_gate_torch_placeholder"]
    inst = MappedInstance(3, K.NOT, cell, (3,))
    assert not inst.is_placed
    with pytest.raises(CompileError, match="mapped instance 3 .* is not placed"):
        _ = inst.placed
    placed = inst.place((5, 0, 5), Orientation(2))
    assert inst.is_placed and inst.origin == (5, 0, 5) and inst.orientation == Orientation(2)
    assert inst.placed is placed and placed.oriented is cell.oriented(Orientation(2))
    assert inst.to_dict()["origin"] == [5, 0, 5] and inst.to_dict()["orientation"] == "west"
    inst.unplace()
    assert (inst.origin, inst.orientation, inst.is_placed) == (None, None, False)


# -- the placeholder library ------------------------------------------------------


def test_library_covers_every_kind_and_both_default_peripherals() -> None:
    assert PRIMITIVE_TECHNOLOGY.name == LIBRARY_NAME == "redc-primitive-placeholder-v1"
    assert PRIMITIVE_TECHNOLOGY.kinds() == frozenset(NON_PERIPHERAL_KINDS)
    assert PRIMITIVE_TECHNOLOGY.peripherals() == (("lever", INPUT, 1), ("2-dig-7-seg", OUTPUT, 8))
    assert len(PRIMITIVE_TECHNOLOGY) == len(CELLS) == len(set(PRIMITIVE_TECHNOLOGY))
    # placeholder_cells() is deterministic and is exactly the default library.
    assert [c.name for c in placeholder_cells()] == list(PRIMITIVE_TECHNOLOGY)
    assert placeholder_cells() == CELLS


@pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.name)
def test_every_cell_is_an_honest_placeholder(cell: PrimitiveCell) -> None:
    assert cell.placeholder is True
    assert cell.structure is None
    assert "PLACEHOLDER" in cell.description
    assert cell.orientations == ORIENTATIONS
    record = cell.to_dict()
    assert record["placeholder"] is True and record["structure"] is None


@pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.name)
def test_every_cell_validates_in_all_four_orientations(cell: PrimitiveCell) -> None:
    for orientation in ORIENTATIONS:
        again = rebuilt(cell.oriented(orientation))
        assert again.kind is cell.kind
        assert again.occupied == {orientation.apply(c) for c in cell.occupied}


@pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.name)
def test_every_cell_keeps_its_pins_two_blocks_apart(cell: PrimitiveCell) -> None:
    for a, b in itertools.combinations(cell.pins, 2):
        delta = sub(b.position, a.position)
        assert delta not in SIGNAL_NEIGHBORHOOD
        assert max(abs(d) for d in delta) >= 2, f"{a.name} and {b.name} are neighbours"


@pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.name)
def test_every_cell_takes_inputs_west_and_drives_outputs_east(cell: PrimitiveCell) -> None:
    roles = dict(cell.voxels)
    xs = [c[0] for c in cell.occupied]
    for pin in cell.pins:
        if pin.direction == "in":
            assert pin.facing is W and pin.position[0] == min(xs)
            assert 1 <= pin.strength <= MAX_SIGNAL_STRENGTH
        else:
            assert pin.facing is E and pin.position[0] == max(xs)
        assert pin.position[1] == 1  # the signal level
        assert roles[below(pin.position)] == "base"  # resting on the cell's own base layer
    assert {c[1] for c, role in cell.voxels if role == "base"} == {0}
    expected = PIN_INTERFACE.get(cell.kind)
    if expected is not None:
        ins = tuple(p.name for p in cell.pins if p.direction == "in")
        outs = tuple(p.name for p in cell.pins if p.direction == "out")
        assert (ins, outs) == expected


@pytest.mark.parametrize("kind", [*GATES, K.REGISTER_BIT], ids=lambda k: k.value)
def test_logic_cells_are_sparse(kind: PrimitiveKind) -> None:
    (cell,) = PRIMITIVE_TECHNOLOGY.candidates(primitive(kind))
    box = Bounds.of(cell.occupied)
    assert box is not None
    assert len(cell.occupied) < box.volume  # never a solid bounding box
    assert {role for _, role in cell.voxels} == {"base", "body"}


@pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.name)
def test_cell_to_dict_is_plain_json(cell: PrimitiveCell) -> None:
    record = cell.to_dict()
    assert_plain_json(record)
    assert json.loads(json.dumps(record)) == record
    assert record["name"] == cell.name and record["kind"] == cell.kind.value
    assert record["latency"] == cell.latency
    assert record["stateful"] is (cell.kind is K.REGISTER_BIT)
    assert record["orientations"] == ["east", "south", "west", "north"]
    assert [tuple(v["coord"]) for v in record["voxels"]] == [c for c, _ in cell.voxels]
    assert [tuple(c) for c in record["keepout"]] == sorted(cell.keepout)
    assert [p["name"] for p in record["pins"]] == [p.name for p in cell.pins]
    assert all(p["facing"] in ("east", "west") for p in record["pins"])
    if cell.peripheral is None:
        assert record["peripheral"] is None
    else:
        kind, direction, width = cell.peripheral
        assert record["peripheral"] == {"kind": kind, "direction": direction.value, "width": width}


# -- library candidate lookup -----------------------------------------------------


@pytest.mark.parametrize("kind", NON_PERIPHERAL_KINDS, ids=lambda k: k.value)
def test_candidates_by_kind(kind: PrimitiveKind) -> None:
    prim = primitive(kind)
    candidates = PRIMITIVE_TECHNOLOGY.candidates(prim)
    assert len(candidates) == 1
    (cell,) = candidates
    assert cell.kind is kind and cell.peripheral is None
    assert {p.name: p.direction for p in cell.pins} == (
        {n: "in" for n in prim.inputs} | {n: "out" for n in prim.outputs}
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (PeripheralSpec("lever", INPUT, 1), ("lever_1bit_placeholder",)),
        (PeripheralSpec("2-dig-7-seg", OUTPUT, 8), ("two_digit_seven_segment_8bit_placeholder",)),
        (PeripheralSpec("lever", INPUT, 2), ()),  # wrong width
        (PeripheralSpec("lever", OUTPUT, 1), ()),  # wrong direction
        (PeripheralSpec("2-dig-7-seg", OUTPUT, 4), ()),
        (PeripheralSpec("button", INPUT, 1), ()),  # unknown device kind
    ],
    ids=["lever", "display", "lever_width", "lever_direction", "display_width", "unknown"],
)
def test_peripheral_candidates_match_kind_direction_and_width(spec: PeripheralSpec, expected: tuple) -> None:
    candidates = PRIMITIVE_TECHNOLOGY.candidates(primitive(K.PERIPHERAL, spec))
    assert tuple(c.name for c in candidates) == expected
    for cell in candidates:
        assert cell.peripheral == (spec.kind, spec.direction, spec.width)


def test_custom_library_keeps_registration_order() -> None:
    first_cell, second_cell = inverter(name="inverter_a"), inverter(name="inverter_b")
    library = PrimitiveTechnologyLibrary("test-lib", [first_cell, second_cell, lever()])
    assert library.candidates(primitive(K.NOT)) == (first_cell, second_cell)
    assert library.candidates(primitive(K.AND)) == ()
    assert library.candidates(primitive(K.PERIPHERAL, PeripheralSpec("lever", INPUT, 1))) == (library["test_lever"],)
    assert list(library) == ["inverter_a", "inverter_b", "test_lever"] and len(library) == 3
    assert library["inverter_b"] is second_cell
    assert library.kinds() == {K.NOT}
    assert library.peripherals() == (("lever", INPUT, 1),)
    with pytest.raises(CompileError, match="duplicate cell 'inverter_a'"):
        PrimitiveTechnologyLibrary("test-lib", [first_cell, inverter(name="inverter_a")])


# -- technology mapping --------------------------------------------------------------


@dataclass(frozen=True)
class ButtonInterface:
    """Realizes every bool input as a ``button`` peripheral the library lacks."""

    def realize_input(self, name: str, typ: IRType) -> PortRealization:
        return PortRealization("button") if typ == BOOL else PADS

    def realize_output(self, name: str, typ: IRType) -> PortRealization:
        return PADS


def test_missing_peripheral_cell_is_named() -> None:
    netlist = synth(INVERTER, ButtonInterface())
    (button,) = [inst for inst in netlist.instances if inst.kind is K.PERIPHERAL]
    with pytest.raises(CompileError) as info:
        map_primitives_to_minecraft(netlist)
    assert str(info.value) == (
        f"technology library 'redc-primitive-placeholder-v1' has no cell for primitive {button.id} "
        "(button input peripheral of width 1)"
    )


def test_missing_gate_cell_is_named() -> None:
    library = PrimitiveTechnologyLibrary("no-inverters", [c for c in placeholder_cells() if c.kind is not K.NOT])
    netlist = synth(INVERTER)
    (gate,) = [inst for inst in netlist.instances if inst.kind is K.NOT]
    with pytest.raises(CompileError, match=re.escape(f"'no-inverters' has no cell for primitive {gate.id} (not)")):
        map_primitives_to_minecraft(netlist, library=library)


def test_missing_cell_closes_the_trace_at_the_techmap_stage() -> None:
    trace = PrimitiveTraceRecorder("basic")
    with pytest.raises(CompileError, match="button input peripheral of width 1"):
        place_and_route_graph(compile_source(INVERTER), interface=ButtonInterface(), trace=trace)
    assert trace.final is not None and trace.final["success"] is False
    assert trace.final["failure"]["stage"] == "techmap"
    assert "button" in trace.final["failure"]["message"]
    assert any(e["type"] == "synthesis_complete" for e in trace.events)
    assert not any(e["phase"] in ("placement", "routing") for e in trace.events)


def test_selector_must_choose_a_candidate() -> None:
    netlist = synth("bool main(bool a, bool b) { return a && b; }")
    (gate,) = [inst for inst in netlist.instances if inst.kind is K.AND]
    or_cell = PRIMITIVE_TECHNOLOGY["or_gate_2x3_placeholder"]

    def wrong_kind(prim, candidates):
        return or_cell if prim.kind is K.AND else candidates[0]

    with pytest.raises(
        CompileError,
        match=re.escape(f"cell selector chose or_gate_2x3_placeholder for primitive {gate.id}, which is not a candidate"),
    ):
        map_primitives_to_minecraft(netlist, select=wrong_kind)
    # A perfectly valid AND cell from outside the library is not a candidate either.
    foreign = placeholder_cell(
        "foreign_and",
        K.AND,
        body=[(1, 1, 0), (1, 1, 1), (1, 1, 2)],
        pins=(PrimitivePin("a", "in", (0, 1, 0), W, 1), PrimitivePin("b", "in", (0, 1, 2), W, 1),
              PrimitivePin("y", "out", (2, 1, 1), E)),
        latency=2,
        description="foreign",
    )  # fmt: skip

    def outsider(prim, candidates):
        return foreign if prim.kind is K.AND else candidates[0]

    with pytest.raises(CompileError, match="cell selector chose foreign_and"):
        map_primitives_to_minecraft(netlist, select=outsider)


def test_selector_sees_every_primitive_with_its_candidates() -> None:
    library = PrimitiveTechnologyLibrary(
        "two-inverters", [*placeholder_cells(), inverter(name="inverter_alt")]
    )
    netlist = synth(INVERTER)
    seen: list[tuple[int, tuple[str, ...]]] = []

    def last_candidate(prim, candidates):
        seen.append((prim.id, tuple(c.name for c in candidates)))
        assert tuple(candidates) == library.candidates(prim)
        return candidates[-1]

    mapped = map_primitives_to_minecraft(netlist, library=library, select=last_candidate)
    assert [prim_id for prim_id, _ in seen] == [inst.id for inst in netlist.instances]
    assert first(mapped, K.NOT).cell.name == "inverter_alt"
    assert mapped.library == "two-inverters"
    default = map_primitives_to_minecraft(netlist, library=library)
    assert first(default, K.NOT).cell.name == "not_gate_torch_placeholder"  # select_first_cell
    cells = library.candidates(netlist.instances[0])
    assert select_first_cell(netlist.instances[0], cells) is cells[0]


@pytest.mark.parametrize(
    ("source", "interface"),
    [(INVERTER, PAD), (FULL_ADDER, PAD), (COUNTER, PAD), (EVERY_KIND, DEFAULT_INTERFACE_POLICY)],
    ids=["inverter", "full_adder", "counter", "every_kind"],
)
def test_techmap_is_one_to_one(source: str, interface: Any) -> None:
    netlist = synth(source, interface)
    mapped = map_primitives_to_minecraft(netlist)
    assert mapped.logical is netlist and mapped.library == LIBRARY_NAME
    assert len(mapped.instances) == len(netlist.instances)
    for index, (inst, prim) in enumerate(zip(mapped.instances, netlist.instances, strict=True)):
        assert inst.id == prim.id == index
        assert inst.realizes == (prim.id,)
        assert inst.kind is prim.kind is inst.cell.kind
        assert inst.cell is PRIMITIVE_TECHNOLOGY.candidates(prim)[0]
        assert {p.name: p.direction for p in inst.cell.pins} == (
            {n: "in" for n in prim.inputs} | {n: "out" for n in prim.outputs}
        )
        assert tuple(p.name for p in inst.cell.pins if p.direction == "in") == prim.inputs
        if prim.peripheral is not None:
            spec = prim.peripheral
            assert inst.cell.peripheral == (spec.kind, spec.direction, spec.width)
    assert [(n.id, n.driver, n.sinks, n.role) for n in mapped.nets] == [
        (n.id, n.driver, n.sinks, n.role) for n in netlist.nets
    ]
    assert all(n.fanout == len(n.sinks) for n in mapped.nets)
    terminals = mapped.net_of_terminal()
    for net in netlist.nets:
        assert terminals[net.driver] == net.id
        assert all(terminals[sink] == net.id for sink in net.sinks)
    assert list(mapped.cells_used()) == list(dict.fromkeys(inst.cell.name for inst in mapped.instances))
    mapped.validate()


def test_every_kind_design_really_uses_every_kind() -> None:
    mapped = mapped_design()
    assert {inst.kind for inst in mapped.instances} == set(K)
    assert {inst.cell.name for inst in mapped.instances} == set(PRIMITIVE_TECHNOLOGY)


def test_mapped_instance_lookup_mirrors_the_logical_netlist() -> None:
    mapped = mapped_design(INVERTER, PAD)
    for inst in mapped.instances:
        assert mapped.instance(inst.id) is inst
    for unknown in (-1, len(mapped.instances)):
        with pytest.raises(CompileError, match=f"unknown primitive instance {unknown}"):
            mapped.logical.instance(unknown)
        with pytest.raises(CompileError):  # not the last instance, not an IndexError
            mapped.instance(unknown)


def test_techmap_assigns_no_coordinates() -> None:
    mapped = mapped_design()
    for inst in mapped.instances:
        assert inst.origin is None and inst.orientation is None and not inst.is_placed
        with pytest.raises(CompileError, match="is not placed"):
            _ = inst.placed
    record = mapped.to_dict()
    assert_plain_json(record)
    assert record["schema"] == "redc.primitive-mapped-netlist.v1" and record["stage"] == "mapped"
    assert all(i["origin"] is None and i["orientation"] is None for i in record["instances"])
    assert [c["name"] for c in record["cells"]] == list(mapped.cells_used())
    assert record["logical"] == mapped.logical.to_dict()


def test_records_preserve_provenance() -> None:
    mapped = mapped_design()
    logical = mapped.logical
    records = design_records(mapped)
    assert records["library"] == LIBRARY_NAME
    assert [r["id"] for r in records["instances"]] == list(range(len(logical.instances)))
    for record, prim in zip(records["instances"], logical.instances, strict=True):
        assert record == instance_record(mapped, prim.id)
        prov = prim.provenance
        assert record["kind"] == prim.kind.value and record["category"] == prim.kind.category
        assert record["cell"] == mapped.instances[prim.id].cell.name
        assert record["realizes"] == [prim.id]
        assert (record["ir_node"], record["role"], record["bit"], record["group"], record["port"]) == (
            prov.ir_node, prov.role, prov.bit, prov.group, prov.port,
        )  # fmt: skip
        assert record.get("attrs") == (dict(prov.attrs) if prov.attrs else None)
        info = logical.ir_nodes.get(prov.ir_node) if prov.ir_node is not None else None
        assert record["ir_op"] == (info.op if info else None)
        assert record["ir_type"] == (info.type.name if info else None)
        assert record["init"] == (None if prim.init is None else int(prim.init))
        assert record["peripheral"] == (None if prim.peripheral is None else prim.peripheral.to_dict())
        assert record["origin"] is None and record["orientation"] is None
        assert record["group"] is None or 0 <= record["group"] < len(records["groups"])
        if prim.is_gate:  # every gate is explained by the IR node it was synthesized for
            assert record["ir_op"] is not None
    assert [n["id"] for n in records["nets"]] == [n.id for n in logical.nets]
    for record, net in zip(records["nets"], logical.nets, strict=True):
        assert record["width"] == 1 and record["role"] == net.role
        assert record["driver"] == net.driver.ref() and record["sinks"] == [s.ref() for s in net.sinks]
    assert_plain_json(records)


def test_techmap_reports_every_choice_to_the_trace() -> None:
    netlist = synth(FULL_ADDER)
    trace = PrimitiveTraceRecorder("detailed")
    mapped = map_primitives_to_minecraft(netlist, trace=trace)
    assert {e["phase"] for e in trace.events} == {"techmap"}
    choices = [e for e in trace.events if e["type"] == "primitive_mapped"]
    assert [e["instance"] for e in choices] == [inst.id for inst in mapped.instances]
    for event, inst in zip(choices, mapped.instances, strict=True):
        assert event["kind"] == inst.kind.value and event["cell"] == inst.cell.name
        assert event["candidates"] == [c.name for c in PRIMITIVE_TECHNOLOGY.candidates(netlist.instances[inst.id])]
    (complete,) = [e for e in trace.events if e["type"] == "techmap_complete"]
    assert complete["library"] == LIBRARY_NAME
    assert (complete["instances"], complete["nets"]) == (len(mapped.instances), len(mapped.nets))
    assert complete["cells"] == dict(sorted(Counter(i.cell.name for i in mapped.instances).items()))
    assert complete["placeholder_cells"] == sorted(mapped.cells_used())
    assert complete["component_voxels"] == sum(len(i.cell.voxels) for i in mapped.instances)
    basic = PrimitiveTraceRecorder("basic")
    map_primitives_to_minecraft(netlist, trace=basic)
    assert [e["type"] for e in basic.events] == ["techmap_complete"]


def test_techmap_requires_a_complete_netlist() -> None:
    netlist = PrimitiveNetlist()
    netlist.add_instance(K.OUTPUT_BIT, Provenance(None, "output_bit"))
    with pytest.raises(CompileError, match="undriven"):
        map_primitives_to_minecraft(netlist)


# -- validate() catches tampering ------------------------------------------------------


def _wide_lever() -> PrimitiveCell:
    return placeholder_cell(
        "lever_2bit_test",
        K.PERIPHERAL,
        body=[(0, 1, 0), (0, 1, 1), (0, 1, 2)],
        pins=(PrimitivePin("b0", "out", (1, 1, 0), E), PrimitivePin("b1", "out", (1, 1, 2), E)),
        latency=None,
        description="test lever with two bits",
        peripheral=("lever", INPUT, 2),
    )


def _renamed_pins_inverter() -> PrimitiveCell:
    """A cell whose pins were changed AFTER validation (bypassing __post_init__)."""
    cell = inverter(name="tampered_inverter")
    object.__setattr__(cell, "pins", (replace(PIN_A, name="x"), PIN_Y))
    return cell


def _swap_first_two(mapped: PrimitivePhysicalNetlist) -> None:
    mapped.instances[0], mapped.instances[1] = mapped.instances[1], mapped.instances[0]


def _set(target: Any, **values: Any) -> None:
    for name, value in values.items():
        setattr(target, name, value)


TAMPERING = [
    ("realizes_another", lambda m: _set(first(m, K.AND), realizes=(first(m, K.AND).id + 1,)), "does not realize"),
    ("many_to_one", lambda m: _set(first(m, K.AND), realizes=(first(m, K.AND).id, 0)), "does not realize"),
    ("instance_id", lambda m: _set(m.instances[0], id=99), "mapped instance 99 does not realize primitive 0"),
    ("reordered_instances", _swap_first_two, "does not realize primitive 0"),
    ("dropped_instance", lambda m: m.instances.pop(), "realize every primitive exactly once"),
    ("instance_kind", lambda m: _set(first(m, K.AND), kind=K.OR), "cannot realize a and"),
    (
        "cell_of_another_kind",
        lambda m: _set(first(m, K.AND), cell=PRIMITIVE_TECHNOLOGY["or_gate_2x3_placeholder"]),
        "cell or_gate_2x3_placeholder (or) cannot realize a and",
    ),
    (
        "peripheral_of_another_width",
        lambda m: _set(next(i for i in m.instances if i.cell.name == "lever_1bit_placeholder"), cell=_wide_lever()),
        "is not a lever input peripheral of width 1",
    ),
    ("cell_pins", lambda m: _set(first(m, K.NOT), cell=_renamed_pins_inverter()), "do not match primitive pins"),
    ("net_driver", lambda m: m.nets.__setitem__(0, replace(m.nets[0], driver=m.nets[1].driver)), "differs"),
    ("net_sinks", lambda m: m.nets.__setitem__(0, replace(m.nets[0], sinks=m.nets[1].sinks)), "differs"),
    ("net_role", lambda m: m.nets.__setitem__(0, replace(m.nets[0], role="clock")), "differs"),
    ("net_id", lambda m: m.nets.__setitem__(0, replace(m.nets[0], id=7)), "differs from logical net 0"),
    ("dropped_net", lambda m: m.nets.pop(), "mirror the logical nets one to one"),
    ("extra_net", lambda m: m.nets.append(m.nets[0]), "mirror the logical nets one to one"),
]


@pytest.mark.parametrize(("tamper", "message"), [t[1:] for t in TAMPERING], ids=[t[0] for t in TAMPERING])
def test_validate_catches_tampering(tamper, message: str) -> None:
    mapped = mapped_design()
    mapped.validate()
    tamper(mapped)
    with pytest.raises(CompileError, match=re.escape(message)):
        mapped.validate()


def test_validate_reports_the_mismatched_pins() -> None:
    mapped = mapped_design()
    gate = first(mapped, K.NOT)
    gate.cell = _renamed_pins_inverter()
    with pytest.raises(CompileError) as info:
        mapped.validate()
    message = str(info.value)
    assert message.startswith(f"mapped instance {gate.id}: cell tampered_inverter pins ")
    assert "{'x': 'in', 'y': 'out'}" in message and "{'a': 'in', 'y': 'out'}" in message
