import pytest

from redc import BOOL, CompileError, IRType
from redc.physical import (
    Component,
    Face,
    Operation,
    Port,
    PortDir,
    PrimitiveGate,
    Wiring,
)

BYTE = IRType(8)


def make_add() -> Operation:
    """A concrete-style Operation built directly from the data model.

    Stand-in for a future ``uint8_add_a-0-0-0_b-0-0-1_out-0-0-2`` subclass:
    two inputs on WEST, one output on EAST, stacked along +z.
    """
    return Operation(
        name="uint8_add",
        latency=0,
        dim=(1, 1, 3),
        inputs=(
            Port("a", BYTE, Face.WEST, (0, 0, 0), PortDir.IN),
            Port("b", BYTE, Face.WEST, (0, 0, 1), PortDir.IN),
        ),
        outputs=(Port("out", BYTE, Face.EAST, (0, 0, 2), PortDir.OUT),),
        op="add",
    )


def test_component_exposes_typed_faced_ports() -> None:
    add = make_add()
    assert add.is_combinational
    assert add.op == "add"
    assert [p.name for p in add.inputs] == ["a", "b"]
    assert all(p.face is Face.WEST for p in add.inputs)
    assert add.outputs[0].face is Face.EAST
    assert all(p.dtype == BYTE for p in add.ports)


def test_ports_resolve_to_absolute_grid_cells() -> None:
    add = make_add()
    b_pin = add.inputs[1]
    assert b_pin.absolute((10, 0, 5)) == (10, 0, 6)


def test_face_normal_points_outward() -> None:
    assert Face.EAST.normal == (1, 0, 0)
    assert Face.TOP.normal == (0, 1, 0)
    assert Face.NORTH.normal == (0, 0, -1)
    assert Face.WEST.axis == 0


def test_latency_marks_sequential() -> None:
    wire = Wiring(
        name="buf",
        latency=2,
        dim=(1, 1, 1),
        inputs=(Port("in", BYTE, Face.WEST, (0, 0, 0), PortDir.IN),),
        outputs=(Port("out", BYTE, Face.EAST, (0, 0, 0), PortDir.OUT),),
    )
    assert not wire.is_combinational


def test_bad_pin_placement_is_rejected() -> None:
    # Claims EAST but sits at x == 0 in a 2-wide box.
    bad = Port("out", BYTE, Face.EAST, (0, 0, 0), PortDir.OUT)
    with pytest.raises(CompileError):
        Component(name="bad", latency=0, dim=(2, 1, 1), inputs=(), outputs=(bad,))

    # Sits outside the footprint entirely.
    oob = Port("out", BYTE, Face.EAST, (0, 0, 9), PortDir.OUT)
    with pytest.raises(CompileError):
        Component(name="oob", latency=0, dim=(1, 1, 1), inputs=(), outputs=(oob,))


def test_gate_family_carries_op() -> None:
    inv = PrimitiveGate(
        name="not",
        latency=1,
        dim=(1, 1, 1),
        inputs=(Port("in0", BOOL, Face.WEST, (0, 0, 0), PortDir.IN),),
        outputs=(Port("out", BOOL, Face.EAST, (0, 0, 0), PortDir.OUT),),
        op="not",
    )
    assert inv.op == "not"
    assert inv.inputs[0].dtype == BOOL


def test_footprint_cells_cover_the_box() -> None:
    add = make_add()
    cells = list(add.footprint_cells((0, 0, 0)))
    assert len(cells) == add.volume == 3
