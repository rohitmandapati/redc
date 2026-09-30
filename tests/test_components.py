import pytest

from redc import BOOL, CompileError, IRType
from redc.physical import (
    ClockSource,
    Component,
    Constant,
    Face,
    InputPad,
    Operation,
    OutputPad,
    Port,
    PortDir,
    PrimitiveGate,
    Register,
    TypeCast,
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


# -- behavior (functional model) ------------------------------------------


def test_operation_behavior_matches_ir_semantics() -> None:
    add = make_add()
    assert add.behavior({"a": 40, "b": 2}) == {"out": 42}
    # wraps at the datatype width like the IR does
    assert add.behavior({"a": 200, "b": 100}) == {"out": (300) & 0xFF}


def make_nand() -> PrimitiveGate:
    return PrimitiveGate(
        name="uint8_nand",
        latency=1,
        dim=(1, 1, 3),
        inputs=(
            Port("in0", BYTE, Face.WEST, (0, 0, 0), PortDir.IN),
            Port("in1", BYTE, Face.WEST, (0, 0, 1), PortDir.IN),
        ),
        outputs=(Port("out", BYTE, Face.EAST, (0, 0, 2), PortDir.OUT),),
        op="nand",
    )


def test_gate_behavior_is_bitwise() -> None:
    assert make_nand().behavior({"in0": 0xF0, "in1": 0xFF}) == {"out": 0x0F}


def test_wiring_behavior_is_identity() -> None:
    buf = Wiring(
        name="buf",
        latency=1,
        dim=(1, 1, 1),
        inputs=(Port("in", BYTE, Face.WEST, (0, 0, 0), PortDir.IN),),
        outputs=(Port("out", BYTE, Face.EAST, (0, 0, 0), PortDir.OUT),),
    )
    assert buf.behavior({"in": 0xAB}) == {"out": 0xAB}


def make_cast() -> TypeCast:
    return TypeCast(
        name="uint8_to_uint4",
        latency=0,
        dim=(1, 1, 1),
        inputs=(Port("in", IRType(8), Face.WEST, (0, 0, 0), PortDir.IN),),
        outputs=(Port("out", IRType(4), Face.EAST, (0, 0, 0), PortDir.OUT),),
    )


def test_typecast_truncates_and_reports_widths() -> None:
    cast = make_cast()
    assert cast.behavior({"in": 0x1F}) == {"out": 0xF}
    assert cast.source_width == 8
    assert cast.result_width == 4


def test_missing_input_pin_is_a_compile_error() -> None:
    with pytest.raises(CompileError):
        make_add().behavior({"a": 1})


# -- registers (stateful) --------------------------------------------------


def make_register() -> Register:
    return Register(
        name="uint8_register",
        latency=1,
        dim=(1, 1, 3),
        init=0,
        inputs=(
            Port("next", BYTE, Face.WEST, (0, 0, 0), PortDir.IN),
            Port("enable", BOOL, Face.WEST, (0, 0, 1), PortDir.IN),
            Port("clk", BOOL, Face.BOTTOM, (0, 0, 2), PortDir.IN),
        ),
        outputs=(Port("out", BYTE, Face.EAST, (0, 0, 2), PortDir.OUT),),
    )


def test_register_holds_and_updates() -> None:
    reg = make_register()
    assert reg.is_stateful
    assert reg.initial() == 0
    assert reg.read(7) == {"out": 7}
    assert reg.step(7, {"next": 42, "enable": 1}) == 42  # enabled: latch next
    assert reg.step(7, {"next": 42, "enable": 0}) == 7  # disabled: hold
    assert reg.index_keys() == frozenset({("register", 8)})


def test_register_requires_its_control_pins() -> None:
    with pytest.raises(CompileError):  # missing enable/clk
        Register(
            name="bad",
            latency=1,
            dim=(1, 1, 1),
            inputs=(Port("next", BYTE, Face.WEST, (0, 0, 0), PortDir.IN),),
            outputs=(Port("out", BYTE, Face.EAST, (0, 0, 0), PortDir.OUT),),
        )


# -- boundary / source cells ----------------------------------------------


def test_constant_is_a_pure_source() -> None:
    const = Constant.of(5, BYTE)
    assert const.is_source
    assert const.behavior({}) == {"out": 5}


def test_input_and_output_pads() -> None:
    pad_in = InputPad.of("x", BYTE)
    assert pad_in.is_source
    assert pad_in.outputs[0].name == "out" and not pad_in.inputs

    pad_out = OutputPad.of("y", BYTE)
    assert pad_out.is_sink
    assert pad_out.behavior({"in": 9}) == {}


def test_clock_source_emits_one_bit() -> None:
    clk = ClockSource.of(period=4)
    assert clk.is_source
    assert clk.period == 4
    assert clk.outputs[0].dtype == BOOL
