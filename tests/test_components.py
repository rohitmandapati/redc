import itertools

import pytest

from redc import BOOL, CompileError, IRType
from redc.ir import SHIFT_AMOUNT, _calculate
from redc.physical import (
    LIBRARY,
    PHYSICAL_TYPES,
    ClockSource,
    Component,
    Constant,
    Face,
    InputPad,
    Operation,
    OperationSignature,
    OutputPad,
    Port,
    PortDir,
    PrimitiveGate,
    Register,
    ResetSource,
    TypeCast,
    Wiring,
)

BYTE = IRType(8)
INT8 = IRType(8, signed=True)
sig = OperationSignature.of


def make_add() -> Operation:
    """A concrete-style Operation built directly from the data model: two
    inputs on WEST, one output on EAST, stacked along +z."""
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
    assert add.signatures() == (sig("add", BYTE, BYTE, BYTE),)


def test_ports_resolve_to_absolute_grid_cells() -> None:
    add = make_add()
    b_pin = add.inputs[1]
    assert b_pin.absolute((10, 0, 5)) == (10, 0, 6)


def test_face_normal_points_outward() -> None:
    assert Face.EAST.normal == (1, 0, 0)
    assert Face.TOP.normal == (0, 1, 0)
    assert Face.NORTH.normal == (0, 0, -1)
    assert Face.WEST.axis == 0


# -- latency is delay, not state ---------------------------------------------


def test_combinational_cell_with_latency_stays_combinational() -> None:
    wire = Wiring(
        name="buf",
        latency=2,
        dim=(1, 1, 1),
        inputs=(Port("in", BYTE, Face.WEST, (0, 0, 0), PortDir.IN),),
        outputs=(Port("out", BYTE, Face.EAST, (0, 0, 0), PortDir.OUT),),
    )
    assert wire.is_combinational and not wire.is_stateful
    gate = LIBRARY["bool_not_in0-0-0-0_out-0-0-0"]
    assert gate.latency == 1 and gate.is_combinational


def test_zero_tick_cell_is_combinational() -> None:
    add = make_add()
    assert add.latency == 0 and add.is_combinational and not add.is_stateful


def test_register_is_stateful_regardless_of_latency() -> None:
    for latency in (0, 1, 5):
        reg = make_register(latency=latency)
        assert reg.is_stateful and not reg.is_combinational


def test_bad_pin_placement_is_rejected() -> None:
    # Claims EAST but sits at x == 0 in a 2-wide box.
    bad = Port("out", BYTE, Face.EAST, (0, 0, 0), PortDir.OUT)
    with pytest.raises(CompileError):
        Component(name="bad", latency=0, dim=(2, 1, 1), inputs=(), outputs=(bad,))

    # Sits outside the footprint entirely.
    oob = Port("out", BYTE, Face.EAST, (0, 0, 9), PortDir.OUT)
    with pytest.raises(CompileError):
        Component(name="oob", latency=0, dim=(1, 1, 1), inputs=(), outputs=(oob,))


@pytest.mark.parametrize(
    "typ", [IRType(3), IRType(5, signed=True), IRType(7), IRType(12), IRType(1)]
)
def test_unsupported_physical_types_cannot_be_pins(typ: IRType) -> None:
    with pytest.raises(CompileError, match="not supported by the Minecraft"):
        InputPad.of("x", typ)


def test_shift_cell_needs_its_amount_operand() -> None:
    with pytest.raises(CompileError, match="operand count"):
        Operation(
            name="uint8_shl_const",
            latency=0,
            dim=(1, 1, 1),
            inputs=(Port("value", BYTE, Face.WEST, (0, 0, 0), PortDir.IN),),
            outputs=(Port("out", BYTE, Face.EAST, (0, 0, 0), PortDir.OUT),),
            op="shl",
        )
    with pytest.raises(CompileError, match="shift type"):  # amount must be uint64
        Operation(
            name="uint8_shl_narrow",
            latency=0,
            dim=(1, 1, 2),
            inputs=(
                Port("value", BYTE, Face.WEST, (0, 0, 0), PortDir.IN),
                Port("amount", BYTE, Face.WEST, (0, 0, 1), PortDir.IN),
            ),
            outputs=(Port("out", BYTE, Face.EAST, (0, 0, 1), PortDir.OUT),),
            op="shl",
        )


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


def test_bool_gates_match_ir() -> None:
    for op in ("and", "or", "xor"):
        (gate,) = LIBRARY.cells(sig(op, BOOL, BOOL, BOOL))
        for a, b in itertools.product((0, 1), repeat=2):
            want = _calculate(op, [a, b], BOOL, [BOOL, BOOL])
            assert gate.behavior({"in0": a, "in1": b}) == {"out": want}
    (inv,) = LIBRARY.cells(sig("not", BOOL, BOOL))
    for a in (0, 1):
        assert inv.behavior({"in0": a}) == {"out": _calculate("not", [a], BOOL, [BOOL])}


def test_wiring_behavior_is_identity() -> None:
    buf = Wiring(
        name="buf",
        latency=1,
        dim=(1, 1, 1),
        inputs=(Port("in", BYTE, Face.WEST, (0, 0, 0), PortDir.IN),),
        outputs=(Port("out", BYTE, Face.EAST, (0, 0, 0), PortDir.OUT),),
    )
    assert buf.behavior({"in": 0xAB}) == {"out": 0xAB}


UINT4 = IRType(4)


def make_cast(src: IRType = BYTE, dst: IRType = UINT4) -> TypeCast:
    return TypeCast(
        name=f"{src.name}_to_{dst.name}",
        latency=0,
        dim=(1, 1, 1),
        inputs=(Port("in", src, Face.WEST, (0, 0, 0), PortDir.IN),),
        outputs=(Port("out", dst, Face.EAST, (0, 0, 0), PortDir.OUT),),
    )


def test_typecast_truncates_and_reports_types() -> None:
    cast = make_cast()
    assert cast.behavior({"in": 0x1F}) == {"out": 0xF}
    assert cast.source_type == BYTE
    assert cast.result_type == IRType(4)
    assert cast.signatures() == (sig("cast", IRType(4), BYTE),)


def test_missing_input_pin_is_a_compile_error() -> None:
    with pytest.raises(CompileError):
        make_add().behavior({"a": 1})


# -- signed semantics: physical behavior == IR _calculate --------------------

INT8_EDGES = [-128, -1, 0, 1, 127]
UINT8_EDGES = [0, 1, 127, 128, 255]


def _bits(typ: IRType, number: int) -> int:
    return typ.bits(number)


@pytest.mark.parametrize(
    "op", ["add", "sub", "mul", "div", "mod", "and", "or", "xor", "eq", "ne", "lt", "le", "gt", "ge"]
)
@pytest.mark.parametrize("typ,edges", [(INT8, INT8_EDGES), (BYTE, UINT8_EDGES)])
def test_binary_cells_match_ir(op: str, typ: IRType, edges: list[int]) -> None:
    result = BOOL if op in {"eq", "ne", "lt", "le", "gt", "ge"} else typ
    cells = LIBRARY.cells(sig(op, result, typ, typ))
    assert cells
    for cell, (a, b) in itertools.product(cells, itertools.product(edges, repeat=2)):
        args = [_bits(typ, a), _bits(typ, b)]
        want = _calculate(op, args, result, [typ, typ])
        assert cell.behavior({"a": args[0], "b": args[1]}) == {"out": want}, (op, a, b)


@pytest.mark.parametrize("op", ["neg", "inv"])
@pytest.mark.parametrize("typ,edges", [(INT8, INT8_EDGES), (BYTE, UINT8_EDGES)])
def test_unary_cells_match_ir(op: str, typ: IRType, edges: list[int]) -> None:
    (cell,) = LIBRARY.cells(sig(op, typ, typ))
    for a in edges:
        want = _calculate(op, [_bits(typ, a)], typ, [typ])
        assert cell.behavior({"a": _bits(typ, a)}) == {"out": want}


@pytest.mark.parametrize("op", ["shl", "shr"])
@pytest.mark.parametrize("typ,edges", [(INT8, INT8_EDGES), (BYTE, UINT8_EDGES)])
def test_shift_cells_match_ir(op: str, typ: IRType, edges: list[int]) -> None:
    (cell,) = LIBRARY.cells(sig(op, typ, typ, SHIFT_AMOUNT))
    for a, amount in itertools.product(edges, [0, 1, 3, 7, 8, 9, 63, 2**64 - 1]):
        want = _calculate(op, [_bits(typ, a), amount], typ, [typ, SHIFT_AMOUNT])
        got = cell.behavior({"value": _bits(typ, a), "amount": amount})
        assert got == {"out": want}, (op, a, amount)


def test_signed_division_and_modulo_truncate_toward_zero() -> None:
    (div,) = LIBRARY.cells(sig("div", INT8, INT8, INT8))
    (mod,) = LIBRARY.cells(sig("mod", INT8, INT8, INT8))
    assert INT8.number(div.behavior({"a": _bits(INT8, -7), "b": 2})["out"]) == -3
    assert INT8.number(mod.behavior({"a": _bits(INT8, -7), "b": 2})["out"]) == -1
    (udiv,) = LIBRARY.cells(sig("div", BYTE, BYTE, BYTE))
    assert udiv.behavior({"a": _bits(BYTE, -7), "b": 2}) == {"out": 249 // 2}


def test_signed_arithmetic_right_shift() -> None:
    (sra,) = LIBRARY.cells(sig("shr", INT8, INT8, SHIFT_AMOUNT))
    assert INT8.number(sra.behavior({"value": _bits(INT8, -128), "amount": 3})["out"]) == -16
    # Shifting past the width saturates to the sign, as in the IR.
    assert INT8.number(sra.behavior({"value": _bits(INT8, -1), "amount": 200})["out"]) == -1


def test_mux_cells_match_ir() -> None:
    for typ in PHYSICAL_TYPES:
        (mux,) = LIBRARY.cells(sig("mux", typ, BOOL, typ, typ))
        assert [p.name for p in mux.inputs] == ["sel", "yes", "no"]
        assert mux.behavior({"sel": 1, "yes": 1, "no": 0}) == {"out": 1}
        assert mux.behavior({"sel": 0, "yes": 1, "no": 0}) == {"out": 0}


def test_every_library_cast_matches_ir() -> None:
    for src, dst in itertools.permutations(PHYSICAL_TYPES, 2):
        (cast,) = LIBRARY.cells(sig("cast", dst, src))
        for number in (src.number(0), 1, -1, src.number(src.mask), src.number(1 << (src.width - 1))):
            bits = src.bits(number)
            want = _calculate("cast", [bits], dst, [src])
            assert cast.behavior({"in": bits}) == {"out": want}, (src, dst, number)


def test_cast_semantics_are_exact() -> None:
    uint16, int16 = IRType(16), IRType(16, signed=True)

    def cast(src: IRType, dst: IRType, number: int) -> int:
        (cell,) = LIBRARY.cells(sig("cast", dst, src))
        return dst.number(cell.behavior({"in": src.bits(number)})["out"])

    assert cast(BYTE, uint16, 0xFF) == 0x00FF  # zero extension
    assert cast(INT8, int16, -1) == -1  # sign extension
    assert cast(INT8, int16, -128) == -128
    assert cast(INT8, uint16, -1) == 0xFFFF  # sign-extend, then reinterpret
    assert cast(uint16, BYTE, 0x1234) == 0x34  # truncation
    assert cast(int16, INT8, 0x0180) == -128  # truncation to destination width
    assert cast(BYTE, BOOL, 2) == 1  # integer -> bool is != 0
    assert cast(BYTE, BOOL, 0) == 0
    assert cast(INT8, BOOL, -128) == 1
    assert cast(BOOL, BYTE, 1) == 1  # bool -> integer is 0 or 1
    assert cast(BOOL, INT8, 1) == 1


# -- registers (stateful) --------------------------------------------------


def make_register(dtype: IRType = BYTE, latency: int = 1) -> Register:
    return Register(
        name=f"{dtype.name}_register",
        latency=latency,
        dim=(1, 1, 3),
        inputs=(
            Port("next", dtype, Face.WEST, (0, 0, 0), PortDir.IN),
            Port("enable", BOOL, Face.WEST, (0, 0, 1), PortDir.IN),
            Port("clk", BOOL, Face.BOTTOM, (0, 0, 2), PortDir.IN),
            Port("rst", BOOL, Face.BOTTOM, (0, 0, 1), PortDir.IN),
        ),
        outputs=(Port("out", dtype, Face.EAST, (0, 0, 2), PortDir.OUT),),
    )


def test_register_holds_and_updates() -> None:
    reg = make_register()
    assert reg.is_stateful
    assert reg.read(7) == {"out": 7}
    assert reg.step(7, {"next": 42, "enable": 1, "rst": 0}, init=0) == 42  # latch
    assert reg.step(7, {"next": 42, "enable": 0, "rst": 0}, init=0) == 7  # hold
    assert reg.signatures() == (sig("register", BYTE, BYTE, BOOL),)
    assert [p.name for p in reg.operand_inputs] == ["next", "enable"]


def test_register_reset_loads_instance_init_and_wins_over_enable() -> None:
    reg = make_register()
    assert reg.reset_value(7) == 7
    assert reg.step(42, {"next": 9, "enable": 0, "rst": 1}, init=7) == 7
    assert reg.step(42, {"next": 9, "enable": 1, "rst": 1}, init=7) == 7  # reset wins
    assert reg.step(42, {"next": 9, "enable": 1, "rst": 0}, init=7) == 9


def test_register_init_is_validated_and_masked() -> None:
    signed = make_register(INT8)
    assert signed.check_init(-1) == 0xFF
    assert signed.check_init(-128) == 0x80
    assert signed.check_init(0xFF) == 0xFF  # raw bit pattern also accepted
    for bad in (-129, 256):
        with pytest.raises(CompileError):
            signed.check_init(bad)
    unsigned = make_register(BYTE)
    with pytest.raises(CompileError):
        unsigned.check_init(-1)
    with pytest.raises(CompileError):
        make_register(BOOL).check_init(2)


def test_register_requires_its_control_pins() -> None:
    with pytest.raises(CompileError):  # missing enable/clk/rst
        Register(
            name="bad",
            latency=1,
            dim=(1, 1, 1),
            inputs=(Port("next", BYTE, Face.WEST, (0, 0, 0), PortDir.IN),),
            outputs=(Port("out", BYTE, Face.EAST, (0, 0, 0), PortDir.OUT),),
        )
    with pytest.raises(CompileError, match="rst"):  # the old 4-pin contract
        Register(
            name="no_reset",
            latency=1,
            dim=(1, 1, 3),
            inputs=(
                Port("next", BYTE, Face.WEST, (0, 0, 0), PortDir.IN),
                Port("enable", BOOL, Face.WEST, (0, 0, 1), PortDir.IN),
                Port("clk", BOOL, Face.BOTTOM, (0, 0, 2), PortDir.IN),
            ),
            outputs=(Port("out", BYTE, Face.EAST, (0, 0, 2), PortDir.OUT),),
        )


# -- boundary / source cells ----------------------------------------------


def test_constant_is_a_pure_source() -> None:
    const = Constant.of(5, BYTE)
    assert const.is_source
    assert const.behavior({}) == {"out": 5}
    # Signedness is part of a constant's identity.
    assert Constant.of(-1, INT8).name != Constant.of(255, BYTE).name


def test_input_and_output_pads() -> None:
    pad_in = InputPad.of("x", BYTE)
    assert pad_in.is_source
    assert pad_in.outputs[0].name == "out" and not pad_in.inputs

    pad_out = OutputPad.of("y", BYTE)
    assert pad_out.is_sink
    assert pad_out.behavior({"in": 9}) == {}


def test_clock_and_reset_sources_emit_one_bit() -> None:
    clk = ClockSource.of(period=4)
    assert clk.is_source and clk.is_combinational
    assert clk.period == 4
    assert clk.outputs[0].dtype == BOOL
    rst = ResetSource.of()
    assert rst.is_source
    assert [(p.name, p.dtype) for p in rst.outputs] == [("rst", BOOL)]
