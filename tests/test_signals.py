import pytest

from redc import BOOL, CompileError, IRType, compile_source, emit_systemverilog
from redc.physical import (
    PHYSICAL_TYPES,
    SignalEncoding,
    is_supported_physical_type,
    require_supported_physical_type,
    signal_layout,
)

SUPPORTED = [BOOL] + [
    IRType(width, signed=signed) for width in (4, 8, 16, 32, 64) for signed in (False, True)
]
UNSUPPORTED = [
    IRType(3),  # uint3
    IRType(5, signed=True),  # int5
    IRType(7),  # uint7
    IRType(12),  # uint12
    IRType(24, signed=True),  # int24
    IRType(1),  # uint1 is not bool
    IRType(1, signed=True),
    IRType(63),
]


def test_supported_type_set_is_exact() -> None:
    assert set(PHYSICAL_TYPES) == set(SUPPORTED)
    assert len(PHYSICAL_TYPES) == 11


@pytest.mark.parametrize("typ", SUPPORTED, ids=lambda t: t.name)
def test_supported_types_are_accepted(typ: IRType) -> None:
    assert is_supported_physical_type(typ)
    assert require_supported_physical_type(typ) is typ


@pytest.mark.parametrize("typ", UNSUPPORTED, ids=lambda t: t.name)
def test_arbitrary_widths_are_rejected(typ: IRType) -> None:
    assert not is_supported_physical_type(typ)
    with pytest.raises(CompileError, match=typ.name):
        require_supported_physical_type(typ)


def test_ir_and_systemverilog_still_accept_any_width() -> None:
    # The restriction is physical only: uint3 compiles and emits as before.
    graph = compile_source("uint3 main(uint3 a, uint3 b) { return a + b; }")
    assert graph.evaluate(in_a=7, in_b=2)["result"] == 1
    assert "logic [2:0]" in emit_systemverilog(graph)


def test_type_names() -> None:
    assert [t.name for t in PHYSICAL_TYPES] == [
        "bool", "uint4", "int4", "uint8", "int8", "uint16", "int16",
        "uint32", "int32", "uint64", "int64",
    ]  # fmt: skip


@pytest.mark.parametrize("width,lanes", [(16, 4), (32, 8), (64, 16)])
def test_wide_values_use_hex_nibble_lanes(width: int, lanes: int) -> None:
    for signed in (False, True):
        layout = signal_layout(IRType(width, signed=signed))
        assert layout.encoding is SignalEncoding.HEX
        assert layout.lane_width == 4
        assert layout.lane_count == lanes
        assert layout.bits == width


@pytest.mark.parametrize("width", [4, 8])
def test_narrow_values_use_binary_lanes(width: int) -> None:
    for signed in (False, True):
        layout = signal_layout(IRType(width, signed=signed))
        assert layout.encoding is SignalEncoding.BINARY
        assert (layout.lane_width, layout.lane_count) == (1, width)


def test_bool_is_a_single_line() -> None:
    layout = signal_layout(BOOL)
    assert layout.encoding is SignalEncoding.BOOL
    assert (layout.lane_width, layout.lane_count) == (1, 1)


def test_signedness_never_changes_layout() -> None:
    for width in (4, 8, 16, 32, 64):
        unsigned = signal_layout(IRType(width))
        signed = signal_layout(IRType(width, signed=True))
        assert unsigned.to_dict() == signed.to_dict()


def test_layout_of_unsupported_type_is_an_error() -> None:
    with pytest.raises(CompileError):
        signal_layout(IRType(12))
