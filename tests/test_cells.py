from pathlib import Path

import pytest
import yaml

from redc import BOOL, CompileError, IRType
from redc.ir import SHIFT_AMOUNT
from redc.physical import (
    GATES,
    LIBRARY,
    OPERATIONS,
    PHYSICAL_TYPES,
    REGISTERS,
    TYPE_CASTS,
    WIRING,
    Face,
    Operation,
    OperationSignature,
    PortDir,
    Register,
    TypeCast,
)
from redc.physical.cells import _loader

CELLS = Path(_loader.__file__).parent
INTS = [t for t in PHYSICAL_TYPES if not t.boolean]
UINT8, INT8 = IRType(8), IRType(8, signed=True)
sig = OperationSignature.of


def test_families_load_from_yaml() -> None:
    assert OPERATIONS and GATES and WIRING and REGISTERS and TYPE_CASTS
    # LIBRARY is the merge of every enumerated family, keyed by convention name.
    assert len(LIBRARY) == sum(
        len(fam) for fam in (OPERATIONS, GATES, WIRING, REGISTERS, TYPE_CASTS)
    )
    assert "uint8_add_a-0-0-0_b-0-0-1_out-0-0-2" in LIBRARY


def test_loaded_operation_matches_its_contract() -> None:
    add = OPERATIONS["uint8_add_a-0-0-0_b-0-0-1_out-0-0-2"]
    assert isinstance(add, Operation)
    assert add.op == "add"
    assert add.latency is None  # unknown until a real circuit is measured
    assert add.dim == (1, 1, 3)
    assert add.nbt is None  # no structure file exists yet
    assert [p.name for p in add.inputs] == ["a", "b"]
    assert add.inputs[0].dtype == UINT8
    assert add.inputs[0].face is Face.WEST
    assert add.inputs[0].direction is PortDir.IN
    assert add.outputs[0].face is Face.EAST
    assert add.outputs[0].offset == (0, 0, 2)


def test_no_cell_references_a_missing_nbt_file() -> None:
    for cell in LIBRARY.values():
        assert cell.nbt is None or (CELLS / cell.nbt).exists(), cell.name


def test_gate_datatype_default_is_bool() -> None:
    inv = GATES["bool_not_in0-0-0-0_out-0-0-0"]
    assert inv.op == "not"
    assert inv.inputs[0].dtype.boolean
    assert len(inv.inputs) == 1
    assert inv.latency == 1 and inv.is_combinational


def test_variants_share_signature_but_differ_in_layout() -> None:
    cells = LIBRARY.cells(sig("add", UINT8, UINT8, UINT8))
    assert [c.name for c in cells] == [
        "uint8_add_a-0-0-0_b-0-0-1_out-0-0-2",  # YAML order, deterministic
        "uint8_add_a-0-0-0_b-0-1-0_out-0-2-0",
    ]
    assert cells[0].dim != cells[1].dim  # the whole point: pickable layouts


def test_lookup_is_by_full_type_not_width() -> None:
    unsigned = LIBRARY.cells(sig("add", UINT8, UINT8, UINT8))
    signed = LIBRARY.cells(sig("add", INT8, INT8, INT8))
    assert unsigned and signed
    assert not {c.name for c in unsigned} & {c.name for c in signed}
    # Mixed-signedness is not a valid signature at all.
    with pytest.raises(CompileError):
        sig("add", UINT8, UINT8, INT8)


def test_every_integer_type_has_arithmetic_compare_shift_and_mux() -> None:
    for t in INTS:
        for op in ("add", "sub", "mul", "div", "mod", "and", "or", "xor"):
            assert LIBRARY.cells(sig(op, t, t, t)), (op, t)
        for op in ("neg", "inv"):
            assert LIBRARY.cells(sig(op, t, t)), (op, t)
        for op in ("eq", "ne", "lt", "le", "gt", "ge"):
            assert LIBRARY.cells(sig(op, BOOL, t, t)), (op, t)
        for op in ("shl", "shr"):
            assert LIBRARY.cells(sig(op, t, t, SHIFT_AMOUNT)), (op, t)
    for t in PHYSICAL_TYPES:
        assert LIBRARY.cells(sig("mux", t, BOOL, t, t)), t
        assert LIBRARY.cells(sig("register", t, t, BOOL)), t
    for op in ("and", "or", "xor"):
        assert LIBRARY.cells(sig(op, BOOL, BOOL, BOOL))
    assert LIBRARY.cells(sig("not", BOOL, BOOL))


def test_unsupported_widths_have_no_cells() -> None:
    uint3 = IRType(3)
    assert LIBRARY.cells(sig("add", uint3, uint3, uint3)) == ()
    assert LIBRARY.cells(sig("cast", IRType(4), uint3)) == ()


def test_register_cells_expose_the_five_pin_contract() -> None:
    for reg in REGISTERS.values():
        assert isinstance(reg, Register)
        assert reg.is_stateful and not reg.is_combinational
        assert [p.name for p in reg.inputs] == ["next", "enable", "clk", "rst"]
        assert [p.name for p in reg.outputs] == ["out"]
        assert not hasattr(reg, "init")  # reset value is per instance


def test_register_yaml_has_no_init() -> None:
    data = yaml.safe_load((CELLS / "register.yaml").read_text())
    for spec in data["variants"].values():
        assert "init" not in spec


def test_loader_rejects_per_cell_init(tmp_path, monkeypatch) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
variants:
  uint8_reg:
    datatype: uint8
    latency: 1
    init: 7
    dim: [1, 1, 3]
    inputs:
      - {name: next, face: WEST, offset: [0, 0, 0]}
      - {name: enable, datatype: bool, face: WEST, offset: [0, 0, 1]}
      - {name: clk, datatype: bool, face: BOTTOM, offset: [0, 0, 2]}
      - {name: rst, datatype: bool, face: BOTTOM, offset: [0, 0, 1]}
    outputs:
      - {name: out, face: EAST, offset: [0, 0, 2]}
"""
    )
    monkeypatch.setattr(_loader, "__file__", str(tmp_path / "_loader.py"))
    with pytest.raises(CompileError, match="init"):
        _loader.load_family("bad.yaml", Register)


def test_loader_preserves_declared_input_order(tmp_path, monkeypatch) -> None:
    (tmp_path / "order.yaml").write_text(
        """
variants:
  uint8_sub_z-a:
    op: sub
    datatype: uint8
    latency: 0
    dim: [1, 1, 3]
    inputs:
      - {name: zeta, face: WEST, offset: [0, 0, 0]}
      - {name: alpha, face: WEST, offset: [0, 0, 1]}
    outputs:
      - {name: out, face: EAST, offset: [0, 0, 2]}
"""
    )
    monkeypatch.setattr(_loader, "__file__", str(tmp_path / "_loader.py"))
    cell = _loader.load_family("order.yaml", Operation)["uint8_sub_z-a"]
    assert [p.name for p in cell.inputs] == ["zeta", "alpha"]  # never sorted
    # Declared order is operand order: zeta - alpha.
    assert cell.behavior({"zeta": 10, "alpha": 3}) == {"out": 7}


def test_variable_shift_cells_take_a_uint64_amount() -> None:
    for t in INTS:
        for op in ("shl", "shr"):
            (cell,) = LIBRARY.cells(sig(op, t, t, SHIFT_AMOUNT))
            assert [p.name for p in cell.inputs] == ["value", "amount"]
            assert cell.inputs[0].dtype == t
            assert cell.inputs[1].dtype == IRType(64)
            assert cell.outputs[0].dtype == t


def test_signed_and_unsigned_shr_are_distinct_cells() -> None:
    (logical,) = LIBRARY.cells(sig("shr", UINT8, UINT8, SHIFT_AMOUNT))
    (arithmetic,) = LIBRARY.cells(sig("shr", INT8, INT8, SHIFT_AMOUNT))
    assert logical.name != arithmetic.name
    assert logical.behavior({"value": 0x80, "amount": 1}) == {"out": 0x40}
    assert arithmetic.behavior({"value": 0x80, "amount": 1}) == {"out": 0xC0}


def test_signed_and_unsigned_comparisons_are_distinct_cells() -> None:
    (unsigned,) = LIBRARY.cells(sig("lt", BOOL, UINT8, UINT8))
    (signed,) = LIBRARY.cells(sig("lt", BOOL, INT8, INT8))
    assert unsigned.name != signed.name
    # 0xFF is 255 unsigned but -1 signed.
    assert unsigned.behavior({"a": 0xFF, "b": 1}) == {"out": 0}
    assert signed.behavior({"a": 0xFF, "b": 1}) == {"out": 1}


def test_cast_index_by_full_source_and_result_type() -> None:
    uint16, int16 = IRType(16), IRType(16, signed=True)
    pairs = [(UINT8, uint16), (INT8, int16), (INT8, uint16), (UINT8, BOOL)]
    names = set()
    for src, dst in pairs:
        (cast,) = LIBRARY.cells(sig("cast", dst, src))
        assert isinstance(cast, TypeCast)
        assert cast.source_type == src and cast.result_type == dst
        names.add(cast.name)
    assert len(names) == len(pairs)
    # One cell per ordered pair of distinct supported types; no identity casts.
    n = len(PHYSICAL_TYPES)
    assert len(TYPE_CASTS) == n * (n - 1)
    assert LIBRARY.cells(sig("cast", UINT8, UINT8)) == ()
