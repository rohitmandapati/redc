from redc import IRType
from redc.physical import (
    GATES,
    LIBRARY,
    OPERATIONS,
    REGISTERS,
    TYPE_CASTS,
    WIRING,
    Face,
    Operation,
    PortDir,
)


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
    assert add.latency == 0
    assert add.dim == (1, 1, 3)
    assert add.nbt == "uint8_add_a-0-0-0_b-0-0-1_out-0-0-2.nbt"
    assert [p.name for p in add.inputs] == ["a", "b"]
    assert add.inputs[0].dtype == IRType(8)  # uint8 family default
    assert add.inputs[0].face is Face.WEST
    assert add.inputs[0].direction is PortDir.IN
    assert add.outputs[0].face is Face.EAST
    assert add.outputs[0].offset == (0, 0, 2)


def test_gate_datatype_default_is_bool() -> None:
    inv = GATES["bool_not_in0-0-0-0_out-0-0-0"]
    assert inv.op == "not"
    assert inv.inputs[0].dtype.boolean
    assert len(inv.inputs) == 1


def test_variants_share_op_but_differ_in_layout() -> None:
    a = OPERATIONS["uint8_add_a-0-0-0_b-0-0-1_out-0-0-2"]
    b = OPERATIONS["uint8_add_a-0-0-0_b-0-1-0_out-0-2-0"]
    assert a.op == b.op == "add"
    assert a.dim != b.dim  # the whole point: pickable layout variations


def test_op_width_index_returns_all_variants() -> None:
    # Both add layouts are reachable by (op, width) for the tech-mapper.
    variants = LIBRARY.variants("add", 8)
    assert len(variants) == 2
    assert all(v.op == "add" for v in variants)
    assert LIBRARY.variants("add", 16) == ()  # no uint16 cells yet


def test_register_indexed_by_data_width() -> None:
    regs = LIBRARY.variants("register", 8)
    assert regs and all(r.is_stateful for r in regs)


def test_cast_index_by_source_and_result_width() -> None:
    casts = LIBRARY.casts(8, 4)
    assert casts
    assert casts[0].source_width == 8 and casts[0].result_width == 4
    assert LIBRARY.casts(4, 8) == ()  # no widening cast defined yet
