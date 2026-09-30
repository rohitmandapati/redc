from redc import IRType
from redc.physical import GATES, LIBRARY, OPERATIONS, WIRING, Face, Operation, PortDir


def test_families_load_from_yaml() -> None:
    assert OPERATIONS and GATES and WIRING
    # LIBRARY is the merge of every family, keyed by convention name.
    assert len(LIBRARY) == len(OPERATIONS) + len(GATES) + len(WIRING)
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
