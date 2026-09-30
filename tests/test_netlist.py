import pytest

from redc import CompileError, IRType
from redc.physical import (
    LIBRARY,
    Constant,
    Net,
    PhysicalNetlist,
)

BYTE = IRType(8)
ADD = LIBRARY["uint8_add_a-0-0-0_b-0-0-1_out-0-0-2"]
REG = LIBRARY.variants("register", 8)[0]


def test_instances_and_nets_get_disjoint_ids() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD)
    r = nl.add(REG)
    net = nl.connect(a.terminal("out"), r.terminal("next"))
    # A single allocator hands out ids, so nothing collides in the grid's owner.
    assert len({a.id, r.id, net.id}) == 3


def test_net_carries_width_and_fanout() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD)
    r1 = nl.add(REG)
    r2 = nl.add(REG)
    net = nl.connect(a.terminal("out"), r1.terminal("next"), r2.terminal("next"))
    assert net.width == 8  # a bus is one net; width lives on the net
    assert net.fanout == 2
    assert net.driver.port.name == "out"


def test_terminal_resolves_to_absolute_cell_when_placed() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD)
    assert not a.is_placed
    with pytest.raises(CompileError):
        _ = a.terminal("b").cell  # unplaced -> no coordinate
    a.origin = (10, 0, 5)
    assert a.terminal("b").cell == (10, 0, 6)  # pin 'b' at offset (0, 0, 1)
    # 'b' sits on WEST, so its wire first steps toward -x -- the abutment target.
    assert a.terminal("b").outward == (9, 0, 6)


def test_unknown_pin_raises() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD)
    with pytest.raises(CompileError):
        a.terminal("nope")


def test_driver_must_be_an_output() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD)
    r = nl.add(REG)
    with pytest.raises(CompileError):  # 'next' is an input -- illegal as a driver
        Net(0, r.terminal("next"), (a.terminal("a"),))


def test_width_mismatch_raises() -> None:
    nl = PhysicalNetlist()
    const4 = nl.add(Constant.of(3, IRType(4)))
    r = nl.add(REG)
    with pytest.raises(CompileError):
        nl.connect(const4.terminal("out"), r.terminal("next"))  # 4-bit into 8-bit


def test_validate_rejects_multiply_driven_input() -> None:
    nl = PhysicalNetlist()
    a1 = nl.add(ADD)
    a2 = nl.add(ADD)
    r = nl.add(REG)
    nl.connect(a1.terminal("out"), r.terminal("next"))
    nl.connect(a2.terminal("out"), r.terminal("next"))  # same input, two drivers
    with pytest.raises(CompileError):
        nl.validate()


def test_validate_accepts_a_well_formed_netlist() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD)
    r = nl.add(REG)
    nl.connect(a.terminal("out"), r.terminal("next"))
    nl.validate()  # no raise


def test_to_dict_snapshot() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD, origin=(0, 0, 0))
    r = nl.add(REG)
    nl.connect(a.terminal("out"), r.terminal("next"))
    snap = nl.to_dict()
    assert len(snap["instances"]) == 2
    assert len(snap["nets"]) == 1
    assert snap["nets"][0]["width"] == 8
    assert snap["nets"][0]["driver"] == {"instance": a.id, "pin": "out"}
