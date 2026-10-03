import json

import pytest

from redc import BOOL, CompileError, IRType
from redc.physical import (
    LIBRARY,
    ClockSource,
    ComponentInstance,
    Constant,
    InputPad,
    Net,
    OperationSignature,
    OutputPad,
    PhysicalNetlist,
    ResetSource,
    SignalEncoding,
)

BYTE = IRType(8)
INT8 = IRType(8, signed=True)
ADD = LIBRARY["uint8_add_a-0-0-0_b-0-0-1_out-0-0-2"]
(REG,) = LIBRARY.cells(OperationSignature.of("register", BYTE, BYTE, BOOL))
(INT8_REG,) = LIBRARY.cells(OperationSignature.of("register", INT8, INT8, BOOL))
(INV,) = LIBRARY.cells(OperationSignature.of("not", BOOL, BOOL))


def test_instances_and_nets_get_disjoint_ids() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD)
    r = nl.add(REG)
    net = nl.connect(a.terminal("out"), r.terminal("next"))
    # A single allocator hands out ids, so nothing collides in the grid's owner.
    assert len({a.id, r.id, net.id}) == 3


def test_net_carries_type_and_fanout() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD)
    r1 = nl.add(REG)
    r2 = nl.add(REG)
    net = nl.connect(a.terminal("out"), r1.terminal("next"), r2.terminal("next"))
    assert net.width == 8  # a bus is one net; width lives on the net
    assert net.dtype == BYTE
    assert net.fanout == 2
    assert net.driver.port.name == "out"
    nl.validate()  # fanout inside one net is the legal form


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


def test_signedness_mismatch_raises() -> None:
    nl = PhysicalNetlist()
    signed = nl.add(Constant.of(-1, INT8))
    r = nl.add(REG)
    with pytest.raises(CompileError, match="int8 vs sink uint8"):
        nl.connect(signed.terminal("out"), r.terminal("next"))
    # The same driver into an exactly-typed sink is fine.
    sr = nl.add(INT8_REG)
    nl.connect(signed.terminal("out"), sr.terminal("next"))
    nl.validate()


def test_explicit_cast_bridges_signedness() -> None:
    nl = PhysicalNetlist()
    signed = nl.add(Constant.of(-1, INT8))
    (cast,) = LIBRARY.cells(OperationSignature.of("cast", BYTE, INT8))
    c = nl.add(cast)
    r = nl.add(REG)
    nl.connect(signed.terminal("out"), c.terminal("in"))
    nl.connect(c.terminal("out"), r.terminal("next"))
    nl.validate()


def test_multiply_driven_input_is_rejected_on_connect() -> None:
    nl = PhysicalNetlist()
    a1 = nl.add(ADD)
    a2 = nl.add(ADD)
    r = nl.add(REG)
    nl.connect(a1.terminal("out"), r.terminal("next"))
    with pytest.raises(CompileError, match="driven by nets"):
        nl.connect(a2.terminal("out"), r.terminal("next"))


def test_validate_rejects_multiply_driven_input() -> None:
    nl = PhysicalNetlist()
    a1 = nl.add(ADD)
    a2 = nl.add(ADD)
    r = nl.add(REG)
    nl.connect(a1.terminal("out"), r.terminal("next"))
    # Bypass connect() to prove validate() re-checks from scratch.
    rogue = Net(99, a2.terminal("out"), (r.terminal("next"),))
    nl.nets[rogue.id] = rogue
    with pytest.raises(CompileError, match="driven by nets"):
        nl.validate()


def test_same_driver_on_two_nets_is_rejected() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD)
    r1 = nl.add(REG)
    r2 = nl.add(REG)
    nl.connect(a.terminal("out"), r1.terminal("next"))
    with pytest.raises(CompileError, match="already drives net"):
        nl.connect(a.terminal("out"), r2.terminal("next"))
    rogue = Net(99, a.terminal("out"), (r2.terminal("next"),))
    nl.nets[rogue.id] = rogue
    with pytest.raises(CompileError, match="already drives net"):
        nl.validate()


def test_duplicate_sink_within_one_net_is_rejected() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD)
    r = nl.add(REG)
    with pytest.raises(CompileError, match="same sink"):
        nl.connect(a.terminal("out"), r.terminal("next"), r.terminal("next"))


def test_foreign_instance_with_colliding_id_is_rejected() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD)
    r = nl.add(REG)
    impostor = ComponentInstance(r.id, REG)  # same id, different object
    with pytest.raises(CompileError, match="not part of this netlist"):
        nl.connect(a.terminal("out"), impostor.terminal("next"))
    other = PhysicalNetlist()
    foreign = other.add(ADD)
    foreign.id = a.id  # force a collision with a real instance id
    with pytest.raises(CompileError, match="not part of this netlist"):
        nl.connect(foreign.terminal("out"), r.terminal("next"))


def test_validate_accepts_a_well_formed_netlist() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD)
    r = nl.add(REG)
    nl.connect(a.terminal("out"), r.terminal("next"))
    nl.validate()  # no raise


# -- register instance state ------------------------------------------------


def test_two_instances_share_a_register_definition_with_different_inits() -> None:
    nl = PhysicalNetlist()
    a = nl.add(REG, init=0)
    b = nl.add(REG, init=7)
    assert a.component is b.component
    assert (a.init, b.init) == (0, 7)
    assert nl.add(REG).init == 0  # default like an IR register
    assert nl.add(INT8_REG, init=-1).init == 0xFF  # masked to raw bits


def test_register_init_is_validated_against_its_type() -> None:
    nl = PhysicalNetlist()
    with pytest.raises(CompileError, match="does not fit uint8"):
        nl.add(REG, init=256)
    with pytest.raises(CompileError, match="does not fit uint8"):
        nl.add(REG, init=-1)
    with pytest.raises(CompileError, match="only registers"):
        nl.add(ADD, init=3)


def test_to_dict_snapshot() -> None:
    nl = PhysicalNetlist()
    a = nl.add(ADD, origin=(0, 0, 0))
    r = nl.add(REG, init=7)
    nl.connect(a.terminal("out"), r.terminal("next"))
    snap = json.loads(json.dumps(nl.to_dict()))  # round-trips as JSON
    assert len(snap["instances"]) == 2
    assert "init" not in snap["instances"][0]
    assert snap["instances"][1]["init"] == 7  # instance-level, not component
    assert len(snap["nets"]) == 1
    assert snap["nets"][0]["width"] == 8
    assert snap["nets"][0]["type"] == {"width": 8, "signed": False, "boolean": False}
    assert snap["nets"][0]["layout"]["encoding"] == "binary"
    assert snap["nets"][0]["driver"] == {"instance": a.id, "pin": "out"}
    assert snap["sequential"] is True


def test_wide_value_is_one_net_with_hex_layout() -> None:
    nl = PhysicalNetlist()
    u64 = IRType(64)
    src = nl.add(InputPad.of("x", u64))
    sinks = [nl.add(OutputPad.of(f"y{i}", u64)) for i in range(3)]
    net = nl.connect(src.terminal("out"), *(s.terminal("in") for s in sinks))
    assert len(nl.nets) == 1
    assert net.layout.encoding is SignalEncoding.HEX
    assert net.layout.lane_count == 16


# -- one global clock / reset domain ------------------------------------------


def _sequential(nl: PhysicalNetlist, registers: int = 2):
    clk = nl.add(ClockSource.of())
    rst = nl.add(ResetSource.of())
    regs = [nl.add(REG, init=i) for i in range(registers)]
    return clk, rst, regs


def _wire_complete(nl, clk, rst, regs) -> None:
    data = nl.add(InputPad.of("d", BYTE))
    en = nl.add(InputPad.of("en", BOOL))
    nl.connect(data.terminal("out"), *(r.terminal("next") for r in regs))
    nl.connect(en.terminal("out"), *(r.terminal("enable") for r in regs))
    nl.connect(clk.terminal("clk"), *(r.terminal("clk") for r in regs))
    nl.connect(rst.terminal("rst"), *(r.terminal("rst") for r in regs))
    for i, r in enumerate(regs):
        nl.connect(r.terminal("out"), nl.add(OutputPad.of(f"q{i}", BYTE)).terminal("in"))


def test_one_clock_and_one_reset_net_reach_every_register() -> None:
    nl = PhysicalNetlist()
    clk, rst, regs = _sequential(nl, 3)
    _wire_complete(nl, clk, rst, regs)
    nl.validate(complete=True)
    clock_nets = [n for n in nl.nets.values() if n.driver.instance is clk]
    assert len(clock_nets) == 1 and clock_nets[0].fanout == 3


def test_second_clock_source_is_rejected() -> None:
    nl = PhysicalNetlist()
    _sequential(nl)
    nl.add(ClockSource.of())
    with pytest.raises(CompileError, match="one ClockSource"):
        nl.validate()


def test_second_reset_source_is_rejected() -> None:
    nl = PhysicalNetlist()
    _sequential(nl)
    nl.add(ResetSource.of())
    with pytest.raises(CompileError, match="one ResetSource"):
        nl.validate()


def test_register_clock_must_come_from_the_global_clock() -> None:
    nl = PhysicalNetlist()
    _, _, regs = _sequential(nl)
    local = nl.add(InputPad.of("my_clk", BOOL))
    with pytest.raises(CompileError, match="global ClockSource"):
        nl.connect(local.terminal("out"), regs[0].terminal("clk"))
    with pytest.raises(CompileError, match="global ResetSource"):
        nl.connect(local.terminal("out"), regs[0].terminal("rst"))


def test_clock_is_not_data() -> None:
    nl = PhysicalNetlist()
    clk, _, regs = _sequential(nl)
    inv = nl.add(INV)
    with pytest.raises(CompileError, match="may only reach register 'clk'"):
        nl.connect(clk.terminal("clk"), regs[0].terminal("clk"), inv.terminal("in0"))


def test_complete_sequential_netlist_needs_clock_and_reset() -> None:
    nl = PhysicalNetlist()
    regs = [nl.add(REG)]
    with pytest.raises(CompileError, match="undriven"):
        nl.validate(complete=True)
    nl2 = PhysicalNetlist()
    clk, rst, regs = _sequential(nl2, 1)
    _wire_complete(nl2, clk, rst, regs)
    nl2.validate(complete=True)
    del nl2.instances[rst.id]
    for net_id in [n.id for n in nl2.nets.values() if n.driver.instance is rst]:
        del nl2.nets[net_id]
    with pytest.raises(CompileError):
        nl2.validate(complete=True)


def test_complete_combinational_netlist_has_no_clock() -> None:
    nl = PhysicalNetlist()
    x = nl.add(InputPad.of("x", BOOL))
    y = nl.add(OutputPad.of("y", BOOL))
    nl.connect(x.terminal("out"), y.terminal("in"))
    nl.validate(complete=True)
    assert not nl.is_sequential
    nl.add(ClockSource.of())
    with pytest.raises(CompileError, match="combinational netlist"):
        nl.validate(complete=True)
