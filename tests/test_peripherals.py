"""Peripheral components and the default boundary-realization policy."""

from pathlib import Path

import pytest

from redc import BOOL, CompileError, IRType, compile_source
from redc.physical import (
    LIBRARY,
    PERIPHERALS,
    Constant,
    DefaultBoundaryPolicy,
    Face,
    InputPad,
    OutputPad,
    PadBoundaryPolicy,
    Peripheral,
    PeripheralDirection,
    PeripheralLibrary,
    PhysicalNetlist,
    Port,
    PortDir,
    lower_to_physical,
)
from redc.physical.cells.peripheral import LEVER, TWO_DIGIT_SEVEN_SEGMENT

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
FIB = (EXAMPLES / "uint8_fib.redc").read_text()
UINT8, INT8 = IRType(8), IRType(8, signed=True)
COUNTER = "uint8 main(uint8 n) { uint8 c = 0; while (c < n) { c = c + 1; } return c; }"
(LEVER_CELL,) = PERIPHERALS.find(LEVER, BOOL, PeripheralDirection.INPUT)
(DISPLAY,) = PERIPHERALS.find(TWO_DIGIT_SEVEN_SEGMENT, UINT8, PeripheralDirection.OUTPUT)


def instances(netlist, cls=None, kind=None):
    return [
        i
        for i in netlist.instances.values()
        if (cls is None or isinstance(i.component, cls))
        and (kind is None or getattr(i.component, "kind", None) == kind)
    ]


def by_label(netlist, label):
    (inst,) = [i for i in netlist.instances.values() if i.label == label]
    return inst


def sink_labels(netlist, inst) -> list[str]:
    (net,) = [n for n in netlist.nets.values() if n.driver.instance is inst]
    return sorted(s.instance.label for s in net.sinks)


def driver(netlist, inst, pin):
    term = inst.terminal(pin)
    (net,) = [n for n in netlist.nets.values() if term in n.sinks]
    return net.driver.instance


# -- F. placeholder geometry -----------------------------------------------------


def test_peripheral_library_loads_placeholder_stubs() -> None:
    assert set(PERIPHERALS) == {LEVER_CELL.name, DISPLAY.name}
    for p in PERIPHERALS.values():
        assert isinstance(p, Peripheral)
        assert all(d > 0 for d in p.dim)
        assert p.nbt is None
        assert p.latency is None  # unknown, not zero
        assert p.signatures() == ()  # never an operation implementation
        assert p.name not in LIBRARY


def test_lever_is_a_bool_input_device() -> None:
    assert LEVER_CELL.kind == "lever"
    assert LEVER_CELL.direction is PeripheralDirection.INPUT
    assert LEVER_CELL.dtype == BOOL
    assert LEVER_CELL.is_source and not LEVER_CELL.is_sink
    assert not LEVER_CELL.inputs
    (out,) = LEVER_CELL.outputs
    assert (out.name, out.dtype, out.direction) == ("out", BOOL, PortDir.OUT)


def test_display_is_a_uint8_output_device() -> None:
    assert DISPLAY.kind == "2-dig-7-seg"
    assert DISPLAY.direction is PeripheralDirection.OUTPUT
    assert DISPLAY.dtype == UINT8
    assert DISPLAY.is_sink and not DISPLAY.is_source
    assert not DISPLAY.outputs
    (pin,) = DISPLAY.inputs
    assert (pin.name, pin.dtype, pin.direction) == ("in", UINT8, PortDir.IN)
    assert DISPLAY.behavior({"in": 42}) == {}


def test_find_is_keyed_by_kind_type_and_direction() -> None:
    assert PERIPHERALS.find(LEVER, UINT8, PeripheralDirection.INPUT) == ()
    assert PERIPHERALS.find(LEVER, BOOL, PeripheralDirection.OUTPUT) == ()
    assert PERIPHERALS.find(TWO_DIGIT_SEVEN_SEGMENT, IRType(16), PeripheralDirection.OUTPUT) == ()


def test_peripheral_shape_is_validated() -> None:
    pin_out = Port("out", BOOL, Face.EAST, (0, 0, 0), PortDir.OUT)
    pin_in = Port("in", BOOL, Face.WEST, (0, 0, 0), PortDir.IN)
    common = {"latency": None, "dim": (1, 1, 1)}
    with pytest.raises(CompileError, match="input peripheral"):
        Peripheral(name="x", inputs=(pin_in,), outputs=(pin_out,), kind="k",
                   direction=PeripheralDirection.INPUT, **common)  # fmt: skip
    with pytest.raises(CompileError, match="output peripheral"):
        Peripheral(name="x", inputs=(), outputs=(pin_out,), kind="k",
                   direction=PeripheralDirection.OUTPUT, **common)  # fmt: skip
    with pytest.raises(CompileError, match="kind"):
        Peripheral(name="x", inputs=(), outputs=(pin_out,), direction=PeripheralDirection.INPUT, **common)
    with pytest.raises(CompileError, match="direction"):
        Peripheral(name="x", inputs=(), outputs=(pin_out,), kind="k", **common)


# -- A. sequential start -> lever ------------------------------------------------


def test_start_becomes_exactly_one_lever() -> None:
    graph = compile_source(COUNTER)
    nl = lower_to_physical(graph)
    levers = instances(nl, Peripheral, LEVER)
    assert len(levers) == 1 and levers[0].label == "start"
    assert not [i for i in instances(nl, InputPad) if i.label == "start"]
    # The lever feeds exactly the IR consumers of the start node.
    (start_node,) = [n for n in graph.live_nodes() if n.get("name") == "start"]
    expected = sorted(
        f"n{n['id']}" for n in graph.live_nodes() if start_node["id"] in n["args"]
    )
    assert sink_labels(nl, levers[0]) == expected


# -- B / E. uint8 result -> display ------------------------------------------------


def test_combinational_uint8_result_drives_the_display() -> None:
    graph = compile_source("uint8 add(uint8 a, uint8 b) { return a + b; }", top="add")
    nl = lower_to_physical(graph)
    assert sorted(i.label for i in instances(nl, InputPad)) == ["in_a", "in_b"]
    assert not instances(nl, OutputPad)
    (display,) = instances(nl, Peripheral, TWO_DIGIT_SEVEN_SEGMENT)
    assert display.label == "result"
    (result_port,) = graph.outputs
    assert driver(nl, display, "in").label == f"n{result_port['node']}"
    assert driver(nl, display, "in").component.op == "add"
    nl.validate(complete=True)


# -- C. sequential topology --------------------------------------------------------


def test_sequential_uint8_boundary_topology() -> None:
    nl = lower_to_physical(compile_source(FIB, top="fib"))
    boundary = {
        i.label: i
        for i in nl.instances.values()
        if isinstance(i.component, (Peripheral, InputPad, OutputPad))
    }
    assert set(boundary) == {"start", "in_n", "result", "done"}
    assert boundary["start"].component is LEVER_CELL
    assert boundary["result"].component is DISPLAY
    assert isinstance(boundary["done"].component, OutputPad)
    assert isinstance(boundary["in_n"].component, InputPad)
    assert all(i.origin is None for i in nl.instances.values())
    nl.validate(complete=True)


# -- D. fallback ----------------------------------------------------------------------


@pytest.mark.parametrize("typ", ["uint16", "int8", "bool"])
def test_non_uint8_result_falls_back_to_output_pad(typ: str) -> None:
    nl = lower_to_physical(compile_source(f"{typ} main({typ} a) {{ return a; }}"))
    assert not instances(nl, Peripheral)
    (pad,) = instances(nl, OutputPad)
    assert pad.label == "result" and pad.component.inputs[0].dtype.name == typ


def test_non_bool_start_or_other_names_use_pads() -> None:
    nl = lower_to_physical(compile_source("uint8 main(uint8 start) { return start; }"))
    assert not instances(nl, Peripheral, LEVER)  # the port is in_start: not a rule


def test_missing_peripheral_variant_degrades_to_pad() -> None:
    policy = DefaultBoundaryPolicy(peripherals=PeripheralLibrary({}))
    nl = lower_to_physical(compile_source(FIB, top="fib"), boundary=policy)
    assert not instances(nl, Peripheral)
    assert by_label(nl, "start").component.name == "start"


def test_pad_policy_has_no_peripherals() -> None:
    nl = lower_to_physical(compile_source(FIB, top="fib"), boundary=PadBoundaryPolicy())
    assert not instances(nl, Peripheral)


# -- netlist rules apply to peripheral terminals --------------------------------------


def test_lever_drives_bool_but_not_uint8() -> None:
    nl = PhysicalNetlist()
    lever = nl.add(LEVER_CELL)
    out = nl.add(OutputPad.of("x", BOOL))
    nl.connect(lever.terminal("out"), out.terminal("in"))
    wide = nl.add(OutputPad.of("y", UINT8))
    with pytest.raises(CompileError, match="type mismatch"):
        nl.connect(lever.terminal("out"), wide.terminal("in"))


def test_display_needs_an_exact_uint8_driver() -> None:
    nl = PhysicalNetlist()
    display = nl.add(DISPLAY)
    with pytest.raises(CompileError, match="type mismatch"):
        nl.connect(nl.add(Constant.of(-1, INT8)).terminal("out"), display.terminal("in"))
    with pytest.raises(CompileError, match="undriven"):
        nl.validate(complete=True)
    nl.connect(nl.add(Constant.of(42, UINT8)).terminal("out"), display.terminal("in"))
    nl.validate(complete=True)


def test_to_dict_marks_peripherals() -> None:
    nl = lower_to_physical(compile_source(FIB, top="fib"))
    entries = {e["label"]: e for e in nl.to_dict()["instances"]}
    assert entries["start"]["kind"] == "peripheral"
    assert entries["start"]["peripheral"] == "lever"
    assert entries["start"]["direction"] == "input"
    assert entries["result"]["peripheral"] == "2-dig-7-seg"
    assert entries["result"]["direction"] == "output"
    assert entries["result"]["origin"] is None
    assert "kind" not in entries["done"]  # ordinary pads stay uncluttered
    assert "kind" not in entries["in_n"]
