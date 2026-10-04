"""The one-bit ``PrimitiveNetlist``: construction invariants, the structure of
synthesized netlists, port interface policies and the ``PrimitiveSimulator``.

* **Invariants.**  ``PrimitiveNetlist.add_instance`` / ``add_group`` /
  ``add_net`` / ``validate`` and the ``PrimitiveBuilder`` make invalid states
  hard to construct: a net is ONE driver output pin fanning out to distinct
  input pins and exactly one bit wide; clock and reset are global control nets
  that reach only register ``clk`` / ``rst`` pins; payloads (register
  ``init``, peripheral spec, provenance role and group) are checked; ids are
  dense and group parents precede their children.  Hand-built netlists
  exercise every rule directly.
* **Structure.**  Synthesized netlists hold only one-bit nets of basis gates
  plus boundary / state primitives; fanout is one net with many sinks; wide
  values exist only as ``LogicalPort`` / ``LogicalBus`` metadata (LSB first);
  dead IR becomes nothing; provenance and the hierarchy queries locate every
  gate; constants are per IR node; ``to_dict()`` is JSON-native.
* **Interfaces.**  ``DefaultInterfacePolicy`` realizes ``start : bool`` as ONE
  lever and ``result : uint8`` as ONE seven-segment display with eight
  independent one-bit pins; ``PadInterfacePolicy`` uses one pad per bit; the
  logical port metadata is the same either way.
* **Simulation.**  ``evaluate_many`` is a bit-parallel ``evaluate``;
  sequential ``step`` / ``run`` / state conversion track ``Graph`` exactly;
  ``rst`` restores every ``init``; combinational loops are detected.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import itertools
import json
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from redc import BOOL, CompileError, Graph, IRType, compile_source
from redc.ir import OPS
from redc.physical_primitive import (
    DEFAULT_INTERFACE_POLICY,
    GATE_KINDS,
    BitNet,
    BitTerminal,
    DefaultInterfacePolicy,
    HierarchyGroup,
    LogicalBus,
    LogicalPort,
    PadInterfacePolicy,
    PeripheralDirection,
    PeripheralSpec,
    PortRealization,
    PrimitiveBuilder,
    PrimitiveInstance,
    PrimitiveKind,
    PrimitiveNetlist,
    PrimitiveSimulator,
    Provenance,
    synthesize_to_primitives,
)
from redc.physical_primitive.netlist import (
    FORBIDDEN_PRIMITIVE_NAMES,
    NON_GATE_KINDS,
    PIN_INTERFACE,
    SCHEMA,
)
from redc.physical_primitive.synthesis.interface import (
    LEVER,
    PADS,
    TWO_DIGIT_SEVEN_SEGMENT,
)

T = BitTerminal
K = PrimitiveKind
INPUT, OUTPUT = PeripheralDirection.INPUT, PeripheralDirection.OUTPUT

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_TOPS = {"uint8_add": "add", "uint8_fib": "fib"}
PAD_POLICY = PadInterfacePolicy()
DEVICE_POLICY = DefaultInterfacePolicy()
POLICIES = [PAD_POLICY, DEVICE_POLICY]
POLICY_IDS = ["pads", "peripherals"]

UINT2 = IRType(2)
UINT4 = IRType(4)
UINT8 = IRType(8)
INT8 = IRType(8, signed=True)

BOUNDARY_KINDS = frozenset({K.INPUT_BIT, K.OUTPUT_BIT, K.CONST0, K.CONST1})
STATE_KINDS = frozenset({K.REGISTER_BIT, K.CLOCK_SOURCE, K.RESET_SOURCE})
#: Hierarchy group kinds (docs/physical-primitive-trace.md, ``design.groups[]``).
GROUP_KINDS = frozenset({"root", "ir_node", "port", "helper", "slice", "stage", "iteration", "control"})


# -- helpers -------------------------------------------------------------------


def fresh() -> PrimitiveNetlist:
    """An empty netlist holding only the root hierarchy group (id 0)."""
    netlist = PrimitiveNetlist()
    netlist.add_group("design", "root", None, None)
    return netlist


def add(netlist: PrimitiveNetlist, kind: PrimitiveKind, **payload: Any) -> PrimitiveInstance:
    """One primitive of ``kind`` in the root group (register bits start at 0)."""
    if kind is K.REGISTER_BIT:
        payload.setdefault("init", False)
    return netlist.add_instance(kind, Provenance(None, kind.value, group=0), **payload)


#: Instance ids of :func:`parts`: two input pads, an AND, a register bit, the
#: global clock and reset, and an output pad.
A, B, GATE, REG, CLK, RST, OUT = range(7)
PART_KINDS = (K.INPUT_BIT, K.INPUT_BIT, K.AND, K.REGISTER_BIT, K.CLOCK_SOURCE, K.RESET_SOURCE, K.OUTPUT_BIT)

#: ``(driver, sinks, role)`` of every net that makes :func:`parts` complete:
#: ``out = reg(a AND b)``.
COMPLETE_WIRING = (
    (T(A, "y"), (T(GATE, "a"),), "data"),
    (T(B, "y"), (T(GATE, "b"),), "data"),
    (T(GATE, "y"), (T(REG, "d"),), "data"),
    (T(CLK, "clk"), (T(REG, "clk"),), "clock"),
    (T(RST, "rst"), (T(REG, "rst"),), "reset"),
    (T(REG, "q"), (T(OUT, "a"),), "data"),
)


def parts(wiring=COMPLETE_WIRING) -> PrimitiveNetlist:
    """The :data:`PART_KINDS` primitives, connected by ``wiring``."""
    netlist = fresh()
    for kind in PART_KINDS:
        add(netlist, kind)
    for driver, sinks, role in wiring:
        netlist.add_net(driver, sinks, role)
    return netlist


def connectivity(netlist: PrimitiveNetlist) -> list[tuple[Any, ...]]:
    """Every net plus what the connectivity indexes say about each terminal."""
    rows: list[tuple[Any, ...]] = [(n.id, n.driver, n.sinks, n.role) for n in netlist.nets]
    for inst in netlist.instances:
        rows += [("drives", inst.id, pin, netlist.net_of_driver(T(inst.id, pin))) for pin in inst.outputs]
        rows += [("driven", inst.id, pin, netlist.driver_of(T(inst.id, pin))) for pin in inst.inputs]
    return rows


def synth(graph: Graph, policy=PAD_POLICY) -> PrimitiveNetlist:
    return synthesize_to_primitives(graph, interface=policy)


@functools.cache
def example_graph(name: str) -> Graph:
    return compile_source((EXAMPLES / f"{name}.redc").read_text(), top=EXAMPLE_TOPS[name])


@functools.cache
def example(name: str, policy) -> tuple[Graph, PrimitiveNetlist]:
    """A shared (never mutated) synthesized example."""
    graph = example_graph(name)
    return graph, synth(graph, policy)


PROGRAMS = {
    "not": "bool main(bool a) { return !a; }",
    "and": "bool main(bool a, bool b) { return a && b; }",
    "half_adder": "uint2 main(bool a, bool b) { return (uint2)a + (uint2)b; }",
    "full_adder": "uint2 main(bool a, bool b, bool c) { return (uint2)a + (uint2)b + (uint2)c; }",
    "uint4_add": "uint4 main(uint4 a, uint4 b) { return a + b; }",
    "select": "int8 main(int8 a, int8 b, bool c) { return c ? a - b : a * b; }",
    "arith_mix": "uint4 main(uint4 a, uint4 b) { return (a / (b | 1)) ^ (a % 3) ^ (a << (b & 3)) ^ (uint4)(a < b); }",
    "counter": "uint2 main(uint2 n) { uint2 x = 0; for (uint2 i = 0; i < n; i++) { x = x + 1; } return x; }",
}


@functools.cache
def program(name: str, policy) -> tuple[Graph, PrimitiveNetlist]:
    """A shared (never mutated) synthesized :data:`PROGRAMS` entry."""
    graph = compile_source(PROGRAMS[name])
    return graph, synth(graph, policy)


def gates(netlist: PrimitiveNetlist) -> list[PrimitiveInstance]:
    return [inst for inst in netlist.instances if inst.is_gate]


def group_named(netlist: PrimitiveNetlist, name: str) -> HierarchyGroup:
    (group,) = [g for g in netlist.groups if g.name == name]
    return group


def assert_json_native(value: Any, where: str = "$") -> None:
    """Only dicts with str keys, lists, str, int, float, bool and None."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert type(key) is str, f"{where}: key {key!r}"
            assert_json_native(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_json_native(item, f"{where}[{index}]")
    else:
        assert value is None or type(value) in (str, int, float, bool), f"{where}: {value!r}"


def py_fib(n: int) -> int:
    """``examples/uint8_fib.redc`` in Python."""
    last1 = last2 = 1
    for _ in range(2, n):
        last1, last2 = last2, last1 + last2
    return last2 & 0xFF


# == (a) PrimitiveNetlist invariants ==============================================


def test_a_complete_hand_built_netlist_validates() -> None:
    netlist = parts()
    netlist.validate(complete=True)
    assert netlist.is_sequential and netlist.gate_count == 1
    assert [net.role for net in netlist.nets] == ["data", "data", "data", "clock", "reset", "data"]
    assert netlist.driver_of(T(REG, "clk")) == T(CLK, "clk")
    assert netlist.net_of_driver(T(GATE, "y")) == netlist.nets[2]
    assert netlist.driver_of(T(A, "y")) is None  # an output pin is never "driven"
    assert netlist.net_of_driver(T(OUT, "a")) is None  # an input pin never drives


def test_a_sink_is_driven_by_at_most_one_net() -> None:
    netlist = parts([])
    netlist.add_net(T(A, "y"), [T(GATE, "a")])
    before = connectivity(netlist)
    message = "net 1: input pin 'a' of primitive 2 is already driven by net 0"
    with pytest.raises(CompileError, match=re.escape(message)):
        netlist.add_net(T(B, "y"), [T(GATE, "a")])
    # ... also when the double-driven pin hides behind a legitimate sink.
    with pytest.raises(CompileError, match=re.escape(message)):
        netlist.add_net(T(B, "y"), [T(GATE, "b"), T(GATE, "a")])
    assert connectivity(netlist) == before  # nothing half-added
    assert netlist.driver_of(T(GATE, "b")) is None


def test_fanout_is_one_net_never_a_second_net_from_the_same_driver() -> None:
    netlist = parts([])
    netlist.add_net(T(A, "y"), [T(GATE, "a")])
    before = connectivity(netlist)
    message = "net 1: pin 'y' of primitive 0 already drives net 0 (fanout must be one net with many sinks)"
    with pytest.raises(CompileError, match=re.escape(message)):
        netlist.add_net(T(A, "y"), [T(GATE, "b")])
    assert connectivity(netlist) == before
    # The legal spelling: one net, many sinks.
    netlist = parts([])
    net = netlist.add_net(T(A, "y"), [T(GATE, "a"), T(GATE, "b"), T(OUT, "a")])
    assert net.fanout == 3 and netlist.fanout_distribution() == {"3": 1}
    assert {netlist.driver_of(s) for s in net.sinks} == {T(A, "y")}


BAD_TERMINALS = [
    pytest.param(T(GATE, "a"), (T(OUT, "a"),), "net 0: primitive 2 (and) has no output pin 'a'", id="driver-is-an-input-pin"),
    pytest.param(T(OUT, "a"), (T(GATE, "a"),), "net 0: primitive 6 (output_bit) has no output pin 'a'", id="output-pad-drives"),
    pytest.param(T(REG, "d"), (T(GATE, "a"),), "net 0: primitive 3 (register_bit) has no output pin 'd'", id="register-d-drives"),
    pytest.param(T(A, "y"), (T(GATE, "y"),), "net 0: primitive 2 (and) has no input pin 'y'", id="sink-is-an-output-pin"),
    pytest.param(T(A, "y"), (T(B, "y"),), "net 0: primitive 1 (input_bit) has no input pin 'y'", id="input-pad-sinks"),
    pytest.param(T(A, "y"), (T(REG, "q"),), "net 0: primitive 3 (register_bit) has no input pin 'q'", id="register-q-sinks"),
    pytest.param(T(A, "y"), (T(GATE, "c"),), "net 0: primitive 2 (and) has no input pin 'c'", id="no-such-pin"),
    pytest.param(T(A, "out"), (T(GATE, "a"),), "net 0: primitive 0 (input_bit) has no output pin 'out'", id="no-such-driver-pin"),
    pytest.param(T(99, "y"), (T(GATE, "a"),), "net 0 references primitive 99, which is not part of this netlist", id="unknown-driver"),
    pytest.param(T(A, "y"), (T(GATE, "a"), T(7, "a")), "net 0 references primitive 7, which is not part of this netlist", id="unknown-sink"),
    pytest.param(T(-1, "y"), (T(GATE, "a"),), "net 0 references primitive -1, which is not part of this netlist", id="negative-id"),
    pytest.param((A, "y"), (T(GATE, "a"),), "net 0: (0, 'y') is not a BitTerminal", id="raw-tuple-driver"),
    pytest.param(T(A, "y"), ((GATE, "a"),), "net 0: (2, 'a') is not a BitTerminal", id="raw-tuple-sink"),
]


@pytest.mark.parametrize(("driver", "sinks", "message"), BAD_TERMINALS)
def test_net_terminals_name_real_pins_of_the_right_direction(driver, sinks, message: str) -> None:
    netlist = parts([])
    with pytest.raises(CompileError, match=re.escape(message)):
        netlist.add_net(driver, sinks)
    assert netlist.nets == [] and connectivity(netlist) == connectivity(parts([]))


def test_a_net_has_sinks_and_lists_each_sink_once() -> None:
    netlist = parts([])
    with pytest.raises(CompileError, match="net 0: has no sinks"):
        netlist.add_net(T(A, "y"), [])
    with pytest.raises(CompileError, match="net 0: lists the same sink pin twice"):
        netlist.add_net(T(A, "y"), [T(GATE, "a"), T(GATE, "b"), T(GATE, "a")])
    assert netlist.nets == []


@pytest.mark.parametrize("width", [0, 2, 8])
def test_every_net_is_exactly_one_bit(width: int) -> None:
    # A wide net cannot even be requested: add_net has no width, pins have no width.
    assert "width" not in inspect.signature(PrimitiveNetlist.add_net).parameters
    assert all(isinstance(pin, str) for ins, outs in PIN_INTERFACE.values() for pin in ins + outs)
    netlist = parts()
    assert {net.width for net in netlist.nets} == {1}
    with pytest.raises(dataclasses.FrozenInstanceError):
        netlist.nets[2].width = width  # type: ignore[misc]
    # Smuggling one in by replacing the record is caught by validate().
    netlist.nets[2] = dataclasses.replace(netlist.nets[2], width=width)
    with pytest.raises(CompileError, match=f"net 2: every primitive net is one bit, got width {width}"):
        netlist.validate()


CONTROL_MISUSE = [
    pytest.param(T(CLK, "clk"), T(GATE, "a"), "clock", "the global clock may only reach register 'clk' pins, not pin 'a' of primitive 2", id="clock-into-gate"),
    pytest.param(T(CLK, "clk"), T(REG, "d"), "clock", "the global clock may only reach register 'clk' pins, not pin 'd' of primitive 3", id="clock-into-register-d"),
    pytest.param(T(CLK, "clk"), T(REG, "rst"), "clock", "the global clock may only reach register 'clk' pins, not pin 'rst' of primitive 3", id="clock-into-register-rst"),
    pytest.param(T(CLK, "clk"), T(OUT, "a"), "clock", "the global clock may only reach register 'clk' pins, not pin 'a' of primitive 6", id="clock-into-output-pad"),
    pytest.param(T(RST, "rst"), T(GATE, "b"), "reset", "the global reset may only reach register 'rst' pins, not pin 'b' of primitive 2", id="reset-into-gate"),
    pytest.param(T(RST, "rst"), T(REG, "d"), "reset", "the global reset may only reach register 'rst' pins, not pin 'd' of primitive 3", id="reset-into-register-d"),
    pytest.param(T(RST, "rst"), T(REG, "clk"), "reset", "register clock pin 'clk' of primitive 3 must be driven by the global clock_source", id="reset-into-register-clk"),
    pytest.param(T(A, "y"), T(REG, "clk"), "data", "register clock pin 'clk' of primitive 3 must be driven by the global clock_source", id="data-into-register-clk"),
    pytest.param(T(REG, "q"), T(REG, "clk"), "data", "register clock pin 'clk' of primitive 3 must be driven by the global clock_source", id="register-output-as-clock"),
    pytest.param(T(A, "y"), T(REG, "rst"), "data", "register reset pin 'rst' of primitive 3 must be driven by the global reset_source", id="data-into-register-rst"),
    pytest.param(T(GATE, "y"), T(REG, "rst"), "data", "register reset pin 'rst' of primitive 3 must be driven by the global reset_source", id="gated-reset"),
]


@pytest.mark.parametrize(("driver", "sink", "role", "message"), CONTROL_MISUSE)
def test_clock_and_reset_are_global_control_nets_only(driver: BitTerminal, sink: BitTerminal, role: str, message: str) -> None:
    netlist = parts([])
    with pytest.raises(CompileError, match=re.escape(f"net 0: {message}")):
        netlist.add_net(driver, [sink], role)
    # Also when the illegal sink rides along with a legal one.
    legal = {"clock": T(REG, "clk"), "reset": T(REG, "rst")}.get(role, T(GATE, "a"))
    if legal != sink:
        with pytest.raises(CompileError, match=re.escape(f"net 0: {message}")):
            netlist.add_net(driver, [legal, sink], role)
    assert netlist.nets == []


ROLE_MISMATCH = [
    pytest.param(T(A, "y"), T(GATE, "a"), "clock", "net 0: role 'clock' but driven by input_bit", id="data-labelled-clock"),
    pytest.param(T(A, "y"), T(GATE, "a"), "", "net 0: role '' but driven by input_bit", id="data-unlabelled"),
    pytest.param(T(CLK, "clk"), T(REG, "clk"), "data", "net 0: role 'data' but driven by clock_source", id="clock-labelled-data"),
    pytest.param(T(RST, "rst"), T(REG, "rst"), "clock", "net 0: role 'clock' but driven by reset_source", id="reset-labelled-clock"),
]


@pytest.mark.parametrize(("driver", "sink", "role", "message"), ROLE_MISMATCH)
def test_a_net_role_names_its_driver(driver: BitTerminal, sink: BitTerminal, role: str, message: str) -> None:
    netlist = parts([])
    with pytest.raises(CompileError, match=re.escape(message)):
        netlist.add_net(driver, [sink], role)
    assert netlist.nets == []


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        (K.CLOCK_SOURCE, "one global clock domain allows one CLOCK_SOURCE, found 2"),
        (K.RESET_SOURCE, "one global reset allows one RESET_SOURCE, found 2"),
    ],
    ids=["clock", "reset"],
)
def test_one_global_clock_and_one_global_reset(kind: PrimitiveKind, message: str) -> None:
    netlist = parts()
    add(netlist, kind)  # a second source, even an unconnected one
    with pytest.raises(CompileError, match=re.escape(message)):
        netlist.validate()
    with pytest.raises(CompileError, match=re.escape(message)):
        netlist.validate(complete=True)


@pytest.mark.parametrize(
    ("drop", "pin"),
    [
        (0, "pin 'a' of primitive 2 (and)"),
        (1, "pin 'b' of primitive 2 (and)"),
        (2, "pin 'd' of primitive 3 (register_bit)"),
        (3, "pin 'clk' of primitive 3 (register_bit)"),
        (4, "pin 'rst' of primitive 3 (register_bit)"),
        (5, "pin 'a' of primitive 6 (output_bit)"),
    ],
    ids=["and-a", "and-b", "register-d", "register-clk", "register-rst", "output-pad"],
)
def test_complete_validation_names_an_undriven_input(drop: int, pin: str) -> None:
    netlist = parts(COMPLETE_WIRING[:drop] + COMPLETE_WIRING[drop + 1 :])
    netlist.validate()  # every structural rule holds; it is merely unfinished
    with pytest.raises(CompileError, match=re.escape(f"incomplete primitive netlist: {pin} is undriven")):
        netlist.validate(complete=True)


def test_an_output_device_needs_every_pin_driven_an_input_device_none() -> None:
    netlist = fresh()
    x = add(netlist, K.INPUT_BIT)
    display = add(netlist, K.PERIPHERAL, peripheral=PeripheralSpec("2-dig-7-seg", OUTPUT, 2))
    add(netlist, K.PERIPHERAL, peripheral=PeripheralSpec("lever", INPUT, 1))  # drives nothing: fine
    netlist.add_net(x.output(), [T(display.id, "b0")])
    netlist.validate()
    with pytest.raises(CompileError, match=re.escape("pin 'b1' of primitive 1 (peripheral) is undriven")):
        netlist.validate(complete=True)
    netlist = fresh()
    x = add(netlist, K.INPUT_BIT)
    display = add(netlist, K.PERIPHERAL, peripheral=PeripheralSpec("2-dig-7-seg", OUTPUT, 2))
    add(netlist, K.PERIPHERAL, peripheral=PeripheralSpec("lever", INPUT, 1))
    netlist.add_net(x.output(), [T(display.id, "b0"), T(display.id, "b1")])
    netlist.validate(complete=True)


@pytest.mark.parametrize("kind", [K.CLOCK_SOURCE, K.RESET_SOURCE], ids=["clock", "reset"])
def test_a_combinational_netlist_has_no_clock_or_reset_source(kind: PrimitiveKind) -> None:
    netlist = fresh()
    for k in (K.INPUT_BIT, K.INPUT_BIT, K.AND, K.OUTPUT_BIT):
        add(netlist, k)
    netlist.add_net(T(0, "y"), [T(2, "a")])
    netlist.add_net(T(1, "y"), [T(2, "b")])
    netlist.add_net(T(2, "y"), [T(3, "a")])
    netlist.validate(complete=True)
    assert not netlist.is_sequential
    add(netlist, kind)
    netlist.validate()
    with pytest.raises(CompileError, match="a combinational primitive netlist has no clock or reset source"):
        netlist.validate(complete=True)


LEVER_SPEC = PeripheralSpec(LEVER, INPUT, 1)
BAD_INSTANCES = [
    pytest.param(K.REGISTER_BIT, {"init": None}, "register bit 0: init must be a bool, got None", id="register-without-init"),
    pytest.param(K.REGISTER_BIT, {"init": 1}, "register bit 0: init must be a bool, got 1", id="register-int-init-1"),
    pytest.param(K.REGISTER_BIT, {"init": 0}, "register bit 0: init must be a bool, got 0", id="register-int-init-0"),
    pytest.param(K.REGISTER_BIT, {"init": "1"}, "register bit 0: init must be a bool, got '1'", id="register-str-init"),
    pytest.param(K.AND, {"init": False}, "primitive 0 (and): only register bits carry init", id="gate-with-init"),
    pytest.param(K.CONST1, {"init": True}, "primitive 0 (const1): only register bits carry init", id="constant-with-init"),
    pytest.param(K.PERIPHERAL, {}, "peripheral 0: missing PeripheralSpec", id="peripheral-without-spec"),
    pytest.param(K.PERIPHERAL, {"peripheral": "lever"}, "peripheral 0: missing PeripheralSpec", id="peripheral-spec-by-name"),
    pytest.param(K.INPUT_BIT, {"peripheral": LEVER_SPEC}, "primitive 0 (input_bit): only peripherals carry a spec", id="pad-with-spec"),
    pytest.param("add", {}, "primitive 0: 'add' is not a PrimitiveKind", id="ir-op-as-kind"),
    pytest.param("and", {}, "primitive 0: 'and' is not a PrimitiveKind", id="kind-by-name"),
]


@pytest.mark.parametrize(("kind", "payload", "message"), BAD_INSTANCES)
def test_instance_payloads_are_checked(kind, payload: dict[str, Any], message: str) -> None:
    netlist = fresh()
    provenance = Provenance(None, "test", group=0)
    with pytest.raises(CompileError, match=re.escape(message)):
        netlist.add_instance(kind, provenance, **payload)
    assert netlist.instances == []
    # The same record assembled by hand is rejected by validate().
    smuggled = PrimitiveNetlist(instances=[PrimitiveInstance(0, kind, provenance, **payload)], groups=list(netlist.groups))
    with pytest.raises(CompileError, match=re.escape(message)):
        smuggled.validate()


@pytest.mark.parametrize(
    ("provenance", "message"),
    [
        pytest.param(Provenance(3, ""), "primitive 0: provenance needs a role", id="empty-role"),
        pytest.param(Provenance(3, None), "primitive 0: provenance needs a role", id="no-role"),  # type: ignore[arg-type]
        pytest.param(Provenance(3, "sum_xor", 0, 1), "primitive 0: unknown provenance group 1", id="unknown-group"),
        pytest.param(Provenance(3, "sum_xor", 0, -1), "primitive 0: unknown provenance group -1", id="negative-group"),
    ],
)
def test_provenance_needs_a_role_and_an_existing_group(provenance: Provenance, message: str) -> None:
    netlist = fresh()
    with pytest.raises(CompileError, match=re.escape(message)):
        netlist.add_instance(K.XOR, provenance)
    assert netlist.instances == []
    smuggled = PrimitiveNetlist(instances=[PrimitiveInstance(0, K.XOR, provenance)], groups=list(netlist.groups))
    with pytest.raises(CompileError, match=re.escape(message)):
        smuggled.validate()


def test_add_group_only_accepts_existing_parents() -> None:
    netlist = PrimitiveNetlist()
    with pytest.raises(CompileError, match=re.escape("group 'orphan': unknown parent group 0")):
        netlist.add_group("orphan", "helper", 0, None)
    root = netlist.add_group("design", "root", None, None)
    for parent in (1, -1, 7):  # 1 would be the new group itself
        with pytest.raises(CompileError, match=re.escape(f"group 'ahead': unknown parent group {parent}")):
            netlist.add_group("ahead", "helper", parent, None)
    adder = netlist.add_group("adder", "helper", root.id, 4)
    bit0 = netlist.add_group("bit0", "slice", adder.id, 4, (("bit", 0),))
    assert [g.id for g in netlist.groups] == [0, 1, 2]
    netlist.validate()
    assert netlist.group_path(bit0.id) == ("design", "adder", "bit0")
    assert netlist.descendants(root.id) == {0, 1, 2} and netlist.descendants(bit0.id) == {2}


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        pytest.param(
            [HierarchyGroup(0, "bit0", "slice", 1, None), HierarchyGroup(1, "design", "root", None, None)],
            "group 0: parent 1 must precede it",
            id="parent-after-child",
        ),
        # A self-parented group would make group_path() loop forever.
        pytest.param([HierarchyGroup(0, "loop", "helper", 0, None)], "group 0: parent 0 must precede it", id="own-parent"),
        pytest.param([HierarchyGroup(0, "design", "root", None, None), HierarchyGroup(1, "a", "helper", 5, None)], "group 1: parent 5 must precede it", id="unknown-parent"),
        pytest.param([HierarchyGroup(1, "design", "root", None, None)], "group 1 is stored at index 0", id="misplaced-id"),
    ],
)
def test_validate_rechecks_hand_assembled_hierarchies(groups: list[HierarchyGroup], message: str) -> None:
    with pytest.raises(CompileError, match=re.escape(message)):
        PrimitiveNetlist(groups=groups).validate()


def test_ids_are_dense() -> None:
    misplaced = PrimitiveNetlist(instances=[PrimitiveInstance(1, K.CONST0, Provenance(None, "const0"))])
    with pytest.raises(CompileError, match="primitive 1 is stored at index 0"):
        misplaced.validate()
    netlist = parts()
    netlist.nets[1] = dataclasses.replace(netlist.nets[1], id=7)
    with pytest.raises(CompileError, match="net 7 is stored at index 1"):
        netlist.validate()


def test_validate_rejects_stale_connectivity_indexes() -> None:
    netlist = parts(COMPLETE_WIRING[:1])
    netlist.nets.append(BitNet(1, T(B, "y"), (T(GATE, "b"),)))  # bypasses add_net
    assert netlist.driver_of(T(GATE, "b")) is None  # the index never heard of it
    with pytest.raises(CompileError, match="connectivity indexes are stale"):
        netlist.validate()
    # A netlist assembled from records rebuilds its indexes ...
    rebuilt = PrimitiveNetlist(instances=list(netlist.instances), nets=list(netlist.nets), groups=list(netlist.groups))
    rebuilt.validate()
    assert rebuilt.driver_of(T(GATE, "b")) == T(B, "y")
    # ... and losing nets behind the indexes' back is caught as well.
    rebuilt.nets.pop()
    with pytest.raises(CompileError, match="connectivity indexes are stale"):
        rebuilt.validate()


BAD_METADATA = [
    pytest.param([LogicalPort("in_x", "input", BOOL, "x", 0, (T(GATE, "a"),))], {}, "primitive 2 (and) has no output pin 'a'", id="input-port-on-an-input-pin"),
    pytest.param([LogicalPort("result", "output", BOOL, None, 0, (T(A, "y"),))], {}, "primitive 0 (input_bit) has no input pin 'y'", id="output-port-on-an-output-pin"),
    pytest.param([LogicalPort("in_x", "inout", BOOL, "x", 0, (T(A, "y"),))], {}, "port 'in_x': bad direction 'inout'", id="bad-direction"),
    pytest.param([LogicalPort("in_x", "input", UINT4, "x", 0, (T(A, "y"), T(B, "y")))], {}, "port 'in_x': 2 bit terminals for a uint4", id="port-width"),
    pytest.param([LogicalPort("in_x", "input", BOOL, "x", 0, (T(42, "y"),))], {}, "references primitive 42", id="port-unknown-instance"),
    pytest.param([], {5: LogicalBus(4, "and", BOOL, (T(GATE, "y"),))}, "bus of IR node 5 is malformed", id="bus-keyed-wrong"),
    pytest.param([], {5: LogicalBus(5, "and", UINT2, (T(GATE, "y"),))}, "bus of IR node 5 is malformed", id="bus-width"),
    pytest.param([], {5: LogicalBus(5, "and", BOOL, (T(GATE, "a"),))}, "primitive 2 (and) has no output pin 'a'", id="bus-on-an-input-pin"),
]


@pytest.mark.parametrize(("ports", "buses", "message"), BAD_METADATA)
def test_ports_and_buses_reference_real_terminals_of_the_right_direction(ports, buses, message: str) -> None:
    netlist = parts()
    netlist.ports = [LogicalPort("in_a", "input", BOOL, "a", 0, (T(A, "y"),)), LogicalPort("result", "output", BOOL, None, 3, (T(OUT, "a"),))]
    netlist.buses = {3: LogicalBus(3, "register", BOOL, (T(REG, "q"),)), 2: LogicalBus(2, "and", BOOL, (T(GATE, "y"),))}
    netlist.validate(complete=True)  # the legal metadata
    netlist.ports += ports
    netlist.buses.update(buses)
    with pytest.raises(CompileError, match=re.escape(message)):
        netlist.validate()


def test_peripheral_specs_are_devices_with_independent_one_bit_pins() -> None:
    with pytest.raises(CompileError, match="a peripheral needs a kind"):
        PeripheralSpec("", INPUT, 1)
    for width in (0, 65, -1):
        with pytest.raises(CompileError, match=re.escape("peripheral 'lever': width must be 1..64")):
            PeripheralSpec("lever", INPUT, width)
    spec = PeripheralSpec(TWO_DIGIT_SEVEN_SEGMENT, OUTPUT, 8)
    assert spec.pins == tuple(f"b{i}" for i in range(8))
    assert spec.to_dict() == {"kind": "2-dig-7-seg", "direction": "output", "width": 8}
    netlist = fresh()
    display = add(netlist, K.PERIPHERAL, peripheral=spec)
    lever = add(netlist, K.PERIPHERAL, peripheral=LEVER_SPEC)
    assert (display.inputs, display.outputs) == (spec.pins, ())  # the circuit drives a display
    assert (lever.inputs, lever.outputs) == ((), ("b0",))  # a lever drives the circuit
    assert lever.output() == T(lever.id, "b0") and display.terminal("b7") == T(display.id, "b7")
    with pytest.raises(CompileError, match=re.escape("primitive 0 (peripheral) has 0 outputs")):
        display.output()
    wide = add(netlist, K.PERIPHERAL, peripheral=PeripheralSpec("switches", INPUT, 4))
    with pytest.raises(CompileError, match=re.escape("primitive 2 (peripheral) has 4 outputs")):
        wide.output()
    assert wide.output("b3") == T(wide.id, "b3")


def test_a_peripheral_direction_must_be_a_peripheral_direction() -> None:
    """A direction given as a plain string must be rejected like an empty kind
    or a zero width: ``inputs`` / ``outputs`` compare it against the enum, so
    such a device silently has NO pins at all, passes
    ``validate(complete=True)`` and only dies later, in ``to_dict()``, with an
    AttributeError."""
    netlist = fresh()
    with pytest.raises(CompileError):
        spec = PeripheralSpec("lever", "input", 1)  # type: ignore[arg-type]
        netlist.add_instance(K.PERIPHERAL, Provenance(None, "peripheral", group=0), peripheral=spec)


def test_instance_pin_helpers() -> None:
    netlist = parts()
    gate = netlist.instance(GATE)
    assert (gate.inputs, gate.outputs) == (("a", "b"), ("y",))
    assert gate.terminal("b") == T(GATE, "b") and gate.output() == T(GATE, "y")
    with pytest.raises(CompileError, match=re.escape("primitive 2 (and) has no pin 'q'")):
        gate.terminal("q")
    with pytest.raises(CompileError, match=re.escape("primitive 2 (and) has no output 'a'")):
        gate.output("a")
    assert netlist.instance(REG).output() == T(REG, "q")
    with pytest.raises(CompileError, match=re.escape("primitive 6 (output_bit) has 0 outputs")):
        netlist.instance(OUT).output()
    for bad in (7, -1):
        with pytest.raises(CompileError, match=f"unknown primitive instance {bad}"):
            netlist.instance(bad)
    assert PIN_INTERFACE[K.REGISTER_BIT] == (("d", "clk", "rst"), ("q",))
    assert set(PIN_INTERFACE) == set(PrimitiveKind) - {K.PERIPHERAL}


# == (a) the PrimitiveBuilder =========================================================


def builder_with_signals(on_emit: Callable[[PrimitiveInstance], None] | None = None) -> tuple[PrimitiveBuilder, dict[str, BitTerminal]]:
    """``x``(0), ``y``(1), ``gate = x AND y``(2), a register bit ``q``(3) and an
    output pad (4) observing the gate; ``pad`` is the pad's input terminal."""
    b = PrimitiveBuilder(on_emit=on_emit)
    x = b.input_bit("in_x", 0)
    y = b.input_bit("in_y", 0)
    gate = b.and_(x, y)
    q = b.register_bit(init=False, bit=0)
    pad = b.output_bit("result", 0, gate)
    return b, {"x": x, "y": y, "gate": gate, "q": q, "pad": pad}


GATE_CALLS: dict[str, Callable[[PrimitiveBuilder, Any, BitTerminal], BitTerminal]] = {
    "and": lambda b, bad, ok: b.and_(ok, bad),
    "and-first": lambda b, bad, ok: b.and_(bad, ok),
    "or": lambda b, bad, ok: b.or_(ok, bad),
    "xor": lambda b, bad, ok: b.xor(bad, ok),
    "not": lambda b, bad, ok: b.not_(bad),
}

NOT_A_HANDLE = "is not a one-bit signal handle"
NOT_AN_OUTPUT = "is not an output terminal of this netlist"
BAD_OPERANDS = [
    pytest.param(lambda s: (s["x"].instance, s["x"].pin), NOT_A_HANDLE, id="raw-tuple"),
    pytest.param(lambda s: (s["x"], s["y"]), NOT_A_HANDLE, id="two-bit-vector"),
    pytest.param(lambda s: (s["x"],), NOT_A_HANDLE, id="one-bit-vector"),
    pytest.param(lambda s: 0, NOT_A_HANDLE, id="int"),
    pytest.param(lambda s: T(99, "y"), NOT_AN_OUTPUT, id="unknown-instance"),
    pytest.param(lambda s: T(s["gate"].instance, "a"), NOT_AN_OUTPUT, id="gate-input-pin"),
    pytest.param(lambda s: T(s["q"].instance, "d"), NOT_AN_OUTPUT, id="register-d-pin"),
    pytest.param(lambda s: s["pad"], NOT_AN_OUTPUT, id="output-pad-pin"),
    pytest.param(lambda s: T(s["x"].instance, "q"), NOT_AN_OUTPUT, id="wrong-pin-name"),
]


@pytest.mark.parametrize(("make", "message"), BAD_OPERANDS)
def test_gates_take_exactly_one_existing_output_bit_per_operand(make, message: str) -> None:
    for name, call in GATE_CALLS.items():
        emitted: list[PrimitiveInstance] = []
        b, s = builder_with_signals(emitted.append)
        before = (len(b.netlist.instances), len(emitted))
        with pytest.raises(CompileError, match=re.escape(message)):
            call(b, make(s), s["x"])
        assert (len(b.netlist.instances), len(emitted)) == before, name  # nothing half-built


def test_connect_drives_each_input_terminal_exactly_once() -> None:
    b, s = builder_with_signals()
    d, clk, rst = b.register_pins(s["q"])
    assert (d, clk, rst) == (T(3, "d"), T(3, "clk"), T(3, "rst"))
    b.connect(s["x"], d)
    with pytest.raises(CompileError, match=re.escape("pin 'd' of primitive 3 is already driven")):
        b.connect(s["y"], d)
    # Gate operands are wired when the gate is made and can never be rewired,
    # so the builder cannot close a combinational loop through gates.
    with pytest.raises(CompileError, match=re.escape("pin 'a' of primitive 2 is already driven")):
        b.connect(s["gate"], T(2, "a"))
    with pytest.raises(CompileError, match=re.escape("pin 'a' of primitive 4 is already driven")):
        b.connect(s["y"], s["pad"])
    for sink in (T(2, "y"), T(99, "a"), s["q"], T(3, "x")):
        with pytest.raises(CompileError, match="is not an input terminal of this netlist"):
            b.connect(s["x"], sink)
    with pytest.raises(CompileError, match=NOT_AN_OUTPUT):
        b.connect(T(2, "b"), clk)
    with pytest.raises(CompileError, match=NOT_A_HANDLE):
        b.connect((0, "y"), clk)  # type: ignore[arg-type]
    # None of the rejected calls consumed clk / rst.
    b.connect(b.clock(), clk)
    b.connect(b.reset(), rst)
    netlist = b.finish()
    assert netlist.driver_of(d) == s["x"] and netlist.driver_of(T(2, "a")) == s["x"]
    assert netlist.driver_of(clk) == T(5, "clk") and netlist.driver_of(rst) == T(6, "rst")


BUILDER_CONTROL_MISUSE = [
    pytest.param(lambda b, s, pins: b.and_(b.clock(), s["x"]), "the global clock may only reach register 'clk' pins, not pin 'a' of primitive 6", id="clock-as-gate-operand"),
    pytest.param(lambda b, s, pins: b.connect(b.clock(), pins[0]), "the global clock may only reach register 'clk' pins, not pin 'd' of primitive 3", id="clock-into-d"),
    pytest.param(lambda b, s, pins: b.connect(b.clock(), pins[2]), "the global clock may only reach register 'clk' pins, not pin 'rst' of primitive 3", id="clock-into-rst"),
    pytest.param(lambda b, s, pins: b.connect(s["x"], pins[1]), "register clock pin 'clk' of primitive 3 must be driven by the global clock_source", id="data-into-clk"),
    pytest.param(lambda b, s, pins: b.connect(b.reset(), pins[1]), "register clock pin 'clk' of primitive 3 must be driven by the global clock_source", id="reset-into-clk"),
    pytest.param(lambda b, s, pins: b.connect(s["gate"], pins[2]), "register reset pin 'rst' of primitive 3 must be driven by the global reset_source", id="gated-reset"),
    pytest.param(lambda b, s, pins: b.output_bit("tap", 0, b.reset()), "the global reset may only reach register 'rst' pins, not pin 'a' of primitive 6", id="reset-observed-by-a-pad"),
]


@pytest.mark.parametrize("validate", [True, False], ids=["validated", "unvalidated"])
@pytest.mark.parametrize(("misuse", "message"), BUILDER_CONTROL_MISUSE)
def test_builder_finish_rejects_clock_and_reset_misuse(misuse, message: str, validate: bool) -> None:
    """``finish(validate=False)`` skips only the completeness check: every net
    is still built through ``add_net`` and its rules."""
    b, s = builder_with_signals()
    misuse(b, s, b.register_pins(s["q"]))
    with pytest.raises(CompileError, match=re.escape(message)):
        b.finish(validate=validate)


def test_builder_makes_one_global_clock_and_one_global_reset() -> None:
    b = PrimitiveBuilder()
    clock, reset = b.clock(), b.reset()
    assert [b.clock(), b.reset(), b.clock()] == [clock, reset, clock]
    assert (clock.pin, reset.pin) == ("clk", "rst")
    counts = b.netlist.kind_counts()
    assert counts["clock_source"] == counts["reset_source"] == 1
    for bit, kind in ((clock, K.CLOCK_SOURCE), (reset, K.RESET_SOURCE)):
        inst = b.netlist.instance(bit.instance)
        assert inst.kind is kind and inst.output() == bit
        assert (inst.provenance.ir_node, inst.provenance.role) == (None, kind.value)
        group = b.netlist.groups[inst.provenance.group]
        assert (group.name, group.kind, group.parent, group.ir_node) == (kind.value, "control", 0, None)


def test_builder_register_bits_carry_a_bool_init() -> None:
    b = PrimitiveBuilder()
    bits = [b.register_bit(init=value, bit=i) for i, value in enumerate((1, 0, True, False, 5))]
    inits = [b.netlist.instance(q.instance).init for q in bits]
    assert inits == [True, False, True, False, True] and {type(v) for v in inits} == {bool}
    assert [q.pin for q in bits] == ["q"] * 5


def test_builder_finish_checks_completeness_and_is_final() -> None:
    b = PrimitiveBuilder()
    q = b.register_bit(init=False, bit=0)
    with pytest.raises(CompileError, match=re.escape("incomplete primitive netlist: pin 'd' of primitive 0 (register_bit) is undriven")):
        b.finish()
    with pytest.raises(CompileError, match="primitive builder is already finished"):
        b.finish()
    with pytest.raises(CompileError, match="primitive builder is finished; no further primitives may be emitted"):
        b.not_(q)
    lenient = PrimitiveBuilder()
    lenient.register_bit(init=True, bit=0)
    netlist = lenient.finish(validate=False)
    netlist.validate()
    with pytest.raises(CompileError, match="is undriven"):
        netlist.validate(complete=True)


def test_builder_checks_payloads_and_roles() -> None:
    b, s = builder_with_signals()
    count = len(b.netlist.instances)
    with pytest.raises(CompileError, match="missing PeripheralSpec"):
        b.peripheral(None, "in_z")  # type: ignore[arg-type]
    with pytest.raises(CompileError, match="provenance needs a role"):
        b.and_(s["x"], s["y"], role="")
    with pytest.raises(CompileError, match="provenance needs a role"):
        b.register_bit(init=False, bit=0, role="")
    assert len(b.netlist.instances) == count


REJECTED_CALLS: dict[str, Callable[[PrimitiveBuilder, BitTerminal, BitTerminal], object]] = {
    "and": lambda b, x, bad: b.and_(x, bad),
    "or": lambda b, x, bad: b.or_(x, bad),
    "xor": lambda b, x, bad: b.xor(x, bad),
    "not": lambda b, x, bad: b.not_(bad),
    "output_bit": lambda b, x, bad: b.output_bit("result", 0, bad),
}


@pytest.mark.parametrize("call", sorted(REJECTED_CALLS))
def test_a_rejected_builder_call_emits_no_primitive(call: str) -> None:
    """Builder calls are all-or-nothing: a call that raises leaves neither a
    half-wired primitive in the netlist nor an emission in the synthesis trace
    (an orphaned output pad would also make the netlist incomplete)."""
    emitted: list[PrimitiveInstance] = []
    b = PrimitiveBuilder(on_emit=emitted.append)
    x = b.input_bit("in_x", 0)
    with pytest.raises(CompileError, match=NOT_AN_OUTPUT):
        REJECTED_CALLS[call](b, x, T(99, "y"))
    assert [inst.kind for inst in b.netlist.instances] == [K.INPUT_BIT]
    assert emitted == list(b.netlist.instances)


def test_finish_makes_one_net_per_driven_signal_in_instance_then_pin_order() -> None:
    emitted: list[PrimitiveInstance] = []
    b = PrimitiveBuilder(on_emit=emitted.append)
    x = b.input_bit("in_x", 0)  # 0
    q = b.register_bit(init=False, bit=0)  # 1
    nq = b.not_(q)  # 2
    g = b.and_(x, nq)  # 3
    d, clk, rst = b.register_pins(q)
    b.connect(g, d)
    b.connect(b.clock(), clk)  # 4
    b.connect(b.reset(), rst)  # 5
    pad = b.output_bit("result", 0, q)  # 6
    b.input_bit("in_unused", 0)  # 7: drives nothing, so it gets no net
    netlist = b.finish()
    assert emitted == netlist.instances
    assert [(net.id, net.driver, net.sinks, net.role) for net in netlist.nets] == [
        (0, x, (T(3, "a"),), "data"),
        (1, q, (T(2, "a"), pad), "data"),  # fanout: one net, sinks in wiring order
        (2, nq, (T(3, "b"),), "data"),
        (3, g, (d,), "data"),
        (4, T(4, "clk"), (clk,), "clock"),
        (5, T(5, "rst"), (rst,), "reset"),
    ]
    assert netlist.net_of_driver(T(7, "y")) is None


def test_builder_scopes_stamp_provenance_and_reuse_each_ir_node_group() -> None:
    b = PrimitiveBuilder()
    with b.port("in_x", "input", 0) as port_group:
        x = b.input_bit("in_x", 0)
    y = b.input_bit("in_y", 0)  # outside every scope: the root group
    with (
        b.ir_node(9, "add", UINT4) as node_group,
        b.scope("adder", width=4) as adder,
        b.scope("bit2", kind="slice", bit=2) as slice_,
    ):
        s = b.xor(x, y, role="sum_xor", bit=2)
    with b.ir_node(9, "add", UINT4) as again:  # re-entering a node reuses its group
        t = b.not_(s, role="late")
    assert again == node_group
    netlist = b.netlist
    group = netlist.groups[node_group]
    assert (group.name, group.kind, group.parent, group.ir_node) == ("n9.add", "ir_node", 0, 9)
    assert dict(group.attrs) == {"op": "add", "type": "uint4"}
    assert (netlist.groups[adder].kind, dict(netlist.groups[adder].attrs)) == ("helper", {"width": 4})
    assert netlist.groups[slice_].ir_node == 9  # inherited from the enclosing IR-node group
    assert netlist.group_path(slice_) == ("design", "n9.add", "adder", "bit2")
    assert netlist.instance(s.instance).provenance == Provenance(9, "sum_xor", 2, slice_)
    assert netlist.instance(t.instance).provenance == Provenance(9, "late", None, node_group)
    assert netlist.instance(x.instance).provenance == Provenance(0, "input_bit", 0, port_group, "in_x")
    assert netlist.instance(y.instance).provenance == Provenance(None, "input_bit", 0, 0, "in_y")
    port = netlist.groups[port_group]
    assert (port.kind, port.ir_node, dict(port.attrs)) == ("port", 0, {"direction": "input", "port": "in_x"})
    # An input port IS its IR node.
    assert (b.group_of_ir_node(0), b.group_of_ir_node(9), b.group_of_ir_node(5)) == (port_group, node_group, None)
    assert netlist.descendants(node_group) == {node_group, adder, slice_}
    expected = [netlist.instance(s.instance), netlist.instance(t.instance)]
    assert netlist.instances_in_group(node_group) == netlist.instances_of_ir_node(9) == expected
    assert netlist.instances_in_group(slice_) == expected[:1]


def test_builder_constants_are_created_once_per_ir_node_and_value() -> None:
    b = PrimitiveBuilder()
    with b.ir_node(3, "add", UINT8) as group3, b.scope("adder"):
        one = b.const(True)
        assert b.const(True) == one and b.const(1) == one
        zero = b.const(False)
    with b.ir_node(3, "add", UINT8):
        assert (b.const(True), b.const(False)) == (one, zero)
    with b.ir_node(4, "sub", UINT8) as group4:
        other = b.const(True)
    root_one = b.const(True)
    assert len({one, zero, other, root_one}) == 4
    netlist = b.netlist
    assert (netlist.kind_counts()["const1"], netlist.kind_counts()["const0"]) == (3, 1)
    # Constants live in their node's TOP-LEVEL group, whatever scope asked.
    assert netlist.instance(one.instance).provenance == Provenance(3, "const1", None, group3)
    assert netlist.instance(zero.instance).provenance == Provenance(3, "const0", None, group3)
    assert netlist.instance(other.instance).provenance == Provenance(4, "const1", None, group4)
    assert netlist.instance(root_one.instance).provenance == Provenance(None, "const1", None, 0)


# == (b) the structure of synthesized netlists ==========================================


def test_the_primitive_vocabulary_has_no_room_for_wide_operations() -> None:
    assert GATE_KINDS == {K.AND, K.OR, K.XOR, K.NOT}
    assert NON_GATE_KINDS == set(PrimitiveKind) - GATE_KINDS
    assert NON_GATE_KINDS == BOUNDARY_KINDS | STATE_KINDS | {K.PERIPHERAL}
    assert FORBIDDEN_PRIMITIVE_NAMES == set(OPS) - {"and", "or", "xor", "not"}
    assert {"add", "sub", "mul", "div", "mod", "mux", "shl", "shr", "eq", "lt", "cast", "neg", "inv"} <= FORBIDDEN_PRIMITIVE_NAMES
    assert {k.value for k in PrimitiveKind}.isdisjoint(FORBIDDEN_PRIMITIVE_NAMES)
    for name in ("add", "mul", "div", "mux", "adder", "register"):
        with pytest.raises(ValueError):
            PrimitiveKind(name)
    assert {k: k.category for k in PrimitiveKind} == {
        **{k: "gate" for k in GATE_KINDS},
        K.REGISTER_BIT: "state",
        K.CONST0: "constant",
        K.CONST1: "constant",
        K.INPUT_BIT: "boundary",
        K.OUTPUT_BIT: "boundary",
        K.CLOCK_SOURCE: "control",
        K.RESET_SOURCE: "control",
        K.PERIPHERAL: "peripheral",
    }
    assert all(k.is_gate == (k in GATE_KINDS) for k in PrimitiveKind)


def _vectors(graph: Graph, rng: random.Random, limit: int = 4096) -> list[dict[str, int]]:
    """Every input combination if there are at most ``limit``, else 400 random ones."""
    names = [p["name"] for p in graph.inputs]
    widths = [p["type"]["width"] for p in graph.inputs]
    if 1 << sum(widths) <= limit:
        return [dict(zip(names, combo)) for combo in itertools.product(*(range(1 << w) for w in widths))]
    return [{n: rng.getrandbits(w) for n, w in zip(names, widths)} for _ in range(400)]


@pytest.mark.parametrize("policy", POLICIES, ids=POLICY_IDS)
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_synthesized_netlists_hold_only_one_bit_basis_logic(name: str, policy) -> None:
    graph, netlist = program(name, policy)
    netlist.validate(complete=True)
    assert netlist.nets and all(net.width == 1 for net in netlist.nets)
    kinds = {inst.kind for inst in netlist.instances}
    allowed = GATE_KINDS | BOUNDARY_KINDS
    if graph.sequential:
        allowed |= STATE_KINDS
    if policy == DEVICE_POLICY:
        allowed |= {K.PERIPHERAL}
    assert kinds & GATE_KINDS and kinds <= allowed
    assert netlist.is_sequential == graph.sequential
    document = netlist.to_dict()
    assert {record["kind"] for record in document["instances"]}.isdisjoint(FORBIDDEN_PRIMITIVE_NAMES)
    assert {record["width"] for record in document["nets"]} == {1}
    for inst in gates(netlist):
        # Gates implement live operations (a register's enable mux included),
        # never boundaries.
        assert netlist.ir_nodes[inst.provenance.ir_node].op in OPS | {"register"}
        assert inst.provenance.role and inst.provenance.port is None
    # ... and they compute the program.
    sim = PrimitiveSimulator(netlist)
    if graph.sequential:
        for n in range(4):
            assert sim.run(in_n=n) == graph.run(in_n=n)
        return
    vectors = _vectors(graph, random.Random(name))
    got = sim.evaluate_many({p["name"]: [v[p["name"]] for v in vectors] for p in graph.inputs})
    for index, vector in enumerate(vectors):
        assert {k: values[index] for k, values in got.items()} == graph.evaluate(**vector), vector


@pytest.mark.parametrize("policy", POLICIES, ids=POLICY_IDS)
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_provenance_agrees_with_the_hierarchy(name: str, policy) -> None:
    _, netlist = program(name, policy)
    (root,) = [g for g in netlist.groups if g.parent is None]
    assert (root.id, root.name, root.kind, root.ir_node) == (0, "design", "root", None)
    assert {g.kind for g in netlist.groups} <= GROUP_KINDS
    for group in netlist.groups[1:]:
        parent = netlist.groups[group.parent]
        if parent.kind == "root":
            assert group.kind in {"ir_node", "port", "control"}
        else:  # everything below an IR node's group belongs to that node
            assert group.ir_node == parent.ir_node and group.kind in {"helper", "slice", "stage", "iteration"}
    for inst in netlist.instances:
        provenance = inst.provenance
        group = netlist.groups[provenance.group]
        assert group.ir_node == provenance.ir_node, inst
        if provenance.ir_node is None:  # only the global control has no IR node
            assert inst.kind in {K.CLOCK_SOURCE, K.RESET_SOURCE} and group.kind == "control"
        if provenance.port is not None:
            assert inst.kind in {K.INPUT_BIT, K.OUTPUT_BIT, K.PERIPHERAL}
            assert group.kind == "port" and dict(group.attrs)["port"] == provenance.port
    # Every live non-input IR node owns exactly one top-level group, named after it.
    top = [g for g in netlist.groups if g.kind == "ir_node"]
    assert sorted(g.ir_node for g in top) == sorted(n for n, info in netlist.ir_nodes.items() if info.op != "input")
    for group in top:
        assert group.name == f"n{group.ir_node}.{netlist.ir_nodes[group.ir_node].op}"


def test_a_signal_used_twice_is_one_net_with_two_sinks() -> None:
    graph = Graph()
    a, b = graph.input("in_a", BOOL), graph.input("in_b", BOOL)
    both = graph.node("and", BOOL, (a, b))
    graph.output("result", graph.node("xor", BOOL, (a, both)))
    graph.output("copy", b)
    netlist = synth(graph)
    (a_bit,) = netlist.port("in_a").bits
    (b_bit,) = netlist.port("in_b").bits
    and_gate = netlist.instance(netlist.buses[both.id].bits[0].instance)
    (xor_gate,) = [g for g in gates(netlist) if g.kind is K.XOR]
    for bit, readers in ((a_bit, {T(and_gate.id, "a"), T(xor_gate.id, "a")}), (b_bit, {T(and_gate.id, "b"), netlist.port("copy").bits[0]})):
        nets = [net for net in netlist.nets if net.driver == bit]
        assert len(nets) == 1, bit
        (net,) = nets
        assert net.fanout == 2 and set(net.sinks) == readers
        assert {netlist.driver_of(sink) for sink in readers} == {bit}
    # Every driver appears once: no net is ever split per sink.
    drivers = [net.driver for net in netlist.nets]
    assert len(drivers) == len(set(drivers))
    assert netlist.fanout_distribution() == {"1": 2, "2": 2}


def test_a_uint8_input_is_eight_input_bits_on_eight_nets() -> None:
    for policy in POLICIES:
        netlist = synth(compile_source("uint8 main(uint8 a) { return ~a; }"), policy)
        port = netlist.port("in_a")
        assert (port.direction, port.type, port.source_name, port.realization) == ("input", UINT8, "a", "pads")
        assert len(port.bits) == 8 and len(port.instances) == 8
        assert netlist.kind_counts()["input_bit"] == 8
        for i, bit in enumerate(port.bits):
            pad = netlist.instance(bit.instance)
            assert pad.kind is K.INPUT_BIT and bit == pad.output()
            assert (pad.provenance.port, pad.provenance.bit, pad.provenance.ir_node) == ("in_a", i, port.ir_node)
        nets = [netlist.net_of_driver(bit) for bit in port.bits]
        assert len({net.id for net in nets if net is not None}) == 8
        assert all(net is not None and net.width == 1 and net.fanout == 1 for net in nets)
        assert len(netlist.nets) == 16 and netlist.gate_count == 8  # 8 input nets + 8 inverter nets


def test_logical_ports_and_buses_are_lsb_first_metadata() -> None:
    graph = Graph()
    a = graph.input("in_a", INT8)
    b = graph.input("in_b", UINT8)
    low, high = graph.constant(0x01, UINT8), graph.constant(0x80, UINT8)
    as_unsigned = graph.node("cast", UINT8, (a,))  # signedness only: pure wiring
    total = graph.node("add", UINT8, (as_unsigned, b))
    graph.output("result", total)
    graph.output("low", low)
    graph.output("high", high)
    netlist = synth(graph)
    netlist.validate(complete=True)
    kind = [netlist.instance(bit.instance).kind for bit in netlist.buses[low.id].bits]
    assert kind == [K.CONST1] + [K.CONST0] * 7  # bits[0] is the least significant bit
    kind = [netlist.instance(bit.instance).kind for bit in netlist.buses[high.id].bits]
    assert kind == [K.CONST0] * 7 + [K.CONST1]
    for port in netlist.ports:
        for i, bit in enumerate(port.bits):
            assert netlist.instance(bit.instance).provenance.bit == i
    # The cast is the SAME signals under another name: no primitive, no net.
    source = netlist.port("in_a").bits
    assert netlist.buses[as_unsigned.id].bits == netlist.buses[a.id].bits == source
    assert netlist.instances_of_ir_node(as_unsigned.id) == []
    names = netlist.bus_memberships()
    for i, bit in enumerate(source):
        assert names[bit] == [(a.id, i), (as_unsigned.id, i)]
        assert sum(net.driver == bit for net in netlist.nets) == 1
    # Buses and ports never add routing: the nets are exactly the drivers with sinks.
    drivers = [net.driver for net in netlist.nets]
    assert len(drivers) == len(set(drivers))
    assert set(drivers) == {
        T(inst.id, pin) for inst in netlist.instances for pin in inst.outputs if netlist.net_of_driver(T(inst.id, pin))
    }
    assert netlist.port_memberships()[netlist.port("result").bits[3]] == [("result", 3)]
    sim = PrimitiveSimulator(netlist)
    assert sim.evaluate(in_a=-1, in_b=1) == graph.evaluate(in_a=-1, in_b=1) == {"result": 0, "low": 1, "high": 128}


def test_dead_ir_produces_no_primitives() -> None:
    alive = synth(compile_source("uint8 main(uint8 a, uint8 b) { return a + b; }"))
    graph = compile_source(
        "uint8 main(uint8 a, uint8 b) { uint8 unused = a * b; uint8 worse = unused / (a | 1); return a + b; }"
    )
    live = {node["id"] for node in graph.live_nodes()}
    dead = [node for node in graph.nodes if node["id"] not in live]
    assert {"mul", "div", "or"} <= {node["op"] for node in dead}
    netlist = synth(graph)
    for node in dead:
        assert node["id"] not in netlist.ir_nodes and node["id"] not in netlist.buses
        assert netlist.instances_of_ir_node(node["id"]) == []
        assert all(group.ir_node != node["id"] for group in netlist.groups)
    assert set(netlist.graph_summary["ops"]) == {"input", "add"}
    assert netlist.summary() == alive.summary()  # exactly the plain adder
    assert netlist.kind_counts()["const0"] == netlist.kind_counts()["const1"] == 0


def test_add_provenance_and_hierarchy_queries() -> None:
    graph, netlist = program("uint4_add", PAD_POLICY)
    (add_node,) = [info for info in netlist.ir_nodes.values() if info.op == "add"]
    node = add_node.id
    group = group_named(netlist, f"n{node}.add")
    assert (group.kind, group.parent, group.ir_node, dict(group.attrs)) == ("ir_node", 0, node, {"op": "add", "type": "uint4"})
    assert netlist.group_path(group.id) == ("design", f"n{node}.add")
    adder = group_named(netlist, "ripple_adder")
    slices = {g.name: g for g in netlist.groups if g.kind == "slice"}
    assert set(slices) == {"bit0", "bit1", "bit2", "bit3"}
    assert netlist.descendants(group.id) == {group.id, adder.id} | {g.id for g in slices.values()}
    assert netlist.descendants(0) == set(range(len(netlist.groups)))
    every_gate = gates(netlist)
    assert len(every_gate) == 5 * 4 - 6  # ripple carry: half adder, two full adders, a sum-only top slice
    assert netlist.instances_in_group(group.id) == every_gate == netlist.instances_in_group(adder.id)
    # The output pads observe the node, so they are attributed to it too (in their port group).
    pads = [inst for inst in netlist.instances if inst.kind is K.OUTPUT_BIT]
    assert netlist.instances_of_ir_node(node) == every_gate + pads
    assert {netlist.groups[p.provenance.group].name for p in pads} == {"port.result"}
    roles = {
        0: {"sum_xor", "carry_generate"},
        1: {"propagate_xor", "sum_xor", "carry_generate", "carry_propagate_and", "carry_or"},
        2: {"propagate_xor", "sum_xor", "carry_generate", "carry_propagate_and", "carry_or"},
        3: {"propagate_xor", "sum_xor"},
    }
    for i in range(4):
        slice_ = slices[f"bit{i}"]
        assert netlist.group_path(slice_.id) == ("design", f"n{node}.add", "ripple_adder", f"bit{i}")
        assert (slice_.parent, slice_.ir_node, dict(slice_.attrs)["bit"]) == (adder.id, node, i)
        members = netlist.instances_in_group(slice_.id)
        assert members == [g for g in every_gate if g.provenance.bit == i]
        assert {m.provenance.role for m in members} == roles[i] and len(members) == len(roles[i])
        assert all(m.provenance.ir_node == node and m.provenance.group == slice_.id for m in members)
    # The sum bits are the bus of the node, LSB first.
    sums = [g for g in every_gate if g.provenance.role == "sum_xor"]
    assert netlist.buses[node].bits == tuple(g.output() for g in sorted(sums, key=lambda g: g.provenance.bit))
    for name, port_node in (("in_a", graph.inputs[0]["node"]), ("in_b", graph.inputs[1]["node"])):
        port_group = group_named(netlist, f"port.{name}")
        assert (port_group.kind, port_group.ir_node) == ("port", port_node)
        members = netlist.instances_in_group(port_group.id)
        assert members == netlist.instances_of_ir_node(port_node)
        assert [(m.kind, m.provenance.port, m.provenance.bit) for m in members] == [(K.INPUT_BIT, name, i) for i in range(4)]


def test_constants_are_shared_within_one_ir_node_only() -> None:
    graph = Graph()
    a = graph.input("in_a", UINT8)
    low, high = graph.constant(0x0F, UINT8), graph.constant(0xF0, UINT8)
    mixed = graph.node("xor", UINT8, (a, low))
    diff = graph.node("sub", UINT8, (mixed, high))  # x + NOT(y) + 1: a constant 1 of its own
    graph.output("result", diff)
    netlist = synth(graph)
    assert (netlist.kind_counts()["const1"], netlist.kind_counts()["const0"]) == (3, 2)
    for node, ones in ((low.id, slice(0, 4)), (high.id, slice(4, 8))):
        bits = netlist.buses[node].bits
        one = set(bits[ones])
        zero = set(bits) - one
        assert len(one) == len(zero) == 1  # one CONST1 and one CONST0 per node, fanned out
        (one_bit,), (zero_bit,) = one, zero
        assert (netlist.instance(one_bit.instance).kind, netlist.instance(zero_bit.instance).kind) == (K.CONST1, K.CONST0)
        top = group_named(netlist, f"n{node}.const")
        for bit in (one_bit, zero_bit):
            provenance = netlist.instance(bit.instance).provenance
            assert (provenance.ir_node, provenance.group, provenance.bit) == (node, top.id, None)
    (carry_in,) = [i for i in netlist.instances_of_ir_node(diff.id) if i.kind is K.CONST1]
    # ... in the node's top-level group, even though the subtractor asked for it.
    assert carry_in.provenance.group == group_named(netlist, f"n{diff.id}.sub").id
    assert carry_in.output() not in netlist.buses[low.id].bits + netlist.buses[high.id].bits
    values = list(range(256))
    assert PrimitiveSimulator(netlist).evaluate_many({"in_a": values})["result"] == [
        graph.evaluate(in_a=v)["result"] for v in values
    ]


@pytest.mark.parametrize(
    ("name", "policy"),
    [("uint4_add", PAD_POLICY), ("counter", DEVICE_POLICY)],
    ids=["combinational", "sequential-peripherals"],
)
def test_to_dict_is_a_json_native_primitive_netlist_document(name: str, policy) -> None:
    graph, netlist = program(name, policy)
    document = netlist.to_dict()
    assert_json_native(document)
    assert json.loads(json.dumps(document)) == document
    assert SCHEMA == document["schema"] == "redc.primitive-netlist.v1"
    assert list(document) == [
        "schema", "backend", "stage", "sequential", "bit_order", "basis", "ir",
        "summary", "ir_nodes", "groups", "ports", "buses", "instances", "nets",
    ]  # fmt: skip
    assert (document["backend"], document["stage"], document["bit_order"]) == ("physical-primitive", "primitive", "lsb_first")
    assert document["basis"] == ["and", "not", "or", "xor"]
    assert document["sequential"] is graph.sequential
    assert document["summary"] == netlist.summary()
    assert document["summary"]["gates"] == sum(document["summary"]["gates_by_kind"].values())
    assert [r["id"] for r in document["instances"]] == list(range(len(netlist.instances)))
    assert [r["id"] for r in document["nets"]] == list(range(len(netlist.nets)))
    assert [r["id"] for r in document["groups"]] == list(range(len(netlist.groups)))
    assert [r["id"] for r in document["ir_nodes"]] == sorted(netlist.ir_nodes)
    for record in document["nets"]:
        assert record["width"] == 1 and record["fanout"] == len(record["sinks"]) >= 1
        assert set(record["driver"]) == {"instance", "pin"}
    for record, inst in zip(document["instances"], netlist.instances):
        assert record["kind"] == inst.kind.value and record["role"] == inst.provenance.role
        assert ("init" in record) == (inst.kind is K.REGISTER_BIT)
        assert ("peripheral" in record) == (inst.kind is K.PERIPHERAL)
        assert ("port" in record) == (inst.provenance.port is not None)
        if inst.kind is K.REGISTER_BIT:
            assert record["init"] == int(inst.init)
    for record in document["ports"]:
        assert len(record["bits"]) == record["type"]["width"]
    if graph.sequential:
        assert {r["kind"] for r in document["instances"]} >= {"register_bit", "clock_source", "reset_source", "peripheral"}
        assert {r["role"] for r in document["nets"]} == {"data", "clock", "reset"}
    # Deterministic: the same graph always gives byte-identical JSON.
    again = synth(compile_source(PROGRAMS[name]), policy).to_dict()
    assert json.dumps(again, sort_keys=True) == json.dumps(document, sort_keys=True)


# == (c) interface policies ==============================================================


def test_interface_policy_rules() -> None:
    assert DEFAULT_INTERFACE_POLICY == DEVICE_POLICY
    assert DEVICE_POLICY.realize_input("start", BOOL) == PortRealization(LEVER)
    assert DEVICE_POLICY.realize_output("result", UINT8) == PortRealization(TWO_DIGIT_SEVEN_SEGMENT)
    for name, typ in (("start", IRType(1)), ("start", UINT8), ("in_start", BOOL), ("result", UINT8)):
        assert DEVICE_POLICY.realize_input(name, typ) == PADS, (name, typ)
    for name, typ in (("result", INT8), ("result", IRType(16)), ("result", BOOL), ("done", BOOL), ("start", BOOL), ("out", UINT8)):
        assert DEVICE_POLICY.realize_output(name, typ) == PADS, (name, typ)
    for name, typ in (("start", BOOL), ("result", UINT8), ("in_a", INT8)):
        assert PAD_POLICY.realize_input(name, typ) == PAD_POLICY.realize_output(name, typ) == PADS
    assert (PADS.peripheral, PADS.label) == (None, "pads")
    assert (PortRealization(LEVER).label, PortRealization(TWO_DIGIT_SEVEN_SEGMENT).label) == ("lever", "2-dig-7-seg")


def test_the_default_policy_makes_start_one_lever() -> None:
    _, netlist = example("uint8_fib", DEVICE_POLICY)
    start = netlist.port("start")
    assert (start.direction, start.type, start.realization) == ("input", BOOL, "lever")
    (lever_id,) = start.instances
    lever = netlist.instance(lever_id)
    assert lever.kind is K.PERIPHERAL and lever.peripheral == PeripheralSpec("lever", INPUT, 1)
    assert (lever.inputs, lever.outputs) == ((), ("b0",))
    assert start.bits == (T(lever_id, "b0"),)
    assert (lever.provenance.port, lever.provenance.role, lever.provenance.ir_node) == ("start", "peripheral", start.ir_node)
    net = netlist.net_of_driver(start.bits[0])
    assert net is not None and net.role == "data" and net.width == 1
    assert netlist.kind_counts()["peripheral"] == 2  # the lever and the display
    assert [p.realization for p in netlist.ports] == ["pads", "lever", "2-dig-7-seg", "pads"]


@pytest.mark.parametrize("name", sorted(EXAMPLE_TOPS))
def test_the_default_policy_makes_result_one_display_with_eight_independent_pins(name: str) -> None:
    _, netlist = example(name, DEVICE_POLICY)
    result = netlist.port("result")
    assert (result.direction, result.type, result.realization) == ("output", UINT8, "2-dig-7-seg")
    (display_id,) = result.instances
    display = netlist.instance(display_id)
    assert display.peripheral == PeripheralSpec("2-dig-7-seg", OUTPUT, 8)
    assert (display.inputs, display.outputs) == (tuple(f"b{i}" for i in range(8)), ())
    assert result.bits == tuple(T(display_id, f"b{i}") for i in range(8))
    assert sum(inst.kind is K.PERIPHERAL and inst.peripheral.kind == "2-dig-7-seg" for inst in netlist.instances) == 1
    assert netlist.kind_counts()["output_bit"] == (1 if name == "uint8_fib" else 0)  # only `done` is a pad
    source = netlist.buses[result.ir_node].bits
    nets = []
    for i, pin in enumerate(result.bits):
        driver = netlist.driver_of(pin)
        assert driver == source[i]  # bit i of the value, LSB first
        net = netlist.net_of_driver(driver)
        assert net is not None and net.width == 1 and pin in net.sinks
        assert sum(sink.instance == display_id for sink in net.sinks) == 1  # one display pin per net
        nets.append(net.id)
    assert len(set(nets)) == 8


@pytest.mark.parametrize("name", sorted(EXAMPLE_TOPS))
def test_the_pad_policy_makes_one_pad_per_bit(name: str) -> None:
    graph, netlist = example(name, PAD_POLICY)
    assert netlist.kind_counts()["peripheral"] == 0
    for port in netlist.ports:
        kind, pin = (K.INPUT_BIT, "y") if port.direction == "input" else (K.OUTPUT_BIT, "a")
        assert port.realization == "pads" and len(port.instances) == port.type.width
        assert port.bits == tuple(T(i, pin) for i in port.instances)
        for i, bit in enumerate(port.bits):
            pad = netlist.instance(bit.instance)
            assert pad.kind is kind and (pad.provenance.port, pad.provenance.bit) == (port.name, i)
    widths = sum(p["type"]["width"] for p in graph.inputs), sum(p["type"]["width"] for p in graph.outputs)
    assert (netlist.kind_counts()["input_bit"], netlist.kind_counts()["output_bit"]) == widths


@pytest.mark.parametrize("name", sorted(EXAMPLE_TOPS))
def test_port_metadata_is_the_same_under_every_policy(name: str) -> None:
    graph = example_graph(name)
    expected = [
        (p["name"], "input", IRType(**p["type"]), p["source_name"], p["node"], p["type"]["width"]) for p in graph.inputs
    ] + [(p["name"], "output", IRType(**p["type"]), None, p["node"], p["type"]["width"]) for p in graph.outputs]
    for policy in POLICIES:
        _, netlist = example(name, policy)
        assert [(p.name, p.direction, p.type, p.source_name, p.ir_node, len(p.bits)) for p in netlist.ports] == expected
        records = netlist.to_dict()["ports"]
        assert [(r["name"], r["realization"]) for r in records] == [(p.name, p.realization) for p in netlist.ports]
        assert netlist.graph_summary["inputs"] == [p["name"] for p in graph.inputs]
        assert netlist.graph_summary["outputs"] == [p["name"] for p in graph.outputs]
    # Everything but the boundary is identical.
    (_, pads), (_, devices) = (example(name, policy) for policy in POLICIES)
    assert pads.gate_count == devices.gate_count
    assert pads.kind_counts()["register_bit"] == devices.kind_counts()["register_bit"]


def test_pads_and_peripherals_simulate_identically() -> None:
    graph, pads = example("uint8_fib", PAD_POLICY)
    _, devices = example("uint8_fib", DEVICE_POLICY)
    for n in (0, 4, 11):
        assert PrimitiveSimulator(pads).run(in_n=n) == PrimitiveSimulator(devices).run(in_n=n) == graph.run(in_n=n)
    graph, pads = example("uint8_add", PAD_POLICY)
    _, devices = example("uint8_add", DEVICE_POLICY)
    rng = random.Random(8)
    columns = {name: [rng.getrandbits(8) for _ in range(256)] for name in ("in_a", "in_b")}
    got = PrimitiveSimulator(devices).evaluate_many(columns)
    assert got == PrimitiveSimulator(pads).evaluate_many(columns)
    assert got["result"] == [(a + b) & 0xFF for a, b in zip(columns["in_a"], columns["in_b"])]


# == (d) the PrimitiveSimulator =============================================================


def _mixed_graph() -> Graph:
    """Signed and unsigned values, a zero extension, three outputs."""
    graph = Graph()
    a = graph.input("in_a", INT8)
    b = graph.input("in_b", UINT4)
    c = graph.input("in_c", BOOL)
    wide = graph.node("cast", INT8, (b,))
    graph.output("diff", graph.node("sub", INT8, (a, wide)))
    graph.output("less", graph.node("lt", BOOL, (a, wide)))
    graph.output("pick", graph.node("mux", UINT4, (c, b, graph.node("inv", UINT4, (b,)))))
    return graph


def test_evaluate_many_is_evaluate_for_every_vector_at_once() -> None:
    graph = _mixed_graph()
    sim = PrimitiveSimulator(synth(graph))
    assert (sim.input_names, sim.output_names, sim.is_sequential) == (["in_a", "in_b", "in_c"], ["diff", "less", "pick"], False)
    rng = random.Random(20261004)
    vectors = [{"in_a": rng.randrange(-128, 128), "in_b": rng.randrange(16), "in_c": rng.getrandbits(1)} for _ in range(300)]
    many = sim.evaluate_many({name: [v[name] for v in vectors] for name in sim.input_names})
    assert set(many) == {"diff", "less", "pick"} and {len(values) for values in many.values()} == {300}
    for k, vector in enumerate(vectors):
        assert sim.evaluate(**vector) == {name: values[k] for name, values in many.items()} == graph.evaluate(**vector)
    single = sim.evaluate_many({name: [vectors[0][name]] for name in sim.input_names})
    assert single == {name: [values[0]] for name, values in many.items()}
    # The whole 13-bit input space in ONE settle.
    space = list(itertools.product(range(-128, 128), range(16), range(2)))
    exhaustive = sim.evaluate_many({name: [v[k] for v in space] for k, name in enumerate(sim.input_names)})
    for index, (a, b, c) in enumerate(space):
        expected = graph.evaluate(in_a=a, in_b=b, in_c=c)
        assert {name: values[index] for name, values in exhaustive.items()} == expected


def test_evaluate_many_settles_every_vector_in_one_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = PrimitiveSimulator(synth(_mixed_graph()))
    masks: list[int] = []
    settle = sim._settle

    def counting(*args):
        masks.append(args[-1])
        return settle(*args)

    monkeypatch.setattr(sim, "_settle", counting)
    rng = random.Random(4096)
    columns = {"in_a": [rng.getrandbits(8) for _ in range(4096)], "in_b": [rng.getrandbits(4) for _ in range(4096)]}
    columns["in_c"] = [rng.getrandbits(1) for _ in range(4096)]
    sim.evaluate_many(columns)
    assert masks == [(1 << 4096) - 1]  # ONE settle, one bit lane per vector


def test_a_gate_may_read_one_signal_on_both_pins() -> None:
    b = PrimitiveBuilder()
    x = b.input_bit("in_x", 0)
    same, never = b.and_(x, x), b.xor(x, x)
    observed = (b.output_bit("same", 0, same), b.output_bit("never", 0, never))
    netlist = b.finish()
    (net,) = [n for n in netlist.nets if n.driver == x]
    assert net.sinks == (T(1, "a"), T(1, "b"), T(2, "a"), T(2, "b"))  # still ONE net
    netlist.ports = [
        LogicalPort("in_x", "input", BOOL, "x", 0, (x,)),
        LogicalPort("same", "output", BOOL, None, 1, observed[:1]),
        LogicalPort("never", "output", BOOL, None, 2, observed[1:]),
    ]
    netlist.validate(complete=True)
    assert PrimitiveSimulator(netlist).evaluate_many({"in_x": [0, 1]}) == {"same": [0, 1], "never": [0, 0]}


@pytest.mark.parametrize(
    ("drop", "pin"),
    [(0, "pin 'a' of primitive 2"), (1, "pin 'b' of primitive 2"), (2, "pin 'd' of primitive 3"), (4, "pin 'rst' of primitive 3"), (5, "pin 'a' of primitive 6")],
    ids=["and-a", "and-b", "register-d", "register-rst", "output-pad"],
)
def test_the_simulator_refuses_a_netlist_it_cannot_settle(drop: int, pin: str) -> None:
    netlist = parts(COMPLETE_WIRING[:drop] + COMPLETE_WIRING[drop + 1 :])
    netlist.ports = [
        LogicalPort("in_a", "input", BOOL, "a", 0, (T(A, "y"),)),
        LogicalPort("in_b", "input", BOOL, "b", 1, (T(B, "y"),)),
        LogicalPort("result", "output", BOOL, None, 3, (T(OUT, "a"),)),
    ]
    with pytest.raises(CompileError, match=re.escape(f"simulation: {pin} is undriven")):
        PrimitiveSimulator(netlist)
    PrimitiveSimulator(parts())  # the complete original is fine (it has no ports, so nothing to observe)


def test_inputs_are_masked_and_outputs_signed_like_the_ir() -> None:
    graph = _mixed_graph()
    sim = PrimitiveSimulator(synth(graph))
    for vector in (
        {"in_a": 200, "in_b": 17, "in_c": 2},
        {"in_a": -129, "in_b": -1, "in_c": -1},
        {"in_a": 2**40 + 5, "in_b": 2**70 + 3, "in_c": 3},
    ):
        assert sim.evaluate(**vector) == graph.evaluate(**vector), vector
    assert sim.evaluate(in_a=-100, in_b=15, in_c=0) == {"diff": -115, "less": 1, "pick": 0}
    columns = {"in_a": [-1, 255, 511], "in_b": [16, 0, 32], "in_c": [1, 3, 5]}
    assert sim.evaluate_many(columns) == {"diff": [-1, -1, -1], "less": [1, 1, 1], "pick": [0, 0, 0]}


def test_simulator_input_errors() -> None:
    graph = _mixed_graph()
    sim = PrimitiveSimulator(synth(graph))
    with pytest.raises(CompileError, match="missing input in_c"):
        graph.evaluate(in_a=1, in_b=2)  # the IR's own wording ...
    with pytest.raises(CompileError, match="missing input in_c"):
        sim.evaluate(in_a=1, in_b=2)  # ... is the simulator's
    with pytest.raises(CompileError, match="missing input in_b"):
        sim.evaluate_many({"in_a": [1], "in_c": [0]})
    with pytest.raises(CompileError, match="every input needs the same number of vectors"):
        sim.evaluate_many({"in_a": [1, 2], "in_b": [1], "in_c": [0, 1]})
    with pytest.raises(CompileError, match="need at least one vector"):
        sim.evaluate_many({"in_a": [], "in_b": [], "in_c": []})
    with pytest.raises(CompileError, match=re.escape("run() is for sequential netlists; use evaluate()")):
        sim.run(in_a=1, in_b=2, in_c=0)
    fib_graph, netlist = example("uint8_fib", PAD_POLICY)
    seq = PrimitiveSimulator(netlist)
    assert (seq.input_names, seq.output_names, seq.is_sequential) == (["in_n", "start"], ["result", "done"], True)
    for call in (lambda: seq.evaluate(in_n=3, start=1), lambda: seq.evaluate_many({"in_n": [3], "start": [1]})):
        with pytest.raises(CompileError, match=re.escape("sequential primitive netlist holds state; use step() or run()")):
            call()
    with pytest.raises(CompileError, match="missing input start"):
        fib_graph.step(fib_graph.reset_state(), in_n=3)
    with pytest.raises(CompileError, match="missing input start"):
        seq.step(seq.reset_state(), in_n=3)
    no_done = _initialized_graph()
    with pytest.raises(CompileError, match="has no 'done' output to wait on"):
        no_done.run(in_x=1)
    with pytest.raises(CompileError, match="has no 'done' output to wait on"):
        PrimitiveSimulator(synth(no_done)).run(in_x=1)


@pytest.mark.parametrize("policy", POLICIES, ids=POLICY_IDS)
def test_fib_transactions_from_reset_match_the_ir(policy) -> None:
    graph, netlist = example("uint8_fib", policy)
    sim = PrimitiveSimulator(netlist)
    for n in (0, 1, 2, 3, 7, 12, 13, 30):
        expected = graph.run(in_n=n)
        assert sim.run(in_n=n) == expected == {"result": py_fib(n), "done": 1}
    for run in (graph.run, sim.run):  # the same cycle budget, the same complaint
        with pytest.raises(CompileError, match="simulation did not finish within 5 cycles"):
            run(max_cycles=5, in_n=30)


def test_fib_back_to_back_transactions_step_like_the_ir() -> None:
    """Several transactions through ONE persistent state (no reset between
    them), compared with ``Graph.step`` and the IR register state every cycle."""
    graph, netlist = example("uint8_fib", PAD_POLICY)
    sim = PrimitiveSimulator(netlist)
    ir_state, state = graph.reset_state(), sim.reset_state()
    assert sim.state_to_ir(state) == ir_state and sim.state_from_ir(ir_state) == state
    results, cycles = [], 0
    for n in (5, 1, 13, 2, 9, 0, 20):
        for cycle in range(100):
            stimulus = {"start": int(cycle == 0), "in_n": n}
            ir_outputs, ir_state = graph.step(ir_state, **stimulus)
            outputs, state = sim.step(state, **stimulus)
            cycles += 1
            assert outputs == ir_outputs, (n, cycle)
            assert sim.state_to_ir(state) == ir_state, (n, cycle)
            assert sim.state_from_ir(ir_state) == state, (n, cycle)
            if cycle > 0 and outputs["done"]:
                results.append(outputs["result"])
                break
        else:
            pytest.fail(f"fib({n}) never finished")
    assert results == [py_fib(n) for n in (5, 1, 13, 2, 9, 0, 20)]
    assert cycles > 40


def test_a_reset_mid_transaction_returns_fib_to_power_up() -> None:
    graph, netlist = example("uint8_fib", PAD_POLICY)
    sim = PrimitiveSimulator(netlist)
    state = sim.reset_state()
    for cycle in range(6):
        _, state = sim.step(state, start=int(cycle == 0), in_n=40)
    assert state != sim.reset_state()
    _, state = sim.step(state, rst=1, start=0, in_n=40)
    assert state == sim.reset_state() and sim.state_to_ir(state) == graph.reset_state()
    # The abandoned transaction leaves no trace: the next one runs like the first ever.
    ir_state = graph.reset_state()
    for cycle in range(60):
        stimulus = {"start": int(cycle == 0), "in_n": 6}
        ir_outputs, ir_state = graph.step(ir_state, **stimulus)
        outputs, state = sim.step(state, **stimulus)
        assert outputs == ir_outputs and sim.state_to_ir(state) == ir_state
        if cycle > 0 and outputs["done"]:
            assert outputs["result"] == py_fib(6) == 8
            break
    else:
        pytest.fail("fib(6) never finished after the reset")


def test_state_conversion_round_trips_and_any_state_steps_like_the_ir() -> None:
    graph, netlist = example("uint8_fib", PAD_POLICY)
    sim = PrimitiveSimulator(netlist)
    registers = {node: IRType(**graph.nodes[node]["type"]) for node in graph.reset_state()}
    bits = {inst.id for inst in netlist.instances if inst.kind is K.REGISTER_BIT}
    assert set(sim.reset_state()) == bits and sum(t.width for t in registers.values()) == len(bits)
    rng = random.Random(77)
    for _ in range(60):
        ir_state = {node: rng.getrandbits(typ.width) for node, typ in registers.items()}
        state = sim.state_from_ir(ir_state)
        assert set(state) == bits and set(state.values()) <= {0, 1}
        assert sim.state_to_ir(state) == ir_state
        # Even unreachable states evolve exactly like the IR's.
        stimulus = {"start": rng.getrandbits(1), "in_n": rng.getrandbits(8)}
        ir_outputs, ir_next = graph.step(ir_state, **stimulus)
        outputs, nxt = sim.step(state, **stimulus)
        assert outputs == ir_outputs and sim.state_to_ir(nxt) == ir_next


def _initialized_graph() -> Graph:
    """Registers with non-zero (and signed) inits, always or conditionally enabled."""
    graph = Graph()
    x = graph.input("in_x", UINT4)
    acc = graph.register(UINT4, init=0b1010)
    flag = graph.register(BOOL, init=1)
    signed = graph.register(INT8, init=-3)
    always = graph.constant(1, BOOL)
    graph.set_register(acc, graph.node("xor", UINT4, (acc, x)), always)
    graph.set_register(flag, graph.node("not", BOOL, (flag,)), always)
    graph.set_register(signed, graph.node("add", INT8, (signed, graph.node("cast", INT8, (x,)))), flag)
    graph.output("acc", acc)
    graph.output("flag", flag)
    graph.output("signed", signed)
    return graph


def test_rst_restores_every_register_init() -> None:
    graph = _initialized_graph()
    netlist = synth(graph)
    sim = PrimitiveSimulator(netlist)
    init = sim.reset_state()
    assert sim.state_to_ir(init) == graph.reset_state() == {1: 0b1010, 2: 1, 3: 0xFD}
    for node, value in graph.reset_state().items():
        for i, q in enumerate(netlist.buses[node].bits):
            assert init[q.instance] == (value >> i) & 1 == int(netlist.instance(q.instance).init)
    rng = random.Random(3)
    state = init
    for _ in range(7):
        _, state = sim.step(state, in_x=rng.getrandbits(4))
    assert state != init
    _, after = sim.step(state, rst=1, in_x=5)
    assert after == init
    for value in range(16):  # held in reset, whatever the inputs say
        _, after = sim.step(after, rst=True, in_x=value)
        assert after == init
    # Released, the netlist tracks the IR from reset again.
    ir_state = graph.reset_state()
    for _ in range(20):
        value = rng.getrandbits(4)
        ir_outputs, ir_state = graph.step(ir_state, in_x=value)
        outputs, after = sim.step(after, in_x=value)
        assert outputs == ir_outputs and sim.state_to_ir(after) == ir_state


@pytest.mark.parametrize(
    ("wiring", "stuck"),
    [
        pytest.param([(T(0, "y"), [T(1, "a")]), (T(1, "y"), [T(0, "a"), T(2, "a")])], [0, 1], id="two-inverters"),
        pytest.param([(T(0, "y"), [T(0, "a"), T(2, "a")])], [0], id="self-loop"),
    ],
)
def test_a_hand_built_combinational_loop_is_detected(wiring, stuck: list[int]) -> None:
    """The builder cannot wire a gate's operand after the gate exists, so a
    loop can only be assembled by hand; it is structurally well-formed (every
    pin driven once) but cannot be simulated."""
    netlist = fresh()
    for kind in (K.NOT, K.NOT, K.OUTPUT_BIT):
        add(netlist, kind)
    for driver, sinks in wiring:
        netlist.add_net(driver, sinks)
    if len(wiring) == 1:  # the spare inverter needs an input too
        x = add(netlist, K.INPUT_BIT)
        netlist.add_net(x.output(), [T(1, "a")])
    netlist.ports = [LogicalPort("result", "output", BOOL, None, 0, (T(2, "a"),))]
    netlist.validate(complete=True)
    with pytest.raises(CompileError, match=re.escape(f"simulation: combinational loop through primitives {stuck}")):
        PrimitiveSimulator(netlist)


def test_a_loop_through_a_register_bit_is_state_not_a_combinational_loop() -> None:
    """A toggle flip-flop, built with the builder: ``q <- NOT(q)``."""
    b = PrimitiveBuilder()
    q = b.register_bit(init=True, bit=0)
    d, clk, rst = b.register_pins(q)
    b.connect(b.not_(q, role="toggle"), d)
    b.connect(b.clock(), clk)
    b.connect(b.reset(), rst)
    sink = b.output_bit("q", 0, q)
    netlist = b.finish()
    netlist.ports = [LogicalPort("q", "output", BOOL, None, 0, (sink,))]
    netlist.buses = {0: LogicalBus(0, "register", BOOL, (q,))}
    netlist.validate(complete=True)
    sim = PrimitiveSimulator(netlist)
    assert sim.is_sequential and sim.input_names == [] and sim.output_names == ["q"]
    state = sim.reset_state()
    assert state == {q.instance: 1} and sim.state_to_ir(state) == {0: 1}
    seen = []
    for _ in range(6):
        outputs, state = sim.step(state)
        seen.append(outputs["q"])
    assert seen == [1, 0, 1, 0, 1, 0]
    for held in (0, 1):  # from either state, reset (not the toggle) decides
        _, state = sim.step(sim.state_from_ir({0: held}), rst=1)
        assert state == {q.instance: 1}
