"""IR Graph -> PhysicalNetlist technology mapping."""

import json
import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from redc import BOOL, CompileError, Graph, IRType, compile_source
from redc.cli import app
from redc.ir import OPS, SHIFT_AMOUNT
from redc.physical import (
    LIBRARY,
    PHYSICAL_TYPES,
    ClockSource,
    CompositeImplementation,
    Constant,
    InputPad,
    Library,
    Operation,
    OperationSignature,
    OutputPad,
    PadBoundaryPolicy,
    Peripheral,
    PhysicalNetlist,
    Register,
    ResetSource,
    TypeCast,
    lower_to_physical,
    simulate,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
UINT8, INT8, UINT16 = IRType(8), IRType(8, signed=True), IRType(16)
sig = OperationSignature.of
ADD_SOURCE = "uint8 add(uint8 a, uint8 b) { return a + b; }"
FIB = (EXAMPLES / "uint8_fib.redc").read_text()


def lower(source: str, top: str = "main", **kwargs) -> PhysicalNetlist:
    return lower_to_physical(compile_source(source, top=top), **kwargs)


def by_label(netlist: PhysicalNetlist, label: str):
    (inst,) = [i for i in netlist.instances.values() if i.label == label]
    return inst


def net_from(netlist: PhysicalNetlist, inst):
    nets = [n for n in netlist.nets.values() if n.driver.instance is inst]
    assert len(nets) <= 1  # one net per driver
    return nets[0] if nets else None


def driver_of(netlist: PhysicalNetlist, inst, pin: str):
    term = inst.terminal(pin)
    (net,) = [n for n in netlist.nets.values() if term in n.sinks]
    return net.driver


def of_type(netlist: PhysicalNetlist, cls) -> list:
    return [i for i in netlist.instances.values() if isinstance(i.component, cls)]


# -- A. simple arithmetic -----------------------------------------------------


def test_simple_add_with_abstract_pads() -> None:
    nl = lower(ADD_SOURCE, top="add", boundary=PadBoundaryPolicy())
    pads_in = of_type(nl, InputPad)
    pads_out = of_type(nl, OutputPad)
    adds = of_type(nl, Operation)
    assert sorted(p.label for p in pads_in) == ["in_a", "in_b"]
    assert [p.label for p in pads_out] == ["result"]
    assert len(adds) == 1 and adds[0].component.op == "add"
    assert len(nl.instances) == 4 and len(nl.nets) == 3
    add = adds[0]
    # Operand order: first IR arg -> pin a, second -> pin b.
    assert driver_of(nl, add, "a").instance.label == "in_a"
    assert driver_of(nl, add, "b").instance.label == "in_b"
    assert driver_of(nl, pads_out[0], "in").instance is add
    assert all(net.dtype == UINT8 and net.width == 8 for net in nl.nets.values())
    assert all(inst.origin is None for inst in nl.instances.values())
    assert not nl.is_sequential and not of_type(nl, ClockSource)
    nl.validate(complete=True)


def test_operand_order_is_semantic_for_non_commutative_ops() -> None:
    nl = lower("int8 main(int8 a, int8 b) { return a - b; }")
    (sub,) = of_type(nl, Operation)
    assert driver_of(nl, sub, "a").instance.label == "in_a"
    assert driver_of(nl, sub, "b").instance.label == "in_b"
    assert simulate.evaluate(nl, {"in_a": 10, "in_b": 3}) == {"result": 7}


# -- B. fanout ----------------------------------------------------------------


def test_fanout_is_one_net_with_many_sinks() -> None:
    nl = lower("uint8 main(uint8 a, uint8 b) { return (a + b) * a + (a ^ b); }")
    pad = by_label(nl, "in_a")
    nets = [n for n in nl.nets.values() if n.driver.instance is pad]
    assert len(nets) == 1
    assert nets[0].fanout == 3  # add.a, mul.b, xor.a
    # No driver terminal appears on two nets anywhere.
    drivers = [n.driver for n in nl.nets.values()]
    assert len(drivers) == len(set(drivers))


def test_one_producer_feeding_both_pins_of_one_cell() -> None:
    nl = lower("uint8 main(uint8 a) { return a * a; }")
    (mul,) = of_type(nl, Operation)
    net = net_from(nl, by_label(nl, "in_a"))
    assert set(net.sinks) == {mul.terminal("a"), mul.terminal("b")}


def test_value_feeding_two_outputs_is_one_net() -> None:
    graph = Graph()
    total = graph.op("add", UINT8, graph.input("a", UINT8), graph.input("b", UINT8))
    graph.output("sum", total)
    graph.output("copy", total)
    nl = lower_to_physical(graph)
    (add,) = of_type(nl, Operation)
    net = net_from(nl, add)
    assert {s.instance.label for s in net.sinks} == {"sum", "copy"}


# -- C. constants -------------------------------------------------------------


def test_constant_becomes_a_constant_cell() -> None:
    nl = lower("uint8 main(uint8 a) { return a + 5; }")
    (const,) = of_type(nl, Constant)
    assert const.component.value == 5
    assert const.component.outputs[0].dtype == UINT8
    (add,) = of_type(nl, Operation)
    assert driver_of(nl, add, "b").instance is const


# -- D. comparisons -----------------------------------------------------------


def test_comparison_uses_operand_typed_cell_with_bool_output() -> None:
    nl = lower("bool main(uint8 a, uint8 b) { return a < b; }")
    (lt,) = of_type(nl, Operation)
    assert lt.component.op == "lt"
    assert lt.component.signatures() == (sig("lt", BOOL, UINT8, UINT8),)
    assert net_from(nl, lt).dtype == BOOL
    (out,) = of_type(nl, OutputPad)  # bool result: no display peripheral
    assert driver_of(nl, out, "in").instance is lt


def test_signed_comparison_selects_the_signed_cell() -> None:
    unsigned = of_type(lower("bool main(uint8 a, uint8 b) { return a < b; }"), Operation)
    signed = of_type(lower("bool main(int8 a, int8 b) { return a < b; }"), Operation)
    assert unsigned[0].component.name != signed[0].component.name
    nl = lower("bool main(int8 a, int8 b) { return a < b; }")
    assert simulate.evaluate(nl, {"in_a": -1, "in_b": 1}) == {"result": 1}


# -- E. casts -----------------------------------------------------------------


@pytest.mark.parametrize(
    "src,dst",
    [("uint8", "uint16"), ("int8", "int16"), ("int8", "uint16"), ("uint16", "uint8")],
)
def test_cast_selects_full_type_cell(src: str, dst: str) -> None:
    nl = lower(f"{dst} main({src} a) {{ return ({dst}) a; }}")
    (cast,) = of_type(nl, TypeCast)
    assert cast.component.source_type.name == src
    assert cast.component.result_type.name == dst


def test_bool_conversion_cast() -> None:
    nl = lower("bool main(uint8 a) { return (bool) a; }")
    (cast,) = of_type(nl, TypeCast)
    assert cast.component.signatures() == (sig("cast", BOOL, UINT8),)


# -- F. sequential --------------------------------------------------------------


def test_sequential_fib_registers_clock_and_reset() -> None:
    graph = compile_source(FIB, top="fib")
    nl = lower_to_physical(graph)
    ir_registers = [n for n in graph.live_nodes() if n["op"] == "register"]
    regs = of_type(nl, Register)
    assert len(regs) == len(ir_registers) == 4
    assert {r.label for r in regs} == {f"n{n['id']}" for n in ir_registers}

    (clk,) = of_type(nl, ClockSource)
    (rst,) = of_type(nl, ResetSource)
    clock_net = net_from(nl, clk)
    reset_net = net_from(nl, rst)
    assert set(clock_net.sinks) == {r.terminal("clk") for r in regs}
    assert set(reset_net.sinks) == {r.terminal("rst") for r in regs}

    for node in ir_registers:
        reg = by_label(nl, f"n{node['id']}")
        assert reg.init == node["init"]
        next_id, enable_id = node["args"]
        for pin, arg in (("next", next_id), ("enable", enable_id)):
            label = driver_of(nl, reg, pin).instance.label
            ir = graph.nodes[arg]
            assert label == (ir["name"] if ir["op"] == "input" else f"n{arg}")
        assert net_from(nl, reg) is not None  # out drives its consumers

    nl.validate(complete=True)
    assert all(inst.origin is None for inst in nl.instances.values())


def test_combinational_graph_has_no_clock_or_reset() -> None:
    nl = lower(ADD_SOURCE, top="add")
    assert not of_type(nl, ClockSource) and not of_type(nl, ResetSource)


# -- G. unsupported -------------------------------------------------------------


def test_unsupported_width_is_a_useful_error() -> None:
    with pytest.raises(CompileError, match=r"input port 'in_a'.*uint3"):
        lower("uint3 main(uint3 a) { return a; }")
    with pytest.raises(CompileError, match=r"IR node \d+ .*uint12"):
        lower("uint8 main(uint8 a) { uint12 w = (uint12) a; return (uint8) (w + 1); }")


def test_example_with_uint3_is_rejected() -> None:
    with pytest.raises(CompileError, match="uint3"):
        lower((EXAMPLES / "uint8_alu.redc").read_text())


def test_missing_cell_names_the_operation_and_types() -> None:
    no_mul = Library({n: c for n, c in LIBRARY.items() if getattr(c, "op", None) != "mul"})
    with pytest.raises(
        CompileError,
        match=r"no physical cell variant for IR node \d+: mul\(uint8, uint8\) -> uint8",
    ):
        lower("uint8 main(uint8 a, uint8 b) { return a * b; }", library=no_mul)


def test_composite_only_operation_is_reported() -> None:
    lib = Library({n: c for n, c in LIBRARY.items() if getattr(c, "op", None) != "mul"})
    mul = sig("mul", UINT8, UINT8, UINT8)
    lib.implementations.register(CompositeImplementation("mul_rewrite", mul, lambda b, ops: ops[0]))
    with pytest.raises(CompileError, match="only composite"):
        lower("uint8 main(uint8 a, uint8 b) { return a * b; }", library=lib)


def test_variant_selector_is_pluggable() -> None:
    chosen = []

    def last(node, signature, candidates):
        chosen.append((signature, len(candidates)))
        return candidates[-1]

    nl = lower(ADD_SOURCE, top="add", select_variant=last)
    (add,) = of_type(nl, Operation)
    assert add.component.name == "uint8_add_a-0-0-0_b-0-1-0_out-0-2-0"
    assert chosen == [(sig("add", UINT8, UINT8, UINT8), 2)]


def test_bad_boundary_policy_is_rejected() -> None:
    class Wrong:
        def realize_input(self, name, typ):
            return InputPad.of(name, UINT16)  # wrong type

        def realize_output(self, name, typ):
            return OutputPad.of(name, typ)

    with pytest.raises(CompileError, match="boundary policy"):
        lower(ADD_SOURCE, top="add", boundary=Wrong())


# -- H/I. library loads and covers every IR operation -------------------------


def test_every_library_stub_loads_with_valid_geometry() -> None:
    assert len(LIBRARY) > 300
    for cell in LIBRARY.values():
        assert all(d > 0 for d in cell.dim)
        assert cell.nbt is None
        assert cell.latency is None or cell.latency >= 0


def _ir_signatures(typ: IRType) -> list[OperationSignature]:
    """Every signature the IR typing rules allow for value type ``typ``."""
    out = []
    for op in sorted(OPS - {"cast", "not"}):
        if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
            out.append(sig(op, BOOL, typ, typ))
        elif op in {"shl", "shr"}:
            out.append(sig(op, typ, typ, SHIFT_AMOUNT))
        elif op == "mux":
            out.append(sig(op, typ, BOOL, typ, typ))
        elif op in {"neg", "inv"}:
            out.append(sig(op, typ, typ))
        else:
            out.append(sig(op, typ, typ, typ))
    out.append(sig("register", typ, typ, BOOL))
    out.extend(sig("cast", dst, typ) for dst in PHYSICAL_TYPES if dst != typ)
    if typ == BOOL:
        out.append(sig("not", BOOL, BOOL))
    return out


@pytest.mark.parametrize("typ", PHYSICAL_TYPES, ids=lambda t: t.name)
def test_every_ir_operation_has_a_cell_for_every_supported_type(typ: IRType) -> None:
    missing = [str(s) for s in _ir_signatures(typ) if not LIBRARY.cells(s)]
    assert not missing


# -- determinism + functional equivalence -------------------------------------


def test_lowering_is_deterministic() -> None:
    graph = compile_source(FIB, top="fib")
    assert lower_to_physical(graph).to_dict() == lower_to_physical(graph).to_dict()


EXAMPLE_TOPS = {
    "array_mux.redc": "main",
    "dot_product.redc": "main",
    "popcount.redc": "main",
    "priority_encoder.redc": "main",
    "uint8_add.redc": "add",
}


@pytest.mark.parametrize("example", sorted(EXAMPLE_TOPS))
def test_mapped_netlist_matches_ir_evaluation(example: str) -> None:
    graph = compile_source((EXAMPLES / example).read_text(), top=EXAMPLE_TOPS[example])
    nl = lower_to_physical(graph)
    nl.validate(complete=True)
    rng = random.Random(example)
    for _ in range(50):
        stimulus = {
            port["name"]: rng.getrandbits(IRType(**port["type"]).width)
            for port in graph.inputs
        }
        assert simulate.evaluate(nl, stimulus) == graph.evaluate(**stimulus)


def test_mapped_fib_matches_ir_cycle_by_cycle() -> None:
    graph = compile_source(FIB, top="fib")
    nl = lower_to_physical(graph)
    ir_state, nl_state = graph.reset_state(), simulate.reset_state(nl)
    starts = [1] + [0] * 15 + [1, 1] + [0] * 6 + [1] + [0] * 12
    for cycle, start in enumerate(starts):
        n = 9 if cycle < 16 else 5
        ir_out, ir_state = graph.step(ir_state, in_n=n, start=start)
        nl_out, nl_state = simulate.step(nl, nl_state, {"in_n": n, "start": start})
        assert nl_out == ir_out, cycle


def test_netlist_simulation_honours_reset() -> None:
    nl = lower(FIB, top="fib")
    state = simulate.reset_state(nl)
    for start in (1, 0, 0):
        _, state = simulate.step(nl, state, {"in_n": 9, "start": start})
    _, state = simulate.step(nl, state, {"in_n": 9, "start": 0}, rst=1)
    assert state == simulate.reset_state(nl)


# -- CLI ------------------------------------------------------------------------


def test_dump_netlist_cli() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["dump-netlist", str(EXAMPLES / "uint8_fib.redc"), "--top", "fib"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["sequential"] is True
    assert any(i.get("peripheral") == "lever" for i in payload["instances"])
    bad = runner.invoke(app, ["dump-netlist", str(EXAMPLES / "uint8_alu.redc")])
    assert bad.exit_code == 1


def test_peripheral_is_not_an_implementation_candidate() -> None:
    for signature in LIBRARY.implementations.signatures():
        assert not any(isinstance(c, Peripheral) for c in LIBRARY.cells(signature))


def test_variant_selector_must_return_a_matching_cell() -> None:
    wrong = LIBRARY["uint8_sub_a-0-0-0_b-0-0-1_out-0-0-2"]
    with pytest.raises(CompileError, match="does not implement"):
        lower(ADD_SOURCE, top="add", select_variant=lambda node, s, c: wrong)
