"""Primitive synthesis: registry, structure and provenance, special nodes,
sequential behaviour and whole programs.

Op-by-op functional equivalence lives in ``test_primitive_synthesis_ops.py``.
This file checks that the backend really EXPLODES every operation into one-bit
basis gates (AND / OR / XOR / NOT) with useful provenance, that the recipe
registry covers every IR op, that inputs / constants / registers / outputs
lower correctly, and that real programs (``examples/*.redc``) simulate exactly
like the IR -- combinationally on many inputs and cycle by cycle from reset.

Some required structure is invisible to every functional test because the
forbidden alternative computes the same function -- an arithmetic shift
filling from a shifted intermediate's top bit instead of the ORIGINAL sign
bit, a signedness-specific multiplier -- so it is pinned structurally:
pin-by-pin barrel wiring, and identical networks for signed and unsigned
typings of every signedness-agnostic op.

Graphs are built with ``Graph.node`` (``Graph.op`` would fold constants and
simplify) and synthesized with pad ports unless a test says otherwise.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from redc import compile_source
from redc.ir import BOOL, OPS, SHIFT_AMOUNT, Graph, IRType
from redc.parser import CompileError
from redc.physical_primitive import (
    GATE_KINDS,
    BitTerminal,
    DefaultInterfacePolicy,
    HierarchyGroup,
    PadInterfacePolicy,
    PrimitiveInstance,
    PrimitiveKind,
    PrimitiveNetlist,
    PrimitiveSimulator,
    synthesize_to_primitives,
)
from redc.physical_primitive.synthesis.logic import bitwise
from redc.physical_primitive.synthesis.recipes import (
    DEFAULT_SYNTHESIS,
    build_default_registry,
)
from redc.signature import OperationSignature
from redc.tracing import TraceLevel

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_TOPS = {"uint8_add": "add", "uint8_fib": "fib"}

UINT4 = IRType(4)
UINT8 = IRType(8)
INT8 = IRType(8, signed=True)
UINT13 = IRType(13)
INT16 = IRType(16, signed=True)
UINT32 = IRType(32)
INT64 = IRType(64, signed=True)

#: Kinds a combinational netlist may contain besides the basis gates.
BOUNDARY_KINDS = frozenset(
    {PrimitiveKind.INPUT_BIT, PrimitiveKind.OUTPUT_BIT, PrimitiveKind.CONST0, PrimitiveKind.CONST1}
)
#: ... and a sequential one.
STATE_KINDS = frozenset(
    {PrimitiveKind.REGISTER_BIT, PrimitiveKind.CLOCK_SOURCE, PrimitiveKind.RESET_SOURCE}
)
FULL_ADDER_ROLES = {"propagate_xor", "sum_xor", "carry_generate", "carry_propagate_and", "carry_or"}


def synth(graph: Graph, **kwargs) -> PrimitiveNetlist:
    kwargs.setdefault("interface", PadInterfacePolicy())
    return synthesize_to_primitives(graph, **kwargs)


def single(op: str, result: IRType, *operands: IRType) -> tuple[Graph, PrimitiveNetlist, int]:
    """``result = op(in_a, in_b, ...)``: the graph, its netlist and the op's node id."""
    graph = Graph()
    args = tuple(graph.input(f"in_{chr(ord('a') + k)}", typ) for k, typ in enumerate(operands))
    node = graph.node(op, result, args)
    graph.output("result", node)
    return graph, synth(graph), node.id


def operands_for(op: str, typ: IRType) -> tuple[IRType, tuple[IRType, ...]]:
    """``(result type, operand types)`` of ``op`` at value type ``typ``."""
    if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        return BOOL, (typ, typ)
    if op in {"shl", "shr"}:
        return typ, (typ, SHIFT_AMOUNT)
    if op == "mux":
        return typ, (BOOL, typ, typ)
    if op == "cast":
        return (INT16 if typ != INT16 else UINT8), (typ,)
    if op == "not":
        return BOOL, (BOOL,)
    if op in {"neg", "inv"}:
        return typ, (typ,)
    return typ, (typ, typ)


def gates(netlist: PrimitiveNetlist) -> list[PrimitiveInstance]:
    return [inst for inst in netlist.instances if inst.is_gate]


def role_counts(instances: list[PrimitiveInstance]) -> Counter[str]:
    return Counter(inst.provenance.role for inst in instances)


def groups(netlist: PrimitiveNetlist, *, kind: str | None = None, name: str | None = None) -> list[HierarchyGroup]:
    return [
        g for g in netlist.groups if (kind is None or g.kind == kind) and (name is None or g.name == name)
    ]


def ancestors(netlist: PrimitiveNetlist, group_id: int | None) -> list[HierarchyGroup]:
    """``group_id`` and every enclosing group, innermost first."""
    chain = []
    while group_id is not None:
        chain.append(netlist.groups[group_id])
        group_id = netlist.groups[group_id].parent
    return chain


def slices_under(netlist: PrimitiveNetlist, group_id: int) -> list[HierarchyGroup]:
    """Every bit-slice group nested (at any depth) below ``group_id``."""
    below = netlist.descendants(group_id) - {group_id}
    return [g for g in netlist.groups if g.id in below and g.kind == "slice"]


def input_driver(netlist: PrimitiveNetlist, inst: PrimitiveInstance, pin: str) -> BitTerminal | None:
    return netlist.driver_of(BitTerminal(inst.id, pin))


def pins(netlist: PrimitiveNetlist, inst: PrimitiveInstance) -> list[BitTerminal | None]:
    """The signals driving ``inst``'s input pins, in pin order (``a``, ``b``)."""
    return [input_driver(netlist, inst, pin) for pin in inst.inputs]


def by_role_and_bit(netlist: PrimitiveNetlist, group_id: int) -> dict[tuple[str, int | None], PrimitiveInstance]:
    """Every primitive at or below ``group_id``, keyed by its (unique) ``(role, bit)``."""
    found: dict[tuple[str, int | None], PrimitiveInstance] = {}
    for inst in netlist.instances_in_group(group_id):
        key = (inst.provenance.role, inst.provenance.bit)
        assert key not in found, f"two primitives share role and bit {key}"
        found[key] = inst
    return found


def assert_clean(
    netlist: PrimitiveNetlist,
    *,
    sequential: bool = False,
    peripherals: bool = False,
    every_bit_observed: bool = True,
) -> None:
    """The structural contract of every synthesized netlist.

    ``every_bit_observed`` additionally demands that no gate is dangling (each
    gate output feeds one net).  That holds whenever every result bit of every
    live node is read -- single-op graphs -- but not for programs that drop
    bits of a live value (a truncating cast of an AND leaves the AND's high
    bit gates unread: liveness is per IR node, not per bit)."""
    netlist.validate(complete=True)
    allowed = GATE_KINDS | BOUNDARY_KINDS
    if sequential:
        allowed |= STATE_KINDS
    if peripherals:
        allowed |= {PrimitiveKind.PERIPHERAL}
    assert {inst.kind for inst in netlist.instances} <= allowed
    assert all(net.width == 1 for net in netlist.nets)
    for inst in netlist.instances:
        if inst.is_gate:
            assert inst.provenance.role
            assert inst.provenance.ir_node in netlist.ir_nodes
            if every_bit_observed:
                assert netlist.net_of_driver(inst.output()) is not None, f"dangling gate {inst}"


# -- the registry --------------------------------------------------------------


def test_default_registry_covers_every_ir_op() -> None:
    assert DEFAULT_SYNTHESIS.ops() == set(OPS)


REPRESENTATIVE_SIGNATURES = [
    (OperationSignature.of("and", UINT8, UINT8, UINT8), "bitwise_and"),
    (OperationSignature.of("or", INT8, INT8, INT8), "bitwise_or"),
    (OperationSignature.of("xor", BOOL, BOOL, BOOL), "bitwise_xor"),
    (OperationSignature.of("inv", INT64, INT64), "bitwise_inv"),
    (OperationSignature.of("not", BOOL, BOOL), "logical_not"),
    (OperationSignature.of("mux", INT8, BOOL, INT8, INT8), "bitwise_mux"),
    (OperationSignature.of("cast", BOOL, UINT8), "cast_to_bool"),
    (OperationSignature.of("cast", UINT8, INT8), "cast_reinterpret"),
    (OperationSignature.of("cast", IRType(1), BOOL), "cast_reinterpret"),
    (OperationSignature.of("cast", UINT8, INT16), "cast_truncate"),
    (OperationSignature.of("cast", INT16, INT8), "cast_sign_extend"),
    (OperationSignature.of("cast", INT16, UINT8), "cast_zero_extend"),
    (OperationSignature.of("cast", INT8, BOOL), "cast_zero_extend"),
    (OperationSignature.of("neg", INT8, INT8), "twos_complement_negate"),
    (OperationSignature.of("add", UINT8, UINT8, UINT8), "ripple_carry_add"),
    (OperationSignature.of("sub", INT8, INT8, INT8), "ripple_carry_subtract"),
    (OperationSignature.of("mul", UINT8, UINT8, UINT8), "array_multiply"),
    (OperationSignature.of("mul", INT8, INT8, INT8), "array_multiply"),  # one network, both signednesses
    (OperationSignature.of("mul", INT64, INT64, INT64), "array_multiply"),
    (OperationSignature.of("div", UINT8, UINT8, UINT8), "restoring_divide_unsigned"),
    (OperationSignature.of("div", INT8, INT8, INT8), "restoring_divide_signed"),
    (OperationSignature.of("mod", UINT32, UINT32, UINT32), "restoring_modulo_unsigned"),
    (OperationSignature.of("mod", INT8, INT8, INT8), "restoring_modulo_signed"),
    (OperationSignature.of("shl", UINT8, UINT8, SHIFT_AMOUNT), "barrel_shift_left"),
    (OperationSignature.of("shr", UINT8, UINT8, SHIFT_AMOUNT), "barrel_shift_right_logical"),
    (OperationSignature.of("shr", INT8, INT8, SHIFT_AMOUNT), "barrel_shift_right_arithmetic"),
    (OperationSignature.of("eq", BOOL, INT8, INT8), "equal_xor_or_tree"),
    (OperationSignature.of("ne", BOOL, UINT8, UINT8), "not_equal_xor_or_tree"),
    (OperationSignature.of("lt", BOOL, UINT8, UINT8), "less_chain_lt_unsigned"),
    (OperationSignature.of("lt", BOOL, INT8, INT8), "less_chain_lt_signed"),
    (OperationSignature.of("le", BOOL, BOOL, BOOL), "less_chain_le_unsigned"),
    (OperationSignature.of("gt", BOOL, INT64, INT64), "less_chain_gt_signed"),
    (OperationSignature.of("ge", BOOL, IRType(1, signed=True), IRType(1, signed=True)), "less_chain_ge_signed"),
]


@pytest.mark.parametrize(
    ("signature", "recipe"), REPRESENTATIVE_SIGNATURES, ids=[str(s) for s, _ in REPRESENTATIVE_SIGNATURES]
)
def test_representative_signatures_resolve_to_the_expected_recipe(
    signature: OperationSignature, recipe: str
) -> None:
    assert DEFAULT_SYNTHESIS.resolve(signature).name == recipe


def test_every_op_resolves_for_every_typing_the_ir_allows() -> None:
    types = [BOOL, IRType(1), IRType(1, signed=True), UINT8, INT8, UINT13, INT64]
    for op in sorted(OPS):
        for typ in types:
            if op == "not" and typ != BOOL:
                continue
            result, operands = operands_for(op, typ)
            assert DEFAULT_SYNTHESIS.supports(OperationSignature(op, result, operands)), (op, typ)


def test_an_unknown_op_has_no_recipe() -> None:
    signature = OperationSignature.of("frobnicate", UINT8, UINT8)
    with pytest.raises(CompileError, match="no primitive synthesis recipe"):
        DEFAULT_SYNTHESIS.resolve(signature)
    assert not DEFAULT_SYNTHESIS.supports(signature)
    # Registers are lowered by the driver itself (state bits + enable mux), never by a recipe.
    assert not DEFAULT_SYNTHESIS.supports(OperationSignature.of("register", UINT8, UINT8, BOOL))


def test_an_exact_registration_wins_over_the_family_recipe() -> None:
    registry = build_default_registry()
    exact = OperationSignature.of("add", UINT8, UINT8, UINT8)

    def xor_only(b, sig, ops):  # deliberately NOT an adder: shows which recipe ran
        return bitwise(b, b.xor, ops[0], ops[1], role="exact_xor")

    registry.register(exact, xor_only, name="exact_add_uint8")
    assert registry.resolve(exact).name == "exact_add_uint8"
    assert registry.resolve(OperationSignature.of("add", INT8, INT8, INT8)).name == "ripple_carry_add"
    with pytest.raises(CompileError, match="already registered"):
        registry.register(exact, xor_only, name="again")

    graph = Graph()
    a, b = graph.input("in_a", UINT8), graph.input("in_b", UINT8)
    graph.output("result", graph.node("add", UINT8, (a, b)))
    netlist = synth(graph, registry=registry)
    assert role_counts(gates(netlist)) == {"exact_xor": 8}
    assert PrimitiveSimulator(netlist).evaluate(in_a=0b1100, in_b=0b1010)["result"] == 0b0110
    # The shared default registry is untouched.
    assert DEFAULT_SYNTHESIS.resolve(exact).name == "ripple_carry_add"


# -- structure: operations really explode into one-bit gates -----------------------


def test_uint8_and_is_eight_and_gates_driving_eight_independent_nets() -> None:
    _, netlist, node = single("and", UINT8, UINT8, UINT8)
    assert_clean(netlist)
    assert [g.kind for g in gates(netlist)] == [PrimitiveKind.AND] * 8
    assert netlist.kind_counts()["and"] == netlist.gate_count == 8
    a_bits, b_bits = netlist.port("in_a").bits, netlist.port("in_b").bits
    result = netlist.buses[node].bits
    assert len(set(result)) == 8
    nets = [netlist.net_of_driver(bit) for bit in result]
    assert len({net.id for net in nets if net is not None}) == 8
    for i, (bit, net) in enumerate(zip(result, nets)):
        gate = netlist.instance(bit.instance)
        assert gate.kind is PrimitiveKind.AND and gate.provenance.bit == i
        assert input_driver(netlist, gate, "a") == a_bits[i]
        assert input_driver(netlist, gate, "b") == b_bits[i]
        assert net is not None and net.width == 1 and net.fanout == 1
        assert net.sinks == (netlist.port("result").bits[i],)


def test_uint8_input_bus_is_eight_input_bits_on_eight_distinct_nets() -> None:
    graph = Graph()
    a = graph.input("in_a", UINT8)
    graph.output("result", graph.node("inv", UINT8, (a,)))
    netlist = synth(graph)
    assert_clean(netlist)
    port = netlist.port("in_a")
    assert port.direction == "input" and port.realization == "pads" and len(port.instances) == 8
    assert netlist.kind_counts()["input_bit"] == 8
    for i, bit in enumerate(port.bits):
        inst = netlist.instance(bit.instance)
        assert inst.kind is PrimitiveKind.INPUT_BIT
        assert (inst.provenance.port, inst.provenance.bit, inst.provenance.ir_node) == ("in_a", i, a.id)
    nets = [netlist.net_of_driver(bit) for bit in port.bits]
    assert len({net.id for net in nets if net is not None}) == 8
    assert all(net is not None and net.width == 1 for net in nets)


def test_uint8_add_is_a_ripple_carry_adder_of_basis_gates_only() -> None:
    _, netlist, _ = single("add", UINT8, UINT8, UINT8)
    assert_clean(netlist)
    assert {inst.kind for inst in netlist.instances} <= GATE_KINDS | BOUNDARY_KINDS
    assert netlist.summary()["gates_by_kind"] == {"and": 13, "not": 0, "or": 6, "xor": 15}
    # Half adder at bit 0, full adders above, no carry out of bit 7.
    assert role_counts(gates(netlist)) == {
        "sum_xor": 8,
        "propagate_xor": 7,
        "carry_generate": 7,
        "carry_propagate_and": 6,
        "carry_or": 6,
    }
    # Ripple, not lookahead: each slice's carry-out (the half adder's AND at
    # bit 0, a full adder's OR above it) feeds only the next slice.
    carries = [i for i in gates(netlist) if i.provenance.role == "carry_or"]
    carries += [i for i in gates(netlist) if i.provenance.role == "carry_generate" and i.provenance.bit == 0]
    assert sorted(i.provenance.bit for i in carries) == list(range(7))
    for inst in carries:
        net = netlist.net_of_driver(inst.output())
        assert net is not None
        readers = {netlist.instance(s.instance).provenance.bit for s in net.sinks}
        assert readers == {inst.provenance.bit + 1}


def test_adder_bit_slices_are_found_through_the_group_hierarchy() -> None:
    _, netlist, node = single("add", UINT8, UINT8, UINT8)
    slices = {g.name: g for g in groups(netlist, kind="slice")}
    assert set(slices) == {f"bit{i}" for i in range(8)}
    (adder,) = groups(netlist, name="ripple_adder")
    assert len(netlist.instances_in_group(adder.id)) == netlist.gate_count
    for i in range(8):
        group = slices[f"bit{i}"]
        assert netlist.group_path(group.id) == ("design", f"n{node}.add", "ripple_adder", f"bit{i}")
        assert dict(group.attrs)["bit"] == i
        members = netlist.instances_in_group(group.id)
        assert all(m.provenance.bit == i and m.provenance.ir_node == node for m in members)
        expected = {0: {"sum_xor", "carry_generate"}, 7: {"propagate_xor", "sum_xor"}}.get(i, FULL_ADDER_ROLES)
        assert sorted(role_counts(members)) == sorted(expected) and len(members) == len(expected)


def test_subtract_and_negate_reuse_the_adder_machinery() -> None:
    _, sub, _ = single("sub", UINT8, UINT8, UINT8)
    sub_roles = role_counts(gates(sub))
    # x + NOT(y) + 1: eight inverters, then FULL adders everywhere (carry-in = constant 1).
    assert sub_roles["subtrahend_not"] == 8 and sub_roles["propagate_xor"] == 8
    assert sub.kind_counts()["const1"] == 1 and sub.gate_count == 45
    (const1,) = [i for i in sub.instances if i.kind is PrimitiveKind.CONST1]
    net = sub.net_of_driver(const1.output())
    assert net is not None and {sub.instance(s.instance).provenance.bit for s in net.sinks} == {0}
    _, neg, _ = single("neg", INT8, INT8)
    # NOT(x) + 1 through the half-adder chain: no constant-zero operand bits.
    assert role_counts(gates(neg)) == {"negate_not": 8, "increment_sum_xor": 8, "increment_carry_and": 7}
    assert neg.kind_counts()["const0"] == 0


def test_multiplier_is_partial_products_plus_one_ripple_adder_per_row() -> None:
    _, netlist, _ = single("mul", UINT8, UINT8, UINT8)
    assert_clean(netlist)
    roles = role_counts(gates(netlist))
    assert roles["partial_product"] == 36  # AND(a[i], b[j]) for i + j < 8 only
    assert roles["sum_xor"] and roles["carry_or"]
    assert {i.kind for i in gates(netlist) if i.provenance.role == "partial_product"} == {PrimitiveKind.AND}
    kinds = netlist.kind_counts()
    assert kinds["xor"] and kinds["or"]
    assert kinds["const0"] == kinds["const1"] == 0  # never feeds constant-zero bits into an adder
    rows = sorted(groups(netlist, kind="iteration"), key=lambda g: dict(g.attrs)["row"])
    assert [g.name for g in rows] == [f"row{j}" for j in range(8)]
    for j, row in enumerate(rows):
        members = netlist.instances_in_group(row.id)
        columns = sorted(m.provenance.bit for m in members if m.provenance.role == "partial_product")
        assert columns == list(range(j, 8))
        adders = [g for g in netlist.groups if g.parent == row.id and g.name == "ripple_adder"]
        if j == 0:
            assert not adders  # row 0 IS the initial accumulator
        else:
            (adder,) = adders
            assert dict(adder.attrs) == {"lsb": j, "width": 8 - j}  # occupied columns j..7 only


#: IR ops whose result bits are ONE function of the operand bit patterns
#: whatever their signedness: ``add`` / ``sub`` / ``mul`` / ``neg`` wrap modulo
#: ``2**W``, bitwise logic and ``mux`` never look at a sign, equality compares
#: patterns and ``shl`` moves raw bits.
SIGNEDNESS_AGNOSTIC_OPS = ("add", "sub", "mul", "neg", "inv", "and", "or", "xor", "mux", "eq", "ne", "shl")


def network(netlist: PrimitiveNetlist) -> tuple[list[tuple], ...]:
    """Everything about a netlist except its type labels: each primitive's kind,
    init and provenance (its group as a path), every net, every group (minus
    its ``type`` attr) and the bits of every bus and port."""
    return (
        [
            (
                inst.kind,
                inst.init,
                inst.provenance.ir_node,
                inst.provenance.role,
                inst.provenance.bit,
                inst.provenance.port,
                inst.provenance.attrs,
                netlist.group_path(inst.provenance.group),
            )
            for inst in netlist.instances
        ],
        [(net.driver, net.sinks, net.role) for net in netlist.nets],
        [(g.name, g.kind, g.parent, g.ir_node, tuple(a for a in g.attrs if a[0] != "type")) for g in netlist.groups],
        [(node, bus.op, bus.bits) for node, bus in sorted(netlist.buses.items())],
        [(port.name, port.direction, port.realization, port.bits) for port in netlist.ports],
    )


@pytest.mark.parametrize("width", [1, 2, 5, 8, 13])
@pytest.mark.parametrize("op", SIGNEDNESS_AGNOSTIC_OPS)
def test_signedness_agnostic_ops_build_one_network_for_signed_and_unsigned(op: str, width: int) -> None:
    """``mul(intW)`` and ``mul(uintW)`` -- like every op whose result is one
    function of the operand bit patterns -- resolve to ONE recipe and build the
    identical primitive network; only the type labels differ.

    For ``mul`` this is the array multiplier's documented argument: a W-bit
    two's-complement value ``v`` and its unsigned pattern ``u`` satisfy
    ``v == u (mod 2**W)``, so ``v1 * v2 == u1 * u2 (mod 2**W)``.  A
    signedness-specific network (sign-magnitude, Baugh-Wooley, an exact
    ``mul(int8)`` registration, ...) computes the very same function, so only
    a structural comparison can catch one."""
    built = []
    for signed in (False, True):
        result, operands = operands_for(op, IRType(width, signed=signed))
        entry = DEFAULT_SYNTHESIS.resolve(OperationSignature(op, result, operands))
        _, netlist, node = single(op, result, *operands)
        built.append((entry, netlist, node))
    (u_entry, u_netlist, u_node), (s_entry, s_netlist, s_node) = built
    assert s_entry is u_entry and s_node == u_node
    assert [p.type for p in s_netlist.ports] != [p.type for p in u_netlist.ports]  # really two typings
    assert network(s_netlist) == network(u_netlist)


@pytest.mark.parametrize("op", ["div", "mod"])
@pytest.mark.parametrize("typ", [UINT8, INT8], ids=lambda t: t.name)
def test_divider_is_an_unrolled_restoring_array(op: str, typ: IRType) -> None:
    _, netlist, node = single(op, typ, typ, typ)
    assert_clean(netlist)
    iterations = sorted(groups(netlist, kind="iteration"), key=lambda g: dict(g.attrs)["iteration"])
    assert [(g.name, dict(g.attrs)) for g in iterations] == [
        (f"iter{k}", {"iteration": k, "quotient_bit": 7 - k}) for k in range(8)
    ]
    for k, group in enumerate(iterations):
        members = netlist.instances_in_group(group.id)
        assert all(m.provenance.ir_node == node for m in members)
        member_roles = set(role_counts(members))
        assert any(r.startswith("restoring_compare") for r in member_roles)
        # div needs no remainder: its last row builds only the comparison.
        complete = not (op == "div" and k == 7)
        assert any(r.startswith("restoring_subtract") for r in member_roles) == complete
        assert any(r.startswith("restoring_select") for r in member_roles) == complete
        # The compare spans the W + 1 = 9-bit shifted remainder.
        (compare,) = [g for g in netlist.groups if g.parent == group.id and g.name == "restoring_compare"]
        assert sorted(dict(g.attrs)["bit"] for g in slices_under(netlist, compare.id)) == list(range(9))
    assert role_counts(gates(netlist))["restoring_compare_ge"] == 8
    names = {g.name for g in netlist.groups}
    assert "divide_by_zero_guard" in names
    if typ.signed:  # magnitudes in, the result sign applied afterwards
        assert {"dividend_abs", "divisor_abs", "quotient_sign" if op == "div" else "remainder_sign"} <= names
    else:
        assert not {"dividend_abs", "divisor_abs", "quotient_sign", "remainder_sign"} & names


@pytest.mark.parametrize(
    ("op", "typ", "stages"),
    [("shl", UINT8, 3), ("shr", UINT8, 3), ("shr", INT8, 3), ("shl", UINT13, 4), ("shr", INT64, 6), ("shl", BOOL, 0)],
    ids=lambda v: v.name if isinstance(v, IRType) else str(v),
)
def test_variable_shift_is_a_staged_mux_barrel_plus_a_64_bit_range_check(
    op: str, typ: IRType, stages: int
) -> None:
    _, netlist, _ = single(op, typ, typ, SHIFT_AMOUNT)
    assert_clean(netlist)
    width = typ.width
    amount = netlist.port("in_b").bits
    found = sorted(groups(netlist, kind="stage"), key=lambda g: dict(g.attrs)["stage"])
    assert [(g.name, dict(g.attrs)) for g in found] == [
        (f"stage{k}", {"stage": k, "distance": 2**k}) for k in range(stages)
    ]
    for k, stage in enumerate(found):
        members = netlist.instances_in_group(stage.id)
        # One shared NOT(amount[k]) and a 2:1 mux (2 AND + 1 OR) per bit.
        assert Counter(m.kind for m in members) == {
            PrimitiveKind.NOT: 1,
            PrimitiveKind.AND: 2 * width,
            PrimitiveKind.OR: width,
        }
        (nsel,) = [m for m in members if m.kind is PrimitiveKind.NOT]
        assert input_driver(netlist, nsel, "a") == amount[k]
        for m in members:
            if m.provenance.role == "shift_stage_yes":
                assert input_driver(netlist, m, "a") == amount[k]
            if m.provenance.role == "shift_stage_no":
                assert input_driver(netlist, m, "a") == nsel.output()
    # too_large is an exact comparison of all 64 amount bits against the constant W.
    (check,) = groups(netlist, name="range_check")
    assert dict(check.attrs) == {"limit": width, "width": 64}
    assert len(slices_under(netlist, check.id)) == 64


#: ``(op, value type)``: left / logical / arithmetic, power-of-two and odd
#: widths, one-bit values (no stage at all) and 64-bit values (six stages).
SHIFTER_WIRING = [
    ("shl", BOOL),
    ("shr", BOOL),
    ("shr", IRType(1, signed=True)),
    ("shr", IRType(2, signed=True)),
    ("shr", IRType(5, signed=True)),
    ("shl", UINT8),
    ("shl", INT8),
    ("shr", UINT8),
    ("shr", INT8),
    ("shl", UINT13),
    ("shr", UINT13),
    ("shr", IRType(13, signed=True)),
    ("shl", INT64),
    ("shr", IRType(64)),
    ("shr", INT64),
]


@pytest.mark.parametrize(("op", "typ"), SHIFTER_WIRING, ids=[f"{op}-{t.name}" for op, t in SHIFTER_WIRING])
def test_barrel_stages_move_the_previous_stage_and_fill_from_the_original_operand(op: str, typ: IRType) -> None:
    """The barrel and its saturation, wired pin by pin.

    Stage ``k`` is ``mux(amount[k], moved, current)``: ``current`` is the
    previous stage's output (the operand itself for stage 0) and ``moved`` is
    ``current`` shifted by ``2**k`` with THE fill entering every vacated
    position -- the node's constant 0 for ``shl`` and logical ``shr``, the
    ORIGINAL operand's sign bit ``in_a[W-1]`` for arithmetic ``shr``.  The
    arithmetic saturation (``amount >= W``) selects copies of that same
    original bit.  Taking the sign from a shifted intermediate instead
    (``current[W-1]``, ``barrel[W-1]``) computes the very same function --
    every stage keeps the sign on top -- so only this structural check tells
    the two apart."""
    _, netlist, node = single(op, typ, typ, SHIFT_AMOUNT)
    width = typ.width
    value, amount = netlist.port("in_a").bits, netlist.port("in_b").bits
    arithmetic = op == "shr" and typ.signed
    if arithmetic:
        fill = value[width - 1]
    else:
        (zero,) = [i for i in netlist.instances_of_ir_node(node) if i.kind is PrimitiveKind.CONST0]
        fill = zero.output()
    stages = sorted(groups(netlist, kind="stage"), key=lambda g: dict(g.attrs)["stage"])
    assert len(stages) == (width - 1).bit_length()
    current = tuple(value)
    for k, stage in enumerate(stages):
        distance = 1 << k
        if op == "shl":
            moved = (fill,) * distance + current[: width - distance]
        else:
            moved = current[distance:] + (fill,) * distance
        members = by_role_and_bit(netlist, stage.id)
        nsel = members.pop(("shift_stage_nsel", k))
        assert pins(netlist, nsel) == [amount[k]]
        outputs = []
        for i in range(width):
            yes = members.pop(("shift_stage_yes", i))
            no = members.pop(("shift_stage_no", i))
            out = members.pop(("shift_stage_out", i))
            assert pins(netlist, yes) == [amount[k], moved[i]], f"stage {k} bit {i}: wrong moved-in signal"
            assert pins(netlist, no) == [nsel.output(), current[i]], f"stage {k} bit {i}: wrong kept signal"
            assert pins(netlist, out) == [yes.output(), no.output()]
            outputs.append(out.output())
        assert not members, f"stage {k}: unexpected primitives {sorted(members)}"
        current = tuple(outputs)
    # Saturation: too_large ? fill : barrel, with in_range = (amount < W) from the range check.
    (check,) = groups(netlist, name="range_check")
    (saturate,) = groups(netlist, name="saturate")
    members = by_role_and_bit(netlist, saturate.id)
    result = netlist.buses[node].bits
    if arithmetic:
        too_large = members.pop(("shift_too_large", 0))
        (in_range,) = pins(netlist, too_large)
    else:
        in_range = input_driver(netlist, members[("shift_in_range_and", 0)], "a")
    assert in_range is not None
    assert netlist.instance(in_range.instance).provenance.group in netlist.descendants(check.id)
    for i in range(width):
        if arithmetic:
            yes = members.pop(("shift_sign_fill_yes", i))
            no = members.pop(("shift_sign_fill_no", i))
            out = members.pop(("shift_sign_fill_out", i))
            assert pins(netlist, yes) == [too_large.output(), fill], f"bit {i}: must saturate to the ORIGINAL sign"
            assert pins(netlist, no) == [in_range, current[i]]
            assert pins(netlist, out) == [yes.output(), no.output()]
            assert result[i] == out.output()
        else:
            mask = members.pop(("shift_in_range_and", i))
            assert pins(netlist, mask) == [in_range, current[i]]
            assert result[i] == mask.output()
    assert not members, f"saturation: unexpected primitives {sorted(members)}"


def test_every_gate_is_attributed_to_the_ir_node_it_implements() -> None:
    graph = Graph()
    a, b = graph.input("in_a", INT8), graph.input("in_b", INT8)
    total = graph.node("add", INT8, (a, b))
    product = graph.node("mul", INT8, (total, b))
    smaller = graph.node("lt", BOOL, (product, a))
    chosen = graph.node("mux", INT8, (smaller, total, product))
    wide = graph.node("cast", INT16, (chosen,))
    graph.output("result", wide)
    graph.output("flag", smaller)
    netlist = synth(graph)
    assert_clean(netlist)
    ops = {total.id: "add", product.id: "mul", smaller.id: "lt", chosen.id: "mux", wide.id: "cast"}
    for inst in gates(netlist):
        node = inst.provenance.ir_node
        assert node in ops and netlist.ir_nodes[node].op == ops[node]
        chain = ancestors(netlist, inst.provenance.group)
        assert all(g.ir_node == node for g in chain[:-1])  # everything below the root
        assert chain[-2].name == f"n{node}.{ops[node]}" and chain[-2].kind == "ir_node"
        assert netlist.group_path(inst.provenance.group)[:2] == ("design", f"n{node}.{ops[node]}")
    # Each node owns exactly the gates its op needs in isolation.
    isolated = {
        total.id: single("add", INT8, INT8, INT8)[1].gate_count,
        product.id: single("mul", INT8, INT8, INT8)[1].gate_count,
        smaller.id: single("lt", BOOL, INT8, INT8)[1].gate_count,
        chosen.id: single("mux", INT8, BOOL, INT8, INT8)[1].gate_count,
        wide.id: 0,  # sign extension is wiring
    }
    assert Counter(i.provenance.ir_node for i in gates(netlist)) == {k: v for k, v in isolated.items() if v}
    assert PrimitiveSimulator(netlist).evaluate(in_a=-7, in_b=5) == graph.evaluate(in_a=-7, in_b=5)


def test_dead_ir_produces_no_primitives() -> None:
    graph = Graph()
    a, b = graph.input("in_a", UINT8), graph.input("in_b", UINT8)
    dead = graph.node("mul", UINT8, (a, b))  # computed but never observed
    deader = graph.node("div", UINT8, (dead, a))
    graph.output("result", graph.node("add", UINT8, (a, b)))
    netlist = synth(graph)
    assert_clean(netlist)
    for node in (dead.id, deader.id):
        assert node not in netlist.ir_nodes and node not in netlist.buses
        assert not netlist.instances_of_ir_node(node)
        assert all(g.ir_node != node for g in netlist.groups)
    assert netlist.gate_count == single("add", UINT8, UINT8, UINT8)[1].gate_count == 34


def test_the_builder_never_simplifies_constant_operands() -> None:
    """``x + 0`` and ``x * 1`` still build the full networks (constant
    propagation would be a separate netlist pass, not synthesis)."""
    for op, constant in (("add", 0), ("mul", 1), ("and", 255)):
        graph = Graph()
        a = graph.input("in_a", UINT8)
        graph.output("result", graph.node(op, UINT8, (a, graph.constant(constant, UINT8))))
        netlist = synth(graph)
        assert netlist.gate_count == single(op, UINT8, UINT8, UINT8)[1].gate_count
        values = list(range(256))
        expected = [graph.evaluate(in_a=v)["result"] for v in values]
        assert PrimitiveSimulator(netlist).evaluate_many({"in_a": values})["result"] == expected


def _mixed_graph() -> Graph:
    graph = Graph()
    a, b = graph.input("in_a", INT8), graph.input("in_b", INT8)
    acc = graph.register(INT8, init=3)
    q = graph.node("div", INT8, (a, b))
    s = graph.node("shr", INT8, (acc, graph.node("cast", SHIFT_AMOUNT, (b,))))
    graph.set_register(acc, graph.node("add", INT8, (q, s)), graph.node("lt", BOOL, (a, b)))
    graph.output("result", acc)
    graph.output("flag", graph.node("ne", BOOL, (q, s)))
    return graph


class _RecordingTrace:
    """The part of a trace recorder synthesis reports to."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def wants(self, level: TraceLevel) -> bool:
        return level <= TraceLevel.BASIC

    def emit(self, phase: str, type: str, *, level: TraceLevel = TraceLevel.BASIC, **data) -> None:
        self.events.append((type, data))


def test_the_synthesis_trace_names_the_network_built_for_each_node() -> None:
    graph = Graph()
    a, b = graph.input("in_a", INT8), graph.input("in_b", INT8)
    quotient = graph.node("div", INT8, (a, b))
    amount = graph.node("cast", SHIFT_AMOUNT, (b,))
    shifted = graph.node("shr", INT8, (quotient, amount))
    smaller = graph.node("lt", BOOL, (shifted, a))
    graph.output("result", smaller)
    trace = _RecordingTrace()
    synth(graph, trace=trace)
    recipes = {
        data["ir_node"]: data["recipe"]
        for kind, data in trace.events
        if kind == "ir_node_synthesized" and data.get("recipe")
    }
    assert recipes == {
        quotient.id: "restoring_divide_signed",
        amount.id: "cast_sign_extend",
        shifted.id: "barrel_shift_right_arithmetic",
        smaller.id: "less_chain_lt_signed",
    }


def test_synthesis_is_deterministic() -> None:
    first = synth(_mixed_graph()).to_dict()
    second = synth(_mixed_graph()).to_dict()
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


ALL_OP_TYPES = [BOOL, IRType(1), IRType(1, signed=True), IRType(5, signed=True), UINT8, INT8]
EVERY_OP_TYPING = [(op, typ) for op in sorted(OPS) for typ in ALL_OP_TYPES if op != "not" or typ == BOOL]


@pytest.mark.parametrize(("op", "typ"), EVERY_OP_TYPING, ids=[f"{op}-{t.name}" for op, t in EVERY_OP_TYPING])
def test_every_recipe_emits_live_one_bit_basis_logic(op: str, typ: IRType) -> None:
    result, operands = operands_for(op, typ)
    _, netlist, node = single(op, result, *operands)
    assert_clean(netlist)
    assert all(inst.provenance.ir_node == node for inst in gates(netlist))
    assert netlist.buses[node].type == result


#: Exact sizes (regression pins documenting the networks); ``W`` = width.
GATE_COUNTS = [
    ("add", UINT8, 34),  # 5W - 6
    ("sub", UINT8, 45),  # 6W - 3
    ("mul", UINT8, 136),
    ("mul", INT8, 136),  # the very same array multiplier
    ("div", UINT8, 962),
    ("mod", UINT8, 1031),
    ("div", INT8, 1032),
    ("mod", INT8, 1100),
    ("shl", UINT8, 525),  # 3 stages of 3W + 1, 64-bit range check (442), W masks
    ("shr", INT8, 542),
    ("lt", UINT8, 50),  # 7W - 6
    ("lt", INT8, 55),  # 7W - 1
    ("eq", UINT8, 16),  # 2W
    ("div", UINT32, 16130),
]


@pytest.mark.parametrize(("op", "typ", "count"), GATE_COUNTS, ids=[f"{o}-{t.name}" for o, t, _ in GATE_COUNTS])
def test_gate_counts(op: str, typ: IRType, count: int) -> None:
    result, operands = operands_for(op, typ)
    assert single(op, result, *operands)[1].gate_count == count


@pytest.mark.parametrize("width", [2, 3, 5, 8, 13, 16, 32, 64])
def test_gate_count_formulas(width: int) -> None:
    """Ripple structures grow linearly (a lookahead adder would not)."""
    unsigned, signed = IRType(width), IRType(width, signed=True)
    assert single("add", unsigned, unsigned, unsigned)[1].gate_count == 5 * width - 6
    assert single("sub", signed, signed, signed)[1].gate_count == 6 * width - 3
    assert single("neg", signed, signed)[1].gate_count == 3 * width - 1
    assert single("lt", BOOL, unsigned, unsigned)[1].gate_count == 7 * width - 6
    assert single("gt", BOOL, signed, signed)[1].gate_count == 7 * width - 1
    assert single("ne", BOOL, unsigned, unsigned)[1].gate_count == 2 * width - 1


# -- casts are wiring ---------------------------------------------------------------


def test_casts_reuse_source_bits() -> None:
    def cast_bits(source: IRType, target: IRType) -> tuple[PrimitiveNetlist, tuple, tuple]:
        _, netlist, node = single("cast", target, source)
        return netlist, netlist.port("in_a").bits, netlist.buses[node].bits

    netlist, src, out = cast_bits(INT8, UINT8)  # signedness only: the very same bits
    assert out == src and netlist.gate_count == 0
    netlist, src, out = cast_bits(BOOL, IRType(1, signed=True))
    assert out == src and netlist.gate_count == 0
    netlist, src, out = cast_bits(INT8, INT16)  # sign extension repeats the sign signal
    assert out[:8] == src and out[8:] == (src[7],) * 8 and netlist.gate_count == 0
    netlist, src, out = cast_bits(UINT8, INT16)  # zero extension uses the node's constant 0
    assert out[:8] == src and netlist.gate_count == 0
    assert {netlist.instance(bit.instance).kind for bit in out[8:]} == {PrimitiveKind.CONST0}
    assert len(set(out[8:])) == 1
    netlist, src, out = cast_bits(INT16, INT8)  # truncation keeps the low bits
    assert out == src[:8] and netlist.gate_count == 0
    netlist, src, out = cast_bits(UINT8, BOOL)  # int -> bool: a balanced OR tree
    assert role_counts(gates(netlist)) == {"truth_or": 7}
    netlist, src, out = cast_bits(IRType(1, signed=True), BOOL)  # one bit is its own truth
    assert out == src and netlist.gate_count == 0
    netlist, src, out = cast_bits(BOOL, UINT8)
    assert out[0] == src[0] and netlist.gate_count == 0


# -- special nodes: inputs, constants, outputs --------------------------------------


def test_const_nodes_become_constant_bits() -> None:
    graph = Graph()
    graph.output("result", graph.constant(0xA5, INT8))
    graph.output("flag", graph.constant(1, BOOL))
    netlist = synth(graph)
    assert_clean(netlist)
    assert netlist.gate_count == 0
    kinds = netlist.kind_counts()
    assert kinds["const0"] == 1 and kinds["const1"] == 2  # one per value per const node
    assert PrimitiveSimulator(netlist).evaluate() == graph.evaluate() == {"result": -91, "flag": 1}


def test_constants_feeding_an_operation_stay_in_their_const_node() -> None:
    graph = Graph()
    a = graph.input("in_a", UINT8)
    three = graph.constant(3, UINT8)
    graph.output("result", graph.node("sub", UINT8, (a, three)))
    netlist = synth(graph)
    const_bits = netlist.buses[three.id].bits
    assert {netlist.instance(b.instance).provenance.ir_node for b in const_bits} == {three.id}
    values = list(range(256))
    expected = [graph.evaluate(in_a=v)["result"] for v in values]
    assert PrimitiveSimulator(netlist).evaluate_many({"in_a": values})["result"] == expected


def test_outputs_pass_through_alias_and_fan_out() -> None:
    graph = Graph()
    a = graph.input("in_a", INT8)
    flag = graph.input("in_flag", BOOL)
    graph.input("in_unused", UINT4)
    graph.output("result", a)  # an input straight to an output
    graph.output("copy", a)  # the same node observed twice
    graph.output("flag", flag)
    netlist = synth(graph)
    assert_clean(netlist)
    assert netlist.gate_count == 0
    for bit in netlist.port("in_a").bits:
        net = netlist.net_of_driver(bit)
        assert net is not None and net.fanout == 2  # ONE net, two output-pad sinks
    # An unused input still has its pads but drives no net.
    assert all(netlist.net_of_driver(bit) is None for bit in netlist.port("in_unused").bits)
    sim = PrimitiveSimulator(netlist)
    for value in (-128, -1, 0, 1, 127):
        stimulus = {"in_a": value, "in_flag": value & 1, "in_unused": 9}
        assert sim.evaluate(**stimulus) == graph.evaluate(**stimulus)


# -- sequential: registers --------------------------------------------------------------


def _register_graph() -> Graph:
    """Signed and unsigned registers with non-zero inits, runtime / constant-true
    / constant-false enables."""
    graph = Graph()
    enable = graph.input("in_en", BOOL)
    x = graph.input("in_x", INT8)
    acc = graph.register(INT8, init=-5)  # signed state, negative init, runtime enable
    count = graph.register(UINT4, init=9)  # always enabled, wraps at 16
    held = graph.register(UINT4, init=6)  # never enabled: holds its init forever
    flag = graph.register(BOOL, init=1)
    graph.set_register(acc, graph.node("add", INT8, (acc, x)), enable)
    graph.set_register(count, graph.node("add", UINT4, (count, graph.constant(1, UINT4))), graph.constant(1, BOOL))
    graph.set_register(held, graph.node("inv", UINT4, (held,)), graph.constant(0, BOOL))
    graph.set_register(flag, graph.node("lt", BOOL, (acc, x)), graph.node("not", BOOL, (enable,)))
    graph.output("result", acc)
    graph.output("count", count)
    graph.output("held", held)
    graph.output("flag", flag)
    graph.output("ahead", graph.node("mul", INT8, (acc, x)))
    return graph


def test_register_state_is_one_bit_registers_with_global_clock_and_reset() -> None:
    graph = _register_graph()
    netlist = synth(graph)
    assert_clean(netlist, sequential=True)
    kinds = netlist.kind_counts()
    assert kinds["register_bit"] == 8 + 4 + 4 + 1
    assert kinds["clock_source"] == kinds["reset_source"] == 1
    for node in graph.registers:
        bus = netlist.buses[node]
        init = graph.nodes[node]["init"]
        for i, q in enumerate(bus.bits):
            reg = netlist.instance(q.instance)
            assert reg.kind is PrimitiveKind.REGISTER_BIT and reg.init == bool((init >> i) & 1)
            d = input_driver(netlist, reg, "d")
            assert d is not None
            mux = netlist.instance(d.instance)  # the IR enable is an ordinary mux in front of d
            assert mux.kind is PrimitiveKind.OR and mux.provenance.ir_node == node
            assert "enable_mux" in netlist.group_path(mux.provenance.group)
    roles = {net.role for net in netlist.nets}
    assert roles == {"data", "clock", "reset"}


def test_registers_step_exactly_like_the_ir_cycle_by_cycle() -> None:
    graph = _register_graph()
    sim = PrimitiveSimulator(synth(graph))
    ir_state, state = graph.reset_state(), sim.reset_state()
    assert sim.state_to_ir(state) == ir_state and sim.state_from_ir(ir_state) == state
    rng = random.Random(20261004)
    enables = set()
    for cycle in range(400):
        stimulus = {"in_en": rng.getrandbits(1), "in_x": rng.getrandbits(8)}
        enables.add(stimulus["in_en"])
        ir_outputs, ir_state = graph.step(ir_state, **stimulus)
        outputs, state = sim.step(state, **stimulus)
        assert outputs == ir_outputs, cycle
        assert sim.state_to_ir(state) == ir_state, cycle
        assert sim.state_from_ir(ir_state) == state, cycle
        assert outputs["held"] == 6
    assert enables == {0, 1}


def test_reset_restores_every_register_init() -> None:
    graph = _register_graph()
    sim = PrimitiveSimulator(synth(graph))
    state = sim.reset_state()
    rng = random.Random(5)
    for _ in range(37):
        _, state = sim.step(state, in_en=1, in_x=rng.getrandbits(8))
    assert state != sim.reset_state()
    _, after_reset = sim.step(state, rst=1, in_en=1, in_x=99)
    assert after_reset == sim.reset_state()
    assert sim.state_to_ir(after_reset) == graph.reset_state()
    # ... and from there the netlist tracks the IR from its own reset state again.
    ir_state = graph.reset_state()
    for _ in range(20):
        stimulus = {"in_en": rng.getrandbits(1), "in_x": rng.getrandbits(8)}
        ir_outputs, ir_state = graph.step(ir_state, **stimulus)
        outputs, after_reset = sim.step(after_reset, **stimulus)
        assert outputs == ir_outputs and sim.state_to_ir(after_reset) == ir_state


def _fib_graph() -> Graph:
    return compile_source((EXAMPLES / "uint8_fib.redc").read_text(), top="fib")


def _py_fib(n: int) -> int:
    last1 = last2 = 1
    for _ in range(2, n):
        last1, last2 = last2, last1 + last2
    return last2 & 0xFF


@pytest.mark.parametrize("policy", [PadInterfacePolicy(), DefaultInterfacePolicy()], ids=["pads", "peripherals"])
def test_fib_runs_like_the_ir(policy) -> None:
    graph = _fib_graph()
    netlist = synth(graph, interface=policy)
    assert_clean(netlist, sequential=True, peripherals=isinstance(policy, DefaultInterfacePolicy))
    sim = PrimitiveSimulator(netlist)
    for n in (0, 1, 2, 3, 5, 8, 13, 14, 25, 60):
        expected = graph.run(in_n=n)
        assert sim.run(in_n=n) == expected
        assert expected["result"] == _py_fib(n)


def test_fib_steps_like_the_ir_with_random_handshakes() -> None:
    graph = _fib_graph()
    sim = PrimitiveSimulator(synth(graph))
    ir_state, state = graph.reset_state(), sim.reset_state()
    assert sim.state_to_ir(state) == ir_state
    rng = random.Random(11)
    n, done = 9, 0
    for cycle in range(700):
        if rng.random() < 0.05:
            n = rng.randrange(30)
        stimulus = {"start": int(rng.random() < 0.15), "in_n": n}
        ir_outputs, ir_state = graph.step(ir_state, **stimulus)
        outputs, state = sim.step(state, **stimulus)
        assert outputs == ir_outputs, cycle
        assert sim.state_to_ir(state) == ir_state, cycle
        done += outputs["done"]
    assert done > 5  # several complete transactions, including start-while-busy pulses


# -- whole programs ------------------------------------------------------------------


def _program_vectors(graph: Graph, rng: random.Random, samples: int = 1500) -> tuple[list[str], list[tuple]]:
    """Every input combination if there are at most 2**16, else edge values
    plus ``samples`` deterministic random vectors."""
    names = [port["name"] for port in graph.inputs]
    types = [IRType(**port["type"]) for port in graph.inputs]
    if sum(t.width for t in types) <= 16:
        return names, list(itertools.product(*(range(1 << t.width) for t in types)))
    edges = [tuple(v & t.mask for t in types) for v in (0, 1, -1, 2**63)]
    edges += [tuple(((1 << (t.width - 1)) - k) & t.mask for t in types) for k in (0, 1)]
    randoms = [tuple(rng.getrandbits(t.width) for t in types) for _ in range(samples)]
    return names, edges + randoms


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.redc")), ids=lambda p: p.stem)
def test_example_programs_match_the_ir(path: Path) -> None:
    graph = compile_source(path.read_text(), top=EXAMPLE_TOPS.get(path.stem, "main"))
    netlist = synth(graph)
    assert_clean(netlist, sequential=graph.sequential, every_bit_observed=False)
    sim = PrimitiveSimulator(netlist)
    rng = random.Random(path.stem)
    if graph.sequential:
        data = [p for p in graph.inputs if p["name"] != "start"]
        for _ in range(12):
            stimulus = {p["name"]: rng.getrandbits(min(IRType(**p["type"]).width, 6)) for p in data}
            assert sim.run(**stimulus) == graph.run(**stimulus), stimulus
        return
    names, vectors = _program_vectors(graph, rng)
    got = sim.evaluate_many({name: [v[k] for v in vectors] for k, name in enumerate(names)})
    for index, vector in enumerate(vectors):
        expected = graph.evaluate(**dict(zip(names, vector)))
        assert {name: values[index] for name, values in got.items()} == expected, vector


def test_peripheral_ports_simulate_like_pads() -> None:
    """``uint8 result`` becomes ONE seven-segment peripheral with eight
    independent one-bit pins; the simulator addresses it by port name."""
    graph = compile_source((EXAMPLES / "uint8_alu.redc").read_text())
    pads = PrimitiveSimulator(synth(graph))
    netlist = synth(graph, interface=DefaultInterfacePolicy())
    assert_clean(netlist, peripherals=True, every_bit_observed=False)
    port = netlist.port("result")
    assert port.realization == "2-dig-7-seg" and len(port.instances) == 1 and len(port.bits) == 8
    devices = PrimitiveSimulator(netlist)
    rng = random.Random(3)
    columns = {
        "in_a": [rng.getrandbits(8) for _ in range(500)],
        "in_b": [rng.getrandbits(8) for _ in range(500)],
        "in_op": [rng.getrandbits(3) for _ in range(500)],
    }
    assert devices.evaluate_many(columns) == pads.evaluate_many(columns)
