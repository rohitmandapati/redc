"""Functional equivalence of every primitive synthesis recipe with the IR.

Each check builds a single-operation graph with ``Graph.node`` -- NOT
``Graph.op``, which would constant-fold or simplify it -- synthesizes it to
one-bit primitives with pad ports, and simulates it BIT-PARALLEL: one
``PrimitiveSimulator.evaluate_many`` call settles every vector of a sweep at
once.  ``Graph.evaluate`` (the IR's own semantics) is the oracle for EVERY
vector; a single-op graph evaluates in a few microseconds, so no shortcut
oracle is needed even for 65 536-vector sweeps.

* exhaustive operand tuples for widths 1-4, unsigned and signed, and ``bool``
  wherever the IR types the op for it;
* exhaustive ``uint8`` / ``int8`` for every binary and comparison op;
* deterministic random vectors (``random.Random`` with fixed seeds) for widths
  5, 7, 8, 13, 16, 32 and 64;
* named edge cases pinned to their documented RedC values (``INT_MIN / -1``,
  division / modulo by zero, shift amounts up to ``2**64 - 1``, extension and
  truncation casts, ...).
"""

from __future__ import annotations

import functools
import itertools
import random
from collections.abc import Iterable, Sequence

import pytest

from redc.ir import BOOL, SHIFT_AMOUNT, Graph, IRType
from redc.physical_primitive import (
    PadInterfacePolicy,
    PrimitiveSimulator,
    synthesize_to_primitives,
)

UINT8 = IRType(8)
INT8 = IRType(8, signed=True)
UINT13 = IRType(13)
INT13 = IRType(13, signed=True)
INT16 = IRType(16, signed=True)
UINT64 = IRType(64)
INT64 = IRType(64, signed=True)
INT1 = IRType(1, signed=True)

ARITHMETIC = ("add", "sub", "mul", "div", "mod")
BITWISE = ("and", "or", "xor")
COMPARISONS = ("eq", "ne", "lt", "le", "gt", "ge")
BINARY = ARITHMETIC + BITWISE + COMPARISONS
SHIFTS = ("shl", "shr")
UNARY = ("neg", "inv")

#: Every type of width 1-4: bool, then unsigned and signed of each width.
TINY_TYPES = (BOOL, *(IRType(w, signed=s) for w in (1, 2, 3, 4) for s in (False, True)))
WIDE_WIDTHS = (5, 7, 8, 13, 16, 32, 64)
CAST_TYPES = TINY_TYPES + tuple(IRType(w, signed=s) for w in WIDE_WIDTHS for s in (False, True))
NAMES = ("in_a", "in_b", "in_c")
BYTE_PAIRS = list(itertools.product(range(256), repeat=2))


def type_id(typ: IRType) -> str:
    return typ.name


def result_type(op: str, typ: IRType) -> IRType:
    return BOOL if op in COMPARISONS else typ


@functools.lru_cache(maxsize=8)
def single_op(op: str, result: IRType, operands: tuple[IRType, ...]) -> tuple[Graph, PrimitiveSimulator]:
    """``result = op(in_a, in_b, ...)`` and the compiled simulation of its primitives."""
    graph = Graph()
    args = tuple(graph.input(name, typ) for name, typ in zip(NAMES, operands))
    graph.output("result", graph.node(op, result, args))
    netlist = synthesize_to_primitives(graph, interface=PadInterfacePolicy())
    return graph, PrimitiveSimulator(netlist)


def assert_matches_ir(
    op: str, result: IRType, operands: Sequence[IRType], vectors: Iterable[Sequence[int]]
) -> None:
    """Simulate every vector in ONE bit-parallel pass; compare each with Graph.evaluate."""
    graph, sim = single_op(op, result, tuple(operands))
    names = NAMES[: len(operands)]
    vectors = list(vectors)
    columns = {name: [vector[k] for vector in vectors] for k, name in enumerate(names)}
    got = sim.evaluate_many(columns)["result"]
    assert len(got) == len(vectors)
    mismatches = []
    for vector, value in zip(vectors, got):
        expected = graph.evaluate(**dict(zip(names, vector)))["result"]
        if value != expected:
            mismatches.append(f"{dict(zip(names, vector))}: primitive {value}, IR {expected}")
    signature = f"{op}({', '.join(t.name for t in operands)}) -> {result.name}"
    assert not mismatches, f"{signature}: {len(mismatches)}/{len(vectors)} mismatches, e.g. {mismatches[:5]}"


def edge_values(typ: IRType) -> list[int]:
    """0, 1, 2, 3, all ones (-1 / unsigned max), all ones - 1, INT_MIN, INT_MIN + 1
    and INT_MAX, as raw bit patterns."""
    top = 1 << (typ.width - 1)
    return sorted({v & typ.mask for v in (0, 1, 2, 3, typ.mask, typ.mask - 1, top, top + 1, top - 1)})


def random_values(rng: random.Random, typ: IRType, count: int) -> list[int]:
    """Raw patterns mixing full-width noise, small magnitudes of either sign
    and neighbours of powers of two (interesting quotients and carries)."""
    width = typ.width
    values = []
    for _ in range(count):
        mode = rng.randrange(4)
        if mode == 0:
            value = rng.getrandbits(width)
        elif mode == 1:
            value = rng.getrandbits(rng.randint(1, width))
        elif mode == 2:
            value = -rng.getrandbits(rng.randint(1, width))
        else:
            value = (1 << rng.randrange(width)) + rng.choice((-1, 0, 1))
        values.append(value & typ.mask)
    return values


def random_pairs(rng: random.Random, typ: IRType, count: int) -> list[tuple[int, int]]:
    """Every pair of edge values, ``count`` random pairs and some equal pairs."""
    left, right = random_values(rng, typ, count), random_values(rng, typ, count)
    edges = edge_values(typ)
    return [*itertools.product(edges, repeat=2), *zip(left, right), *((v, v) for v in left[:20])]


def shift_amounts(width: int) -> list[int]:
    """Every distance 0 .. W+2 plus huge uint64 amounts (the amount is ALWAYS
    uint64: 2**32, 2**63, 2**64 - 1, ...)."""
    far = {31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 2**32 - 1, 2**32, 2**32 + width}
    far |= {2**63, 2**63 + 1, 2**64 - width, 2**64 - 1}
    return sorted(set(range(width + 3)) | far)


# -- exhaustive: widths 1-4 (bool, uintN, intN) ---------------------------------


@pytest.mark.parametrize("typ", TINY_TYPES, ids=type_id)
@pytest.mark.parametrize("op", BINARY)
def test_tiny_binary_ops_exhaustive(op: str, typ: IRType) -> None:
    values = range(1 << typ.width)
    assert_matches_ir(op, result_type(op, typ), (typ, typ), itertools.product(values, repeat=2))


@pytest.mark.parametrize("typ", TINY_TYPES, ids=type_id)
@pytest.mark.parametrize("op", UNARY)
def test_tiny_unary_ops_exhaustive(op: str, typ: IRType) -> None:
    assert_matches_ir(op, typ, (typ,), [(v,) for v in range(1 << typ.width)])


def test_logical_not_exhaustive() -> None:
    assert_matches_ir("not", BOOL, (BOOL,), [(0,), (1,)])


@pytest.mark.parametrize("typ", TINY_TYPES, ids=type_id)
@pytest.mark.parametrize("op", SHIFTS)
def test_tiny_shifts_every_value_and_boundary_amount(op: str, typ: IRType) -> None:
    vectors = itertools.product(range(1 << typ.width), shift_amounts(typ.width))
    assert_matches_ir(op, typ, (typ, SHIFT_AMOUNT), vectors)


@pytest.mark.parametrize("typ", TINY_TYPES, ids=type_id)
def test_tiny_mux_exhaustive(typ: IRType) -> None:
    values = range(1 << typ.width)
    assert_matches_ir("mux", typ, (BOOL, typ, typ), itertools.product((0, 1), values, values))


# -- exhaustive: uint8 / int8 ---------------------------------------------------


@pytest.mark.parametrize("typ", (UINT8, INT8), ids=type_id)
@pytest.mark.parametrize("op", BINARY)
def test_byte_binary_ops_exhaustive(op: str, typ: IRType) -> None:
    assert_matches_ir(op, result_type(op, typ), (typ, typ), BYTE_PAIRS)


@pytest.mark.parametrize("typ", (UINT8, INT8), ids=type_id)
@pytest.mark.parametrize("op", UNARY)
def test_byte_unary_ops_exhaustive(op: str, typ: IRType) -> None:
    assert_matches_ir(op, typ, (typ,), [(v,) for v in range(256)])


@pytest.mark.parametrize("typ", (UINT8, INT8), ids=type_id)
@pytest.mark.parametrize("op", SHIFTS)
def test_byte_shifts_every_value_and_boundary_amount(op: str, typ: IRType) -> None:
    assert_matches_ir(op, typ, (typ, SHIFT_AMOUNT), itertools.product(range(256), shift_amounts(8)))


@pytest.mark.parametrize("typ", (UINT8, INT8), ids=type_id)
def test_byte_mux_exhaustive(typ: IRType) -> None:
    vectors = ((sel, a, b) for sel in (0, 1) for a, b in BYTE_PAIRS)
    assert_matches_ir("mux", typ, (BOOL, typ, typ), vectors)


# -- deterministic random: widths 5, 7, 8, 13, 16, 32, 64 ------------------------


@pytest.mark.parametrize("signed", (False, True), ids=("unsigned", "signed"))
@pytest.mark.parametrize("width", WIDE_WIDTHS)
@pytest.mark.parametrize("op", BINARY)
def test_wide_binary_ops_random(op: str, width: int, signed: bool) -> None:
    typ = IRType(width, signed=signed)
    rng = random.Random(f"binary/{op}/{typ.name}")
    assert_matches_ir(op, result_type(op, typ), (typ, typ), random_pairs(rng, typ, 300))


@pytest.mark.parametrize("signed", (False, True), ids=("unsigned", "signed"))
@pytest.mark.parametrize("width", WIDE_WIDTHS)
@pytest.mark.parametrize("op", SHIFTS)
def test_wide_shifts_random(op: str, width: int, signed: bool) -> None:
    typ = IRType(width, signed=signed)
    rng = random.Random(f"shift/{op}/{typ.name}")
    values = edge_values(typ) + random_values(rng, typ, 40)
    amounts = [
        *shift_amounts(width),
        *(rng.randrange(width) for _ in range(10)),
        *(rng.getrandbits(64) for _ in range(10)),
        *(rng.getrandbits(rng.randint(1, 64)) for _ in range(10)),
    ]
    assert_matches_ir(op, typ, (typ, SHIFT_AMOUNT), itertools.product(values, amounts))


@pytest.mark.parametrize("signed", (False, True), ids=("unsigned", "signed"))
@pytest.mark.parametrize("width", WIDE_WIDTHS)
def test_wide_unary_ops_and_mux_random(width: int, signed: bool) -> None:
    typ = IRType(width, signed=signed)
    rng = random.Random(f"unary/{typ.name}")
    values = edge_values(typ) + random_values(rng, typ, 200)
    for op in UNARY:
        assert_matches_ir(op, typ, (typ,), [(v,) for v in values])
    vectors = [(rng.getrandbits(1), a, b) for a, b in random_pairs(rng, typ, 200)]
    assert_matches_ir("mux", typ, (BOOL, typ, typ), vectors)


@pytest.mark.parametrize("source", CAST_TYPES, ids=type_id)
def test_casts_to_every_type(source: IRType) -> None:
    """Every (source, target) pair over bool and widths 1-8, 13, 16, 32, 64:
    widening, narrowing, signedness-only, to bool and from bool."""
    if source.width <= 8:
        values = list(range(1 << source.width))
    else:
        values = edge_values(source) + random_values(random.Random(f"cast/{source.name}"), source, 200)
    for target in CAST_TYPES:
        assert_matches_ir("cast", target, (source,), [(v,) for v in values])


# -- named edge cases ---------------------------------------------------------------

#: ``(op, type, a, b, documented RedC result)``; the IR must agree with the
#: documented value and the primitive netlist with both.
EDGE_CASES = [
    ("add", UINT8, 255, 1, 0),  # carry out of the top bit wraps away
    ("add", UINT8, 255, 255, 254),
    ("add", INT8, 127, 1, -128),  # signed overflow wraps
    ("add", INT8, -128, -1, 127),
    ("sub", UINT8, 0, 1, 255),  # borrow wraps
    ("sub", UINT8, 5, 7, 254),
    ("sub", INT8, -128, 1, 127),
    ("sub", INT8, 0, -128, -128),
    ("mul", UINT8, 16, 16, 0),  # only the low W product bits survive
    ("mul", UINT8, 255, 255, 1),
    ("mul", INT8, -128, -1, -128),
    ("mul", INT8, -3, 5, -15),
    ("mul", INT8, -7, -9, 63),
    ("mul", INT64, -(2**63), 3, -(2**63)),
    ("div", UINT8, 200, 0, 0),  # division by zero is 0
    ("div", INT8, -7, 0, 0),
    ("mod", UINT8, 200, 0, 0),  # modulo by zero is 0
    ("mod", INT8, -128, 0, 0),
    ("div", INT8, -128, -1, -128),  # INT_MIN / -1 wraps to INT_MIN
    ("mod", INT8, -128, -1, 0),  # INT_MIN % -1 == 0
    ("div", INT8, -7, 2, -3),  # truncation toward zero
    ("div", INT8, 7, -2, -3),  # negative divisor
    ("div", INT8, -7, -2, 3),
    ("mod", INT8, -7, 2, -1),  # the remainder takes the DIVIDEND's sign
    ("mod", INT8, 7, -2, 1),
    ("mod", INT8, -7, -2, -1),
    ("div", INT8, -128, 2, -64),
    ("mod", INT8, -128, 3, -2),
    ("div", INT8, 127, -128, 0),
    ("mod", INT8, 127, -128, 127),
    ("div", UINT8, 255, 1, 255),
    ("mod", UINT8, 255, 16, 15),
    ("div", INT64, -(2**63), -1, -(2**63)),
    ("mod", INT64, -(2**63), -1, 0),
    ("div", UINT64, 2**64 - 1, 3, (2**64 - 1) // 3),
    ("mod", UINT64, 2**64 - 1, 2**63, 2**63 - 1),
    ("lt", INT8, -1, 0, 1),  # signed comparison crossing zero
    ("lt", INT8, 0, -1, 0),
    ("lt", UINT8, 255, 0, 0),  # the same bit patterns compared unsigned
    ("gt", INT8, -128, 127, 0),
    ("gt", UINT8, 128, 127, 1),
    ("ge", INT8, -1, 1, 0),
    ("le", UINT8, 0, 255, 1),
    ("le", INT8, -5, -5, 1),  # equal operands
    ("ge", INT8, -5, -5, 1),
    ("lt", INT8, -5, -5, 0),
    ("gt", INT8, -5, -5, 0),
    ("eq", INT8, -5, -5, 1),
    ("ne", INT8, -5, -5, 0),
    ("ne", UINT8, 0, 255, 1),
    ("lt", INT64, -(2**63), 2**63 - 1, 1),
    ("lt", UINT64, 2**63 - 1, 2**63, 1),
]


def _edge_id(case: tuple[str, IRType, int, int, int]) -> str:
    op, typ, a, b, _ = case
    return f"{op}-{typ.name}-{a}-{b}"


@pytest.mark.parametrize(("op", "typ", "a", "b", "expected"), EDGE_CASES, ids=[_edge_id(c) for c in EDGE_CASES])
def test_named_arithmetic_and_comparison_edge_cases(op: str, typ: IRType, a: int, b: int, expected: int) -> None:
    graph, sim = single_op(op, result_type(op, typ), (typ, typ))
    assert graph.evaluate(in_a=a, in_b=b)["result"] == expected
    assert sim.evaluate(in_a=a, in_b=b)["result"] == expected


#: ``(op, type, value, amount, documented result)``.
SHIFT_EDGES = [
    ("shl", UINT8, 0b1011_0001, 0, 0b1011_0001),  # by 0
    ("shl", UINT8, 0b1011_0001, 7, 0b1000_0000),  # by W - 1
    ("shl", UINT8, 0xFF, 8, 0),  # by W
    ("shl", UINT8, 0xFF, 9, 0),  # by W + 1
    ("shl", UINT8, 1, 63, 0),
    ("shl", UINT8, 1, 64, 0),
    ("shl", UINT8, 1, 2**32, 0),
    ("shl", UINT8, 1, 2**63, 0),
    ("shl", UINT8, 1, 2**64 - 1, 0),
    ("shl", UINT13, 1, 12, 4096),  # a non-power-of-two width
    ("shl", UINT13, 1, 13, 0),
    ("shl", UINT13, 1, 15, 0),  # 13..15 set no amount bit above the barrel's stages
    ("shl", INT64, 1, 63, -(2**63)),
    ("shl", INT64, 1, 64, 0),
    ("shl", UINT64, 3, 63, 2**63),
    ("shr", UINT8, 0x80, 7, 1),  # logical: zero fill
    ("shr", UINT8, 0xFF, 8, 0),
    ("shr", UINT8, 0xFF, 2**64 - 1, 0),
    ("shr", UINT13, 0x1FFF, 12, 1),
    ("shr", UINT13, 0x1FFF, 13, 0),
    ("shr", UINT13, 0x1FFF, 14, 0),
    ("shr", INT8, -128, 0, -128),  # arithmetic: sign fill ...
    ("shr", INT8, -128, 1, -64),
    ("shr", INT8, -128, 7, -1),
    ("shr", INT8, -128, 8, -1),  # ... saturating to copies of the sign
    ("shr", INT8, -2, 2**63, -1),
    ("shr", INT8, 127, 6, 1),
    ("shr", INT8, 127, 7, 0),
    ("shr", INT8, 127, 2**64 - 1, 0),
    ("shr", INT13, -4096, 12, -1),
    ("shr", INT13, -4096, 13, -1),
    ("shr", INT13, -4096, 15, -1),
    ("shr", INT64, -(2**63), 63, -1),
    ("shr", INT64, -(2**63), 64, -1),
    ("shr", INT64, 2**62, 64, 0),
    ("shr", INT64, -(2**63), 2**32, -1),
    ("shl", BOOL, 1, 0, 1),
    ("shl", BOOL, 1, 1, 0),
    ("shr", BOOL, 1, 1, 0),
    ("shr", INT1, 1, 5, -1),
]


@pytest.mark.parametrize(
    ("op", "typ", "value", "amount", "expected"), SHIFT_EDGES, ids=[_edge_id(c) for c in SHIFT_EDGES]
)
def test_named_shift_edge_cases(op: str, typ: IRType, value: int, amount: int, expected: int) -> None:
    graph, sim = single_op(op, typ, (typ, SHIFT_AMOUNT))
    assert graph.evaluate(in_a=value, in_b=amount)["result"] == expected
    assert sim.evaluate(in_a=value, in_b=amount)["result"] == expected


#: ``(source, target, value, documented result)``.
CAST_EDGES = [
    (INT8, INT16, -1, -1),  # sign extension
    (INT8, INT16, -128, -128),
    (INT8, IRType(16), -1, 0xFFFF),  # a signed source sign-extends even into an unsigned type
    (UINT8, INT16, 255, 255),  # zero extension
    (UINT8, UINT64, 0x80, 0x80),
    (INT16, INT8, -129, 127),  # truncation keeps the low bits
    (IRType(16), UINT8, 0x1234, 0x34),
    (INT64, IRType(1), -1, 1),
    (INT8, UINT8, -1, 255),  # signedness only
    (UINT8, INT8, 200, -56),
    (INT8, BOOL, -128, 1),  # int -> bool is "!= 0"
    (UINT8, BOOL, 0, 0),
    (UINT8, BOOL, 0x10, 1),
    (INT64, BOOL, -(2**63), 1),
    (BOOL, INT8, 1, 1),  # bool -> int zero-extends
    (BOOL, UINT8, 1, 1),
    (BOOL, INT1, 1, -1),
    (INT1, INT8, 1, -1),  # int1 holds -1 or 0
    (INT1, BOOL, 1, 1),
]


def _cast_id(case: tuple[IRType, IRType, int, int]) -> str:
    source, target, value, _ = case
    return f"{source.name}-to-{target.name}-{value}"


@pytest.mark.parametrize(("source", "target", "value", "expected"), CAST_EDGES, ids=[_cast_id(c) for c in CAST_EDGES])
def test_named_cast_edge_cases(source: IRType, target: IRType, value: int, expected: int) -> None:
    graph, sim = single_op("cast", target, (source,))
    assert graph.evaluate(in_a=value)["result"] == expected
    assert sim.evaluate(in_a=value)["result"] == expected


@pytest.mark.parametrize("op", (*COMPARISONS, "sub", "xor", "add", "mul", "div", "mod"))
def test_one_value_used_as_both_operands(op: str) -> None:
    """``x op x`` -- the same input bits feed both operand positions."""
    graph = Graph()
    a = graph.input("in_a", INT8)
    graph.output("result", graph.node(op, result_type(op, INT8), (a, a)))
    sim = PrimitiveSimulator(synthesize_to_primitives(graph, interface=PadInterfacePolicy()))
    values = list(range(256))
    assert sim.evaluate_many({"in_a": values})["result"] == [graph.evaluate(in_a=v)["result"] for v in values]
