"""Adversarial differential harness: one-bit primitive netlists vs the IR.

Written independently of ``test_primitive_synthesis.py`` and
``test_primitive_synthesis_ops.py`` (nothing is imported from them).  Every
check builds its graph with ``Graph.node`` -- never ``Graph.op``, which would
constant-fold or simplify -- synthesizes it with
:func:`~redc.physical_primitive.synthesize_to_primitives` and compares the
primitive circuit with the IR's own semantics:

* combinational: ``PrimitiveSimulator.evaluate_many`` (bit-parallel) against
  ``Graph.evaluate``.  The 2**18-vector sweeps call ``redc.ir._calculate``
  directly -- exactly what ``Graph.evaluate`` computes for a one-node graph --
  after checking that shortcut against ``Graph.evaluate`` on a sample;
* sequential: ``PrimitiveSimulator.step`` / ``run`` against ``Graph.step`` /
  ``run`` cycle by cycle, including reset pulses and state round trips;
* :class:`IndependentEvaluator` (written here, sharing no code with
  :mod:`redc.physical_primitive.simulate`) re-evaluates many netlists from
  their nets alone, so a simulator bug cannot mask a synthesis bug.

Coverage: every binary op over widths 1-7 exhaustively (width 9 too for div
and mod), widths 8/9/16/31/32/33/63/64 with edge-biased vectors (0, 1, 2, -1,
-2, INT_MIN, INT_MIN+1, INT_MAX, UMAX, UMAX-1, powers of two +-1); all 121
casts among 11 types; shifts by 0..W+2, every power of two and huge uint64
amounts; mux over every type; ONE IR node feeding several operands; random
typed DAGs; hand-built sequential graphs (back-edges to later nodes, register
chains, constant enables, signed inits, no inputs at all) and compiled RedC
programs; default-policy peripherals; and the port-name ambiguity of
hand-built graphs.  Every random choice uses ``random.Random`` with a fixed
seed.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cache

import pytest

from redc import compile_source
from redc.ir import BOOL, OPS, SHIFT_AMOUNT, Graph, IRType, Value, _calculate
from redc.parser import CompileError
from redc.physical_primitive import (
    GATE_KINDS,
    BitTerminal,
    DefaultInterfacePolicy,
    InterfacePolicy,
    PadInterfacePolicy,
    PrimitiveKind,
    PrimitiveNetlist,
    PrimitiveSimulator,
    PrimitiveTraceRecorder,
    synthesize_to_primitives,
)
from redc.physical_primitive.synthesis.recipes import DEFAULT_SYNTHESIS
from redc.tracing import TraceLevel

PADS = PadInterfacePolicy()

UINT3, INT3 = IRType(3), IRType(3, signed=True)
UINT4 = IRType(4)
UINT8, INT8 = IRType(8), IRType(8, signed=True)
UINT16, INT16 = IRType(16), IRType(16, signed=True)
UINT64, INT64 = IRType(64), IRType(64, signed=True)
UINT1, INT1 = IRType(1), IRType(1, signed=True)

ARITHMETIC = ("add", "sub", "mul", "div", "mod")
BITWISE = ("and", "or", "xor")
COMPARISONS = ("eq", "ne", "lt", "le", "gt", "ge")
BINARY = ARITHMETIC + BITWISE + COMPARISONS
SHIFTS = ("shl", "shr")
#: Ops whose networks grow quadratically with the width.
HEAVY = frozenset({"mul", "div", "mod"})

NAMES = ("in_a", "in_b", "in_c")
#: Vectors checked against ``Graph.evaluate`` itself when a fast oracle is used.
ORACLE_SAMPLE = 512

COMBINATIONAL_KINDS = GATE_KINDS | {
    PrimitiveKind.INPUT_BIT,
    PrimitiveKind.OUTPUT_BIT,
    PrimitiveKind.CONST0,
    PrimitiveKind.CONST1,
}
SEQUENTIAL_KINDS = COMBINATIONAL_KINDS | {
    PrimitiveKind.REGISTER_BIT,
    PrimitiveKind.CLOCK_SOURCE,
    PrimitiveKind.RESET_SOURCE,
}


def int_types(widths: Iterable[int]) -> tuple[IRType, ...]:
    """``bool`` followed by ``uintW`` and ``intW`` for every width."""
    return (BOOL, *(IRType(w, signed=s) for w in widths for s in (False, True)))


def type_name(typ: IRType) -> str:
    return typ.name


def result_of(op: str, typ: IRType) -> IRType:
    return BOOL if op in COMPARISONS else typ


# -- synthesis + a structural sanity net ------------------------------------------


def synth(graph: Graph, policy: InterfacePolicy = PADS) -> PrimitiveNetlist:
    """Synthesize (which ends with ``validate(complete=True)``) and check that
    only basis gates, state and boundaries exist."""
    netlist = synthesize_to_primitives(graph, interface=policy)
    allowed = SEQUENTIAL_KINDS if netlist.is_sequential else COMBINATIONAL_KINDS
    if not isinstance(policy, PadInterfacePolicy):
        allowed = allowed | {PrimitiveKind.PERIPHERAL}
    stray = {inst.kind for inst in netlist.instances} - allowed
    assert not stray, f"non-basis primitives survived: {sorted(k.value for k in stray)}"
    return netlist


# -- an evaluator that shares no code with PrimitiveSimulator -------------------------


class IndependentEvaluator:
    """Evaluates a :class:`PrimitiveNetlist` from its nets alone.

    Gates run in a depth-first topological order (post-order from each gate's
    operands); values are bit-parallel Python ints (bit ``k`` = vector ``k``).
    A register bit loads its ``init`` while ``rst`` is high, else latches the
    signal on ``d``; ``clk`` / ``rst`` must come from the global sources."""

    def __init__(self, netlist: PrimitiveNetlist) -> None:
        self.netlist = netlist
        driver: dict[BitTerminal, BitTerminal] = {}
        for net in netlist.nets:
            for sink in net.sinks:
                assert sink not in driver, f"{sink.describe()} has two drivers"
                driver[sink] = net.driver
        self.driver = driver
        instances = netlist.instances
        gates = {inst.id for inst in instances if inst.kind in GATE_KINDS}
        order: list[int] = []
        seen: set[int] = set()
        for root in sorted(gates):
            stack = [(root, False)]
            while stack:
                gid, finished = stack.pop()
                if finished:
                    order.append(gid)
                    continue
                if gid in seen:
                    continue
                seen.add(gid)
                stack.append((gid, True))
                for pin in instances[gid].inputs:
                    source = driver[BitTerminal(gid, pin)].instance
                    if source in gates and source not in seen:
                        stack.append((source, False))
        self.order = order
        self.registers = [inst for inst in instances if inst.kind is PrimitiveKind.REGISTER_BIT]
        for reg in self.registers:
            clk = instances[driver[BitTerminal(reg.id, "clk")].instance].kind
            rst = instances[driver[BitTerminal(reg.id, "rst")].instance].kind
            assert clk is PrimitiveKind.CLOCK_SOURCE and rst is PrimitiveKind.RESET_SOURCE

    def _sources(
        self, columns: Mapping[str, Sequence[int]], count: int, state: Mapping[int, int] | None
    ) -> dict[BitTerminal, int]:
        mask = (1 << count) - 1
        values: dict[BitTerminal, int] = {}
        for inst in self.netlist.instances:
            if inst.kind is PrimitiveKind.CONST0:
                values[BitTerminal(inst.id, "y")] = 0
            elif inst.kind is PrimitiveKind.CONST1:
                values[BitTerminal(inst.id, "y")] = mask
            elif inst.kind is PrimitiveKind.REGISTER_BIT:
                assert state is not None, "a sequential netlist needs a state"
                values[BitTerminal(inst.id, "q")] = mask if state[inst.id] else 0
        for port in self.netlist.ports:
            if port.direction != "input":
                continue
            column = [port.type.bits(v) for v in columns[port.name]]
            assert len(column) == count
            for i, terminal in enumerate(port.bits):
                values[terminal] = sum(1 << k for k, v in enumerate(column) if (v >> i) & 1)
        return values

    def _settle(self, values: dict[BitTerminal, int], mask: int) -> dict[BitTerminal, int]:
        instances, driver = self.netlist.instances, self.driver
        for gid in self.order:
            kind = instances[gid].kind
            a = values[driver[BitTerminal(gid, "a")]]
            if kind is PrimitiveKind.NOT:
                y = ~a & mask
            else:
                b = values[driver[BitTerminal(gid, "b")]]
                if kind is PrimitiveKind.AND:
                    y = a & b
                elif kind is PrimitiveKind.OR:
                    y = a | b
                else:
                    assert kind is PrimitiveKind.XOR
                    y = a ^ b
            values[BitTerminal(gid, "y")] = y
        return values

    def _outputs(self, values: Mapping[BitTerminal, int], count: int) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for port in self.netlist.ports:
            if port.direction != "output":
                continue
            raw = [0] * count
            for i, sink in enumerate(port.bits):
                bits = values[self.driver[sink]]
                while bits:
                    low = bits & -bits
                    raw[low.bit_length() - 1] |= 1 << i
                    bits ^= low
            result[port.name] = [port.type.number(v) for v in raw]
        return result

    def evaluate_many(self, columns: Mapping[str, Sequence[int]], count: int) -> dict[str, list[int]]:
        values = self._settle(self._sources(columns, count, None), (1 << count) - 1)
        return self._outputs(values, count)

    def reset_state(self) -> dict[int, int]:
        return {reg.id: int(bool(reg.init)) for reg in self.registers}

    def step(
        self, state: Mapping[int, int], inputs: Mapping[str, int], *, rst: int = 0
    ) -> tuple[dict[str, int], dict[int, int]]:
        values = self._settle(self._sources({k: [v] for k, v in inputs.items()}, 1, state), 1)
        outputs = {name: column[0] for name, column in self._outputs(values, 1).items()}
        nxt = {
            reg.id: int(bool(reg.init)) if rst else values[self.driver[BitTerminal(reg.id, "d")]] & 1
            for reg in self.registers
        }
        return outputs, nxt

    def ir_state(self, state: Mapping[int, int]) -> dict[int, int]:
        """Register-bit state -> IR register state, via the netlist's buses."""
        return {
            node: sum(state[bit.instance] << i for i, bit in enumerate(bus.bits))
            for node, bus in self.netlist.buses.items()
            if self.netlist.ir_nodes[node].op == "register"
        }


# -- stimulus ------------------------------------------------------------------------


@cache
def edge_values(typ: IRType) -> tuple[int, ...]:
    """Raw patterns of 0, +-1, +-2, +-3, INT_MIN, INT_MIN+1, INT_MAX, INT_MAX-1,
    UMAX, UMAX-1 and every power of two +-1 (and their negations)."""
    top = 1 << (typ.width - 1)
    raw = {0, 1, 2, 3, -1, -2, -3, top, top + 1, top - 1, top - 2, typ.mask, typ.mask - 1}
    for k in range(typ.width):
        p = 1 << k
        raw |= {p - 1, p, p + 1, -p, -p + 1, -p - 1}
    return tuple(sorted({v & typ.mask for v in raw}))


@cache
def core_values(typ: IRType) -> tuple[int, ...]:
    """A small edge set whose full cross product stays cheap at any width."""
    top, half = 1 << (typ.width - 1), 1 << (typ.width // 2)
    raw = {0, 1, 2, 3, 7, -1, -2, -3, -7, top, top + 1, top - 1, typ.mask - 1, half, half - 1, half + 1, -half}
    return tuple(sorted({v & typ.mask for v in raw}))


def biased(rng: random.Random, typ: IRType) -> int:
    """One raw pattern, heavily biased towards edge cases."""
    width = typ.width
    mode = rng.random()
    if mode < 0.3:
        return rng.choice(edge_values(typ))
    if mode < 0.5:
        small = rng.getrandbits(rng.randint(1, min(width, 6)))
        return (-small if rng.random() < 0.5 else small) & typ.mask
    if mode < 0.62:
        near = (1 << rng.randrange(width)) + rng.randint(-2, 2)
        return (-near if rng.random() < 0.5 else near) & typ.mask
    if mode < 0.72:
        density = rng.choice((0.1, 0.9))
        return sum(1 << i for i in range(width) if rng.random() < density)
    return rng.getrandbits(width)


def all_values(typ: IRType, rng: random.Random, limit_bits: int = 9, extra: int = 200) -> list[int]:
    """Every value of a narrow type, else the edge set plus biased randoms."""
    if typ.width <= limit_bits:
        return list(range(1 << typ.width))
    return list(edge_values(typ)) + [biased(rng, typ) for _ in range(extra)]


def wide_pairs(rng: random.Random, typ: IRType, count: int) -> list[tuple[int, int]]:
    """Core edge cross product, random edge pairs, biased pairs, equal and
    adjacent pairs, and exact multiples (+-1) for the dividers."""
    pairs = list(itertools.product(core_values(typ), repeat=2))
    edges = edge_values(typ)
    pairs += [(rng.choice(edges), rng.choice(edges)) for _ in range(count)]
    pairs += [(biased(rng, typ), biased(rng, typ)) for _ in range(count)]
    for _ in range(count // 4):
        a = biased(rng, typ)
        pairs += [(a, a), (a, (a + 1) & typ.mask), ((a + 1) & typ.mask, a)]
        divisor = biased(rng, typ) or 1
        multiple = typ.number(divisor) * typ.number(biased(rng, typ))
        pairs += [(multiple & typ.mask, divisor), ((multiple + rng.choice((-1, 1))) & typ.mask, divisor)]
    return pairs


def boundary_amounts(width: int) -> tuple[int, ...]:
    """0 .. W+2, the barrel's own range edge (``2**stages +- 1``) and huge
    uint64 amounts (2**31, 2**32, 2**63, 2**64-1, W + 2**32, ...)."""
    stages = (width - 1).bit_length()
    amounts = set(range(width + 3)) | set(range(max(0, (1 << stages) - 1), (1 << stages) + 2))
    amounts |= {63, 64, 65, 127, 128, 2**31, 2**32, 2**63, 2**64 - 1, width + 2**32, 2**64 - width}
    amounts |= {2**63 + width, 64 + width, 2**32 - 1}
    return tuple(sorted(a & (2**64 - 1) for a in amounts))


def shift_amounts(width: int) -> tuple[int, ...]:
    """:func:`boundary_amounts` plus every power of two and ``2**k - 1``."""
    powers = {1 << k for k in range(64)} | {(1 << k) - 1 for k in range(1, 65)}
    return tuple(sorted(set(boundary_amounts(width)) | {a & (2**64 - 1) for a in powers}))


# -- the combinational comparator -----------------------------------------------------

Oracle = Callable[[Sequence[int]], Mapping[str, int]]


def single(
    op: str, result: IRType, operands: Sequence[IRType], *, shared: bool = False
) -> tuple[Graph, tuple[str, ...]]:
    """``result = op(...)`` built with ``Graph.node``.  ``shared`` feeds ONE
    input node to every operand (``add(x, x)``, never folded by the IR)."""
    graph = Graph()
    if shared:
        assert len(set(operands)) == 1
        x = graph.input("in_a", operands[0])
        node = graph.node(op, result, (x,) * len(operands))
        names: tuple[str, ...] = ("in_a",)
    else:
        args = tuple(graph.input(name, typ) for name, typ in zip(NAMES, operands))
        node = graph.node(op, result, args)
        names = NAMES[: len(operands)]
    graph.output("result", node)
    return graph, names


def single_oracle(op: str, result: IRType, operands: Sequence[IRType], *, shared: bool = False) -> Oracle:
    """``Graph.evaluate`` of a one-node graph, minus the per-call overhead."""
    types = list(operands)

    def oracle(vector: Sequence[int]) -> Mapping[str, int]:
        values = [vector[0]] * len(types) if shared else list(vector)
        raw = [t.bits(v) for t, v in zip(types, values)]
        return {"result": result.number(_calculate(op, raw, result, types))}

    return oracle


def assert_combinational(
    graph: Graph,
    names: Sequence[str],
    vectors: Iterable[Sequence[int]],
    *,
    label: str,
    oracle: Oracle | None = None,
    policy: InterfacePolicy = PADS,
    independent: bool = False,
) -> PrimitiveNetlist:
    """Simulate every vector bit-parallel and compare each output with the IR."""
    netlist = synth(graph, policy)
    rows = [tuple(v) for v in vectors]
    columns = {name: [row[k] for row in rows] for k, name in enumerate(names)}
    got = PrimitiveSimulator(netlist).evaluate_many(columns)
    cross = IndependentEvaluator(netlist).evaluate_many(columns, len(rows)) if independent else None
    mismatches: list[str] = []
    for index, row in enumerate(rows):
        if oracle is None or index < ORACLE_SAMPLE:
            expected = graph.evaluate(**dict(zip(names, row)))
            if oracle is not None:
                assert dict(oracle(row)) == expected, f"{label}: test oracle disagrees with Graph.evaluate at {row}"
        else:
            expected = oracle(row)
        for port, value in expected.items():
            if got[port][index] != value:
                mismatches.append(f"{dict(zip(names, row))}: {port} primitive={got[port][index]} IR={value}")
            if cross is not None and cross[port][index] != value:
                mismatches.append(f"{dict(zip(names, row))}: {port} independent={cross[port][index]} IR={value}")
    assert not mismatches, f"{label}: {len(mismatches)} mismatches over {len(rows)} vectors, e.g. {mismatches[:6]}"
    return netlist


# -- registry --------------------------------------------------------------------------


def test_default_registry_covers_exactly_the_ir_ops() -> None:
    assert DEFAULT_SYNTHESIS.ops() == set(OPS)


# -- binary ops: exhaustive -------------------------------------------------------------

SMALL_TYPES = int_types(range(1, 8))


@pytest.mark.parametrize("typ", SMALL_TYPES, ids=type_name)
@pytest.mark.parametrize("op", BINARY)
def test_binary_op_exhaustive_widths_1_to_7(op: str, typ: IRType) -> None:
    result = result_of(op, typ)
    graph, names = single(op, result, (typ, typ))
    values = range(1 << typ.width)
    assert_combinational(
        graph,
        names,
        itertools.product(values, repeat=2),
        label=f"{op}({typ.name}, {typ.name})",
        independent=typ.width <= 5,
    )


@pytest.mark.parametrize("signed", (False, True), ids=("uint9", "int9"))
@pytest.mark.parametrize("op", ("div", "mod"))
def test_dividers_exhaustive_width_9(op: str, signed: bool) -> None:
    typ = IRType(9, signed=signed)
    result = result_of(op, typ)
    graph, names = single(op, result, (typ, typ))
    assert_combinational(
        graph,
        names,
        itertools.product(range(512), repeat=2),
        label=f"{op}({typ.name}, {typ.name})",
        oracle=single_oracle(op, result, (typ, typ)),
    )


# -- binary ops: wide, edge-biased ------------------------------------------------------

WIDE_WIDTHS = (8, 9, 16, 31, 32, 33, 63, 64)


@pytest.mark.parametrize("signed", (False, True), ids=("unsigned", "signed"))
@pytest.mark.parametrize("width", WIDE_WIDTHS)
@pytest.mark.parametrize("op", BINARY)
def test_binary_op_wide_edge_biased(op: str, width: int, signed: bool) -> None:
    typ = IRType(width, signed=signed)
    rng = random.Random(f"binary/{op}/{typ.name}")
    result = result_of(op, typ)
    graph, names = single(op, result, (typ, typ))
    assert_combinational(
        graph,
        names,
        wide_pairs(rng, typ, 120 if op in HEAVY and width > 32 else 300),
        label=f"{op}({typ.name}, {typ.name})",
        independent=width in (9, 33),
    )


# -- one IR node feeding several operands ----------------------------------------------

SHARED_TYPES = int_types((1, 2, 3, 5, 8, 9, 16, 33, 64))
#: The quadratic ops stop at 33 bits here (a 64-bit divider costs a second).
SHARED_CASES = [
    (op, typ) for op in BINARY for typ in SHARED_TYPES if op not in HEAVY or typ.width <= 33
]


@pytest.mark.parametrize(("op", "typ"), SHARED_CASES, ids=[f"{op}-{t.name}" for op, t in SHARED_CASES])
def test_binary_op_with_one_node_as_both_operands(op: str, typ: IRType) -> None:
    rng = random.Random(f"shared/{op}/{typ.name}")
    result = result_of(op, typ)
    graph, names = single(op, result, (typ, typ), shared=True)
    assert_combinational(
        graph,
        names,
        [(v,) for v in all_values(typ, rng)],
        label=f"{op}(x, x) over {typ.name}",
        independent=True,
    )


@pytest.mark.parametrize("op", SHIFTS)
def test_uint64_shift_by_itself(op: str) -> None:
    """``shl(x, x)`` / ``shr(x, x)`` -- the only typing where the value and the
    uint64 amount can be the very same node."""
    rng = random.Random(f"self-shift/{op}")
    values = sorted(set(range(70)) | set(edge_values(UINT64)) | {biased(rng, UINT64) for _ in range(200)})
    graph, names = single(op, UINT64, (UINT64, UINT64), shared=True)
    assert_combinational(graph, names, [(v,) for v in values], label=f"{op}(x, x) over uint64", independent=True)


@pytest.mark.parametrize("typ", (UINT8, INT8, INT3, IRType(13, signed=True), INT64, UINT64), ids=type_name)
def test_shift_by_a_cast_of_the_shifted_value(typ: IRType) -> None:
    """The amount is the value itself, widened to uint64 (a negative signed
    value sign-extends into a huge amount)."""
    rng = random.Random(f"cast-shift/{typ.name}")
    graph = Graph()
    x = graph.input("in_a", typ)
    amount = graph.node("cast", SHIFT_AMOUNT, (x,))
    graph.output("left", graph.node("shl", typ, (x, amount)))
    graph.output("right", graph.node("shr", typ, (x, amount)))
    assert_combinational(graph, ["in_a"], [(v,) for v in all_values(typ, rng)], label=f"shift {typ.name} by itself")


# -- comparisons on equal / adjacent operands ----------------------------------------------

COMPARE_TYPES = int_types((1, 2, 3, 4, 7, 8, 9, 16, 31, 32, 33, 63, 64))


@pytest.mark.parametrize("typ", COMPARE_TYPES, ids=type_name)
def test_comparisons_on_equal_and_adjacent_values(typ: IRType) -> None:
    rng = random.Random(f"compare/{typ.name}")
    graph = Graph()
    a, b = graph.input("in_a", typ), graph.input("in_b", typ)
    for op in COMPARISONS:
        graph.output(op, graph.node(op, BOOL, (a, b)))
    values = list(edge_values(typ)) + [biased(rng, typ) for _ in range(100)]
    vectors = []
    for v in values:
        for delta in (0, 1, -1, 2):
            vectors += [(v, (v + delta) & typ.mask), ((v + delta) & typ.mask, v)]
        vectors.append((v, v ^ (1 << (typ.width - 1))))  # differ only in the sign bit
    assert_combinational(graph, ["in_a", "in_b"], vectors, label=f"comparisons over {typ.name}", independent=True)


# -- unary ops -----------------------------------------------------------------------------

UNARY_TYPES = int_types((1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 31, 32, 33, 63, 64))


@pytest.mark.parametrize("typ", UNARY_TYPES, ids=type_name)
def test_unary_ops(typ: IRType) -> None:
    rng = random.Random(f"unary/{typ.name}")
    graph = Graph()
    x = graph.input("in_a", typ)
    graph.output("neg", graph.node("neg", typ, (x,)))
    graph.output("inv", graph.node("inv", typ, (x,)))
    if typ == BOOL:
        graph.output("not", graph.node("not", BOOL, (x,)))
    values = all_values(typ, rng, extra=400)
    assert_combinational(graph, ["in_a"], [(v,) for v in values], label=f"neg/inv over {typ.name}", independent=True)


# -- shifts ----------------------------------------------------------------------------------

SHIFT_TYPES = int_types((1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 31, 32, 33, 63, 64))


@pytest.mark.parametrize("typ", SHIFT_TYPES, ids=type_name)
@pytest.mark.parametrize("op", SHIFTS)
def test_shift_every_boundary_amount(op: str, typ: IRType) -> None:
    rng = random.Random(f"shift/{op}/{typ.name}")
    amounts = shift_amounts(typ.width)
    # The core values meet EVERY amount; the rest meet the boundary amounts
    # (narrow types: every value) or random amounts.
    vectors = list(itertools.product(core_values(typ), amounts))
    if typ.width <= 9:
        vectors += itertools.product(range(1 << typ.width), boundary_amounts(typ.width))
    else:
        vectors += [(biased(rng, typ), rng.choice(amounts)) for _ in range(1500)]
    vectors += [(biased(rng, typ), rng.getrandbits(64)) for _ in range(200)]
    graph, names = single(op, typ, (typ, SHIFT_AMOUNT))
    assert_combinational(
        graph, names, vectors, label=f"{op}({typ.name}, uint64)", independent=typ.width in (5, 33)
    )


# -- casts -------------------------------------------------------------------------------------

CAST_TYPES = (BOOL, UINT1, INT1, UINT3, INT3, UINT8, INT8, UINT16, INT16, UINT64, INT64)


@pytest.mark.parametrize("target", CAST_TYPES, ids=type_name)
@pytest.mark.parametrize("source", CAST_TYPES, ids=type_name)
def test_cast_every_pair(source: IRType, target: IRType) -> None:
    rng = random.Random(f"cast/{source.name}/{target.name}")
    graph, names = single("cast", target, (source,))
    values = all_values(source, rng, limit_bits=8, extra=100)
    assert_combinational(
        graph,
        names,
        [(v,) for v in values],
        label=f"cast {source.name} -> {target.name}",
        independent=True,
    )


def test_cast_chains_through_every_type() -> None:
    """``x -> t1 -> t2 -> t3`` for many deterministic type chains in ONE graph
    (reinterpreted / extended bits feed further casts)."""
    rng = random.Random("cast-chains")
    graph = Graph()
    x8 = graph.input("in_a", INT8)
    x64 = graph.input("in_b", UINT64)
    for k in range(60):
        value = x8 if k % 2 else x64
        for _ in range(3):
            value = graph.node("cast", rng.choice(CAST_TYPES), (value,))
        graph.output(f"chain{k}", value)
    vectors = [(a, rng.choice(edge_values(UINT64))) for a in range(256)]
    vectors += [(biased(rng, INT8), biased(rng, UINT64)) for _ in range(300)]
    assert_combinational(graph, ["in_a", "in_b"], vectors, label="cast chains", independent=True)


# -- mux ---------------------------------------------------------------------------------------

MUX_TYPES = int_types((1, 2, 3, 4, 5, 7, 8, 9, 16, 31, 32, 33, 63, 64))


@pytest.mark.parametrize("typ", MUX_TYPES, ids=type_name)
def test_mux_every_type(typ: IRType) -> None:
    rng = random.Random(f"mux/{typ.name}")
    if typ.width <= 3:
        pairs = list(itertools.product(range(1 << typ.width), repeat=2))
    else:
        pairs = list(itertools.product(core_values(typ), repeat=2)) + [
            (biased(rng, typ), biased(rng, typ)) for _ in range(300)
        ]
    graph, names = single("mux", typ, (BOOL, typ, typ))
    vectors = [(sel, a, b) for sel in (0, 1) for a, b in pairs]
    assert_combinational(graph, names, vectors, label=f"mux(bool, {typ.name}, {typ.name})", independent=True)


@pytest.mark.parametrize("typ", MUX_TYPES, ids=type_name)
def test_mux_with_shared_operand_nodes(typ: IRType) -> None:
    """``mux(s, x, x)`` (never folded by ``Graph.node``), plus for ``bool``
    every way the select can also be a data operand."""
    rng = random.Random(f"mux-shared/{typ.name}")
    graph = Graph()
    sel, x = graph.input("in_a", BOOL), graph.input("in_b", typ)
    graph.output("same_branches", graph.node("mux", typ, (sel, x, x)))
    if typ == BOOL:
        graph.output("all_select", graph.node("mux", BOOL, (sel, sel, sel)))
        graph.output("select_yes", graph.node("mux", BOOL, (sel, sel, x)))
        graph.output("select_no", graph.node("mux", BOOL, (sel, x, sel)))
        graph.output("x_selects", graph.node("mux", BOOL, (x, sel, x)))
    vectors = [(s, v) for s in (0, 1) for v in all_values(typ, rng, extra=100)]
    assert_combinational(graph, ["in_a", "in_b"], vectors, label=f"shared mux over {typ.name}", independent=True)


# -- random typed DAGs ----------------------------------------------------------------------------

FUZZ_TYPES = (
    *int_types((1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16)),
    IRType(31),
    IRType(32, signed=True),
    IRType(33, signed=True),
    IRType(63),
    UINT64,
    INT64,
)
#: Widest type the quadratic ops (mul / div / mod) may use inside a fuzzed DAG.
FUZZ_HEAVY_LIMIT = 16


def fit(graph: Graph, value: Value, typ: IRType) -> Value:
    """``value`` itself if it already has type ``typ``, else a cast node."""
    return value if value.type == typ else graph.node("cast", typ, (value,))


def fuzz_node(graph: Graph, pool: Sequence[Value], rng: random.Random) -> Value:
    """One random well-typed node over ``pool`` (any IR op or a constant),
    built with ``Graph.node``; casts are inserted where a type does not fit."""

    def pick() -> Value:  # favour recent values: deep chains, not just wide fans
        return pool[-1 - min(int(rng.expovariate(0.5)), len(pool) - 1)]

    def pick_type(heavy: bool) -> IRType:
        typ = pick().type if rng.random() < 0.6 else rng.choice(FUZZ_TYPES)
        if heavy and typ.width > FUZZ_HEAVY_LIMIT:
            typ = IRType(rng.choice((5, 8, 13, 16)), signed=typ.signed)
        return typ

    if rng.random() < 0.1:
        typ = rng.choice(FUZZ_TYPES)
        return graph.constant(biased(rng, typ), typ)
    op = rng.choice(sorted(OPS))
    if op == "cast":
        return graph.node("cast", rng.choice(FUZZ_TYPES), (pick(),))
    if op == "not":
        return graph.node("not", BOOL, (fit(graph, pick(), BOOL),))
    if op in ("neg", "inv"):
        typ = pick_type(False)
        return graph.node(op, typ, (fit(graph, pick(), typ),))
    if op == "mux":
        typ = pick_type(False)
        operands = (fit(graph, pick(), BOOL), fit(graph, pick(), typ), fit(graph, pick(), typ))
        return graph.node("mux", typ, operands)
    if op in SHIFTS:
        typ = pick_type(False)
        if rng.random() < 0.3:
            amount = graph.constant(rng.randrange(typ.width + 3), SHIFT_AMOUNT)
        else:
            amount = fit(graph, pick(), SHIFT_AMOUNT)
        return graph.node(op, typ, (fit(graph, pick(), typ), amount))
    if op in COMPARISONS:
        typ = pick_type(False)
        return graph.node(op, BOOL, (fit(graph, pick(), typ), fit(graph, pick(), typ)))
    typ = pick_type(op in HEAVY)
    return graph.node(op, typ, (fit(graph, pick(), typ), fit(graph, pick(), typ)))


def _fuzz_inputs(graph: Graph, rng: random.Random, low: int, high: int) -> list[tuple[str, IRType]]:
    ports = [(f"in_{k}", rng.choice(FUZZ_TYPES)) for k in range(rng.randint(low, high))]
    for name, typ in ports:
        graph.input(name, typ)
    return ports


def _fuzz_outputs(graph: Graph, pool: Sequence[Value], rng: random.Random, extra: int) -> None:
    observed = {pool[-1].id: pool[-1]}
    for _ in range(rng.randint(0, extra)):
        value = rng.choice(pool)
        observed.setdefault(value.id, value)
    for k, value in enumerate(observed.values()):
        graph.output(f"out_{k}", value)


def random_dag(rng: random.Random) -> tuple[Graph, list[tuple[str, IRType]]]:
    """A random well-typed combinational DAG over every IR op."""
    graph = Graph()
    ports = _fuzz_inputs(graph, rng, 1, 4)
    pool = [Value(port["node"], IRType(**port["type"])) for port in graph.inputs]
    for _ in range(rng.randint(4, 14)):
        pool.append(fuzz_node(graph, pool, rng))
    _fuzz_outputs(graph, pool, rng, 3)
    return graph, ports


def random_sequential_graph(rng: random.Random) -> tuple[Graph, list[tuple[str, IRType]]]:
    """Registers (random types, edge-biased signed inits) declared BEFORE the
    random logic that reads them; each next / enable is then a random LATER
    node (a cast is added where needed) or a constant-0 / constant-1 enable."""
    graph = Graph()
    ports = _fuzz_inputs(graph, rng, 0, 3)
    pool = [Value(port["node"], IRType(**port["type"])) for port in graph.inputs]
    registers = []
    for _ in range(rng.randint(1, 4)):
        typ = rng.choice(FUZZ_TYPES)
        registers.append(graph.register(typ, init=typ.number(biased(rng, typ))))
    pool += registers
    for _ in range(rng.randint(3, 12)):
        pool.append(fuzz_node(graph, pool, rng))
    for register in registers:
        nxt = fit(graph, rng.choice(pool), register.type)
        choice = rng.random()
        if choice < 0.15:
            enable = graph.constant(1, BOOL)
        elif choice < 0.25:
            enable = graph.constant(0, BOOL)
        else:
            enable = fit(graph, rng.choice(pool), BOOL)
        graph.set_register(register, nxt, enable)
    _fuzz_outputs(graph, [*pool, *registers], rng, 4)
    return graph, ports


@pytest.mark.parametrize("seed", range(160))
def test_random_dag(seed: int) -> None:
    rng = random.Random(f"dag/{seed}")
    graph, ports = random_dag(rng)
    vectors = [tuple(biased(rng, t) for _, t in ports) for _ in range(48)]
    vectors += [tuple(0 for _ in ports), tuple(t.mask for _, t in ports)]
    assert_combinational(
        graph, [name for name, _ in ports], vectors, label=f"random DAG {seed}", independent=True
    )


def test_single_vector_evaluate_agrees_with_evaluate_many() -> None:
    rng = random.Random("single-evaluate")
    for _ in range(12):
        graph, ports = random_dag(rng)
        sim = PrimitiveSimulator(synth(graph))
        for _ in range(8):
            inputs = {name: biased(rng, t) for name, t in ports}
            assert sim.evaluate(**inputs) == graph.evaluate(**inputs), inputs


def test_a_detailed_trace_does_not_change_the_circuit() -> None:
    rng = random.Random("trace")
    graph, _ports = random_dag(rng)
    plain = synth(graph)
    traced = synthesize_to_primitives(graph, interface=PADS, trace=PrimitiveTraceRecorder(TraceLevel.DETAILED))
    assert traced.to_dict() == plain.to_dict()


# -- sequential: hand-built graphs ---------------------------------------------------------------

Stimulus = Callable[[random.Random, int], dict[str, int]]


def assert_steps_like_ir(
    graph: Graph,
    stimulus: Stimulus,
    *,
    cycles: int,
    seed: str,
    resets: Sequence[int] = (),
    policy: InterfacePolicy = PADS,
    independent: bool = True,
) -> None:
    """Step the IR, the simulator and (unless ``independent=False``, for very
    large netlists) the independent evaluator in lockstep.  On a ``resets``
    cycle the netlist gets ``rst`` high: its outputs are still this cycle's,
    and the next state is every register's ``init``."""
    netlist = synth(graph, policy)
    sim = PrimitiveSimulator(netlist)
    ref = IndependentEvaluator(netlist) if independent else None
    ir_state, state = graph.reset_state(), sim.reset_state()
    assert sim.state_to_ir(state) == ir_state
    assert sim.state_from_ir(ir_state) == state
    ref_state = ref.reset_state() if ref is not None else {}
    assert ref is None or ref.ir_state(ref_state) == ir_state
    rng = random.Random(seed)
    for cycle in range(cycles):
        inputs = stimulus(rng, cycle)
        rst = int(cycle in resets)
        ir_outputs, ir_next = graph.step(ir_state, **inputs)
        if rst:
            ir_next = graph.reset_state()
        outputs, nxt = sim.step(state, rst=rst, **inputs)
        wrong = {k: (outputs.get(k), v) for k, v in ir_outputs.items() if outputs.get(k) != v}
        assert not wrong, f"cycle {cycle} inputs {inputs}: (primitive, IR) outputs differ {wrong}"
        assert sim.state_to_ir(nxt) == ir_next, f"cycle {cycle}: state {sim.state_to_ir(nxt)} vs IR {ir_next}"
        assert sim.state_from_ir(ir_next) == nxt
        if ref is not None:
            ref_outputs, ref_state = ref.step(ref_state, inputs, rst=rst)
            assert ref_outputs == ir_outputs, f"cycle {cycle}: independent {ref_outputs} vs IR {ir_outputs}"
            assert ref.ir_state(ref_state) == ir_next, f"cycle {cycle}: independent state differs from the IR"
        ir_state, state = ir_next, nxt


def _backedge_graph() -> Graph:
    """Registers created BEFORE the nodes their next / enable reference, an
    enable driven by the register itself, constant-0 / constant-1 enables, a
    self-holding register and signed (INT_MIN, negative) init values."""
    g = Graph()
    x = g.input("in_x", INT8)
    en = g.input("in_en", BOOL)
    k = g.input("in_k", UINT3)
    acc = g.register(INT8, init=-5)
    count = g.register(UINT4, init=9)
    toggle = g.register(BOOL, init=1)
    wide = g.register(INT64, init=-(2**63))
    big = g.register(UINT64, init=2**64 - 2)
    held = g.register(INT16, init=-300)
    parity = g.register(BOOL, init=0)
    under = g.node("lt", BOOL, (acc, g.constant(50, INT8)))
    g.set_register(acc, g.node("add", INT8, (acc, x)), g.node("and", BOOL, (under, en)))
    g.set_register(count, g.node("sub", UINT4, (count, g.constant(1, UINT4))), g.constant(1, BOOL))
    g.set_register(toggle, g.node("not", BOOL, (toggle,)), toggle)
    widened = g.node("add", INT64, (wide, g.node("cast", INT64, (x,))))
    g.set_register(wide, g.node("shr", INT64, (widened, g.node("cast", SHIFT_AMOUNT, (k,)))), en)
    g.set_register(big, g.node("mul", UINT64, (big, g.node("cast", UINT64, (x,)))), g.constant(0, BOOL))
    g.set_register(held, held, en)
    g.set_register(parity, g.node("xor", BOOL, (parity, g.node("cast", BOOL, (count,)))), g.node("ne", BOOL, (x, acc)))
    for name, value in (
        ("acc", acc),
        ("count", count),
        ("toggle", toggle),
        ("wide", wide),
        ("big", big),
        ("held", held),
        ("parity", parity),
    ):
        g.output(name, value)
    g.output("mealy", g.node("xor", INT8, (acc, x)))  # depends on this cycle's input
    g.output("ratio", g.node("div", INT8, (acc, x)))
    return g


def _backedge_stimulus(rng: random.Random, cycle: int) -> dict[str, int]:
    return {"in_x": biased(rng, INT8), "in_en": int(rng.random() < 0.6), "in_k": rng.randrange(8)}


def test_registers_with_back_edges_constant_enables_and_signed_inits() -> None:
    assert_steps_like_ir(
        _backedge_graph(), _backedge_stimulus, cycles=120, seed="backedge", resets=(37, 38, 101)
    )


@pytest.mark.parametrize("seed", range(40))
def test_random_sequential_graph(seed: int) -> None:
    rng = random.Random(f"sequential/{seed}")
    graph, ports = random_sequential_graph(rng)

    def stimulus(rng: random.Random, cycle: int) -> dict[str, int]:
        return {name: biased(rng, typ) for name, typ in ports}

    resets = [cycle for cycle in range(40) if rng.random() < 0.06]
    assert_steps_like_ir(graph, stimulus, cycles=40, seed=f"sequential-steps/{seed}", resets=resets)


def _chain_graph() -> Graph:
    """Registers feeding registers: a pipeline closed by a back-edge to a LATER
    register, a swap pair, a register visible only through another one,
    constant next values, next == enable, and one register on two outputs."""
    g = Graph()
    x = g.input("in_x", UINT8)
    en = g.input("in_en", BOOL)
    one, zero = g.constant(1, BOOL), g.constant(0, BOOL)
    loop = g.register(UINT8, init=0x11)
    r1 = g.register(UINT8, init=1)
    r2 = g.register(UINT8, init=2)
    r3 = g.register(UINT8, init=3)
    g.set_register(r1, g.node("add", UINT8, (x, loop)), one)
    g.set_register(r2, r1, one)
    g.set_register(r3, r2, en)
    g.set_register(loop, r3, one)  # back-edge to a register created later
    s1 = g.register(INT16, init=-1)
    s2 = g.register(INT16, init=-32768)
    g.set_register(s1, s2, en)
    g.set_register(s2, s1, en)
    hidden = g.register(UINT8, init=0x5A)
    seen = g.register(UINT8, init=0)
    g.set_register(hidden, g.node("xor", UINT8, (hidden, x)), en)
    g.set_register(seen, g.node("add", UINT8, (seen, hidden)), one)
    fixed = g.register(UINT8, init=0xFF)
    g.set_register(fixed, g.constant(0x42, UINT8), en)
    frozen = g.register(INT8, init=-128)
    g.set_register(frozen, g.node("cast", INT8, (x,)), zero)
    flag = g.register(BOOL, init=0)
    g.set_register(flag, en, en)
    g.output("r3", r3)
    g.output("swap_diff", g.node("sub", INT16, (s1, s2)))
    g.output("s1", s1)
    g.output("seen", seen)
    g.output("fixed", fixed)
    g.output("frozen", frozen)
    g.output("flag", flag)
    g.output("r1", r1)
    g.output("r1_again", r1)
    return g


def test_register_chains_swaps_and_hidden_state() -> None:
    def stimulus(rng: random.Random, cycle: int) -> dict[str, int]:
        return {"in_x": biased(rng, UINT8), "in_en": int(rng.random() < 0.5)}

    assert_steps_like_ir(_chain_graph(), stimulus, cycles=200, seed="chain", resets=(0, 77))


def _handshake_graph() -> Graph:
    """A hand-built start/done FSM: ``acc = a * n`` by repeated signed addition."""
    g = Graph()
    start = g.input("start", BOOL)
    a = g.input("in_a", INT16)
    n = g.input("in_n", UINT4)
    running = g.register(BOOL, init=0)
    acc = g.register(INT16, init=-1)
    left = g.register(UINT4, init=0)
    accept = g.node("and", BOOL, (start, g.node("not", BOOL, (running,))))
    finished = g.node("eq", BOOL, (left, g.constant(0, UINT4)))
    busy = g.node("and", BOOL, (running, g.node("not", BOOL, (finished,))))
    step = g.node("or", BOOL, (accept, busy))
    g.set_register(acc, g.node("mux", INT16, (accept, g.constant(0, INT16), g.node("add", INT16, (acc, a)))), step)
    left_next = g.node("mux", UINT4, (accept, n, g.node("sub", UINT4, (left, g.constant(1, UINT4)))))
    g.set_register(left, left_next, step)
    g.set_register(running, step, g.constant(1, BOOL))
    g.output("result", g.node("cast", UINT8, (acc,)))  # a uint8 result: the 7-segment display
    g.output("full", acc)
    g.output("done", g.node("and", BOOL, (running, finished)))
    return g


@pytest.mark.parametrize("policy", (PADS, DefaultInterfacePolicy()), ids=("pads", "peripherals"))
def test_handshake_fsm_runs_and_steps_like_the_ir(policy: InterfacePolicy) -> None:
    graph = _handshake_graph()
    sim = PrimitiveSimulator(synth(graph, policy))
    rng = random.Random("handshake")
    for _ in range(25):
        inputs = {"in_a": biased(rng, INT16), "in_n": rng.randrange(16)}
        assert sim.run(**inputs) == graph.run(**inputs), inputs

    def stimulus(rng: random.Random, cycle: int) -> dict[str, int]:
        return {"start": int(rng.random() < 0.2), "in_a": biased(rng, INT16), "in_n": rng.randrange(16)}

    assert_steps_like_ir(graph, stimulus, cycles=150, seed="handshake-steps", resets=(60,), policy=policy)


# -- compiled RedC programs ------------------------------------------------------------------------

GCD = """
int16 main(int16 a, int16 b) {
    int16 x = a;
    int16 y = b;
    while (y != 0) {
        int16 t = x % y;
        x = y;
        y = t;
    }
    return x;
}
"""

COLLATZ = """
uint8 main(uint16 n) {
    uint8 steps = 0;
    uint16 x = n;
    while (x > 1 && steps < 40) {
        if ((x & 1) == 1) { x = x * 3 + 1; } else { x = x >> 1; }
        steps = steps + 1;
    }
    return steps;
}
"""

MIXED = """
int32 main(int32 a, uint8 n) {
    int32 acc = a;
    for (uint8 i = 0; i < n; i = i + 1) {
        acc = acc * 3 - acc / 7 + (int32)i;
        if (acc > 100000 || acc < -100000) { acc = acc % 9973; }
    }
    return acc;
}
"""

COMBINATIONAL = """
int16 main(int16 a, int16 b, uint8 s) {
    int16 r = 0;
    if (b != 0) { r = a / b + a % b; }
    r = r ^ (a >> s) ^ (int16)((uint16)b << s);
    if (a < b) { r = -r; }
    return r;
}
"""


def _program_inputs(graph: Graph, rng: random.Random, small: Mapping[str, int]) -> dict[str, int]:
    inputs = {}
    for port in graph.inputs:
        if port["name"] == "start":
            continue
        typ = IRType(**port["type"])
        bound = small.get(port["name"])
        inputs[port["name"]] = rng.randrange(bound) if bound else biased(rng, typ)
    return inputs


@pytest.mark.parametrize(
    ("source", "small", "cycles"),
    [(GCD, {}, 90), (COLLATZ, {}, 90), (MIXED, {"in_n": 9}, 50)],
    ids=("gcd", "collatz", "mixed"),
)
def test_compiled_sequential_programs(source: str, small: Mapping[str, int], cycles: int) -> None:
    graph = compile_source(source)
    assert graph.sequential
    sim = PrimitiveSimulator(synth(graph))
    rng = random.Random(f"program/{source[:40]}")
    for _ in range(14):
        inputs = _program_inputs(graph, rng, small)
        assert sim.run(**inputs) == graph.run(**inputs), inputs

    def stimulus(rng: random.Random, cycle: int) -> dict[str, int]:
        return {**_program_inputs(graph, rng, small), "start": int(rng.random() < 0.15)}

    # ``mixed`` is ~40k gates (32-bit mul, div and mod): simulator vs IR only.
    assert_steps_like_ir(
        graph, stimulus, cycles=cycles, seed="program-steps", resets=(cycles // 2,), independent=source != MIXED
    )


def test_compiled_combinational_program() -> None:
    graph = compile_source(COMBINATIONAL)
    rng = random.Random("combinational-program")
    vectors = [(biased(rng, INT16), biased(rng, INT16), rng.choice((0, 1, 7, 15, 16, 17, 255))) for _ in range(1500)]
    vectors += [(a, b, s) for a in core_values(INT16) for b in core_values(INT16) for s in (0, 3, 16)]
    assert_combinational(graph, ["in_a", "in_b", "in_s"], vectors, label="compiled program", independent=True)


# -- sequential graphs without any input -------------------------------------------------------


def _free_running_graph() -> Graph:
    """No inputs at all: an int3 counter (signed init) with a constant-1
    enable, an int1 negated every cycle, and a uint64 shifted by the counter
    while ``counter < 2`` -- ``done`` when the counter reaches 3."""
    g = Graph()
    count = g.register(INT3, init=-4)
    sign = g.register(INT1, init=-1)
    wide = g.register(UINT64, init=2**64 - 1)
    one = g.constant(1, BOOL)
    g.set_register(count, g.node("add", INT3, (count, g.constant(1, INT3))), one)
    g.set_register(sign, g.node("neg", INT1, (sign,)), one)
    shifted = g.node("shl", UINT64, (wide, g.node("cast", SHIFT_AMOUNT, (count,))))
    g.set_register(wide, shifted, g.node("lt", BOOL, (count, g.constant(2, INT3))))
    g.output("count", count)
    g.output("sign", sign)
    g.output("wide", wide)
    g.output("done", g.node("eq", BOOL, (count, g.constant(3, INT3))))
    return g


def test_input_free_sequential_graph_runs_and_steps_like_the_ir() -> None:
    graph = _free_running_graph()
    assert PrimitiveSimulator(synth(graph)).run() == graph.run()
    assert_steps_like_ir(graph, lambda rng, cycle: {}, cycles=40, seed="free", resets=(13,))


# -- default interface policy: lever input, 7-segment output ---------------------------------------


@pytest.mark.parametrize("variant", ("const", "zero_extend", "sign_extend", "lever_logic", "input_alias"))
def test_default_policy_peripherals_compute_like_the_ir(variant: str) -> None:
    """``start: bool`` becomes a lever and ``result: uint8`` a display whose
    eight pins may share drivers (constants, extension copies, the lever)."""
    graph = Graph()
    start = graph.input("start", BOOL)
    x = graph.input("in_x", INT3)
    byte = graph.input("in_b", UINT8)
    if variant == "const":
        result = graph.constant(0xA5, UINT8)
    elif variant == "zero_extend":
        result = graph.node("cast", UINT8, (start,))
    elif variant == "sign_extend":
        result = graph.node("cast", UINT8, (x,))
    elif variant == "lever_logic":
        chosen = graph.node("mux", BOOL, (start, start, graph.node("cast", BOOL, (x,))))
        result = graph.node("add", UINT8, (graph.node("cast", UINT8, (chosen,)), byte))
    else:
        result = byte
    graph.output("result", result)
    graph.output("echo", start)
    vectors = [(s, v, b) for s in (0, 1) for v in range(8) for b in (0, 1, 0x7F, 0x80, 0xFE, 0xFF)]
    netlist = assert_combinational(
        graph, ["start", "in_x", "in_b"], vectors, label=f"default policy {variant}", policy=DefaultInterfacePolicy(), independent=True
    )
    assert netlist.port("result").realization == "2-dig-7-seg" and netlist.port("start").realization == "lever"


def test_out_of_range_python_ints_are_masked_like_the_ir() -> None:
    graph = Graph()
    a, s = graph.input("in_a", UINT8), graph.input("in_s", INT8)
    graph.output("sum", graph.node("add", UINT8, (a, a)))
    graph.output("neg", graph.node("neg", INT8, (s,)))
    graph.output("lt", graph.node("lt", BOOL, (s, graph.node("cast", INT8, (a,)))))
    sim = PrimitiveSimulator(synth(graph))
    for a_value, s_value in ((-1, 300), (2**70 + 3, -129), (-256, 2**64), (255, -1), (256, 128)):
        assert sim.evaluate(in_a=a_value, in_s=s_value) == graph.evaluate(in_a=a_value, in_s=s_value)


# -- port-name ambiguity in hand-built graphs ---------------------------------------------------
#
# ``compile_source`` refuses colliding port names ("flattened input port names
# collide"), but ``Graph.validate`` accepts them and so does primitive
# synthesis.  Either outcome is acceptable here -- reject the ambiguous
# interface with a CompileError, or simulate exactly like the IR -- but a
# silently different answer is not.


def test_duplicate_input_port_names_are_rejected_or_simulated_like_the_ir() -> None:
    graph = Graph()
    wide = graph.input("in_a", UINT8)
    signed = graph.input("in_a", INT8)  # same port name, different type: a second node
    graph.output("result", graph.node("add", UINT8, (wide, graph.node("cast", UINT8, (signed,)))))
    graph.validate()
    assert graph.evaluate(in_a=5) == {"result": 10}  # the IR feeds BOTH nodes
    try:
        netlist = synth(graph)
    except CompileError:
        return
    assert [port.name for port in netlist.ports if port.direction == "input"] == ["in_a", "in_a"]
    assert PrimitiveSimulator(netlist).evaluate(in_a=5) == graph.evaluate(in_a=5), (
        "two LogicalPorts are both named 'in_a'; PrimitiveSimulator keys ports by name, so one "
        "port's bits are never driven (they settle to 0) instead of the ambiguity being rejected"
    )


def test_duplicate_done_ports_are_rejected_or_run_like_the_ir() -> None:
    graph = Graph()
    phase = graph.register(BOOL, init=0)
    graph.set_register(phase, graph.node("not", BOOL, (phase,)), graph.constant(1, BOOL))
    graph.output("done", phase)  # Graph.run waits on the FIRST "done" port
    graph.output("done", graph.node("not", BOOL, (phase,)))
    graph.validate()
    try:
        netlist = synth(graph)
    except CompileError:
        return
    assert PrimitiveSimulator(netlist).run(max_cycles=10) == graph.run(max_cycles=10), (
        "Graph.run waits on the FIRST 'done' port, PrimitiveSimulator.run on the LAST one "
        "(its port dict keeps the last name), so the transaction ends on a different cycle"
    )
