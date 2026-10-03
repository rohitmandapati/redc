from pathlib import Path
from typing import Any

import pytest

from redc import BOOL, CompileError, Graph, IRType, compile_source
from redc.ir import SHIFT_AMOUNT
from redc.physical import (
    LIBRARY,
    CompositeImplementation,
    DirectCellImplementation,
    ImplementationRegistry,
    OperationSignature,
    RecipeBuilder,
    is_supported_physical_type,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
UINT8, INT8 = IRType(8), IRType(8, signed=True)
sig = OperationSignature.of


# -- signatures ---------------------------------------------------------------


def test_signatures_carry_full_types() -> None:
    assert sig("add", UINT8, UINT8, UINT8) != sig("add", INT8, INT8, INT8)
    assert sig("lt", BOOL, UINT8, UINT8) != sig("lt", BOOL, INT8, INT8)
    assert sig("shr", UINT8, UINT8, SHIFT_AMOUNT) != sig("shr", INT8, INT8, SHIFT_AMOUNT)
    casts = {
        sig("cast", IRType(16), UINT8),
        sig("cast", IRType(16, signed=True), INT8),
        sig("cast", IRType(16), INT8),
        sig("cast", BOOL, UINT8),
    }
    assert len(casts) == 4
    assert str(sig("shr", INT8, INT8, SHIFT_AMOUNT)) == "shr(int8, uint64) -> int8"


def test_signatures_obey_ir_typing_rules() -> None:
    with pytest.raises(CompileError, match="operand count"):
        sig("shl", UINT8, UINT8)  # a shift always has a variable amount
    with pytest.raises(CompileError, match="shift type"):
        sig("shr", UINT8, UINT8, UINT8)  # amount must be uint64
    with pytest.raises(CompileError, match="comparison"):
        sig("lt", UINT8, UINT8, UINT8)  # compares produce bool
    with pytest.raises(CompileError, match="mux"):
        sig("mux", UINT8, UINT8, UINT8, UINT8)
    with pytest.raises(CompileError, match="register"):
        sig("register", UINT8, UINT8)
    with pytest.raises(CompileError, match="cast"):
        sig("cast", UINT8)
    # Physical-only ops (e.g. nand) are free-form.
    assert sig("nand", BOOL, BOOL, BOOL).op == "nand"


def test_signature_from_ir_node() -> None:
    graph = Graph()
    value = graph.input("v", INT8)
    amount = graph.input("n", SHIFT_AMOUNT)
    shifted = graph.op("shr", INT8, value, amount)
    widened = graph.cast(shifted, IRType(16))
    graph.output("r", widened)
    nodes = {node["id"]: node for node in graph.live_nodes()}
    assert OperationSignature.from_ir_node(graph, nodes[shifted.id]) == sig(
        "shr", INT8, INT8, SHIFT_AMOUNT
    )
    assert OperationSignature.from_ir_node(graph, nodes[widened.id]) == sig(
        "cast", IRType(16), INT8
    )
    with pytest.raises(CompileError, match="boundary"):
        OperationSignature.from_ir_node(graph, nodes[value.id])


# -- registry -----------------------------------------------------------------


def test_library_registers_direct_cells() -> None:
    candidates = LIBRARY.candidates(sig("add", UINT8, UINT8, UINT8))
    assert candidates
    assert all(isinstance(c, DirectCellImplementation) for c in candidates)
    assert [c.name for c in candidates] == [c.name for c in LIBRARY.cells(candidates[0].signatures[0])]
    assert LIBRARY.candidates(sig("add", IRType(3), IRType(3), IRType(3))) == ()


def test_candidate_order_is_deterministic() -> None:
    from redc.physical.cells import Library

    rebuilt = Library(dict(LIBRARY))
    for signature in LIBRARY.implementations.signatures():
        assert [c.name for c in rebuilt.candidates(signature)] == [
            c.name for c in LIBRARY.candidates(signature)
        ]


class RecordingBuilder:
    """Stand-in for the future mapper: records what a recipe would emit."""

    def __init__(self) -> None:
        self.emitted: list[tuple[OperationSignature, tuple[Any, ...]]] = []

    def emit(self, signature: OperationSignature, *operands: Any) -> Any:
        self.emitted.append((signature, operands))
        return f"%{len(self.emitted)}"

    def constant(self, value: int, typ: IRType) -> Any:
        return f"#{typ.name}:{value}"


def _sub_via_add(b: RecipeBuilder, ops: tuple[Any, ...]) -> Any:
    a, x = ops
    inverted = b.emit(sig("inv", UINT8, UINT8), x)
    negated = b.emit(sig("add", UINT8, UINT8, UINT8), inverted, b.constant(1, UINT8))
    return b.emit(sig("add", UINT8, UINT8, UINT8), a, negated)



def test_composite_plugs_into_the_registry_beside_direct_cells() -> None:
    sub = sig("sub", UINT8, UINT8, UINT8)
    composite = CompositeImplementation("uint8_sub_via_add", sub, _sub_via_add)
    registry = ImplementationRegistry(
        DirectCellImplementation(cell) for cell in LIBRARY.cells(sub)
    )
    registry.register(composite)
    names = [c.name for c in registry.candidates(sub)]
    assert names == [LIBRARY.cells(sub)[0].name, "uint8_sub_via_add"]
    assert registry.direct_cells(sub) == LIBRARY.cells(sub)

    # One operation -> many sub-operations, each itself a registry signature.
    builder = RecordingBuilder()
    result = composite.expand(builder, ("%a", "%b"))
    assert result == "%3"
    assert [s for s, _ in builder.emitted] == [
        sig("inv", UINT8, UINT8),
        sig("add", UINT8, UINT8, UINT8),
        sig("add", UINT8, UINT8, UINT8),
    ]
    assert all(LIBRARY.cells(s) for s, _ in builder.emitted)
    with pytest.raises(CompileError, match="expected 2 operands"):
        composite.expand(builder, ("%a",))


def test_composite_applicability_can_depend_on_constant_operands() -> None:
    mul = sig("mul", UINT8, UINT8, UINT8)
    only_power_of_two = CompositeImplementation(
        "example_guarded",
        mul,
        recipe=lambda b, ops: ops[0],
        applies=lambda consts: consts[1] is not None and consts[1] & (consts[1] - 1) == 0,
    )
    assert only_power_of_two.applicable([None, 8])
    assert not only_power_of_two.applicable([None, 6])
    assert not only_power_of_two.applicable([None, None])


def test_registering_the_same_implementation_twice_is_rejected() -> None:
    cell = LIBRARY["uint8_add_a-0-0-0_b-0-0-1_out-0-0-2"]
    registry = ImplementationRegistry([DirectCellImplementation(cell)])
    with pytest.raises(CompileError, match="already registered"):
        registry.register(DirectCellImplementation(cell))


# -- operand order is semantic ------------------------------------------------


def test_cell_inputs_zip_with_ir_arguments() -> None:
    graph = Graph()
    a = graph.input("a", INT8)
    b = graph.input("b", INT8)
    diff = graph.op("sub", INT8, a, b)
    graph.output("r", diff)
    node = graph.nodes[diff.id]
    (cell,) = LIBRARY.cells(OperationSignature.from_ir_node(graph, node))
    stimulus = {"a": INT8.bits(-100), "b": INT8.bits(50)}
    pins = dict(zip((p.name for p in cell.operand_inputs), (stimulus["a"], stimulus["b"])))
    want = graph.evaluate(a=-100, b=50)["r"]
    assert INT8.number(cell.behavior(pins)["out"]) == want


# -- live IR is the lowering input ------------------------------------------


def test_dead_operations_are_not_live() -> None:
    graph = compile_source(
        """
        uint8 main(uint8 a, uint8 b) {
            uint8 unused = a * b;
            return a + b;
        }
        """
    )
    all_ops = {node["op"] for node in graph.nodes}
    live_ops = {node["op"] for node in graph.live_nodes()}
    assert "mul" in all_ops and "mul" not in live_ops


EXAMPLE_TOPS = {
    "array_mux.redc": "main",
    "dot_product.redc": "main",
    "popcount.redc": "main",
    "priority_encoder.redc": "main",
    "uint8_add.redc": "add",
    "uint8_alu.redc": "main",
    "uint8_fib.redc": "fib",
}


@pytest.mark.parametrize("example", sorted(EXAMPLE_TOPS))
def test_live_example_operations_resolve_to_direct_cells(example: str) -> None:
    """Readiness check for tech-map: every live operation whose types are all
    physically supported has at least one direct cell.  (``uint8_alu`` uses a
    ``uint3`` selector, which the physical backend must reject, not map.)"""
    graph = compile_source((EXAMPLES / example).read_text(), top=EXAMPLE_TOPS[example])
    unsupported: set[str] = set()
    for node in graph.live_nodes():
        if node["op"] in {"input", "const"}:
            continue
        signature = OperationSignature.from_ir_node(graph, node)
        types = (signature.result_type, *signature.operand_types)
        if not all(is_supported_physical_type(t) for t in types):
            unsupported.add(str(signature))
            continue
        assert LIBRARY.cells(signature), f"{example}: no cell for {signature}"
    if example == "uint8_alu.redc":
        assert unsupported  # the uint3 operations are flagged, not mapped
    else:
        assert not unsupported
