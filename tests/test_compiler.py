import pytest

from redc import CompileError, compile_source


def evaluate(source: str, **inputs: int) -> int:
    ports = {f"in_{name}": value for name, value in inputs.items()}
    return compile_source(source).evaluate(**ports)["result"]


def test_straight_line_assignments_create_new_values() -> None:
    source = """
        uint8 main(uint8 a, uint8 b) {
            uint8 result = a + b;
            result ^= 15;
            return result;
        }
    """
    graph = compile_source(source)
    assert graph.evaluate(in_a=250, in_b=10)["result"] == 11
    assert {node["op"] for node in graph.live_nodes()} == {
        "input",
        "add",
        "xor",
        "const",
    }


def test_functions_inline_into_one_acyclic_graph() -> None:
    source = """
        uint8 twice(uint8 x) { return x + x; }
        uint8 main(uint8 value) { return twice(value) + 1; }
    """
    graph = compile_source(source)
    assert graph.evaluate(in_value=130)["result"] == 5
    assert all(arg < node["id"] for node in graph.nodes for arg in node["args"])


def test_expressions_keep_fixed_width_semantics() -> None:
    assert evaluate("uint8 main(uint8 a) { return a + 10; }", a=250) == 4
    assert evaluate("bool main(int8 a) { return a < 0; }", a=-1) == 1
    assert evaluate("int8 main(int8 a) { return a + 1; }", a=127) == -128


def test_rejects_hidden_state_and_uninitialized_reads() -> None:
    with pytest.raises(CompileError, match="globals must be const"):
        compile_source("uint8 state = 0; uint8 main() { return state; }")
    with pytest.raises(CompileError, match="before initialization"):
        compile_source("uint8 main() { uint8 x; return x; }")
    with pytest.raises(CompileError, match="recursive"):
        compile_source("uint8 main() { return main(); }")


def test_structured_control_is_deferred_to_the_next_stage() -> None:
    with pytest.raises(CompileError, match="straight-line lowering"):
        compile_source("uint8 main(bool choose) { if (choose) return 1; return 0; }")
