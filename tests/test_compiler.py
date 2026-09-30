from pathlib import Path

import pytest

from redc import CompileError, compile_source, emit_systemverilog

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def py_fib(n: int) -> int:
    last1 = last2 = 1
    for _ in range(2, n):
        last1, last2 = last2, last1 + last2
    return last2


def test_straight_line_stays_combinational() -> None:
    graph = compile_source(
        """
        uint8 main(uint8 a, uint8 b) {
            uint8 result = a + b;
            result ^= 15;
            return result;
        }
        """
    )
    assert graph.sequential is False
    assert graph.evaluate(in_a=250, in_b=10)["result"] == 11


def test_runtime_if_merges_with_a_mux() -> None:
    graph = compile_source("uint8 main(bool choose) { if (choose) return 1; return 0; }")
    assert graph.sequential is False
    assert graph.evaluate(in_choose=1)["result"] == 1
    assert graph.evaluate(in_choose=0)["result"] == 0


def test_compile_time_bounded_loop_still_unrolls() -> None:
    graph = compile_source(
        """
        uint8 main(uint8 a) {
            uint8 sum = 0;
            for (uint8 i = 0; i < 4; i = i + 1) { sum = sum + a; }
            return sum;
        }
        """
    )
    assert graph.sequential is False
    assert graph.evaluate(in_a=3)["result"] == 12


def test_runtime_bounded_for_loop_lowers_to_registers() -> None:
    graph = compile_source((EXAMPLES / "uint8_fib.redc").read_text(), top="fib")
    assert graph.sequential is True
    assert "start" in {port["name"] for port in graph.inputs}
    assert "done" in {port["name"] for port in graph.outputs}
    for n in range(1, 14):
        assert graph.run(in_n=n)["result"] == py_fib(n)


def test_runtime_bounded_while_loop_counts_up() -> None:
    graph = compile_source(
        "uint8 main(uint8 n) { uint8 c = 0; while (c < n) { c = c + 1; } return c; }"
    )
    assert graph.sequential is True
    for n in range(10):
        assert graph.run(in_n=n)["result"] == n


def test_sequential_backend_emits_a_clocked_module() -> None:
    graph = compile_source((EXAMPLES / "uint8_fib.redc").read_text(), top="fib")
    verilog = emit_systemverilog(graph, "redc_fib")
    assert "always_ff @(posedge clk or posedge rst)" in verilog
    assert "input logic clk" in verilog
    assert "output logic done" in verilog


def test_runtime_bounded_loop_only_in_top_function() -> None:
    source = """
        uint8 helper(uint8 n) {
            uint8 s = 0;
            for (uint8 i = 0; i < n; i = i + 1) { s = s + 1; }
            return s;
        }
        uint8 main(uint8 n) { return helper(n); }
    """
    with pytest.raises(CompileError, match="top-level"):
        compile_source(source)


def test_runtime_bounded_loop_must_be_unconditional() -> None:
    source = """
        uint8 main(uint8 n, bool go) {
            uint8 s = 0;
            if (go) { for (uint8 i = 0; i < n; i = i + 1) { s = s + 1; } }
            return s;
        }
    """
    with pytest.raises(CompileError, match="unconditionally"):
        compile_source(source)


def test_nested_runtime_bounded_loops_are_rejected() -> None:
    source = """
        uint8 main(uint8 n) {
            uint8 s = 0;
            for (uint8 i = 0; i < n; i = i + 1) {
                for (uint8 j = 0; j < n; j = j + 1) { s = s + 1; }
            }
            return s;
        }
    """
    with pytest.raises(CompileError, match="nested"):
        compile_source(source)


def test_backend_registry_is_extensible() -> None:
    from redc.backends import available_backends, get_backend

    names = available_backends()
    assert "systemverilog" in names and "ir-json" in names

    graph = compile_source("uint8 main(uint8 a, uint8 b) { return a + b; }")
    for name in names:
        backend = get_backend(name)
        artifact = backend.emit(graph, "redc_main")
        assert isinstance(artifact, str) and artifact
        assert backend.extension.startswith(".")

    with pytest.raises(CompileError, match="unknown backend"):
        get_backend("vhdl")
