import json

import pytest

from redc import BOOL, CompileError, Graph, IRType


def test_graph_is_topological_and_evaluates() -> None:
    graph = Graph()
    byte = IRType(8)
    a = graph.input("in_a", byte)
    b = graph.input("in_b", byte)
    total = graph.op("add", byte, a, b)
    selected = graph.mux(graph.input("choose_sum", BOOL), total, a)
    graph.output("result", selected)

    graph.validate()
    assert graph.evaluate(in_a=250, in_b=10, choose_sum=1) == {"result": 4}
    assert all(arg < node["id"] for node in graph.nodes for arg in node["args"])


def test_json_is_a_versioned_target_neutral_contract() -> None:
    graph = Graph()
    value = graph.input("value", IRType(4))
    graph.output("result", value)

    payload = json.loads(graph.to_json())
    assert payload["schema"] == "redc.comb.v1"
    assert payload["combinational"] is True
    assert payload["inputs"][0]["type"]["width"] == 4


def test_validator_rejects_cycles_and_state() -> None:
    graph = Graph()
    byte = IRType(8)
    a = graph.input("a", byte)
    b = graph.input("b", byte)
    result = graph.op("add", byte, a, b)
    graph.output("result", result)

    graph.nodes[result.id]["args"][0] = result.id
    with pytest.raises(CompileError, match="cycle"):
        graph.validate()

    graph.nodes[result.id]["args"] = [a.id, b.id]
    graph.nodes[result.id]["op"] = "flip_flop"
    with pytest.raises(CompileError, match="stateful or unknown"):
        graph.validate()


def _counter_to(limit: int) -> Graph:
    """A minimal sequential graph: count up, assert done when count == limit."""
    graph = Graph()
    byte = IRType(8)
    graph.input("start", BOOL)  # unused here, but present for the handshake
    count = graph.register(byte)
    incremented = graph.op("add", byte, count, graph.constant(1, byte))
    done = graph.op("eq", BOOL, count, graph.constant(limit, byte))
    graph.set_register(count, incremented, graph.constant(1, BOOL))
    graph.output("result", count)
    graph.output("done", done)
    return graph


def test_registers_make_the_graph_sequential() -> None:
    graph = _counter_to(3)
    assert graph.sequential is True
    graph.validate()  # a register-mediated cycle is legal
    payload = json.loads(graph.to_json())
    assert payload["schema"] == "redc.seq.v1"
    assert payload["sequential"] is True
    assert payload["clock"]["name"] == "clk"


def test_sequential_graph_simulates_over_time() -> None:
    graph = _counter_to(3)
    result = graph.run()
    assert result == {"result": 3, "done": 1}


def test_evaluate_rejects_sequential_and_run_rejects_combinational() -> None:
    sequential = _counter_to(3)
    with pytest.raises(CompileError, match="use run"):
        sequential.evaluate()

    combinational = Graph()
    combinational.output("result", combinational.input("a", IRType(8)))
    with pytest.raises(CompileError, match="evaluate"):
        combinational.run(a=1)
