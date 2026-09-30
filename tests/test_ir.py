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
    graph.nodes[result.id]["op"] = "register"
    with pytest.raises(CompileError, match="stateful"):
        graph.validate()
