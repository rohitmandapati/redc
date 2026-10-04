"""Abstractions shared by both physical backends, and the import seam between them.

``OperationSignature`` (exact-typed operation keys) and ``TraceLevel``
(replay-trace verbosity) used to live inside the coarse ``redc.physical``
backend.  They moved to the target-neutral modules :mod:`redc.signature` and
:mod:`redc.tracing` so that ``redc.physical_primitive`` can key its synthesis
recipes and trace levels by the very same classes WITHOUT importing the coarse
backend.  This file pins that refactoring down:

* the coarse re-exports are the SAME objects (``is``), not look-alikes;
* their behaviour is unchanged (the assertions of ``tests/test_implementation.py``
  and the old ``TraceLevel.parse`` rules, error messages included);
* one key works in both registries, one level means the same in both recorders;
* neither backend package drags the other in at import time.  That is checked
  in a fresh interpreter (``sys.executable -c ...``), because this test
  process imports both backends.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import redc.physical
import redc.physical.implementation
import redc.physical.pnr
import redc.physical.pnr.config
import redc.physical_primitive.pnr
import redc.physical_primitive.pnr.config
import redc.physical_primitive.synthesis.lower
import redc.physical_primitive.synthesis.registry
import redc.signature
import redc.tracing
from redc import BOOL, CompileError, Graph, IRType
from redc.ir import SHIFT_AMOUNT
from redc.physical import LIBRARY
from redc.physical.pnr import PnRConfig, TraceRecorder
from redc.physical_primitive import PrimitivePnRConfig, PrimitiveTraceRecorder
from redc.physical_primitive.synthesis.recipes import DEFAULT_SYNTHESIS
from redc.signature import REGISTER_OP, OperationSignature
from redc.tracing import TraceLevel

UINT8, INT8 = IRType(8), IRType(8, signed=True)
sig = OperationSignature.of


# -- OperationSignature: one class, unchanged -------------------------------------------


def test_operation_signature_is_one_class_everywhere() -> None:
    assert redc.signature.OperationSignature is redc.physical.OperationSignature
    assert redc.physical.OperationSignature is redc.physical.implementation.OperationSignature
    assert OperationSignature.__module__ == "redc.signature"
    # The primitive backend keys its recipes by the very same class.
    assert redc.physical_primitive.synthesis.registry.OperationSignature is OperationSignature
    assert redc.physical_primitive.synthesis.lower.OperationSignature is OperationSignature
    assert REGISTER_OP is redc.physical.implementation.REGISTER_OP
    assert REGISTER_OP == "register"
    assert {"REGISTER_OP", "OperationSignature"} <= set(redc.physical.implementation.__all__)
    assert "OperationSignature" in redc.physical.__all__
    assert set(redc.signature.__all__) == {"REGISTER_OP", "OperationSignature"}


def test_one_signature_keys_both_backends() -> None:
    coarse = redc.physical.OperationSignature.of("add", UINT8, UINT8, UINT8)
    shared = redc.signature.OperationSignature.of("add", UINT8, UINT8, UINT8)
    assert coarse == shared and hash(coarse) == hash(shared)
    assert LIBRARY.candidates(shared) and LIBRARY.candidates(shared) == LIBRARY.candidates(coarse)
    assert DEFAULT_SYNTHESIS.resolve(coarse).name == DEFAULT_SYNTHESIS.resolve(shared).name == "ripple_carry_add"
    register = sig(REGISTER_OP, UINT8, UINT8, BOOL)
    assert LIBRARY.cells(register)  # the coarse backend maps registers to cells ...
    assert not DEFAULT_SYNTHESIS.supports(register)  # ... the primitive one lowers them itself


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
    assert sig("cast", IRType(1), BOOL) != sig("cast", BOOL, IRType(1))  # bool is not uint1


@pytest.mark.parametrize(
    ("signature", "text"),
    [
        (sig("shr", INT8, INT8, SHIFT_AMOUNT), "shr(int8, uint64) -> int8"),
        (sig("add", UINT8, UINT8, UINT8), "add(uint8, uint8) -> uint8"),
        (sig("cast", IRType(16, signed=True), INT8), "cast(int8) -> int16"),
        (sig("mux", INT8, BOOL, INT8, INT8), "mux(bool, int8, int8) -> int8"),
        (sig(REGISTER_OP, UINT8, UINT8, BOOL), "register(uint8, bool) -> uint8"),
        (sig("nand", BOOL, BOOL, BOOL), "nand(bool, bool) -> bool"),
        (sig("frobnicate", UINT8), "frobnicate() -> uint8"),
    ],
    ids=lambda v: v if isinstance(v, str) else v.op,
)
def test_signature_str(signature: OperationSignature, text: str) -> None:
    assert str(signature) == text


def test_of_is_the_positional_spelling_of_the_constructor() -> None:
    assert sig("sub", INT8, INT8, INT8) == OperationSignature("sub", INT8, (INT8, INT8))
    signature = sig("lt", BOOL, UINT8, UINT8)
    assert (signature.op, signature.result_type, signature.operand_types) == ("lt", BOOL, (UINT8, UINT8))
    assert type(signature.operand_types) is tuple
    assert {signature: 1}[sig("lt", BOOL, UINT8, UINT8)] == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        signature.op = "le"  # type: ignore[misc]
    assert not hasattr(signature, "__dict__")  # slots, as before


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("shl", UINT8, UINT8), "incorrect operand count for shl"),  # a shift always has a variable amount
        (("neg", UINT8), "incorrect operand count for neg"),
        (("shr", UINT8, UINT8, UINT8), "shift type mismatch"),  # the amount must be uint64
        (("lt", UINT8, UINT8, UINT8), "comparison type mismatch"),  # compares produce bool
        (("lt", BOOL, UINT8, INT8), "comparison type mismatch"),
        (("mux", UINT8, UINT8, UINT8, UINT8), "mux type mismatch"),
        (("not", UINT8, UINT8), "logical negation type mismatch"),
        (("add", UINT8, INT8, INT8), "arithmetic type mismatch"),
        ((REGISTER_OP, UINT8, UINT8), "register signature must be register(T next, bool enable) -> T"),
        ((REGISTER_OP, UINT8, UINT8, UINT8), "register signature must be register(T next, bool enable) -> T"),
        ((REGISTER_OP, UINT8, INT8, BOOL), "register signature must be register(T next, bool enable) -> T"),
        (("cast", UINT8), "cast signature takes exactly one operand"),
        (("cast", UINT8, UINT8, UINT8), "cast signature takes exactly one operand"),
        (("", UINT8), "operation signature needs an op"),
    ],
    ids=lambda v: v if isinstance(v, str) else "-".join(getattr(t, "name", str(t)) for t in v),
)
def test_signatures_obey_ir_typing_rules(args: tuple, message: str) -> None:
    with pytest.raises(CompileError, match=f"^{re.escape(message)}$"):
        sig(*args)


def test_well_typed_and_physical_only_signatures_are_accepted() -> None:
    assert sig("nand", BOOL, BOOL, BOOL).op == "nand"  # physical-only ops are free-form
    assert sig("frobnicate", UINT8).operand_types == ()
    assert sig("cast", UINT8, UINT8).operand_types == (UINT8,)  # widths alone never decide a cast
    assert sig(REGISTER_OP, INT8, INT8, BOOL).result_type == INT8
    assert sig("mux", INT8, BOOL, INT8, INT8).operand_types[0] == BOOL
    assert sig("shl", UINT8, UINT8, SHIFT_AMOUNT).operand_types[1] == SHIFT_AMOUNT


def test_signature_from_ir_node() -> None:
    graph = Graph()
    value = graph.input("v", INT8)
    amount = graph.input("n", SHIFT_AMOUNT)
    shifted = graph.op("shr", INT8, value, amount)
    widened = graph.cast(shifted, IRType(16))
    state = graph.register(IRType(16), init=5)
    graph.set_register(state, widened, graph.op("lt", BOOL, value, graph.constant(0, INT8)))
    graph.output("r", state)
    nodes = {node["id"]: node for node in graph.live_nodes()}
    assert OperationSignature.from_ir_node(graph, nodes[shifted.id]) == sig("shr", INT8, INT8, SHIFT_AMOUNT)
    assert OperationSignature.from_ir_node(graph, nodes[widened.id]) == sig("cast", IRType(16), INT8)
    assert OperationSignature.from_ir_node(graph, nodes[state.id]) == sig(REGISTER_OP, IRType(16), IRType(16), BOOL)
    for boundary, op in ((value, "input"), (graph.constant(0, INT8), "const")):
        with pytest.raises(CompileError, match=f"^{op} nodes are boundary cells, not operations$"):
            OperationSignature.from_ir_node(graph, graph.nodes[boundary.id])


# -- TraceLevel: one enum, unchanged ----------------------------------------------------


def test_trace_level_is_one_enum_everywhere() -> None:
    assert redc.tracing.TraceLevel is redc.physical.pnr.TraceLevel
    assert redc.physical.pnr.TraceLevel is redc.physical.pnr.config.TraceLevel
    assert redc.physical_primitive.pnr.TraceLevel is TraceLevel
    assert redc.physical_primitive.pnr.config.TraceLevel is TraceLevel
    assert TraceLevel.__module__ == "redc.tracing"
    assert "TraceLevel" in redc.physical.pnr.config.__all__ and "TraceLevel" in redc.physical.pnr.__all__
    assert redc.tracing.__all__ == ["TraceLevel"]


def test_trace_levels_are_ordered_integers_with_lowercase_labels() -> None:
    assert [(level.name, int(level), level.label) for level in TraceLevel] == [
        ("NONE", 0, "none"),
        ("BASIC", 1, "basic"),
        ("DETAILED", 2, "detailed"),
        ("SEARCH", 3, "search"),
    ]
    assert TraceLevel.NONE < TraceLevel.BASIC < TraceLevel.DETAILED < TraceLevel.SEARCH
    assert TraceLevel.DETAILED == 2 and isinstance(TraceLevel.SEARCH, int)


@pytest.mark.parametrize(
    ("value", "level"),
    [
        ("none", TraceLevel.NONE),
        ("basic", TraceLevel.BASIC),
        ("detailed", TraceLevel.DETAILED),
        ("search", TraceLevel.SEARCH),
        ("BASIC", TraceLevel.BASIC),  # names are case-insensitive
        ("Search", TraceLevel.SEARCH),
        (0, TraceLevel.NONE),
        (2, TraceLevel.DETAILED),
        (TraceLevel.SEARCH, TraceLevel.SEARCH),
    ],
    ids=repr,
)
def test_trace_level_parse(value: str | int, level: TraceLevel) -> None:
    assert TraceLevel.parse(value) is level
    assert TraceLevel.parse(level.label) is level


@pytest.mark.parametrize("value", ["loud", "", " basic", "1", "detail"])
def test_trace_level_parse_rejects_unknown_names(value: str) -> None:
    message = f"unknown trace level {value!r}; use one of none, basic, detailed, search"
    with pytest.raises(CompileError) as caught:
        TraceLevel.parse(value)
    assert str(caught.value) == message
    assert caught.value.__suppress_context__  # `from None`: no KeyError chained in


@pytest.mark.parametrize("value", [-1, 4, 99])
def test_trace_level_parse_of_an_unknown_number_is_a_value_error(value: int) -> None:
    """Unchanged: integers go straight to ``TraceLevel(value)``."""
    with pytest.raises(ValueError, match="is not a valid TraceLevel"):
        TraceLevel.parse(value)


@pytest.mark.parametrize("label", ["none", "basic", "detailed", "search"])
def test_a_trace_level_means_the_same_in_both_backends(label: str) -> None:
    coarse, primitive = TraceRecorder(label), PrimitiveTraceRecorder(label)
    assert coarse.level is primitive.level is TraceLevel.parse(label)
    for event_level in TraceLevel:
        assert coarse.wants(event_level) == primitive.wants(event_level), event_level
        expected = TraceLevel.parse(label) != TraceLevel.NONE and TraceLevel.parse(label) >= event_level
        assert primitive.wants(event_level) == expected
    coarse_config, primitive_config = PnRConfig(trace_level=label), PrimitivePnRConfig(trace_level=label)
    assert coarse_config.trace_level is primitive_config.trace_level is TraceLevel.parse(label)
    assert coarse_config.to_dict()["trace_level"] == primitive_config.to_dict()["trace_level"] == label


@pytest.mark.parametrize(
    "make",
    [
        lambda level: PnRConfig(trace_level=level),
        lambda level: PrimitivePnRConfig(trace_level=level),
        TraceRecorder,
        PrimitiveTraceRecorder,
    ],
    ids=["coarse-config", "primitive-config", "coarse-recorder", "primitive-recorder"],
)
def test_both_backends_reject_unknown_trace_levels_the_same_way(make) -> None:
    message = "unknown trace level 'verbose'; use one of none, basic, detailed, search"
    with pytest.raises(CompileError, match=f"^{re.escape(message)}$"):
        make("verbose")


# -- the import seam between the backends -------------------------------------------------


def loaded_modules(tmp_path: Path, code: str) -> list[str]:
    """Run ``code`` in a fresh interpreter and return every ``redc`` module it
    loaded.  The working directory is ``tmp_path`` and no bytecode is written,
    so nothing outside ``tmp_path`` changes."""
    probe = (
        f"{code}\n"
        "import json as _json, sys as _sys\n"
        "print(_json.dumps(sorted(m for m in _sys.modules if m == 'redc' or m.startswith('redc.'))))\n"
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=tmp_path, env=env, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def coarse(modules: list[str]) -> list[str]:
    return [m for m in modules if m == "redc.physical" or m.startswith("redc.physical.")]


def primitive(modules: list[str]) -> list[str]:
    return [m for m in modules if m == "redc.physical_primitive" or m.startswith("redc.physical_primitive.")]


def test_importing_the_primitive_backend_does_not_import_the_coarse_backend(tmp_path: Path) -> None:
    modules = loaded_modules(tmp_path, "import redc.physical_primitive")
    assert "redc.physical_primitive" in modules
    assert {"redc.signature", "redc.tracing"} <= set(modules)  # the shared seam
    assert coarse(modules) == []


def test_running_the_primitive_pipeline_does_not_import_the_coarse_backend(tmp_path: Path) -> None:
    code = """
from redc import compile_source
from redc.physical_primitive import (
    PadInterfacePolicy, PrimitivePnRConfig, PrimitiveSimulator, PrimitiveTraceRecorder, place_and_route_graph,
)
from redc.physical_backends import get_physical_backend
graph = compile_source("bool main(bool a, bool b) { return a && b; }")
netlist, mapped, result = place_and_route_graph(
    graph, PrimitivePnRConfig(), interface=PadInterfacePolicy(), trace=PrimitiveTraceRecorder("detailed")
)
assert result.success, result.failure
assert PrimitiveSimulator(netlist).evaluate(in_a=1, in_b=1) == {"result": 1}
assert result.to_design_dict()["schema"] == "redc.physical-primitive.v1"
backend = get_physical_backend("physical-primitive")
assert backend.dump_netlist(graph, stage="mapped")["schema"] == "redc.primitive-mapped-netlist.v1"
"""
    modules = loaded_modules(tmp_path, code)
    assert "redc.physical_primitive.synthesis.recipes" in modules  # it really ran
    assert coarse(modules) == []


def test_importing_the_coarse_backend_still_works_without_the_primitive_backend(tmp_path: Path) -> None:
    code = """
import redc.physical
from redc.physical import LIBRARY, OperationSignature
from redc.ir import IRType
u8 = IRType(8)
assert LIBRARY.cells(OperationSignature.of("add", u8, u8, u8))
"""
    modules = loaded_modules(tmp_path, code)
    assert "redc.physical" in modules and "redc.signature" in modules
    assert primitive(modules) == []


def test_running_the_coarse_pipeline_does_not_import_the_primitive_backend(tmp_path: Path) -> None:
    code = """
from redc import compile_source
from redc.physical import lower_to_physical
from redc.physical.pnr import PnRConfig, TraceLevel, TraceRecorder, place_and_route
graph = compile_source("bool main(bool a) { return !a; }")
result = place_and_route(lower_to_physical(graph), PnRConfig(), trace=TraceRecorder(TraceLevel.BASIC))
assert result.success
"""
    modules = loaded_modules(tmp_path, code)
    assert {"redc.physical.pnr", "redc.tracing", "redc.signature"} <= set(modules)
    assert primitive(modules) == []


def test_the_shared_modules_need_neither_backend(tmp_path: Path) -> None:
    modules = loaded_modules(tmp_path, "import redc.signature, redc.tracing")
    assert {"redc.signature", "redc.tracing"} <= set(modules)
    assert coarse(modules) == primitive(modules) == []
