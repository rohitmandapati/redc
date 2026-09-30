"""Typed intermediate representation for combinational and sequential circuits.

The IR is deliberately independent of SystemVerilog and Minecraft.  Ordinary
(combinational) nodes may only reference nodes created before them, so the
combinational sub-graph is acyclic by construction and checked again by
:meth:`Graph.validate`.

State is modelled by a single node type, ``register``.  A register holds a value
across clock ticks: it exposes its *current* value as an ordinary operand and
carries two back-edges, ``next`` (the value latched on the next tick when the
register is enabled) and ``enable``.  These back-edges are the only edges allowed
to point at a node created later, so every cycle in the graph must pass through a
register.  A graph that contains registers is *sequential*; one that does not is
*combinational* and behaves exactly as before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .parser import CompileError


@dataclass(frozen=True, slots=True)
class IRType:
    width: int
    signed: bool = False
    boolean: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.width <= 64:
            raise CompileError("integer widths must be between 1 and 64 bits")
        if self.boolean and (self.width != 1 or self.signed):
            raise CompileError("bool must be an unsigned one-bit type")

    @property
    def mask(self) -> int:
        return (1 << self.width) - 1

    def bits(self, value: int) -> int:
        return value & self.mask

    def number(self, bits: int) -> int:
        bits &= self.mask
        sign_bit = 1 << (self.width - 1)
        return bits - (1 << self.width) if self.signed and bits & sign_bit else bits

    def to_dict(self) -> dict[str, int | bool]:
        return {"width": self.width, "signed": self.signed, "boolean": self.boolean}


BOOL = IRType(1, boolean=True)


def type_from_name(name: str) -> IRType | None:
    """Convert a source type spelling into an IR type."""
    if name == "void":
        return None
    if name == "bool":
        return BOOL
    signed = not name.startswith("u")
    digits = name[3:] if signed else name[4:]
    return IRType(int(digits) if digits else 32, signed=signed)


@dataclass(frozen=True, slots=True)
class Value:
    id: int
    type: IRType


@dataclass(frozen=True, slots=True)
class Literal:
    number: int


OPS = {
    "cast",
    "mux",
    "not",
    "neg",
    "inv",
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "shl",
    "shr",
    "and",
    "or",
    "xor",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
}


def _calculate(op: str, args: list[int], typ: IRType, input_types: list[IRType]) -> int:
    nums = [arg_type.number(value) for arg_type, value in zip(input_types, args)]
    if op == "cast":
        answer = int(nums[0] != 0) if typ.boolean else nums[0]
    elif op == "mux":
        answer = args[1] if args[0] else args[2]
    elif op == "not":
        answer = int(not args[0])
    elif op == "neg":
        answer = -nums[0]
    elif op == "inv":
        answer = ~args[0]
    elif op == "add":
        answer = nums[0] + nums[1]
    elif op == "sub":
        answer = nums[0] - nums[1]
    elif op == "mul":
        answer = nums[0] * nums[1]
    elif op in {"div", "mod"}:
        if nums[1] == 0:
            answer = 0
        else:
            quotient = abs(nums[0]) // abs(nums[1])
            quotient = -quotient if (nums[0] < 0) != (nums[1] < 0) else quotient
            answer = quotient if op == "div" else nums[0] - quotient * nums[1]
    elif op == "shl":
        answer = 0 if args[1] >= typ.width else args[0] << args[1]
    elif op == "shr":
        answer = nums[0] >> min(args[1], typ.width)
    elif op == "and":
        answer = args[0] & args[1]
    elif op == "or":
        answer = args[0] | args[1]
    elif op == "xor":
        answer = args[0] ^ args[1]
    elif op == "eq":
        answer = nums[0] == nums[1]
    elif op == "ne":
        answer = nums[0] != nums[1]
    elif op == "lt":
        answer = nums[0] < nums[1]
    elif op == "le":
        answer = nums[0] <= nums[1]
    elif op == "gt":
        answer = nums[0] > nums[1]
    elif op == "ge":
        answer = nums[0] >= nums[1]
    else:
        raise CompileError(f"unknown IR operation {op}")
    return int(answer) & typ.mask


class Graph:
    """A typed DAG of pure bit-vector operations."""

    def __init__(self, max_nodes: int = 100_000) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.inputs: list[dict[str, Any]] = []
        self.outputs: list[dict[str, Any]] = []
        self.registers: list[int] = []
        self.max_nodes = max_nodes
        self._cache: dict[tuple[Any, ...], Value] = {}

    @property
    def sequential(self) -> bool:
        """Whether the graph holds state (and therefore needs a clock)."""
        return bool(self.registers)

    def node(
        self, op: str, typ: IRType, args: tuple[Value, ...] = (), **attrs: Any
    ) -> Value:
        key = (op, typ, tuple(arg.id for arg in args), tuple(sorted(attrs.items())))
        if key in self._cache:
            return self._cache[key]
        if len(self.nodes) >= self.max_nodes:
            raise CompileError(f"IR node budget exceeded ({self.max_nodes})")
        value = Value(len(self.nodes), typ)
        self.nodes.append(
            {
                "id": value.id,
                "op": op,
                "type": typ.to_dict(),
                "args": [arg.id for arg in args],
                **attrs,
            }
        )
        self._cache[key] = value
        return value

    def constant(self, number: int, typ: IRType) -> Value:
        return self.node("const", typ, value=typ.bits(number))

    def constant_value(self, value: Value) -> int | None:
        return self.nodes[value.id].get("value")

    def input(self, name: str, typ: IRType, *, source_name: str | None = None) -> Value:
        value = self.node("input", typ, name=name)
        self.inputs.append(
            {
                "name": name,
                "source_name": source_name or name,
                "node": value.id,
                "type": typ.to_dict(),
            }
        )
        return value

    def output(self, name: str, value: Value) -> None:
        self.outputs.append(
            {"name": name, "node": value.id, "type": value.type.to_dict()}
        )

    def cast(self, value: Value | Literal, typ: IRType) -> Value:
        if isinstance(value, Literal):
            return self.constant(value.number, typ)
        if value.type == typ:
            return value
        return self.op("cast", typ, value)

    def truth(self, value: Value) -> Value:
        return self.cast(value, BOOL)

    def op(self, op: str, typ: IRType, *args: Value) -> Value:
        constants = [self.constant_value(arg) for arg in args]
        if op == "mux":
            if constants[0] is not None:
                return args[1] if constants[0] else args[2]
            if args[1] == args[2]:
                return args[1]
        if all(value is not None for value in constants):
            result = _calculate(op, constants, typ, [arg.type for arg in args])
            return self.constant(result, typ)
        if op in {"eq", "le", "ge"} and args[0] == args[1]:
            return self.constant(1, BOOL)
        if op in {"ne", "lt", "gt", "xor", "sub"} and args[0] == args[1]:
            return self.constant(0, typ)
        if len(args) == 2 and args[0].type == args[1].type == typ:
            left, right = constants
            if op in {"add", "or", "xor"}:
                if left == 0:
                    return args[1]
                if right == 0:
                    return args[0]
            if op == "sub" and right == 0:
                return args[0]
            if op == "mul":
                if left == 0 or right == 0:
                    return self.constant(0, typ)
                if left == 1:
                    return args[1]
                if right == 1:
                    return args[0]
        return self.node(op, typ, tuple(args))

    def mux(self, condition: Value, yes: Value, no: Value) -> Value:
        if yes.type != no.type:
            raise CompileError("mux operands must have the same type")
        return self.op("mux", yes.type, condition, yes, no)

    def register(self, typ: IRType, init: int = 0) -> Value:
        """Allocate a state element.

        The register's ``next`` and ``enable`` inputs are wired later with
        :meth:`set_register`; until then the node is a leaf holding ``init``.
        Registers are never de-duplicated: each one is a distinct piece of
        state even if two happen to be wired identically.
        """
        if len(self.nodes) >= self.max_nodes:
            raise CompileError(f"IR node budget exceeded ({self.max_nodes})")
        value = Value(len(self.nodes), typ)
        self.nodes.append(
            {
                "id": value.id,
                "op": "register",
                "type": typ.to_dict(),
                "args": [],
                "init": typ.bits(init),
            }
        )
        self.registers.append(value.id)
        return value

    def set_register(self, register: Value, next_value: Value, enable: Value) -> None:
        """Wire a register's update path (its two back-edges)."""
        node = self.nodes[register.id]
        if node["op"] != "register":
            raise CompileError("set_register target is not a register")
        if next_value.type != register.type:
            raise CompileError("register next value type mismatch")
        if enable.type != BOOL:
            raise CompileError("register enable must be a bool")
        node["args"] = [next_value.id, enable.id]

    def live_nodes(self) -> list[dict[str, Any]]:
        live = {port["node"] for port in self.inputs}
        pending = [port["node"] for port in self.outputs]
        while pending:
            node_id = pending.pop()
            if node_id in live:
                continue
            live.add(node_id)
            pending.extend(self.nodes[node_id]["args"])
        return [node for node in self.nodes if node["id"] in live]

    def validate(self) -> None:
        for node in self.nodes:
            op = node["op"]
            if op not in OPS | {"input", "const", "register"}:
                raise CompileError("stateful or unknown operation in combinational IR")
            # Registers are the only nodes allowed to carry back-edges: every
            # cycle in the graph must pass through one.  Every other node may
            # only reference nodes that already exist, keeping the
            # combinational sub-graph acyclic.
            if op == "register":
                if any(arg < 0 or arg >= len(self.nodes) for arg in node["args"]):
                    raise CompileError("register references an invalid node")
            elif any(arg < 0 or arg >= node["id"] for arg in node["args"]):
                raise CompileError("cycle or invalid edge in combinational IR")
            typ = IRType(**node["type"])
            arity = (
                0
                if op in {"input", "const"}
                else 1
                if op in {"cast", "not", "neg", "inv"}
                else 3
                if op == "mux"
                else 2
            )
            if len(node["args"]) != arity:
                raise CompileError(f"incorrect operand count for {op}")
            if op == "register":
                next_type = IRType(**self.nodes[node["args"][0]]["type"])
                enable_type = IRType(**self.nodes[node["args"][1]]["type"])
                if next_type != typ or enable_type != BOOL:
                    raise CompileError("register type mismatch")
                continue
            arg_types = [IRType(**self.nodes[arg]["type"]) for arg in node["args"]]
            if op == "mux" and not (
                arg_types[0] == BOOL and arg_types[1] == arg_types[2] == typ
            ):
                raise CompileError("mux type mismatch")
            if op in {"eq", "ne", "lt", "le", "gt", "ge"} and not (
                arg_types[0] == arg_types[1] and typ == BOOL
            ):
                raise CompileError("comparison type mismatch")
            if op in {"shl", "shr"} and not (
                arg_types[0] == typ and arg_types[1] == IRType(64)
            ):
                raise CompileError("shift type mismatch")
            if op == "not" and not (arg_types[0] == typ == BOOL):
                raise CompileError("logical negation type mismatch")
            if op in {
                "add",
                "sub",
                "mul",
                "div",
                "mod",
                "and",
                "or",
                "xor",
                "neg",
                "inv",
            } and any(arg_type != typ for arg_type in arg_types):
                raise CompileError("arithmetic type mismatch")
        for port in self.outputs:
            if not 0 <= port["node"] < len(self.nodes):
                raise CompileError(f"output {port['name']} is not driven")

    def evaluate(self, **inputs: int) -> dict[str, int]:
        if self.sequential:
            raise CompileError(
                "sequential graph holds state; use run() to simulate it over time"
            )
        values: dict[int, int] = {}
        for node in self.live_nodes():
            typ = IRType(**node["type"])
            if node["op"] == "input":
                if node["name"] not in inputs:
                    raise CompileError(f"missing input {node['name']}")
                values[node["id"]] = typ.bits(inputs[node["name"]])
            elif node["op"] == "const":
                values[node["id"]] = node["value"]
            else:
                values[node["id"]] = _calculate(
                    node["op"],
                    [values[arg] for arg in node["args"]],
                    typ,
                    [IRType(**self.nodes[arg]["type"]) for arg in node["args"]],
                )
        return {
            port["name"]: IRType(**port["type"]).number(values[port["node"]])
            for port in self.outputs
        }

    def _combinational_pass(
        self,
        live: list[dict[str, Any]],
        state: dict[int, int],
        inputs: dict[str, int],
        start: int,
    ) -> dict[int, int]:
        """One cycle of pure logic given the current register state."""
        values: dict[int, int] = {}
        for node in live:
            op, node_id = node["op"], node["id"]
            typ = IRType(**node["type"])
            if op == "register":
                values[node_id] = state[node_id]
            elif op == "input":
                name = node["name"]
                if name == "start":
                    values[node_id] = start
                elif name in inputs:
                    values[node_id] = typ.bits(inputs[name])
                else:
                    raise CompileError(f"missing input {name}")
            elif op == "const":
                values[node_id] = node["value"]
            else:
                values[node_id] = _calculate(
                    op,
                    [values[arg] for arg in node["args"]],
                    typ,
                    [IRType(**self.nodes[arg]["type"]) for arg in node["args"]],
                )
        return values

    def run(self, *, max_cycles: int = 100_000, **inputs: int) -> dict[str, int]:
        """Simulate a sequential graph under the start/done handshake.

        The implicit ``start`` input is pulsed on the first cycle; data inputs
        are held stable for the whole run.  The register state advances one tick
        per cycle until the ``done`` output asserts, at which point the settled
        outputs are returned.  Combinational graphs (no registers, no ``done``)
        should use :meth:`evaluate` instead.
        """
        live = self.live_nodes()
        register_nodes = [node for node in live if node["op"] == "register"]
        if not register_nodes:
            raise CompileError("run() is for sequential graphs; use evaluate()")
        done = next((port for port in self.outputs if port["name"] == "done"), None)
        if done is None:
            raise CompileError("sequential graph has no 'done' output to wait on")
        has_start = any(
            node["op"] == "input" and node["name"] == "start" for node in live
        )
        state = {node["id"]: node["init"] for node in register_nodes}
        for cycle in range(max_cycles):
            start = 1 if has_start and cycle == 0 else 0
            values = self._combinational_pass(live, state, inputs, start)
            if cycle > 0 and values[done["node"]]:
                return {
                    port["name"]: IRType(**port["type"]).number(values[port["node"]])
                    for port in self.outputs
                }
            state = {
                node["id"]: (
                    IRType(**node["type"]).bits(values[node["args"][0]])
                    if values[node["args"][1]]
                    else state[node["id"]]
                )
                for node in register_nodes
            }
        raise CompileError(f"simulation did not finish within {max_cycles} cycles")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        nodes = self.live_nodes()
        sequential = any(node["op"] == "register" for node in nodes)
        payload: dict[str, Any] = {
            "schema": "redc.seq.v1" if sequential else "redc.comb.v1",
            "combinational": not sequential,
            "sequential": sequential,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": nodes,
        }
        if sequential:
            # A single implicit clock domain drives every register; the reset is
            # asynchronous and loads each register's `init` value.
            payload["clock"] = {"name": "clk", "reset": "rst", "edge": "posedge"}
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"
