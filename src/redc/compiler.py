"""Lower straight-line RedC functions into the combinational IR.

This first lowering pass handles pure expressions, local assignments, and
inlined function calls. 
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import BOOL, Graph, IRType, Literal, Value, type_from_name
from .parser import AST, CompileError, parse


@dataclass(frozen=True)
class Limits:
    loop_iterations: int = 1024
    nodes: int = 100_000
    steps: int = 100_000
    array_elements: int = 1024
    call_depth: int = 64


@dataclass(frozen=True, slots=True)
class Binding:
    type: IRType
    value: Value | None
    const: bool = False


class Environment:
    def __init__(self, scopes: list[dict[str, Binding]] | None = None) -> None:
        self.scopes = scopes or [{}]

    def push(self) -> None:
        self.scopes.append({})

    def pop(self) -> None:
        self.scopes.pop()

    def lookup(self, name: str, where: AST) -> tuple[dict[str, Binding], Binding]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope, scope[name]
        where.fail(f"undeclared name '{name}'")


BINARY_OPS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "%": "mod",
    "<<": "shl",
    ">>": "shr",
    "&": "and",
    "|": "or",
    "^": "xor",
    "==": "eq",
    "!=": "ne",
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
}


class Compiler:
    def __init__(self, limits: Limits | None = None) -> None:
        self.limits = limits or Limits()
        if any(value <= 0 for value in vars(self.limits).values()):
            raise CompileError("all compilation limits must be positive")
        self.graph = Graph(self.limits.nodes)
        self.functions: dict[str, AST] = {}
        self.globals: dict[str, Binding] = {}
        self.call_stack: list[str] = []
        self.steps = 0
        self.true = self.graph.constant(1, BOOL)
        self.false = self.graph.constant(0, BOOL)

    def tick(self, where: AST) -> None:
        self.steps += 1
        if self.steps > self.limits.steps:
            where.fail(f"compile-time work budget exceeded ({self.limits.steps})")

    def materialize(self, value: Value | Literal, where: AST) -> Value:
        if isinstance(value, Value):
            return value
        number = value.number
        if -(1 << 31) <= number < (1 << 31):
            typ = IRType(32, signed=True)
        elif -(1 << 63) <= number < (1 << 63):
            typ = IRType(64, signed=True)
        elif 0 <= number < (1 << 64):
            typ = IRType(64)
        else:
            where.fail("integer literal exceeds the supported 64-bit range")
        return self.graph.constant(number, typ)

    def convert(self, value: Value | Literal, typ: IRType | None, where: AST) -> Value:
        if typ is None:
            where.fail("void is not a value type")
        if isinstance(value, Literal):
            self.materialize(value, where)
            return self.graph.cast(value, typ)
        return self.graph.cast(value, typ)

    def pair(
        self, left: Value | Literal, right: Value | Literal, where: AST
    ) -> tuple[Value, Value]:
        if isinstance(left, Literal) and isinstance(right, Value):
            return self.convert(left, right.type, where), right
        if isinstance(right, Literal) and isinstance(left, Value):
            return left, self.convert(right, left.type, where)
        left_value = self.materialize(left, where)
        right_value = self.materialize(right, where)
        if left_value.type == right_value.type:
            return left_value, right_value
        if left_value.type.boolean:
            return self.graph.cast(left_value, right_value.type), right_value
        if right_value.type.boolean:
            return left_value, self.graph.cast(right_value, left_value.type)
        if left_value.type.signed != right_value.type.signed:
            where.fail("mixed signed and unsigned operands require an explicit cast")
        typ = IRType(
            max(left_value.type.width, right_value.type.width), left_value.type.signed
        )
        return self.graph.cast(left_value, typ), self.graph.cast(right_value, typ)

    def binary(
        self,
        operator: str,
        left: Value | Literal,
        right: Value | Literal,
        where: AST,
    ) -> Value:
        if operator in {"<<", ">>"}:
            left_value = self.materialize(left, where)
            right_value = self.graph.cast(self.materialize(right, where), IRType(64))
            return self.graph.op(
                BINARY_OPS[operator], left_value.type, left_value, right_value
            )
        left_value, right_value = self.pair(left, right, where)
        typ = (
            BOOL if operator in {"==", "!=", "<", "<=", ">", ">="} else left_value.type
        )
        return self.graph.op(BINARY_OPS[operator], typ, left_value, right_value)

    def expression(self, node: AST, env: Environment) -> Value | Literal | None:
        self.tick(node)
        kind, children = node.kind, node.children
        if kind == "integer":
            spelling = children[0]
            base = (
                16
                if spelling.lower().startswith("0x")
                else 2
                if spelling.lower().startswith("0b")
                else 10
            )
            return Literal(int(spelling, base))
        if kind in {"true_value", "false_value"}:
            return self.true if kind == "true_value" else self.false
        if kind == "variable":
            _, binding = env.lookup(children[0], node)
            if binding.value is None:
                node.fail(f"'{children[0]}' may be read before initialization")
            return binding.value
        if kind == "unary":
            operator, operand = children
            value = self.expression(operand, env)
            if isinstance(value, Literal) and operator in {"+", "-", "~"}:
                return Literal(
                    value.number
                    if operator == "+"
                    else -value.number
                    if operator == "-"
                    else ~value.number
                )
            materialized = self.materialize(value, node)
            if operator == "+":
                return materialized
            if operator == "!":
                return self.graph.op("not", BOOL, self.graph.truth(materialized))
            return self.graph.op(
                "neg" if operator == "-" else "inv", materialized.type, materialized
            )
        if kind == "cast":
            return self.convert(
                self.expression(children[1], env), type_from_name(children[0]), node
            )
        if kind == "binary":
            left_node, operator, right_node = children
            left = self.expression(left_node, env)
            if operator in {"&&", "||"}:
                left_value = self.graph.truth(self.materialize(left, left_node))
                right_value = self.graph.truth(
                    self.materialize(self.expression(right_node, env), right_node)
                )
                return self.graph.op(
                    "and" if operator == "&&" else "or", BOOL, left_value, right_value
                )
            return self.binary(operator, left, self.expression(right_node, env), node)
        if kind == "ternary":
            condition = self.graph.truth(
                self.materialize(self.expression(children[0], env), node)
            )
            yes, no = self.pair(
                self.expression(children[1], env),
                self.expression(children[2], env),
                node,
            )
            return self.graph.mux(condition, yes, no)
        if kind == "call":
            arguments = children[1].children if len(children) > 1 else ()
            return self.call(
                children[0], [self.expression(arg, env) for arg in arguments], node
            )
        node.fail(f"unsupported expression '{kind}' in straight-line lowering")

    @staticmethod
    def _declaration_parts(node: AST) -> tuple[bool, str, str, AST | None]:
        children = list(node.children)
        const = children[0] == "const"
        if const:
            children.pop(0)
        type_name, name = children.pop(0), children.pop(0)
        if any(
            isinstance(child, AST) and child.kind == "array_size" for child in children
        ):
            node.fail("arrays require the structured-control lowering stage")
        initializer = next(
            (
                child
                for child in children
                if isinstance(child, AST) and child.kind == "initializer"
            ),
            None,
        )
        return const, type_name, name, initializer

    def declare(self, node: AST, env: Environment, *, global_: bool = False) -> None:
        const, type_name, name, initializer = self._declaration_parts(node)
        typ = type_from_name(type_name)
        if typ is None:
            node.fail("variables cannot have type void")
        if name in env.scopes[-1]:
            node.fail(f"duplicate declaration '{name}'")
        if global_ and not const:
            node.fail("globals must be const; persistent state is unsupported")
        value = None
        if initializer is not None:
            init = initializer.children[0]
            if init.kind == "array_init":
                init.fail("array initializers require an array declaration")
            value = self.convert(self.expression(init, env), typ, init)
        elif const:
            node.fail("const declarations require an initializer")
        if global_ and value is not None and self.graph.constant_value(value) is None:
            node.fail("global constant initializer must be known at compile time")
        env.scopes[-1][name] = Binding(typ, value, const)

    def assign(
        self, target: AST, rhs: Value | Literal | None, env: Environment, operator: str
    ) -> None:
        if target.kind != "variable":
            target.fail(
                "array assignment requires the structured-control lowering stage"
            )
        name = target.children[0]
        scope, binding = env.lookup(name, target)
        if binding.const:
            target.fail(f"cannot assign to const '{name}'")
        if operator != "=":
            if binding.value is None:
                target.fail(f"'{name}' may be read before initialization")
            rhs = self.binary(operator[:-1], binding.value, rhs, target)
        scope[name] = Binding(binding.type, self.convert(rhs, binding.type, target))

    def statement(
        self, node: AST, env: Environment, return_type: IRType | None
    ) -> Value | None:
        self.tick(node)
        kind, children = node.kind, node.children
        if kind == "block":
            env.push()
            try:
                return self.sequence(children, env, return_type)
            finally:
                env.pop()
        if kind in {"decl_stmt", "declaration"}:
            self.declare(children[0] if kind == "decl_stmt" else node, env)
            return None
        if kind == "assignment":
            self.assign(
                children[0], self.expression(children[2], env), env, children[1]
            )
            return None
        if kind in {"post_update", "pre_update"}:
            target, operator = (
                children if kind == "post_update" else (children[1], children[0])
            )
            self.assign(target, Literal(1), env, "+=" if operator == "++" else "-=")
            return None
        if kind == "expr_stmt":
            self.expression(children[0], env)
            return None
        if kind == "return_stmt":
            if return_type is None and children:
                node.fail("void function cannot return a value")
            if return_type is not None and not children:
                node.fail("non-void function must return a value")
            return (
                self.convert(self.expression(children[0], env), return_type, node)
                if children
                else None
            )
        if kind == "empty_stmt":
            return None
        node.fail(f"unsupported statement '{kind}' in straight-line lowering")

    def sequence(
        self, statements: tuple[AST, ...], env: Environment, return_type: IRType | None
    ) -> Value | None:
        for statement in statements:
            result = self.statement(statement, env, return_type)
            if statement.kind == "return_stmt" or result is not None:
                return result
        return None

    @staticmethod
    def _function_parts(function: AST) -> tuple[str, str, tuple[AST, ...], AST]:
        return_type, name, *rest = function.children
        parameters = rest[0].children if rest[0].kind == "parameters" else ()
        return return_type, name, parameters, rest[-1]

    def call(
        self, name: str, arguments: list[Value | Literal | None], where: AST
    ) -> Value | None:
        if name not in self.functions:
            where.fail(f"unknown function '{name}'")
        if name in self.call_stack:
            where.fail("recursive calls are not supported")
        if len(self.call_stack) >= self.limits.call_depth:
            where.fail("function expansion depth exceeded")
        return_name, _, parameters, body = self._function_parts(self.functions[name])
        if len(arguments) != len(parameters):
            where.fail(f"'{name}' expects {len(parameters)} arguments")
        env = Environment([self.globals.copy(), {}])
        for parameter, argument in zip(parameters, arguments):
            type_name, parameter_name, *shape = parameter.children
            if shape:
                parameter.fail(
                    "array parameters require the structured-control lowering stage"
                )
            typ = type_from_name(type_name)
            if typ is None:
                parameter.fail("parameters cannot have type void")
            env.scopes[-1][parameter_name] = Binding(
                typ, self.convert(argument, typ, parameter)
            )
        self.call_stack.append(name)
        try:
            result = self.sequence(body.children, env, type_from_name(return_name))
        finally:
            self.call_stack.pop()
        if type_from_name(return_name) is not None and result is None:
            where.fail(f"function '{name}' does not return a value")
        return result

    def compile(self, source: str, top: str = "main") -> Graph:
        self.__init__(self.limits)
        ast = parse(source)
        for item in ast.children:
            if item.kind == "function":
                name = item.children[1]
                if name in self.functions:
                    item.fail(f"duplicate function '{name}'")
                self.functions[name] = item
        global_env = Environment([self.globals])
        for item in ast.children:
            if item.kind == "global_decl":
                self.declare(item.children[0], global_env, global_=True)
        if top not in self.functions:
            raise CompileError(f"top function '{top}' not found")
        return_type, _, parameters, _ = self._function_parts(self.functions[top])
        if type_from_name(return_type) is None:
            self.functions[top].fail("top function must return a scalar value")
        arguments = []
        for parameter in parameters:
            type_name, name, *shape = parameter.children
            if shape:
                parameter.fail(
                    "array inputs require the structured-control lowering stage"
                )
            typ = type_from_name(type_name)
            if typ is None:
                parameter.fail("input cannot have type void")
            arguments.append(self.graph.input(f"in_{name}", typ, source_name=name))
        result = self.call(top, arguments, self.functions[top])
        if not isinstance(result, Value):
            self.functions[top].fail("top function did not produce a scalar value")
        self.graph.output("result", result)
        self.graph.validate()
        return self.graph


def compile_source(
    source: str, top: str = "main", limits: Limits | None = None
) -> Graph:
    try:
        return Compiler(limits).compile(source, top)
    except RecursionError:
        raise CompileError(
            "source nesting exceeds the compiler recursion limit"
        ) from None
