"""Elaborate RedC source into a hardware graph.

Branches execute on separate environments and merge with mux nodes. Function
calls inline. A loop whose continuation condition is known at compile time is
unrolled and emits repeated combinational hardware, exactly as before, and
source-level assignment there only creates a new signal value.

A loop whose bound depends on a runtime value cannot be unrolled, so it is
lowered instead into a clocked finite state machine: its loop-carried variables
become registers, one iteration runs per clock tick, and the enclosing function
gains a ``start``/``done`` handshake. This is what makes stateful programs such
as the Fibonacci example compile. Sequential lowering is intentionally scoped in
v1 to a single, unconditional runtime-bounded loop in the top-level function.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .ir import BOOL, Graph, IRType, Literal, Value, type_from_name
from .parser import AST, CompileError, parse


@dataclass(frozen=True)
class Limits:
    loop_iterations: int = 1024
    nodes: int = 100_000
    steps: int = 100_000
    array_elements: int = 1024
    call_depth: int = 64


@dataclass(frozen=True)
class Binding:
    type: IRType
    values: tuple[Value | None, ...]
    array: bool = False
    const: bool = False


@dataclass
class Path:
    scopes: list[dict[str, Binding]]
    guard: Value
    flow: str = "normal"
    result: Value | None = None

    def copy(self) -> Path:
        return Path(
            [scope.copy() for scope in self.scopes], self.guard, self.flow, self.result
        )

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
        # Sequential-lowering state (populated when a runtime-bounded loop is
        # elaborated into a clocked FSM).
        self.in_sequential_loop = False
        self.start: Value | None = None
        self.done: Value | None = None

    def tick(self, where: AST) -> None:
        self.steps += 1
        if self.steps > self.limits.steps:
            where.fail(f"compile-time work budget exceeded ({self.limits.steps})")

    def materialize(self, value, where: AST) -> Value:
        if isinstance(value, Literal):
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
        if not isinstance(value, Value):
            where.fail(
                "expected a scalar value (arrays and void are not scalar expressions)"
            )
        return value

    def convert(self, value, typ: IRType | None, where: AST) -> Value:
        if typ is None:
            where.fail("void is not a value type")
        if isinstance(value, Literal):
            self.materialize(value, where)
            return self.graph.cast(value, typ)
        return self.graph.cast(self.materialize(value, where), typ)

    def contextual_literal(self, literal: Literal, typ: IRType, where: AST) -> Value:
        low = -(1 << (typ.width - 1)) if typ.signed else 0
        high = (1 << (typ.width - int(typ.signed))) - 1
        if not low <= literal.number <= high:
            where.fail(
                f"literal {literal.number} does not fit the other operand; "
                "widen the operand or explicitly cast the literal"
            )
        return self.convert(literal, typ, where)

    def pair(self, left, right, where: AST) -> tuple[Value, Value]:
        if isinstance(left, Literal) and isinstance(right, Value):
            return self.contextual_literal(left, right.type, where), right
        if isinstance(right, Literal) and isinstance(left, Value):
            return left, self.contextual_literal(right, left.type, where)
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

    def binary(self, operator: str, left, right, where: AST) -> Value:
        if operator in {"<<", ">>"}:
            left_value = self.materialize(left, where)
            right_value = self.materialize(right, where)
            constant = self.graph.constant_value(right_value)
            if constant is not None and right_value.type.number(constant) < 0:
                where.fail("negative constant shift count")
            right_value = self.graph.cast(right_value, IRType(64))
            return self.graph.op(
                BINARY_OPS[operator], left_value.type, left_value, right_value
            )
        left_value, right_value = self.pair(left, right, where)
        typ = (
            BOOL if operator in {"==", "!=", "<", "<=", ">", ">="} else left_value.type
        )
        return self.graph.op(BINARY_OPS[operator], typ, left_value, right_value)

    def constant_integer(self, expression: AST, path: Path, label: str) -> int:
        value = self.materialize(self.expression(expression, path), expression)
        bits = self.graph.constant_value(value)
        if bits is None:
            expression.fail(f"{label} must be known at compile time")
        return value.type.number(bits)

    def expression(self, node: AST, path: Path):
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
            _, binding = path.lookup(children[0], node)
            if any(value is None for value in binding.values):
                node.fail(f"'{children[0]}' may be read before initialization")
            return binding if binding.array else binding.values[0]
        if kind == "index":
            _, binding = path.lookup(children[0], node)
            if not binding.array:
                node.fail(f"'{children[0]}' is not an array")
            index = self.materialize(self.expression(children[1], path), node)
            constant = self.graph.constant_value(index)
            if constant is not None:
                position = index.type.number(constant)
                if not 0 <= position < len(binding.values):
                    node.fail("constant array index out of bounds")
                if binding.values[position] is None:
                    node.fail("array element read before initialization")
                return binding.values[position]
            if any(value is None for value in binding.values):
                node.fail("dynamic array read requires every element to be initialized")
            index = self.index_type(index, len(binding.values))
            result = self.graph.constant(0, binding.type)
            for position, value in enumerate(binding.values):
                equal = self.graph.op(
                    "eq", BOOL, index, self.graph.constant(position, index.type)
                )
                result = self.graph.mux(equal, value, result)
            return result
        if kind == "unary":
            operator, operand = children
            value = self.expression(operand, path)
            if isinstance(value, Literal) and operator in {"+", "-", "~"}:
                return Literal(
                    value.number
                    if operator == "+"
                    else -value.number
                    if operator == "-"
                    else ~value.number
                )
            value = self.materialize(value, node)
            if operator == "+":
                return value
            if operator == "!":
                return self.graph.op("not", BOOL, self.graph.truth(value))
            return self.graph.op("neg" if operator == "-" else "inv", value.type, value)
        if kind == "cast":
            return self.convert(
                self.expression(children[1], path), type_from_name(children[0]), node
            )
        if kind == "binary":
            left_node, operator, right_node = children
            left = self.expression(left_node, path)
            if operator in {"&&", "||"}:
                left_value = self.graph.truth(self.materialize(left, left_node))
                constant = self.graph.constant_value(left_value)
                if constant is not None and (
                    (operator == "&&" and not constant)
                    or (operator == "||" and constant)
                ):
                    return left_value
                right_value = self.graph.truth(
                    self.materialize(self.expression(right_node, path), right_node)
                )
                return self.graph.op(
                    "and" if operator == "&&" else "or", BOOL, left_value, right_value
                )
            return self.binary(operator, left, self.expression(right_node, path), node)
        if kind == "ternary":
            condition = self.graph.truth(
                self.materialize(self.expression(children[0], path), node)
            )
            constant = self.graph.constant_value(condition)
            if constant is not None:
                return self.expression(children[1] if constant else children[2], path)
            yes, no = self.pair(
                self.expression(children[1], path),
                self.expression(children[2], path),
                node,
            )
            return self.graph.mux(condition, yes, no)
        if kind == "call":
            arguments = children[1].children if len(children) > 1 else ()
            return self.call(
                children[0],
                [self.expression(argument, path) for argument in arguments],
                node,
            )
        node.fail(f"unsupported expression {kind}")

    def index_type(self, index: Value, size: int) -> Value:
        width = max(index.type.width, size.bit_length() + int(index.type.signed))
        return self.graph.cast(index, IRType(width, index.type.signed))

    def declare(self, node: AST, path: Path, *, global_: bool = False) -> None:
        children = list(node.children)
        const = children[0] == "const"
        if const:
            children.pop(0)
        typ = type_from_name(children.pop(0))
        name = children.pop(0)
        if typ is None:
            node.fail("variables cannot have type void")
        if name in path.scopes[-1]:
            node.fail(f"duplicate declaration '{name}'")
        size_node = next(
            (
                child
                for child in children
                if isinstance(child, AST) and child.kind == "array_size"
            ),
            None,
        )
        init_node = next(
            (
                child
                for child in children
                if isinstance(child, AST) and child.kind == "initializer"
            ),
            None,
        )
        array = size_node is not None
        size = (
            self.constant_integer(size_node.children[0], path, "array length")
            if array
            else 1
        )
        if not 1 <= size <= self.limits.array_elements:
            node.fail(f"array length must be 1..{self.limits.array_elements}")
        if global_ and not const:
            node.fail("globals must be const; persistent state is unsupported")
        values: list[Value | None] = [None] * size
        if init_node:
            initializer = init_node.children[0]
            if array:
                if initializer.kind != "array_init":
                    node.fail("array initialization requires { ... }")
                if len(initializer.children) > size:
                    node.fail("too many array initializer elements")
                values = [
                    self.convert(self.expression(item, path), typ, item)
                    for item in initializer.children
                ]
                values += [self.graph.constant(0, typ)] * (size - len(values))
            else:
                if initializer.kind == "array_init":
                    node.fail("scalar initialization requires an expression")
                values = [
                    self.convert(self.expression(initializer, path), typ, initializer)
                ]
        elif const:
            node.fail("const declarations require an initializer")
        if global_ and any(
            value is None or self.graph.constant_value(value) is None
            for value in values
        ):
            node.fail("global constant initializer must be known at compile time")
        path.scopes[-1][name] = Binding(typ, tuple(values), array, const)

    def assign(self, target: AST, rhs, path: Path, operator: str = "=") -> None:
        name = target.children[0]
        scope, binding = path.lookup(name, target)
        if binding.const:
            target.fail(f"cannot assign to const '{name}'")
        if operator != "=":
            rhs = self.binary(operator[:-1], self.expression(target, path), rhs, target)
        rhs = self.convert(rhs, binding.type, target)
        if target.kind == "variable":
            if binding.array:
                target.fail("whole-array assignment is unsupported")
            values = (rhs,)
        else:
            if not binding.array:
                target.fail(f"'{name}' is not an array")
            index = self.materialize(self.expression(target.children[1], path), target)
            constant = self.graph.constant_value(index)
            values = list(binding.values)
            if constant is not None:
                position = index.type.number(constant)
                if not 0 <= position < len(values):
                    target.fail("constant array index out of bounds")
                values[position] = rhs
            else:
                if any(value is None for value in values):
                    target.fail("dynamic array write requires an initialized array")
                index = self.index_type(index, len(values))
                for position, value in enumerate(values):
                    equal = self.graph.op(
                        "eq", BOOL, index, self.graph.constant(position, index.type)
                    )
                    values[position] = self.graph.mux(equal, rhs, value)
        scope[name] = replace(binding, values=tuple(values))

    def merge(self, paths: list[Path]) -> list[Path]:
        groups: dict[tuple, Path] = {}
        for path in paths:
            if self.graph.constant_value(path.guard) == 0:
                continue
            shape = tuple(tuple(sorted(scope)) for scope in path.scopes)
            key = (path.flow, shape)
            if key not in groups:
                groups[key] = path.copy()
                continue
            old = groups[key]
            for old_scope, new_scope in zip(old.scopes, path.scopes):
                for name, binding in old_scope.items():
                    other = new_scope[name]
                    values = tuple(
                        None
                        if left is None or right is None
                        else self.graph.mux(path.guard, right, left)
                        for left, right in zip(binding.values, other.values)
                    )
                    old_scope[name] = replace(binding, values=values)
            if path.result is not None:
                old.result = (
                    path.result
                    if old.result is None
                    else self.graph.mux(path.guard, path.result, old.result)
                )
            old.guard = self.graph.op("or", BOOL, old.guard, path.guard)
        return list(groups.values())

    def sequence(
        self,
        statements: tuple[AST, ...] | list[AST],
        paths: list[Path],
        loop: int = 0,
        switch: int = 0,
    ) -> list[Path]:
        for statement in statements:
            next_paths = []
            for path in paths:
                next_paths.extend(
                    self.statement(statement, path, loop, switch)
                    if path.flow == "normal"
                    else [path]
                )
            paths = self.merge(next_paths)
        return paths

    def scoped(self, statement: AST, path: Path, loop: int, switch: int) -> list[Path]:
        path = path.copy()
        path.scopes.append({})
        body = statement.children if statement.kind == "block" else (statement,)
        paths = self.sequence(body, [path], loop, switch)
        for result in paths:
            result.scopes.pop()
        return paths

    def statement(
        self, node: AST, path: Path, loop: int = 0, switch: int = 0
    ) -> list[Path]:
        self.tick(node)
        kind, children = node.kind, node.children
        if kind == "block":
            return self.scoped(node, path, loop, switch)
        if kind in {"decl_stmt", "declaration"}:
            self.declare(children[0] if kind == "decl_stmt" else node, path)
        elif kind == "assignment":
            self.assign(
                children[0], self.expression(children[2], path), path, children[1]
            )
        elif kind in {"post_update", "pre_update"}:
            target, operator = (
                children if kind == "post_update" else (children[1], children[0])
            )
            self.assign(target, Literal(1), path, "+=" if operator == "++" else "-=")
        elif kind == "expr_stmt":
            self.expression(children[0], path)
        elif kind == "return_stmt":
            return_type = type_from_name(
                self.functions[self.call_stack[-1]].children[0]
            )
            if return_type is None and children:
                node.fail("void function cannot return a value")
            if return_type is not None and not children:
                node.fail("non-void function must return a value")
            path.result = (
                self.convert(self.expression(children[0], path), return_type, node)
                if children
                else None
            )
            path.flow = "return"
        elif kind == "break_stmt":
            if not (loop or switch):
                node.fail("break requires a loop or switch")
            path.flow = "break"
        elif kind == "continue_stmt":
            if not loop:
                node.fail("continue requires a loop")
            path.flow = "continue"
        elif kind == "if_stmt":
            condition = self.graph.truth(
                self.materialize(self.expression(children[0], path), node)
            )
            constant = self.graph.constant_value(condition)
            if constant is not None:
                if constant:
                    return self.scoped(children[1], path, loop, switch)
                return (
                    self.scoped(children[2], path, loop, switch)
                    if len(children) > 2
                    else [path]
                )
            yes, no = path.copy(), path.copy()
            yes.guard = self.graph.op("and", BOOL, path.guard, condition)
            no.guard = self.graph.op(
                "and", BOOL, path.guard, self.graph.op("not", BOOL, condition)
            )
            return self.merge(
                self.scoped(children[1], yes, loop, switch)
                + (
                    self.scoped(children[2], no, loop, switch)
                    if len(children) > 2
                    else [no]
                )
            )
        elif kind in {"for_stmt", "while_stmt", "do_stmt"}:
            return self.unroll(node, path, loop, switch)
        elif kind == "switch_stmt":
            return self.lower_switch(node, path, loop, switch)
        elif kind != "empty_stmt":
            node.fail(f"unsupported statement {kind}")
        return [path]

    def unroll(self, node: AST, path: Path, loop: int, switch: int) -> list[Path]:
        children, kind = node.children, node.kind
        path = path.copy()
        path.scopes.append({})
        step = None
        if kind == "for_stmt":
            initializer, condition, step_node, body = children
            if initializer.children:
                self.statement(initializer.children[0], path, loop, switch)
            condition_expression = condition.children[0] if condition.children else None
            step = step_node.children[0] if step_node.children else None
        elif kind == "while_stmt":
            condition_expression, body = children
        else:
            body, condition_expression = children
        if kind != "do_stmt" and condition_expression is not None:
            probe = self.materialize(
                self.expression(condition_expression, path), condition_expression
            )
            if self.graph.constant_value(probe) is None:
                result = self.lower_sequential_loop(
                    node, path, condition_expression, step, body, loop, switch
                )
                for output in result:
                    output.scopes.pop()
                return self.merge(result)
        active, done = [path], []
        iteration = 0
        while active:
            running = []
            for current in active:
                if (
                    kind == "do_stmt" and iteration == 0
                ) or condition_expression is None:
                    test = 1
                else:
                    test = self.constant_integer(
                        condition_expression,
                        current,
                        "loop condition (use a fixed bound and if/break for runtime decisions)",
                    )
                if test:
                    running.append(current)
                else:
                    done.append(current)
            if not running:
                break
            if iteration >= self.limits.loop_iterations:
                node.fail(
                    f"loop unroll limit exceeded ({self.limits.loop_iterations}); "
                    "check the bound, increment, and integer overflow"
                )
            next_active = []
            for current in running:
                for output in self.scoped(body, current, loop + 1, switch):
                    if output.flow == "return":
                        done.append(output)
                    elif output.flow == "break":
                        output.flow = "normal"
                        done.append(output)
                    else:
                        output.flow = "normal"
                        if step:
                            self.statement(step, output, loop + 1, switch)
                        next_active.append(output)
            active = self.merge(next_active)
            iteration += 1
        for output in done:
            output.scopes.pop()
        return self.merge(done)

    def collect_assigned(self, node: AST, names: set[str]) -> None:
        """Record every name that a statement (or its descendants) assigns to."""
        if node.kind == "assignment":
            target = node.children[0]
            if target.kind in {"variable", "index"}:
                names.add(target.children[0])
        elif node.kind in {"post_update", "pre_update"}:
            target = node.children[0 if node.kind == "post_update" else 1]
            if target.kind in {"variable", "index"}:
                names.add(target.children[0])
        for child in node.children:
            if isinstance(child, AST):
                self.collect_assigned(child, names)

    def control_start(self) -> Value:
        """The implicit ``start`` handshake input, created on first use."""
        if self.start is None:
            self.start = self.graph.input("start", BOOL, source_name="start")
        return self.start

    def lower_sequential_loop(
        self,
        node: AST,
        entry: Path,
        condition_expression: AST,
        step: AST | None,
        body: AST,
        loop: int,
        switch: int,
    ) -> list[Path]:
        """Lower a runtime-bounded loop into a clocked FSM with registers.

        Loop-carried variables become registers seeded (on ``start``) with their
        pre-loop values; the loop body is elaborated once to derive each
        register's next-iteration value; a ``running`` register plus the loop
        condition sequence one iteration per clock and raise ``done`` on exit.
        """
        if len(self.call_stack) != 1:
            node.fail(
                "runtime-bounded loops are only supported in the top-level function "
                "in v1 (called functions must use compile-time bounds)"
            )
        if self.graph.constant_value(entry.guard) != 1:
            node.fail(
                "a runtime-bounded loop must run unconditionally; placing one inside "
                "a runtime branch is not supported in v1"
            )
        if self.in_sequential_loop:
            node.fail("nested runtime-bounded loops are not supported in v1")
        if self.done is not None:
            node.fail("only one runtime-bounded loop is supported per program in v1")

        assigned: set[str] = set()
        self.collect_assigned(body, assigned)
        if step is not None:
            self.collect_assigned(step, assigned)

        # A carried variable is one that is assigned in the loop and already
        # exists outside the body; names assigned but declared inside the body
        # are fresh each iteration and never resolve here.
        carried: list[tuple[str, dict[str, Binding], Binding]] = []
        for name in sorted(assigned):
            for scope in reversed(entry.scopes):
                if name in scope:
                    carried.append((name, scope, scope[name]))
                    break

        self.in_sequential_loop = True
        try:
            registers: dict[str, Value] = {}
            preloop: dict[str, Value] = {}
            for name, _, binding in carried:
                if binding.array:
                    node.fail(
                        f"array variable '{name}' cannot be carried through a "
                        "runtime-bounded loop in v1"
                    )
                if binding.values[0] is None:
                    node.fail(
                        f"loop variable '{name}' must be initialized before a "
                        "runtime-bounded loop"
                    )
                preloop[name] = binding.values[0]
                registers[name] = self.graph.register(binding.type)

            # Elaborate the condition and one iteration against the register
            # outputs (the "current" state).
            state = entry.copy()
            for name, _, binding in carried:
                scope, current = state.lookup(name, node)
                scope[name] = replace(current, values=(registers[name],))

            condition = self.graph.truth(
                self.materialize(self.expression(condition_expression, state), node)
            )

            body_paths = self.scoped(body, state, loop + 1, switch)
            for output in body_paths:
                if output.flow != "normal":
                    node.fail(
                        "break, continue, and return inside a runtime-bounded loop "
                        "are not supported in v1"
                    )
            body_paths = self.merge(body_paths)
            if len(body_paths) != 1:
                node.fail("could not converge the loop body into a single next state")
            body_state = body_paths[0]
            if step is not None:
                self.statement(step, body_state, loop + 1, switch)

            # Wire the FSM: start loads the pre-loop values, each running tick
            # loads the next-iteration values, exit holds the final values.
            start = self.control_start()
            running = self.graph.register(BOOL)
            active = self.graph.op("and", BOOL, running, condition)
            enable = self.graph.op("or", BOOL, start, active)
            for name, _, _ in carried:
                register = registers[name]
                next_value = self.graph.mux(
                    start, preloop[name], body_state.lookup(name, node)[1].values[0]
                )
                self.graph.set_register(register, next_value, enable)
            self.graph.set_register(running, enable, self.true)
            self.done = self.graph.op(
                "and", BOOL, running, self.graph.op("not", BOOL, condition)
            )

            # After the loop the carried variables read from their registers,
            # which hold the settled final values when `done` asserts.
            for name, scope, binding in carried:
                scope[name] = replace(binding, values=(registers[name],))
        finally:
            self.in_sequential_loop = False
        return [entry]

    def lower_switch(self, node: AST, path: Path, loop: int, switch: int) -> list[Path]:
        selector = self.materialize(self.expression(node.children[0], path), node)
        arms = node.children[1:]
        cases = []
        default = None
        seen = set()
        for index, arm in enumerate(arms):
            if arm.kind == "default_arm":
                if default is not None:
                    arm.fail("duplicate default label")
                default = index
            else:
                raw = self.constant_integer(arm.children[0], path, "case label")
                bits = raw & selector.type.mask
                if bits in seen:
                    arm.fail("duplicate case label after conversion to switch type")
                seen.add(bits)
                equal = self.graph.op(
                    "eq", BOOL, selector, self.graph.constant(bits, selector.type)
                )
                cases.append((index, equal))
        matched = self.false
        for _, condition in cases:
            matched = self.graph.op("or", BOOL, matched, condition)
        cases.append((default, self.graph.op("not", BOOL, matched)))
        results = []
        for start, condition in cases:
            current = path.copy()
            current.guard = self.graph.op("and", BOOL, current.guard, condition)
            if self.graph.constant_value(current.guard) == 0:
                continue
            if start is None:
                results.append(current)
                continue
            current.scopes.append({})
            statements = []
            for arm in arms[start:]:
                body = arm.children[1:] if arm.kind == "case_arm" else arm.children
                if any(statement.kind == "decl_stmt" for statement in body):
                    arm.fail("put switch-local declarations inside { } blocks")
                statements.extend(body)
            outputs = self.sequence(statements, [current], loop, switch + 1)
            for output in outputs:
                output.scopes.pop()
                if output.flow == "break":
                    output.flow = "normal"
            results.extend(outputs)
        return self.merge(results)

    def call(self, name: str, arguments: list, where: AST):
        if name not in self.functions:
            where.fail(f"unknown function '{name}'")
        if name in self.call_stack:
            where.fail("recursive calls are not supported")
        if len(self.call_stack) >= self.limits.call_depth:
            where.fail("function expansion depth exceeded")
        function = self.functions[name]
        return_type_name, _, *rest = function.children
        parameters = rest[0].children if rest[0].kind == "parameters" else ()
        body = rest[-1]
        if len(arguments) != len(parameters):
            where.fail(f"'{name}' expects {len(parameters)} arguments")
        path = Path([self.globals.copy(), {}], self.true)
        for parameter, argument in zip(parameters, arguments):
            type_name, parameter_name, *shape = parameter.children
            typ = type_from_name(type_name)
            if typ is None:
                parameter.fail("parameters cannot have type void")
            if parameter_name in path.scopes[-1]:
                parameter.fail("duplicate parameter name")
            if shape:
                size = self.constant_integer(
                    shape[0].children[0], path, "parameter array length"
                )
                if (
                    not isinstance(argument, Binding)
                    or not argument.array
                    or len(argument.values) != size
                ):
                    parameter.fail("array argument has the wrong shape")
                values = tuple(
                    self.convert(value, typ, parameter) for value in argument.values
                )
                binding = Binding(typ, values, True)
            else:
                binding = Binding(typ, (self.convert(argument, typ, parameter),))
            path.scopes[-1][parameter_name] = binding
        self.call_stack.append(name)
        try:
            outputs = self.sequence(body.children, [path])
        finally:
            self.call_stack.pop()
        return_type = type_from_name(return_type_name)
        if return_type is None:
            return None
        if any(output.flow != "return" for output in outputs):
            function.fail(f"function '{name}' does not return on every reachable path")
        if not outputs:
            function.fail(f"function '{name}' has no return")
        result = outputs[0].result
        for output in outputs[1:]:
            result = self.graph.mux(output.guard, output.result, result)
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
        globals_path = Path([self.globals], self.true)
        for item in ast.children:
            if item.kind == "global_decl":
                self.declare(item.children[0], globals_path, global_=True)
        if set(self.functions) & set(self.globals):
            raise CompileError("a global and a function cannot have the same name")
        if top not in self.functions:
            raise CompileError(f"top function '{top}' not found")
        function = self.functions[top]
        if type_from_name(function.children[0]) is None:
            function.fail("top function must return a scalar value")
        parameters = (
            function.children[2].children
            if function.children[2].kind == "parameters"
            else ()
        )
        arguments = []
        for parameter in parameters:
            type_name, name, *shape = parameter.children
            typ = type_from_name(type_name)
            if typ is None:
                parameter.fail("input cannot have type void")
            if shape:
                size = self.constant_integer(
                    shape[0].children[0], globals_path, "input array length"
                )
                if not 1 <= size <= self.limits.array_elements:
                    parameter.fail("input array too large or empty")
                values = tuple(
                    self.graph.input(f"in_{name}_{index}", typ, source_name=name)
                    for index in range(size)
                )
                arguments.append(Binding(typ, values, True))
            else:
                arguments.append(self.graph.input(f"in_{name}", typ, source_name=name))
        port_names = [port["name"] for port in self.graph.inputs]
        if len(set(port_names)) != len(port_names):
            function.fail("flattened input port names collide; rename parameters")
        result = self.call(top, arguments, function)
        self.graph.output("result", result)
        if self.done is not None:
            # A runtime-bounded loop made the top function sequential: expose the
            # handshake so a caller knows when `result` has settled.
            self.graph.output("done", self.done)
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
