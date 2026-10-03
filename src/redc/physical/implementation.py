"""Operation signatures and the physical implementation registry.

This is the lookup layer a future technology mapper queries.  It deliberately
does NOT traverse a :class:`~redc.ir.Graph` or build a
:class:`~redc.physical.netlist.PhysicalNetlist`; it only answers "which physical
implementations exist for this exactly-typed operation?".

    OperationSignature  ->  ImplementationRegistry  ->  zero or more candidates

* :class:`OperationSignature` -- an operation plus its COMPLETE result and operand
  types.  ``add(uint8, uint8) -> uint8`` and ``add(int8, int8) -> int8`` are
  different keys even if one cell could serve both, and
  ``shr(int8, uint64) -> int8`` (arithmetic) is distinct from
  ``shr(uint8, uint64) -> uint8`` (logical).  Casts are keyed by their full
  source and destination types, never by widths alone.
* :class:`PhysicalImplementation` -- the candidate protocol.  Two kinds exist:

  - :class:`DirectCellImplementation` -- exactly one library
    :class:`~redc.physical.components.Component`;
  - :class:`CompositeImplementation` -- a recipe that expands one operation into
    several sub-operations (each itself resolved through the registry), e.g. a
    wide operation built from narrow cells.  Recipes are declared here but are
    only ever *executed* by the future mapper.

* :class:`ImplementationRegistry` -- signature -> candidates, in deterministic
  registration order, so "pick the first direct cell" is a reproducible v1
  policy and smarter (placement-aware) selection can come later.

Two related-but-separate phases must stay separate:

* *Target-neutral optimization* rewrites ``Graph -> Graph`` (``x * 8 -> x << 3``,
  ``x + 0 -> x``, constant propagation).  It belongs with the IR, not here.
* *Physical implementation* turns one live IR operation into one or more
  physical cells.  That is what this registry serves.

The mapper must lower ``graph.live_nodes()``, never ``graph.nodes``: dead IR
operations must never become Minecraft components.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ..ir import BOOL, OPS, Graph, IRType, check_operation_types
from ..parser import CompileError

if TYPE_CHECKING:
    from .components import Component

#: Signature op of a state element (the IR ``register`` node).  IR operations
#: are type-checked by the IR's own rules; other physical-only op names (gates
#: such as ``nand`` that a composite recipe may target) are free-form.
REGISTER_OP = "register"


@dataclass(frozen=True, slots=True)
class OperationSignature:
    """An operation together with its exact result and operand types.

    Operand order is the IR argument order (``mux`` = select, yes, no; shifts =
    value, amount; ``register`` = next, enable).  IR operations are checked with
    :func:`redc.ir.check_operation_types`, so a signature can never claim a
    typing the IR itself would reject (e.g. a one-input shift).
    """

    op: str
    result_type: IRType
    operand_types: tuple[IRType, ...]

    def __post_init__(self) -> None:
        if not self.op:
            raise CompileError("operation signature needs an op")
        if self.op == "cast":
            if len(self.operand_types) != 1:
                raise CompileError("cast signature takes exactly one operand")
        elif self.op in OPS:
            check_operation_types(self.op, self.result_type, self.operand_types)
        elif self.op == REGISTER_OP and self.operand_types != (self.result_type, BOOL):
            raise CompileError(
                "register signature must be register(T next, bool enable) -> T"
            )

    @classmethod
    def of(cls, op: str, result_type: IRType, *operand_types: IRType) -> OperationSignature:
        return cls(op, result_type, tuple(operand_types))

    @classmethod
    def from_ir_node(cls, graph: Graph, node: dict[str, Any]) -> OperationSignature:
        """The signature of one IR operation node (not a traversal: the caller
        decides which nodes to ask about, and must only ask about live ones).

        ``input``/``const`` nodes are boundary cells built on demand, not library
        operations, so they have no signature."""
        if node["op"] in {"input", "const"}:
            raise CompileError(f"{node['op']} nodes are boundary cells, not operations")
        return cls(
            node["op"],
            IRType(**node["type"]),
            tuple(IRType(**graph.nodes[arg]["type"]) for arg in node["args"]),
        )

    def __str__(self) -> str:
        operands = ", ".join(t.name for t in self.operand_types)
        return f"{self.op}({operands}) -> {self.result_type.name}"


class PhysicalImplementation(Protocol):
    """A way to realize one or more operation signatures physically."""

    @property
    def name(self) -> str: ...

    @property
    def signatures(self) -> tuple[OperationSignature, ...]: ...


@dataclass(frozen=True)
class DirectCellImplementation:
    """One operation realized by exactly one library component."""

    component: Component

    @property
    def name(self) -> str:
        return self.component.name

    @property
    def signatures(self) -> tuple[OperationSignature, ...]:
        return self.component.signatures()


class RecipeBuilder(Protocol):
    """What a composite recipe may do while expanding.  The future mapper
    supplies the concrete builder; handles are opaque to the recipe."""

    def emit(self, signature: OperationSignature, *operands: Any) -> Any:
        """Instantiate a sub-operation (resolved through the registry)."""
        ...

    def constant(self, value: int, typ: IRType) -> Any:
        """A hardwired constant operand."""
        ...


#: ``recipe(builder, operands) -> result handle``.
Recipe = Callable[[RecipeBuilder, tuple[Any, ...]], Any]


@dataclass(frozen=True)
class CompositeImplementation:
    """One operation realized as several sub-operations.

    ``applies`` optionally restricts the rewrite to operand values known at map
    time (``None`` entries are runtime operands) -- the hook a later
    constant-shift or strength-reduction rule would use.
    """

    name: str
    signature: OperationSignature
    recipe: Recipe
    applies: Callable[[tuple[int | None, ...]], bool] | None = None

    @property
    def signatures(self) -> tuple[OperationSignature, ...]:
        return (self.signature,)

    def applicable(self, constants: Sequence[int | None]) -> bool:
        return self.applies is None or self.applies(tuple(constants))

    def expand(self, builder: RecipeBuilder, operands: Sequence[Any]) -> Any:
        if len(operands) != len(self.signature.operand_types):
            raise CompileError(
                f"{self.name}: expected {len(self.signature.operand_types)} "
                f"operands, got {len(operands)}"
            )
        return self.recipe(builder, tuple(operands))


class ImplementationRegistry:
    """Signature -> implementation candidates, in registration order."""

    def __init__(self, implementations: Iterable[PhysicalImplementation] = ()) -> None:
        self._by_signature: dict[OperationSignature, list[PhysicalImplementation]] = {}
        for implementation in implementations:
            self.register(implementation)

    def register(self, implementation: PhysicalImplementation) -> None:
        for signature in implementation.signatures:
            bucket = self._by_signature.setdefault(signature, [])
            if any(existing.name == implementation.name for existing in bucket):
                raise CompileError(
                    f"implementation {implementation.name!r} already registered "
                    f"for {signature}"
                )
            bucket.append(implementation)

    def candidates(self, signature: OperationSignature) -> tuple[PhysicalImplementation, ...]:
        """Every implementation of ``signature``, deterministically ordered."""
        return tuple(self._by_signature.get(signature, ()))

    def direct_cells(self, signature: OperationSignature) -> tuple[Component, ...]:
        """The single-cell candidates' components, in registration order."""
        return tuple(
            candidate.component
            for candidate in self.candidates(signature)
            if isinstance(candidate, DirectCellImplementation)
        )

    def supports(self, signature: OperationSignature) -> bool:
        return signature in self._by_signature

    def signatures(self) -> tuple[OperationSignature, ...]:
        """Every signature with at least one candidate, in first-seen order."""
        return tuple(self._by_signature)
