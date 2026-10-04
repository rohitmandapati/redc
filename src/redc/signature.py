"""Exact-typed operation signatures, shared by every lowering target.

An :class:`OperationSignature` names one IR operation together with its COMPLETE
result and operand types.  It is target-neutral -- it says nothing about
Minecraft cells, gates or geometry -- so both physical backends key their
implementation lookups by it:

* ``redc.physical`` indexes coarse library cells by signature
  (:class:`~redc.physical.implementation.ImplementationRegistry`);
* ``redc.physical_primitive`` indexes bit-level synthesis recipes by signature
  (:class:`~redc.physical_primitive.synthesis.registry.SynthesisRegistry`).

``add(uint8, uint8) -> uint8`` and ``add(int8, int8) -> int8`` are different keys
even if one implementation could serve both, and ``shr(int8, uint64) -> int8``
(arithmetic) is distinct from ``shr(uint8, uint64) -> uint8`` (logical).  Casts
are keyed by their full source and destination types, never by widths alone.

This module used to live inside :mod:`redc.physical.implementation`, which still
re-exports it unchanged, so ``redc.physical.OperationSignature`` remains the very
same class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ir import BOOL, OPS, Graph, IRType, check_operation_types
from .parser import CompileError

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


__all__ = ["REGISTER_OP", "OperationSignature"]
