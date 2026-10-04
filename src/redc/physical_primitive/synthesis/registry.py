"""The synthesis-recipe registry, keyed by exact :class:`OperationSignature`.

A recipe answers "which one-bit gates compute this exactly-typed operation?":

    recipe(builder, signature, operands) -> result BitVector

``operands`` are LSB-first bit vectors in IR operand order; ``signature``
carries the full result and operand types (so a recipe can branch on
signedness).  Recipes emit only basis gates through the builder and know
nothing about netlist ids, Minecraft or geometry.

Resolution order for a signature:

1. an *exact* recipe registered for that very signature (a specialized
   ``add(uint8, uint8) -> uint8`` could be registered here later);
2. the first applicable *family* recipe registered for the signature's op --
   a generic recipe valid for every typing the IR allows, optionally narrowed
   by an ``applies(signature)`` predicate.

The same exact-typed key concept as the coarse backend's
:class:`~redc.physical.implementation.ImplementationRegistry`, but resolving to
bit-level synthesis recipes instead of library cells.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ...parser import CompileError
from ...signature import OperationSignature
from ..netlist import BitVector
from .builder import PrimitiveBuilder

#: ``recipe(builder, signature, operands) -> result bits`` (LSB first).
SynthesisRecipe = Callable[[PrimitiveBuilder, OperationSignature, tuple[BitVector, ...]], BitVector]


@dataclass(frozen=True)
class RecipeEntry:
    """One registered recipe.  ``signature`` is set for exact entries; family
    entries carry the ``op`` and an optional ``applies`` predicate."""

    name: str
    op: str
    recipe: SynthesisRecipe
    signature: OperationSignature | None = None
    applies: Callable[[OperationSignature], bool] | None = None

    def matches(self, signature: OperationSignature) -> bool:
        if self.signature is not None:
            return self.signature == signature
        return signature.op == self.op and (self.applies is None or self.applies(signature))

    def synthesize(
        self, builder: PrimitiveBuilder, signature: OperationSignature, operands: tuple[BitVector, ...]
    ) -> BitVector:
        if len(operands) != len(signature.operand_types):
            raise CompileError(
                f"{self.name}: expected {len(signature.operand_types)} operands, got {len(operands)}"
            )
        for operand, typ in zip(operands, signature.operand_types):
            if len(operand) != typ.width:
                raise CompileError(
                    f"{self.name}: operand of type {typ.name} has {len(operand)} bits"
                )
        result = self.recipe(builder, signature, operands)
        if len(result) != signature.result_type.width:
            raise CompileError(
                f"{self.name}: produced {len(result)} bits for a {signature.result_type.name} result"
            )
        return tuple(result)


class SynthesisRegistry:
    """Signature -> synthesis recipe, deterministically ordered."""

    def __init__(self) -> None:
        self._exact: dict[OperationSignature, RecipeEntry] = {}
        self._families: dict[str, list[RecipeEntry]] = {}

    def register(self, signature: OperationSignature, recipe: SynthesisRecipe, *, name: str) -> None:
        """A recipe for exactly ``signature`` (wins over family recipes)."""
        if signature in self._exact:
            raise CompileError(f"a recipe is already registered for {signature}")
        self._exact[signature] = RecipeEntry(name, signature.op, recipe, signature=signature)

    def register_family(
        self,
        op: str,
        recipe: SynthesisRecipe,
        *,
        name: str,
        applies: Callable[[OperationSignature], bool] | None = None,
    ) -> None:
        """A generic recipe for every (applicable) typing of ``op``."""
        bucket = self._families.setdefault(op, [])
        if any(entry.name == name for entry in bucket):
            raise CompileError(f"family recipe {name!r} is already registered for {op}")
        bucket.append(RecipeEntry(name, op, recipe, applies=applies))

    def resolve(self, signature: OperationSignature) -> RecipeEntry:
        exact = self._exact.get(signature)
        if exact is not None:
            return exact
        for entry in self._families.get(signature.op, ()):
            if entry.matches(signature):
                return entry
        raise CompileError(f"no primitive synthesis recipe for {signature}")

    def supports(self, signature: OperationSignature) -> bool:
        try:
            self.resolve(signature)
        except CompileError:
            return False
        return True

    def ops(self) -> frozenset[str]:
        """Every op with at least one recipe."""
        return frozenset(self._families) | frozenset(s.op for s in self._exact)

    def entries(self) -> tuple[RecipeEntry, ...]:
        return tuple(self._exact.values()) + tuple(e for b in self._families.values() for e in b)


__all__ = ["RecipeEntry", "SynthesisRecipe", "SynthesisRegistry"]
