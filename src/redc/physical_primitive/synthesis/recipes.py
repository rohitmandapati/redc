"""Family recipes for every IR operation, registered into
:data:`DEFAULT_SYNTHESIS`.

Each recipe is ``(builder, signature, operands) -> result bits`` (LSB first)
and emits ONLY basis gates (AND / OR / XOR / NOT) plus per-node constants.  The
recipes are thin adapters; the structures live in the helper modules:

* :mod:`.logic`      -- ``and`` / ``or`` / ``xor`` (one gate per bit), ``inv``,
  ``not``, ``mux`` (shared ``NOT(sel)``);
* :mod:`.cast`       -- ``cast`` (wiring, plus an OR tree for ``-> bool``);
* :mod:`.arithmetic` -- ``add`` / ``sub`` / ``neg`` (one ripple-carry adder),
  ``mul`` (array multiplier), ``div`` / ``mod`` (restoring division);
* :mod:`.compare`    -- ``eq`` / ``ne`` (XOR + OR tree) and ``lt`` / ``le`` /
  ``gt`` / ``ge`` (MSB-first less-than chain);
* :mod:`.shift`      -- ``shl`` / ``shr`` (barrel stages + 64-bit range check).

Where the structure depends on signedness the op gets one family entry per
case, selected by an ``applies`` predicate (``div`` -> ``restoring_divide_signed``
vs ``restoring_divide_unsigned``), so the registry -- and a synthesis trace's
``recipe`` field -- names the network actually built.  Every op in
:data:`redc.ir.OPS` has a recipe for every typing the IR allows.
"""

from __future__ import annotations

from collections.abc import Callable

from ...signature import OperationSignature
from ..netlist import BitVector
from .arithmetic import add, divide, modulo, multiply, negate, subtract
from .builder import PrimitiveBuilder
from .cast import CAST_KINDS, cast, cast_kind
from .compare import comparison, operands_signed
from .logic import bitwise, invert, mux_vector
from .registry import SynthesisRecipe, SynthesisRegistry
from .shift import shift_left, shift_right

Predicate = Callable[[OperationSignature], bool]


def synth_and(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return bitwise(b, b.and_, ops[0], ops[1], role="and")


def synth_or(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return bitwise(b, b.or_, ops[0], ops[1], role="or")


def synth_xor(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return bitwise(b, b.xor, ops[0], ops[1], role="xor")


def synth_inv(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return invert(b, ops[0], role="inv")


def synth_not(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return (b.not_(ops[0][0], role="logical_not", bit=0),)


def synth_mux(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    sel, yes, no = ops
    with b.scope("mux", kind="helper"):
        return mux_vector(b, sel[0], yes, no, role="mux")


def synth_cast(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return cast(b, ops[0], sig.operand_types[0], sig.result_type)


def synth_neg(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return negate(b, ops[0])


def synth_add(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return add(b, ops[0], ops[1])


def synth_sub(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return subtract(b, ops[0], ops[1])


def synth_mul(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return multiply(b, ops[0], ops[1])


def synth_div(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return divide(b, ops[0], ops[1], signed=sig.result_type.signed)


def synth_mod(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return modulo(b, ops[0], ops[1], signed=sig.result_type.signed)


def synth_shl(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return shift_left(b, ops[0], ops[1])


def synth_shr(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return shift_right(b, ops[0], ops[1], signed=sig.result_type.signed)


def synth_compare(b: PrimitiveBuilder, sig: OperationSignature, ops: tuple[BitVector, ...]) -> BitVector:
    return (comparison(b, sig, ops[0], ops[1]),)


def _signed_result(sig: OperationSignature) -> bool:
    return sig.result_type.signed


def _unsigned_result(sig: OperationSignature) -> bool:
    return not sig.result_type.signed


def _unsigned_operands(sig: OperationSignature) -> bool:
    return not operands_signed(sig)


def _is_cast(kind: str) -> Predicate:
    def applies(sig: OperationSignature) -> bool:
        return cast_kind(sig.operand_types[0], sig.result_type) == kind

    return applies


def _by_signedness(
    registry: SynthesisRegistry,
    op: str,
    recipe: SynthesisRecipe,
    name: str,
    *,
    signed: Predicate,
    unsigned: Predicate,
) -> None:
    registry.register_family(op, recipe, name=f"{name}_signed", applies=signed)
    registry.register_family(op, recipe, name=f"{name}_unsigned", applies=unsigned)


def build_default_registry() -> SynthesisRegistry:
    """A fresh registry with one family recipe (per signedness case) for every IR op."""
    registry = SynthesisRegistry()
    registry.register_family("and", synth_and, name="bitwise_and")
    registry.register_family("or", synth_or, name="bitwise_or")
    registry.register_family("xor", synth_xor, name="bitwise_xor")
    registry.register_family("inv", synth_inv, name="bitwise_inv")
    registry.register_family("not", synth_not, name="logical_not")
    registry.register_family("mux", synth_mux, name="bitwise_mux")
    for kind in CAST_KINDS:
        registry.register_family("cast", synth_cast, name=f"cast_{kind}", applies=_is_cast(kind))
    registry.register_family("neg", synth_neg, name="twos_complement_negate")
    registry.register_family("add", synth_add, name="ripple_carry_add")
    registry.register_family("sub", synth_sub, name="ripple_carry_subtract")
    registry.register_family("mul", synth_mul, name="array_multiply")
    _by_signedness(
        registry, "div", synth_div, "restoring_divide", signed=_signed_result, unsigned=_unsigned_result
    )
    _by_signedness(
        registry, "mod", synth_mod, "restoring_modulo", signed=_signed_result, unsigned=_unsigned_result
    )
    registry.register_family("shl", synth_shl, name="barrel_shift_left")
    registry.register_family("shr", synth_shr, name="barrel_shift_right_arithmetic", applies=_signed_result)
    registry.register_family("shr", synth_shr, name="barrel_shift_right_logical", applies=_unsigned_result)
    registry.register_family("eq", synth_compare, name="equal_xor_or_tree")
    registry.register_family("ne", synth_compare, name="not_equal_xor_or_tree")
    for op in ("lt", "le", "gt", "ge"):
        _by_signedness(
            registry, op, synth_compare, f"less_chain_{op}", signed=operands_signed, unsigned=_unsigned_operands
        )
    return registry


#: The registry :func:`~redc.physical_primitive.synthesis.lower.synthesize_to_primitives`
#: uses by default.
DEFAULT_SYNTHESIS = build_default_registry()

__all__ = ["DEFAULT_SYNTHESIS", "build_default_registry"]
