"""The ``cast`` recipe: re-typing a bit vector -- almost always pure wiring.

The IR rule (:func:`redc.ir._calculate`) is ``target is bool ? number(x) != 0
: number(x) mod 2**W_dst``, where ``number`` interprets ``x`` with the SOURCE
type.  Bit-blasted, each case (:func:`cast_kind`) is:

* ``to_bool`` -- any source to ``bool``: OR-reduce every source bit (a balanced
  tree); a one-bit source is its own truth value, so it needs no gate.
* ``reinterpret`` -- same width, signedness only (``bool`` -> ``uint1`` /
  ``int1`` included): the very same bit handles, no gate.
* ``truncate`` -- narrowing: the low ``W_dst`` source bits.
* ``zero_extend`` -- widening an unsigned source (``bool`` included): the
  node's constant-0 signal fills the new high bits.
* ``sign_extend`` -- widening a signed source: the source sign bit ``x[W-1]``
  is repeated -- the SAME signal handle, so one net simply fans out further.

Only ``to_bool`` emits gates.  The recipes module registers one family entry
per kind, so a synthesis trace names the case that was applied.
"""

from __future__ import annotations

from ...ir import IRType
from ...parser import CompileError
from ..netlist import Bit, BitVector
from .builder import PrimitiveBuilder
from .logic import or_reduce

CAST_KINDS = ("to_bool", "reinterpret", "truncate", "sign_extend", "zero_extend")


def cast_kind(source: IRType, target: IRType) -> str:
    """Which of :data:`CAST_KINDS` a ``source -> target`` cast is."""
    if target.boolean:
        return "to_bool"
    if target.width == source.width:
        return "reinterpret"
    if target.width < source.width:
        return "truncate"
    return "sign_extend" if source.signed else "zero_extend"


def truth(b: PrimitiveBuilder, x: BitVector) -> Bit:
    """``x != 0``: a balanced OR tree (no gate for a one-bit ``x``)."""
    if len(x) == 1:
        return x[0]
    with b.scope("truth", kind="helper", width=len(x)):
        return or_reduce(b, x, role="truth_or")


def truncate(x: BitVector, width: int) -> BitVector:
    """The low ``width`` bits (pure wiring)."""
    return tuple(x[:width])


def zero_extend(b: PrimitiveBuilder, x: BitVector, width: int) -> BitVector:
    """``x`` followed by ``width - len(x)`` copies of the node's constant 0."""
    return tuple(x) + (b.const(False),) * (width - len(x))


def sign_extend(x: BitVector, width: int) -> BitVector:
    """``x`` followed by ``width - len(x)`` copies of its own sign bit."""
    return tuple(x) + (x[-1],) * (width - len(x))


def cast(b: PrimitiveBuilder, x: BitVector, source: IRType, target: IRType) -> BitVector:
    """``x`` (of type ``source``) converted to ``target`` with the IR's semantics."""
    if len(x) != source.width:
        raise CompileError(f"cast: {len(x)} bits for a {source.name} source")
    kind = cast_kind(source, target)
    if kind == "to_bool":
        return (truth(b, x),)
    if kind == "reinterpret":
        return tuple(x)
    if kind == "truncate":
        return truncate(x, target.width)
    if kind == "sign_extend":
        return sign_extend(x, target.width)
    return zero_extend(b, x, target.width)


__all__ = [
    "CAST_KINDS",
    "cast",
    "cast_kind",
    "sign_extend",
    "truncate",
    "truth",
    "zero_extend",
]
