"""Comparison recipes: equality and the MSB-first less-than chain.

There is no comparator primitive: every comparison is a network of basis gates
(AND / OR / XOR / NOT) emitted through the
:class:`~redc.physical_primitive.synthesis.builder.PrimitiveBuilder`.  One
convention is used everywhere (bit vectors are LSB first):

* ``different[i] = XOR(x[i], y[i])``; ``x != y`` is ``OR_REDUCE(different)`` (a
  balanced tree) and ``x == y`` is ``NOT(OR_REDUCE(different))``.
* :func:`unsigned_less` walks the bits from the MSB down, carrying ``less``
  (already decided: ``x < y``) and ``equal_prefix`` (every higher bit equal)::

      bit_equal    = NOT(XOR(x[i], y[i]))
      x_less_y     = AND(NOT(x[i]), y[i])
      decisive     = AND(equal_prefix, x_less_y)
      less         = OR(less, decisive)
      equal_prefix = AND(equal_prefix, bit_equal)

  Above the MSB ``equal_prefix`` is "true" and ``less`` is "false", so the MSB
  starts the chain WITHOUT constants (``less = x_less_y``, ``equal_prefix =
  bit_equal``).  The LSB's ``equal_prefix`` update would be read by no later
  bit, so neither it nor the LSB's ``bit_equal`` is built.
* :func:`signed_less` (two's complement): when the sign bits differ, ``x < y``
  iff ``x`` is negative; otherwise the raw unsigned comparison of the bit
  patterns decides -- ``MUX(XOR(sx, sy), sx, unsigned_less(x, y))``, the mux
  expanded by :func:`~.logic.mux_bit`.
* :func:`comparison` maps the six IR comparisons onto those helpers:
  ``gt(x, y) = lt(y, x)``, ``le = OR(lt, eq)``, ``ge = NOT(lt)``.  Signedness
  follows the OPERAND type (``bool`` is unsigned); the result is one ``bool``.

Gate counts for operand width ``W >= 2``: ``ne`` ``2W - 1``, ``eq`` ``2W``,
unsigned ``lt`` ``7W - 6``, signed ``lt`` ``7W - 1``.

Every helper takes a ``role`` prefix (gate roles become ``f"{role}_{base}"``)
and a ``name`` for its hierarchy group, so a caller can label nested use: the
restoring divider's comparator emits ``restoring_compare_lt_*`` gates.
"""

from __future__ import annotations

from ...parser import CompileError
from ...signature import OperationSignature
from ..netlist import Bit, BitVector
from .builder import PrimitiveBuilder
from .logic import mux_bit, or_reduce


def _role(prefix: str, base: str) -> str:
    """Gate role ``base`` labelled for nested use (``restoring_compare_lt_x_not``)."""
    return f"{prefix}_{base}" if prefix else base


def _pair_width(name: str, x: BitVector, y: BitVector) -> int:
    if len(x) != len(y) or not x:
        raise CompileError(f"{name}: operand widths {len(x)} and {len(y)} must be equal and non-zero")
    return len(x)


def different_bits(b: PrimitiveBuilder, x: BitVector, y: BitVector, *, role: str = "") -> BitVector:
    """``different[i] = XOR(x[i], y[i])`` -- one XOR per bit."""
    _pair_width("different_bits", x, y)
    return tuple(b.xor(p, q, role=_role(role, "diff_xor"), bit=i) for i, (p, q) in enumerate(zip(x, y)))


def not_equal(
    b: PrimitiveBuilder, x: BitVector, y: BitVector, *, role: str = "", name: str = "not_equal"
) -> Bit:
    """``x != y`` as ``OR_REDUCE(different)`` (a one-bit operand needs only the XOR)."""
    width = _pair_width(name, x, y)
    with b.scope(name, kind="helper", width=width):
        return or_reduce(b, different_bits(b, x, y, role=role), role=_role(role, "diff_or"))


def equal(b: PrimitiveBuilder, x: BitVector, y: BitVector, *, role: str = "", name: str = "equal") -> Bit:
    """``x == y`` as ``NOT(OR_REDUCE(different))``."""
    width = _pair_width(name, x, y)
    with b.scope(name, kind="helper", width=width):
        any_different = or_reduce(b, different_bits(b, x, y, role=role), role=_role(role, "diff_or"))
        return b.not_(any_different, role=_role(role, "eq_not"), bit=0)


def _bit_equal(b: PrimitiveBuilder, x: Bit, y: Bit, role: str, bit: int) -> Bit:
    diff = b.xor(x, y, role=_role(role, "lt_diff_xor"), bit=bit)
    return b.not_(diff, role=_role(role, "lt_bit_equal"), bit=bit)


def _bit_less(b: PrimitiveBuilder, x: Bit, y: Bit, role: str, bit: int) -> Bit:
    x_not = b.not_(x, role=_role(role, "lt_x_not"), bit=bit)
    return b.and_(x_not, y, role=_role(role, "lt_bit_less"), bit=bit)


def unsigned_less(
    b: PrimitiveBuilder, x: BitVector, y: BitVector, *, role: str = "", name: str = "unsigned_less"
) -> Bit:
    """``x < y`` on raw (unsigned) bit patterns: the MSB-first chain above.

    Each bit position is one ``bit{i}`` slice group.  ``W = 1`` is just
    ``AND(NOT(x[0]), y[0])``."""
    width = _pair_width(name, x, y)
    msb = width - 1
    with b.scope(name, kind="helper", width=width):
        with b.scope(f"bit{msb}", kind="slice", bit=msb):
            # Above the MSB everything is equal and nothing is decided yet.
            equal_prefix = _bit_equal(b, x[msb], y[msb], role, msb) if msb > 0 else None
            less = _bit_less(b, x[msb], y[msb], role, msb)
        for i in range(msb - 1, -1, -1):
            assert equal_prefix is not None  # every bit above the LSB extends the prefix
            with b.scope(f"bit{i}", kind="slice", bit=i):
                # The LSB's prefix update would have no reader: never build it.
                bit_equal = _bit_equal(b, x[i], y[i], role, i) if i > 0 else None
                x_less_y = _bit_less(b, x[i], y[i], role, i)
                decisive = b.and_(equal_prefix, x_less_y, role=_role(role, "lt_decisive"), bit=i)
                less = b.or_(less, decisive, role=_role(role, "lt_accumulate"), bit=i)
                if bit_equal is not None:
                    equal_prefix = b.and_(equal_prefix, bit_equal, role=_role(role, "lt_equal_prefix"), bit=i)
        return less


def signed_less(
    b: PrimitiveBuilder, x: BitVector, y: BitVector, *, role: str = "", name: str = "signed_less"
) -> Bit:
    """Two's-complement ``x < y``: ``MUX(XOR(sx, sy), sx, unsigned_less(x, y))``.

    Same signs: two's complement preserves order within one sign, so the raw
    pattern comparison is exact.  Different signs: the negative one is smaller."""
    width = _pair_width(name, x, y)
    msb = width - 1
    with b.scope(name, kind="helper", width=width):
        sx, sy = x[msb], y[msb]
        signs_differ = b.xor(sx, sy, role=_role(role, "lt_sign_differ_xor"), bit=msb)
        same_sign_less = unsigned_less(b, x, y, role=role)
        select = _role(role, "lt_sign_select")
        signs_agree = b.not_(signs_differ, role=f"{select}_nsel", bit=msb)
        return mux_bit(b, signs_differ, sx, same_sign_less, nsel=signs_agree, bit=msb, role=select)


def less(b: PrimitiveBuilder, x: BitVector, y: BitVector, *, signed: bool, role: str = "") -> Bit:
    """``x < y`` with the signedness of the operand type."""
    if signed:
        return signed_less(b, x, y, role=role)
    return unsigned_less(b, x, y, role=role)


def operands_signed(signature: OperationSignature) -> bool:
    """Whether a comparison orders its operands as signed numbers (``bool`` and
    every ``uint`` compare unsigned)."""
    return signature.operand_types[0].signed


def comparison(b: PrimitiveBuilder, signature: OperationSignature, x: BitVector, y: BitVector) -> Bit:
    """The ``bool`` result of IR comparison ``signature.op`` (eq ne lt le gt ge)."""
    op = signature.op
    signed = operands_signed(signature)
    if op == "eq":
        return equal(b, x, y)
    if op == "ne":
        return not_equal(b, x, y)
    if op == "lt":
        return less(b, x, y, signed=signed)
    if op == "gt":
        return less(b, y, x, signed=signed)  # x > y  ==  y < x
    if op == "le":
        lt = less(b, x, y, signed=signed)
        eq = equal(b, x, y)
        return b.or_(lt, eq, role="le_or", bit=0)
    if op == "ge":
        return b.not_(less(b, x, y, signed=signed), role="ge_not", bit=0)
    raise CompileError(f"{signature} is not a comparison")


__all__ = [
    "comparison",
    "different_bits",
    "equal",
    "less",
    "not_equal",
    "operands_signed",
    "signed_less",
    "unsigned_less",
]
