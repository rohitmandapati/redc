"""Shift recipes: a staged barrel shifter plus an exact amount range check.

The amount operand of ``shl`` / ``shr`` is ALWAYS ``uint64``
(:data:`redc.ir.SHIFT_AMOUNT`), whatever the value width ``W``.  The IR
semantics (:func:`redc.ir._calculate`) are

* ``shl``: ``0`` if ``amount >= W`` else ``(x << amount) mod 2**W``;
* ``shr``: ``number(x) >> min(amount, W)`` -- logical for unsigned ``x``,
  arithmetic for signed ``x`` (an over-long shift leaves only sign copies).

Structure (only basis gates; Python decides nothing but the stage layout):

* **Barrel.**  Stage ``k`` (group ``stage{k}``, ``distance = 2**k``, built only
  for ``2**k < W``) is a :func:`~.logic.mux_vector` on ``amount[k]`` between
  the current vector and the vector moved by ``2**k``, with ONE shared
  ``NOT(amount[k])`` per stage.  The ``ceil(log2 W)`` stages reach every
  distance ``0 .. W-1``; vacated positions take the fill: constant 0 for
  ``shl`` and unsigned ``shr``, the ORIGINAL operand's sign bit ``x[W-1]`` for
  signed ``shr`` (never a shifted intermediate).
* **Range.**  :func:`amount_in_range` is ``amount < W`` as a full 64-bit
  unsigned comparison (:func:`~.compare.unsigned_less`) against the
  bit-blasted constant ``W``, and ``too_large = NOT(amount_in_range)``.  It is
  exact for every width; looking only at the amount bits above the barrel's
  stages would be WRONG for a non-power-of-two width (``W = 13``: amounts
  ``13 .. 15`` set no bit above bit 3 yet are too large).
* **Saturation.**  ``too_large ? fill : barrel``.  For a constant-zero fill this
  is ``AND(in_range, barrel[i])`` per bit (``too_large ? 0 : v == AND(NOT
  too_large, v)``, and ``NOT too_large`` is the comparator output itself).
  The signed fill is a runtime bit, so it is a real 2:1 mux selected by an
  explicit ``too_large`` gate whose shared complement is ``in_range``.
"""

from __future__ import annotations

from ...parser import CompileError
from ..netlist import Bit, BitVector
from .builder import PrimitiveBuilder
from .compare import unsigned_less
from .logic import constant_vector, mux_vector


def _role(prefix: str, base: str) -> str:
    return f"{prefix}_{base}" if prefix else base


def stage_count(width: int) -> int:
    """Barrel stages a W-bit value needs: the number of ``k`` with ``2**k < W``."""
    return (width - 1).bit_length()


def barrel_shift(
    b: PrimitiveBuilder,
    x: BitVector,
    amount: BitVector,
    *,
    left: bool,
    fill: Bit,
    role: str = "",
) -> BitVector:
    """``x`` shifted by ``amount mod 2**stage_count(W)`` with ``fill`` moving in.

    Correct for every ``amount < W`` (higher amounts are the caller's
    saturation).  A one-bit value has no stage at all."""
    width = len(x)
    stages = stage_count(width)
    if len(amount) < stages:
        raise CompileError(f"barrel shift: a {len(amount)}-bit amount cannot drive {stages} stages")
    current = tuple(x)
    for k in range(stages):
        distance = 1 << k
        with b.scope(f"stage{k}", kind="stage", stage=k, distance=distance):
            if left:
                moved = (fill,) * distance + current[: width - distance]
            else:
                moved = current[distance:] + (fill,) * distance
            # One NOT(amount[k]) shared by the whole stage (provenance bit = k).
            keep = b.not_(amount[k], role=_role(role, "shift_stage_nsel"), bit=k)
            current = mux_vector(b, amount[k], moved, current, nsel=keep, role=_role(role, "shift_stage"))
    return current


def amount_in_range(b: PrimitiveBuilder, amount: BitVector, width: int, *, role: str = "shift_range") -> Bit:
    """``amount < width`` -- a ``len(amount)``-bit unsigned comparison against
    the constant ``width`` (``too_large`` is its complement)."""
    with b.scope("range_check", kind="helper", width=len(amount), limit=width):
        limit = constant_vector(b, width, len(amount))
        return unsigned_less(b, amount, limit, role=role)


def _zero_unless_in_range(b: PrimitiveBuilder, in_range: Bit, barrel: BitVector) -> BitVector:
    """``too_large ? 0 : barrel`` as ``AND(in_range, barrel[i])``."""
    with b.scope("saturate", kind="helper"):
        return tuple(b.and_(in_range, bit, role="shift_in_range_and", bit=i) for i, bit in enumerate(barrel))


def shift_left(b: PrimitiveBuilder, x: BitVector, amount: BitVector) -> BitVector:
    """IR ``shl``: zero fill; any ``amount >= W`` gives 0."""
    width = len(x)
    with b.scope("left_shifter", kind="helper", width=width):
        barrel = barrel_shift(b, x, amount, left=True, fill=b.const(False))
        return _zero_unless_in_range(b, amount_in_range(b, amount, width), barrel)


def shift_right(b: PrimitiveBuilder, x: BitVector, amount: BitVector, *, signed: bool) -> BitVector:
    """IR ``shr``: logical (zero fill, ``amount >= W`` gives 0) for unsigned
    ``x``; arithmetic (sign fill, ``amount >= W`` gives W copies of the sign)
    for signed ``x``."""
    width = len(x)
    name = "arithmetic_right_shifter" if signed else "logical_right_shifter"
    with b.scope(name, kind="helper", width=width):
        fill = x[width - 1] if signed else b.const(False)
        barrel = barrel_shift(b, x, amount, left=False, fill=fill)
        in_range = amount_in_range(b, amount, width)
        if not signed:
            return _zero_unless_in_range(b, in_range, barrel)
        with b.scope("saturate", kind="helper"):
            too_large = b.not_(in_range, role="shift_too_large", bit=0)
            return mux_vector(b, too_large, (fill,) * width, barrel, nsel=in_range, role="shift_sign_fill")


__all__ = [
    "amount_in_range",
    "barrel_shift",
    "shift_left",
    "shift_right",
    "stage_count",
]
