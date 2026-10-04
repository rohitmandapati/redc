"""Arithmetic recipes: ONE ripple-carry adder and everything built on it.

There is no adder, multiplier or divider primitive: every helper here emits
only basis gates (AND / OR / XOR / NOT) through the
:class:`~redc.physical_primitive.synthesis.builder.PrimitiveBuilder`, and the
"adder", "multiplier", ... exist only as provenance groups.  Bit vectors are
LSB first; ``W`` is the operand width and every result wraps modulo ``2**W``
exactly like :func:`redc.ir._calculate` (which masks every answer to the
result width).  Nothing is ever evaluated on the host: Python only decides the
STRUCTURE (widths, signedness), never a runtime value.

* :func:`ripple_add` -- the single adder architecture (no carry-lookahead).
  Bit 0 without a carry-in is a half adder (``sum = XOR(a, b)``, ``carry =
  AND(a, b)``); every other bit is a full adder::

      p = XOR(a, b);  sum = XOR(p, c);  g = AND(a, b);  pc = AND(p, c);  cout = OR(g, pc)

  The carry out of the top bit is never built (the sum wraps to W bits).  Gate
  roles are ``propagate_xor``, ``sum_xor``, ``carry_generate``,
  ``carry_propagate_and`` and ``carry_or``, one ``bit{i}`` slice group per bit.
  ``add`` is the same network for signed and unsigned operands.
* :func:`increment` -- the ripple adder specialised to an all-zero second
  operand: a half-adder chain ``sum[i] = XOR(x[i], c[i])``, ``c[i+1] =
  AND(x[i], c[i])`` with ``c[0] = carry_in`` (constant-zero operand bits are
  never fed into full adders).
* :func:`subtract` -- ``x - y = x + NOT(y) + 1``: the ripple adder with every
  subtrahend bit inverted and carry-in = the node's constant 1 (bit 0 is then a
  full adder).  No separate subtractor architecture.
* :func:`negate` -- ``-x = NOT(x) + 1`` via :func:`increment` (carry-in 1).
* :func:`conditional_twos_complement` -- ``s ? -x : x`` as ``(x XOR s) + s``
  (:func:`increment` with carry-in ``s``).  Exact for every pattern, including
  ``INT_MIN``, whose magnitude ``2**(W-1)`` is representable as an UNSIGNED
  W-bit intermediate.
* :func:`multiply` -- an array multiplier (no Booth, no Wallace tree); see its
  docstring for why one network serves signed and unsigned operands.
* :func:`unsigned_divmod` -- unrolled restoring division, ONE helper shared by
  ``div`` and ``mod``; :func:`divide` / :func:`modulo` add RedC's signed
  handling (truncation toward zero, remainder follows the dividend) and the
  divide-by-zero rule (result 0).

Every helper takes a ``role`` prefix (gate roles become ``f"{role}_{base}"``)
and a ``name`` for its hierarchy group, so nested use stays legible: the
divider's subtractor emits ``restoring_subtract_propagate_xor`` gates inside a
``restoring_subtract`` group inside an ``iter{k}`` group.
"""

from __future__ import annotations

from ...parser import CompileError
from ..netlist import Bit, BitVector
from .builder import PrimitiveBuilder
from .compare import unsigned_less
from .logic import mux_vector, or_reduce


def _role(prefix: str, base: str) -> str:
    """Gate role ``base`` labelled for nested use (``restoring_subtract_sum_xor``)."""
    return f"{prefix}_{base}" if prefix else base


def _width(name: str, *vectors: BitVector) -> int:
    widths = {len(vector) for vector in vectors}
    if len(widths) != 1 or 0 in widths:
        raise CompileError(f"{name}: operand widths {sorted(widths)} must be equal and non-zero")
    return widths.pop()


# -- addition ---------------------------------------------------------------


def ripple_add(
    b: PrimitiveBuilder,
    x: BitVector,
    y: BitVector,
    carry_in: Bit | None = None,
    *,
    role: str = "",
    name: str = "ripple_adder",
    bit_offset: int = 0,
) -> BitVector:
    """``(x + y + carry_in) mod 2**W`` as a ripple-carry adder.

    Without ``carry_in`` bit 0 is a half adder; with it (and at every higher
    bit) a full adder.  The top bit's carry-out is not built.  ``bit_offset``
    is the absolute bit position of ``x[0]`` -- it names the slices and the
    provenance bits of an adder over a sub-range (a multiplier row)."""
    width = _width(name, x, y)
    out: list[Bit] = []
    carry = carry_in
    with b.scope(name, kind="helper", width=width, lsb=bit_offset):
        for i in range(width):
            pos = bit_offset + i
            top = i == width - 1  # its carry-out would wrap away: never built
            if carry is None:
                with b.scope(f"bit{pos}", kind="slice", bit=pos, adder="half"):
                    out.append(b.xor(x[i], y[i], role=_role(role, "sum_xor"), bit=pos))
                    if not top:
                        carry = b.and_(x[i], y[i], role=_role(role, "carry_generate"), bit=pos)
                continue
            with b.scope(f"bit{pos}", kind="slice", bit=pos, adder="full"):
                propagate = b.xor(x[i], y[i], role=_role(role, "propagate_xor"), bit=pos)
                out.append(b.xor(propagate, carry, role=_role(role, "sum_xor"), bit=pos))
                if not top:
                    generate = b.and_(x[i], y[i], role=_role(role, "carry_generate"), bit=pos)
                    passed = b.and_(propagate, carry, role=_role(role, "carry_propagate_and"), bit=pos)
                    carry = b.or_(generate, passed, role=_role(role, "carry_or"), bit=pos)
    return tuple(out)


def add(b: PrimitiveBuilder, x: BitVector, y: BitVector, *, role: str = "") -> BitVector:
    """``(x + y) mod 2**W`` -- identical for signed and unsigned operands."""
    return ripple_add(b, x, y, role=role)


def increment(
    b: PrimitiveBuilder, x: BitVector, carry_in: Bit, *, role: str = "", name: str = "incrementer"
) -> BitVector:
    """``(x + carry_in) mod 2**W``: the ripple adder with an all-zero second
    operand, i.e. the half-adder chain ``sum[i] = XOR(x[i], c[i])``,
    ``c[i+1] = AND(x[i], c[i])``, ``c[0] = carry_in`` (top carry not built)."""
    width = _width(name, x)
    out: list[Bit] = []
    carry = carry_in
    with b.scope(name, kind="helper", width=width):
        for i in range(width):
            with b.scope(f"bit{i}", kind="slice", bit=i, adder="half"):
                out.append(b.xor(x[i], carry, role=_role(role, "increment_sum_xor"), bit=i))
                if i < width - 1:
                    carry = b.and_(x[i], carry, role=_role(role, "increment_carry_and"), bit=i)
    return tuple(out)


def subtract(
    b: PrimitiveBuilder,
    x: BitVector,
    y: BitVector,
    *,
    role: str = "",
    name: str = "subtractor",
) -> BitVector:
    """``(x - y) mod 2**W = x + NOT(y) + 1``: :func:`ripple_add` over the
    inverted subtrahend with carry-in = constant 1."""
    width = _width(name, x, y)
    with b.scope(name, kind="helper", width=width):
        inverted = tuple(b.not_(bit, role=_role(role, "subtrahend_not"), bit=i) for i, bit in enumerate(y))
        return ripple_add(b, x, inverted, b.const(True), role=role)


def negate(b: PrimitiveBuilder, x: BitVector, *, role: str = "", name: str = "negator") -> BitVector:
    """``(-x) mod 2**W = NOT(x) + 1`` (``-INT_MIN`` wraps to ``INT_MIN``)."""
    width = _width(name, x)
    with b.scope(name, kind="helper", width=width):
        inverted = tuple(b.not_(bit, role=_role(role, "negate_not"), bit=i) for i, bit in enumerate(x))
        return increment(b, inverted, b.const(True), role=role)


def conditional_twos_complement(
    b: PrimitiveBuilder, x: BitVector, s: Bit, *, role: str = "", name: str = "conditional_negate"
) -> BitVector:
    """``s ? -x : x`` (mod ``2**W``) as ``(x XOR s) + s``.

    Turns a two's-complement value into its magnitude when ``s`` is its sign
    bit -- exact for ``INT_MIN`` too, whose magnitude ``2**(W-1)`` is a valid
    UNSIGNED W-bit pattern -- and applies a result sign afterwards."""
    width = _width(name, x)
    with b.scope(name, kind="helper", width=width):
        flipped = tuple(
            b.xor(bit, s, role=_role(role, "conditional_invert_xor"), bit=i) for i, bit in enumerate(x)
        )
        return increment(b, flipped, s, role=role)


# -- multiplication ---------------------------------------------------------


def multiply(
    b: PrimitiveBuilder, x: BitVector, y: BitVector, *, role: str = "", name: str = "array_multiplier"
) -> BitVector:
    """``(x * y) mod 2**W`` as an array multiplier.

    Partial products ``partial[j][i + j] = AND(x[i], y[j])`` are built only for
    ``i + j < W`` (higher columns would be truncated away).  Row 0
    (``AND(x[i], y[0])``) IS the initial accumulator.  Each later row ``j`` --
    ``x`` shifted left by ``j`` and gated by ``y[j]`` -- is zero below column
    ``j``, so adding it cannot change accumulator bits ``0 .. j-1`` (zero
    addend, zero carry): the shared :func:`ripple_add` runs only over the
    occupied columns ``j .. W-1``, never on explicit constant-zero bits, and its
    top carry is dropped (low ``W`` bits only).  Row ``j``'s gates live in group
    ``row{j}`` (``kind="iteration"``, ``iteration=j``).

    Signed and unsigned operands share this exact raw-bit network: a W-bit
    two's-complement value ``v`` and its unsigned pattern ``u`` satisfy
    ``v == u (mod 2**W)``, so ``v1 * v2 == u1 * u2 (mod 2**W)`` -- the low W
    product bits do not depend on the signedness (only the discarded high
    half would)."""
    width = _width(name, x, y)
    with b.scope(name, kind="helper", width=width):
        with b.scope("row0", kind="iteration", iteration=0, row=0):
            accumulator = [b.and_(x[i], y[0], role=_role(role, "partial_product"), bit=i) for i in range(width)]
        for j in range(1, width):
            with b.scope(f"row{j}", kind="iteration", iteration=j, row=j):
                row = tuple(
                    b.and_(x[col - j], y[j], role=_role(role, "partial_product"), bit=col)
                    for col in range(j, width)
                )
                accumulator[j:] = ripple_add(b, tuple(accumulator[j:]), row, role=role, bit_offset=j)
        return tuple(accumulator)


# -- division -----------------------------------------------------------------


def unsigned_divmod(
    b: PrimitiveBuilder,
    n: BitVector,
    d: BitVector,
    *,
    need_remainder: bool = True,
    name: str = "restoring_divider",
) -> tuple[BitVector, BitVector | None]:
    """``(n // d, n % d)`` for unsigned W-bit patterns: unrolled RESTORING division.

    ::

        remainder = W constant-zero bits
        for i = W-1 down to 0:                       # group iter{k}, k = W-1-i
            shifted   = (remainder << 1) | n[i]      # W+1 bits, shifted[0] = n[i]
            ge        = NOT(unsigned_less(shifted, zero_extend(d)))   # restoring_compare
            candidate = shifted - d                  # restoring_subtract
            remainder = MUX(ge, candidate, shifted)  # restoring_select
            quotient[i] = ge

    The compare needs all ``W + 1`` bits of ``shifted`` (it can reach
    ``2d - 1``), but the remainder never does: it is always ``< d < 2**W``, so
    ``shifted - d`` (taken only when ``shifted >= d``) and ``shifted`` (kept
    only when ``shifted < d``) both fit in W bits.  The subtractor and the
    select are therefore W bits wide; the low W bits of a difference depend
    only on the low W bits of its operands, so ``candidate`` is exactly
    ``shifted - d``.  Building the always-zero top remainder bit would only
    leave dead gates (the next shift drops it).  The select reuses the
    comparator's ``unsigned_less`` output as the mux's ``NOT(ge)``.

    ``need_remainder=False`` (``div``) skips the last row's subtract and select,
    which feed only the final remainder; the second element is then ``None``.
    ``d == 0`` yields an all-ones quotient and remainder ``n``; callers apply
    the IR's divide-by-zero rule (:func:`zero_if_divisor_zero`)."""
    width = _width(name, n, d)
    decided: list[Bit] = []  # quotient bits, MSB first
    with b.scope(name, kind="helper", width=width):
        zero = b.const(False)
        remainder: BitVector = (zero,) * width
        divisor = (*d, zero)  # d zero-extended to W + 1 bits
        for k in range(width):
            i = width - 1 - k  # the quotient bit this row decides
            with b.scope(f"iter{k}", kind="iteration", iteration=k, quotient_bit=i):
                shifted = (n[i], *remainder)  # (remainder << 1) | n[i], W + 1 bits
                with b.scope("restoring_compare", kind="helper"):
                    below = unsigned_less(b, shifted, divisor, role="restoring_compare")
                    ge = b.not_(below, role="restoring_compare_ge", bit=i)
                decided.append(ge)
                if k == width - 1 and not need_remainder:
                    break  # the final remainder is not wanted: build nothing for it
                candidate = subtract(b, shifted[:width], d, role="restoring_subtract", name="restoring_subtract")
                with b.scope("restoring_select", kind="helper"):
                    remainder = mux_vector(b, ge, candidate, shifted[:width], nsel=below, role="restoring_select")
    return tuple(reversed(decided)), (remainder if need_remainder else None)


def zero_if_divisor_zero(b: PrimitiveBuilder, value: BitVector, divisor: BitVector) -> BitVector:
    """RedC's divide-by-zero rule: ``divisor == 0 ? 0 : value``, as
    ``AND(value[i], OR_REDUCE(divisor))``."""
    with b.scope("divide_by_zero_guard", kind="helper"):
        nonzero = or_reduce(b, divisor, role="divisor_nonzero_or")
        return tuple(b.and_(bit, nonzero, role="divide_by_zero_mask", bit=i) for i, bit in enumerate(value))


def _magnitudes(b: PrimitiveBuilder, x: BitVector, y: BitVector) -> tuple[BitVector, BitVector]:
    """``(|x|, |y|)`` as unsigned W-bit patterns (``|INT_MIN| = 2**(W-1)``)."""
    msb = len(x) - 1
    abs_x = conditional_twos_complement(b, x, x[msb], role="dividend_abs", name="dividend_abs")
    abs_y = conditional_twos_complement(b, y, y[msb], role="divisor_abs", name="divisor_abs")
    return abs_x, abs_y


def divide(b: PrimitiveBuilder, x: BitVector, y: BitVector, *, signed: bool) -> BitVector:
    """RedC ``div``: 0 for a zero divisor; signed division truncates toward
    zero (``q = sign(x) XOR sign(y) ? -(|x| // |y|) : |x| // |y|``, so
    ``INT_MIN / -1`` wraps to ``INT_MIN``); the result is masked to W bits."""
    width = _width("divide", x, y)
    with b.scope("signed_division" if signed else "unsigned_division", kind="helper", width=width):
        if signed:
            msb = width - 1
            quotient_sign = b.xor(x[msb], y[msb], role="quotient_sign_xor", bit=msb)
            abs_x, abs_y = _magnitudes(b, x, y)
            magnitude, _ = unsigned_divmod(b, abs_x, abs_y, need_remainder=False)
            quotient = conditional_twos_complement(
                b, magnitude, quotient_sign, role="quotient_sign", name="quotient_sign"
            )
        else:
            quotient, _ = unsigned_divmod(b, x, y, need_remainder=False)
        return zero_if_divisor_zero(b, quotient, y)


def modulo(b: PrimitiveBuilder, x: BitVector, y: BitVector, *, signed: bool) -> BitVector:
    """RedC ``mod``: ``x - trunc(x / y) * y``, 0 for a zero divisor.  With
    truncating division the remainder takes the DIVIDEND's sign:
    ``r = sign(x) ? -(|x| % |y|) : |x| % |y|`` (so ``INT_MIN % -1 == 0``)."""
    width = _width("modulo", x, y)
    with b.scope("signed_modulo" if signed else "unsigned_modulo", kind="helper", width=width):
        if signed:
            abs_x, abs_y = _magnitudes(b, x, y)
            _, magnitude = unsigned_divmod(b, abs_x, abs_y)
            assert magnitude is not None
            remainder = conditional_twos_complement(
                b, magnitude, x[width - 1], role="remainder_sign", name="remainder_sign"
            )
        else:
            _, unsigned_remainder = unsigned_divmod(b, x, y)
            assert unsigned_remainder is not None
            remainder = unsigned_remainder
        return zero_if_divisor_zero(b, remainder, y)


__all__ = [
    "add",
    "conditional_twos_complement",
    "divide",
    "increment",
    "modulo",
    "multiply",
    "negate",
    "ripple_add",
    "subtract",
    "unsigned_divmod",
    "zero_if_divisor_zero",
]
