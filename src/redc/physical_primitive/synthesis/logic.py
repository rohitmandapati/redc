"""Shared Boolean structure: bitwise maps, reductions and the 2:1 mux.

Every helper emits only basis gates through a
:class:`~redc.physical_primitive.synthesis.builder.PrimitiveBuilder`; "mux" and
"reduce" are synthesis-time *structures*, never primitives.

Conventions used throughout synthesis:

* bit vectors are LSB first (``bits[0]`` is the least significant bit);
* reductions are BALANCED trees built level by level over adjacent pairs
  (``[b0, b1, b2, b3, b4] -> [b0.b1, b2.b3, b4] -> [b0.b1.b2.b3, b4] -> ...``),
  so logic depth is ``ceil(log2(n))`` and the structure is deterministic;
* a 2:1 mux is ``OR(AND(sel, yes), AND(NOT(sel), no))`` with ``NOT(sel)``
  computed once and shared by every bit of a vector mux.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ...parser import CompileError
from ..netlist import Bit, BitVector
from .builder import PrimitiveBuilder

GateFn = Callable[..., Bit]


def bitwise(
    b: PrimitiveBuilder, gate: GateFn, a: BitVector, c: BitVector, *, role: str
) -> BitVector:
    """``out[i] = gate(a[i], c[i])`` -- one gate per bit."""
    if len(a) != len(c):
        raise CompileError(f"bitwise {role}: width mismatch {len(a)} vs {len(c)}")
    return tuple(gate(x, y, role=role, bit=i) for i, (x, y) in enumerate(zip(a, c)))


def invert(b: PrimitiveBuilder, a: BitVector, *, role: str = "invert") -> BitVector:
    """``out[i] = NOT(a[i])``."""
    return tuple(b.not_(x, role=role, bit=i) for i, x in enumerate(a))


def _reduce(gate: GateFn, bits: Sequence[Bit], role: str) -> Bit:
    if not bits:
        raise CompileError(f"{role}: cannot reduce an empty bit vector")
    level = list(bits)
    depth = 0
    while len(level) > 1:
        nxt = [gate(level[k], level[k + 1], role=role, bit=depth) for k in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
        depth += 1
    return level[0]


def and_reduce(b: PrimitiveBuilder, bits: Sequence[Bit], *, role: str = "and_reduce") -> Bit:
    """1 iff every bit is 1 (balanced AND tree; one bit needs no gate)."""
    return _reduce(b.and_, bits, role)


def or_reduce(b: PrimitiveBuilder, bits: Sequence[Bit], *, role: str = "or_reduce") -> Bit:
    """1 iff any bit is 1 (balanced OR tree; one bit needs no gate)."""
    return _reduce(b.or_, bits, role)


def xor_reduce(b: PrimitiveBuilder, bits: Sequence[Bit], *, role: str = "xor_reduce") -> Bit:
    """Parity of the bits (balanced XOR tree)."""
    return _reduce(b.xor, bits, role)


def mux_bit(
    b: PrimitiveBuilder,
    sel: Bit,
    yes: Bit,
    no: Bit,
    *,
    nsel: Bit | None = None,
    bit: int | None = None,
    role: str = "mux",
) -> Bit:
    """``sel ? yes : no`` as ``OR(AND(sel, yes), AND(NOT(sel), no))``.

    Pass a precomputed ``nsel`` (``NOT(sel)``) to share it across bits."""
    if nsel is None:
        nsel = b.not_(sel, role=f"{role}_nsel")
    taken = b.and_(sel, yes, role=f"{role}_yes", bit=bit)
    other = b.and_(nsel, no, role=f"{role}_no", bit=bit)
    return b.or_(taken, other, role=f"{role}_out", bit=bit)


def mux_vector(
    b: PrimitiveBuilder,
    sel: Bit,
    yes: BitVector,
    no: BitVector,
    *,
    role: str = "mux",
    nsel: Bit | None = None,
) -> BitVector:
    """Bitwise ``sel ? yes : no``; ``NOT(sel)`` is computed once for all bits."""
    if len(yes) != len(no):
        raise CompileError(f"{role}: width mismatch {len(yes)} vs {len(no)}")
    if nsel is None:
        nsel = b.not_(sel, role=f"{role}_nsel")
    return tuple(mux_bit(b, sel, y, n, nsel=nsel, bit=i, role=role) for i, (y, n) in enumerate(zip(yes, no)))


def constant_vector(b: PrimitiveBuilder, value: int, width: int) -> BitVector:
    """``width`` constant bits of ``value`` (LSB first), shared per IR node."""
    return tuple(b.const(bool((value >> i) & 1)) for i in range(width))


__all__ = [
    "and_reduce",
    "bitwise",
    "constant_vector",
    "invert",
    "mux_bit",
    "mux_vector",
    "or_reduce",
    "xor_reduce",
]
