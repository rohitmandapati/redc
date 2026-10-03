"""Minecraft physical type contract and signal-encoding metadata.

The RedC language and target-neutral IR accept any integer width from 1 to 64.
The Minecraft physical backend deliberately supports far fewer:

    bool, uint4/int4, uint8/int8, uint16/int16, uint32/int32, uint64/int64

:func:`is_supported_physical_type` / :func:`require_supported_physical_type` are
the single gate for that contract.  Every :class:`~redc.physical.components.Port`
is checked against it, so an arbitrary IR width (``uint3``, ``int24``, ...) can
never masquerade as a physical signal; the restriction never touches the
compiler or the SystemVerilog backend.

Logical values stay ordinary two's-complement bit vectors.  How a value is laid
out *spatially* is a separate, purely physical choice described by
:class:`PhysicalSignalLayout`:

* ``bool``     -> :attr:`SignalEncoding.BOOL`, one 1-bit lane.
* 4/8-bit      -> :attr:`SignalEncoding.BINARY`, one 1-bit lane per bit.
* 16/32/64-bit -> :attr:`SignalEncoding.HEX`, one 4-bit (nibble) lane per hex
  digit, carried as redstone signal strength 0-15.

Signedness never changes the layout.  A wide value remains ONE logical
:class:`~redc.physical.netlist.Net`; the layout only tells future placement,
routing and NBT realization how many physical lanes to reserve.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..ir import BOOL, IRType
from ..parser import CompileError

#: Integer widths the Minecraft backend implements (each signed and unsigned).
PHYSICAL_WIDTHS = (4, 8, 16, 32, 64)

#: Every supported physical datatype, in a fixed order: ``bool`` then each
#: width as ``uint`` followed by ``int``.
PHYSICAL_TYPES: tuple[IRType, ...] = (BOOL,) + tuple(
    IRType(width, signed=signed)
    for width in PHYSICAL_WIDTHS
    for signed in (False, True)
)

_SUPPORTED = frozenset(PHYSICAL_TYPES)

#: Widths at or above which values are routed as nibble (hex) lanes.
HEX_MIN_WIDTH = 16
NIBBLE = 4


def is_supported_physical_type(typ: IRType) -> bool:
    """Whether ``typ`` is a datatype the Minecraft backend can realize."""
    return typ in _SUPPORTED


def require_supported_physical_type(typ: IRType, what: str = "signal") -> IRType:
    """Return ``typ`` unchanged, or raise if Minecraft cannot realize it."""
    if not is_supported_physical_type(typ):
        supported = ", ".join(t.name for t in PHYSICAL_TYPES)
        raise CompileError(
            f"{what}: type {typ.name} is not supported by the Minecraft physical "
            f"backend (supported: {supported})"
        )
    return typ


class SignalEncoding(Enum):
    """How a logical value is represented in redstone."""

    BOOL = "bool"  # one on/off line
    BINARY = "binary"  # one on/off line per bit
    HEX = "hex"  # one signal-strength (0-15) line per nibble


@dataclass(frozen=True, slots=True)
class PhysicalSignalLayout:
    """Physical lane structure of one logical signal.

    ``lane_width * lane_count == logical_type.width`` always holds: a lane is one
    physical line carrying ``lane_width`` bits of the value.
    """

    logical_type: IRType
    encoding: SignalEncoding
    lane_width: int
    lane_count: int

    @property
    def bits(self) -> int:
        return self.lane_width * self.lane_count

    def to_dict(self) -> dict[str, str | int]:
        return {
            "encoding": self.encoding.value,
            "lane_width": self.lane_width,
            "lane_count": self.lane_count,
        }


def signal_layout(typ: IRType) -> PhysicalSignalLayout:
    """The physical layout for a supported datatype (see the module docstring)."""
    require_supported_physical_type(typ)
    if typ.boolean:
        return PhysicalSignalLayout(typ, SignalEncoding.BOOL, 1, 1)
    if typ.width >= HEX_MIN_WIDTH:
        return PhysicalSignalLayout(
            typ, SignalEncoding.HEX, NIBBLE, typ.width // NIBBLE
        )
    return PhysicalSignalLayout(typ, SignalEncoding.BINARY, 1, typ.width)
