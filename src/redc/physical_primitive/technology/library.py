"""The primitive Minecraft technology library: primitive kind -> candidate cells.

!!! EVERY CELL HERE IS A PLACEHOLDER !!!
No in-game structure has been built or verified for any of them
(``placeholder=True``, ``structure=None``).  Each one is a deterministic,
honest FOOTPRINT -- sparse occupied blocks, pin endpoints with facing and
drive strength, keep-out blocks -- sized like a plausible compact redstone
implementation so that placement, routing and electrical legalization can be
developed and tested at true block resolution.  The internal blocks are
abstract ``body`` volume; nothing claims they compute the right function.
Latencies are rough estimates in redstone ticks.

All cells use the same conventions: a ``base`` layer at local y=0 (so every
pin endpoint rests on the cell), signal level y=1, inputs on the local west
face, outputs on the local east face, adjacent pins at least two blocks apart
so different nets' endpoint dust can never touch.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

from ...parser import CompileError
from ..geometry import Direction
from ..netlist import PeripheralDirection, PrimitiveInstance, PrimitiveKind
from ..redstone import MAX_SIGNAL_STRENGTH
from .cells import PrimitiveCell, PrimitivePin, placeholder_cell

W, E = Direction.WEST, Direction.EAST

LIBRARY_NAME = "redc-primitive-placeholder-v1"


def _in(name: str, x: int, z: int) -> PrimitivePin:
    return PrimitivePin(name, "in", (x, 1, z), W, 1)


def _out(name: str, x: int, z: int, strength: int = MAX_SIGNAL_STRENGTH) -> PrimitivePin:
    return PrimitivePin(name, "out", (x, 1, z), E, strength)


def _box(x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> list[tuple[int, int, int]]:
    return [(x, y, z) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1) for z in range(z0, z1 + 1)]


def _two_input(kind: PrimitiveKind, latency: int, strength: int, what: str) -> PrimitiveCell:
    return placeholder_cell(
        f"{kind.value}_gate_2x3_placeholder",
        kind,
        body=_box(1, 2, 1, 1, 0, 2),
        pins=(_in("a", 0, 0), _in("b", 0, 2), _out("y", 3, 1, strength)),
        latency=latency,
        description=(
            f"PLACEHOLDER 2-input {what}: inputs a/b on the west face two blocks apart, "
            "output y on the east face; 2x1x3 abstract body on a 4x3 base."
        ),
    )


def placeholder_cells() -> tuple[PrimitiveCell, ...]:
    """Every placeholder cell, in a fixed order (the v1 default candidates)."""
    K = PrimitiveKind
    return (
        placeholder_cell(
            "not_gate_torch_placeholder",
            K.NOT,
            body=[(1, 1, 0), (2, 1, 0)],
            pins=(_in("a", 0, 0), _out("y", 3, 0)),
            latency=1,
            description=(
                "PLACEHOLDER inverter, laid out like a torch inverter (input dust -> block "
                "-> wall torch -> output dust) but not verified in-game."
            ),
        ),
        _two_input(K.AND, 2, MAX_SIGNAL_STRENGTH, "AND (torch-style)"),
        _two_input(K.OR, 1, 13, "OR (diode-merge style; drives strength 13)"),
        _two_input(K.XOR, 3, MAX_SIGNAL_STRENGTH, "XOR"),
        placeholder_cell(
            "register_bit_dff_placeholder",
            K.REGISTER_BIT,
            body=_box(1, 3, 1, 2, 0, 4),
            pins=(_in("d", 0, 0), _in("clk", 0, 2), _in("rst", 0, 4), _out("q", 4, 2)),
            latency=2,
            description=(
                "PLACEHOLDER one-bit edge-triggered register with asynchronous reset "
                "(d / clk / rst on the west face, q east); abstract 3x2x5 latch body."
            ),
        ),
        placeholder_cell(
            "const0_anchor_placeholder",
            K.CONST0,
            body=[(0, 1, 0)],
            pins=(_out("y", 1, 0, 0),),
            latency=0,
            body_role="anchor",
            description="PLACEHOLDER constant 0: an unpowered anchor block; its net is never powered.",
        ),
        placeholder_cell(
            "const1_source_placeholder",
            K.CONST1,
            body=[(0, 1, 0)],
            pins=(_out("y", 1, 0),),
            latency=0,
            body_role="power_source",
            description="PLACEHOLDER constant 1: a permanent power source (e.g. a block of redstone).",
        ),
        placeholder_cell(
            "input_bit_pad_placeholder",
            K.INPUT_BIT,
            body=[(0, 1, 0)],
            pins=(_out("y", 1, 0),),
            latency=0,
            body_role="input_pad",
            description="PLACEHOLDER one-bit input pad driven from outside the design.",
        ),
        placeholder_cell(
            "output_bit_pad_placeholder",
            K.OUTPUT_BIT,
            body=[(1, 1, 0)],
            pins=(_in("a", 0, 0),),
            latency=0,
            body_role="output_pad",
            description="PLACEHOLDER one-bit output pad (e.g. a lamp) observing the design.",
        ),
        placeholder_cell(
            "clock_source_placeholder",
            K.CLOCK_SOURCE,
            body=_box(0, 2, 1, 2, 0, 2),
            pins=(_out("clk", 3, 1),),
            latency=0,
            body_role="clock",
            description="PLACEHOLDER global clock generator (one per design).",
        ),
        placeholder_cell(
            "reset_source_placeholder",
            K.RESET_SOURCE,
            body=[(0, 1, 0), (0, 2, 0)],
            pins=(_out("rst", 1, 0),),
            latency=0,
            body_role="reset",
            description="PLACEHOLDER global active-high reset (e.g. a button), one per design.",
        ),
        placeholder_cell(
            "lever_1bit_placeholder",
            K.PERIPHERAL,
            body=_box(0, 1, 1, 2, 0, 1),
            pins=(_out("b0", 2, 0),),
            latency=None,
            body_role="lever",
            peripheral=("lever", PeripheralDirection.INPUT, 1),
            description="PLACEHOLDER lever the player flips: one bool into the circuit.",
        ),
        placeholder_cell(
            "two_digit_seven_segment_8bit_placeholder",
            K.PERIPHERAL,
            body=_box(1, 3, 1, 5, 0, 14),
            pins=tuple(_in(f"b{k}", 0, 2 * k) for k in range(8)),
            latency=None,
            body_role="display",
            peripheral=("2-dig-7-seg", PeripheralDirection.OUTPUT, 8),
            description=(
                "PLACEHOLDER two-digit seven-segment display consuming one uint8 as eight "
                "independent one-bit inputs b0..b7 (two blocks apart on its west face)."
            ),
        ),
    )


class PrimitiveTechnologyLibrary(Mapping[str, PrimitiveCell]):
    """Name lookup plus "which cells can realize this primitive?" lookup.

    Non-peripheral kinds are keyed by :class:`PrimitiveKind`; peripherals by
    ``(kind name, direction, width)``.  Candidates come back in registration
    order, so "pick the first" is a deterministic v1 policy."""

    def __init__(self, name: str, cells: Iterable[PrimitiveCell]) -> None:
        self.name = name
        self._by_name: dict[str, PrimitiveCell] = {}
        self._by_kind: dict[PrimitiveKind, list[PrimitiveCell]] = {}
        self._by_peripheral: dict[tuple[str, PeripheralDirection, int], list[PrimitiveCell]] = {}
        for cell in cells:
            if cell.name in self._by_name:
                raise CompileError(f"technology library {name!r}: duplicate cell {cell.name!r}")
            self._by_name[cell.name] = cell
            if cell.peripheral is not None:
                self._by_peripheral.setdefault(cell.peripheral, []).append(cell)
            else:
                self._by_kind.setdefault(cell.kind, []).append(cell)

    def __getitem__(self, name: str) -> PrimitiveCell:
        return self._by_name[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    def candidates(self, instance: PrimitiveInstance) -> tuple[PrimitiveCell, ...]:
        """Every cell able to realize ``instance`` (possibly empty)."""
        if instance.kind is PrimitiveKind.PERIPHERAL:
            spec = instance.peripheral
            assert spec is not None
            return tuple(self._by_peripheral.get((spec.kind, spec.direction, spec.width), ()))
        return tuple(self._by_kind.get(instance.kind, ()))

    def kinds(self) -> frozenset[PrimitiveKind]:
        return frozenset(self._by_kind)

    def peripherals(self) -> tuple[tuple[str, PeripheralDirection, int], ...]:
        return tuple(self._by_peripheral)


#: The default (all-placeholder) library.
PRIMITIVE_TECHNOLOGY = PrimitiveTechnologyLibrary(LIBRARY_NAME, placeholder_cells())

__all__ = ["LIBRARY_NAME", "PRIMITIVE_TECHNOLOGY", "PrimitiveTechnologyLibrary", "placeholder_cells"]
