"""The cell library: all component variations, loaded from YAML at import.

``LIBRARY`` is the merged lookup the placer and router resolve against.  Beyond
name lookup (it still behaves like the old ``dict``), it carries the two indexes
the tech-mapper needs: :meth:`Library.variants` to find every cell implementing
an IR operation at a given datapath width, and :meth:`Library.casts` for type
casts.  Boundary cells (constants, pads, clock source) are parametric and built
on demand, so they are intentionally *not* part of the library.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..components import Component, TypeCast
from .operation import OPERATIONS
from .primitive_gate import GATES
from .register import REGISTERS
from .type_cast import TYPE_CASTS
from .wiring import WIRING


class Library(Mapping):
    """Name lookup for cells, plus ``(op, width)`` and cast indexes."""

    def __init__(self, components: dict[str, Component]) -> None:
        self._by_name: dict[str, Component] = dict(components)
        self._by_op: dict[tuple[str, int], list[Component]] = {}
        self._by_cast: dict[tuple[int, int], list[Component]] = {}
        for cell in self._by_name.values():
            for key in cell.index_keys():
                self._by_op.setdefault(key, []).append(cell)
            if isinstance(cell, TypeCast):
                self._by_cast.setdefault(
                    (cell.source_width, cell.result_width), []
                ).append(cell)

    # -- Mapping protocol: keep the old dict-style name lookup working --------
    def __getitem__(self, name: str) -> Component:
        return self._by_name[name]

    def __iter__(self):
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    # -- tech-map queries ----------------------------------------------------
    def variants(self, op: str, width: int) -> tuple[Component, ...]:
        """Every cell implementing ``op`` at ``width`` (operand width for
        comparisons, data width for registers), in no particular order."""
        return tuple(self._by_op.get((op, width), ()))

    def casts(self, source_width: int, result_width: int) -> tuple[Component, ...]:
        """Every cell casting ``source_width`` bits to ``result_width`` bits."""
        return tuple(self._by_cast.get((source_width, result_width), ()))


#: Every enumerated cell variation, keyed by its convention name.
LIBRARY = Library({**OPERATIONS, **GATES, **WIRING, **REGISTERS, **TYPE_CASTS})

__all__ = [
    "GATES",
    "LIBRARY",
    "OPERATIONS",
    "REGISTERS",
    "TYPE_CASTS",
    "WIRING",
    "Library",
]
