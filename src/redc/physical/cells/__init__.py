"""The cell library: all component variations, loaded from YAML at import.

``LIBRARY`` is the merged lookup the placer and router resolve against.  Beyond
name lookup (it still behaves like the old ``dict``), it owns an
:class:`~redc.physical.implementation.ImplementationRegistry` in which every
operation cell is registered as a
:class:`~redc.physical.implementation.DirectCellImplementation` under its exact
:class:`~redc.physical.implementation.OperationSignature`.  Composite (one-to-
many) implementations can be registered into the same registry later without
changing it.  Candidates come back in a deterministic order: family order below,
then YAML order.

Boundary cells (constants, pads, clock and reset sources) are parametric and
built on demand, so they are intentionally *not* part of the library.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from ..components import Component
from ..implementation import (
    DirectCellImplementation,
    ImplementationRegistry,
    OperationSignature,
    PhysicalImplementation,
)
from .operation import OPERATIONS
from .primitive_gate import GATES
from .register import REGISTERS
from .type_cast import TYPE_CASTS
from .wiring import WIRING


class Library(Mapping[str, Component]):
    """Name lookup for cells, plus exact-signature implementation lookup."""

    def __init__(self, components: dict[str, Component]) -> None:
        self._by_name: dict[str, Component] = dict(components)
        self.implementations = ImplementationRegistry(
            DirectCellImplementation(cell)
            for cell in self._by_name.values()
            if cell.signatures()
        )

    # -- Mapping protocol: keep the old dict-style name lookup working --------
    def __getitem__(self, name: str) -> Component:
        return self._by_name[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    # -- tech-map queries ----------------------------------------------------
    def candidates(self, signature: OperationSignature) -> tuple[PhysicalImplementation, ...]:
        """Every implementation (direct or composite) of ``signature``."""
        return self.implementations.candidates(signature)

    def cells(self, signature: OperationSignature) -> tuple[Component, ...]:
        """Every single cell implementing exactly ``signature``, in a
        deterministic order (v1 tech-map may simply take the first)."""
        return self.implementations.direct_cells(signature)


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
