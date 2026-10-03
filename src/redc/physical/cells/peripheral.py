"""Peripheral devices, loaded from ``peripheral.yaml`` at import.

Peripherals are boundary realization choices, not IR operations, so they live in
their own :class:`PeripheralLibrary` keyed by ``(kind, datatype, direction)``
rather than in the operation-signature registry.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from ...ir import IRType
from ..components import Peripheral, PeripheralDirection
from ._loader import load_family

#: Stable peripheral kind identifiers used in the YAML ``kind`` field.
LEVER = "lever"
TWO_DIGIT_SEVEN_SEGMENT = "2-dig-7-seg"


class PeripheralLibrary(Mapping[str, Peripheral]):
    """Name lookup plus ``(kind, datatype, direction)`` lookup for peripherals."""

    def __init__(self, peripherals: dict[str, Peripheral]) -> None:
        self._by_name = dict(peripherals)

    def __getitem__(self, name: str) -> Peripheral:
        return self._by_name[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    def find(
        self, kind: str, dtype: IRType, direction: PeripheralDirection
    ) -> tuple[Peripheral, ...]:
        """Every variant of ``kind`` carrying ``dtype`` in ``direction``, in YAML
        order (deterministic)."""
        return tuple(
            p
            for p in self._by_name.values()
            if p.kind == kind and p.dtype == dtype and p.direction is direction
        )


#: Convention-name -> :class:`Peripheral` for every variant in the YAML.
PERIPHERALS = PeripheralLibrary(load_family("peripheral.yaml", Peripheral))
