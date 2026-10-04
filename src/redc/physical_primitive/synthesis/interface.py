"""How each module port is realized at the primitive level: one-bit pads, or
ONE conceptual peripheral device with independent one-bit pins.

This is the primitive backend's equivalent of the coarse backend's
:class:`~redc.physical.boundary.BoundaryPolicy`, moved to the synthesis seam
because the choice is a property of the *interface*, not of Minecraft geometry:
once RedC source can annotate ports (``bool<Lever> start``,
``uint8<2-dig-7-seg> result``) a policy built from those annotations replaces
:class:`DefaultInterfacePolicy`.  The peripheral ``kind`` is just a name here;
the primitive technology library maps it to a (placeholder) device geometry.

Either realization exposes exactly ``width`` independent one-bit connections
grouped under one :class:`~redc.physical_primitive.netlist.LogicalPort`, so the
simulator and every later phase treat both identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ...ir import BOOL, IRType

#: Interface kind names, spelled exactly like the coarse backend's peripheral
#: kinds and the planned source annotations.
LEVER = "lever"
TWO_DIGIT_SEVEN_SEGMENT = "2-dig-7-seg"

UINT8 = IRType(8)


@dataclass(frozen=True, slots=True)
class PortRealization:
    """``peripheral=None`` realizes the port as one pad primitive per bit;
    otherwise as one PERIPHERAL of that kind with one pin per bit."""

    peripheral: str | None = None

    @property
    def label(self) -> str:
        return self.peripheral or "pads"


PADS = PortRealization()


class InterfacePolicy(Protocol):
    """Chooses how one module port is realized."""

    def realize_input(self, name: str, typ: IRType) -> PortRealization: ...

    def realize_output(self, name: str, typ: IRType) -> PortRealization: ...


@dataclass(frozen=True)
class DefaultInterfacePolicy:
    """The temporary v1 policy (same rules as the coarse backend):

    * input ``start : bool``    -> one ``lever`` peripheral;
    * output ``result : uint8`` -> one ``2-dig-7-seg`` display peripheral with
      eight independent one-bit inputs;
    * every other port          -> one pad primitive per bit."""

    def realize_input(self, name: str, typ: IRType) -> PortRealization:
        if name == "start" and typ == BOOL:
            return PortRealization(LEVER)
        return PADS

    def realize_output(self, name: str, typ: IRType) -> PortRealization:
        if name == "result" and typ == UINT8:
            return PortRealization(TWO_DIGIT_SEVEN_SEGMENT)
        return PADS


@dataclass(frozen=True)
class PadInterfacePolicy:
    """Every port becomes one pad primitive per bit (no peripherals)."""

    def realize_input(self, name: str, typ: IRType) -> PortRealization:
        return PADS

    def realize_output(self, name: str, typ: IRType) -> PortRealization:
        return PADS


DEFAULT_INTERFACE_POLICY = DefaultInterfacePolicy()

__all__ = [
    "DEFAULT_INTERFACE_POLICY",
    "LEVER",
    "PADS",
    "TWO_DIGIT_SEVEN_SEGMENT",
    "DefaultInterfacePolicy",
    "InterfacePolicy",
    "PadInterfacePolicy",
    "PortRealization",
]
