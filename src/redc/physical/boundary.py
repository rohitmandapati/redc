"""Boundary realization: which component stands for each module port.

Technology mapping asks a :class:`BoundaryPolicy` what to instantiate for every
graph input and output, so the choice between an abstract pad and a real
Minecraft :class:`~redc.physical.components.Peripheral` lives in exactly one
replaceable place.  Today the choice is a fixed default; once RedC source can
annotate ports (``bool<Lever> start``, ``uint8<2-dig-7-seg> result``) a policy
built from that interface metadata replaces :class:`DefaultBoundaryPolicy`
without touching the mapper or :class:`~redc.physical.netlist.PhysicalNetlist`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..ir import BOOL, IRType
from .cells.peripheral import (
    LEVER,
    PERIPHERALS,
    TWO_DIGIT_SEVEN_SEGMENT,
    PeripheralLibrary,
)
from .components import Component, InputPad, OutputPad, PeripheralDirection

UINT8 = IRType(8)


class BoundaryPolicy(Protocol):
    """Chooses the physical component realizing one module port.

    ``realize_input`` must return a component with no inputs and one output of
    type ``typ``; ``realize_output`` one with a single input of type ``typ`` and
    no outputs."""

    def realize_input(self, name: str, typ: IRType) -> Component: ...

    def realize_output(self, name: str, typ: IRType) -> Component: ...


@dataclass(frozen=True)
class DefaultBoundaryPolicy:
    """The temporary v1 policy, until ports carry peripheral annotations:

    * input ``start : bool``    -> lever peripheral (the user launches the FSM);
    * output ``result : uint8`` -> two-digit seven-segment display peripheral;
    * every other port          -> abstract :class:`InputPad` / :class:`OutputPad`
      (including ``done``, and ``result`` of any other type).

    A rule only applies when the library actually has a matching variant, so a
    missing peripheral degrades to a pad instead of a wrong device."""

    peripherals: PeripheralLibrary = field(default_factory=lambda: PERIPHERALS)

    def realize_input(self, name: str, typ: IRType) -> Component:
        if name == "start" and typ == BOOL:
            found = self.peripherals.find(LEVER, typ, PeripheralDirection.INPUT)
            if found:
                return found[0]
        return InputPad.of(name, typ)

    def realize_output(self, name: str, typ: IRType) -> Component:
        if name == "result" and typ == UINT8:
            found = self.peripherals.find(
                TWO_DIGIT_SEVEN_SEGMENT, typ, PeripheralDirection.OUTPUT
            )
            if found:
                return found[0]
        return OutputPad.of(name, typ)


@dataclass(frozen=True)
class PadBoundaryPolicy:
    """Every port becomes an abstract pad (no peripherals at all)."""

    def realize_input(self, name: str, typ: IRType) -> Component:
        return InputPad.of(name, typ)

    def realize_output(self, name: str, typ: IRType) -> Component:
        return OutputPad.of(name, typ)


DEFAULT_BOUNDARY_POLICY = DefaultBoundaryPolicy()
