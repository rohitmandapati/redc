"""The technology-mapped, block-level primitive netlist.

:func:`~redc.physical_primitive.techmap.map_primitives_to_minecraft` produces a
:class:`PrimitivePhysicalNetlist` from a
:class:`~redc.physical_primitive.netlist.PrimitiveNetlist`: every logical
primitive is realized by one :class:`MappedInstance` of a
:class:`~redc.physical_primitive.technology.PrimitiveCell`.  Technology mapping
decides WHICH cell realizes each primitive, never WHERE: every mapped instance
starts with ``origin is None`` and ``orientation is None``; placement assigns
both.

v1 mapping is one-to-one, so a mapped instance's id equals the id of the
logical primitive it realizes (``realizes == (id,)``) and each
:class:`PhysicalBitNet` keeps its logical net's id and terminals.  The
``realizes`` tuple is the seam for a later many-to-one mapping (e.g. fusing an
AND followed by a NOT into one NAND cell).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..parser import CompileError
from .geometry import Coord, Orientation
from .netlist import BitTerminal, PrimitiveKind, PrimitiveNetlist
from .technology.cells import PlacedCell, PrimitiveCell

SCHEMA = "redc.primitive-mapped-netlist.v1"


@dataclass(eq=False)
class MappedInstance:
    """One placed-or-placeable technology cell instance."""

    id: int
    kind: PrimitiveKind
    cell: PrimitiveCell
    realizes: tuple[int, ...]
    origin: Coord | None = None
    orientation: Orientation | None = None
    _placed: PlacedCell | None = field(default=None, repr=False)

    @property
    def is_placed(self) -> bool:
        return self.origin is not None

    def place(self, origin: Coord, orientation: Orientation) -> PlacedCell:
        placed = self.cell.oriented(orientation).translate(origin)
        self.origin, self.orientation, self._placed = origin, orientation, placed
        return placed

    def unplace(self) -> None:
        self.origin = self.orientation = self._placed = None

    @property
    def placed(self) -> PlacedCell:
        if self._placed is None:
            raise CompileError(f"mapped instance {self.id} ({self.cell.name}) is not placed")
        return self._placed

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "cell": self.cell.name,
            "realizes": list(self.realizes),
            "origin": None if self.origin is None else list(self.origin),
            "orientation": None if self.orientation is None else self.orientation.name,
        }


@dataclass(frozen=True, slots=True)
class PhysicalBitNet:
    """One one-bit net between mapped-instance pins (same id as its logical net)."""

    id: int
    driver: BitTerminal
    sinks: tuple[BitTerminal, ...]
    role: str = "data"

    @property
    def fanout(self) -> int:
        return len(self.sinks)


@dataclass
class PrimitivePhysicalNetlist:
    """Mapped instances + one-bit nets, linked to the logical netlist."""

    logical: PrimitiveNetlist
    library: str
    instances: list[MappedInstance]
    nets: list[PhysicalBitNet]

    def instance(self, instance_id: int) -> MappedInstance:
        if not 0 <= instance_id < len(self.instances):
            raise CompileError(f"unknown mapped instance {instance_id}")
        return self.instances[instance_id]

    def cells_used(self) -> dict[str, PrimitiveCell]:
        """Distinct technology cells in first-use order."""
        used: dict[str, PrimitiveCell] = {}
        for inst in self.instances:
            used.setdefault(inst.cell.name, inst.cell)
        return used

    def net_of_terminal(self) -> dict[BitTerminal, int]:
        """Every connected terminal -> its net id."""
        index: dict[BitTerminal, int] = {}
        for net in self.nets:
            index[net.driver] = net.id
            for sink in net.sinks:
                index[sink] = net.id
        return index

    def validate(self) -> None:
        """Mapping is 1:1, kinds match, every logical pin exists on the cell
        with the same direction, and nets mirror the logical nets exactly."""
        logical = self.logical
        if len(self.instances) != len(logical.instances):
            raise CompileError("technology mapping must realize every primitive exactly once")
        for index, inst in enumerate(self.instances):
            prim = logical.instances[index]
            if inst.id != index or inst.realizes != (prim.id,):
                raise CompileError(f"mapped instance {inst.id} does not realize primitive {index}")
            if inst.kind is not prim.kind or inst.cell.kind is not prim.kind:
                raise CompileError(
                    f"mapped instance {inst.id}: cell {inst.cell.name} ({inst.cell.kind.value}) "
                    f"cannot realize a {prim.kind.value}"
                )
            if prim.peripheral is not None:
                spec = prim.peripheral
                if inst.cell.peripheral != (spec.kind, spec.direction, spec.width):
                    raise CompileError(
                        f"mapped instance {inst.id}: cell {inst.cell.name} is not a {spec.kind} "
                        f"{spec.direction.value} peripheral of width {spec.width}"
                    )
            cell_pins = {p.name: p.direction for p in inst.cell.pins}
            want = {pin: "in" for pin in prim.inputs} | {pin: "out" for pin in prim.outputs}
            if cell_pins != want:
                raise CompileError(
                    f"mapped instance {inst.id}: cell {inst.cell.name} pins {cell_pins} do not match "
                    f"primitive pins {want}"
                )
        if len(self.nets) != len(logical.nets):
            raise CompileError("physical nets must mirror the logical nets one to one")
        for net, lnet in zip(self.nets, logical.nets):
            if (net.id, net.driver, net.sinks, net.role) != (lnet.id, lnet.driver, lnet.sinks, lnet.role):
                raise CompileError(f"physical net {net.id} differs from logical net {lnet.id}")

    def to_dict(self) -> dict[str, Any]:
        """The ``redc.primitive-mapped-netlist.v1`` debug view (unplaced)."""
        return {
            "schema": SCHEMA,
            "backend": "physical-primitive",
            "stage": "mapped",
            "library": self.library,
            "cells": [cell.to_dict() for cell in self.cells_used().values()],
            "instances": [inst.to_dict() for inst in self.instances],
            "nets": [
                {
                    "id": net.id,
                    "role": net.role,
                    "driver": net.driver.ref(),
                    "sinks": [s.ref() for s in net.sinks],
                    "fanout": net.fanout,
                }
                for net in self.nets
            ],
            "logical": self.logical.to_dict(),
        }


__all__ = ["SCHEMA", "MappedInstance", "PhysicalBitNet", "PrimitivePhysicalNetlist"]
