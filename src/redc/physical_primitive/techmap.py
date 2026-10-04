"""Minecraft primitive technology mapping: logical primitives -> technology cells.

    synthesis   : add<uint8>  -> AND / XOR / OR graph        (what Boolean circuit?)
    tech mapping: logical XOR -> a Minecraft XOR cell         (which cell realizes it?)  <- here
    placement   : XOR instance -> world origin + orientation  (where?)
    routing     : connect its one-bit pins                    (how is it wired?)

The mapper picks a :class:`~redc.physical_primitive.technology.PrimitiveCell`
for every primitive from the library's deterministic candidate list via a
replaceable :data:`CellSelector` (v1: the first candidate).  It preserves the
logical primitive id, provenance (through ``realizes`` and the linked logical
netlist), bit-net connectivity and hierarchy.  It assigns NO coordinates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from ..parser import CompileError
from ..tracing import TraceLevel
from .netlist import PrimitiveInstance, PrimitiveNetlist
from .physical import MappedInstance, PhysicalBitNet, PrimitivePhysicalNetlist
from .synthesis.lower import SynthesisTrace
from .technology.cells import PrimitiveCell
from .technology.library import PRIMITIVE_TECHNOLOGY, PrimitiveTechnologyLibrary

#: ``(primitive, candidates) -> chosen cell`` -- the seam where a later,
#: placement- or timing-aware selector replaces :func:`select_first_cell`.
CellSelector = Callable[[PrimitiveInstance, Sequence[PrimitiveCell]], PrimitiveCell]


def select_first_cell(instance: PrimitiveInstance, candidates: Sequence[PrimitiveCell]) -> PrimitiveCell:
    """v1 policy: the first candidate in the library's deterministic order."""
    return candidates[0]


def map_primitives_to_minecraft(
    netlist: PrimitiveNetlist,
    *,
    library: PrimitiveTechnologyLibrary = PRIMITIVE_TECHNOLOGY,
    select: CellSelector = select_first_cell,
    trace: SynthesisTrace | None = None,
) -> PrimitivePhysicalNetlist:
    """Realize every primitive of ``netlist`` with a technology cell (unplaced)."""
    netlist.validate(complete=True)
    instances: list[MappedInstance] = []
    detailed = trace is not None and trace.wants(TraceLevel.DETAILED)
    for prim in netlist.instances:
        candidates = library.candidates(prim)
        if not candidates:
            what = prim.kind.value
            if prim.peripheral is not None:
                spec = prim.peripheral
                what = f"{spec.kind} {spec.direction.value} peripheral of width {spec.width}"
            raise CompileError(
                f"technology library {library.name!r} has no cell for primitive {prim.id} ({what})"
            )
        cell = select(prim, candidates)
        if not isinstance(cell, PrimitiveCell) or cell not in candidates:
            chosen = cell.name if isinstance(cell, PrimitiveCell) else repr(cell)
            raise CompileError(
                f"cell selector chose {chosen} for primitive {prim.id}, which is not a candidate"
            )
        instances.append(MappedInstance(prim.id, prim.kind, cell, (prim.id,)))
        if detailed:
            assert trace is not None
            trace.emit(
                "techmap", "primitive_mapped", level=TraceLevel.DETAILED,
                instance=prim.id, kind=prim.kind.value, cell=cell.name,
                candidates=[c.name for c in candidates],
            )  # fmt: skip
    nets = [PhysicalBitNet(net.id, net.driver, net.sinks, net.role) for net in netlist.nets]
    mapped = PrimitivePhysicalNetlist(netlist, library.name, instances, nets)
    mapped.validate()
    if trace is not None:
        trace.emit("techmap", "techmap_complete", **techmap_summary(mapped))
    return mapped


def techmap_summary(mapped: PrimitivePhysicalNetlist) -> dict[str, Any]:
    cells = Counter(inst.cell.name for inst in mapped.instances)
    return {
        "library": mapped.library,
        "instances": len(mapped.instances),
        "nets": len(mapped.nets),
        "cells": dict(sorted(cells.items())),
        "placeholder_cells": sorted({i.cell.name for i in mapped.instances if i.cell.placeholder}),
        "component_voxels": sum(len(i.cell.voxels) for i in mapped.instances),
    }


__all__ = ["CellSelector", "map_primitives_to_minecraft", "select_first_cell", "techmap_summary"]
