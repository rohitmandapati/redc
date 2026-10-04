"""``physical-primitive``: a second, independent Minecraft physical backend that
lowers EVERYTHING to one-bit primitive redstone logic.

    redc.ir.Graph
        |  primitive synthesis            (:mod:`.synthesis`)
        v
    PrimitiveNetlist                      (:mod:`.netlist`)   technology-neutral, one-bit
        |  Minecraft primitive tech-map   (:mod:`.techmap`, :mod:`.technology`)
        v
    PrimitivePhysicalNetlist              (:mod:`.physical`)  unplaced block-level cells
        |  placement -> single-bit routing -> redstone electrical legalization
        v                                 (:mod:`.pnr`)
    LegalizedPrimitiveDesign  ->  redc.physical-primitive.v1 JSON + replay trace

Unlike the coarse ``redc.physical`` backend (wide components, bus-valued nets,
abstract grid cells), here a W-bit value is W separate one-bit signals and W
separately routed nets, and ONE grid coordinate is ONE Minecraft block.  No
opaque adder, multiplier, divider, comparator, shifter, mux or wide register
survives synthesis: they are all AND / OR / XOR / NOT gates and one-bit
register bits.  Huge circuits are expected and intentional.
"""

from .backend import PrimitivePhysicalBackend, place_and_route_graph
from .netlist import (
    GATE_KINDS,
    Bit,
    BitNet,
    BitTerminal,
    BitVector,
    HierarchyGroup,
    LogicalBus,
    LogicalPort,
    PeripheralDirection,
    PeripheralSpec,
    PrimitiveInstance,
    PrimitiveKind,
    PrimitiveNetlist,
    Provenance,
)
from .physical import MappedInstance, PhysicalBitNet, PrimitivePhysicalNetlist
from .pnr import (
    PHYSICAL_SCHEMA,
    TRACE_SCHEMA,
    PrimitivePnRConfig,
    PrimitivePnRResult,
    PrimitiveTraceRecorder,
    place_and_route_primitive,
)
from .simulate import PrimitiveSimulator
from .synthesis import (
    DEFAULT_INTERFACE_POLICY,
    DefaultInterfacePolicy,
    InterfacePolicy,
    PadInterfacePolicy,
    PortRealization,
    PrimitiveBuilder,
    SynthesisRegistry,
    synthesize_to_primitives,
)
from .techmap import map_primitives_to_minecraft, select_first_cell
from .technology import (
    PRIMITIVE_TECHNOLOGY,
    PrimitiveCell,
    PrimitivePin,
    PrimitiveTechnologyLibrary,
)

BACKEND_NAME = "physical-primitive"

__all__ = [
    "BACKEND_NAME",
    "DEFAULT_INTERFACE_POLICY",
    "GATE_KINDS",
    "PHYSICAL_SCHEMA",
    "PRIMITIVE_TECHNOLOGY",
    "TRACE_SCHEMA",
    "Bit",
    "BitNet",
    "BitTerminal",
    "BitVector",
    "DefaultInterfacePolicy",
    "HierarchyGroup",
    "InterfacePolicy",
    "LogicalBus",
    "LogicalPort",
    "MappedInstance",
    "PadInterfacePolicy",
    "PeripheralDirection",
    "PeripheralSpec",
    "PhysicalBitNet",
    "PortRealization",
    "PrimitiveBuilder",
    "PrimitiveCell",
    "PrimitiveInstance",
    "PrimitiveKind",
    "PrimitiveNetlist",
    "PrimitivePhysicalBackend",
    "PrimitivePhysicalNetlist",
    "PrimitivePin",
    "PrimitivePnRConfig",
    "PrimitivePnRResult",
    "PrimitiveSimulator",
    "PrimitiveTechnologyLibrary",
    "PrimitiveTraceRecorder",
    "Provenance",
    "SynthesisRegistry",
    "map_primitives_to_minecraft",
    "place_and_route_graph",
    "place_and_route_primitive",
    "select_first_cell",
    "synthesize_to_primitives",
]
