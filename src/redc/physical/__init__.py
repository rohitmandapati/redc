"""Physical layer: the Minecraft-specific mapping target for the IR.

Contracts established here (technology mapping itself comes later):

* :mod:`.signals` -- the supported Minecraft datatypes (bool, u/int4/8/16/32/64)
  and how each is laid out physically (bool / binary / hex nibble lanes).
* :mod:`.implementation` -- exact-typed :class:`OperationSignature` keys and the
  :class:`ImplementationRegistry` of direct-cell and composite candidates.
* :mod:`.components` / :mod:`.cells` -- cell definitions and the YAML library.
* :mod:`.netlist` -- instances + nets, one global clock and reset domain.
* :mod:`.grid` -- the occupancy grid for placement and routing.
"""

from .cells import GATES, LIBRARY, OPERATIONS, REGISTERS, TYPE_CASTS, WIRING, Library
from .components import (
    Boundary,
    Clock,
    ClockSource,
    Component,
    Constant,
    Face,
    InputPad,
    Operation,
    OutputPad,
    Port,
    PortDir,
    PrimitiveGate,
    Register,
    ResetSource,
    TypeCast,
    Wiring,
)
from .grid import EMPTY, MAX_HEIGHT, CellKind, Grid
from .implementation import (
    CompositeImplementation,
    DirectCellImplementation,
    ImplementationRegistry,
    OperationSignature,
    PhysicalImplementation,
    RecipeBuilder,
)
from .netlist import ComponentInstance, Net, PhysicalNetlist, Terminal
from .signals import (
    PHYSICAL_TYPES,
    PhysicalSignalLayout,
    SignalEncoding,
    is_supported_physical_type,
    require_supported_physical_type,
    signal_layout,
)

__all__ = [
    "EMPTY",
    "GATES",
    "LIBRARY",
    "MAX_HEIGHT",
    "OPERATIONS",
    "PHYSICAL_TYPES",
    "REGISTERS",
    "TYPE_CASTS",
    "WIRING",
    "Boundary",
    "CellKind",
    "Clock",
    "ClockSource",
    "Component",
    "ComponentInstance",
    "CompositeImplementation",
    "Constant",
    "DirectCellImplementation",
    "Face",
    "Grid",
    "ImplementationRegistry",
    "InputPad",
    "Library",
    "Net",
    "Operation",
    "OperationSignature",
    "OutputPad",
    "PhysicalImplementation",
    "PhysicalNetlist",
    "PhysicalSignalLayout",
    "Port",
    "PortDir",
    "PrimitiveGate",
    "RecipeBuilder",
    "Register",
    "ResetSource",
    "SignalEncoding",
    "Terminal",
    "TypeCast",
    "Wiring",
    "is_supported_physical_type",
    "require_supported_physical_type",
    "signal_layout",
]
