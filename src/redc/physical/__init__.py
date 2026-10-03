"""Physical layer: the Minecraft-specific mapping target for the IR.

Contracts established here (technology mapping itself comes later):

* :mod:`.signals` -- the supported Minecraft datatypes (bool, u/int4/8/16/32/64)
  and how each is laid out physically (bool / binary / hex nibble lanes).
* :mod:`.implementation` -- exact-typed :class:`OperationSignature` keys and the
  :class:`ImplementationRegistry` of direct-cell and composite candidates.
* :mod:`.components` / :mod:`.cells` -- cell definitions and the YAML library.
* :mod:`.netlist` -- instances + nets, one global clock and reset domain.
* :mod:`.boundary` -- which pad or :class:`Peripheral` realizes each module port.
* :mod:`.techmap` -- :func:`lower_to_physical`: IR Graph -> unplaced netlist.
* :mod:`.simulate` -- functional netlist simulation, to check tech-map vs the IR.
* :mod:`.grid` -- the occupancy grid for placement and routing.
"""

from .boundary import (
    DEFAULT_BOUNDARY_POLICY,
    BoundaryPolicy,
    DefaultBoundaryPolicy,
    PadBoundaryPolicy,
)
from .cells import GATES, LIBRARY, OPERATIONS, REGISTERS, TYPE_CASTS, WIRING, Library
from .cells.peripheral import PERIPHERALS, PeripheralLibrary
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
    Peripheral,
    PeripheralDirection,
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
from .techmap import lower_to_physical, select_first_variant

__all__ = [
    "DEFAULT_BOUNDARY_POLICY",
    "EMPTY",
    "GATES",
    "LIBRARY",
    "MAX_HEIGHT",
    "OPERATIONS",
    "PERIPHERALS",
    "PHYSICAL_TYPES",
    "REGISTERS",
    "TYPE_CASTS",
    "WIRING",
    "Boundary",
    "BoundaryPolicy",
    "CellKind",
    "Clock",
    "ClockSource",
    "Component",
    "ComponentInstance",
    "CompositeImplementation",
    "Constant",
    "DefaultBoundaryPolicy",
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
    "PadBoundaryPolicy",
    "Peripheral",
    "PeripheralDirection",
    "PeripheralLibrary",
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
    "lower_to_physical",
    "require_supported_physical_type",
    "select_first_variant",
    "signal_layout",
]
