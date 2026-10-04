"""Backend-neutral Minecraft circuits: representation, simulation, timing.

    any RedC Minecraft backend
              |  materialize
              v
    MinecraftPhysicalDesign          (:mod:`.design`)   blocks + abstract components + ports
              |
       +------+-------+
       v              v
    RedstoneSimulator   static timing analysis      (:mod:`.simulator`, :mod:`.sta`)
       (dynamic)        (analytic)
       \\_____ one timing model (:mod:`.timing`) and one geometry-derived
              connectivity (:mod:`.connectivity`) shared by both _____/

Nothing in this package knows RedC IR operations, primitive synthesis or any
backend's component classes: only blocks, coordinates, block states, ports,
probes and (explicitly ABSTRACT) components with declared behaviour.
"""

from .abstract import Capture, TimingViolation
from .behaviors import SUPPORTED_BLOCKS, UNSUPPORTED_MECHANICS, Diagnostic
from .blocks import Coord, MinecraftBlock
from .connectivity import CompiledWorld, SimulationError, compile_world
from .design import (
    ABSTRACT_MODE,
    BLOCK_MODE,
    DESIGN_SCHEMA,
    MINECRAFT_VERSION,
    AbstractComponent,
    ComponentPin,
    MinecraftPhysicalDesign,
    Port,
    PortBit,
    Probe,
)
from .simulator import SIMULATION_MODEL, RedstoneSimulator, simulate_combinational
from .timing import (
    TIMING_MODEL,
    CombinationalArc,
    ComponentTiming,
    SequentialTiming,
    repeater_delay_gt,
)
from .units import GAME_TICKS_PER_REDSTONE_TICK, gt_to_rt, rt_to_gt

__all__ = [
    "ABSTRACT_MODE",
    "BLOCK_MODE",
    "DESIGN_SCHEMA",
    "GAME_TICKS_PER_REDSTONE_TICK",
    "MINECRAFT_VERSION",
    "SIMULATION_MODEL",
    "SUPPORTED_BLOCKS",
    "TIMING_MODEL",
    "UNSUPPORTED_MECHANICS",
    "AbstractComponent",
    "Capture",
    "CombinationalArc",
    "CompiledWorld",
    "ComponentPin",
    "ComponentTiming",
    "Coord",
    "Diagnostic",
    "MinecraftBlock",
    "MinecraftPhysicalDesign",
    "Port",
    "PortBit",
    "Probe",
    "RedstoneSimulator",
    "SequentialTiming",
    "SimulationError",
    "TimingViolation",
    "compile_world",
    "gt_to_rt",
    "repeater_delay_gt",
    "rt_to_gt",
    "simulate_combinational",
]
