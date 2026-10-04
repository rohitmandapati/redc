"""Per-block behaviour of the supported redstone subset (``redc-redstone-sim-v1``).

Every supported block id maps to ONE :class:`BlockBehavior` in
:data:`BEHAVIORS`.  A behaviour answers two kinds of questions:

* **static** (geometry only, used by :func:`redc.minecraft.connectivity.compile_world`,
  which the simulator AND static timing analysis share): is the block a
  redstone conductor, can dust rest on it, does dust connect to it sideways,
  which neighbours does it power (weakly / strongly) when on, which
  neighbours does it read its input from, and is its placement valid;
* **dynamic** (used by the simulator): what happens when its input changes
  (:meth:`BlockBehavior.on_input`) and when its scheduled tick fires
  (:meth:`BlockBehavior.on_tick`).  Delays come from :mod:`redc.minecraft.timing`.

Java Edition facts encoded here (the "signal" a block gives toward a
neighbour is Java's ``getSignal``; "strong" is ``getDirectSignal``):

* dust: see :mod:`redc.minecraft.connectivity` (shape, attenuation);
* repeater: reads ONLY the block behind it (the blockstate ``facing`` side);
  powers only the block in front, strongly; delay 1..4 redstone ticks; a
  pulse shorter than the delay is extended to the delay;
* redstone torch: reads the block it is attached to (below for a standing
  torch, behind for a wall torch); is lit iff that block is unpowered;
  powers every neighbour except its attachment, strongly powers the block
  above; toggles one redstone tick after its input changes (no extension);
* lever: powers every neighbour, strongly powers its attachment block;
  its state is set from outside (a port);
* redstone block: always on; powers every neighbour, strongly powers nothing;
* redstone lamp: a conductor that lights at once when any neighbour gives it a
  signal and turns off two redstone ticks after the last signal goes away;
* stone / abstract blocks: conductors dust rests on (an abstract block is
  inert: it can be powered but nothing reads it);
* glass: dust rests on it, but it is NOT a conductor (cannot be powered,
  never cuts a dust staircase).

Explicitly UNSUPPORTED (an explicit diagnostic, never a silent
approximation): every other block id (comparators, observers, pistons,
buttons, ...), repeater locking, torch burnout (detected and reported as an
unstable circuit), quasi-connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .blocks import (
    ABSTRACT_ID,
    AIR_ID,
    DIRECTION_VECTORS,
    GLASS_ID,
    HORIZONTAL_DIRECTIONS,
    LAMP_ID,
    LEVER_ID,
    OPPOSITE,
    REDSTONE_BLOCK_ID,
    REPEATER_ID,
    STONE_ID,
    TORCH_ID,
    WALL_TORCH_ID,
    WIRE_ID,
    Coord,
    MinecraftBlock,
    direction_between,
    offset,
)
from .timing import LAMP_OFF_DELAY_GT, TORCH_DELAY_GT, repeater_delay_gt

if TYPE_CHECKING:
    from .simulator import RedstoneSimulator

#: Reading modes of a reader's input coordinate (see ``connectivity``):
#: ``raw`` -- dust there counts with its full power regardless of its shape
#: (a repeater reading the dust behind it); ``signal`` -- dust there counts
#: only if it points at / sits on the reader (Java ``getSignal``).
READ_RAW = "raw"
READ_SIGNAL = "signal"

#: Torch burnout: Java burns a torch out after 8 toggles within 60 game ticks.
BURNOUT_TOGGLES = 8
BURNOUT_WINDOW_GT = 60


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A problem found in a design.  ``code`` is a ``simulation/...`` failure code."""

    code: str
    message: str
    coord: Coord | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "coord": None if self.coord is None else list(self.coord)}


class BlockBehavior:
    """Base class: an inert, non-conducting, non-supporting block."""

    block_id = ""
    #: Java ``isRedstoneConductor``: a full opaque block that can be powered
    #: and that cuts a dust staircase passing over it.
    conductor = False
    #: Dust, repeaters and torches may rest on top of it.
    sturdy_top = False
    #: Java ``isSignalSource``: dust beside it connects to it.
    signal_source = False
    #: Not a real Minecraft block (forces abstract simulation mode).
    abstract = False
    #: Kind of power source this block is (``None`` = not a source).
    source_kind: str | None = None
    #: Device kind the simulator instantiates for it (``None`` = static block).
    device_kind: str | None = None

    # -- static ----------------------------------------------------------------

    def validate(self, view: WorldView, coord: Coord, block: MinecraftBlock) -> list[Diagnostic]:
        return []

    def connects_sideways(self, block: MinecraftBlock, direction: str) -> bool:
        """Whether dust whose neighbour (in ``direction``) is this block connects to it."""
        return self.signal_source

    def emits_toward(self, block: MinecraftBlock, coord: Coord, target: Coord) -> bool:
        """Whether, when on, this block gives a signal to the neighbour ``target``."""
        return False

    def strongly_powers(self, block: MinecraftBlock, coord: Coord, target: Coord) -> bool:
        """Whether, when on, this block strongly powers the neighbour ``target``."""
        return False

    def inputs(self, block: MinecraftBlock, coord: Coord) -> list[tuple[Coord, str]]:
        """``(coordinate, read mode)`` pairs this block takes its input from."""
        return []

    def initially_on(self, block: MinecraftBlock) -> bool:
        return False

    def static_state(self, block: MinecraftBlock) -> dict[str, Any]:
        return {}

    # -- dynamic -----------------------------------------------------------------

    def on_input(self, sim: RedstoneSimulator, device: int, value: int) -> None:
        """The device's input strength changed to ``value``."""

    def on_tick(self, sim: RedstoneSimulator, device: int) -> None:
        """A tick this device scheduled fires now."""


class WorldView:
    """Read-only geometric queries the static rules need."""

    def __init__(self, blocks: dict[Coord, MinecraftBlock]) -> None:
        self.blocks = blocks

    def block(self, coord: Coord) -> MinecraftBlock | None:
        return self.blocks.get(coord)

    def behavior(self, coord: Coord) -> BlockBehavior:
        block = self.blocks.get(coord)
        return AIR if block is None else BEHAVIORS.get(block.id, UNSUPPORTED)

    def is_conductor(self, coord: Coord) -> bool:
        return self.behavior(coord).conductor

    def is_sturdy_top(self, coord: Coord) -> bool:
        return self.behavior(coord).sturdy_top

    def is_wire(self, coord: Coord) -> bool:
        block = self.blocks.get(coord)
        return block is not None and block.id == WIRE_ID


def _needs_support(view: WorldView, coord: Coord, what: str) -> list[Diagnostic]:
    below = offset(coord, "down")
    if view.is_sturdy_top(below):
        return []
    found = view.block(below)
    return [
        Diagnostic(
            "simulation/unsupported_placement",
            f"{what} at {list(coord)} rests on {found or 'air'}, which cannot hold it (it would pop off)",
            coord,
        )
    ]


class _Air(BlockBehavior):
    block_id = AIR_ID


class _Unsupported(BlockBehavior):
    block_id = "<unsupported>"


class _Solid(BlockBehavior):
    """A full opaque conductor (stone)."""

    block_id = STONE_ID
    conductor = True
    sturdy_top = True


class _Glass(BlockBehavior):
    block_id = GLASS_ID
    sturdy_top = True


class _Abstract(_Solid):
    block_id = ABSTRACT_ID
    abstract = True


class _Wire(BlockBehavior):
    """Redstone dust.  Its electrical rules live in ``connectivity`` because
    they depend on the whole neighbourhood; the simulator recomputes dust
    networks itself (dust has no scheduled state)."""

    block_id = WIRE_ID

    def connects_sideways(self, block: MinecraftBlock, direction: str) -> bool:
        return True

    def validate(self, view: WorldView, coord: Coord, block: MinecraftBlock) -> list[Diagnostic]:
        return _needs_support(view, coord, "redstone dust")


class _Repeater(BlockBehavior):
    block_id = REPEATER_ID
    source_kind = "repeater"
    device_kind = "repeater"

    @staticmethod
    def input_side(block: MinecraftBlock) -> str:
        return str(block.get("facing"))

    @staticmethod
    def output_side(block: MinecraftBlock) -> str:
        return OPPOSITE[str(block.get("facing"))]

    @staticmethod
    def delay_rt(block: MinecraftBlock) -> int:
        return int(block.get("delay", 1))

    def validate(self, view: WorldView, coord: Coord, block: MinecraftBlock) -> list[Diagnostic]:
        problems = _needs_support(view, coord, "repeater")
        if block.get("facing") not in HORIZONTAL_DIRECTIONS:
            problems.append(Diagnostic("simulation/unsupported_state", f"repeater at {list(coord)} needs a horizontal facing", coord))
        if block.get("delay", 1) not in (1, 2, 3, 4):
            problems.append(Diagnostic("simulation/unsupported_state", f"repeater at {list(coord)} delay must be 1..4", coord))
        if block.get("locked") is True:
            problems.append(Diagnostic("simulation/unsupported_mechanic", f"repeater at {list(coord)} is locked (unsupported)", coord))
        if problems:
            return problems
        # Locking: a diode pointing into a repeater's SIDE locks it -- unsupported.
        axis = DIRECTION_VECTORS[self.input_side(block)]
        for direction in HORIZONTAL_DIRECTIONS:
            vector = DIRECTION_VECTORS[direction]
            if vector == axis or vector == (-axis[0], 0, -axis[2]):
                continue
            side = offset(coord, direction)
            other = view.block(side)
            if other is not None and other.id == REPEATER_ID and offset(side, self.output_side(other)) == coord:
                problems.append(
                    Diagnostic(
                        "simulation/unsupported_mechanic",
                        f"repeater at {list(side)} points into the side of the repeater at {list(coord)}: "
                        "repeater locking is not modelled",
                        coord,
                    )
                )
        return problems

    def connects_sideways(self, block: MinecraftBlock, direction: str) -> bool:
        # Dust connects to a repeater's front and back, never its sides.
        return direction in (self.input_side(block), self.output_side(block))

    def emits_toward(self, block: MinecraftBlock, coord: Coord, target: Coord) -> bool:
        return target == offset(coord, self.output_side(block))

    def strongly_powers(self, block: MinecraftBlock, coord: Coord, target: Coord) -> bool:
        return target == offset(coord, self.output_side(block))

    def inputs(self, block: MinecraftBlock, coord: Coord) -> list[tuple[Coord, str]]:
        return [(offset(coord, self.input_side(block)), READ_RAW)]

    def static_state(self, block: MinecraftBlock) -> dict[str, Any]:
        return {"delay_gt": repeater_delay_gt(self.delay_rt(block))}

    def on_input(self, sim: RedstoneSimulator, device: int, value: int) -> None:
        powered = sim.device_on(device)
        if powered != (value > 0) and not sim.has_pending_tick(device):
            delay = sim.devices[device].static["delay_gt"]
            sim.schedule_tick(device, delay)
            sim.record("repeater_scheduled", coord=sim.devices[device].coord, at=sim.time + delay)

    def on_tick(self, sim: RedstoneSimulator, device: int) -> None:
        powered = sim.device_on(device)
        should = sim.device_input(device) > 0
        if powered and not should:
            sim.set_device_output(device, False)
            sim.record("repeater_output_changed", coord=sim.devices[device].coord, powered=False)
        elif not powered:
            sim.set_device_output(device, True)
            sim.record("repeater_output_changed", coord=sim.devices[device].coord, powered=True)
            if not should:
                # Pulse extension: the repeater stays on for a full delay.
                delay = sim.devices[device].static["delay_gt"]
                sim.schedule_tick(device, delay)
                sim.record("repeater_scheduled", coord=sim.devices[device].coord, at=sim.time + delay)


class _Torch(BlockBehavior):
    """A standing redstone torch (attached to the block below)."""

    block_id = TORCH_ID
    signal_source = True
    source_kind = "torch"
    device_kind = "torch"

    def attached(self, block: MinecraftBlock, coord: Coord) -> Coord:
        return offset(coord, "down")

    def validate(self, view: WorldView, coord: Coord, block: MinecraftBlock) -> list[Diagnostic]:
        return _needs_support(view, coord, "redstone torch")

    def emits_toward(self, block: MinecraftBlock, coord: Coord, target: Coord) -> bool:
        return target != self.attached(block, coord) and direction_between(coord, target) is not None

    def strongly_powers(self, block: MinecraftBlock, coord: Coord, target: Coord) -> bool:
        return target == offset(coord, "up")

    def inputs(self, block: MinecraftBlock, coord: Coord) -> list[tuple[Coord, str]]:
        return [(self.attached(block, coord), READ_SIGNAL)]

    def on_input(self, sim: RedstoneSimulator, device: int, value: int) -> None:
        lit = sim.device_on(device)
        if lit == (value > 0) and not sim.has_pending_tick(device):
            sim.schedule_tick(device, TORCH_DELAY_GT)
            sim.record("torch_scheduled", coord=sim.devices[device].coord, at=sim.time + TORCH_DELAY_GT)

    def on_tick(self, sim: RedstoneSimulator, device: int) -> None:
        lit = sim.device_on(device)
        powered = sim.device_input(device) > 0
        if lit == powered:
            sim.set_device_output(device, not lit)
            sim.record("torch_changed", coord=sim.devices[device].coord, lit=not lit)
            sim.note_torch_toggle(device)


class _WallTorch(_Torch):
    block_id = WALL_TORCH_ID

    def attached(self, block: MinecraftBlock, coord: Coord) -> Coord:
        return offset(coord, OPPOSITE[str(block.get("facing"))])

    def validate(self, view: WorldView, coord: Coord, block: MinecraftBlock) -> list[Diagnostic]:
        if block.get("facing") not in HORIZONTAL_DIRECTIONS:
            return [Diagnostic("simulation/unsupported_state", f"wall torch at {list(coord)} needs a horizontal facing", coord)]
        wall = self.attached(block, coord)
        if view.behavior(wall).conductor or view.behavior(wall).sturdy_top:
            return []
        return [
            Diagnostic(
                "simulation/unsupported_placement",
                f"wall torch at {list(coord)} hangs on {view.block(wall) or 'air'} (it would pop off)",
                coord,
            )
        ]


class _Lever(BlockBehavior):
    block_id = LEVER_ID
    signal_source = True
    source_kind = "lever"
    device_kind = "lever"

    def attached(self, block: MinecraftBlock, coord: Coord) -> Coord:
        face = block.get("face", "floor")
        if face == "floor":
            return offset(coord, "down")
        if face == "ceiling":
            return offset(coord, "up")
        return offset(coord, OPPOSITE[str(block.get("facing"))])

    def validate(self, view: WorldView, coord: Coord, block: MinecraftBlock) -> list[Diagnostic]:
        if block.get("face", "floor") not in ("floor", "wall", "ceiling") or block.get("facing", "north") not in HORIZONTAL_DIRECTIONS:
            return [Diagnostic("simulation/unsupported_state", f"lever at {list(coord)} has a bad face/facing", coord)]
        mount = self.attached(block, coord)
        if view.behavior(mount).conductor or view.behavior(mount).sturdy_top:
            return []
        return [Diagnostic("simulation/unsupported_placement", f"lever at {list(coord)} is attached to nothing", coord)]

    def emits_toward(self, block: MinecraftBlock, coord: Coord, target: Coord) -> bool:
        return direction_between(coord, target) is not None

    def strongly_powers(self, block: MinecraftBlock, coord: Coord, target: Coord) -> bool:
        return target == self.attached(block, coord)


class _RedstoneBlock(BlockBehavior):
    block_id = REDSTONE_BLOCK_ID
    sturdy_top = True
    signal_source = True
    source_kind = "redstone_block"
    device_kind = "constant"

    def emits_toward(self, block: MinecraftBlock, coord: Coord, target: Coord) -> bool:
        return direction_between(coord, target) is not None

    def initially_on(self, block: MinecraftBlock) -> bool:
        return True


class _Lamp(_Solid):
    block_id = LAMP_ID
    device_kind = "lamp"

    def inputs(self, block: MinecraftBlock, coord: Coord) -> list[tuple[Coord, str]]:
        return [(offset(coord, d), READ_SIGNAL) for d in ("down", "up", "north", "south", "west", "east")]

    def on_input(self, sim: RedstoneSimulator, device: int, value: int) -> None:
        lit = sim.device_on(device)
        if value > 0 and not lit:
            sim.set_device_output(device, True)
            sim.record("lamp_changed", coord=sim.devices[device].coord, lit=True)
        elif value == 0 and lit and not sim.has_pending_tick(device):
            sim.schedule_tick(device, LAMP_OFF_DELAY_GT)

    def on_tick(self, sim: RedstoneSimulator, device: int) -> None:
        if sim.device_on(device) and sim.device_input(device) == 0:
            sim.set_device_output(device, False)
            sim.record("lamp_changed", coord=sim.devices[device].coord, lit=False)


AIR = _Air()
UNSUPPORTED = _Unsupported()

#: The supported block ids -> behaviour.  Anything else is a diagnostic.
BEHAVIORS: dict[str, BlockBehavior] = {
    b.block_id: b
    for b in (
        AIR,
        _Solid(),
        _Glass(),
        _Abstract(),
        _Wire(),
        _Repeater(),
        _Torch(),
        _WallTorch(),
        _Lever(),
        _RedstoneBlock(),
        _Lamp(),
    )
}

SUPPORTED_BLOCKS: tuple[str, ...] = tuple(sorted(BEHAVIORS))

#: What is deliberately not modelled (exported in every simulation report).
UNSUPPORTED_MECHANICS: tuple[str, ...] = (
    "comparators",
    "observers",
    "pistons and quasi-connectivity",
    "buttons and pressure plates",
    "repeater locking",
    "torch burnout (detected and reported, never simulated)",
    (
        "Java tick priorities and block-update order within one game tick "
        "(same-tick events run in scheduling order; RedC cells must not depend on it)"
    ),
)


def behavior_of(block: MinecraftBlock) -> BlockBehavior:
    return BEHAVIORS.get(block.id, UNSUPPORTED)


__all__ = [
    "BEHAVIORS",
    "BURNOUT_TOGGLES",
    "BURNOUT_WINDOW_GT",
    "READ_RAW",
    "READ_SIGNAL",
    "SUPPORTED_BLOCKS",
    "UNSUPPORTED_MECHANICS",
    "BlockBehavior",
    "Diagnostic",
    "WorldView",
    "behavior_of",
]
