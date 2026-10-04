"""Minecraft blocks as plain data: a namespaced id plus block-state properties.

A :class:`MinecraftBlock` says WHAT occupies one block position, exactly as a
structure file or ``/setblock`` would (``minecraft:repeater[delay=2,facing=west]``).
It carries no behaviour: how a block behaves electrically is decided by
:mod:`redc.minecraft.behaviors`, keyed by the id.

Block-state values follow Minecraft's own conventions, which are NOT always
the intuitive ones.  In particular ``minecraft:repeater[facing=...]`` names
the side its INPUT faces: a repeater with ``facing=west`` takes its input from
the west and outputs to the east.  :func:`repeater` therefore takes the OUTPUT
direction (what a router thinks in) and stores the blockstate facing.

Coordinates: ``(x, y, z)`` with x east, y up, z south (Minecraft's axes); one
coordinate is one block.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

Coord = tuple[int, int, int]
StateValue = str | int | bool

#: Unit vector of every direction name Minecraft uses in block states.
DIRECTION_VECTORS: dict[str, Coord] = {
    "north": (0, 0, -1),
    "south": (0, 0, 1),
    "east": (1, 0, 0),
    "west": (-1, 0, 0),
    "up": (0, 1, 0),
    "down": (0, -1, 0),
}
#: Horizontal directions, clockwise seen from above (x east, z south).
HORIZONTAL_DIRECTIONS: tuple[str, ...] = ("east", "south", "west", "north")
#: All six directions in the fixed order used wherever neighbours are scanned.
ALL_DIRECTIONS: tuple[str, ...] = ("down", "up", "north", "south", "west", "east")
OPPOSITE: dict[str, str] = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
    "up": "down",
    "down": "up",
}
_BY_VECTOR = {v: k for k, v in DIRECTION_VECTORS.items()}


def offset(coord: Coord, direction: str) -> Coord:
    dx, dy, dz = DIRECTION_VECTORS[direction]
    return (coord[0] + dx, coord[1] + dy, coord[2] + dz)


def direction_between(a: Coord, b: Coord) -> str | None:
    """The direction name of the unit step ``a -> b`` (``None`` if not adjacent)."""
    return _BY_VECTOR.get((b[0] - a[0], b[1] - a[1], b[2] - a[2]))


def rotate_direction(direction: str, quarter_turns: int) -> str:
    """``direction`` turned clockwise (seen from above) ``quarter_turns`` times;
    ``up`` / ``down`` are unchanged."""
    if direction not in HORIZONTAL_DIRECTIONS:
        return direction
    return HORIZONTAL_DIRECTIONS[(HORIZONTAL_DIRECTIONS.index(direction) + quarter_turns) % 4]


def rotate_coord(coord: Coord, quarter_turns: int) -> Coord:
    """Rotate a local coordinate clockwise about +y (the same rotation as
    :class:`redc.physical_primitive.geometry.Orientation`)."""
    x, y, z = coord
    for _ in range(quarter_turns % 4):
        x, z = -z, x
    return (x, y, z)


#: Block-state keys holding a horizontal direction (rotated with the block).
_DIRECTIONAL_KEYS = ("facing",)


@dataclass(frozen=True, slots=True)
class MinecraftBlock:
    """One block: namespaced id + block-state properties (sorted, hashable)."""

    id: str
    state: tuple[tuple[str, StateValue], ...] = ()

    def __post_init__(self) -> None:
        if ":" not in self.id:
            raise ValueError(f"block id {self.id!r} must be namespaced (e.g. 'minecraft:stone')")
        keys = [k for k, _ in self.state]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError(f"block {self.id}: state keys must be unique and sorted, got {keys}")

    @classmethod
    def of(cls, block_id: str, **state: StateValue) -> MinecraftBlock:
        return cls(block_id, tuple(sorted(state.items())))

    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self.state:
            if k == key:
                return v
        return default

    @property
    def properties(self) -> dict[str, StateValue]:
        return dict(self.state)

    def with_state(self, **changes: StateValue) -> MinecraftBlock:
        merged = dict(self.state)
        merged.update(changes)
        return MinecraftBlock(self.id, tuple(sorted(merged.items())))

    def rotated(self, quarter_turns: int) -> MinecraftBlock:
        """The same block turned clockwise about +y (horizontal facings rotate)."""
        if not quarter_turns % 4:
            return self
        changes = {
            key: rotate_direction(str(value), quarter_turns)
            for key, value in self.state
            if key in _DIRECTIONAL_KEYS
        }
        return self.with_state(**changes) if changes else self

    def __str__(self) -> str:
        if not self.state:
            return self.id
        props = ",".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in self.state)
        return f"{self.id}[{props}]"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "state": dict(self.state)} if self.state else {"id": self.id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MinecraftBlock:
        return cls.of(str(data["id"]), **dict(data.get("state") or {}))


# -- the blocks RedC places ------------------------------------------------------

AIR_ID = "minecraft:air"
STONE_ID = "minecraft:stone"
GLASS_ID = "minecraft:glass"
WIRE_ID = "minecraft:redstone_wire"
REPEATER_ID = "minecraft:repeater"
TORCH_ID = "minecraft:redstone_torch"
WALL_TORCH_ID = "minecraft:redstone_wall_torch"
REDSTONE_BLOCK_ID = "minecraft:redstone_block"
LEVER_ID = "minecraft:lever"
LAMP_ID = "minecraft:redstone_lamp"
#: A RedC-only marker: one voxel of an ABSTRACT (not materialized) component.
ABSTRACT_ID = "redc:abstract_block"


def air() -> MinecraftBlock:
    return MinecraftBlock(AIR_ID)


def stone() -> MinecraftBlock:
    return MinecraftBlock(STONE_ID)


def glass() -> MinecraftBlock:
    return MinecraftBlock(GLASS_ID)


def redstone_wire() -> MinecraftBlock:
    return MinecraftBlock(WIRE_ID)


def repeater(output: str, delay: int = 1) -> MinecraftBlock:
    """A repeater whose signal LEAVES toward ``output`` (blockstate facing is
    the opposite, input side) with ``delay`` redstone ticks (1..4)."""
    if output not in HORIZONTAL_DIRECTIONS:
        raise ValueError(f"repeater output must be horizontal, got {output!r}")
    if not 1 <= delay <= 4:
        raise ValueError(f"repeater delay must be 1..4 redstone ticks, got {delay}")
    return MinecraftBlock.of(REPEATER_ID, facing=OPPOSITE[output], delay=delay)


def repeater_output(block: MinecraftBlock) -> str:
    """The direction a repeater's signal leaves toward."""
    return OPPOSITE[str(block.get("facing"))]


def redstone_torch() -> MinecraftBlock:
    """A standing torch (attached to the block below it)."""
    return MinecraftBlock(TORCH_ID)


def wall_torch(facing: str) -> MinecraftBlock:
    """A wall torch pointing ``facing`` (attached to the block behind it)."""
    if facing not in HORIZONTAL_DIRECTIONS:
        raise ValueError(f"wall torch facing must be horizontal, got {facing!r}")
    return MinecraftBlock.of(WALL_TORCH_ID, facing=facing)


def redstone_block() -> MinecraftBlock:
    return MinecraftBlock(REDSTONE_BLOCK_ID)


def lever(face: str = "floor", facing: str = "north") -> MinecraftBlock:
    """A lever on the ``floor`` / ``ceiling`` / ``wall`` (wall: attached to the
    block behind ``facing``).  Its ``powered`` state is driven by a port."""
    if face not in ("floor", "wall", "ceiling") or facing not in HORIZONTAL_DIRECTIONS:
        raise ValueError(f"bad lever face/facing {face!r}/{facing!r}")
    return MinecraftBlock.of(LEVER_ID, face=face, facing=facing)


def redstone_lamp() -> MinecraftBlock:
    return MinecraftBlock(LAMP_ID)


def abstract_block() -> MinecraftBlock:
    return MinecraftBlock(ABSTRACT_ID)


__all__ = [
    "ABSTRACT_ID",
    "AIR_ID",
    "ALL_DIRECTIONS",
    "DIRECTION_VECTORS",
    "GLASS_ID",
    "HORIZONTAL_DIRECTIONS",
    "LAMP_ID",
    "LEVER_ID",
    "OPPOSITE",
    "REDSTONE_BLOCK_ID",
    "REPEATER_ID",
    "STONE_ID",
    "TORCH_ID",
    "WALL_TORCH_ID",
    "WIRE_ID",
    "Coord",
    "MinecraftBlock",
    "abstract_block",
    "air",
    "direction_between",
    "glass",
    "lever",
    "offset",
    "redstone_block",
    "redstone_lamp",
    "redstone_torch",
    "redstone_wire",
    "repeater",
    "repeater_output",
    "rotate_coord",
    "rotate_direction",
    "stone",
    "wall_torch",
]
