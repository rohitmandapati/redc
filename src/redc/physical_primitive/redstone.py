"""The explicit (conservative) Minecraft redstone model routing is checked against.

A wire is a ROUTE, never a component: a one-bit net is realized as a tree of
*signal blocks* -- redstone dust, or a repeater inserted by electrical
legalization -- each resting on a *support* block directly below it.  A
support must be a full, opaque redstone CONDUCTOR (:data:`SUPPORT_MATERIAL`):
on a non-conductor such as glass a staircase only carries signal UP, never
down, and the router descends freely.
Placement, the router, the legalizer and the independent verifier all use the
rules below, so "legal" means the same thing in every phase.

Facts of Java Edition redstone the model relies on
--------------------------------------------------
* Dust needs a solid block underneath; dust cannot sit on dust.
* Dust connects to dust in the four horizontal neighbours on the same level,
  and diagonally one block up/down (a "staircase") UNLESS an opaque block sits
  directly above the lower dust (that block cuts the connection).
* Signal strength starts at 15 next to a source and drops by 1 per dust block;
  dust at strength 0 is unpowered.  A repeater re-drives its front to 15, takes
  input only from its back and outputs only to its front (it is a diode), and
  delays by 1..4 redstone ticks.
* Dust weakly powers the block below it and the blocks it points into; weak
  power never reaches other dust, but it does trigger mechanisms (torches
  attached to that block, repeaters facing away from it, ...).

Conservative v1 rules (``N`` = a signal block of net ``n``)
-----------------------------------------------------------
1. **Signal neighbourhood.**  The 12 blocks that could ever connect to ``N``
   are :data:`SIGNAL_NEIGHBORHOOD`: 4 horizontal + 8 diagonal one-up/one-down.
   No signal block of ANOTHER net may be in it (that would short two nets).
   Within ONE net, the signal blocks in each other's neighbourhood must be
   exactly the tree's parent/child pairs -- the realized electrical graph IS
   the routing tree, so repeater direction and signal strength are exact and a
   net can never loop back through its own repeater (latch).
2. **Support.**  ``support_of(N)`` (directly below) is a solid block owned by
   the route, or -- only for a pin's endpoint block -- the cell's own base.  A
   support is never another net's support, never a signal or pin block, never
   inside a component or a keep-out.
3. **Vertical transitions.**  A one-up/one-down step between two signal blocks
   is legal only if :func:`clearance_of` (the block directly above the LOWER
   one) stays AIR (any non-conductor would do; air is the safe choice): it may
   not hold a support, signal, pin or component voxel.  There is no
   straight-up move: dust cannot climb a vertical column in mid-air.
4. **Crossings.**  Because of rules 1-3, two nets cross only with at least two
   blocks of vertical separation (the upper route's support sits directly above
   the lower dust, which is legal: an opaque block above dust is harmless).
5. **Repeaters** sit only on straight, level segments: parent at the back,
   exactly one child at the front, same y; never on a pin block.
6. **Components** declare keep-out voxels where no route signal or support
   block may go (a clearance may: it stays air), so routes cannot weakly power
   a cell's internals; pin endpoint blocks are reserved for the pin's own net
   and are entered/left only along the pin's facing.

What is deliberately NOT modelled (documented limitations): quasi-connectivity,
comparators, block update order, tick-accurate timing races, and the internal
behaviour of the PLACEHOLDER gate geometries -- the gate cells are honest
footprints, not verified in-game circuits.
"""

from __future__ import annotations

from enum import Enum

from .geometry import Coord

#: Strength a source drives into adjacent dust, and the repeater refresh level.
MAX_SIGNAL_STRENGTH = 15

#: What route supports and clearances must be made of.  The design file states
#: it, so a materializer cannot choose, e.g., glass supports (non-conductors
#: break every descending staircase).
SUPPORT_MATERIAL = "minecraft:stone"
SUPPORT_REQUIREMENT = "full opaque redstone conductor"
CLEARANCE_MATERIAL = "minecraft:air"

#: Default repeater delay in redstone ticks (one tick = two game ticks).
REPEATER_DELAY_TICKS = 1

#: Lowest y a signal block may occupy (its support needs y - 1 >= 0).
MIN_SIGNAL_Y = 1

#: The four horizontal unit steps.
HORIZONTAL_STEPS: tuple[Coord, ...] = ((1, 0, 0), (0, 0, 1), (-1, 0, 0), (0, 0, -1))

#: Every block that could electrically connect to a signal block: the 4
#: horizontal neighbours and the 8 diagonal one-up / one-down neighbours.
#: Straight up/down is absent: a block directly above dust is its neighbour's
#: support or air, never dust (dust cannot rest on dust).
SIGNAL_NEIGHBORHOOD: tuple[Coord, ...] = tuple(
    (dx, dy, dz) for dy in (0, 1, -1) for (dx, _, dz) in HORIZONTAL_STEPS
)

#: The router's moves: flat steps first, then climbs, then descents.  Every
#: move lands in the signal neighbourhood (each step is an electrical edge).
MOVES: tuple[Coord, ...] = SIGNAL_NEIGHBORHOOD


class ElementKind(Enum):
    """What a route places at one block."""

    DUST = "dust"
    REPEATER = "repeater"
    SUPPORT = "support"  # a solid block under a signal block
    CLEARANCE = "clearance"  # must stay air (vertical-transition headroom)


def support_of(cell: Coord) -> Coord:
    """The block a signal block at ``cell`` rests on."""
    return (cell[0], cell[1] - 1, cell[2])


def clearance_of(a: Coord, b: Coord) -> Coord | None:
    """For a step between signal blocks ``a`` and ``b``: the block that must
    stay air so the diagonal connection exists -- the one directly above the
    LOWER block.  ``None`` for a level step."""
    if a[1] == b[1]:
        return None
    lower = a if a[1] < b[1] else b
    return (lower[0], lower[1] + 1, lower[2])


def is_move(a: Coord, b: Coord) -> bool:
    """Whether ``b`` is one legal routing step from ``a``."""
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    return abs(dx) + abs(dz) == 1 and abs(dy) <= 1


def neighborhood(cell: Coord) -> list[Coord]:
    """The 12 blocks that could electrically connect to a signal block at ``cell``."""
    x, y, z = cell
    return [(x + dx, y + dy, z + dz) for dx, dy, dz in SIGNAL_NEIGHBORHOOD]


def horizontal_direction(a: Coord, b: Coord) -> Coord:
    """The horizontal unit step from ``a`` toward ``b`` (one routing move)."""
    return (b[0] - a[0], 0, b[2] - a[2])


__all__ = [
    "CLEARANCE_MATERIAL",
    "HORIZONTAL_STEPS",
    "MAX_SIGNAL_STRENGTH",
    "MIN_SIGNAL_Y",
    "MOVES",
    "REPEATER_DELAY_TICKS",
    "SIGNAL_NEIGHBORHOOD",
    "SUPPORT_MATERIAL",
    "SUPPORT_REQUIREMENT",
    "ElementKind",
    "clearance_of",
    "horizontal_direction",
    "is_move",
    "neighborhood",
    "support_of",
]
