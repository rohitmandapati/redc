"""The sparse block grid of the ``physical-primitive`` backend and the redstone
model helpers it is built on.

* placement: :meth:`BlockGrid.check_placement` explains every rejection
  (overlap, keep-out entry, pins / approaches covered, pins that would touch,
  height limit) and :meth:`BlockGrid.place` records cells and tracks bounds;
* routing claims are reference counted per net, the adjacency index counts
  each net's signal blocks around every block, and ``release(keep=pins)``
  restores every table;
* :meth:`BlockGrid.conflicts` flags each cross-net violation kind of the
  conservative redstone model -- and nothing that model allows;
* history and congestion statistics;
* the :mod:`redc.physical_primitive.redstone` helpers.

Grids are built by hand -- cells as explicit block lists, pins as
:class:`PinSite`, claims exactly as the router commits a branch -- so every
case is one block-level situation.  The last tests check the grid a real
place-and-route run leaves behind.
"""

from __future__ import annotations

import itertools
from collections import Counter
from itertools import pairwise
from typing import Any

import pytest

from redc import CompileError, compile_source
from redc.physical_primitive import (
    PRIMITIVE_TECHNOLOGY,
    BitTerminal,
    PadInterfacePolicy,
    PrimitivePnRConfig,
    place_and_route_graph,
)
from redc.physical_primitive.geometry import (
    IDENTITY,
    Bounds,
    Coord,
    Direction,
    Orientation,
    below,
)
from redc.physical_primitive.grid import (
    HISTORY_INCREMENT,
    STATIC_BODY,
    STATIC_KEEPOUT,
    STATIC_PIN,
    BlockGrid,
    Conflict,
    NetClaims,
    PinSite,
)
from redc.physical_primitive.redstone import (
    HORIZONTAL_STEPS,
    MOVES,
    SIGNAL_NEIGHBORHOOD,
    clearance_of,
    horizontal_direction,
    is_move,
    neighborhood,
    support_of,
)
from redc.physical_primitive.technology import PlacedCell

E, W = Direction.EAST, Direction.WEST
MAX_Y = 9
NOT = "not_gate_torch_placeholder"
AND = "and_gate_2x3_placeholder"

# -- helpers ---------------------------------------------------------------------

#: Instance 0, spelled out block by block (the placeholder inverter's shape): a
#: base row, a two-block body, keep-out around and above the body and beside
#: the pins; pin ``a`` (0, 1, 0) faces west (approach (-1, 1, 0)), pin ``y``
#: (3, 1, 0) faces east (approach (4, 1, 0)).
BODY0 = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (1, 1, 0), (2, 1, 0)]
KEEPOUT0 = [
    (1, 2, 0), (2, 2, 0), (1, 1, 1), (1, 1, -1), (2, 1, 1), (2, 1, -1),
    (0, 1, 1), (0, 1, -1), (3, 1, 1), (3, 1, -1),
]  # fmt: skip


def site(
    cell: Coord,
    *,
    instance: int = 0,
    pin: str = "a",
    direction: str = "in",
    facing: Direction = W,
    strength: int = 1,
    net: int | None = None,
) -> PinSite:
    return PinSite(cell, instance, pin, direction, facing, strength, net)


PINS0 = [site((0, 1, 0), net=1), site((3, 1, 0), pin="y", direction="out", facing=E, strength=15, net=2)]


def grid_with_cell() -> BlockGrid:
    grid = BlockGrid(max_y=MAX_Y)
    assert grid.check_placement(BODY0, KEEPOUT0, [((0, 1, 0), (-1, 1, 0)), ((3, 1, 0), (4, 1, 0))]) is None
    grid.place(0, BODY0, KEEPOUT0, PINS0)
    return grid


def grid_with_pin(cell: Coord = (0, 1, 0), facing: Direction = W) -> BlockGrid:
    """Only one pin endpoint (no body / keep-out): isolates the pin rules."""
    grid = BlockGrid(max_y=MAX_Y)
    grid.place(0, [], [], [site(cell, facing=facing)])
    return grid


def library_cell(name: str, origin: Coord, orientation: Orientation = IDENTITY) -> PlacedCell:
    return PRIMITIVE_TECHNOLOGY[name].oriented(orientation).translate(origin)


def check(grid: BlockGrid, cell: PlacedCell) -> str | None:
    return grid.check_placement(cell.occupied, cell.keepout, [(p.position, p.approach) for p in cell.pins.values()])


def put(grid: BlockGrid, instance: int, cell: PlacedCell, nets: dict[str, int] | None = None) -> None:
    nets = nets or {}
    sites = [
        PinSite(p.position, instance, p.name, p.direction, p.facing, p.strength, nets.get(p.name))
        for p in cell.pins.values()
    ]
    grid.place(instance, cell.occupied, cell.keepout, sites)


def claim(grid: BlockGrid, layer: str, net: int, cell: Coord) -> None:
    getattr(grid, f"claim_{layer}")(net, cell)


def claim_route(grid: BlockGrid, net: int, path: tuple[Coord, ...], pins: set[Coord]) -> None:
    """Claim ``path`` exactly as the router commits a branch: every new non-pin
    block is a signal with its support, every staircase needs its clearance."""
    for a, b in pairwise(path):
        if b not in pins:
            grid.claim_signal(net, b)
            grid.claim_support(net, support_of(b))
        clear = clearance_of(a, b)
        if clear is not None:
            grid.claim_clearance(net, clear)


def expected_adjacency(grid: BlockGrid) -> dict[Coord, dict[int, int]]:
    """The adjacency index recomputed from scratch from the signal table."""
    index: dict[Coord, dict[int, int]] = {}
    for cell, nets in grid.signal.items():
        for net in nets:
            for nb in neighborhood(cell):
                users = index.setdefault(nb, {})
                users[net] = users.get(net, 0) + 1
    return index


def expected_used(grid: BlockGrid) -> dict[Coord, int]:
    """Total claims per block (every kind, every net), recomputed from scratch."""
    used: dict[Coord, int] = {}
    for table in (grid.signal, grid.support, grid.clearance):
        for cell, users in table.items():
            used[cell] = used.get(cell, 0) + sum(users.values())
    return used


def expected_static(grid: BlockGrid) -> dict[Coord, int]:
    """The STATIC_* bits of every block, recomputed from the placement tables."""
    static: dict[Coord, int] = {}
    for table, bit in ((grid.body, STATIC_BODY), (grid.keepout, STATIC_KEEPOUT), (grid.pins, STATIC_PIN)):
        for cell in table:
            static[cell] = static.get(cell, 0) | bit
    return static


def assert_indexes_consistent(grid: BlockGrid) -> None:
    """Every incremental index equals its from-scratch recomputation."""
    assert grid.adjacent == expected_adjacency(grid)
    assert grid.used == expected_used(grid)
    assert grid.static == expected_static(grid)


def tables(grid: BlockGrid) -> dict[str, Any]:
    """A deep copy of every routing table (to compare states)."""
    return {
        name: {cell: dict(users) for cell, users in getattr(grid, name).items()}
        for name in ("signal", "support", "clearance", "adjacent")
    } | {
        "used": dict(grid.used),
        "claims": {net: (dict(c.signals), dict(c.supports), dict(c.clearances)) for net, c in grid.claims.items()},
    }


# -- construction and placement ------------------------------------------------------


def test_grid_needs_room_for_a_support_and_a_signal() -> None:
    with pytest.raises(CompileError, match="max_y >= 2"):
        BlockGrid(max_y=1)
    grid = BlockGrid(max_y=2)
    assert grid.in_height((0, 0, 0)) and grid.in_height((0, 2, 0))
    assert not grid.in_height((0, 3, 0)) and not grid.in_height((0, -1, 0))
    assert grid.placement_bounds is None and grid.conflicts() == []


PLACEMENT_REJECTIONS = [
    ("overlap", [(9, 0, 9), (2, 1, 0)], [], [], "overlaps instance 0 at [2, 1, 0]"),
    ("keepout_entry", [(1, 2, 0)], [], [], "enters the keep-out of instance 0 at [1, 2, 0]"),
    ("body_on_pin", [(0, 1, 0)], [], [], "covers pin 'a' of instance 0 at [0, 1, 0]"),
    ("body_on_input_approach", [(-1, 1, 0)], [], [], "covers pin 'a' of instance 0 at [-1, 1, 0]"),
    ("body_on_output_approach", [(4, 1, 0)], [], [], "covers pin 'y' of instance 0 at [4, 1, 0]"),
    ("keepout_covers_body", [], [(2, 1, 0)], [], "keep-out covers instance 0 at [2, 1, 0]"),
    ("keepout_covers_pin", [], [(3, 1, 0)], [], "keep-out covers pin 'y' of instance 0 at [3, 1, 0]"),
    ("keepout_covers_approach", [], [(-1, 1, 0)], [], "keep-out covers pin 'a' of instance 0 at [-1, 1, 0]"),
    ("body_above_height_limit", [(5, MAX_Y + 1, 5)], [], [], "block [5, 10, 5] is outside the height limit 0..9"),
    ("body_below_ground", [(5, -1, 5)], [], [], "block [5, -1, 5] is outside the height limit 0..9"),
    ("pin_above_height_limit", [], [], [((5, MAX_Y + 1, 5), (6, MAX_Y + 1, 5))], "is outside the height limit"),
    ("pin_in_a_body", [], [], [((1, 1, 0), (1, 1, -1))], "pin block [1, 1, 0] is occupied or kept out"),
    ("pin_in_a_keepout", [], [], [((1, 2, 0), (1, 2, -1))], "pin block [1, 2, 0] is occupied or kept out"),
    ("approach_in_a_keepout", [], [], [((2, 1, 2), (2, 1, 1))], "pin block [2, 1, 1] is occupied or kept out"),
    ("pin_on_a_pin", [], [], [((0, 1, 0), (-1, 1, 0))], "pin [0, 1, 0] collides with another pin"),
    ("pin_on_an_approach", [], [], [((-1, 1, 0), (-2, 1, 0))], "pin [-1, 1, 0] sits on another pin's approach"),
    (
        "facing_pins_share_an_approach",
        [],
        [],
        [((-2, 1, 0), (-1, 1, 0))],
        "pin approach [-1, 1, 0] is another pin's endpoint or approach",
    ),
    (
        "pin_touches_a_pin_one_down",
        [],
        [],
        [((3, 2, 1), (3, 2, 0))],
        "pin block [3, 2, 1] would touch pin 'y' of instance 0",
    ),
    (
        "approach_touches_an_approach",
        [],
        [],
        [((-1, 1, 2), (-1, 1, 1))],
        "pin block [-1, 1, 1] would touch pin 'a' of instance 0",
    ),
]


@pytest.mark.parametrize(
    ("occupied", "keepout", "pins", "reason"),
    [case[1:] for case in PLACEMENT_REJECTIONS],
    ids=[case[0] for case in PLACEMENT_REJECTIONS],
)
def test_check_placement_explains_every_rejection(occupied, keepout, pins, reason: str) -> None:
    grid = grid_with_cell()
    before = (dict(grid.body), dict(grid.keepout), dict(grid.pins), dict(grid.approaches), grid.placement_bounds)
    answer = grid.check_placement(occupied, keepout, pins)
    assert answer is not None and reason in answer, answer
    # Checking never writes anything.
    assert (grid.body, grid.keepout, grid.pins, grid.approaches, grid.placement_bounds) == before


@pytest.mark.parametrize(
    ("occupied", "keepout", "pins"),
    [
        ([(10, 0, 10), (10, 1, 10)], [(11, 1, 10)], []),  # far away
        ([], [(1, 2, 0), (1, 1, 1)], []),  # keep-outs of different instances may overlap
        ([], [], [((0, 1, 2), (-1, 1, 2))]),  # a pin two blocks from another pin
        ([], [], [((7, 1, 0), (6, 1, 0))]),  # an input approach two blocks from an output approach
    ],
    ids=["far_away", "overlapping_keepouts", "pins_two_apart", "approaches_two_apart"],
)
def test_check_placement_accepts_legal_neighbours(occupied, keepout, pins) -> None:
    assert grid_with_cell().check_placement(occupied, keepout, pins) is None


@pytest.mark.parametrize(
    ("endpoint", "approach", "reason"),
    [
        ((0, 1, 1), (-1, 1, 1), "pin block [0, 1, 1] would touch pin 'a'"),  # level side neighbour
        ((1, 2, 0), (2, 2, 0), "pin block [1, 2, 0] would touch pin 'a'"),  # one up, diagonally
        ((1, 0, 0), (2, 0, 0), "pin block [1, 0, 0] would touch pin 'a'"),  # one down, diagonally
        ((0, 1, 2), (0, 1, 1), "pin block [0, 1, 1] would touch pin 'a'"),  # its approach is the neighbour
        ((0, 1, 1), (0, 1, 0), "pin approach [0, 1, 0] is another pin's endpoint or approach"),
    ],
    ids=["level", "one_up", "one_down", "approach_beside_pin", "approach_on_pin"],
)
def test_pins_may_not_touch_in_the_signal_neighbourhood(endpoint: Coord, approach: Coord, reason: str) -> None:
    answer = grid_with_pin().check_placement([], [], [(endpoint, approach)])
    assert answer is not None and reason in answer, answer
    # An in-plane diagonal, or two blocks straight above, is no contact.
    assert grid_with_pin().check_placement([], [], [((1, 1, 1), (2, 1, 1))]) is None
    assert grid_with_pin().check_placement([], [], [((0, 3, 0), (0, 3, 1))]) is None


def test_library_cells_pack_side_by_side() -> None:
    grid = BlockGrid(max_y=MAX_Y)
    first = library_cell(NOT, (0, 0, 0))
    assert check(grid, first) is None
    put(grid, 0, first)
    # Along z: two blocks apart the keep-outs overlap (allowed), one block apart
    # the new body would sit in the first cell's keep-out.
    assert check(grid, library_cell(NOT, (0, 0, 2))) is None
    assert check(grid, library_cell(NOT, (0, 0, 1))).startswith("enters the keep-out of instance 0")
    # Along x (signal flow): the next cell's input approach needs one free block
    # between it and this cell's output approach (4, 1, 0).
    assert check(grid, library_cell(NOT, (4, 0, 0))) == "pin [4, 1, 0] sits on another pin's approach"
    assert "pin approach [4, 1, 0] is another pin's" in check(grid, library_cell(NOT, (5, 0, 0)))
    assert "pin block [5, 1, 0] would touch pin 'y'" in check(grid, library_cell(NOT, (6, 0, 0)))
    assert check(grid, library_cell(NOT, (7, 0, 0))) is None
    # A rotated cell is checked by its rotated blocks.
    assert check(grid, library_cell(AND, (12, 0, 0), Orientation(1))) is None


def test_place_records_bodies_keepouts_pins_and_approaches() -> None:
    grid = BlockGrid(max_y=MAX_Y)
    a, b = library_cell(NOT, (0, 0, 0)), library_cell(NOT, (0, 0, 2))
    put(grid, 0, a, {"a": 3, "y": 4})
    assert check(grid, b) is None
    put(grid, 1, b)
    assert grid.body == {**{c: 0 for c in a.occupied}, **{c: 1 for c in b.occupied}}
    shared = a.keepout & b.keepout
    assert (1, 1, 1) in shared
    assert set(grid.keepout) == a.keepout | b.keepout
    assert all(grid.keepout[c] == 0 for c in shared)  # the first instance keeps ownership
    pin_a = grid.pins[(0, 1, 0)]
    assert (pin_a.instance, pin_a.pin, pin_a.direction, pin_a.facing, pin_a.net) == (0, "a", "in", W, 3)
    assert pin_a.ref() == {"instance": 0, "pin": "a"}
    assert grid.approaches[(-1, 1, 0)] is pin_a
    assert grid.approaches[(4, 1, 0)] is grid.pins[(3, 1, 0)]
    assert grid.pins[(3, 1, 0)].net == 4 and grid.pins[(0, 1, 2)].net is None  # unconnected pin
    assert len(grid.pins) == len(grid.approaches) == 4
    assert grid.is_blocked((1, 1, 0)) and grid.is_blocked((1, 2, 0))
    assert not grid.is_blocked((0, 1, 0)) and not grid.is_blocked((-1, 1, 0))
    with pytest.raises(CompileError, match=r"block \[1, 1, 0\] is already occupied"):
        grid.place(2, [(1, 1, 0)], [], [])
    assert list(grid.occupied_blocks()) == [(c, "body", owner) for c, owner in grid.body.items()]
    # The router's one-lookup index of what placement put where.
    assert grid.static[(1, 1, 0)] == STATIC_BODY and grid.static[(1, 1, 1)] == STATIC_KEEPOUT
    assert grid.static[(0, 1, 0)] == STATIC_PIN and (-1, 1, 0) not in grid.static  # approaches stay free
    assert grid.static == expected_static(grid) and grid.used == {}


def test_place_tracks_the_placement_bounds() -> None:
    grid = BlockGrid(max_y=MAX_Y)
    grid.place(5, [], [], [])
    assert grid.placement_bounds is None
    a = library_cell(NOT, (0, 0, 0))
    put(grid, 0, a)
    assert grid.placement_bounds == a.bounds == Bounds((0, 0, -1), (3, 2, 1))
    b = library_cell(AND, (10, 0, 6), Orientation(1))
    put(grid, 1, b)
    assert grid.placement_bounds == a.bounds.union(b.bounds)
    everything = [*a.occupied, *a.keepout, *b.occupied, *b.keepout]
    everything += [p.position for cell in (a, b) for p in cell.pins.values()]
    assert grid.placement_bounds == Bounds.of(everything)
    # Approach blocks are reserved, but they are routing room, not the placement.
    assert (-1, 1, 0) in grid.approaches and not grid.placement_bounds.contains((-1, 1, 0))
    # A lone pin still grows the bounds.
    grid.place(2, [], [], [site((40, 3, -20), instance=2)])
    assert grid.placement_bounds.hi[0] == 40 and grid.placement_bounds.lo[2] == -20


# -- routing claims ----------------------------------------------------------------------

DRIVER, SINK = (0, 1, 0), (4, 1, 0)
#: A staircase route between two pins: up onto (2, 2, 0), down onto (3, 1, 0).
PATH = (DRIVER, (1, 1, 0), (2, 2, 0), (3, 1, 0), SINK)


def test_claims_are_reference_counted() -> None:
    grid = BlockGrid(max_y=MAX_Y)
    grid.claim_signal(1, DRIVER)
    grid.claim_signal(1, DRIVER)
    assert grid.signal == {DRIVER: {1: 2}}
    assert grid.net_signals(1) == {DRIVER: 2} and grid.net_signals(2) == {}
    # The adjacency index counts DISTINCT signal blocks: a second claim adds nothing.
    assert grid.adjacent == {nb: {1: 1} for nb in neighborhood(DRIVER)}
    grid.claim_support(1, below(DRIVER))
    grid.claim_support(1, below(DRIVER))
    grid.claim_clearance(1, (0, 2, 0))
    grid.claim_clearance(2, (0, 2, 0))
    assert grid.support == {(0, 0, 0): {1: 2}}
    assert grid.clearance == {(0, 2, 0): {1: 1, 2: 1}}
    assert grid.claims[1] == NetClaims({DRIVER: 2}, {(0, 0, 0): 2}, {(0, 2, 0): 1})
    assert grid.claims[2] == NetClaims({}, {}, {(0, 2, 0): 1})
    assert list(grid.occupied_blocks()) == [(DRIVER, "signal", 1), ((0, 0, 0), "support", 1)]  # never air
    assert grid.used == {DRIVER: 2, (0, 0, 0): 2, (0, 2, 0): 2}  # every claim of every kind and net
    assert grid.static == {}  # routing never touches the placement index


def test_adjacency_index_counts_each_nets_signal_blocks() -> None:
    grid = BlockGrid(max_y=MAX_Y)
    for net, cell in [(1, (0, 1, 0)), (1, (2, 1, 0)), (2, (1, 1, 1)), (3, (1, 2, 0))]:
        grid.claim_signal(net, cell)
    # (1, 1, 0) is level with both net-1 blocks and with net 2's block; the
    # net-3 block straight above it is NOT a neighbour.
    assert grid.adjacent[(1, 1, 0)] == {1: 2, 2: 1}
    # (1, 2, 0) is one up, diagonally, from all three of them.
    assert grid.adjacent[(1, 2, 0)] == {1: 2, 2: 1}
    assert grid.adjacent[(0, 1, 0)] == {3: 1}  # one down from net 3's block
    assert grid.adjacent[(1, 1, 1)] == {3: 1}  # net-1 blocks are in-plane diagonals: no contact
    assert (5, 1, 5) not in grid.adjacent
    assert grid.adjacent == expected_adjacency(grid)


def test_release_keeps_pins_and_restores_every_table() -> None:
    grid = BlockGrid(max_y=MAX_Y)
    pins = {DRIVER, SINK}
    for pin in pins:
        grid.claim_signal(7, pin)
    grid.claim_signal(7, DRIVER)  # claimed twice: still ONE reservation is kept
    # Net 8 shares one staircase clearance (legal: air is air) and routes elsewhere.
    grid.claim_signal(8, (10, 1, 10))
    grid.claim_support(8, (10, 0, 10))
    grid.claim_clearance(8, (1, 2, 0))
    claim_route(grid, 7, PATH, pins)
    assert grid.net_signals(7) == {DRIVER: 2, SINK: 1, (1, 1, 0): 1, (2, 2, 0): 1, (3, 1, 0): 1}
    assert grid.support == {(1, 0, 0): {7: 1}, (2, 1, 0): {7: 1}, (3, 0, 0): {7: 1}, (10, 0, 10): {8: 1}}
    assert grid.clearance == {(1, 2, 0): {8: 1, 7: 1}, (3, 2, 0): {7: 1}}
    assert_indexes_consistent(grid)
    assert grid.conflicts() == []

    grid.release(7, keep=pins)
    assert grid.signal == {DRIVER: {7: 1}, SINK: {7: 1}, (10, 1, 10): {8: 1}}
    assert grid.support == {(10, 0, 10): {8: 1}}
    assert grid.clearance == {(1, 2, 0): {8: 1}}
    assert grid.used == {DRIVER: 1, SINK: 1, (10, 1, 10): 1, (10, 0, 10): 1, (1, 2, 0): 1}
    assert_indexes_consistent(grid)
    assert grid.claims[7] == NetClaims(signals={DRIVER: 1, SINK: 1})
    after_rip_up = tables(grid)

    for _ in range(3):  # route / rip up again and again: no reference leaks
        claim_route(grid, 7, PATH, pins)
        grid.release(7, keep=pins)
        assert tables(grid) == after_rip_up

    grid.release(7)  # without keep the net disappears entirely
    assert 7 not in grid.claims
    for table in (grid.signal, grid.support, grid.clearance, grid.adjacent):
        assert all(7 not in users for users in table.values())
    assert_indexes_consistent(grid)
    grid.release(7)  # releasing an unknown net is a no-op
    grid.release(99, keep=pins)
    assert grid.signal == {(10, 1, 10): {8: 1}}


def test_release_leaves_exactly_the_kept_pins() -> None:
    grid = BlockGrid(max_y=MAX_Y)
    pins = {DRIVER, SINK}
    for pin in pins:
        grid.claim_signal(7, pin)
    claim_route(grid, 7, PATH, pins)
    grid.release(7, keep=pins)
    assert grid.signal == {DRIVER: {7: 1}, SINK: {7: 1}}
    assert grid.support == {} and grid.clearance == {}
    assert grid.adjacent == {nb: {7: 1} for pin in pins for nb in neighborhood(pin)}
    assert grid.used == {DRIVER: 1, SINK: 1}
    assert grid.claims == {7: NetClaims(signals={DRIVER: 1, SINK: 1})}


def test_release_of_every_net_empties_the_grid() -> None:
    grid = BlockGrid(max_y=MAX_Y)
    claim_route(grid, 1, PATH, {DRIVER, SINK})
    claim_route(grid, 2, ((0, 1, 5), (0, 2, 6), (0, 3, 7), (0, 2, 8)), set())
    for net in (1, 2):
        grid.release(net)
    assert (grid.signal, grid.support, grid.clearance, grid.adjacent, grid.used, grid.claims) == ({},) * 6


# -- conflicts -----------------------------------------------------------------------------

CONFLICTS = [
    ("shared_signal", [("signal", 1, (0, 1, 0)), ("signal", 2, (0, 1, 0))], "shared_signal", [(0, 1, 0)]),
    ("shared_support", [("support", 1, (0, 0, 0)), ("support", 2, (0, 0, 0))], "shared_support", [(0, 0, 0)]),
    (
        "signal_on_support",
        [("signal", 1, (0, 1, 0)), ("support", 2, (0, 1, 0))],
        "signal_on_support",
        [(0, 1, 0)],
    ),
    (
        "clearance_holds_a_signal",
        [("clearance", 1, (0, 2, 0)), ("signal", 2, (0, 2, 0))],
        "clearance_blocked",
        [(0, 2, 0)],
    ),
    (
        "clearance_holds_a_support",
        [("clearance", 1, (0, 2, 0)), ("support", 2, (0, 2, 0))],
        "clearance_blocked",
        [(0, 2, 0)],
    ),
    (
        "adjacent_level",
        [("signal", 1, (0, 1, 0)), ("signal", 2, (1, 1, 0))],
        "adjacent_signals",
        [(0, 1, 0), (1, 1, 0)],
    ),
    (
        "adjacent_diagonal_one_up",
        [("signal", 1, (0, 1, 0)), ("signal", 2, (1, 2, 0))],
        "adjacent_signals",
        [(0, 1, 0), (1, 2, 0)],
    ),
    (
        "adjacent_diagonal_one_down",
        [("signal", 1, (0, 2, 0)), ("signal", 2, (0, 1, 1))],
        "adjacent_signals",
        [(0, 1, 1), (0, 2, 0)],
    ),
]


@pytest.mark.parametrize(
    ("claims", "kind", "cells"), [c[1:] for c in CONFLICTS], ids=[c[0] for c in CONFLICTS]
)
def test_conflicts_detect_each_kind(claims, kind: str, cells: list[Coord]) -> None:
    grid = BlockGrid(max_y=MAX_Y)
    for layer, net, cell in claims:
        claim(grid, layer, net, cell)
    assert grid.conflicts() == [Conflict(kind, tuple(cells), (1, 2))]
    assert grid.conflicts()[0].to_dict() == {"kind": kind, "cells": [list(c) for c in cells], "nets": [1, 2]}


LEGAL_CLAIMS = [
    (
        "same_net_adjacency",
        [("signal", 1, (0, 1, 0)), ("signal", 1, (1, 1, 0)), ("signal", 1, (2, 2, 0)),
         ("support", 1, (1, 0, 0)), ("support", 1, (2, 1, 0)), ("clearance", 1, (1, 2, 0))],
    ),
    (
        "two_nets_share_a_clearance",
        [("clearance", 1, (0, 2, 0)), ("clearance", 2, (0, 2, 0)), ("signal", 1, (0, 1, 0)),
         ("signal", 2, (5, 1, 5))],
    ),
    ("in_plane_diagonal", [("signal", 1, (0, 1, 0)), ("signal", 2, (1, 1, 1))]),
    ("two_blocks_apart", [("signal", 1, (0, 1, 0)), ("signal", 2, (2, 1, 0))]),
    ("two_up_diagonally", [("signal", 1, (0, 1, 0)), ("signal", 2, (1, 3, 0))]),
    (
        "crossing_two_blocks_above",
        [("signal", 1, (0, 1, 0)), ("support", 1, (0, 0, 0)), ("signal", 2, (0, 3, 0)),
         ("support", 2, (0, 2, 0))],
    ),
    ("supports_side_by_side", [("support", 1, (0, 0, 0)), ("support", 2, (1, 0, 0))]),
]  # fmt: skip


@pytest.mark.parametrize("claims", [c[1] for c in LEGAL_CLAIMS], ids=[c[0] for c in LEGAL_CLAIMS])
def test_conflicts_ignore_what_the_model_allows(claims) -> None:
    grid = BlockGrid(max_y=MAX_Y)
    for layer, net, cell in claims:
        claim(grid, layer, net, cell)
    assert grid.conflicts() == []


def test_conflicts_are_complete_and_deterministically_ordered() -> None:
    grid = BlockGrid(max_y=MAX_Y)
    for layer, net, cell in [
        ("support", 5, (9, 0, 9)), ("support", 4, (9, 0, 9)),
        ("signal", 2, (5, 1, 5)), ("signal", 1, (5, 1, 5)),
        ("signal", 3, (1, 1, 0)), ("signal", 1, (0, 1, 0)),
        ("clearance", 6, (9, 0, 9)),
    ]:  # fmt: skip
        claim(grid, layer, net, cell)
    expected = [
        Conflict("adjacent_signals", ((0, 1, 0), (1, 1, 0)), (1, 3)),
        Conflict("shared_signal", ((5, 1, 5),), (1, 2)),
        Conflict("shared_support", ((9, 0, 9),), (4, 5)),
        Conflict("clearance_blocked", ((9, 0, 9),), (4, 5, 6)),
    ]
    assert grid.conflicts() == expected
    assert grid.conflicts() == expected  # reading conflicts changes nothing
    # A pair is reported once, whichever block is visited first.
    assert sum(1 for c in grid.conflicts() if c.kind == "adjacent_signals") == 1


# -- history and congestion --------------------------------------------------------------


def test_add_history_counts_distinct_blocks() -> None:
    grid = BlockGrid(max_y=MAX_Y)
    a, b = (0, 1, 0), (1, 1, 0)
    assert grid.add_history([a, a, b]) == 2
    assert grid.history == {a: HISTORY_INCREMENT, b: HISTORY_INCREMENT}
    assert grid.add_history([a], 2.5) == 1
    assert grid.history[a] == HISTORY_INCREMENT + 2.5
    assert grid.add_history([]) == 0
    assert grid.add_history(iter([b, b])) == 1 and grid.history[b] == 2 * HISTORY_INCREMENT


def test_congestion_stats() -> None:
    grid = BlockGrid(max_y=MAX_Y)
    assert grid.congestion_stats() == {
        "conflicts": 0, "conflict_cells": 0, "history_cells": 0, "history_total": 0.0, "history_max": 0.0,
    }  # fmt: skip
    for net, cell in [(1, (0, 1, 0)), (2, (0, 1, 0)), (1, (5, 1, 0)), (3, (6, 2, 0))]:
        grid.claim_signal(net, cell)
    grid.add_history([(0, 1, 0), (5, 1, 0)])
    grid.add_history([(0, 1, 0)], 3.0)
    stats = grid.congestion_stats()
    # shared_signal at (0,1,0) + adjacent_signals (5,1,0)/(6,2,0): three distinct cells.
    assert stats == {
        "conflicts": 2,
        "conflict_cells": 3,
        "history_cells": 2,
        "history_total": 2 * HISTORY_INCREMENT + 3.0,
        "history_max": HISTORY_INCREMENT + 3.0,
    }
    assert type(stats["history_total"]) is float and type(stats["history_max"]) is float


# -- the redstone model helpers ----------------------------------------------------------


def test_support_is_directly_below() -> None:
    assert support_of((3, 5, -2)) == (3, 4, -2) == below((3, 5, -2))


@pytest.mark.parametrize(
    ("a", "b", "clear"),
    [
        ((0, 1, 0), (1, 1, 0), None),  # level step: no clearance
        ((0, 1, 0), (1, 2, 0), (0, 2, 0)),  # climb: above the lower (starting) block
        ((0, 2, 0), (1, 1, 0), (1, 2, 0)),  # descent: above the lower (ending) block
        ((5, 3, 5), (5, 4, 4), (5, 4, 5)),
        ((-2, 7, 3), (-2, 6, 4), (-2, 7, 4)),
    ],
    ids=["level", "up", "down", "up_north", "down_south"],
)
def test_clearance_is_above_the_lower_block(a: Coord, b: Coord, clear: Coord | None) -> None:
    assert clearance_of(a, b) == clear
    assert clearance_of(b, a) == clear  # the same air for either direction
    if clear is not None:
        lower = min(a, b, key=lambda c: c[1])
        assert clear == (lower[0], lower[1] + 1, lower[2])
        assert clear[1] == max(a[1], b[1]) and clear not in (a, b)


@pytest.mark.parametrize(
    ("delta", "legal"),
    [
        ((1, 0, 0), True), ((0, 0, -1), True), ((-1, 1, 0), True), ((0, -1, 1), True),
        ((0, 1, 0), False), ((0, -1, 0), False),  # no straight-up / straight-down move
        ((1, 0, 1), False), ((2, 0, 0), False), ((1, 2, 0), False), ((0, 0, 0), False), ((1, -1, 1), False),
    ],
)  # fmt: skip
def test_is_move(delta: Coord, legal: bool) -> None:
    origin = (3, 4, 5)
    target = (origin[0] + delta[0], origin[1] + delta[1], origin[2] + delta[2])
    assert is_move(origin, target) is legal
    assert is_move(target, origin) is legal


def test_neighbourhood_is_exactly_twelve_blocks() -> None:
    cell = (4, 2, -3)
    hood = neighborhood(cell)
    assert len(hood) == len(set(hood)) == 12
    assert cell not in hood
    assert (4, 3, -3) not in hood and (4, 1, -3) not in hood  # nothing straight above / below
    cube = {(cell[0] + dx, cell[1] + dy, cell[2] + dz) for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3)}
    assert set(hood) == {c for c in cube if is_move(cell, c)}
    assert Counter(c[1] - cell[1] for c in hood) == {0: 4, 1: 4, -1: 4}  # level, one up, one down
    assert all(cell in neighborhood(nb) for nb in hood)  # the relation is symmetric


def test_moves_are_the_neighbourhood_and_never_vertical() -> None:
    assert MOVES == SIGNAL_NEIGHBORHOOD
    assert MOVES[:4] == HORIZONTAL_STEPS  # flat steps first, then climbs, then descents
    assert [m[1] for m in MOVES] == [0] * 4 + [1] * 4 + [-1] * 4
    assert all(abs(dx) + abs(dz) == 1 for dx, _dy, dz in MOVES)
    assert (0, 1, 0) not in MOVES and (0, -1, 0) not in MOVES
    assert all((-dx, -dy, -dz) in SIGNAL_NEIGHBORHOOD for dx, dy, dz in SIGNAL_NEIGHBORHOOD)
    assert horizontal_direction((0, 1, 0), (1, 2, 0)) == (1, 0, 0)
    assert horizontal_direction((0, 2, 0), (0, 1, -1)) == (0, 0, -1)


# -- the grid a real place-and-route run leaves behind -----------------------------------


@pytest.mark.parametrize(
    "source",
    ["uint2 main(bool a, bool b) { return (uint2)a + (uint2)b; }", "uint4 main(uint4 a, uint4 b) { return a + b; }"],
    ids=["half_adder", "uint4_add"],
)
def test_routed_grid_matches_its_route_trees(source: str) -> None:
    _netlist, mapped, result = place_and_route_graph(
        compile_source(source), PrimitivePnRConfig(), interface=PadInterfacePolicy()
    )
    assert result.success, result.failure
    grid = result.grid
    assert grid is not None
    assert grid.conflicts() == []
    assert_indexes_consistent(grid)  # no incremental index drifted through all the rip-ups
    assert set(grid.claims) == set(result.routes)
    for net_id, tree in result.routes.items():
        claims = grid.claims[net_id]
        assert set(claims.signals) == set(tree.cells)
        assert set(claims.supports) == set(tree.supports)
        assert set(claims.clearances) == set(tree.clearances)
    assert set(grid.signal) == {c for tree in result.routes.values() for c in tree.cells}
    assert set(grid.support) == {c for tree in result.routes.values() for c in tree.supports}
    # Every pin of every placed cell is reserved for its net, with its approach.
    terminals = mapped.net_of_terminal()
    body: set[Coord] = set()
    for inst in mapped.instances:
        body |= inst.placed.occupied
        for pin in inst.placed.pins.values():
            pin_site = grid.pins[pin.position]
            assert (pin_site.instance, pin_site.pin) == (inst.id, pin.name)
            assert pin_site.net == terminals.get(BitTerminal(inst.id, pin.name))
            assert grid.approaches[pin.approach] is pin_site
    assert set(grid.body) == body
    assert not set(grid.signal) & (set(grid.body) | set(grid.keepout))
    assert not set(grid.support) & (set(grid.body) | set(grid.keepout) | set(grid.pins))
