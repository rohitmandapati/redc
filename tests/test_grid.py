import pytest

from redc import CompileError
from redc.physical import EMPTY, MAX_HEIGHT, CellKind, Grid


def test_place_and_read_back() -> None:
    grid = Grid()
    grid.place(3, 5, 7, owner=42)
    assert grid.owner_at(3, 5, 7) == 42
    assert grid.kind_at(3, 5, 7) == CellKind.COMPONENT
    assert not grid.is_free(3, 5, 7)
    # An untouched cell reads as free without allocating anything.
    assert grid.is_free(0, 0, 0)
    assert grid.owner_at(100, 0, 100) == EMPTY


def test_height_is_capped() -> None:
    grid = Grid()  # default height == 24
    grid.place(0, MAX_HEIGHT - 1, 0, owner=1)  # top cell is legal
    with pytest.raises(CompileError):
        grid.place(0, MAX_HEIGHT, 0, owner=1)
    with pytest.raises(CompileError):
        grid.place(0, -1, 0, owner=1)
    with pytest.raises(CompileError):
        Grid(height=MAX_HEIGHT + 1)


def test_horizontal_growth_and_negative_coords() -> None:
    grid = Grid()
    grid.place(0, 0, 0, owner=1)
    # Writes far away in both directions force the backing box to grow.
    grid.place(-50, 0, -50, owner=2)
    grid.place(200, 3, 175, owner=3)
    assert grid.owner_at(0, 0, 0) == 1
    assert grid.owner_at(-50, 0, -50) == 2
    assert grid.owner_at(200, 3, 175) == 3


def test_horizontal_caps_limit_span_but_default_is_unbounded() -> None:
    # Default grid: caps are humongous, so far-flung writes are fine.
    Grid().place(1_000_000, 0, 1_000_000, owner=1)

    # A width cap of 3 allows a 3-cell span (0..2) but not a 4th column.
    grid = Grid(max_x=3, max_z=3)
    grid.place(0, 0, 0, owner=1)
    grid.place(2, 0, 2, owner=1)  # span is now exactly 3 in both axes
    with pytest.raises(CompileError):
        grid.place(3, 0, 0, owner=1)  # would make width 4
    with pytest.raises(CompileError):
        grid.place(0, 0, 3, owner=1)  # would make depth 4

    # The span is what's capped, not the coordinate: a shifted 3-wide run is ok.
    shifted = Grid(max_x=3)
    shifted.place(-1, 0, 0, owner=1)
    shifted.place(1, 0, 0, owner=1)  # span -1..1 == 3
    with pytest.raises(CompileError):
        shifted.place(2, 0, 0, owner=1)

    with pytest.raises(CompileError):
        Grid(max_x=0)


def test_double_placement_raises_but_rip_up_frees() -> None:
    grid = Grid()
    grid.place(1, 1, 1, owner=1)
    with pytest.raises(CompileError):
        grid.place(1, 1, 1, owner=2)
    grid.rip_up(1, 1, 1)
    assert grid.is_free(1, 1, 1)
    grid.place(1, 1, 1, owner=2)  # now reusable
    assert grid.owner_at(1, 1, 1) == 2


def test_occupied_cells_and_snapshot() -> None:
    grid = Grid()
    grid.place(0, 0, 0, owner=1, kind=CellKind.PIN)
    grid.reserve(-2, 4, 6, owner=1)

    cells = sorted(grid.occupied_cells())
    assert cells == [
        (-2, 4, 6, 1, CellKind.RESERVED),
        (0, 0, 0, 1, CellKind.PIN),
    ]

    snapshot = grid.to_dict()
    assert snapshot["height"] == MAX_HEIGHT
    assert len(snapshot["cells"]) == 2


# -- routing layer ---------------------------------------------------------


def test_neighbors_are_six_connected_and_in_bounds() -> None:
    grid = Grid()
    # Open space: all six axis-aligned neighbours.
    assert set(grid.neighbors(5, 5, 5)) == {
        (6, 5, 5),
        (4, 5, 5),
        (5, 6, 5),
        (5, 4, 5),
        (5, 5, 6),
        (5, 5, 4),
    }
    # The floor drops the -y neighbour; the ceiling drops the +y neighbour.
    assert (0, -1, 0) not in set(grid.neighbors(0, 0, 0))
    assert (0, MAX_HEIGHT, 0) not in set(grid.neighbors(0, MAX_HEIGHT - 1, 0))


def test_component_bodies_block_routing_but_wires_do_not() -> None:
    grid = Grid()
    grid.place(1, 0, 0, owner=7)  # a component body
    assert not grid.is_routable(1, 0, 0)
    assert (1, 0, 0) not in set(grid.neighbors(0, 0, 0))
    assert grid.routing_cost(1, 0, 0) == float("inf")
    # Free space and committed wires are routable.
    grid.place(2, 0, 0, owner=8, kind=CellKind.WIRE)
    assert grid.is_routable(2, 0, 0)


def test_routing_cost_grows_with_congestion() -> None:
    grid = Grid()
    base = grid.routing_cost(0, 0, 0)
    grid.claim(1, 0, 0, 0)  # one net: within capacity, no overuse penalty
    assert grid.routing_cost(0, 0, 0) == base
    grid.claim(2, 0, 0, 0)  # second net: overused -> more expensive
    assert grid.routing_cost(0, 0, 0) > base


def test_claim_allows_overuse_and_rip_up_net_frees_the_whole_route() -> None:
    grid = Grid()
    grid.claim(1, 0, 0, 0)
    grid.claim(1, 1, 0, 0)  # net 1 spans two cells
    grid.claim(2, 0, 0, 0)  # net 2 overuses the first cell (place() would raise)
    assert grid.occupancy_at(0, 0, 0) == 2
    assert grid.overused() == [(0, 0, 0)]
    assert grid.route_of(1) == frozenset({(0, 0, 0), (1, 0, 0)})

    grid.rip_up_net(1)
    assert grid.occupancy_at(0, 0, 0) == 1
    assert grid.occupancy_at(1, 0, 0) == 0
    assert grid.overused() == []
    assert grid.route_of(1) == frozenset()


def test_claim_refuses_blocked_cells() -> None:
    grid = Grid()
    grid.place(0, 0, 0, owner=1)  # component body
    with pytest.raises(CompileError):
        grid.claim(9, 0, 0, 0)


def test_history_accumulates_only_on_overused_cells() -> None:
    grid = Grid()
    grid.claim(1, 0, 0, 0)
    grid.claim(2, 0, 0, 0)  # overused
    before = grid.routing_cost(0, 0, 0)
    grid.add_history()  # one negotiation iteration
    assert grid.routing_cost(0, 0, 0) > before  # permanently pricier now
    assert grid.history_at(0, 0, 0) > 0.0

    # A cell that was never overused gains no history.
    grid.claim(3, 5, 5, 5)
    grid.add_history()
    assert grid.history_at(5, 5, 5) == 0.0


def test_reclaiming_a_cell_by_the_same_net_counts_once() -> None:
    # A fanout tree's branches share a trunk: that trunk holds ONE net.
    grid = Grid()
    for _ in range(3):
        grid.claim(7, 0, 0, 0)
    grid.claim(7, 1, 0, 0)
    assert grid.occupancy_at(0, 0, 0) == 1
    assert grid.overused() == []
    grid.rip_up_net(7)
    assert grid.occupancy_at(0, 0, 0) == 0  # released exactly once
    assert grid.occupancy_at(1, 0, 0) == 0
    assert grid.max_occupancy() == 0


def test_net_aware_cost_prices_the_sharing_that_would_result() -> None:
    grid = Grid()
    base = grid.routing_cost(0, 0, 0)
    grid.claim(1, 0, 0, 0)
    # For another net, using a cell net 1 holds would overuse it.
    assert grid.routing_cost(0, 0, 0, net=2) > base
    # For net 1 itself (another branch of its own tree) it is not shared.
    assert grid.routing_cost(0, 0, 0, net=1) == base
    # The net-less form keeps reporting the current state.
    assert grid.routing_cost(0, 0, 0) == base


def test_commit_routes_marks_wires_and_refuses_overuse() -> None:
    grid = Grid()
    grid.claim(1, 0, 0, 0)
    grid.claim(1, 1, 0, 0)
    grid.claim(2, 1, 0, 0)  # overused
    with pytest.raises(CompileError, match="overused"):
        grid.commit_routes()
    grid.rip_up_net(2)
    assert grid.commit_routes() == 2
    assert grid.kind_at(0, 0, 0) == CellKind.WIRE
    assert grid.owner_at(1, 0, 0) == 1
    assert grid.route_of(1) == frozenset({(0, 0, 0), (1, 0, 0)})
    kinds = {c["kind"] for c in grid.to_dict()["cells"]}
    assert kinds == {int(CellKind.WIRE)}


def test_congestion_stats_summarise_history() -> None:
    grid = Grid()
    assert grid.congestion_stats()["overused"] == 0
    grid.claim(1, 0, 0, 0)
    grid.claim(2, 0, 0, 0)
    grid.add_history(increment=2.0)
    stats = grid.congestion_stats()
    assert stats["overused"] == 1
    assert stats["max_occupancy"] == 2
    assert stats["history_cells"] == 1
    assert stats["history_max"] == 2.0


def test_growth_along_one_axis_does_not_inflate_the_other() -> None:
    # Regression: growing x used to double z as well (and anchor the spare room
    # on the wrong side), so a design spreading east one column at a time grew
    # the backing arrays exponentially.  Inspects private capacity on purpose.
    grid = Grid(height=4)
    for i in range(200):
        grid.place(i * 6, 1, 0, owner=i)
    assert grid._nz == 1
    assert grid._nx <= 2 * (199 * 6 + 1)
    for i in range(1, 100):
        grid.place(0, 1, -i * 5, owner=1000 + i)  # now grow toward -z
    assert grid._nz <= 2 * (99 * 5 + 1)
    assert grid.owner_at(0, 1, 0) == 0 and grid.owner_at(1194, 1, 0) == 199
    assert grid.owner_at(0, 1, -495) == 1099
