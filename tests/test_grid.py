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
