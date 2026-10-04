"""Adversarial review of the ``physical-primitive`` redstone legality model.

The router, the legalizer and :func:`verify_design` all judge designs against
the SAME conservative model (:mod:`redc.physical_primitive.redstone`), so a
hole in the model would be invisible to all three.  This file attacks the
model from the outside:

* :class:`JavaWorld` re-implements the Java Edition update rules the model
  claims to rely on, from the block world alone -- no neighbourhoods, no
  clearances, no trees.  ``RedStoneWireBlock.calculateTargetStrength`` decides
  which neighbouring wire a wire reads (level always; one block up only if the
  block beside it is a redstone *conductor* and the block above the lower wire
  is not; one block down only if the block beside the upper wire is not a
  conductor), ``getConnectingSide`` plus the single-connection line extension
  decides which blocks a wire weakly powers, and ``DiodeBlock`` makes a
  repeater read only the block behind it and drive only the block in front.
  Cell voxels and route supports are conductors; keep-outs and clearances are
  air.  Power is the least fixed point from all-off (what a world settles to
  when the drivers switch on).
* Real place-and-route output under several geometries must agree with it.
* Hand-placed designs (:func:`hand_design`) build exact adversarial
  geometries: crossings over repeaters, cut and uncut staircases, a latch, a
  sink entered from the side, routes over cells and over foreign pins.  The
  model must flag everything Java would get wrong; where it also flags
  something Java gets right, the test documents the rule as conservative.
* Mutated realized routes (``dataclasses.replace``) must be REPORTED by
  :func:`verify_design` -- an accepted illegal mutation is a verifier hole,
  and so is a crash.

Everything is deterministic (the placement fuzz uses a fixed seed).
"""

from __future__ import annotations

import dataclasses
import functools
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations

import pytest

from redc import compile_source
from redc.physical_primitive import (
    DEFAULT_INTERFACE_POLICY,
    PRIMITIVE_TECHNOLOGY,
    BitTerminal,
    PadInterfacePolicy,
    PrimitivePhysicalNetlist,
    PrimitivePnRConfig,
    PrimitiveTechnologyLibrary,
    PrimitiveTraceRecorder,
    map_primitives_to_minecraft,
    place_and_route_graph,
    synthesize_to_primitives,
)
from redc.physical_primitive.geometry import IDENTITY, ORIENTATIONS, Coord, Direction
from redc.physical_primitive.grid import BlockGrid, PinSite
from redc.physical_primitive.pnr import (
    LegalizationFailure,
    RealizedRoute,
    RouteBranch,
    RouteRequest,
    RouteTree,
    legalize_route,
    route_requests,
    verify_design,
)
from redc.physical_primitive.redstone import (
    MAX_SIGNAL_STRENGTH,
    SIGNAL_NEIGHBORHOOD,
    ElementKind,
    clearance_of,
)

PADS = PadInterfacePolicy()
UP, DOWN = (0, 1, 0), (0, -1, 0)
SIDES: tuple[Coord, ...] = ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1))
NORTH_SOUTH = frozenset({(0, 0, 1), (0, 0, -1)})
EAST_WEST = frozenset({(1, 0, 0), (-1, 0, 0)})


def add(a: Coord, b: Coord) -> Coord:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def neg(v: Coord) -> Coord:
    return (-v[0], -v[1], -v[2])


# -- an independent Java Edition redstone world --------------------------------------------


@dataclass(frozen=True)
class Block:
    """``kind`` is ``conductor`` (a full opaque block), ``dust`` or ``repeater``
    (``facing`` = its OUTPUT direction).  ``owner`` is ``("cell", id)`` or
    ``("net", id)``."""

    kind: str
    owner: tuple[str, int]
    facing: Coord | None = None


@dataclass
class JavaWorld:
    """The built design as Java sees it: every non-air block, plus any block
    two artifacts both tried to occupy."""

    blocks: dict[Coord, Block]
    collisions: list[Coord] = field(default_factory=list)

    @classmethod
    def build(cls, mapped: PrimitivePhysicalNetlist, realized: Mapping[int, RealizedRoute]) -> JavaWorld:
        world = cls({})
        for inst in mapped.instances:
            for cell in inst.placed.occupied:
                world.put(cell, Block("conductor", ("cell", inst.id)))
        for net, route in realized.items():
            for cell in route.supports:
                world.put(cell, Block("conductor", ("net", net)))
            for e in route.elements:
                if e.kind is ElementKind.REPEATER:
                    assert e.facing is not None
                    world.put(e.coord, Block("repeater", ("net", net), e.facing.vector))
                elif e.kind is ElementKind.DUST:
                    world.put(e.coord, Block("dust", ("net", net)))
                elif e.kind is ElementKind.SUPPORT:  # a "signal" realized as a solid block
                    world.put(e.coord, Block("conductor", ("net", net)))
                # ElementKind.CLEARANCE: the block stays air
        return world

    def put(self, cell: Coord, block: Block) -> None:
        if cell in self.blocks:
            self.collisions.append(cell)
        self.blocks[cell] = block

    def kind(self, cell: Coord) -> str:
        block = self.blocks.get(cell)
        return "air" if block is None else block.kind

    def conductor(self, cell: Coord) -> bool:
        return self.kind(cell) == "conductor"

    def wire(self, cell: Coord) -> bool:
        return self.kind(cell) == "dust"

    def wire_reads(self, p: Coord) -> list[Coord]:
        """The wires the wire at ``p`` takes power from (calculateTargetStrength)."""
        out = []
        roof = self.conductor(add(p, UP))
        for d in SIDES:
            n = add(p, d)
            if self.wire(n):
                out.append(n)
            if self.conductor(n):
                if not roof and self.wire(add(n, UP)):
                    out.append(add(n, UP))
            elif self.wire(add(n, DOWN)):
                out.append(add(n, DOWN))
        return out

    def connects_to(self, cell: Coord, d: Coord) -> bool:
        block = self.blocks.get(cell)
        if block is None:
            return False
        if block.kind == "dust":
            return True
        return block.kind == "repeater" and d in (block.facing, neg(block.facing))

    def shape(self, p: Coord) -> set[Coord]:
        """Connected sides of the wire at ``p`` (getConnectingSide + line extension)."""
        can_climb = not self.conductor(add(p, UP))
        sides = set()
        for d in SIDES:
            n = add(p, d)
            climbs = can_climb and self.conductor(n) and self.wire(add(n, UP))
            drops = not self.conductor(n) and self.wire(add(n, DOWN))
            if climbs or drops or self.connects_to(n, d):
                sides.add(d)
        if not sides & NORTH_SOUTH:
            sides |= EAST_WEST
        if not sides & EAST_WEST:
            sides |= NORTH_SOUTH
        return sides

    def weakly_powers(self, p: Coord) -> list[Coord]:
        """Blocks a powered wire at ``p`` powers: the one below and every side it points into."""
        return [add(p, DOWN), *(add(p, d) for d in sorted(self.shape(p)))]

    def steady_power(self, sources: Mapping[Coord, int]) -> tuple[dict[Coord, int], dict[Coord, bool]]:
        wires = [c for c, b in self.blocks.items() if b.kind == "dust"]
        repeaters = [c for c, b in self.blocks.items() if b.kind == "repeater"]
        reads = {p: self.wire_reads(p) for p in wires}
        front = {r: add(r, self.blocks[r].facing) for r in repeaters}
        back = {r: add(r, neg(self.blocks[r].facing)) for r in repeaters}
        fronting: dict[Coord, list[Coord]] = {}
        for r in repeaters:
            fronting.setdefault(front[r], []).append(r)
        weak_into: dict[Coord, list[Coord]] = {}
        for p in wires:
            for b in self.weakly_powers(p):
                if self.conductor(b):
                    weak_into.setdefault(b, []).append(p)
        power = {p: 0 for p in wires}
        on = {r: False for r in repeaters}

        def strong(cell: Coord) -> bool:
            return self.conductor(cell) and any(on[r] for r in fronting.get(cell, ()))

        def repeater_input(r: Coord) -> int:
            b = back[r]
            kind = self.kind(b)
            if kind == "dust":
                return power[b]
            if kind == "repeater":
                return MAX_SIGNAL_STRENGTH if on[b] and front[b] == r else 0
            if kind == "conductor":
                return MAX_SIGNAL_STRENGTH if strong(b) else max((power[p] for p in weak_into.get(b, ())), default=0)
            return 0

        changed = True
        while changed:
            changed = False
            for r in repeaters:
                if not on[r] and repeater_input(r) >= 1:
                    on[r] = changed = True
            for p in wires:
                level = sources.get(p, 0)
                for q in reads[p]:
                    level = max(level, power[q] - 1)
                if any(on[r] for r in fronting.get(p, ())) or any(strong(add(p, d)) for d in (*SIDES, UP, DOWN)):
                    level = MAX_SIGNAL_STRENGTH
                if level > power[p]:
                    power[p] = level
                    changed = True
        return power, on

    def latched(self) -> list[Coord]:
        """Repeaters whose output reaches their own input (a self-holding latch)."""
        succ: dict[Coord, set[Coord]] = {}
        for p, b in self.blocks.items():
            if b.kind == "dust":
                for q in self.wire_reads(p):
                    succ.setdefault(q, set()).add(p)
            elif b.kind == "repeater":
                succ.setdefault(add(p, neg(b.facing)), set()).add(p)
                succ.setdefault(p, set()).add(add(p, b.facing))
        found = []
        for r, b in sorted(self.blocks.items()):
            if b.kind != "repeater":
                continue
            seen, todo = set(), [add(r, b.facing)]
            while todo:
                cell = todo.pop()
                if cell == r:
                    found.append(r)
                    break
                if cell not in seen:
                    seen.add(cell)
                    todo.extend(succ.get(cell, ()))
        return found


def java_problems(
    mapped: PrimitivePhysicalNetlist, requests: Mapping[int, RouteRequest], realized: Mapping[int, RealizedRoute]
) -> list[tuple]:
    """Every way the realized routes misbehave in Java Edition (empty = sound)."""
    world = JavaWorld.build(mapped, realized)
    problems: list[tuple] = [("collision", c) for c in world.collisions]
    net_of, parent = {}, {}
    for net, route in realized.items():
        for e in route.elements:
            net_of[e.coord], parent[e.coord] = net, e.parent
    for cell, block in sorted(world.blocks.items()):
        if block.kind in ("dust", "repeater") and not world.conductor(add(cell, DOWN)):
            problems.append(("floating", cell))
    # electrical edges must be exactly the routed tree edges
    for p in sorted(c for c, b in world.blocks.items() if b.kind == "dust"):
        for q in world.wire_reads(p):
            if net_of[p] != net_of[q]:
                problems.append(("short", p, q))
            elif parent.get(p) != q and parent.get(q) != p:
                problems.append(("extra_connection", p, q))
    for net, route in sorted(realized.items()):
        for e in route.elements:
            if e.parent is None:
                continue
            up, here = world.blocks.get(e.parent), world.blocks.get(e.coord)
            if up is None or here is None:
                problems.append(("broken_connection", e.parent, e.coord))
            elif up.kind == here.kind == "dust":
                if e.parent not in world.wire_reads(e.coord) or e.coord not in world.wire_reads(e.parent):
                    problems.append(("broken_connection", e.parent, e.coord))
            elif here.kind == "repeater" and add(e.coord, neg(here.facing)) != e.parent:
                problems.append(("repeater_misplaced", e.coord))
            elif up.kind == "repeater" and add(e.parent, up.facing) != e.coord:
                problems.append(("repeater_misplaced", e.parent))
            elif "conductor" in (up.kind, here.kind):
                problems.append(("broken_connection", e.parent, e.coord))
    for cell, block in sorted(world.blocks.items()):
        if block.kind == "repeater":
            for end in (add(cell, block.facing), add(cell, neg(block.facing))):
                other = world.blocks.get(end)
                if other is not None and other.kind != "conductor" and other.owner != block.owner:
                    problems.append(("short", cell, end))
    problems += [("latch", r) for r in world.latched()]
    # steady-state power: recorded strengths, sinks, and the sink dust pointing INTO its cell
    power, on = world.steady_power({r.driver.cell: r.driver.strength for r in requests.values()})
    for net, route in sorted(realized.items()):
        for e in route.elements:
            if e.kind is ElementKind.DUST:
                actual = power.get(e.coord, 0)
            elif e.kind is ElementKind.REPEATER:
                behind = add(e.coord, neg(e.facing.vector))
                actual = power.get(behind, MAX_SIGNAL_STRENGTH if on.get(behind) else 0)
            else:
                continue
            if actual != e.strength:
                problems.append(("strength", e.coord, e.strength, actual))
        request = requests.get(net)
        if request is None or request.driver.strength == 0:
            continue
        for sink in request.sinks:
            if power.get(sink.cell, 0) < sink.strength:
                problems.append(("weak_sink", sink.cell, power.get(sink.cell, 0)))
            inside = add(sink.cell, neg(sink.facing.vector))
            if world.wire(sink.cell) and inside not in world.weakly_powers(sink.cell):
                problems.append(("sink_not_driven", sink.cell))
    # no wire weakly powers a cell voxel, except a pin's dust powering its own cell
    pin_owner = {p.position: inst.id for inst in mapped.instances for p in inst.placed.pins.values()}
    for p in sorted(c for c, b in world.blocks.items() if b.kind == "dust"):
        for b in world.weakly_powers(p):
            block = world.blocks.get(b)
            if block is not None and block.owner[0] == "cell" and pin_owner.get(p) != block.owner[1]:
                problems.append(("powers_cell", p, b))
    return problems


def problem_kinds(problems: list[tuple]) -> set[str]:
    return {p[0] for p in problems}


def foreign_weak_power(mapped: PrimitivePhysicalNetlist, realized: Mapping[int, RealizedRoute]) -> list[tuple]:
    """``(wire, block)`` pairs where a route wire weakly powers a block that is
    neither air, its own net's, nor (for pin dust) its own cell's.  Harmless
    by itself in Java, but the only way a route could reach a foreign
    support (and through it a repeater or mechanism reading that block)."""
    world = JavaWorld.build(mapped, realized)
    pin_owner = {p.position: inst.id for inst in mapped.instances for p in inst.placed.pins.values()}
    found = []
    for p, block in sorted(world.blocks.items()):
        if block.kind != "dust":
            continue
        for b in world.weakly_powers(p):
            other = world.blocks.get(b)
            if other is None or other.owner == block.owner:
                continue
            if other.owner[0] == "cell" and pin_owner.get(p) == other.owner[1]:
                continue
            found.append((p, b))
    return found


# -- 1. the model's geometry against Java's wire rules --------------------------------------


def pair_world(a: Coord, b: Coord, *, conductors: tuple[Coord, ...] = (), glass: tuple[Coord, ...] = ()) -> JavaWorld:
    """Two lone wires on stone supports, plus extra ``conductors``.  A support
    listed in ``glass`` is glass: dust stands on it, but it is no redstone
    conductor, so for every wire read it behaves exactly like air."""
    blocks = {add(c, DOWN): Block("conductor", ("net", 0)) for c in (a, b) if add(c, DOWN) not in glass}
    blocks.update({c: Block("conductor", ("net", 0)) for c in conductors})
    blocks.update({a: Block("dust", ("net", 1)), b: Block("dust", ("net", 2))})
    return JavaWorld(blocks)


OFFSETS = [
    (dx, dy, dz)
    for dx in range(-2, 3)
    for dy in range(-2, 3)
    for dz in range(-2, 3)
    if (dx, dz) != (0, 0) or abs(dy) == 2  # a wire never sits on a wire
]


@pytest.mark.parametrize("offset", OFFSETS, ids=str)
def test_signal_neighbourhood_is_exactly_where_java_wires_connect(offset: Coord) -> None:
    """Rule 1's 12-block neighbourhood is complete and tight: two lone wires
    (staircase headroom left as air) connect in Java iff one is in the
    other's neighbourhood."""
    a = (0, 3, 0)
    b = add(a, offset)
    world = pair_world(a, b)
    connected = b in world.wire_reads(a) or a in world.wire_reads(b)
    assert connected == (offset in SIGNAL_NEIGHBORHOOD)


@pytest.mark.parametrize("step", [s for s in SIGNAL_NEIGHBORHOOD if s[1]], ids=str)
def test_clearance_is_the_block_java_checks_for_a_staircase(step: Coord) -> None:
    """A conductor at :func:`clearance_of` (above the LOWER wire) cuts the
    staircase both ways; a conductor above the UPPER wire is harmless."""
    a = (0, 3, 0)
    b = add(a, step)
    lower, upper = (a, b) if a[1] < b[1] else (b, a)
    assert clearance_of(a, b) == clearance_of(b, a) == add(lower, UP)
    cut = pair_world(a, b, conductors=(clearance_of(a, b),))
    assert b not in cut.wire_reads(a) and a not in cut.wire_reads(b)
    roofed = pair_world(a, b, conductors=(add(upper, UP),))
    assert b in roofed.wire_reads(a) and a in roofed.wire_reads(b)


def test_a_staircase_on_a_non_conductive_support_only_carries_signal_upward() -> None:
    """Why rule 2's supports must be redstone CONDUCTORS (stone), not just any
    block dust can stand on: on glass / glowstone / an upside-down slab the
    lower wire no longer reads the upper one (the up-read needs a conductor
    beside the lower wire), so every DESCENDING step of a route dies."""
    lower, upper = (0, 3, 0), (1, 4, 0)
    glass = pair_world(lower, upper, glass=(add(upper, DOWN),))
    assert lower in glass.wire_reads(upper)  # signal still climbs ...
    assert upper not in glass.wire_reads(lower)  # ... but cannot come back down
    stone = pair_world(lower, upper)
    assert upper in stone.wire_reads(lower) and lower in stone.wire_reads(upper)


# -- 2. real place-and-route output is sound in Java ----------------------------------------

DESIGNS = {
    "not": ("bool main(bool a) { return !a; }", PADS),
    "or": ("bool main(bool a, bool b) { return a || b; }", PADS),  # drives strength 13
    "half_adder": ("uint2 main(bool a, bool b) { return (uint2)a + (uint2)b; }", PADS),  # constant-0 nets
    "full_adder": ("uint2 main(bool a, bool b, bool c) { return (uint2)a + (uint2)b + (uint2)c; }", PADS),
    "uint4_add": ("uint4 main(uint4 a, uint4 b) { return a + b; }", PADS),
    "mux": ("uint4 main(bool s, uint4 a, uint4 b) { return s ? a : b; }", PADS),
    # clock / reset fan-out trees, register bits and constant-1 sources
    "counter": ("uint2 main(uint2 n) { uint2 x = 0; for (uint2 i = 0; i < n; i++) { x = x + 1; } return x; }", PADS),
    # a 5-block-tall display with eight input pins two blocks apart
    "display": ("uint8 main(bool start, uint8 a) { return start ? a : 0; }", DEFAULT_INTERFACE_POLICY),
}
GEOMETRIES = {
    "default": {},
    "flat": {"max_y": 3},  # crossings squeezed into y = 1 and y = 3
    "wide": {"channel_width": 20},  # repeaters everywhere, OR outputs included
}
SMALL = ("not", "or", "half_adder", "full_adder", "uint4_add", "mux")
JAVA_CASES = [(n, g) for n in SMALL for g in sorted(GEOMETRIES)] + [
    ("counter", "default"),
    ("display", "default"),
    ("display", "flat"),
]


@functools.cache
def bare_library() -> PrimitiveTechnologyLibrary:
    """Every library cell with its keep-out removed (legal: keep-outs are each cell's choice)."""
    cells = []
    for cell in PRIMITIVE_TECHNOLOGY.values():
        bare = dataclasses.replace(cell, keepout=frozenset())
        assert bare._oriented is not cell._oriented  # never share the rotation cache
        cells.append(bare)
    return PrimitiveTechnologyLibrary("bare", cells)


@functools.cache
def pnr(name: str, geometry: str = "default", *, bare: bool = False):
    source, interface = DESIGNS[name]
    config = PrimitivePnRConfig(**GEOMETRIES[geometry])
    library = bare_library() if bare else PRIMITIVE_TECHNOLOGY
    _netlist, mapped, result = place_and_route_graph(
        compile_source(source), config, interface=interface, library=library, trace=PrimitiveTraceRecorder("none")
    )
    assert result.success, result.failure
    return mapped, result


@pytest.mark.parametrize(("name", "geometry"), JAVA_CASES, ids=[f"{n}-{g}" for n, g in JAVA_CASES])
def test_routed_designs_behave_in_java_exactly_as_recorded(name: str, geometry: str) -> None:
    mapped, result = pnr(name, geometry)
    assert java_problems(mapped, result.routed.requests, result.realized) == []
    # no route wire weakly powers another net's support (or anything foreign)
    assert foreign_weak_power(mapped, result.realized) == []


@pytest.mark.parametrize("name", ["half_adder", "uint4_add"])
def test_route_dust_powers_only_its_own_net_even_without_keepouts(name: str) -> None:
    """Every non-pin route wire has a parent and a child, so its Java shape
    points only at its own net: even with keep-out-free cells no route wire
    weakly powers a cell voxel.  Keep-outs therefore guard the other
    direction -- cell internals (torches, power-source bodies) powering
    adjacent route dust, mechanisms beside route supports -- which the
    placeholder bodies cannot exhibit."""
    mapped, result = pnr(name, bare=True)
    assert all(not inst.cell.keepout for inst in mapped.instances)
    assert java_problems(mapped, result.routed.requests, result.realized) == []
    assert foreign_weak_power(mapped, result.realized) == []


def test_the_java_check_sees_repeaters_or_drive_13_and_constant_nets() -> None:
    """The geometries above really exercise what they are meant to."""
    _mapped, result = pnr("uint4_add", "wide")
    drives = {r.driver.strength for r in result.routed.requests.values()}
    assert {13, 15} <= drives
    assert sum(len(r.repeaters) for r in result.realized.values()) > 20
    _mapped, result = pnr("half_adder")
    assert any(not r.powered for r in result.realized.values())
    _mapped, result = pnr("full_adder", "flat")
    crossings = [
        (c, below)
        for r in result.realized.values()
        for c in r.supports
        for below in [add(c, DOWN)]
        if any(below in o.tree.cell_set for o in result.realized.values() if o.net != r.net)
    ]
    assert crossings, "no route support sits directly above another net's signal"


# -- 3. hand-placed adversarial geometry ---------------------------------------------------

NOT2 = "uint2 main(uint2 a) { return ~a; }"
#: in[0] (0) -> NOT (2) -> out[0] (4) along z = 0; in[1] (1) at z = -6 -> NOT (3) -> out[1] (5) at z = 10.
NOT2_ORIGINS = {0: (0, 0, 0), 1: (0, 0, -6), 2: (24, 0, 0), 3: (24, 0, 10), 4: (34, 0, 0), 5: (34, 0, 10)}


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


def path(*points: Coord) -> tuple[Coord, ...]:
    """Blocks from waypoint to waypoint: each leg runs along one horizontal
    axis and climbs / descends one block per step until its height is reached."""
    out = [points[0]]
    for b in points[1:]:
        x, y, z = a = out[-1]
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        steps = abs(dx) + abs(dz)
        assert (dx == 0 or dz == 0) and abs(dy) <= steps, f"bad leg {a} -> {b}"
        for i in range(steps):
            x, z = x + _sign(dx), z + _sign(dz)
            y += _sign(dy) if i < abs(dy) else 0
            out.append((x, y, z))
    return tuple(out)


@dataclass
class Hand:
    mapped: PrimitivePhysicalNetlist
    requests: dict[int, RouteRequest]

    def realize(self, net: int, *paths: tuple[Coord, ...]) -> RealizedRoute:
        request = self.requests[net]
        sinks = {s.cell: s for s in request.sinks}
        branches = tuple(RouteBranch(sinks[p[-1]], p) for p in paths)
        route = legalize_route(RouteTree(net, request.driver, request.driver.cell, branches), request)
        assert not isinstance(route, LegalizationFailure), route.message
        return route

    def verdict(self, realized: Mapping[int, RealizedRoute]) -> list[tuple[str, frozenset[Coord]]]:
        found = verify_design(self.mapped, self.requests, realized, max_y=9)
        return [(v.kind, frozenset(v.cells)) for v in found]

    def java(self, realized: Mapping[int, RealizedRoute]) -> list[tuple]:
        return java_problems(self.mapped, self.requests, realized)


@functools.cache
def hand_design(source: str = NOT2, origins: tuple[tuple[int, Coord], ...] = tuple(NOT2_ORIGINS.items())) -> Hand:
    netlist = synthesize_to_primitives(compile_source(source), interface=PADS)
    mapped = map_primitives_to_minecraft(netlist)
    grid = BlockGrid(max_y=9)
    net_of = mapped.net_of_terminal()
    for inst_id, origin in origins:
        placed = mapped.instances[inst_id].place(origin, IDENTITY)
        pins = [(p.position, p.approach) for p in placed.pins.values()]
        assert grid.check_placement(placed.occupied, placed.keepout, pins) is None
        sites = [
            PinSite(p.position, inst_id, p.name, p.direction, p.facing, p.strength, net_of.get(BitTerminal(inst_id, p.name)))
            for p in placed.pins.values()
        ]
        grid.place(inst_id, placed.occupied, placed.keepout, sites)
    return Hand(mapped, {r.net: r for r in route_requests(mapped, grid)})


#: net 1 detours west of everything, never near net 0.
AROUND = path((1, 1, -6), (2, 1, -6), (2, 1, -8), (-3, 1, -8), (-3, 1, 13), (23, 1, 13), (23, 1, 10), (24, 1, 10))
#: net 1 crosses two blocks above net 0's repeater at (16, 1, 0).
OVER_THE_REPEATER = path(
    (1, 1, -6), (16, 1, -6), (16, 1, -3), (16, 3, -1), (16, 3, 1), (16, 1, 3), (16, 1, 10), (24, 1, 10)
)


def baseline(hand: Hand, **routes: tuple[Coord, ...]) -> dict[int, RealizedRoute]:
    """All four NOT2 nets straight, net 1 crossing over net 0's repeater
    (override any net with ``n<id>=path``)."""
    paths = {
        "n0": path((1, 1, 0), (24, 1, 0)),
        "n1": OVER_THE_REPEATER,
        "n2": path((27, 1, 0), (34, 1, 0)),
        "n3": path((27, 1, 10), (34, 1, 10)),
    }
    paths.update(routes)
    return {int(k[1:]): hand.realize(int(k[1:]), p) for k, p in paths.items()}


def test_hand_baseline_crosses_two_blocks_over_a_repeater_legally() -> None:
    """Rule 4: the upper route's support may sit directly above a REPEATER of
    the lower route (a repeater ignores the block above it)."""
    hand = hand_design()
    realized = baseline(hand)
    assert [e.coord for e in realized[0].repeaters] == [(16, 1, 0)]
    assert (16, 2, 0) in realized[1].supports
    assert hand.verdict(realized) == []
    assert hand.java(realized) == []


def test_dust_diagonal_to_a_repeater_side_is_flagged_although_java_ignores_it() -> None:
    """Conservative-but-correct: a wire one block up beside a repeater never
    touches it in Java (a repeater reads only its back and dust does not
    connect diagonally to repeaters), yet rule 1 treats repeaters like dust."""
    hand = hand_design()
    beside = path((1, 1, -6), (16, 1, -6), (16, 1, -2), (16, 2, -1), (16, 3, 0), (16, 3, 1), (16, 1, 3),
                  (16, 1, 10), (24, 1, 10))  # fmt: skip
    realized = baseline(hand, n1=beside)
    assert hand.verdict(realized) == [("short", frozenset({(16, 1, 0), (16, 2, -1)}))]
    assert hand.java(realized) == []


def test_a_cut_staircase_diagonal_is_flagged_although_java_keeps_it_apart() -> None:
    """Conservative-but-correct: net 1's support right above net 0's wire cuts
    the diagonal between net 1's (10, 2, -1) and net 0's (10, 1, 0)."""
    hand = hand_design()
    hop = path((1, 1, -6), (10, 1, -6), (10, 1, -2), (10, 2, -1), (10, 3, 0), (10, 3, 1), (10, 1, 3),
               (10, 1, 10), (24, 1, 10))  # fmt: skip
    realized = baseline(hand, n1=hop)
    assert hand.verdict(realized) == [("short", frozenset({(10, 1, 0), (10, 2, -1)}))]
    assert hand.java(realized) == []


def test_an_uncut_staircase_between_two_nets_is_a_short_in_both() -> None:
    """Net 1 running one block up and one block beside net 0 connects to it
    through every open staircase -- except where legalization put a repeater
    on that run (Java dust never connects diagonally to a repeater; rule 1
    still flags it, conservatively)."""
    hand = hand_design()
    parallel = path((1, 1, -6), (4, 1, -6), (4, 1, -2), (4, 2, -1), (12, 2, -1), (12, 1, -2), (12, 1, -6),
                    (20, 1, -6), (20, 1, -3), (20, 3, -1), (20, 3, 1), (20, 1, 3), (20, 1, 10), (24, 1, 10))  # fmt: skip
    realized = baseline(hand, n1=parallel)
    repeaters = {e.coord for e in realized[1].repeaters}
    assert (11, 2, -1) in repeaters
    pairs = {cells for kind, cells in hand.verdict(realized) if kind == "short"}
    java = {frozenset(p[1:]) for p in hand.java(realized) if p[0] == "short"}
    beside = {frozenset({(x, 1, 0), (x, 2, -1)}) for x in range(4, 13)}
    assert beside <= pairs
    assert java == {pair for pair in beside if not pair & repeaters}


def test_a_net_looping_back_beside_its_own_repeater_is_a_latch() -> None:
    """Rule 1 within one net: dust after the repeater touching dust before it
    feeds the repeater its own output -- a latch Java would hold forever."""
    hand = hand_design()
    u_turn = path((1, 1, 0), (17, 1, 0), (17, 1, 1), (15, 1, 1), (15, 1, 3), (23, 1, 3), (23, 1, 0), (24, 1, 0))
    realized = baseline(hand, n0=u_turn, n1=AROUND)
    assert (16, 1, 0) in {e.coord for e in realized[0].repeaters}
    assert hand.verdict(realized) == [
        ("self_loop", frozenset({(15, 1, 0), (15, 1, 1)})),
        ("self_loop", frozenset({(16, 1, 0), (16, 1, 1)})),
    ]
    java = hand.java(realized)
    assert ("latch", (16, 1, 0)) in java
    assert "extra_connection" in problem_kinds(java)


def test_a_foreign_support_in_a_staircase_clearance_breaks_the_staircase() -> None:
    hand = hand_design()
    climb = path((1, 1, 0), (8, 1, 0), (9, 2, 0), (20, 2, 0), (21, 1, 0), (24, 1, 0))
    over = path((1, 1, -6), (8, 1, -6), (8, 1, -3), (8, 3, -1), (8, 3, 1), (8, 1, 3), (8, 1, 10), (24, 1, 10))
    realized = baseline(hand, n0=climb, n1=over)
    assert (8, 2, 0) in realized[0].clearances and (8, 2, 0) in realized[1].supports
    assert ("clearance_blocked", frozenset({(8, 2, 0)})) in hand.verdict(realized)
    java = hand.java(realized)
    assert ("broken_connection", (8, 1, 0), (9, 2, 0)) in java


def test_a_route_resting_on_its_own_staircase_clearance_breaks_itself() -> None:
    """Intra-net rule 3: a zig-zag whose next support is the headroom of the
    staircase it just climbed cuts that staircase."""
    hand = hand_design()
    zigzag = path((1, 1, 0), (8, 1, 0), (9, 2, 0), (8, 3, 0), (8, 3, 2), (20, 3, 2), (22, 1, 2), (22, 1, 0), (24, 1, 0))
    realized = baseline(hand, n0=zigzag, n1=AROUND)
    assert ("clearance_blocked", frozenset({(8, 2, 0)})) in hand.verdict(realized)
    assert ("broken_connection", (8, 1, 0), (9, 2, 0)) in hand.java(realized)


def test_a_sink_entered_from_the_side_does_not_drive_its_cell() -> None:
    """Rule 6: input dust entered sideways is a north-south line that no longer
    points into the cell east of it -- Java never powers the gate input."""
    hand = hand_design()
    sideways = path((1, 1, 0), (22, 1, 0), (22, 1, -2), (24, 1, -2), (24, 1, 0))
    realized = baseline(hand, n0=sideways, n1=AROUND)
    kinds = {kind for kind, _ in hand.verdict(realized)}
    assert {"pin_facing", "signal_in_component"} <= kinds
    assert ("sink_not_driven", (24, 1, 0)) in hand.java(realized)


def test_a_driver_left_against_its_facing_is_rejected() -> None:
    hand = hand_design()
    backwards = path((1, 1, 0), (1, 1, -2), (22, 1, -2), (22, 1, 0), (24, 1, 0))
    realized = baseline(hand, n0=backwards, n1=AROUND)
    kinds = {kind for kind, _ in hand.verdict(realized)}
    assert {"pin_facing", "signal_in_component"} <= kinds


def test_a_route_may_cross_three_blocks_above_a_cell_body() -> None:
    """Over NOT 2's body (y = 1, keep-out layer y = 2) a route needs its support
    at y = 3 or higher; at y = 2 the support sits in the keep-out."""
    hand = hand_design()
    high = path((1, 1, -6), (25, 1, -6), (25, 1, -5), (25, 4, -2), (25, 4, 2), (25, 1, 5), (25, 1, 7), (22, 1, 7),
                (22, 1, 10), (24, 1, 10))  # fmt: skip
    realized = baseline(hand, n1=high)
    assert (25, 3, 0) in realized[1].supports
    assert hand.verdict(realized) == []
    assert hand.java(realized) == []
    low = path((1, 1, -6), (25, 1, -6), (25, 1, -5), (25, 3, -3), (25, 3, 3), (25, 1, 5), (25, 1, 7), (22, 1, 7),
               (22, 1, 10), (24, 1, 10))  # fmt: skip
    realized = baseline(hand, n1=low)
    assert ("support_in_component", frozenset({(25, 2, 0)})) in hand.verdict(realized)


def test_a_route_may_pass_two_blocks_above_a_foreign_pin() -> None:
    """A support directly above another net's input pin is harmless: weak power
    never reaches the pin dust and the pin still points into its cell."""
    hand = hand_design()
    over_pin = path((1, 1, -6), (24, 1, -6), (24, 1, -4), (24, 3, -2), (24, 3, 2), (24, 1, 4), (24, 1, 7),
                    (22, 1, 7), (22, 1, 10), (24, 1, 10))  # fmt: skip
    realized = baseline(hand, n1=over_pin)
    assert (24, 2, 0) in realized[1].supports  # right above NOT 2's input pin (24, 1, 0)
    assert hand.verdict(realized) == []
    assert hand.java(realized) == []


# -- 4. verify_design must report every broken realized route ---------------------------------


def wide_not() -> tuple[PrimitivePhysicalNetlist, dict[int, RouteRequest], dict[int, RealizedRoute]]:
    mapped, result = pnr("not", "wide")
    realized = dict(result.realized)
    assert verify_design(mapped, result.routed.requests, realized, max_y=result.geometry.max_y) == []
    assert any(r.repeaters for r in realized.values())
    return mapped, result.routed.requests, realized


def verdict(mapped, requests, realized) -> list[str]:
    return [v.kind for v in verify_design(mapped, requests, realized, max_y=9)]


def replace_elements(route: RealizedRoute, change) -> RealizedRoute:
    return dataclasses.replace(route, elements=tuple(change(e) for e in route.elements))


def _drop_a_support(route: RealizedRoute) -> RealizedRoute:
    return dataclasses.replace(route, supports=route.supports[1:])


def _flip_the_repeater(route: RealizedRoute) -> RealizedRoute:
    first = route.repeaters[0].coord
    return replace_elements(
        route, lambda e: dataclasses.replace(e, facing=e.facing.opposite) if e.coord == first else e
    )


def _strip_the_repeaters(route: RealizedRoute) -> RealizedRoute:
    return replace_elements(route, lambda e: dataclasses.replace(e, kind=ElementKind.DUST, facing=None))


@pytest.mark.parametrize(
    ("mutate", "model", "java"),
    [
        (_drop_a_support, "unsupported", "floating"),
        (_flip_the_repeater, "repeater", "repeater_misplaced"),
        (_strip_the_repeaters, "weak_signal", "weak_sink"),
    ],
    ids=lambda v: v.__name__[1:] if callable(v) else v,
)
def test_verify_and_java_agree_that_a_broken_route_is_broken(mutate, model: str, java: str) -> None:
    mapped, requests, realized = wide_not()
    mutated = {**realized, 0: mutate(realized[0])}
    assert model in verdict(mapped, requests, mutated)
    assert java in problem_kinds(java_problems(mapped, requests, mutated))


@pytest.mark.parametrize("kind", [ElementKind.SUPPORT, ElementKind.CLEARANCE], ids=lambda k: k.value)
def test_verify_rejects_a_signal_block_that_is_not_dust_or_a_repeater(kind: ElementKind) -> None:
    """A route element realized as a solid block (or left as air) breaks the
    wire in Java -- the sink goes dark -- so verification must not pass it."""
    mapped, requests, realized = wide_not()
    route = realized[0]
    victim = route.elements[5].coord
    broken = dataclasses.replace(
        route, elements=tuple(dataclasses.replace(e, kind=kind) if e.coord == victim else e for e in route.elements)
    )
    mutated = {**realized, 0: broken}
    assert {"strength", "weak_sink"} <= problem_kinds(java_problems(mapped, requests, mutated))
    assert verdict(mapped, requests, mutated), f"a {kind.value} element passed verification"


def _drop_element(route: RealizedRoute) -> RealizedRoute:
    return dataclasses.replace(route, elements=route.elements[:5] + route.elements[6:])


def _detach_branch(route: RealizedRoute) -> RealizedRoute:
    (branch,) = route.tree.branches
    tree = dataclasses.replace(route.tree, branches=(RouteBranch(branch.sink, branch.path[3:]),))
    return dataclasses.replace(route, tree=tree)


def _split_out_of_order(route: RealizedRoute) -> RealizedRoute:
    """The same blocks as two branches listed tail first (the tail starts on
    a block the second branch only adds later)."""
    (branch,) = route.tree.branches
    head = RouteBranch(dataclasses.replace(branch.sink, cell=branch.path[6]), branch.path[:7])
    tail = RouteBranch(branch.sink, branch.path[6:])
    return dataclasses.replace(route, tree=dataclasses.replace(route.tree, branches=(tail, head)))


@pytest.mark.parametrize("mutate", [_drop_element, _detach_branch, _split_out_of_order], ids=lambda f: f.__name__[1:])
def test_verify_reports_malformed_routes_instead_of_crashing(mutate) -> None:
    mapped, requests, realized = wide_not()
    mutated = {**realized, 0: mutate(realized[0])}
    assert verdict(mapped, requests, mutated)


def test_verify_checks_routes_of_nets_it_was_not_asked_about() -> None:
    """verify_design walks ``requests`` only, so a realized route whose net is
    not requested is never looked at -- not even one lying on every block of
    another net.  (Today the pipeline builds both from the same nets, so this
    is a defence-in-depth gap: ``final.realized`` and the metrics would still
    include such a route.)"""
    mapped, requests, realized = wide_not()
    stray = dataclasses.replace(realized[0], net=99)
    mutated = {**realized, 99: stray}
    assert verdict(mapped, requests, mutated), "a route sharing every block of net 0 passed verification"


def _unpowered_claims_power(realized: dict[int, RealizedRoute], net: int) -> RealizedRoute:
    route = realized[net]
    elements = tuple(dataclasses.replace(e, strength=MAX_SIGNAL_STRENGTH) for e in route.elements)
    return dataclasses.replace(route, elements=elements)


def test_verify_checks_the_recorded_strength_of_unpowered_nets() -> None:
    """A constant-0 net is never powered in Java; a design file recording its
    dust at strength 15 is wrong, yet the strength check skips undriven nets."""
    mapped, result = pnr("half_adder")
    requests, realized = result.routed.requests, dict(result.realized)
    net = next(n for n, r in sorted(realized.items()) if not r.powered)
    mutated = {**realized, net: _unpowered_claims_power(realized, net)}
    assert "strength" in problem_kinds(java_problems(mapped, requests, mutated))
    assert verdict(mapped, requests, mutated), "an unpowered net recorded at strength 15 passed verification"


@pytest.mark.parametrize("field_", ["powered", "sinks"])
def test_verify_checks_the_recorded_electrical_summary(field_: str) -> None:
    """``powered`` and the per-sink reports (strength, repeaters, delay) are
    part of the physical design file and feed the metrics
    (``min_sink_strength``, ``critical_path_ticks``); verification never
    re-derives them."""
    mapped, requests, realized = wide_not()
    route = realized[0]
    if field_ == "powered":
        broken = dataclasses.replace(route, powered=False)
    else:
        (report,) = route.sinks
        broken = dataclasses.replace(route, sinks=(dataclasses.replace(report, strength=1, repeaters=0, delay_ticks=9),))
    assert verdict(mapped, requests, {**realized, 0: broken}), f"a falsified {field_!r} record passed verification"


# -- 5. electrical legalization at the strength limits -------------------------------------


def straight(drive: int, length: int) -> tuple[RouteTree, RouteRequest]:
    driver = PinSite((0, 1, 0), 0, "y", "out", Direction.EAST, drive, 0)
    sink = PinSite((length, 1, 0), 1, "a", "in", Direction.WEST, 1, 0)
    route = tuple((x, 1, 0) for x in range(length + 1))
    return RouteTree(0, driver, driver.cell, (RouteBranch(sink, route),)), RouteRequest(0, "data", driver, (sink,))


@pytest.mark.parametrize("drive", [MAX_SIGNAL_STRENGTH, 13])
def test_a_driver_feeds_exactly_drive_dust_blocks_before_a_repeater(drive: int) -> None:
    """Off-by-one guard: the pin dust is ``drive``, each later dust one less,
    so ``drive`` blocks (pin included) reach the sink at 1 and one more block
    needs a repeater -- whose output dust is back at 15."""
    tree, request = straight(drive, drive - 1)
    route = legalize_route(tree, request)
    assert isinstance(route, RealizedRoute) and not route.repeaters
    assert [e.strength for e in route.elements] == list(range(drive, 0, -1))
    tree, request = straight(drive, drive)
    route = legalize_route(tree, request)
    assert isinstance(route, RealizedRoute)
    (repeater,) = route.repeaters
    assert repeater.coord == (drive - 1, 1, 0) and repeater.strength == 2
    assert route.sinks[0].strength == MAX_SIGNAL_STRENGTH


def test_a_constant_zero_net_is_never_repeated_however_long() -> None:
    tree, request = straight(0, 60)
    route = legalize_route(tree, request)
    assert isinstance(route, RealizedRoute)
    assert not route.powered and not route.repeaters
    assert {e.strength for e in route.elements} == {0}


# -- 6. placement keeps pins of different cells apart in every orientation ---------------------


def test_check_placement_never_lets_pins_or_approaches_of_two_cells_touch() -> None:
    """Fuzz rotated library cells into one grid: whatever check_placement
    accepts keeps every pin endpoint AND approach (both carry the pin's net's
    dust) out of every other cell's pins' neighbourhoods, off bodies and out
    of keep-outs."""
    cells = [c for c in PRIMITIVE_TECHNOLOGY.values() if c.peripheral is None]
    rng = random.Random(20261004)
    grid = BlockGrid(max_y=9)
    placed = []
    for _ in range(400):
        cell = rng.choice(cells)
        orientation = rng.choice(ORIENTATIONS)
        origin = (rng.randrange(-12, 13), 0, rng.randrange(-12, 13))
        candidate = cell.oriented(orientation).translate(origin)
        pins = [(p.position, p.approach) for p in candidate.pins.values()]
        if grid.check_placement(candidate.occupied, candidate.keepout, pins) is None:
            sites = [PinSite(p.position, len(placed), p.name, p.direction, p.facing, p.strength, None)
                     for p in candidate.pins.values()]  # fmt: skip
            grid.place(len(placed), candidate.occupied, candidate.keepout, sites)
            placed.append(candidate)
    assert len(placed) >= 10
    assert len({p.orientation for p in placed}) == 4
    hood = set(SIGNAL_NEIGHBORHOOD) | {(0, 0, 0)}
    for a, b in combinations(placed, 2):
        ends_a = {c for p in a.pins.values() for c in (p.position, p.approach)}
        ends_b = {c for p in b.pins.values() for c in (p.position, p.approach)}
        for x in ends_a:
            for y in ends_b:
                assert (y[0] - x[0], y[1] - x[1], y[2] - x[2]) not in hood, (x, y)
        assert not ends_a & (b.occupied | b.keepout)
        assert not ends_b & (a.occupied | a.keepout)
