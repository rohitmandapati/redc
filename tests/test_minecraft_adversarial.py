"""The redstone simulator as a PHYSICAL oracle: take real place-and-route
output, break it the way a buggy router / legalizer / materializer could --
without telling the compiler -- and require the simulation to notice.

Every mutation keeps the design's metadata (net annotations, components,
ports) unchanged; only blocks move.  The simulator derives connectivity from
the blocks, so it must see the short, the reversed diode, the weak signal or
the gap."""

from __future__ import annotations

import dataclasses
import functools
import itertools

import pytest

from redc import compile_source
from redc.minecraft import MinecraftPhysicalDesign, RedstoneSimulator, SimulationError
from redc.minecraft.blocks import (
    OPPOSITE,
    redstone_wire,
    repeater,
    repeater_output,
    stone,
)
from redc.minecraft.closure import close_timing
from redc.minecraft.sta import analyze
from redc.physical_primitive import (
    PadInterfacePolicy,
    PrimitivePnRConfig,
    place_and_route_graph,
)

ADD3 = "uint3 main(uint3 a, uint3 b) { return a + b; }"
NOT = "bool main(bool a) { return !a; }"


@functools.cache
def routed(source: str, channel_width: int = 6):
    graph = compile_source(source)
    _netlist, _mapped, result = place_and_route_graph(
        graph, PrimitivePnRConfig(channel_width=channel_width), interface=PadInterfacePolicy()
    )
    assert result.success, result.failure
    return graph, result.timing.world


def vectors(graph):
    ports = [(p["name"], graph.nodes[p["node"]]["type"]["width"]) for p in graph.inputs]
    for values in itertools.product(*[range(1 << w) for _, w in ports]):
        yield dict(zip((n for n, _ in ports), values))


def disagrees(graph, world: MinecraftPhysicalDesign) -> bool:
    """Whether the redstone simulation of ``world`` departs from the IR anywhere
    (a wrong output, a weak input, an unstable circuit or a refused world)."""
    try:
        sim = RedstoneSimulator(world)
        for vector in vectors(graph):
            if sim.evaluate(max_ticks=2_000, **vector) != graph.evaluate(**vector) or sim.warnings:
                return True
    except SimulationError:
        return True
    return False


def with_blocks(world: MinecraftPhysicalDesign, changes: dict) -> MinecraftPhysicalDesign:
    blocks = dict(world.blocks)
    blocks.update(changes)
    return dataclasses.replace(world, blocks=blocks)


def nets_of(world: MinecraftPhysicalDesign) -> dict[int, list[tuple[int, int, int]]]:
    nets: dict[int, list] = {}
    for coord, net in sorted(world.annotations.items()):
        nets.setdefault(net, []).append(coord)
    return nets


def test_the_unmodified_world_agrees() -> None:
    graph, world = routed(ADD3)
    assert not disagrees(graph, world)


def test_a_physical_short_between_two_nets_is_caught() -> None:
    """Join two nets' dust with one extra dust block: the compiler still
    believes they are separate nets; the simulator sees one network."""
    graph, world = routed(ADD3)
    wires = {c for c, b in world.blocks.items() if b.id == "minecraft:redstone_wire"}
    owner = world.annotations
    found = None
    for coord in sorted(wires):
        x, y, z = coord
        for dx, dz in ((2, 0), (0, 2)):
            other = (x + dx, y, z + dz)
            middle = (x + dx // 2, y, z + dz // 2)
            below = (middle[0], y - 1, middle[2])
            if (
                other in wires
                and owner.get(other) != owner.get(coord)
                and middle not in world.blocks
                and world.blocks.get(below, stone()).id == "minecraft:stone"
            ):
                found = middle
                break
        if found:
            break
    assert found is not None, "no two nets run two blocks apart in this layout"
    support = (found[0], found[1] - 1, found[2])
    assert disagrees(graph, with_blocks(world, {found: redstone_wire(), support: stone()}))


def test_a_reversed_repeater_is_caught() -> None:
    graph, world = routed(ADD3)
    reps = sorted(c for c, b in world.blocks.items() if b.id == "minecraft:repeater")
    assert reps
    caught = 0
    for coord in reps[:6]:
        block = world.blocks[coord]
        flipped = repeater(OPPOSITE[repeater_output(block)], int(block.get("delay")))
        caught += disagrees(graph, with_blocks(world, {coord: flipped}))
    assert caught == min(6, len(reps))


def test_a_missing_repeater_leaves_a_too_weak_signal() -> None:
    graph, world = routed(NOT, channel_width=20)  # long nets: every one needs a repeater
    reps = sorted(c for c, b in world.blocks.items() if b.id == "minecraft:repeater")
    assert reps
    for coord in reps:
        assert disagrees(graph, with_blocks(world, {coord: redstone_wire()}))


def test_a_gap_in_a_route_disconnects_it() -> None:
    graph, world = routed(ADD3)
    for net, cells in sorted(nets_of(world).items())[:6]:
        dust = [c for c in cells if world.blocks[c].id == "minecraft:redstone_wire"]
        middle = dust[len(dust) // 2]
        broken = dict(world.blocks)
        del broken[middle]
        assert disagrees(graph, dataclasses.replace(world, blocks=broken)), net


def test_a_removed_support_is_refused() -> None:
    _graph, world = routed(ADD3)
    dust = next(
        c for c, b in sorted(world.blocks.items())
        if b.id == "minecraft:redstone_wire" and world.blocks[(c[0], c[1] - 1, c[2])].id == "minecraft:stone"
    )  # fmt: skip
    below = (dust[0], dust[1] - 1, dust[2])
    broken = dict(world.blocks)
    del broken[below]
    with pytest.raises(SimulationError) as info:
        RedstoneSimulator(dataclasses.replace(world, blocks=broken))
    assert info.value.code == "simulation/unsupported_placement"


def test_labels_never_influence_connectivity() -> None:
    """Scrambling every annotation changes nothing the simulator or STA do."""
    graph, world = routed(ADD3)
    scrambled = dataclasses.replace(world, annotations={c: -1 for c in world.annotations})
    a, b = RedstoneSimulator(world, trace="full"), RedstoneSimulator(scrambled, trace="full")
    for vector in itertools.islice(vectors(graph), 12):
        assert a.evaluate(**vector) == b.evaluate(**vector)
    assert a.events == b.events
    ra = close_timing(analyze(world)).report["combinational"]
    rb = close_timing(analyze(scrambled)).report["combinational"]
    assert ra["settle"] == rb["settle"] and len(ra["critical_path"]["steps"]) == len(rb["critical_path"]["steps"])


def test_every_simulated_output_change_lies_within_the_static_timing_window() -> None:
    """For every single-input transition: no output changes before STA's
    earliest arrival or after its latest -- simulation and STA share one model."""
    graph, world = routed(ADD3)
    report = close_timing(analyze(world)).report["combinational"]
    earliest, latest = report["earliest_change"]["gt"], report["settle"]["gt"]
    sim = RedstoneSimulator(world, trace="events")
    sim.run_until_stable()
    all_vectors = list(vectors(graph))
    for before, after in itertools.pairwise(all_vectors):
        sim.evaluate(**before)
        start = sim.time
        sim.events.clear()
        for name, value in after.items():
            sim.set_input(name, value)
        sim.run_until_stable()
        changes = [e["t"] - start for e in sim.events if e["type"] == "output_changed"]
        assert all(earliest <= t <= latest for t in changes), (before, after, changes)


def test_the_minecraft_layer_imports_no_backend() -> None:
    """``redc.minecraft`` is backend-neutral: importing all of it pulls in
    neither physical backend (``redc.physical``, ``redc.physical_primitive``),
    the physical-backend registry nor the viewer."""
    import subprocess
    import sys

    code = (
        "import sys, redc.minecraft, redc.minecraft.sta, redc.minecraft.closure, redc.minecraft.testbench, "
        "redc.minecraft.characterize, redc.minecraft.playback; "
        "print(sorted(m for m in sys.modules if m.startswith(('redc.physical', 'redc.viewer'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
