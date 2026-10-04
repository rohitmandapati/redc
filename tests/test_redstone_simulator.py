"""The backend-neutral redstone simulator, tested on hand-built block worlds
(no RedC compilation): every behaviour of the supported v1 subset, the
event engine's determinism and its explicit refusals."""

from __future__ import annotations

import json

import pytest

from redc.minecraft import (
    ABSTRACT_MODE,
    BLOCK_MODE,
    AbstractComponent,
    ComponentPin,
    ComponentTiming,
    MinecraftBlock,
    MinecraftPhysicalDesign,
    Port,
    PortBit,
    Probe,
    RedstoneSimulator,
    SimulationError,
)
from redc.minecraft.blocks import (
    abstract_block,
    glass,
    lever,
    redstone_block,
    redstone_lamp,
    redstone_torch,
    redstone_wire,
    repeater,
    stone,
    wall_torch,
)
from redc.minecraft.connectivity import compile_world, dust_shape
from redc.minecraft.timing import (
    TORCH_DELAY_GT,
    CombinationalArc,
    SequentialTiming,
    repeater_delay_gt,
    uniform_arcs,
)
from redc.minecraft.units import gt_to_rt, rt_to_gt

Coord = tuple[int, int, int]
LEVER_AT: Coord = (-1, 1, 0)


def dust_run(blocks: dict, start: Coord, n: int, step: Coord = (1, 0, 0)) -> list[Coord]:
    """``n`` dust blocks from ``start`` along ``step``, each on a stone support."""
    cells = []
    x, y, z = start
    for _ in range(n):
        blocks[(x, y - 1, z)] = stone()
        blocks[(x, y, z)] = redstone_wire()
        cells.append((x, y, z))
        x, y, z = x + step[0], y + step[1], z + step[2]
    return cells


def with_lever(blocks: dict, at: Coord = LEVER_AT) -> Port:
    blocks[(at[0], at[1] - 1, at[2])] = stone()
    blocks[at] = lever("floor", "east")
    return Port("a", "in", (PortBit("lever", at),))


def observe(name: str, coord: Coord, threshold: int = 1) -> Port:
    return Port(name, "out", (PortBit("dust", coord, threshold),))


def design(blocks: dict, *ports: Port, **kw) -> MinecraftPhysicalDesign:
    return MinecraftPhysicalDesign(dict(blocks), ports=tuple(ports), source_backend="test", **kw)


def first_change(sim: RedstoneSimulator, port: str, value: int) -> int:
    """The tick at which output ``port`` first reads ``value``."""
    while sim.read_output(port) != value:
        assert sim.step() is not None, f"{port} never became {value}"
    return sim.time


# -- 1-3, 11: dust ----------------------------------------------------------------------


def test_dust_attenuates_one_level_per_block_from_15() -> None:
    blocks: dict = {}
    cells = dust_run(blocks, (0, 1, 0), 18)
    sim = RedstoneSimulator(design(blocks, with_lever(blocks)))
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert [sim.dust_at(c) for c in cells] == [15 - i for i in range(15)] + [0, 0, 0]
    sim.set_input("a", 0)
    sim.run_until_stable()
    assert all(sim.dust_at(c) == 0 for c in cells)


def test_dust_staircase_connects_only_through_an_air_clearance() -> None:
    def build(clearance: MinecraftBlock | None) -> tuple[RedstoneSimulator, Coord]:
        blocks: dict = {}
        dust_run(blocks, (0, 1, 0), 3)  # x = 0..2 at y=1
        blocks[(3, 1, 0)] = stone()  # support of the upper dust, beside the lower end
        blocks[(3, 2, 0)] = redstone_wire()
        if clearance is not None:
            blocks[(2, 2, 0)] = clearance  # directly above the lower dust
        return RedstoneSimulator(design(blocks, with_lever(blocks))), (3, 2, 0)

    sim, top = build(None)
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert sim.dust_at(top) == 12  # 15, 14, 13 below, then one step up
    sim, top = build(stone())
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert sim.dust_at(top) == 0  # an opaque block above the lower dust cuts the stair


def test_staircase_on_glass_carries_signal_up_but_not_down() -> None:
    def build(lever_low: bool) -> tuple[RedstoneSimulator, Coord, Coord]:
        blocks: dict = {(0, 0, 0): stone(), (0, 1, 0): redstone_wire(), (1, 1, 0): glass(), (1, 2, 0): redstone_wire()}
        low, high = (0, 1, 0), (1, 2, 0)
        if lever_low:
            port = with_lever(blocks, (-1, 1, 0))
        else:
            blocks[(2, 1, 0)] = stone()
            blocks[(2, 2, 0)] = lever("floor", "west")
            port = Port("a", "in", (PortBit("lever", (2, 2, 0)),))
        return RedstoneSimulator(design(blocks, port)), low, high

    sim, low, high = build(lever_low=True)
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert (sim.dust_at(low), sim.dust_at(high)) == (15, 14)
    sim, low, high = build(lever_low=False)
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert (sim.dust_at(low), sim.dust_at(high)) == (0, 15)


def test_connectivity_comes_from_geometry_not_from_labels() -> None:
    """Two routes the producer labels as different nets touch: the simulator
    connects them, because they touch."""
    blocks: dict = {}
    mine = dust_run(blocks, (0, 1, 0), 4)
    theirs = dust_run(blocks, (3, 1, 1), 4)  # (3,1,1) is beside (3,1,0)
    labels = {c: "net 1" for c in mine} | {c: "net 2" for c in theirs}
    sim = RedstoneSimulator(design(blocks, with_lever(blocks), annotations=labels))
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert sim.dust_at(theirs[0]) == 11 and sim.dust_at(theirs[-1]) == 8


def test_diagonal_dust_in_one_plane_does_not_connect() -> None:
    blocks: dict = {}
    dust_run(blocks, (0, 1, 0), 3)
    other = dust_run(blocks, (3, 1, 1), 2)  # touches (2,1,0) only at a corner
    sim = RedstoneSimulator(design(blocks, with_lever(blocks)))
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert all(sim.dust_at(c) == 0 for c in other)


def test_branching_dust_powers_every_branch_with_its_own_attenuation() -> None:
    blocks: dict = {}
    trunk = dust_run(blocks, (0, 1, 0), 4)
    north = dust_run(blocks, (3, 1, -1), 5, (0, 0, -1))
    south = dust_run(blocks, (3, 1, 1), 2, (0, 0, 1))
    sim = RedstoneSimulator(design(blocks, with_lever(blocks)))
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert [sim.dust_at(c) for c in trunk] == [15, 14, 13, 12]
    assert [sim.dust_at(c) for c in north] == [11, 10, 9, 8, 7]
    assert [sim.dust_at(c) for c in south] == [11, 10]


def test_dust_shape_follows_java_connection_rules() -> None:
    blocks: dict = {}
    dust_run(blocks, (0, 1, 0), 1)
    view = compile_world(design(blocks)).view
    assert dust_shape(view, (0, 1, 0)) == ("east", "south", "west", "north")  # isolated: a cross
    dust_run(blocks, (1, 1, 0), 1)
    view = compile_world(design(blocks)).view
    assert dust_shape(view, (0, 1, 0)) == ("east", "west")  # one connection: a line
    dust_run(blocks, (0, 1, 1), 1)
    view = compile_world(design(blocks)).view
    assert dust_shape(view, (0, 1, 0)) == ("east", "south")  # a corner points nowhere else


def test_weak_power_from_dust_never_reaches_other_dust_but_strong_power_does() -> None:
    blocks: dict = {}
    dust_run(blocks, (0, 1, 0), 1)  # points east into the block at (1,1,0)
    blocks[(1, 1, 0)] = stone()
    blocks[(1, 0, 0)] = stone()
    beyond = dust_run(blocks, (2, 1, 0), 1)[0]
    sim = RedstoneSimulator(design(blocks, with_lever(blocks)))
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert sim.dust_at(beyond) == 0  # the block is only WEAKLY powered
    blocks[(0, 1, 0)] = repeater("east", 1)  # a repeater facing in powers it STRONGLY
    sim = RedstoneSimulator(design(blocks, with_lever(blocks)))
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert sim.dust_at(beyond) == 15


# -- 4-10: repeaters -------------------------------------------------------------------------


def repeater_line(delays: list[int], *, facing: str = "east") -> tuple[MinecraftPhysicalDesign, Coord]:
    blocks: dict = {}
    port = with_lever(blocks)
    x = 0
    for delay in delays:
        dust_run(blocks, (x, 1, 0), 1)
        blocks[(x + 1, 0, 0)] = stone()
        blocks[(x + 1, 1, 0)] = repeater(facing, delay)
        x += 2
    out = dust_run(blocks, (x, 1, 0), 1)[0]
    return design(blocks, port, observe("y", out)), out


def test_straight_repeater_conducts_and_refreshes_to_15() -> None:
    d, out = repeater_line([1])
    sim = RedstoneSimulator(d)
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert sim.dust_at(out) == 15 and sim.read_output("y") == 1


def test_reversed_repeater_does_not_conduct() -> None:
    d, out = repeater_line([1], facing="west")
    sim = RedstoneSimulator(d)
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert sim.dust_at(out) == 0 and sim.read_output("y") == 0


@pytest.mark.parametrize("delay", [1, 2, 3, 4])
def test_repeater_delay_is_its_configured_redstone_ticks(delay: int) -> None:
    d, _out = repeater_line([delay])
    sim = RedstoneSimulator(d)
    sim.run_until_stable()
    sim.schedule_input("a", 1, at_tick=10)
    assert first_change(sim, "y", 1) - 10 == rt_to_gt(delay) == repeater_delay_gt(delay)
    sim.schedule_input("a", 0, at_tick=40)
    assert first_change(sim, "y", 0) - 40 == rt_to_gt(delay)
    assert gt_to_rt(rt_to_gt(delay)) == delay


def test_chained_repeater_delays_add_up() -> None:
    d, _out = repeater_line([2, 3, 1])
    sim = RedstoneSimulator(d)
    sim.schedule_input("a", 1, at_tick=4)
    assert first_change(sim, "y", 1) - 4 == rt_to_gt(2 + 3 + 1)


def test_a_pulse_shorter_than_the_delay_is_extended() -> None:
    d, _out = repeater_line([4])
    sim = RedstoneSimulator(d, trace="events")
    sim.schedule_input("a", 1, at_tick=10)
    sim.schedule_input("a", 0, at_tick=12)  # a 1-redstone-tick pulse into a 4-tick repeater
    on = first_change(sim, "y", 1)
    off = first_change(sim, "y", 0)
    assert (on, off) == (10 + 8, 10 + 16)


def test_repeater_locking_is_an_explicit_unsupported_mechanic() -> None:
    blocks: dict = {(0, 0, 0): stone(), (0, 1, 0): repeater("east"), (0, 0, 1): stone(), (0, 1, 1): repeater("north")}
    with pytest.raises(SimulationError) as info:
        RedstoneSimulator(design(blocks))
    assert info.value.code == "simulation/unsupported_mechanic"


# -- 12-15: torches, sources, levers ---------------------------------------------------


def torch_inverter() -> tuple[MinecraftPhysicalDesign, Coord]:
    blocks: dict = {}
    port = with_lever(blocks)
    dust_run(blocks, (0, 1, 0), 1)
    blocks[(1, 0, 0)] = stone()
    blocks[(1, 1, 0)] = stone()
    blocks[(2, 1, 0)] = wall_torch("east")
    out = dust_run(blocks, (3, 1, 0), 1)[0]
    return design(blocks, port, observe("y", out)), out


def test_torch_inverts() -> None:
    d, out = torch_inverter()
    sim = RedstoneSimulator(d)
    sim.run_until_stable()
    assert sim.read_output("y") == 1 and sim.dust_at(out) == 15
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert sim.read_output("y") == 0


def test_torch_toggles_one_redstone_tick_after_its_input() -> None:
    d, _out = torch_inverter()
    sim = RedstoneSimulator(d)
    assert first_change(sim, "y", 1) == TORCH_DELAY_GT  # cold start: unlit, lights after 1 rt
    sim.run_until_stable()
    sim.schedule_input("a", 1, at_tick=20)
    assert first_change(sim, "y", 0) - 20 == TORCH_DELAY_GT == rt_to_gt(1)
    sim.schedule_input("a", 0, at_tick=30)
    assert first_change(sim, "y", 1) - 30 == TORCH_DELAY_GT


def test_standing_torch_reads_the_block_below_it() -> None:
    blocks: dict = {}
    port = with_lever(blocks)
    dust_run(blocks, (0, 1, 0), 2)
    blocks[(1, 2, 0)] = stone()  # cuts nothing: no stair here; dust (1,1,0) powers... nothing above
    blocks[(1, 0, 0)] = stone()
    blocks[(2, 0, 0)] = stone()
    blocks[(2, 1, 0)] = stone()
    blocks[(2, 2, 0)] = redstone_torch()  # on the block the dust points into
    out = (3, 2, 0)
    blocks[(3, 1, 0)] = stone()
    blocks[out] = redstone_wire()
    sim = RedstoneSimulator(design(blocks, port, observe("y", out)))
    sim.run_until_stable()
    assert sim.read_output("y") == 1
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert sim.read_output("y") == 0


def test_redstone_block_powers_adjacent_dust_from_the_start() -> None:
    blocks: dict = {(0, 0, 0): stone(), (0, 1, 0): redstone_block()}
    cells = dust_run(blocks, (1, 1, 0), 3)
    sim = RedstoneSimulator(design(blocks, observe("y", cells[-1])))
    assert sim.time == 0 and [sim.dust_at(c) for c in cells] == [15, 14, 13]
    assert sim.is_stable()


def test_lever_transitions_follow_the_stimulus() -> None:
    blocks: dict = {}
    cells = dust_run(blocks, (0, 1, 0), 3)
    sim = RedstoneSimulator(design(blocks, with_lever(blocks), observe("y", cells[-1])), trace="events")
    seen = []
    for tick, value in ((0, 1), (6, 0), (8, 1), (30, 0)):
        sim.schedule_input("a", value, at_tick=tick)
    while sim.step() is not None:
        seen.append((sim.time, sim.read_output("y")))
    assert seen == [(0, 1), (6, 0), (8, 1), (30, 0)]
    changes = [e for e in sim.events if e["type"] == "input_changed"]
    assert [(e["t"], e["value"]) for e in changes] == [(0, 1), (6, 0), (8, 1), (30, 0)]


def test_lamp_lights_at_once_and_turns_off_two_redstone_ticks_later() -> None:
    blocks: dict = {}
    cells = dust_run(blocks, (0, 1, 0), 2)
    blocks[(2, 1, 0)] = redstone_lamp()
    blocks[(2, 0, 0)] = stone()
    lamp = Port("y", "out", (PortBit("lamp", (2, 1, 0)),))
    sim = RedstoneSimulator(design(blocks, with_lever(blocks), lamp))
    sim.schedule_input("a", 1, at_tick=10)
    assert first_change(sim, "y", 1) == 10 and sim.dust_at(cells[-1]) == 14
    sim.schedule_input("a", 0, at_tick=20)
    assert first_change(sim, "y", 0) == 20 + rt_to_gt(2)


# -- 16-19: the engine ---------------------------------------------------------------------


def busy_design() -> MinecraftPhysicalDesign:
    """A branching circuit with repeaters, torches and a lamp."""
    blocks: dict = {}
    port = with_lever(blocks)
    dust_run(blocks, (0, 1, 0), 3)
    blocks[(3, 0, 0)] = stone()
    blocks[(3, 1, 0)] = repeater("east", 2)
    dust_run(blocks, (4, 1, 0), 3)
    dust_run(blocks, (2, 1, 1), 4, (0, 0, 1))
    blocks[(2, 1, 5)] = stone()
    blocks[(2, 0, 5)] = stone()
    blocks[(2, 1, 6)] = wall_torch("south")
    dust_run(blocks, (2, 1, 7), 3, (0, 0, 1))
    blocks[(7, 1, 0)] = redstone_lamp()
    blocks[(7, 0, 0)] = stone()
    return design(
        blocks, port, Port("y", "out", (PortBit("lamp", (7, 1, 0)),)), observe("z", (2, 1, 9)),
        probes=(Probe("trunk", (1, 1, 0)),),
    )  # fmt: skip


def stimulate(sim: RedstoneSimulator) -> list[tuple[int, dict[str, int]]]:
    for tick, value in ((3, 1), (9, 0), (10, 1), (40, 0), (44, 1)):
        sim.schedule_input("a", value, at_tick=tick)
    seen = []
    while sim.step() is not None:
        seen.append((sim.time, sim.read_outputs()))
    return seen


def test_identical_design_and_stimulus_give_identical_event_sequences() -> None:
    runs = []
    for _ in range(3):
        sim = RedstoneSimulator(busy_design(), trace="full")
        seen = stimulate(sim)
        runs.append((seen, sim.events, sim.snapshot(), sim.stats))
    assert runs[0] == runs[1] == runs[2]
    assert any(e["type"] == "wire_strength_changed" for e in runs[0][1])


def test_a_settled_circuit_is_stable() -> None:
    sim = RedstoneSimulator(busy_design())
    sim.set_input("a", 1)
    last = sim.run_until_stable()
    assert sim.is_stable() and sim.pending_events == 0 and sim.next_event_time() is None
    before = sim.snapshot()
    sim.run_until(last + 1000)
    assert sim.snapshot()[1:] == before[1:]  # nothing moves without stimulus
    assert sim.read_outputs() == {"y": 1, "z": 0}  # z is behind the torch inverter


def test_run_until_stops_before_the_given_tick() -> None:
    d, _out = repeater_line([1])
    sim = RedstoneSimulator(d)
    sim.schedule_input("a", 1, at_tick=10)
    sim.run_until(12)
    assert sim.time == 12 and sim.read_output("y") == 0 and sim.pending_events == 1
    sim.run_until(13)
    assert sim.read_output("y") == 1


def test_a_torch_feeding_itself_is_reported_as_unstable_burnout() -> None:
    """Torch -> dust -> staircase onto its own block: a 2-redstone-tick
    clock that Java burns out.  The simulator refuses rather than pretends."""
    blocks: dict = {(0, 0, 0): stone(), (0, 1, 0): stone(), (1, 1, 0): wall_torch("east")}
    path = dust_run(blocks, (1, 1, -1), 1) + dust_run(blocks, (0, 1, -1), 2, (-1, 0, 0))
    path += dust_run(blocks, (-1, 1, 0), 1)
    blocks[(0, 2, 0)] = redstone_wire()  # on top of the torch's block, one stair up from (-1,1,0)
    sim = RedstoneSimulator(design(blocks))
    with pytest.raises(SimulationError) as info:
        sim.run_until_stable()
    assert info.value.code == "simulation/unstable" and "burn" in str(info.value)


def test_an_oscillating_abstract_loop_is_bounded_and_reported() -> None:
    blocks: dict = {}
    loop = dust_run(blocks, (0, 1, 0), 4)
    inverter = AbstractComponent(
        "osc", "combinational",
        (ComponentPin("a", "in", loop[-1], 1), ComponentPin("y", "out", loop[0], 15)),
        ComponentTiming(uniform_arcs(["a"], ["y"], 4)), truth_tables=(("y", "10"),),
    )  # fmt: skip
    sim = RedstoneSimulator(MinecraftPhysicalDesign(blocks, components=(inverter,)))
    with pytest.raises(SimulationError) as info:
        sim.run_until_stable(max_ticks=200)
    assert info.value.code == "simulation/unstable"
    assert sim.time <= 200 and sim.pending_events > 0


def test_serialized_design_simulates_identically() -> None:
    original = busy_design()
    text = json.dumps(original.to_dict())
    loaded = MinecraftPhysicalDesign.from_dict(json.loads(text))
    assert loaded.fingerprint() == original.fingerprint()
    assert loaded.to_dict() == original.to_dict()
    a, b = RedstoneSimulator(original, trace="full"), RedstoneSimulator(loaded, trace="full")
    assert stimulate(a) == stimulate(b)
    assert a.events == b.events


# -- 20: abstract components -----------------------------------------------------------------


def and_gate_design(delay_a: int = 4, delay_b: int = 6) -> MinecraftPhysicalDesign:
    blocks: dict = {}
    a = dust_run(blocks, (0, 1, 0), 3)
    b = dust_run(blocks, (0, 1, 4), 3)
    y = dust_run(blocks, (5, 1, 2), 3)
    blocks[(4, 1, 2)] = abstract_block()
    blocks[(4, 0, 2)] = stone()
    gate = AbstractComponent(
        "and0", "combinational",
        (ComponentPin("a", "in", a[-1], 1), ComponentPin("b", "in", b[-1], 1), ComponentPin("y", "out", y[0], 15)),
        ComponentTiming((CombinationalArc("a", "y", delay_a, delay_a), CombinationalArc("b", "y", delay_b, delay_b))),
        truth_tables=(("y", "0001"),), voxels=((4, 1, 2),),
    )  # fmt: skip
    return MinecraftPhysicalDesign(
        blocks, components=(gate,),
        ports=(Port("a", "in", (PortBit("source", a[0]),)), Port("b", "in", (PortBit("source", b[0]),)),
               observe("y", y[-1])),
    )  # fmt: skip


def test_abstract_component_mode_is_never_called_block_accurate() -> None:
    d = and_gate_design()
    assert d.mode == ABSTRACT_MODE != BLOCK_MODE
    assert RedstoneSimulator(d).report()["mode"] == "abstract-components"
    assert busy_design().mode == BLOCK_MODE


@pytest.mark.parametrize("a, b", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_abstract_and_gate_truth_table(a: int, b: int) -> None:
    sim = RedstoneSimulator(and_gate_design())
    assert sim.evaluate(a=a, b=b) == {"y": a & b}


def test_abstract_arcs_are_transport_delays_per_input() -> None:
    sim = RedstoneSimulator(and_gate_design(4, 6))
    sim.schedule_input("a", 1, at_tick=10)
    sim.schedule_input("b", 1, at_tick=10)
    assert first_change(sim, "y", 1) == 16  # the slower arc (b, 6 gt) decides
    sim = RedstoneSimulator(and_gate_design(4, 6))
    sim.schedule_input("b", 1, at_tick=0)
    sim.schedule_input("a", 1, at_tick=10)
    assert first_change(sim, "y", 1) == 14


def dff_design(**timing: int) -> MinecraftPhysicalDesign:
    blocks: dict = {}
    d = dust_run(blocks, (0, 1, 0), 2)
    clk = dust_run(blocks, (0, 1, 2), 2)
    rst = dust_run(blocks, (0, 1, 4), 2)
    q = dust_run(blocks, (4, 1, 2), 2)
    seq = SequentialTiming("clk", "d", "q", "rst", timing.get("ctq", 4), timing.get("ctq", 4), timing.get("setup", 2),
                           timing.get("hold", 2), 2, 2)  # fmt: skip
    reg = AbstractComponent(
        "r0", "dff",
        (ComponentPin("d", "in", d[-1], 1), ComponentPin("clk", "in", clk[-1], 1), ComponentPin("rst", "in", rst[-1], 1),
         ComponentPin("q", "out", q[0], 15)),
        ComponentTiming(sequential=seq), init=1,
    )  # fmt: skip
    return MinecraftPhysicalDesign(
        blocks, components=(reg,),
        ports=(Port("d", "in", (PortBit("source", d[0]),)), Port("clk", "in", (PortBit("source", clk[0]),), role="clock"),
               Port("rst", "in", (PortBit("source", rst[0]),), role="reset"), observe("q", q[-1])),
    )  # fmt: skip


def test_abstract_register_captures_on_the_rising_edge_and_resets_asynchronously() -> None:
    sim = RedstoneSimulator(dff_design(), trace="events")
    sim.schedule_input("rst", 1, at_tick=0)
    sim.schedule_input("rst", 0, at_tick=10)
    assert first_change(sim, "q", 1) == 2  # reset -> init 1 after reset->Q
    sim.schedule_input("d", 0, at_tick=20)
    sim.schedule_input("clk", 1, at_tick=30)
    sim.schedule_input("clk", 0, at_tick=40)
    assert first_change(sim, "q", 0) == 34  # clk->Q = 4 gt
    sim.schedule_input("d", 1, at_tick=50)
    sim.schedule_input("clk", 1, at_tick=60)  # falling edges never capture
    sim.run_until_stable()
    assert sim.read_output("q") == 1 and not sim.violations
    assert [(c.edge_gt, c.value) for c in sim.captures] == [(30, 0), (60, 1)]
    assert any(e["type"] == "clock_edge" for e in sim.events)


def test_abstract_register_reports_setup_and_hold_violations() -> None:
    sim = RedstoneSimulator(dff_design(setup=4, hold=4))
    sim.schedule_input("d", 1, at_tick=28)  # 2 gt before the edge: setup 4 violated
    sim.schedule_input("clk", 1, at_tick=30)
    sim.schedule_input("d", 0, at_tick=32)  # 2 gt after the edge: hold 4 violated
    sim.run_until_stable()
    assert [v.kind for v in sim.violations] == ["setup", "hold"]


# -- 21: refusals ---------------------------------------------------------------------------------


def test_unsupported_block_gives_an_explicit_diagnostic() -> None:
    blocks: dict = {(0, 0, 0): stone(), (0, 1, 0): MinecraftBlock.of("minecraft:comparator", facing="west")}
    with pytest.raises(SimulationError) as info:
        RedstoneSimulator(design(blocks))
    assert info.value.code == "simulation/unsupported_block"
    assert "comparator" in str(info.value)


def test_dust_on_air_is_refused_not_simulated() -> None:
    with pytest.raises(SimulationError) as info:
        RedstoneSimulator(design({(0, 5, 0): redstone_wire()}))
    assert info.value.code == "simulation/unsupported_placement"


def test_a_component_pin_off_dust_is_a_binding_error() -> None:
    gate = AbstractComponent(
        "g", "combinational", (ComponentPin("y", "out", (0, 1, 0), 15),), ComponentTiming(), truth_tables=(("y", "1"),),
    )  # fmt: skip
    with pytest.raises(SimulationError) as info:
        RedstoneSimulator(MinecraftPhysicalDesign({(0, 0, 0): stone()}, components=(gate,)))
    assert info.value.code == "simulation/bad_binding"


def test_weak_signal_at_an_abstract_input_is_reported() -> None:
    blocks: dict = {}
    run = dust_run(blocks, (0, 1, 0), 6)
    gate = AbstractComponent(
        "buf", "combinational",
        (ComponentPin("a", "in", run[-1], 12), ComponentPin("y", "out", dust_run(blocks, (0, 1, 3), 1)[0], 15)),
        ComponentTiming(uniform_arcs(["a"], ["y"], 2)), truth_tables=(("y", "01"),),
    )  # fmt: skip
    sim = RedstoneSimulator(MinecraftPhysicalDesign(blocks, components=(gate,), ports=(Port("a", "in", (PortBit("source", run[0]),)),)))
    sim.set_input("a", 1)
    sim.run_until_stable()
    assert [w.code for w in sim.warnings] == ["simulation/weak_input"]
