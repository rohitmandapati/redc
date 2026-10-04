"""A compact recording of a short redstone simulation, for viewer playback.

The recording is what a viewer needs to ANIMATE real signal propagation over
the placed-and-routed blocks:

* ``initial_dust`` / ``initial_devices`` -- the settled state at ``start_gt``
  (``[x, y, z, strength]`` / ``[x, y, z, on]``);
* ``dust`` -- every later dust strength change, ``[t, x, y, z, strength]``;
* ``devices`` -- every repeater / torch / lamp / lever change, ``[t, x, y, z, on]``;
* ``marks`` -- notable moments: input changes, clock edges, register clock
  edges and captures.

Sequential designs are recorded from just before the first clock edge (after
reset) through ``cycles`` clock periods; combinational designs from one input
change until the outputs settle.  The recording is capped (``truncated``
says so) to keep traces small; it is a visualization aid, never a
verification input.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .closure import TimingClosure
from .connectivity import CompiledWorld
from .design import MinecraftPhysicalDesign
from .simulator import SIMULATION_MODEL, RedstoneSimulator
from .units import round_up_to_rt, rt_to_gt

MAX_DUST_CHANGES = 60_000
MARK_TYPES = ("input_changed", "clock_edge", "register_clock_edge", "register_capture", "timing_violation")
DEVICE_TYPES = {
    "repeater_output_changed": "powered",
    "torch_changed": "lit",
    "lamp_changed": "lit",
}


def _snapshot(sim: RedstoneSimulator) -> tuple[list[list[int]], list[list[int]]]:
    world = sim.world
    dust = [[*world.dust_coords[d], p] for d, p in enumerate(sim.dust_power) if p > 0]
    devices = [
        [*dev.coord, int(sim.device_state[dev.index])]
        for dev in world.devices
        if dev.behavior is not None and dev.kind in ("repeater", "torch", "lamp", "lever") and sim.device_state[dev.index]
    ]
    return dust, devices


def record_playback(
    design: MinecraftPhysicalDesign,
    closure: TimingClosure,
    *,
    world: CompiledWorld | None = None,
    inputs: Mapping[str, int] | None = None,
    cycles: int = 2,
) -> dict[str, Any]:
    """Record a short simulation of ``design`` (see the module docstring)."""
    sim = RedstoneSimulator(design, world=world, trace="full")
    clock = design.role_port("clock")
    reset = design.role_port("reset")
    values = dict(inputs or {})
    if reset is not None:
        sim.set_input(reset.name, 1)
    for port in design.inputs:
        if port.role == "data":
            sim.set_input(port.name, values.get(port.name, 1 if port.name == "start" else 0))
    sim.run_until_stable()
    if clock is not None and closure.period_gt is not None and closure.high_gt is not None:
        release = round_up_to_rt(sim.time) + rt_to_gt(1)
        if reset is not None:
            sim.schedule_input(reset.name, 0, at_tick=release)
        first = release + closure.recovery_gap_gt
        start = first - rt_to_gt(1)
        end = first + cycles * closure.period_gt
        sim.run_until(start)
        for k in range(cycles):
            edge = first + k * closure.period_gt
            sim.schedule_input(clock.name, 1, at_tick=edge)
            sim.schedule_input(clock.name, 0, at_tick=edge + closure.high_gt)
            if k == 0 and "start" in {p.name for p in design.inputs}:
                sim.schedule_input("start", 0, at_tick=edge + closure.input_offset_gt)
    else:
        start = sim.time
        for port in design.inputs:
            if port.role == "data":
                sim.schedule_input(port.name, (1 << port.width) - 1, at_tick=start + rt_to_gt(1))
        end = start
    initial_dust, initial_devices = _snapshot(sim)
    sim.events.clear()
    if clock is not None and closure.period_gt is not None:
        sim.run_until(end)
    else:
        end = sim.run_until_stable()
    dust: list[list[int]] = []
    devices: list[list[int]] = []
    marks: list[dict[str, Any]] = []
    truncated = False
    for event in sim.events:
        kind = event["type"]
        if kind == "wire_strength_changed":
            for x, y, z, s in event["changes"]:
                if len(dust) >= MAX_DUST_CHANGES:
                    truncated = True
                    break
                dust.append([event["t"], x, y, z, s])
        elif kind in DEVICE_TYPES:
            devices.append([event["t"], *event["coord"], int(bool(event[DEVICE_TYPES[kind]]))])
        elif kind in MARK_TYPES and len(marks) < 2000:
            marks.append({k: v for k, v in event.items() if k in ("t", "type", "port", "bit", "value", "edge", "component")})
    return {
        "model": SIMULATION_MODEL,
        "mode": design.mode,
        "start_gt": start,
        "end_gt": end,
        "cycles": cycles if clock is not None else 0,
        "truncated": truncated,
        "initial_dust": initial_dust,
        "initial_devices": initial_devices,
        "dust": dust,
        "devices": devices,
        "marks": marks,
    }


__all__ = ["MAX_DUST_CHANGES", "record_playback"]
