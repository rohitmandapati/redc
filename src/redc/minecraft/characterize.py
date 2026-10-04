"""Characterize a block-level technology cell by SIMULATING it.

Given a cell's blocks and its pin endpoints, :func:`characterize_cell` builds a
small test harness around it, exactly the way a route meets the cell in a
real design:

* every pin gets its endpoint dust plus one stub dust block on its approach
  side (so the pin dust has the same shape -- a straight line along the pin
  facing -- that a routed net gives it);
* every input is driven through its stub at the cell's MINIMUM required
  strength (stub ``threshold + 1`` -> pin ``threshold``), so a cell that only
  works with full-strength inputs fails characterization;
* every output pin's dust is observed.

It then measures:

* the truth table (every input combination, settled);
* every single-input transition (each input, each state of the others, both
  directions): how long until each output's LAST change (and its first),
  whether the output glitched, and the output strength;

and derives conservative :class:`~redc.minecraft.timing.CombinationalArc` s:
``min`` = the earliest output change seen for that input, ``max`` = the
latest settle.  :meth:`Characterization.check` compares the result with a
cell's DECLARED function and timing, so a technology library can carry
cells whose behaviour was proven by the simulator, not hand-entered.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .blocks import Coord, MinecraftBlock, offset, redstone_wire, stone
from .connectivity import SimulationError
from .design import MinecraftPhysicalDesign, Port, PortBit
from .simulator import RedstoneSimulator
from .timing import CombinationalArc, ComponentTiming
from .units import ticks_record


@dataclass(frozen=True)
class CellPin:
    """A pin endpoint in the cell's coordinates.  ``facing`` points AWAY from
    the cell (where the route comes from / goes to); ``strength`` is the
    minimum an input needs, or the strength an output promises."""

    name: str
    direction: str
    coord: Coord
    facing: str
    strength: int


@dataclass(frozen=True)
class Transition:
    input: str
    output: str
    others: tuple[tuple[str, int], ...]
    input_edge: str  # "rise" | "fall"
    output_edge: str | None  # None: the output did not change
    first_gt: int | None
    last_gt: int | None
    changes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "output": self.output,
            "others": dict(self.others),
            "input_edge": self.input_edge,
            "output_edge": self.output_edge,
            "first": ticks_record(self.first_gt),
            "last": ticks_record(self.last_gt),
            "changes": self.changes,
        }


@dataclass
class Characterization:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    truth_tables: dict[str, str]
    transitions: list[Transition]
    min_output_strength: dict[str, int | None]
    timing: ComponentTiming
    diagnostics: list[str] = field(default_factory=list)

    def check(
        self,
        truth_tables: Mapping[str, str],
        timing: ComponentTiming | None = None,
        output_strength: Mapping[str, int] | None = None,
    ) -> list[str]:
        """Every way the measured behaviour contradicts a declaration (empty = verified).

        A declared arc passes if it is CONSERVATIVE: declared min <= measured
        min and declared max >= measured max."""
        problems = list(self.diagnostics)
        for out, table in truth_tables.items():
            got = self.truth_tables.get(out)
            if got != table:
                problems.append(f"{self.name}: output {out!r} truth table is {got}, declared {table}")
        if timing is not None:
            measured = {(a.from_pin, a.to_pin): a for a in self.timing.arcs}
            for arc in timing.arcs:
                seen = measured.get((arc.from_pin, arc.to_pin))
                if seen is None:
                    continue
                if arc.min_gt > seen.min_gt or arc.max_gt < seen.max_gt:
                    problems.append(
                        f"{self.name}: arc {arc.from_pin}->{arc.to_pin} declared [{arc.min_gt}, {arc.max_gt}] gt "
                        f"but measured [{seen.min_gt}, {seen.max_gt}] gt"
                    )
        for out, promised in (output_strength or {}).items():
            low = self.min_output_strength.get(out)
            if low is not None and low < promised:
                problems.append(f"{self.name}: output {out!r} reaches only strength {low} < promised {promised}")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "truth_tables": dict(self.truth_tables),
            "min_output_strength": dict(self.min_output_strength),
            "timing": self.timing.to_dict(),
            "transitions": [t.to_dict() for t in self.transitions],
            "diagnostics": list(self.diagnostics),
        }


def harness(
    blocks: Mapping[Coord, MinecraftBlock], pins: Sequence[CellPin]
) -> tuple[MinecraftPhysicalDesign, dict[str, Coord]]:
    """The test bench world: the cell plus pin dust, stubs and their supports.
    Returns the design and each output's observed coordinate."""
    world = dict(blocks)
    ports: list[Port] = []
    observed: dict[str, Coord] = {}

    def lay(coord: Coord) -> None:
        below = offset(coord, "down")
        if below not in world:
            world[below] = stone()
        world[coord] = redstone_wire()

    for pin in pins:
        stub = offset(pin.coord, pin.facing)
        lay(pin.coord)
        lay(stub)
        if pin.direction == "in":
            drive = min(15, pin.strength + 1)
            ports.append(Port(pin.name, "in", (PortBit("source", stub, drive),)))
        else:
            ports.append(Port(pin.name, "out", (PortBit("dust", pin.coord, 1),)))
            observed[pin.name] = pin.coord
    return MinecraftPhysicalDesign(world, ports=tuple(ports), source_backend="characterization"), observed


def characterize_cell(
    name: str,
    blocks: Mapping[Coord, MinecraftBlock],
    pins: Sequence[CellPin],
    *,
    max_ticks: int = 2_000,
) -> Characterization:
    """Simulate ``blocks`` (see the module docstring)."""
    design, observed = harness(blocks, pins)
    inputs = tuple(p.name for p in pins if p.direction == "in")
    outputs = tuple(p.name for p in pins if p.direction == "out")
    diagnostics: list[str] = []
    tables = {o: "" for o in outputs}
    strength: dict[str, int | None] = {o: None for o in outputs}
    try:
        settled: dict[tuple[int, ...], dict[str, int]] = {}
        for code in range(1 << len(inputs)):
            values = tuple((code >> i) & 1 for i in range(len(inputs)))
            sim = RedstoneSimulator(design)
            sim.evaluate(max_ticks=max_ticks, **dict(zip(inputs, values)))
            result = {o: sim.read_output(o) for o in outputs}
            settled[values] = result
            for o in outputs:
                tables[o] += str(result[o])
                if result[o]:
                    level = sim.dust_at(observed[o])
                    current = strength[o]
                    strength[o] = level if current is None else min(current, level)
            if sim.warnings:
                diagnostics.extend(w.message for w in sim.warnings)
        transitions: list[Transition] = []
        for i, pin in enumerate(inputs):
            for others in itertools.product((0, 1), repeat=len(inputs) - 1):
                base = list(others[:i]) + [0] + list(others[i:])
                for start_bit in (0, 1):
                    start = list(base)
                    start[i] = start_bit
                    sim = RedstoneSimulator(design)
                    sim.evaluate(max_ticks=max_ticks, **dict(zip(inputs, start)))
                    before = {o: sim.read_output(o) for o in outputs}
                    t0 = sim.time
                    sim.set_input(pin, 1 - start_bit)
                    history: dict[str, list[int]] = {o: [] for o in outputs}
                    last = dict(before)
                    while sim.step() is not None:
                        if sim.time - t0 > max_ticks:
                            raise SimulationError("simulation/unstable", f"{name} does not settle")
                        for o in outputs:
                            value = sim.read_output(o)
                            if value != last[o]:
                                history[o].append(sim.time - t0)
                                last[o] = value
                    for o in outputs:
                        changes = history[o]
                        final = last[o]
                        edge = None if final == before[o] else ("rise" if final else "fall")
                        transitions.append(
                            Transition(
                                pin, o, tuple((n, v) for n, v in zip(inputs, start) if n != pin),
                                "rise" if start_bit == 0 else "fall", edge,
                                changes[0] if changes else None, changes[-1] if changes else None, len(changes),
                            )
                        )  # fmt: skip
    except SimulationError as error:
        diagnostics.append(str(error))
        return Characterization(name, inputs, outputs, tables, [], strength, ComponentTiming(source="characterized"),
                                diagnostics)  # fmt: skip
    arcs = []
    for pin in inputs:
        for o in outputs:
            seen = [t for t in transitions if t.input == pin and t.output == o and t.first_gt is not None]
            if not seen:
                continue
            low = min(t.first_gt for t in seen if t.first_gt is not None)
            high = max(t.last_gt for t in seen if t.last_gt is not None)
            arcs.append(CombinationalArc(pin, o, low, high))
    return Characterization(
        name, inputs, outputs, tables, transitions, strength, ComponentTiming(tuple(arcs), source="characterized"),
        diagnostics,
    )  # fmt: skip


__all__ = ["CellPin", "Characterization", "Transition", "characterize_cell", "harness"]
