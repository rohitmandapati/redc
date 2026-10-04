"""``MinecraftPhysicalDesign``: a materialized Minecraft circuit, backend-neutral.

Any RedC Minecraft backend lowers into this one representation; the redstone
simulator (:mod:`redc.minecraft.simulator`), static timing analysis
(:mod:`redc.minecraft.sta`) and future exporters (NBT, schematics) consume it.
It knows only Minecraft-level facts:

* ``blocks`` -- coordinate -> :class:`~redc.minecraft.blocks.MinecraftBlock`
  (air is implicit);
* ``components`` -- :class:`AbstractComponent` black boxes standing in for
  cells that have NO block-level implementation yet.  A component declares
  its Boolean function (truth tables) and timing arcs; its pins are block
  coordinates where it drives / reads redstone dust.  Its voxels are
  ``redc:abstract_block`` markers.  A design containing any component (or
  any externally injected ``source`` port) is simulated in
  ``abstract-components`` mode, never as Minecraft-accurate;
* ``ports`` -- the named, multi-bit interface: each bit is a lever the test
  bench flips (``lever``), an externally injected source (``source``,
  abstract), a dust block whose strength is read (``dust``), a lamp
  (``lamp``) or ``open`` -- an input bit the circuit never uses (nothing is
  routed from it).  A port's ``role`` is ``data``, ``clock`` or ``reset``;
* ``probes`` -- named observation points (e.g. one register bit's output) for
  verification; reading them never changes the circuit;
* ``annotations`` -- per-block labels from the producing backend (net ids,
  instance ids).  They are for REPORTS ONLY: no simulator or timing decision
  ever looks at them -- connectivity always comes from the block geometry.

Nothing here names RedC IR operations, primitive kinds or backend classes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..parser import CompileError
from .blocks import ABSTRACT_ID, Coord, MinecraftBlock, abstract_block
from .timing import TIMING_MODEL, ComponentTiming

DESIGN_SCHEMA = "redc.minecraft-design.v1"
#: The Java Edition redstone the model follows (a documented SUBSET of it).
MINECRAFT_VERSION = "java-1.20 (redc-redstone-sim-v1 subset)"

ABSTRACT_MODE = "abstract-components"
BLOCK_MODE = "block-accurate"

PORT_BIT_KINDS = {"in": ("lever", "source", "open"), "out": ("dust", "lamp")}
PORT_ROLES = ("data", "clock", "reset")


def _coord(value: Iterable[Any]) -> Coord:
    x, y, z = value
    return (int(x), int(y), int(z))


@dataclass(frozen=True, slots=True)
class ComponentPin:
    """One pin of an abstract component, at a block coordinate.

    ``direction="out"``: the component injects ``strength`` into the dust at
    ``coord`` while the output is 1.  ``direction="in"``: the component reads
    the dust at ``coord``; it sees 1 iff the strength is ``>= strength``."""

    name: str
    direction: str
    coord: Coord
    strength: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "direction": self.direction, "coord": list(self.coord), "strength": self.strength}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ComponentPin:
        return cls(str(data["name"]), str(data["direction"]), _coord(data["coord"]), int(data["strength"]))


@dataclass(frozen=True)
class AbstractComponent:
    """A black box with declared behaviour (ABSTRACT SIMULATION only).

    ``function``:

    * ``combinational`` -- each output's ``truth_tables`` entry is a string of
      ``2**n`` bits over the input pins in pin order: character ``k`` is the
      output for inputs where input ``i`` is bit ``i`` of ``k``.  Delays come
      from ``timing.arcs`` (transport delay: the output at time ``T`` is the
      function of each input's value at ``T - delay(input -> output)``);
    * ``dff`` -- a rising-edge register: pins named by ``timing.sequential``
      (data, clock, optional asynchronous active-high reset, q), reset value
      ``init``.
    """

    name: str
    function: str
    pins: tuple[ComponentPin, ...]
    timing: ComponentTiming
    truth_tables: tuple[tuple[str, str], ...] = ()
    init: int = 0
    voxels: tuple[Coord, ...] = ()
    labels: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        where = f"abstract component {self.name!r}"
        names = [p.name for p in self.pins]
        if len(set(names)) != len(names):
            raise CompileError(f"{where}: duplicate pin names {names}")
        for pin in self.pins:
            if pin.direction not in ("in", "out"):
                raise CompileError(f"{where}: pin {pin.name!r} direction must be 'in' or 'out'")
            low = 1 if pin.direction == "in" else 0
            if not low <= pin.strength <= 15:
                raise CompileError(f"{where}: pin {pin.name!r} strength {pin.strength} out of range")
        ins, outs = self.input_names, self.output_names
        if self.function == "combinational":
            if self.timing.sequential is not None:
                raise CompileError(f"{where}: a combinational component has no sequential timing")
            tables = dict(self.truth_tables)
            if set(tables) != set(outs) or len(tables) != len(self.truth_tables):
                raise CompileError(f"{where}: needs exactly one truth table per output pin {list(outs)}")
            for out, table in tables.items():
                if len(table) != 1 << len(ins) or set(table) - {"0", "1"}:
                    raise CompileError(f"{where}: truth table of {out!r} must be {1 << len(ins)} bits")
            for out in outs:
                for name in ins:
                    arcs = [a for a in self.timing.arcs if a.from_pin == name and a.to_pin == out]
                    if len(arcs) != 1:
                        raise CompileError(f"{where}: needs exactly one timing arc {name} -> {out}")
                    if arcs[0].min_gt < 1:
                        raise CompileError(f"{where}: arc {name} -> {out} must delay at least one game tick")
            known = {(i, o) for i in ins for o in outs}
            if any((a.from_pin, a.to_pin) not in known for a in self.timing.arcs):
                raise CompileError(f"{where}: timing arc between unknown pins")
        elif self.function == "dff":
            seq = self.timing.sequential
            if seq is None or self.timing.arcs:
                raise CompileError(f"{where}: a dff needs sequential timing and no combinational arcs")
            want_in = {seq.data_pin, seq.clock_pin} | ({seq.reset_pin} if seq.reset_pin else set())
            if set(ins) != want_in or tuple(outs) != (seq.q_pin,):
                raise CompileError(f"{where}: dff pins must be inputs {sorted(want_in)} and output {seq.q_pin!r}")
            if self.init not in (0, 1):
                raise CompileError(f"{where}: init must be 0 or 1")
        else:
            raise CompileError(f"{where}: unknown function {self.function!r} (combinational or dff)")

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.pins if p.direction == "in")

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.pins if p.direction == "out")

    def pin(self, name: str) -> ComponentPin:
        for pin in self.pins:
            if pin.name == name:
                return pin
        raise CompileError(f"abstract component {self.name!r} has no pin {name!r}")

    def table(self, output: str) -> str:
        return dict(self.truth_tables)[output]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "function": self.function,
            "simulation": ABSTRACT_MODE,
            "pins": [p.to_dict() for p in self.pins],
            "truth_tables": dict(self.truth_tables),
            "init": self.init,
            "timing": self.timing.to_dict(),
            "voxels": [list(c) for c in self.voxels],
            "labels": dict(self.labels),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AbstractComponent:
        return cls(
            name=str(data["name"]),
            function=str(data["function"]),
            pins=tuple(ComponentPin.from_dict(p) for p in data["pins"]),
            timing=ComponentTiming.from_dict(data["timing"]),
            truth_tables=tuple(sorted((str(k), str(v)) for k, v in dict(data.get("truth_tables") or {}).items())),
            init=int(data.get("init", 0)),
            voxels=tuple(_coord(c) for c in data.get("voxels", ())),
            labels=tuple(sorted(dict(data.get("labels") or {}).items())),
        )


@dataclass(frozen=True, slots=True)
class PortBit:
    """How one bit of a port touches the world (see the module docstring)."""

    kind: str
    coord: Coord
    #: ``source``: the strength injected; ``dust``: the threshold read as 1.
    #: Default: 15 for a source, 1 for observed dust (unused for levers / lamps).
    strength: int = 0

    def __post_init__(self) -> None:
        if self.strength == 0:
            object.__setattr__(self, "strength", 1 if self.kind == "dust" else 15)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "coord": list(self.coord), "strength": self.strength}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PortBit:
        return cls(str(data["kind"]), _coord(data["coord"]), int(data.get("strength", 0)))


@dataclass(frozen=True)
class Port:
    """A named interface value, ``bits`` LSB first."""

    name: str
    direction: str
    bits: tuple[PortBit, ...]
    signed: bool = False
    role: str = "data"

    def __post_init__(self) -> None:
        where = f"port {self.name!r}"
        if self.direction not in PORT_BIT_KINDS:
            raise CompileError(f"{where}: direction must be 'in' or 'out'")
        if not self.bits:
            raise CompileError(f"{where}: needs at least one bit")
        if self.role not in PORT_ROLES:
            raise CompileError(f"{where}: role must be one of {PORT_ROLES}")
        if self.role != "data" and (self.direction != "in" or len(self.bits) != 1):
            raise CompileError(f"{where}: a {self.role} port is a one-bit input")
        for bit in self.bits:
            if bit.kind not in PORT_BIT_KINDS[self.direction]:
                raise CompileError(f"{where}: an {self.direction} bit is one of {PORT_BIT_KINDS[self.direction]}")
            if not 1 <= bit.strength <= 15:
                raise CompileError(f"{where}: bit strength {bit.strength} out of range")

    @property
    def width(self) -> int:
        return len(self.bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "role": self.role,
            "width": self.width,
            "signed": self.signed,
            "bits": [b.to_dict() for b in self.bits],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Port:
        return cls(
            str(data["name"]),
            str(data["direction"]),
            tuple(PortBit.from_dict(b) for b in data["bits"]),
            bool(data.get("signed", False)),
            str(data.get("role", "data")),
        )


@dataclass(frozen=True, slots=True)
class Probe:
    """A named, non-intrusive observation of one dust block (1 iff strength >= threshold)."""

    name: str
    coord: Coord
    threshold: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "coord": list(self.coord), "threshold": self.threshold}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Probe:
        return cls(str(data["name"]), _coord(data["coord"]), int(data.get("threshold", 1)))


@dataclass
class MinecraftPhysicalDesign:
    """A materialized Minecraft circuit (see the module docstring)."""

    blocks: dict[Coord, MinecraftBlock]
    components: tuple[AbstractComponent, ...] = ()
    ports: tuple[Port, ...] = ()
    probes: tuple[Probe, ...] = ()
    annotations: dict[Coord, Any] = field(default_factory=dict)
    source_backend: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    timing_model: str = TIMING_MODEL
    minecraft_version: str = MINECRAFT_VERSION

    def __post_init__(self) -> None:
        self.blocks = {c: b for c, b in self.blocks.items() if b.id != "minecraft:air"}
        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            raise CompileError("abstract component names must be unique")
        ports = [p.name for p in self.ports]
        if len(set(ports)) != len(ports):
            raise CompileError(f"port names must be unique, got {ports}")
        probes = [p.name for p in self.probes]
        if len(set(probes)) != len(probes):
            raise CompileError("probe names must be unique")
        for role in ("clock", "reset"):
            if sum(1 for p in self.ports if p.role == role) > 1:
                raise CompileError(f"at most one {role} port")
        for comp in self.components:
            for voxel in comp.voxels:
                block = self.blocks.get(voxel)
                if block is None or block.id != ABSTRACT_ID:
                    raise CompileError(
                        f"abstract component {comp.name!r} voxel {list(voxel)} must be a {ABSTRACT_ID} block"
                    )

    # -- queries ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """``block-accurate`` only if nothing in the design is abstract."""
        if self.components or any(b.id.startswith("redc:") for b in self.blocks.values()):
            return ABSTRACT_MODE
        if any(bit.kind == "source" for p in self.ports for bit in p.bits):
            return ABSTRACT_MODE
        return BLOCK_MODE

    def port(self, name: str) -> Port:
        for port in self.ports:
            if port.name == name:
                return port
        raise CompileError(f"design has no port {name!r} (ports: {', '.join(p.name for p in self.ports)})")

    def role_port(self, role: str) -> Port | None:
        return next((p for p in self.ports if p.role == role), None)

    @property
    def inputs(self) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.direction == "in")

    @property
    def outputs(self) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.direction == "out")

    def block_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for block in self.blocks.values():
            counts[block.id] = counts.get(block.id, 0) + 1
        return dict(sorted(counts.items()))

    # -- serialization ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        palette: dict[MinecraftBlock, int] = {}
        rows = []
        for coord in sorted(self.blocks):
            block = self.blocks[coord]
            index = palette.setdefault(block, len(palette))
            rows.append([*coord, index])
        return {
            "schema": DESIGN_SCHEMA,
            "source_backend": self.source_backend,
            "minecraft_version": self.minecraft_version,
            "timing_model": self.timing_model,
            "mode": self.mode,
            "coordinates": "integer [x, y, z] = one block; x east, y up, z south; air is implicit",
            "palette": [b.to_dict() for b in palette],
            "blocks": rows,
            "block_counts": self.block_counts(),
            "components": [c.to_dict() for c in self.components],
            "ports": [p.to_dict() for p in self.ports],
            "probes": [p.to_dict() for p in self.probes],
            "annotations": [[*c, self.annotations[c]] for c in sorted(self.annotations)],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MinecraftPhysicalDesign:
        if data.get("schema") != DESIGN_SCHEMA:
            raise CompileError(f"not a {DESIGN_SCHEMA} design (schema {data.get('schema')!r})")
        palette = [MinecraftBlock.from_dict(b) for b in data["palette"]]
        blocks = {(int(x), int(y), int(z)): palette[int(i)] for x, y, z, i in data["blocks"]}
        return cls(
            blocks=blocks,
            components=tuple(AbstractComponent.from_dict(c) for c in data.get("components", ())),
            ports=tuple(Port.from_dict(p) for p in data.get("ports", ())),
            probes=tuple(Probe.from_dict(p) for p in data.get("probes", ())),
            annotations={(int(r[0]), int(r[1]), int(r[2])): r[3] for r in data.get("annotations", ())},
            source_backend=str(data.get("source_backend", "unknown")),
            metadata=dict(data.get("metadata") or {}),
            timing_model=str(data.get("timing_model", TIMING_MODEL)),
            minecraft_version=str(data.get("minecraft_version", MINECRAFT_VERSION)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def fingerprint(self) -> str:
        """SHA-256 over the physical content (blocks, components, ports,
        probes) -- NOT metadata or annotations.  A timing report or simulation
        result records the fingerprint of the world it analysed, so a stale
        report (computed before a route changed) is detectable."""
        record = self.to_dict()
        physical = {k: record[k] for k in ("palette", "blocks", "components", "ports", "probes")}
        text = json.dumps(physical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def add_component(
    blocks: dict[Coord, MinecraftBlock], component: AbstractComponent
) -> AbstractComponent:
    """Mark ``component``'s voxels as abstract blocks in ``blocks`` (helper for producers)."""
    for voxel in component.voxels:
        if voxel in blocks and blocks[voxel].id != ABSTRACT_ID:
            raise CompileError(f"abstract component {component.name!r} voxel {list(voxel)} is already {blocks[voxel]}")
        blocks[voxel] = abstract_block()
    return component


__all__ = [
    "ABSTRACT_MODE",
    "BLOCK_MODE",
    "DESIGN_SCHEMA",
    "MINECRAFT_VERSION",
    "AbstractComponent",
    "ComponentPin",
    "MinecraftPhysicalDesign",
    "Port",
    "PortBit",
    "Probe",
    "add_component",
]
