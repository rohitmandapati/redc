"""The ``physical-primitive`` physical backend: the full pipeline behind
``redc pnr --backend physical-primitive``.

    Graph --synthesize_to_primitives--> PrimitiveNetlist
          --map_primitives_to_minecraft--> PrimitivePhysicalNetlist
          --place_and_route_primitive--> PrimitivePnRResult (+ replay trace)

Every phase reports into ONE :class:`PrimitiveTraceRecorder`, created before
synthesis starts, so the trace captures synthesis and technology mapping too
and survives a failure in any phase (the failing phase is recorded as the
``final.failure.stage``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ir import Graph
from ..parser import CompileError
from ..physical_backends import PnROptions, PnRPaths, PnRRun
from .netlist import PrimitiveNetlist
from .physical import PrimitivePhysicalNetlist
from .pnr.config import PrimitivePnRConfig
from .pnr.design import PHYSICAL_SCHEMA, PrimitivePnRResult, place_and_route_primitive
from .pnr.trace import TRACE_SCHEMA, PrimitiveTraceRecorder
from .synthesis.interface import (
    DEFAULT_INTERFACE_POLICY,
    InterfacePolicy,
    PadInterfacePolicy,
)
from .synthesis.lower import synthesize_to_primitives
from .techmap import map_primitives_to_minecraft
from .technology.library import PRIMITIVE_TECHNOLOGY, PrimitiveTechnologyLibrary

#: ``--interface`` choices: how module ports are realized.
INTERFACE_POLICIES: dict[str, InterfacePolicy] = {
    "default": DEFAULT_INTERFACE_POLICY,
    "pads": PadInterfacePolicy(),
}


def interface_policy(name: str | None) -> InterfacePolicy:
    if name is None:
        return DEFAULT_INTERFACE_POLICY
    try:
        return INTERFACE_POLICIES[name]
    except KeyError:
        raise CompileError(
            f"unknown interface policy {name!r}; use one of {', '.join(INTERFACE_POLICIES)}"
        ) from None


def place_and_route_graph(
    graph: Graph,
    config: PrimitivePnRConfig | None = None,
    *,
    interface: InterfacePolicy = DEFAULT_INTERFACE_POLICY,
    library: PrimitiveTechnologyLibrary = PRIMITIVE_TECHNOLOGY,
    trace: PrimitiveTraceRecorder | None = None,
    source: str | None = None,
    top: str | None = None,
) -> tuple[PrimitiveNetlist, PrimitivePhysicalNetlist, PrimitivePnRResult]:
    """Run the whole primitive pipeline on a validated graph.

    Returns ``(logical netlist, mapped netlist, P&R result)``.  Raises
    :class:`CompileError` only if synthesis or tech-mapping fails (after
    closing the trace with that stage); P&R failure is ``result.success``."""
    config = config or PrimitivePnRConfig()
    recorder = trace if trace is not None else PrimitiveTraceRecorder(config.trace_level)
    graph.validate()
    if recorder.source is None:
        recorder.begin_source(path=source, top=top)
    stage = "synthesis"
    try:
        netlist = synthesize_to_primitives(graph, interface=interface, trace=recorder)
        recorder.ir = netlist.graph_summary
        stage = "techmap"
        mapped = map_primitives_to_minecraft(netlist, library=library, trace=recorder)
        stage = "pnr"
        result = place_and_route_primitive(mapped, config, trace=recorder)
    except CompileError as error:
        recorder.fail(str(error), stage=stage)
        raise
    return netlist, mapped, result


@dataclass
class _PrimitivePrepared:
    graph: Graph
    config: PrimitivePnRConfig
    interface: InterfacePolicy
    recorder: PrimitiveTraceRecorder
    source: str | None
    top: str


class PrimitivePhysicalBackend:
    """Adapter registering the primitive pipeline in :mod:`redc.physical_backends`."""

    name = "physical-primitive"
    summary = (
        "Block-resolution Minecraft backend: every value bit-blasted to one-bit AND/OR/XOR/NOT gates, "
        "one routed net per bit, redstone legalization (redc.physical_primitive)."
    )
    trace_schema = TRACE_SCHEMA
    design_schema = PHYSICAL_SCHEMA
    netlist_stages: tuple[str, ...] = ("primitive", "mapped")
    options = frozenset(
        {"spacing", "routing_margin", "max_route_iterations", "max_attempts", "max_height", "channel_width", "interface"}
    )

    def default_paths(self, stem: str) -> PnRPaths:
        return PnRPaths(
            Path("build") / f"{stem}.primitive.physical.json", Path("build") / f"{stem}.primitive.pnr.json"
        )

    def viewer_path(self, trace_path: Path) -> Path:
        """``<stem>.primitive.pnr.html`` next to the trace: strips a trailing
        ``.json`` / ``.jsonl`` and then ``.pnr`` / ``.primitive``, so
        ``inv.v2.primitive.pnr.json`` -> ``inv.v2.primitive.pnr.html``."""
        name = trace_path.name
        for suffix in (".jsonl", ".json", ".pnr", ".primitive"):
            name = name.removesuffix(suffix)
        return trace_path.with_name(name + ".primitive.pnr.html")

    def dump_netlist(self, graph: Graph, *, stage: str | None = None, interface: str | None = None) -> dict[str, Any]:
        stage = stage or "primitive"
        if stage not in self.netlist_stages:
            raise CompileError(
                f"the physical-primitive backend has no netlist stage {stage!r} "
                f"(use {' or '.join(self.netlist_stages)})"
            )
        netlist = synthesize_to_primitives(graph, interface=interface_policy(interface))
        if stage == "primitive":
            return netlist.to_dict()
        return map_primitives_to_minecraft(netlist).to_dict()

    def prepare(self, graph: Graph, options: PnROptions, *, source: Path | None, top: str) -> _PrimitivePrepared:
        defaults = PrimitivePnRConfig()
        config = PrimitivePnRConfig(
            max_y=defaults.max_y if options.max_height is None else options.max_height,
            component_spacing=defaults.component_spacing if options.spacing is None else options.spacing,
            channel_width=defaults.channel_width if options.channel_width is None else options.channel_width,
            routing_margin=defaults.routing_margin if options.routing_margin is None else options.routing_margin,
            max_routing_iterations=(
                defaults.max_routing_iterations
                if options.max_route_iterations is None
                else options.max_route_iterations
            ),
            max_pnr_attempts=defaults.max_pnr_attempts if options.max_attempts is None else options.max_attempts,
            trace_level=options.trace_level,  # type: ignore[arg-type]  # parsed in __post_init__
        )
        recorder = PrimitiveTraceRecorder(config.trace_level)
        path = None if source is None else source.as_posix()
        recorder.begin_source(path=path, top=top)
        return _PrimitivePrepared(graph, config, interface_policy(options.interface), recorder, path, top)

    def run(self, prepared: _PrimitivePrepared) -> PnRRun:
        _netlist, _mapped, result = place_and_route_graph(
            prepared.graph,
            prepared.config,
            interface=prepared.interface,
            trace=prepared.recorder,
            source=prepared.source,
            top=prepared.top,
        )
        if not result.success:
            failure = result.failure
            return PnRRun(False, result.attempts, failure.message if failure else "unknown failure", None, "")
        m = result.metrics
        bounds = m["final"]["bounds"]
        summary = (
            f"P&R: {m['techmap']['components']} cells ({m['logical']['primitive_gates']} gates, "
            f"{m['logical']['register_bits']} register bits), {m['routing']['routed_nets']} one-bit nets, "
            f"{m['routing']['dust_blocks']} dust + {m['routing']['repeaters']} repeaters, "
            f"{m['routing']['iterations']} routing iteration(s), {m['routing']['rip_ups']} rip-up(s), "
            f"attempt {result.geometry.attempt}, bounds {bounds['dims'] if bounds else '-'} blocks"
        )
        return PnRRun(True, result.attempts, None, result.to_design_dict(), summary)

    def write_viewer(self, trace: Mapping[str, Any], path: Path, *, title: str) -> Path:
        from ..viewer import write_trace_html

        if trace.get("design") is None:
            stage = ((trace.get("final") or {}).get("failure") or {}).get("stage", "an early stage")
            raise CompileError(f"no design to show: the run failed during {stage} (the trace was written)")
        return write_trace_html(trace, path, title=title)


__all__ = ["INTERFACE_POLICIES", "PrimitivePhysicalBackend", "interface_policy", "place_and_route_graph"]
