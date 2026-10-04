"""Physical (place-and-route) backends, selectable with ``redc pnr --backend``.

This registry is deliberately separate from :mod:`redc.backends`.  An *output*
backend (``redc build --backend systemverilog``) is one function
``Graph -> text artifact``.  A *physical* backend is a multi-phase pipeline
with several artifacts -- lowering, placement, routing, a replay trace, a
viewer and a final design file -- and its own configuration knobs:

* ``physical``           -- the coarse Minecraft backend (:mod:`redc.physical`):
  wide components, bus-valued nets, abstract grid cells (the default);
* ``physical-primitive`` -- every value bit-blasted to one-bit AND/OR/XOR/NOT
  gates and register bits, routed one bit per net at block resolution with
  redstone legalization (:mod:`redc.physical_primitive`).

A backend implements :class:`PhysicalBackend`.  The CLI drives every backend the
same way: ``prepare`` (errors here exit before any trace exists), ``run``
inside ``try/finally`` so the replay trace is ALWAYS written -- also when P&R
fails or raises -- then the viewer and, on success, the final design file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol

from .ir import Graph
from .parser import CompileError

DEFAULT_PHYSICAL_BACKEND = "physical"


@dataclass(frozen=True)
class PnROptions:
    """P&R knobs as the CLI received them.  ``None`` = the backend's default;
    an option a backend does not understand is an error, never ignored."""

    trace_level: str = "basic"
    spacing: int | None = None
    routing_margin: int | None = None
    max_route_iterations: int | None = None
    max_attempts: int | None = None
    # coarse ``physical`` only
    grid_height: int | None = None
    layer_gap: int | None = None
    # ``physical-primitive`` only
    max_height: int | None = None
    channel_width: int | None = None
    interface: str | None = None

    def given(self) -> dict[str, Any]:
        """Every explicitly set option except the trace level."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "trace_level" and getattr(self, f.name) is not None
        }


@dataclass(frozen=True)
class PnRPaths:
    output: Path
    trace: Path


@dataclass
class PnRRun:
    """What one P&R run produced, in CLI terms."""

    success: bool
    attempts: int
    failure: str | None
    design: dict[str, Any] | None
    summary: str


class TraceWriter(Protocol):
    """The part of a trace recorder the CLI needs."""

    events: list[dict[str, Any]]

    def fail(self, message: str) -> None: ...

    def write(self, path: str | Path) -> Path: ...

    def to_dict(self) -> dict[str, Any]: ...


class PreparedPnR(Protocol):
    @property
    def recorder(self) -> TraceWriter: ...


class PhysicalBackend(Protocol):
    """A selectable place-and-route pipeline (see the module docstring)."""

    name: str
    summary: str
    trace_schema: str
    design_schema: str
    netlist_stages: tuple[str, ...]
    #: :class:`PnROptions` field names this backend accepts.
    options: frozenset[str]

    def default_paths(self, stem: str) -> PnRPaths: ...

    def viewer_path(self, trace_path: Path) -> Path: ...

    def dump_netlist(self, graph: Graph, *, stage: str | None = None, interface: str | None = None) -> dict[str, Any]: ...

    def prepare(self, graph: Graph, options: PnROptions, *, source: Path | None, top: str) -> PreparedPnR: ...

    def run(self, prepared: Any) -> PnRRun: ...

    def write_viewer(self, trace: Mapping[str, Any], path: Path, *, title: str) -> Path: ...


#: CLI spelling of each option, for error messages.
OPTION_FLAGS = {
    "spacing": "--spacing",
    "routing_margin": "--routing-margin",
    "max_route_iterations": "--max-route-iterations",
    "max_attempts": "--max-attempts",
    "grid_height": "--grid-height",
    "layer_gap": "--layer-gap",
    "max_height": "--max-height",
    "channel_width": "--channel-width",
    "interface": "--interface",
}


def check_options(backend: PhysicalBackend, options: PnROptions) -> None:
    """Reject options the selected backend does not understand."""
    unsupported = sorted(set(options.given()) - backend.options)
    if unsupported:
        flags = ", ".join(OPTION_FLAGS.get(name, name) for name in unsupported)
        owners = []
        for name in unsupported:
            others = [b.name for b in _registry().values() if name in b.options and b.name != backend.name]
            if others:
                owners.append(f"{OPTION_FLAGS.get(name, name)} belongs to {', '.join(others)}")
        hint = f" ({'; '.join(owners)})" if owners else ""
        raise CompileError(f"the {backend.name} backend does not support {flags}{hint}")


@dataclass
class _CoarsePrepared:
    config: Any
    netlist: Any
    recorder: Any


class CoarsePhysicalBackend:
    """Thin adapter around the existing :mod:`redc.physical` flow (unchanged)."""

    name = "physical"
    summary = "Coarse Minecraft backend: wide components, bus-valued nets, abstract grid cells (redc.physical)."
    trace_schema = "redc.pnr.trace.v1"
    design_schema = "redc.physical.v1"
    netlist_stages: tuple[str, ...] = ("physical",)
    options = frozenset(
        {"spacing", "routing_margin", "max_route_iterations", "max_attempts", "grid_height", "layer_gap"}
    )

    def default_paths(self, stem: str) -> PnRPaths:
        return PnRPaths(Path("build") / f"{stem}.physical.json", Path("build") / f"{stem}.pnr.json")

    def viewer_path(self, trace_path: Path) -> Path:
        return trace_path.with_name(trace_path.name.split(".")[0] + ".pnr.html")

    def dump_netlist(self, graph: Graph, *, stage: str | None = None, interface: str | None = None) -> dict[str, Any]:
        from .physical import lower_to_physical

        if stage not in (None, "physical"):
            raise CompileError(f"the physical backend has no netlist stage {stage!r} (only 'physical')")
        if interface is not None:
            raise CompileError("--interface is only supported by the physical-primitive backend")
        return lower_to_physical(graph).to_dict()

    def prepare(self, graph: Graph, options: PnROptions, *, source: Path | None, top: str) -> _CoarsePrepared:
        from .physical import lower_to_physical
        from .physical.pnr import PnRConfig, TraceLevel, TraceRecorder

        grid_height = 24 if options.grid_height is None else options.grid_height
        config = PnRConfig(
            grid_height=grid_height,
            base_y=min(PnRConfig.base_y, grid_height - 1),
            component_spacing=3 if options.spacing is None else options.spacing,
            layer_gap=4 if options.layer_gap is None else options.layer_gap,
            routing_margin=4 if options.routing_margin is None else options.routing_margin,
            max_routing_iterations=40 if options.max_route_iterations is None else options.max_route_iterations,
            max_pnr_attempts=4 if options.max_attempts is None else options.max_attempts,
            trace_level=TraceLevel.parse(options.trace_level),
        )
        netlist = lower_to_physical(graph)
        return _CoarsePrepared(config, netlist, TraceRecorder(config.trace_level))

    def run(self, prepared: _CoarsePrepared) -> PnRRun:
        from .physical.pnr import place_and_route

        result = place_and_route(prepared.netlist, prepared.config, trace=prepared.recorder)
        if not result.success:
            failure = result.failure
            return PnRRun(False, result.attempts, failure.message if failure else "unknown failure", None, "")
        metrics = result.metrics
        bounds = metrics["final"]["bounds"]
        summary = (
            f"P&R: {metrics['placement']['component_count']} components, "
            f"{metrics['routing']['routed_nets']} nets, {metrics['routing']['wire_cells']} wire cells, "
            f"{metrics['routing']['iterations']} routing iteration(s), attempt {result.geometry.attempt}, "
            f"bounds {bounds['dims'] if bounds else '-'}"
        )
        return PnRRun(True, result.attempts, None, result.to_physical_dict(), summary)

    def write_viewer(self, trace: Mapping[str, Any], path: Path, *, title: str) -> Path:
        from .viewer import write_pnr_html

        return write_pnr_html(trace, path, title=title)


def _registry() -> dict[str, PhysicalBackend]:
    from .physical_primitive.backend import PrimitivePhysicalBackend

    backends: list[PhysicalBackend] = [CoarsePhysicalBackend(), PrimitivePhysicalBackend()]
    return {b.name: b for b in backends}


def physical_backends() -> dict[str, PhysicalBackend]:
    """Every registered physical backend, by name."""
    return _registry()


def available_physical_backends() -> list[str]:
    """Registered physical backend names, default first, then alphabetical."""
    names = sorted(_registry())
    names.remove(DEFAULT_PHYSICAL_BACKEND)
    return [DEFAULT_PHYSICAL_BACKEND, *names]


def get_physical_backend(name: str) -> PhysicalBackend:
    """Look up a physical backend by name, or fail listing the valid choices.
    Never falls back silently."""
    registry = _registry()
    try:
        return registry[name]
    except KeyError:
        choices = ", ".join(available_physical_backends())
        raise CompileError(f"unknown physical backend '{name}'; available physical backends: {choices}") from None


__all__ = [
    "DEFAULT_PHYSICAL_BACKEND",
    "OPTION_FLAGS",
    "CoarsePhysicalBackend",
    "PhysicalBackend",
    "PnROptions",
    "PnRPaths",
    "PnRRun",
    "available_physical_backends",
    "check_options",
    "get_physical_backend",
    "physical_backends",
]
