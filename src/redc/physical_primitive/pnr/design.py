"""The primitive place-and-route driver, its stage artifacts and the final design.

    PrimitivePhysicalNetlist                       (tech-mapped, unplaced)
        --place_design-->        PlacedPrimitiveDesign
        --NegotiatedRedstoneRouter--> RoutedPrimitiveDesign      (geometric routes)
        --legalize_route-->      LegalizedPrimitiveDesign        (dust / repeaters)
        --verify_design-->       final ``redc.physical-primitive.v1`` JSON

:func:`place_and_route_primitive` runs attempts; each starts from a fresh grid
with deterministically wider spacing, taller routing room and a larger search
margin.  A net that electrical legalization rejects (no repeater site keeps
its signal alive) is ripped up and rerouted with a penalty, then the routing is
renegotiated.  Every attempt -- failed ones included -- stays in the trace, and
a failed run still returns a result whose trace can be written.

Three timing notions are kept apart: *state* (only REGISTER_BIT cells hold
it), *logical depth* (gates on the longest combinational path) and *physical
delay* (cell latencies plus repeater ticks along the realized routes).
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...parser import CompileError
from ...tracing import TraceLevel
from ..geometry import Bounds, coord_list
from ..grid import BlockGrid
from ..netlist import GATE_KINDS, BitTerminal, PrimitiveKind
from ..physical import PrimitivePhysicalNetlist
from ..redstone import (
    CLEARANCE_MATERIAL,
    MIN_SIGNAL_Y,
    SUPPORT_MATERIAL,
    SUPPORT_REQUIREMENT,
)
from .config import AttemptGeometry, PrimitivePnRConfig
from .legalize import LegalizationFailure, RealizedRoute, legalize_route
from .placement import PlacedPrimitiveDesign, PlacementError, place_design
from .records import design_records, instance_record
from .routing import (
    NegotiatedRedstoneRouter,
    RouteRequest,
    RouteTree,
    RoutingOutcome,
    route_requests,
)
from .trace import BACKEND, COORDINATE_SYSTEM, PrimitiveTraceRecorder
from .verify import Violation, verify_design

PHYSICAL_SCHEMA = "redc.physical-primitive.v1"


@dataclass
class RoutedPrimitiveDesign:
    """Geometric routing stage: one directed tree per one-bit net."""

    placed: PlacedPrimitiveDesign
    requests: dict[int, RouteRequest]
    routes: dict[int, RouteTree]
    search_bounds: Bounds
    iterations: int
    rip_ups: int


@dataclass
class LegalizedPrimitiveDesign:
    """Electrical stage: every signal block typed (dust / repeater)."""

    routed: RoutedPrimitiveDesign
    realized: dict[int, RealizedRoute]
    rounds: int


@dataclass(frozen=True)
class PrimitivePnRFailure:
    """Why primitive P&R failed, as of the last attempt."""

    stage: str  # placement | routing | legalization | verification
    reason: str
    message: str
    attempt: int
    attempts: int
    net: int | None = None
    iteration: int | None = None
    details: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "reason": self.reason,
            "message": self.message,
            "attempt": self.attempt,
            "attempts": self.attempts,
            "net": self.net,
            "iteration": self.iteration,
            "details": list(self.details),
        }


class PrimitivePnRError(CompileError):
    """Raised by :meth:`PrimitivePnRResult.raise_for_failure`."""

    def __init__(self, result: PrimitivePnRResult) -> None:
        self.result = result
        failure = result.failure
        message = failure.message if failure else "place-and-route failed"
        super().__init__(f"place-and-route failed after {result.attempts} attempt(s): {message}")


@dataclass
class PrimitivePnRResult:
    """Outcome of :func:`place_and_route_primitive` (of its LAST attempt)."""

    success: bool
    mapped: PrimitivePhysicalNetlist
    trace: PrimitiveTraceRecorder
    attempts: int
    geometry: AttemptGeometry
    placed: PlacedPrimitiveDesign | None = None
    routed: RoutedPrimitiveDesign | None = None
    legalized: LegalizedPrimitiveDesign | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    failure: PrimitivePnRFailure | None = None
    violations: list[Violation] = field(default_factory=list)

    @property
    def grid(self) -> BlockGrid | None:
        return None if self.placed is None else self.placed.grid

    @property
    def routes(self) -> dict[int, RouteTree]:
        return {} if self.routed is None else self.routed.routes

    @property
    def realized(self) -> dict[int, RealizedRoute]:
        return {} if self.legalized is None else self.legalized.realized

    @property
    def design_bounds(self) -> Bounds | None:
        return _design_bounds(self.mapped, self.routes, self.realized)

    def raise_for_failure(self) -> None:
        if not self.success:
            raise PrimitivePnRError(self)

    def to_design_dict(self) -> dict[str, Any]:
        """The final physical truth: every block of every cell and route."""
        self.raise_for_failure()
        assert self.legalized is not None
        mapped = self.mapped
        records = design_records(mapped)
        instances = []
        net_of = mapped.net_of_terminal()
        for inst in mapped.instances:
            placed = inst.placed
            record = instance_record(mapped, inst.id)
            record["bounds"] = placed.bounds.to_dict()
            record["voxels"] = [{"coord": coord_list(c), "role": role} for c, role in placed.voxels()]
            record["keepout"] = [coord_list(c) for c in sorted(placed.keepout)]
            record["pins"] = [
                {
                    "name": p.name,
                    "direction": p.direction,
                    "position": coord_list(p.position),
                    "facing": p.facing.label,
                    "strength": p.strength,
                    "net": net_of.get(BitTerminal(inst.id, p.name)),
                }
                for p in placed.pins.values()
            ]
            instances.append(record)
        nets = []
        for record in records["nets"]:
            route = self.legalized.realized[record["id"]]
            nets.append({**record, "route": route.tree.to_dict(), "realized": route.to_dict()})
        bounds = self.design_bounds
        return {
            "schema": PHYSICAL_SCHEMA,
            "backend": BACKEND,
            "generator": "redc",
            "coordinate_system": COORDINATE_SYSTEM,
            "source": self.trace.source,
            "ir": mapped.logical.graph_summary,
            "library": mapped.library,
            "placeholder_geometry": any(i.cell.placeholder for i in mapped.instances),
            "materials": {
                "support": SUPPORT_MATERIAL,
                "support_requirement": SUPPORT_REQUIREMENT,
                "clearance": CLEARANCE_MATERIAL,
                "dust": "minecraft:redstone_wire",
                "repeater": "minecraft:repeater",
                "repeater_facing": "elements[].facing is the OUTPUT direction; blockstate_facing is the "
                "minecraft:repeater[facing=...] value (the INPUT side)",
            },
            "logical": {
                "summary": records["summary"],
                "ir_nodes": records["ir_nodes"],
                "groups": records["groups"],
                "ports": records["ports"],
                "buses": records["buses"],
            },
            "cells": records["cells"],
            "instances": instances,
            "nets": nets,
            "bounds": bounds.to_dict() if bounds else None,
            "metrics": self.metrics,
        }


def place_and_route_primitive(
    mapped: PrimitivePhysicalNetlist,
    config: PrimitivePnRConfig | None = None,
    *,
    trace: PrimitiveTraceRecorder | None = None,
) -> PrimitivePnRResult:
    """Place, route and legalize ``mapped`` (mutates only instance placement).

    Never raises on P&R failure: check ``result.success``."""
    config = config or PrimitivePnRConfig()
    recorder = trace if trace is not None else PrimitiveTraceRecorder(config.trace_level)
    mapped.validate()
    recorder.begin(mapped, config)
    recorder.emit(
        "pnr",
        "pnr_begin",
        instances=len(mapped.instances),
        nets=len(mapped.nets),
        max_attempts=config.max_pnr_attempts,
    )
    result: PrimitivePnRResult | None = None
    for attempt in range(config.max_pnr_attempts):
        result = _attempt(mapped, config, recorder, attempt)
        if result.success:
            break
    assert result is not None
    recorder.emit("pnr", "pnr_end", success=result.success, attempts=result.attempts)
    recorder.finish(_final_state(result))
    return result


def _attempt(
    mapped: PrimitivePhysicalNetlist, config: PrimitivePnRConfig, trace: PrimitiveTraceRecorder, attempt: int
) -> PrimitivePnRResult:
    geometry = config.attempt_geometry(attempt)
    attempts = attempt + 1
    for inst in mapped.instances:
        inst.unplace()
    grid = BlockGrid(max_y=geometry.max_y)
    trace.emit("pnr", "pnr_attempt_begin", **geometry.to_dict())
    result = PrimitivePnRResult(False, mapped, trace, attempts, geometry)

    def fail(stage: str, reason: str, message: str, **extra: Any) -> PrimitivePnRResult:
        result.failure = PrimitivePnRFailure(stage, reason, message, attempt, attempts, **extra)
        result.metrics = compute_metrics(result)
        return _end_attempt(result)

    try:
        result.placed = place_design(mapped, grid, geometry, trace)
    except PlacementError as error:
        trace.emit("placement", "placement_failed", attempt=attempt, reason=str(error))
        return fail("placement", "placement", str(error))
    requests = {r.net: r for r in route_requests(mapped, grid)}
    extent = grid.placement_bounds
    assert extent is not None
    margin = geometry.routing_margin
    search_bounds = extent.expand(margin, margin, (MIN_SIGNAL_Y, geometry.max_y))
    def keyframe_state() -> dict[str, Any]:
        # What a replay needs besides routes and congestion to restart here.
        current = {} if result.legalized is None else result.legalized.realized
        alive = sorted(n for n, r in current.items() if router.routes.get(n) is r.tree)
        return {
            "placement": _placement_records(mapped),
            "realized": [current[n].to_dict() for n in alive],
        }

    router = NegotiatedRedstoneRouter(
        grid, list(requests.values()), bounds=search_bounds, config=config, trace=trace,
        keyframe_state=keyframe_state,
    )  # fmt: skip
    outcome = router.run()
    result.routed = _routed(result.placed, requests, outcome, search_bounds)
    if not outcome.success:
        failure = outcome.failure
        assert failure is not None
        return fail("routing", failure.reason, failure.message, net=failure.net, iteration=failure.iteration,
                    details=(failure.to_dict(),))  # fmt: skip

    for round_ in range(config.max_legalization_rounds + 1):
        realized, failures = legalize_all(outcome.routes, requests, trace, round_)
        result.legalized = LegalizedPrimitiveDesign(result.routed, realized, round_ + 1)
        if not failures:
            break
        if round_ == config.max_legalization_rounds:
            first = failures[0]
            return fail("legalization", "signal_strength", first.message, net=first.net,
                        details=tuple(f.to_dict() for f in failures[:50]))  # fmt: skip
        penalize = [c for f in failures for c in f.cells]
        outcome = router.repair([f.net for f in failures], reason="legalization", penalize=penalize)
        result.routed = _routed(result.placed, requests, outcome, search_bounds)
        # Keep only realized routes whose tree survived the repair: a ripped-up
        # net lost its realized route (exactly what replaying the trace shows).
        kept = {n: r for n, r in realized.items() if outcome.routes.get(n) is r.tree}
        result.legalized = LegalizedPrimitiveDesign(result.routed, kept, round_ + 1)
        if not outcome.success:
            failure = outcome.failure
            assert failure is not None
            return fail("routing", failure.reason, failure.message, net=failure.net,
                        iteration=failure.iteration, details=(failure.to_dict(),))  # fmt: skip

    assert result.legalized is not None
    violations = verify_design(mapped, requests, result.legalized.realized, max_y=geometry.max_y)
    result.violations = violations
    if violations:
        for violation in violations[:200]:
            trace.emit("legalization", "illegal_transition", **violation.to_dict())
        return fail("verification", violations[0].kind, violations[0].message,
                    details=tuple(v.to_dict() for v in violations[:50]))  # fmt: skip
    result.success = True
    result.metrics = compute_metrics(result)
    bounds = result.design_bounds
    trace.emit(
        "finalize",
        "design_finalized",
        attempt=attempt,
        nets=len(result.legalized.realized),
        dust=result.metrics["routing"]["dust_blocks"],
        repeaters=result.metrics["routing"]["repeaters"],
        supports=result.metrics["routing"]["support_blocks"],
        bounds=bounds.to_dict() if bounds else None,
    )
    return _end_attempt(result)


def _routed(
    placed: PlacedPrimitiveDesign,
    requests: dict[int, RouteRequest],
    outcome: RoutingOutcome,
    bounds: Bounds,
) -> RoutedPrimitiveDesign:
    return RoutedPrimitiveDesign(placed, requests, dict(outcome.routes), bounds, outcome.iterations, outcome.rip_ups)


def legalize_all(
    routes: Mapping[int, RouteTree],
    requests: Mapping[int, RouteRequest],
    trace: PrimitiveTraceRecorder | None,
    round_: int = 0,
) -> tuple[dict[int, RealizedRoute], list[LegalizationFailure]]:
    """Legalize every routed net (id order); emits the legalization trace."""
    if trace:
        trace.emit("legalization", "legalization_begin", round=round_, nets=len(routes))
    detailed = trace is not None and trace.wants(TraceLevel.DETAILED)
    realized: dict[int, RealizedRoute] = {}
    failures: list[LegalizationFailure] = []
    for net_id in sorted(routes):
        tree, request = routes[net_id], requests[net_id]
        if detailed:
            assert trace is not None
            trace.emit("legalization", "route_legalization_begin", level=TraceLevel.DETAILED, net=net_id,
                       round=round_, length=tree.length, drive=request.driver.strength)  # fmt: skip
        outcome = legalize_route(tree, request)
        if isinstance(outcome, LegalizationFailure):
            failures.append(outcome)
            if trace:
                trace.emit("legalization", "legalization_failed", round=round_, **outcome.to_dict())
            continue
        realized[net_id] = outcome
        if trace:
            if detailed:
                trace.emit(
                    "legalization", "signal_strength_scan", level=TraceLevel.DETAILED, net=net_id,
                    round=round_, powered=outcome.powered,
                    cells=[[*coord_list(e.coord), e.strength] for e in outcome.elements],
                )  # fmt: skip
            for element in outcome.repeaters:
                trace.emit(
                    "legalization", "repeater_inserted", net=net_id, round=round_,
                    coord=coord_list(element.coord),
                    facing=element.facing.label if element.facing else None,
                    input_strength=element.strength, delay=element.delay,
                )  # fmt: skip
            trace.emit("legalization", "route_realized", round=round_, **outcome.to_dict())
    if trace:
        trace.emit(
            "legalization",
            "legalization_complete",
            round=round_,
            realized=len(realized),
            failures=len(failures),
            repeaters=sum(len(r.repeaters) for r in realized.values()),
        )
    return realized, failures


def _end_attempt(result: PrimitivePnRResult) -> PrimitivePnRResult:
    result.trace.emit(
        "pnr",
        "pnr_attempt_end",
        attempt=result.geometry.attempt,
        status="success" if result.success else "failed",
        failure=result.failure.to_dict() if result.failure else None,
        metrics=result.metrics or None,
    )
    return result


def _design_bounds(
    mapped: PrimitivePhysicalNetlist, routes: Mapping[int, RouteTree], realized: Mapping[int, RealizedRoute]
) -> Bounds | None:
    bounds: Bounds | None = None
    for inst in mapped.instances:
        if inst.is_placed:
            box = Bounds.of(inst.placed.occupied)
            if box is not None:
                bounds = box.union(bounds)
    for net_id, tree in routes.items():
        cells = list(tree.cells)
        route = realized.get(net_id)
        if route is not None:
            cells.extend(route.supports)
        box = Bounds.of(cells)
        if box is not None:
            bounds = box.union(bounds)
    return bounds


def logic_depth(mapped: PrimitivePhysicalNetlist) -> int:
    """Gates on the longest combinational path (register bits cut paths)."""
    logical = mapped.logical
    depth: dict[int, int] = {}
    deps: dict[int, list[int]] = {}
    users: dict[int, list[int]] = {}
    for net in logical.nets:
        driver = net.driver.instance
        for sink in net.sinks:
            if logical.instances[sink.instance].kind in GATE_KINDS:
                deps.setdefault(sink.instance, []).append(driver)
                users.setdefault(driver, []).append(sink.instance)
    gates = [i.id for i in logical.instances if i.kind in GATE_KINDS]
    pending = {g: sum(1 for d in deps.get(g, ()) if logical.instances[d].kind in GATE_KINDS) for g in gates}
    ready = [g for g in gates if pending[g] == 0]
    heapq.heapify(ready)
    while ready:
        gate = heapq.heappop(ready)
        depth[gate] = 1 + max((depth.get(d, 0) for d in deps.get(gate, ())), default=0)
        for user in users.get(gate, ()):
            pending[user] -= 1
            if pending[user] == 0:
                heapq.heappush(ready, user)
    return max(depth.values(), default=0)


def physical_delay(mapped: PrimitivePhysicalNetlist, realized: Mapping[int, RealizedRoute]) -> dict[str, Any]:
    """Estimated settle time in redstone ticks: cell latency plus repeater
    delay along each route, over the longest combinational path."""
    logical = mapped.logical
    route_delay: dict[tuple[int, str], int] = {}
    for route in realized.values():
        for report in route.sinks:
            route_delay[(report.sink.instance, report.sink.pin)] = report.delay_ticks
    unknown = sorted({i.cell.name for i in mapped.instances if i.cell.latency is None})
    arrival: dict[int, int] = {}
    deps: dict[int, list[tuple[int, int]]] = {}
    users: dict[int, list[int]] = {}
    for net in logical.nets:
        for sink in net.sinks:
            hop = route_delay.get((sink.instance, sink.pin), 0)
            deps.setdefault(sink.instance, []).append((net.driver.instance, hop))
            users.setdefault(net.driver.instance, []).append(sink.instance)
    gates = {i.id for i in logical.instances if i.kind in GATE_KINDS}
    pending = {g: sum(1 for d, _ in deps.get(g, ()) if d in gates) for g in gates}
    ready = [g for g in gates if pending[g] == 0]
    heapq.heapify(ready)

    def start(inst_id: int) -> int:
        cell = mapped.instances[inst_id].cell
        return arrival.get(inst_id, cell.latency or 0) if inst_id in gates else (cell.latency or 0)

    while ready:
        gate = heapq.heappop(ready)
        latency = mapped.instances[gate].cell.latency or 0
        arrival[gate] = latency + max((start(d) + hop for d, hop in deps.get(gate, ())), default=0)
        for user in users.get(gate, ()):
            if user in pending:
                pending[user] -= 1
                if pending[user] == 0:
                    heapq.heappush(ready, user)
    worst = 0
    for inst in logical.instances:
        if inst.kind in (PrimitiveKind.OUTPUT_BIT, PrimitiveKind.REGISTER_BIT, PrimitiveKind.PERIPHERAL):
            for d, hop in deps.get(inst.id, ()):
                worst = max(worst, start(d) + hop)
    return {"critical_path_ticks": worst, "unknown_latency_cells": unknown, "placeholder_estimate": True}


def compute_metrics(result: PrimitivePnRResult) -> dict[str, Any]:
    """Deterministic metrics for engineering and visualization (no wall-clock)."""
    mapped = result.mapped
    logical = mapped.logical
    summary = logical.summary()
    routes = result.routes
    realized = result.realized
    grid = result.grid
    lengths = [t.length for t in routes.values()]
    vertical = sum(
        1 for t in routes.values() for c, up in t.parent.items() if up is not None and up[1] != c[1]
    )
    dust = sum(len(r.dust) for r in realized.values())
    repeaters = sum(len(r.repeaters) for r in realized.values())
    supports = sum(len(r.supports) for r in realized.values())
    sink_strengths = [s.strength for r in realized.values() if r.powered for s in r.sinks]
    bounds = result.design_bounds
    placement_bounds = None if grid is None else grid.placement_bounds
    return {
        "ir": {
            "live_nodes": logical.graph_summary.get("live_nodes"),
            "ops": logical.graph_summary.get("ops"),
        },
        "logical": {
            "logical_ports": summary["logical_ports"],
            "logical_buses": summary["logical_buses"],
            "primitive_gates": summary["gates"],
            "gates_by_kind": summary["gates_by_kind"],
            "primitives_by_kind": summary["by_kind"],
            "register_bits": summary["register_bits"],
            "constant_bits": summary["constant_bits"],
            "one_bit_nets": summary["nets"],
            "fanout_distribution": summary["fanout"],
            "max_fanout": summary["max_fanout"],
            "logic_depth": logic_depth(mapped),
        },
        "techmap": {
            "library": mapped.library,
            "components": len(mapped.instances),
            "cells": dict(sorted(Counter(i.cell.name for i in mapped.instances).items())),
            "component_voxels": sum(len(i.cell.voxels) for i in mapped.instances),
            "placeholder_cells": any(i.cell.placeholder for i in mapped.instances),
        },
        "placement": {
            "component_count": sum(1 for i in mapped.instances if i.is_placed),
            "occupied_component_voxels": 0 if grid is None else len(grid.body),
            "keepout_voxels": 0 if grid is None else len(grid.keepout),
            "probes": 0 if result.placed is None else result.placed.probes,
            "bounds": placement_bounds.to_dict() if placement_bounds else None,
        },
        "routing": {
            "routed_nets": len(routes),
            "iterations": 0 if result.routed is None else result.routed.iterations,
            "rip_ups": 0 if result.routed is None else result.routed.rip_ups,
            "total_routed_length": sum(lengths),
            "average_routed_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "max_routed_length": max(lengths, default=0),
            "total_branch_steps": sum(t.steps for t in routes.values()),
            "vertical_steps": vertical,
            "dust_blocks": dust,
            "repeaters": repeaters,
            "support_blocks": supports,
            "clearance_blocks": len({c for r in realized.values() for c in r.clearances}),
        },
        "legalization": {
            "rounds": 0 if result.legalized is None else result.legalized.rounds,
            "repeaters": repeaters,
            "min_sink_strength": min(sink_strengths, default=None),
        },
        "timing": {
            "state_elements": summary["register_bits"],
            "logic_depth": logic_depth(mapped),
            **physical_delay(mapped, realized),
        },
        "pnr": {"attempts": result.attempts, "attempt": result.geometry.attempt},
        "final": {
            "bounds": bounds.to_dict() if bounds else None,
            "volume": bounds.volume if bounds else 0,
            "blocks": (0 if grid is None else len(grid.body)) + dust + repeaters + supports,
        },
    }


def _placement_records(mapped: PrimitivePhysicalNetlist) -> list[dict[str, Any]]:
    return [
        {
            "instance": inst.id,
            "origin": None if inst.origin is None else list(inst.origin),
            "orientation": None if inst.orientation is None else inst.orientation.name,
            "bounds": inst.placed.bounds.to_dict() if inst.is_placed else None,
        }
        for inst in mapped.instances
    ]


def _final_state(result: PrimitivePnRResult) -> dict[str, Any]:
    """The trace's ``final`` block: the end state of the LAST attempt."""
    mapped = result.mapped
    grid = result.grid
    bounds = result.design_bounds
    conflicts = [] if grid is None or result.success else [c.to_dict() for c in grid.conflicts()[:500]]
    return {
        "success": result.success,
        "attempt": result.geometry.attempt,
        "attempts": result.attempts,
        "geometry": result.geometry.to_dict(),
        "failure": result.failure.to_dict() if result.failure else None,
        "placement": _placement_records(mapped),
        "routes": [result.routes[n].to_dict() for n in sorted(result.routes)],
        "realized": [result.realized[n].to_dict() for n in sorted(result.realized)],
        "conflicts": conflicts,
        "violations": [v.to_dict() for v in result.violations[:200]],
        "design_bounds": bounds.to_dict() if bounds else None,
        "search_bounds": None if result.routed is None else result.routed.search_bounds.to_dict(),
        "metrics": result.metrics,
    }


__all__ = [
    "PHYSICAL_SCHEMA",
    "LegalizedPrimitiveDesign",
    "PrimitivePnRError",
    "PrimitivePnRFailure",
    "PrimitivePnRResult",
    "RoutedPrimitiveDesign",
    "compute_metrics",
    "legalize_all",
    "logic_depth",
    "physical_delay",
    "place_and_route_primitive",
]
