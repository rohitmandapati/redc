"""The place-and-route driver: attempts, verification, metrics, serialization.

:func:`place_and_route` runs the whole first-pass physical flow on an unplaced
:class:`~redc.physical.netlist.PhysicalNetlist`:

    for each attempt (spacing grows deterministically every retry):
        fresh Grid -> levelized placement -> negotiated-congestion routing
        success: commit wires, verify, measure, stop
        failure: record why, spread the design out, try again

P&R changes only instance origins, routing geometry, grid occupancy and the
trace -- never the netlist's logic or connectivity.  A failed run still returns
a result (``success=False``) whose trace holds every attempt, so callers can
always write the replay before reporting the error.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...parser import CompileError
from ..grid import CellKind, Grid
from ..netlist import PhysicalNetlist
from .config import AttemptGeometry, PnRConfig
from .geometry import Bounds, Coord, coord_list, escape_cell, footprint, manhattan
from .placement import Placement, PlacementError, place_netlist
from .records import design_records
from .routing import NegotiatedRouter, RoutedNet, RouteEndpoint, RouteRequest
from .trace import COORDINATE_SYSTEM, TraceRecorder

PHYSICAL_SCHEMA = "redc.physical.v1"


@dataclass(frozen=True)
class PnRFailure:
    """Why place-and-route failed, as of the last attempt."""

    stage: str  # placement | routing | verification
    reason: str  # placement | unroutable | congestion | verification
    message: str
    attempt: int
    attempts: int
    net: int | None = None
    sink: dict[str, Any] | None = None
    iteration: int | None = None
    overused: tuple[Coord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "reason": self.reason,
            "message": self.message,
            "attempt": self.attempt,
            "attempts": self.attempts,
            "net": self.net,
            "sink": self.sink,
            "iteration": self.iteration,
            "overused": [coord_list(c) for c in self.overused],
        }


class PnRError(CompileError):
    """Raised by :meth:`PnRResult.raise_for_failure`; carries the result."""

    def __init__(self, result: PnRResult) -> None:
        self.result = result
        failure = result.failure
        message = failure.message if failure else "place-and-route failed"
        attempts = failure.attempts if failure else result.attempts
        super().__init__(f"place-and-route failed after {attempts} attempt(s): {message}")


@dataclass
class PnRResult:
    """Outcome of :func:`place_and_route` (of its last attempt, on failure)."""

    success: bool
    netlist: PhysicalNetlist
    grid: Grid
    routes: dict[int, RoutedNet]
    geometry: AttemptGeometry
    attempts: int
    trace: TraceRecorder
    metrics: dict[str, Any] = field(default_factory=dict)
    search_bounds: Bounds | None = None
    failure: PnRFailure | None = None

    @property
    def design_bounds(self) -> Bounds | None:
        """Bounding box of every placed body and routed wire."""
        return _design_bounds(self.netlist, self.routes)

    def raise_for_failure(self) -> None:
        if not self.success:
            raise PnRError(self)

    def to_physical_dict(self) -> dict[str, Any]:
        """The concise final design: placed instances, exact origins and
        geometry, and final routes only (no transient search or rip-up data)."""
        self.raise_for_failure()
        records = design_records(self.netlist)
        boxes = {
            inst.id: Bounds.box(inst.origin, inst.component.dim).to_dict()
            for inst in self.netlist.instances.values()
            if inst.origin is not None
        }
        for inst in records["instances"]:
            inst["bounds"] = boxes[inst["id"]]
        for net in records["nets"]:
            net["route"] = self.routes[net["id"]].to_dict()
        bounds = self.design_bounds
        return {
            "schema": PHYSICAL_SCHEMA,
            "coordinate_system": COORDINATE_SYSTEM,
            "grid": {
                "height": self.grid.height,
                "bounds": bounds.to_dict() if bounds else None,
            },
            **records,
            "metrics": self.metrics,
        }


def place_and_route(
    netlist: PhysicalNetlist,
    config: PnRConfig | None = None,
    *,
    trace: TraceRecorder | None = None,
) -> PnRResult:
    """Place and route ``netlist`` (mutates only instance origins).

    Returns a :class:`PnRResult`; it does not raise on P&R failure -- check
    ``result.success`` or call ``result.raise_for_failure()``."""
    config = config or PnRConfig()
    recorder = trace if trace is not None else TraceRecorder(config.trace_level)
    netlist.validate()
    for inst in netlist.instances.values():
        inst.origin = None
    recorder.begin(netlist, config)
    recorder.emit(
        "pnr",
        "pnr_begin",
        instances=len(netlist.instances),
        nets=len(netlist.nets),
        max_attempts=config.max_pnr_attempts,
    )

    result: PnRResult | None = None
    for attempt in range(config.max_pnr_attempts):
        result = _attempt(netlist, config, recorder, attempt)
        if result.success:
            break
    assert result is not None
    recorder.emit("pnr", "pnr_end", success=result.success, attempts=result.attempts)
    recorder.finish(_final_state(result))
    return result


def _attempt(
    netlist: PhysicalNetlist, config: PnRConfig, trace: TraceRecorder, attempt: int
) -> PnRResult:
    geometry = config.attempt_geometry(attempt)
    attempts = attempt + 1
    for inst in netlist.instances.values():
        inst.origin = None
    grid = Grid(height=config.grid_height)
    trace.emit("pnr", "pnr_attempt_begin", **geometry.to_dict())
    result = PnRResult(
        success=False,
        netlist=netlist,
        grid=grid,
        routes={},
        geometry=geometry,
        attempts=attempts,
        trace=trace,
    )

    try:
        placement = place_netlist(netlist, grid, geometry, config, trace)
        requests, escape_owner = route_requests(netlist, placement)
    except PlacementError as error:
        trace.emit("placement", "placement_failed", attempt=attempt, reason=str(error))
        result.failure = PnRFailure("placement", "placement", str(error), attempt, attempts)
        return _end_attempt(result)

    extent = placement.extent
    assert extent is not None
    result.search_bounds = extent.expand_horizontal(geometry.routing_margin, grid.height)
    router = NegotiatedRouter(
        grid,
        requests,
        bounds=result.search_bounds,
        config=config,
        trace=trace,
        escape_owner=escape_owner,
        design_bounds=placement.bounds,
        placement=_placement_record(netlist),
    )
    outcome = router.run()
    result.routes = outcome.routes
    result.metrics = compute_metrics(netlist, grid, outcome.routes, outcome.iterations)
    if not outcome.success:
        failure = outcome.failure
        assert failure is not None
        result.failure = PnRFailure(
            "routing",
            failure.reason,
            failure.message,
            attempt,
            attempts,
            net=failure.net,
            sink=failure.sink.to_dict() if failure.sink else None,
            iteration=failure.iteration,
            overused=failure.overused,
        )
        return _end_attempt(result)

    try:
        committed = grid.commit_routes()
        verify_design(netlist, grid, outcome.routes, requests, result.search_bounds)
    except CompileError as error:
        result.failure = PnRFailure("verification", "verification", str(error), attempt, attempts)
        return _end_attempt(result)
    result.success = True
    result.metrics = compute_metrics(netlist, grid, outcome.routes, outcome.iterations)
    trace.emit(
        "finalize",
        "routes_finalized",
        attempt=attempt,
        nets=len(outcome.routes),
        wire_cells=committed,
        bounds=result.design_bounds.to_dict() if result.design_bounds else None,
    )
    return _end_attempt(result)


def _end_attempt(result: PnRResult) -> PnRResult:
    result.trace.emit(
        "pnr",
        "pnr_attempt_end",
        attempt=result.geometry.attempt,
        status="success" if result.success else "failed",
        failure=result.failure.to_dict() if result.failure else None,
        metrics=result.metrics or None,
    )
    return result


def route_requests(
    netlist: PhysicalNetlist, placement: Placement
) -> tuple[list[RouteRequest], dict[Coord, int]]:
    """One :class:`RouteRequest` per net (id order) plus the escape-cell owner
    map (``-1`` = an unconnected pin).  Raises :class:`PlacementError` if two
    pins carrying different nets would share one escape cell."""
    net_of: dict[tuple[int, str], int] = {}
    for net in netlist.nets.values():
        for terminal in (net.driver, *net.sinks):
            net_of[(terminal.instance.id, terminal.port.name)] = net.id
    owner: dict[Coord, int] = {}
    for inst in sorted(netlist.instances.values(), key=lambda i: i.id):
        assert inst.origin is not None
        for port in inst.component.ports:
            cell = escape_cell(port, inst.origin)
            net_id = net_of.get((inst.id, port.name), -1)
            previous = owner.get(cell)
            if previous is not None and previous != net_id and -1 not in (previous, net_id):
                raise PlacementError(
                    f"pin {port.name!r} of instance {inst.id} shares escape cell "
                    f"{coord_list(cell)} with net {previous} but carries net {net_id}"
                )
            if previous is None or previous == -1:
                owner[cell] = net_id

    def endpoint(terminal) -> RouteEndpoint:
        return RouteEndpoint(terminal.instance.id, terminal.port.name, terminal.outward)

    requests = [
        RouteRequest(net.id, endpoint(net.driver), tuple(endpoint(s) for s in net.sinks))
        for net in sorted(netlist.nets.values(), key=lambda n: n.id)
    ]
    return requests, owner


def verify_design(
    netlist: PhysicalNetlist,
    grid: Grid,
    routes: Mapping[int, RoutedNet],
    requests: list[RouteRequest],
    bounds: Bounds,
) -> None:
    """Independently re-check a committed design; raises on any violation."""
    for inst in netlist.instances.values():
        if inst.origin is None:
            raise CompileError(f"instance {inst.id} is not placed")
        for cell in footprint(inst.component, inst.origin):
            if grid.owner_at(*cell) != inst.id:
                raise CompileError(f"instance {inst.id} body cell {coord_list(cell)} is not its own")
            if grid.kind_at(*cell) not in (CellKind.COMPONENT, CellKind.PIN):
                raise CompileError(f"instance {inst.id} body cell {coord_list(cell)} is not a body")
    if grid.overused():
        raise CompileError(f"{len(grid.overused())} cells are overused after routing")
    for request in requests:
        routed = routes.get(request.net)
        if routed is None:
            raise CompileError(f"net {request.net} has no route")
        if routed.root != request.driver.coord:
            raise CompileError(f"net {request.net} does not start at its driver escape")
        tree = {routed.root}
        goals = set()
        for branch in routed.branches:
            if branch.start not in tree:
                raise CompileError(f"net {request.net} has a branch detached from its tree")
            for a, b in zip(branch.path, branch.path[1:]):
                if manhattan(a, b) != 1:
                    raise CompileError(f"net {request.net} path jumps {list(a)} -> {list(b)}")
            tree.update(branch.path)
            goals.add(branch.goal)
        missing = {s.coord for s in request.sinks} - goals
        if missing:
            raise CompileError(f"net {request.net} misses sink escapes {sorted(missing)}")
        for cell in routed.cells:
            if not bounds.contains(cell):
                raise CompileError(f"net {request.net} leaves the search bounds at {list(cell)}")
            if grid.owner_at(*cell) != request.net or grid.kind_at(*cell) != CellKind.WIRE:
                raise CompileError(f"net {request.net} cell {list(cell)} is not its committed wire")
        if grid.route_of(request.net) != routed.cell_set:
            raise CompileError(f"net {request.net} grid claim differs from its tree")


def _design_bounds(netlist: PhysicalNetlist, routes: Mapping[int, RoutedNet]) -> Bounds | None:
    bounds: Bounds | None = None
    for inst in netlist.instances.values():
        if inst.origin is not None:
            bounds = Bounds.box(inst.origin, inst.component.dim).union(bounds)
    for routed in routes.values():
        box = Bounds.of(routed.cells)
        if box is not None:
            bounds = box.union(bounds)
    return bounds


def compute_metrics(
    netlist: PhysicalNetlist,
    grid: Grid,
    routes: Mapping[int, RoutedNet],
    iterations: int,
) -> dict[str, Any]:
    """Deterministic placement / routing / final metrics (no wall-clock time)."""
    placed = [i for i in netlist.instances.values() if i.origin is not None]
    body: Bounds | None = None
    for inst in placed:
        assert inst.origin is not None
        body = Bounds.box(inst.origin, inst.component.dim).union(body)
    component_cells = sum(inst.component.volume for inst in placed)
    wire = set().union(*(r.cell_set for r in routes.values())) if routes else set()
    final = _design_bounds(netlist, routes)
    stats = grid.congestion_stats()
    return {
        "placement": {
            "component_count": len(placed),
            "component_cells": component_cells,
            "bounds": body.to_dict() if body else None,
        },
        "routing": {
            "net_count": len(netlist.nets),
            "routed_nets": len(routes),
            "wire_cells": len(wire),
            "total_net_cells": sum(r.length for r in routes.values()),
            "total_branch_length": sum(r.steps for r in routes.values()),
            "max_net_length": max((r.length for r in routes.values()), default=0),
            "total_fanout": sum(n.fanout for n in netlist.nets.values()),
            "overused_cells": stats["overused"],
            "max_occupancy": stats["max_occupancy"],
            "iterations": iterations,
        },
        "final": {
            "component_volume": component_cells,
            "wire_volume": len(wire),
            "bounds": final.to_dict() if final else None,
            "volume": final.volume if final else 0,
        },
    }


def _placement_record(netlist: PhysicalNetlist) -> list[dict[str, Any]]:
    return [
        {
            "instance": inst.id,
            "origin": coord_list(inst.origin) if inst.origin is not None else None,
        }
        for inst in sorted(netlist.instances.values(), key=lambda i: i.id)
    ]


_KIND_NAMES = {kind: kind.name.lower() for kind in CellKind}


def _final_state(result: PnRResult) -> dict[str, Any]:
    """The trace's ``final`` block: the end state of the last attempt."""
    grid = result.grid
    overused = sorted(grid.overused())
    congestion = []
    for cell in overused:
        nets = sorted(n for n, r in result.routes.items() if cell in r.cell_set)
        congestion.append(
            {
                "coord": coord_list(cell),
                "occupancy": grid.occupancy_at(*cell),
                "history": grid.history_at(*cell),
                "nets": nets,
            }
        )
    cells = sorted(grid.occupied_cells(), key=lambda c: (c[0], c[1], c[2]))
    bounds = result.design_bounds
    return {
        "success": result.success,
        "attempt": result.geometry.attempt,
        "attempts": result.attempts,
        "geometry": result.geometry.to_dict(),
        "failure": result.failure.to_dict() if result.failure else None,
        "placement": [
            {
                **entry,
                "bounds": (
                    Bounds.box(tuple(entry["origin"]), result.netlist.instances[entry["instance"]].component.dim).to_dict()
                    if entry["origin"] is not None
                    else None
                ),
            }
            for entry in _placement_record(result.netlist)
        ],
        "routes": [result.routes[n].to_dict() for n in sorted(result.routes)],
        "congestion": congestion,
        "grid": {
            "height": grid.height,
            "cells": [
                {"coord": [x, y, z], "owner": owner, "kind": _KIND_NAMES[kind]}
                for x, y, z, owner, kind in cells
            ],
        },
        "design_bounds": bounds.to_dict() if bounds else None,
        "search_bounds": result.search_bounds.to_dict() if result.search_bounds else None,
        "metrics": result.metrics,
    }
