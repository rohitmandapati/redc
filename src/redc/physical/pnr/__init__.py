"""First-pass 3D place-and-route with a versioned replay trace.

    PhysicalNetlist --place_and_route()--> PnRResult (+ redc.pnr.trace.v1 trace)

* :mod:`.config`    -- :class:`PnRConfig`, :class:`TraceLevel`;
* :mod:`.geometry`  -- cell coordinates, :class:`Bounds`, pin escape cells;
* :mod:`.placement` -- legality predicate + levelized baseline placer;
* :mod:`.search`    -- deterministic bounded six-connected A*;
* :mod:`.routing`   -- routing trees + negotiated-congestion router;
* :mod:`.trace`     -- the event-sourced replay recorder;
* :mod:`.records`   -- plain-JSON design records shared by trace and output;
* :mod:`.design`    -- the attempt loop, verification, metrics, serialization.
"""

from .config import AttemptGeometry, PnRConfig, TraceLevel
from .design import PHYSICAL_SCHEMA, PnRError, PnRFailure, PnRResult, place_and_route
from .geometry import Bounds, Coord, escape_cell
from .placement import (
    Placement,
    PlacementError,
    check_placement,
    place_instance,
    place_netlist,
    placement_layers,
)
from .routing import (
    NegotiatedRouter,
    RouteBranch,
    RoutedNet,
    RouteEndpoint,
    RouteRequest,
    RoutingFailure,
    RoutingOutcome,
)
from .search import SearchResult, astar
from .trace import TRACE_SCHEMA, TraceRecorder, load_trace

__all__ = [
    "PHYSICAL_SCHEMA",
    "TRACE_SCHEMA",
    "AttemptGeometry",
    "Bounds",
    "Coord",
    "NegotiatedRouter",
    "Placement",
    "PlacementError",
    "PnRConfig",
    "PnRError",
    "PnRFailure",
    "PnRResult",
    "RouteBranch",
    "RouteEndpoint",
    "RouteRequest",
    "RoutedNet",
    "RoutingFailure",
    "RoutingOutcome",
    "SearchResult",
    "TraceLevel",
    "TraceRecorder",
    "astar",
    "check_placement",
    "escape_cell",
    "load_trace",
    "place_and_route",
    "place_instance",
    "place_netlist",
    "placement_layers",
]
