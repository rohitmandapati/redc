"""Block-resolution place-and-route for one-bit primitive designs.

    PrimitivePhysicalNetlist --place_and_route_primitive()--> PrimitivePnRResult
                                        (+ redc.physical-primitive.pnr.v1 replay trace)

* :mod:`.config`    -- :class:`PrimitivePnRConfig`, :class:`AttemptGeometry`;
* :mod:`.placement` -- levelized, bit-slice-aware block placement;
* :mod:`.search`    -- redstone-aware A* for one branch of one net;
* :mod:`.routing`   -- directed route trees + negotiated-congestion router;
* :mod:`.legalize`  -- signal strength and repeater insertion;
* :mod:`.verify`    -- independent geometric + electrical verification;
* :mod:`.trace`     -- the event-sourced replay recorder;
* :mod:`.records`   -- plain-JSON design records;
* :mod:`.design`    -- the attempt loop, stage artifacts, metrics and export.
"""

from .config import WORLD_HEIGHT, AttemptGeometry, PrimitivePnRConfig, TraceLevel
from .design import (
    PHYSICAL_SCHEMA,
    LegalizedPrimitiveDesign,
    PrimitivePnRError,
    PrimitivePnRFailure,
    PrimitivePnRResult,
    RoutedPrimitiveDesign,
    compute_metrics,
    legalize_all,
    logic_depth,
    place_and_route_primitive,
)
from .legalize import (
    LegalizationFailure,
    RealizedRoute,
    RouteElement,
    SinkReport,
    legalize_route,
)
from .placement import (
    PlacedPrimitiveDesign,
    PlacementError,
    place_design,
    placement_columns,
)
from .routing import (
    NegotiatedRedstoneRouter,
    RouteBranch,
    RouteRequest,
    RouteTree,
    RoutingFailure,
    RoutingOutcome,
    route_requests,
)
from .search import NetState, SearchResult, search_branch, validate_branch
from .trace import BACKEND, PHASES, TRACE_SCHEMA, PrimitiveTraceRecorder, load_trace
from .verify import Violation, verify_design

__all__ = [
    "BACKEND",
    "PHASES",
    "PHYSICAL_SCHEMA",
    "TRACE_SCHEMA",
    "WORLD_HEIGHT",
    "AttemptGeometry",
    "LegalizationFailure",
    "LegalizedPrimitiveDesign",
    "NegotiatedRedstoneRouter",
    "NetState",
    "PlacedPrimitiveDesign",
    "PlacementError",
    "PrimitivePnRConfig",
    "PrimitivePnRError",
    "PrimitivePnRFailure",
    "PrimitivePnRResult",
    "PrimitiveTraceRecorder",
    "RealizedRoute",
    "RouteBranch",
    "RouteElement",
    "RouteRequest",
    "RouteTree",
    "RoutedPrimitiveDesign",
    "RoutingFailure",
    "RoutingOutcome",
    "SearchResult",
    "SinkReport",
    "TraceLevel",
    "Violation",
    "compute_metrics",
    "legalize_all",
    "legalize_route",
    "load_trace",
    "logic_depth",
    "place_and_route_primitive",
    "place_design",
    "placement_columns",
    "route_requests",
    "search_branch",
    "validate_branch",
    "verify_design",
]
