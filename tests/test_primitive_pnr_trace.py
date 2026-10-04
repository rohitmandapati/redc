"""The ``redc.physical-primitive.pnr.v1`` replay trace of the primitive backend.

Checks the documented contract (``docs/physical-primitive-trace.md``): the
header and its block coordinate system, the self-contained UNPLACED design,
every event family per trace level (synthesis, technology mapping, attempts,
placement probes, routing and negotiation, electrical legalization), that
replaying the events reproduces ``final`` exactly, failure traces, determinism
and the ``.json`` / ``.jsonl`` round trip.

Designs are tiny (one gate, a half / full adder, a 2-bit counter) so the file
runs in seconds; a rejected placement probe and router congestion are forced
on hand-built :class:`BlockGrid` s.
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from redc import CompileError, compile_source
from redc.ir import Graph
from redc.physical_primitive import (
    DefaultInterfacePolicy,
    PadInterfacePolicy,
    PrimitiveNetlist,
    PrimitivePhysicalNetlist,
    PrimitivePnRConfig,
    PrimitivePnRResult,
    PrimitiveTechnologyLibrary,
    PrimitiveTraceRecorder,
    SynthesisRegistry,
    map_primitives_to_minecraft,
    place_and_route_graph,
    synthesize_to_primitives,
)
from redc.physical_primitive.geometry import Bounds, Direction
from redc.physical_primitive.grid import BlockGrid, Conflict, PinSite
from redc.physical_primitive.pnr import (
    BACKEND,
    PHASES,
    TRACE_SCHEMA,
    LegalizationFailure,
    NegotiatedRedstoneRouter,
    PrimitivePnRError,
    RouteRequest,
    Violation,
    load_trace,
    place_design,
)
from redc.physical_primitive.pnr import design as pnr_design
from redc.physical_primitive.redstone import REPEATER_DELAY_TICKS
from redc.physical_primitive.synthesis import lower

DOC = Path(__file__).resolve().parent.parent / "docs" / "physical-primitive-trace.md"

NOT = "bool main(bool a) { return !a; }"
HALF_ADDER = "uint2 main(bool a, bool b) { return (uint2)a + (uint2)b; }"
FULL_ADDER = "uint2 main(bool a, bool b, bool c) { return (uint2)a + (uint2)b + (uint2)c; }"
COUNTER = "uint2 main(uint2 n) { uint2 x = 0; for (uint2 i = 0; i < n; i++) { x = x + 1; } return x; }"
#: ``result: uint8`` becomes ONE 2-dig-7-seg display under the default interface
#: policy, and that placeholder cell is five blocks tall: it cannot be placed
#: below ``max_y = 5``.
DISPLAY = "uint8 main(uint8 a) { return a; }"

PADS = PadInterfacePolicy()
LEVELS = ("none", "basic", "detailed", "search")
#: Routing can never succeed: one A* expansion per branch, no distance bonus.
NO_EXPANSIONS = {"max_astar_expansions": 1, "expansions_per_block": 0}

ORIENTATIONS = ("east", "south", "west", "north")
DIRECTIONS = {"east": (1, 0, 0), "south": (0, 0, 1), "west": (-1, 0, 0), "north": (0, 0, -1)}
KIND_CATEGORY = {
    "and": "gate", "or": "gate", "xor": "gate", "not": "gate",
    "register_bit": "state", "const0": "constant", "const1": "constant",
    "input_bit": "boundary", "output_bit": "boundary",
    "clock_source": "control", "reset_source": "control", "peripheral": "peripheral",
}  # fmt: skip
GROUP_KINDS = {"root", "ir_node", "port", "helper", "slice", "stage", "iteration", "control"}
CONFLICT_KINDS = {"shared_signal", "shared_support", "signal_on_support", "clearance_blocked", "adjacent_signals"}

COORD_KEYS = frozenset(
    {"coord", "origin", "root", "start", "goal", "min", "max", "dims", "position", "from", "to"}
)
COORD_LIST_KEYS = frozenset({"path", "cells", "voxels", "keepout", "supports", "clearances"})
ENVELOPE = ("seq", "phase", "type")

#: Record shapes from the doc: ``(required keys, optional keys)``.
BOUNDS = ({"min", "max", "dims", "volume"}, set())
ROUTE = ({"net", "driver", "root", "branches", "cells", "length", "fanout"}, set())
REALIZED = (
    {"net", "powered", "elements", "repeaters", "supports", "clearances", "sinks", "min_strength",
     "max_delay_ticks"},
    set(),
)  # fmt: skip
RECORDS = {
    "header": (
        {"schema", "generator", "backend", "trace_level", "coordinate_system", "source", "ir", "design",
         "config", "events", "final"},
        set(),
    ),
    "coordinate_system": ({"units", "x", "y", "z", "cell"}, set()),
    "ir": ({"live_nodes", "ops", "sequential", "inputs", "outputs"}, set()),
    "design": (
        {"library", "cells", "instances", "nets", "groups", "ports", "buses", "ir_nodes", "summary"}, set()
    ),
    "cell": (
        {"name", "kind", "placeholder", "structure", "latency", "stateful", "description", "orientations",
         "peripheral", "voxels", "keepout", "pins"},
        set(),
    ),
    "pin": ({"name", "direction", "position", "facing", "strength"}, set()),
    "instance": (
        {"id", "kind", "category", "cell", "realizes", "ir_node", "ir_op", "ir_type", "role", "bit", "group",
         "port", "init", "peripheral", "origin", "orientation"},
        {"attrs"},
    ),
    "net": ({"id", "role", "width", "fanout", "driver", "sinks", "logical", "ports"}, set()),
    "group": ({"id", "name", "kind", "parent", "ir_node"}, {"attrs"}),
    "port": ({"name", "direction", "type", "source_name", "ir_node", "realization", "bits"}, set()),
    "bus": ({"ir_node", "op", "type", "bits"}, set()),
    "ir_node": ({"id", "op", "type", "args"}, {"name", "value", "init"}),
    "final": (
        {"success", "attempt", "attempts", "geometry", "failure", "placement", "routes", "realized", "conflicts",
         "violations", "design_bounds", "search_bounds", "metrics"},
        set(),
    ),
    "failure": ({"stage", "reason", "message", "attempt", "attempts", "net", "iteration", "details"}, set()),
    "geometry": ({"attempt", "component_spacing", "channel_width", "routing_margin", "max_y"}, set()),
    "placement": ({"instance", "origin", "orientation", "bounds"}, set()),
    "route": ROUTE,
    "branch": ({"sink", "start", "goal", "path"}, set()),
    "realized": REALIZED,
    "element": ({"coord", "kind", "parent", "strength", "delay"}, {"facing", "blockstate_facing"}),
    "sink": ({"instance", "pin", "coord", "strength", "required", "repeaters", "delay_ticks", "distance"}, set()),
}  # fmt: skip

#: Documented payload of every event type (beyond seq / phase / type); extra
#: fields are allowed (forward compatibility), missing ones are not.
EVENT_FIELDS = {
    "synthesis_begin": {"live_nodes", "ops", "sequential"},
    "ir_node_synthesized": {"ir_node", "op", "ir_type", "synthesis_pass", "recipe", "group", "primitives",
                            "first_instance", "last_instance", "by_kind"},
    "primitive_emitted": {"instance", "kind", "ir_node", "role", "bit", "group"},
    "synthesis_complete": {"instances", "gates", "gates_by_kind", "register_bits", "nets", "fanout"},
    "primitive_mapped": {"instance", "kind", "cell", "candidates"},
    "techmap_complete": {"library", "instances", "nets", "cells", "placeholder_cells", "component_voxels"},
    "pnr_begin": {"instances", "nets", "max_attempts"},
    "pnr_attempt_begin": {"attempt", "component_spacing", "channel_width", "routing_margin", "max_y"},
    "placement_begin": {"attempt", "component_spacing", "channel_width", "columns", "blocks"},
    "component_place_attempt": {"instance", "origin", "orientation", "voxels", "keepout", "bounds"},
    "component_place_rejected": {"instance", "origin", "orientation", "reason"},
    "component_placed": {"instance", "column", "block", "origin", "orientation", "bounds"},
    "design_bounds_changed": BOUNDS[0],
    "placement_complete": {"attempt", "component_count", "component_voxels", "keepout_voxels", "probes", "bounds"},
    "placement_failed": {"attempt", "reason"},
    "routing_begin": {"nets", "order", "search_bounds", "present_factor"},
    "routing_iteration_begin": {"iteration", "present_factor", "nets"},
    "net_route_begin": {"net", "iteration", "driver", "sinks", "fanout"},
    "branch_route_begin": {"net", "iteration", "sink", "goal"},
    "route_search_expand": {"net", "iteration", "sink", "coord", "g", "h", "f", "frontier"},
    "route_transition_blocked": {"net", "iteration", "sink", "from", "to", "reason"},
    "branch_search_stats": {"net", "iteration", "sink", "attempt", "mode", "expansions", "found", "blocked"},
    "branch_search_relaxed": {"net", "iteration", "sink", "mode", "expansions"},
    "net_route_restarted": {"net", "iteration", "first_sink", "restart"},
    "branch_path_rejected": {"net", "iteration", "sink", "reason", "coord", "path"},
    "branch_route_found": {"net", "iteration", "sink", "start", "goal", "path", "length"},
    "branch_route_failed": {"net", "iteration", "sink", "goal", "reason", "expansions", "message"},
    "net_route_committed": ROUTE[0],
    "net_rip_up": {"net", "iteration", "reason", "cells", "length"},
    "routing_iteration_end": {"iteration", "present_factor", "rerouted", "changed", "conflicts", "conflict_cells",
                              "routed_cells", "history_total"},
    "congestion_snapshot": {"iteration", "present_factor", "conflicts", "cells", "history_cells"},
    "physical_conflict": {"iteration", "kind", "cells", "nets"},
    "keyframe": {"iteration", "placement", "routes", "realized", "congestion"},
    "routing_complete": {"iteration", "iterations", "routed_nets", "rip_ups", "failure"},
    "routing_failed": {"iteration", "iterations", "routed_nets", "rip_ups", "failure"},
    "legalization_begin": {"round", "nets"},
    "route_legalization_begin": {"net", "round", "length", "drive"},
    "signal_strength_scan": {"net", "round", "powered", "cells"},
    "repeater_inserted": {"net", "round", "coord", "facing", "input_strength", "delay"},
    "legalization_failed": {"net", "round", "reason", "message", "coord", "sink"},
    "route_realized": REALIZED[0],
    "legalization_complete": {"round", "realized", "failures", "repeaters"},
    "illegal_transition": {"kind", "message", "cells", "nets"},
    "design_finalized": {"attempt", "nets", "dust", "repeaters", "supports", "bounds"},
    "pnr_attempt_end": {"attempt", "status", "failure", "metrics"},
    "pnr_end": {"success", "attempts"},
}  # fmt: skip


class Run(NamedTuple):
    graph: Graph
    netlist: PrimitiveNetlist
    mapped: PrimitivePhysicalNetlist
    result: PrimitivePnRResult
    trace: dict[str, Any]


def pnr(source: str = HALF_ADDER, level: str = "basic", *, interface=PADS, **config: Any) -> Run:
    """Compile ``source`` and run the whole primitive pipeline with a fresh recorder."""
    graph = compile_source(source)
    recorder = PrimitiveTraceRecorder(level)
    netlist, mapped, result = place_and_route_graph(
        graph,
        PrimitivePnRConfig(trace_level=level, **config),
        interface=interface,
        trace=recorder,
        source="examples/test.redc",
        top="main",
    )
    return Run(graph, netlist, mapped, result, recorder.to_dict())


#: Read-only runs shared between tests (nothing below mutates a cached run).
cached = functools.cache(pnr)


def documented_events() -> dict[str, tuple[str, str]]:
    """``type -> (phase, lowest level)``, parsed from the Events table of the doc."""
    table: dict[str, tuple[str, str]] = {}
    for line in DOC.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\| (`\w+`(?: / `\w+`)*) \| (\w+) \| (\w+)", line)
        if match:
            for name in re.findall(r"`(\w+)`", match.group(1)):
                table[name] = (match.group(2), match.group(3))
    return table


def of_type(trace: dict[str, Any], *types: str) -> list[dict[str, Any]]:
    return [e for e in trace["events"] if e["type"] in types]


def payload(event: dict[str, Any], *extra: str) -> dict[str, Any]:
    """An event without its envelope (``seq`` / ``phase`` / ``type``) and ``extra`` keys."""
    return {k: v for k, v in event.items() if k not in ENVELOPE and k not in extra}


def attempts_of(trace: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """The events of each P&R attempt, ``pnr_attempt_begin`` .. ``pnr_attempt_end``."""
    blocks: list[list[dict[str, Any]]] = []
    inside = False
    for event in trace["events"]:
        if event["type"] == "pnr_attempt_begin":
            blocks.append([])
            inside = True
        if inside:
            blocks[-1].append(event)
        if event["type"] == "pnr_attempt_end":
            inside = False
    return blocks


def fields(value: Any):
    """Yield ``(owning dict, key, value)`` for every nested dict field."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield value, key, item
            yield from fields(item)
    elif isinstance(value, list):
        for item in value:
            yield from fields(item)


def is_coord(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(type(c) is int for c in value)


def assert_plain_json(value: Any) -> None:
    if isinstance(value, dict):
        assert all(isinstance(k, str) for k in value), list(value)
        for item in value.values():
            assert_plain_json(item)
    elif isinstance(value, list):
        for item in value:
            assert_plain_json(item)
    else:
        assert value is None or type(value) in (bool, int, float, str), repr(value)


def rotate(local: list[int], orientation: str) -> tuple[int, int, int]:
    """``orientation`` applied to a local coordinate: ``south`` is ``(x, y, z) -> (-z, y, x)``."""
    x, y, z = local
    for _ in range(ORIENTATIONS.index(orientation)):
        x, z = -z, x
    return (x, y, z)


def absolute(local: list[int], origin: list[int], orientation: str) -> list[int]:
    x, y, z = rotate(local, orientation)
    return [origin[0] + x, origin[1] + y, origin[2] + z]


def bounds_of(cells: list[list[int]]) -> dict[str, Any]:
    lo = [min(c[i] for c in cells) for i in range(3)]
    hi = [max(c[i] for c in cells) for i in range(3)]
    dims = [hi[i] - lo[i] + 1 for i in range(3)]
    return {"min": lo, "max": hi, "dims": dims, "volume": dims[0] * dims[1] * dims[2]}


def cell_blocks(cell: dict[str, Any], origin: list[int], orientation: str) -> dict[str, Any]:
    """Absolute voxels / keep-out / pin blocks of a placed cell, from its JSON definition."""
    return {
        "voxels": sorted(absolute(v["coord"], origin, orientation) for v in cell["voxels"]),
        "keepout": sorted(absolute(c, origin, orientation) for c in cell["keepout"]),
        "pins": {p["name"]: absolute(p["position"], origin, orientation) for p in cell["pins"]},
        "facing": {
            p["name"]: ORIENTATIONS[(ORIENTATIONS.index(p["facing"]) + ORIENTATIONS.index(orientation)) % 4]
            for p in cell["pins"]
        },
    }


def placed_cells(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Every placed instance's absolute blocks, computed from the trace alone."""
    cells = {c["name"]: c for c in trace["design"]["cells"]}
    instances = {i["id"]: i for i in trace["design"]["instances"]}
    return {
        p["instance"]: cell_blocks(cells[instances[p["instance"]]["cell"]], p["origin"], p["orientation"])
        for p in trace["final"]["placement"]
        if p["origin"] is not None
    }


def replay(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply ``events`` as the doc's "Replaying" section says; the state after the last one."""

    def fresh(attempt: int | None) -> dict[str, Any]:
        return {"attempt": attempt, "placement": {}, "routes": {}, "realized": {}, "congestion": None}

    state = fresh(None)
    for event in events:
        kind = event["type"]
        if kind == "pnr_attempt_begin":
            state = fresh(event["attempt"])
        elif kind == "component_placed":
            state["placement"][event["instance"]] = (event["origin"], event["orientation"])
        elif kind == "net_route_committed":
            state["routes"][event["net"]] = payload(event, "iteration")
        elif kind == "net_rip_up":
            assert event["net"] in state["routes"], f"rip-up of net {event['net']}, which has no route"
            del state["routes"][event["net"]]
            state["realized"].pop(event["net"], None)
        elif kind == "route_realized":
            state["realized"][event["net"]] = payload(event, "round")
        elif kind == "congestion_snapshot":
            state["congestion"] = event["conflicts"]
        elif kind == "keyframe":
            state["routes"] = {route["net"]: route for route in event["routes"]}
    return state


def route_parents(route: dict[str, Any]) -> dict[tuple[int, ...], tuple[int, ...] | None]:
    parent: dict[tuple[int, ...], tuple[int, ...] | None] = {tuple(route["root"]): None}
    for branch in route["branches"]:
        for a, b in pairwise(branch["path"]):
            parent.setdefault(tuple(b), tuple(a))
    return parent


def pin_index(trace: dict[str, Any]) -> dict[int, dict[str, dict[str, Any]]]:
    """``instance -> pin name -> pin record`` from the header's cell definitions."""
    cells = {c["name"]: c for c in trace["design"]["cells"]}
    return {i["id"]: {p["name"]: p for p in cells[i["cell"]]["pins"]} for i in trace["design"]["instances"]}


def assert_shape(record: dict[str, Any], name: str) -> None:
    required, optional = RECORDS[name]
    assert required <= set(record) <= required | optional, (name, sorted(set(record) ^ required))


def assert_event_fields(events: list[dict[str, Any]]) -> set[str]:
    """Every DOCUMENTED event carries its documented payload; returns the types
    checked (undocumented types fail ``test_events_are_sequenced_documented...``)."""
    checked = set()
    for event in events:
        if event["type"] in EVENT_FIELDS:
            missing = EVENT_FIELDS[event["type"]] - set(event)
            assert not missing, (event["type"], sorted(missing))
            checked.add(event["type"])
    return checked


def force_legalization_failure(monkeypatch: pytest.MonkeyPatch, net: int, times: int | None = None) -> dict:
    """Make electrical legalization reject ``net`` (the first ``times`` times;
    ``None`` = always), as if no repeater site could keep its signal alive."""
    real = pnr_design.legalize_route
    calls = {"forced": 0}

    def legalize(tree, request):
        if tree.net == net and (times is None or calls["forced"] < times):
            calls["forced"] += 1
            return LegalizationFailure(tree.net, "signal too weak (forced)", tree.root, None, tuple(tree.cells))
        return real(tree, request)

    monkeypatch.setattr(pnr_design, "legalize_route", legalize)
    return calls


# -- header ---------------------------------------------------------------------------


def test_documented_event_table_parses() -> None:
    table = documented_events()
    assert len(table) >= 40
    assert table["route_search_expand"] == ("routing", "search")
    assert table["routing_failed"] == table["routing_complete"] == ("routing", "basic")
    assert {phase for phase, _ in table.values()} == set(PHASES)
    assert {level for _, level in table.values()} == {"basic", "detailed", "search"}


def test_header_identifies_a_block_resolution_primitive_trace() -> None:
    run = cached()
    trace = run.trace
    assert trace["schema"] == TRACE_SCHEMA == "redc.physical-primitive.pnr.v1"
    assert trace["generator"] == "redc"
    assert trace["backend"] == BACKEND == "physical-primitive"
    assert trace["trace_level"] == "basic"
    system = trace["coordinate_system"]
    assert (system["units"], system["x"], system["y"], system["z"]) == ("blocks", "east", "up", "south")
    assert trace["source"] == {"path": "examples/test.redc", "top": "main"}
    assert trace["ir"] == run.netlist.graph_summary
    assert trace["ir"]["live_nodes"] == len(run.graph.live_nodes())
    assert trace["ir"]["sequential"] is False
    assert {"library", "cells", "instances", "nets", "groups", "ports", "buses", "ir_nodes", "summary"} <= set(
        trace["design"]
    )
    assert trace["config"] == PrimitivePnRConfig().to_dict()
    assert trace["config"]["units"] == "blocks" and trace["config"]["trace_level"] == "basic"
    assert list(trace)[-2:] == ["events", "final"]
    assert trace["final"]["success"] is True


@pytest.mark.parametrize(
    ("source", "level", "config"),
    [
        (HALF_ADDER, "basic", {}),
        (HALF_ADDER, "detailed", {}),
        (HALF_ADDER, "search", {}),
        (NOT, "basic", {"max_pnr_attempts": 3, **NO_EXPANSIONS}),
    ],
    ids=["basic", "detailed", "search", "astar-budget-exhausted"],
)
def test_events_are_sequenced_documented_and_level_filtered(source: str, level: str, config: dict) -> None:
    trace = cached(source, level, **config).trace
    events = trace["events"]
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert all(list(e)[:3] == list(ENVELOPE) for e in events)
    table = documented_events()
    for event in events:
        assert event["type"] in table, f"undocumented event type {event['type']!r}"
        phase, lowest = table[event["type"]]
        assert event["phase"] == phase in PHASES, event["type"]
        assert LEVELS.index(lowest) <= LEVELS.index(level), f"{event['type']} recorded at {level}"
    # Pipeline order: synthesis, then technology mapping, then place and route.
    phases = [e["phase"] for e in events]
    assert phases[0] == "synthesis" and events[0]["type"] == "synthesis_begin"
    last_synthesis = max(i for i, p in enumerate(phases) if p == "synthesis")
    techmap = [i for i, p in enumerate(phases) if p == "techmap"]
    assert last_synthesis < min(techmap) and max(techmap) < phases.index("pnr")
    assert events[-1]["type"] == "pnr_end"


def test_header_design_is_unplaced_and_self_contained() -> None:
    run = cached(FULL_ADDER)
    design = run.trace["design"]
    assert design["library"] == run.mapped.library == "redc-primitive-placeholder-v1"
    cells = {c["name"]: c for c in design["cells"]}
    assert len(cells) == len(design["cells"])
    assert [i["id"] for i in design["instances"]] == list(range(len(run.mapped.instances)))
    for inst, mapped in zip(design["instances"], run.mapped.instances, strict=True):
        assert inst["origin"] is None and inst["orientation"] is None  # placement is an EVENT
        assert inst["realizes"] == [inst["id"]]
        assert inst["kind"] == mapped.kind.value and inst["category"] == KIND_CATEGORY[inst["kind"]]
        cell = cells[inst["cell"]]
        assert cell["kind"] == inst["kind"] and cell["placeholder"] is True and cell["structure"] is None
        assert "east" in cell["orientations"] and set(cell["orientations"]) <= set(ORIENTATIONS)
    for cell in cells.values():
        assert cell["voxels"] and all(set(v) == {"coord", "role"} for v in cell["voxels"])
        occupied = {tuple(v["coord"]) for v in cell["voxels"]}
        assert not occupied & {tuple(c) for c in cell["keepout"]}
        for pin in cell["pins"]:
            assert pin["direction"] in ("in", "out") and pin["facing"] in ORIENTATIONS
            x, y, z = pin["position"]
            assert (x, y - 1, z) in occupied  # the endpoint dust rests on the cell's own base
            assert 0 <= pin["strength"] <= 15
    pins = pin_index(run.trace)
    assert [n["id"] for n in design["nets"]] == list(range(len(run.mapped.nets)))
    for net in design["nets"]:
        assert net["width"] == 1 and net["fanout"] == len(net["sinks"]) >= 1
        assert pins[net["driver"]["instance"]][net["driver"]["pin"]]["direction"] == "out"
        assert all(pins[s["instance"]][s["pin"]]["direction"] == "in" for s in net["sinks"])
    groups = design["groups"]
    assert [g["id"] for g in groups] == list(range(len(groups)))
    assert groups[0]["kind"] == "root" and groups[0]["parent"] is None
    assert all(g["kind"] in GROUP_KINDS for g in groups)
    assert all(g["parent"] is None or g["parent"] < g["id"] for g in groups)
    assert {n["id"] for n in design["ir_nodes"]} == {n["id"] for n in run.graph.live_nodes()}
    assert [p["name"] for p in design["ports"]] == ["in_a", "in_b", "in_c", "result"]
    assert all(p["realization"] == "pads" for p in design["ports"])
    assert design["summary"] == run.netlist.summary()
    # Provenance: ir_op / ir_type name the IR node, and the group belongs to it.
    nodes = {n["id"]: n for n in design["ir_nodes"]}
    for inst in design["instances"]:
        node = nodes.get(inst["ir_node"])
        assert (inst["ir_op"], inst["ir_type"]) == ((node["op"], node["type"]["name"]) if node else (None, None))
        assert groups[inst["group"]]["ir_node"] == inst["ir_node"]
        assert cells[inst["cell"]]["stateful"] is (inst["kind"] == "register_bit")
    # Bits are LSB first: bit i of a pad port is the pad of bit i.
    for port in design["ports"]:
        records = [design["instances"][b["instance"]] for b in port["bits"]]
        assert [(r["port"], r["bit"]) for r in records] == [(port["name"], i) for i in range(len(records))]


def test_every_referenced_id_exists() -> None:
    trace = cached(COUNTER, "detailed").trace
    design = trace["design"]
    pins = pin_index(trace)
    nets = {n["id"] for n in design["nets"]}
    groups = {g["id"] for g in design["groups"]}
    ir_nodes = {n["id"] for n in design["ir_nodes"]}
    ports = {p["name"] for p in design["ports"]}
    for record in design["instances"]:
        assert record["group"] in groups and (record["ir_node"] is None or record["ir_node"] in ir_nodes)
    for net in design["nets"]:
        assert all(entry["ir_node"] in ir_nodes for entry in net["logical"])
        assert all(entry["port"] in ports for entry in net["ports"])
    for record in (*design["ports"], *design["buses"]):
        assert all(bit["pin"] in pins[bit["instance"]] for bit in record["bits"])
    checked = Counter()
    for owner, key, value in fields({"events": trace["events"], "final": trace["final"]}):
        if value is None:
            continue
        if key == "instance":
            assert value in pins, (owner.get("type"), value)
            if "pin" in owner:
                assert owner["pin"] in pins[value], owner
        elif key in ("first_instance", "last_instance"):
            assert value in pins
        elif key == "net":
            assert value in nets, (owner.get("type"), value)
        elif key in ("nets", "order") and isinstance(value, list):
            assert set(value) <= nets, (owner.get("type"), key)
        elif key == "instances" and isinstance(value, list):
            assert set(value) <= set(pins), owner.get("type")
        elif key == "group":
            assert value in groups
        elif key == "ir_node":
            assert value in ir_nodes
        else:
            continue
        checked[key] += 1
    assert {"instance", "net", "nets", "order", "instances", "group", "ir_node"} <= set(checked)


def test_records_have_the_documented_shape() -> None:
    trace = cached(COUNTER, "detailed").trace
    design, final = trace["design"], trace["final"]
    assert_shape(trace, "header")
    assert_shape(trace["coordinate_system"], "coordinate_system")
    assert_shape(trace["ir"], "ir")
    assert_shape(design, "design")
    for name, records in (("cell", design["cells"]), ("instance", design["instances"]), ("net", design["nets"]),
                          ("group", design["groups"]), ("port", design["ports"]), ("bus", design["buses"]),
                          ("ir_node", design["ir_nodes"])):  # fmt: skip
        assert records
        for record in records:
            assert_shape(record, name)
    for cell in design["cells"]:
        for pin in cell["pins"]:
            assert_shape(pin, "pin")
    assert_shape(final, "final")
    assert_shape(final["geometry"], "geometry")
    for entry in final["placement"]:
        assert_shape(entry, "placement")
        assert set(entry["bounds"]) == BOUNDS[0]
    for bounds in (final["design_bounds"], final["search_bounds"]):
        assert set(bounds) == BOUNDS[0]
    for route in final["routes"]:
        assert_shape(route, "route")
        for branch in route["branches"]:
            assert_shape(branch, "branch")
    for realized in final["realized"]:
        assert_shape(realized, "realized")
        for element in realized["elements"]:
            assert_shape(element, "element")
            assert ("facing" in element) is (element["kind"] == "repeater")
        for report in realized["sinks"]:
            assert_shape(report, "sink")
    failed = cached(NOT, max_pnr_attempts=3, **NO_EXPANSIONS).trace["final"]
    assert_shape(failed, "final")
    assert_shape(failed["failure"], "failure")


def test_event_payloads_carry_the_documented_fields() -> None:
    runs = [
        cached(HALF_ADDER, "search"),
        cached(COUNTER, "detailed"),
        cached(FULL_ADDER, "basic", keyframe_interval=1),
        cached(NOT, max_pnr_attempts=3, **NO_EXPANSIONS),
        cached(DISPLAY, interface=DefaultInterfacePolicy(), max_y=3, retry_height_growth=2, max_pnr_attempts=3),
    ]
    checked = set().union(*(assert_event_fields(run.trace["events"]) for run in runs))
    table = documented_events()
    assert set(EVENT_FIELDS) == set(table), sorted(set(EVENT_FIELDS) ^ set(table))
    # Events that only a forced failure produces are covered by the tests below.
    assert set(table) - checked <= {"component_place_rejected", "legalization_failed", "illegal_transition",
                                    "branch_path_rejected", "physical_conflict", "net_route_restarted"}  # fmt: skip


#: ``(event type, key)`` pairs where a coordinate-list key holds a COUNT instead.
COUNT_FIELDS = {("component_place_attempt", "keepout"), ("design_finalized", "supports")}


@pytest.mark.parametrize(
    ("source", "level", "config", "expected"),
    [
        (HALF_ADDER, "search", {}, {"coord", "origin", "root", "path", "cells", "voxels", "from", "to"}),
        (COUNTER, "detailed", {}, {"coord", "origin", "root", "start", "goal", "parent", "supports", "voxels"}),
        (NOT, "detailed", NO_EXPANSIONS, {"coord", "origin", "goal", "voxels", "position", "min", "max"}),
    ],
    ids=["search", "detailed-sequential", "failure"],
)
def test_every_coordinate_is_an_xyz_int_triple(source: str, level: str, config: dict, expected: set) -> None:
    trace = cached(source, level, **config).trace
    seen = Counter()
    for owner, key, value in fields({k: v for k, v in trace.items() if k != "source"}):  # source.path is a file
        if value is None:
            continue
        if key in COORD_KEYS:
            assert is_coord(value), (owner.get("type"), key, value)
        elif key == "parent" and owner.get("kind") in ("dust", "repeater"):
            assert is_coord(value), owner
        elif key in COORD_LIST_KEYS:
            if key == "cells" and (owner is trace["design"] or isinstance(value, dict)):
                continue  # cell DEFINITIONS (walked below) or a {cell name: count} tally
            if (owner.get("type"), key) in COUNT_FIELDS:
                assert type(value) is int
            elif owner.get("type") == "signal_strength_scan":  # [x, y, z, strength]
                assert all(len(c) == 4 and is_coord(c[:3]) and type(c[3]) is int for c in value)
            else:
                for item in value:
                    assert is_coord(item) or (isinstance(item, dict) and is_coord(item["coord"])), (key, item)
        else:
            continue
        seen[key] += 1
    assert expected <= set(seen)


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_trace_is_json_native_and_round_trips(tmp_path: Path, suffix: str) -> None:
    run = cached(COUNTER)
    trace = run.trace
    assert_plain_json(trace)
    json.dumps(trace, allow_nan=False)  # no NaN / Infinity either
    assert json.loads(run.result.trace.to_json()) == trace
    path = run.result.trace.write(tmp_path / "nested" / f"counter.primitive.pnr{suffix}")
    assert load_trace(path) == trace
    if suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [r["record"] for r in records] == ["header"] + ["event"] * len(trace["events"]) + ["final"]
        assert records[0]["schema"] == TRACE_SCHEMA and "events" not in records[0]
        assert records[-1]["final"] == trace["final"]
    # Metrics survive the round trip unchanged (no tuples, no non-string keys).
    assert load_trace(path)["final"]["metrics"] == run.result.metrics


def test_load_trace_rejects_other_documents(tmp_path: Path) -> None:
    for name, text in (
        ("bogus.json", '{"schema": "nope"}'),
        ("coarse.json", '{"schema": "redc.pnr.trace.v1", "events": []}'),
        ("design.json", '{"schema": "redc.physical-primitive.v1"}'),
        ("bogus.jsonl", '{"record": "header", "schema": "redc.pnr.trace.v1"}\n'),
    ):
        (tmp_path / name).write_text(text, encoding="utf-8")
        with pytest.raises(CompileError, match="not a redc.physical-primitive.pnr.v1 trace"):
            load_trace(tmp_path / name)


# -- synthesis and technology mapping ----------------------------------------------


def test_synthesis_events_cover_every_live_ir_node() -> None:
    run = cached(COUNTER, "detailed")
    trace, live = run.trace, run.graph.live_nodes()
    design = trace["design"]
    (begin,) = of_type(trace, "synthesis_begin")
    assert begin["live_nodes"] == len(live)
    assert begin["ops"] == dict(sorted(Counter(n["op"] for n in live).items()))
    assert begin["sequential"] is True
    synthesized = of_type(trace, "ir_node_synthesized")
    assert {e["ir_node"] for e in synthesized} == {n["id"] for n in live}
    passes = Counter((e["op"], e["synthesis_pass"]) for e in synthesized)
    registers = sum(1 for n in live if n["op"] == "register")
    assert passes[("register", "A")] == passes[("register", "D")] == registers > 0
    nodes = {n["id"]: n for n in design["ir_nodes"]}
    groups = {g["id"]: g for g in design["groups"]}
    covered: list[int] = []
    for event in synthesized:
        node = nodes[event["ir_node"]]
        assert event["op"] == node["op"] and event["ir_type"] == node["type"]["name"]
        assert event["synthesis_pass"] in ("A", "B", "D")
        is_recipe = event["synthesis_pass"] == "B" and node["op"] != "const"
        assert (event["recipe"] is not None) == is_recipe, event
        assert groups[event["group"]]["ir_node"] == event["ir_node"]
        made = list(range(event["first_instance"], event["last_instance"] + 1)) if event["primitives"] else []
        assert len(made) == event["primitives"]
        assert event["by_kind"] == dict(sorted(Counter(design["instances"][i]["kind"] for i in made).items()))
        covered.extend(made)
    # Every primitive is reported exactly once, except the output pads (pass C).
    outputs = [i["id"] for i in design["instances"] if i["kind"] == "output_bit"]
    assert sorted(covered + outputs) == list(range(len(design["instances"])))
    (complete,) = of_type(trace, "synthesis_complete")
    assert payload(complete) == run.netlist.summary() == design["summary"]
    types = [e["type"] for e in trace["events"]]
    assert types.index("synthesis_complete") > max(i for i, t in enumerate(types) if t == "ir_node_synthesized")
    # DETAILED: one primitive_emitted per primitive, in id order, matching the header.
    emitted = of_type(trace, "primitive_emitted")
    assert [e["instance"] for e in emitted] == list(range(len(design["instances"])))
    for event, record in zip(emitted, design["instances"], strict=True):
        assert (event["kind"], event["ir_node"], event["role"], event["bit"], event["group"]) == (
            record["kind"], record["ir_node"], record["role"], record["bit"], record["group"]
        )  # fmt: skip


def test_techmap_events_describe_the_chosen_cells() -> None:
    trace = cached(COUNTER, "detailed").trace
    design = trace["design"]
    cells = {c["name"]: c for c in design["cells"]}
    (complete,) = of_type(trace, "techmap_complete")
    assert complete["library"] == design["library"]
    assert complete["instances"] == len(design["instances"]) and complete["nets"] == len(design["nets"])
    assert complete["cells"] == dict(sorted(Counter(i["cell"] for i in design["instances"]).items()))
    assert complete["placeholder_cells"] == sorted(cells)  # every v1 cell is a placeholder
    assert complete["component_voxels"] == sum(len(cells[i["cell"]]["voxels"]) for i in design["instances"])
    mapped = of_type(trace, "primitive_mapped")
    assert [e["instance"] for e in mapped] == list(range(len(design["instances"])))
    for event, record in zip(mapped, design["instances"], strict=True):
        assert event["kind"] == record["kind"] and event["cell"] == record["cell"]
        assert event["cell"] in event["candidates"]
    types = [e["type"] for e in trace["events"]]
    assert types.index("synthesis_complete") < types.index("techmap_complete") < types.index("pnr_begin")


# -- attempts --------------------------------------------------------------------------


def test_every_failed_attempt_stays_in_the_trace() -> None:
    run = cached(NOT, max_pnr_attempts=3, **NO_EXPANSIONS)
    trace, result = run.trace, run.result
    config = PrimitivePnRConfig(max_pnr_attempts=3, **NO_EXPANSIONS)
    assert not result.success and result.attempts == 3
    (begin,) = of_type(trace, "pnr_begin")
    assert begin["max_attempts"] == 3
    assert begin["instances"] == len(run.mapped.instances) and begin["nets"] == len(run.mapped.nets)
    begins = of_type(trace, "pnr_attempt_begin")
    assert [payload(e) for e in begins] == [config.attempt_geometry(k).to_dict() for k in range(3)]
    ends = of_type(trace, "pnr_attempt_end")
    assert [(e["attempt"], e["status"]) for e in ends] == [(0, "failed"), (1, "failed"), (2, "failed")]
    for k, end in enumerate(ends):
        failure = end["failure"]
        assert (failure["stage"], failure["reason"], failure["attempt"], failure["attempts"]) == (
            "routing", "unroutable", k, k + 1
        )  # fmt: skip
        assert failure["message"] and end["metrics"]["pnr"] == {"attempts": k + 1, "attempt": k}
    assert len(attempts_of(trace)) == 3 and all(b[-1]["type"] == "pnr_attempt_end" for b in attempts_of(trace))
    # Each attempt stops at its first unroutable branch.
    assert [e["reason"] for e in of_type(trace, "branch_route_failed")] == ["expansion_limit"] * 3
    (end,) = of_type(trace, "pnr_end")
    assert (end["success"], end["attempts"]) == (False, 3)
    final = trace["final"]
    assert (final["success"], final["attempt"], final["attempts"]) == (False, 2, 3)
    assert final["geometry"] == config.attempt_geometry(2).to_dict()
    assert final["failure"] == ends[-1]["failure"] == result.failure.to_dict()


def test_a_later_attempt_can_succeed_after_a_failed_one() -> None:
    run = cached(DISPLAY, interface=DefaultInterfacePolicy(), max_y=3, retry_height_growth=2, max_pnr_attempts=3)
    trace, result = run.trace, run.result
    assert result.success and result.attempts == 2 and result.geometry.attempt == 1
    begins = of_type(trace, "pnr_attempt_begin")
    assert [(e["attempt"], e["max_y"]) for e in begins] == [(0, 3), (1, 5)]
    assert begins[1]["component_spacing"] > begins[0]["component_spacing"]
    ends = of_type(trace, "pnr_attempt_end")
    assert [e["status"] for e in ends] == ["failed", "success"]
    assert ends[0]["failure"]["stage"] == "placement" and ends[1]["failure"] is None
    (failed,) = of_type(trace, "placement_failed")
    assert failed["attempt"] == 0 and failed["reason"] == ends[0]["failure"]["message"]
    first, second = attempts_of(trace)
    assert "routing_begin" not in {e["type"] for e in first}
    assert {"routing_complete", "design_finalized"} <= {e["type"] for e in second}
    (finalized,) = of_type(trace, "design_finalized")
    assert finalized["attempt"] == 1
    final = trace["final"]
    assert (final["success"], final["attempt"], final["attempts"], final["failure"]) == (True, 1, 2, None)
    state = replay(trace["events"])
    assert state["attempt"] == 1 and len(state["placement"]) == len(run.mapped.instances)


# -- placement -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "config"),
    [(FULL_ADDER, {}), (NOT, {"max_pnr_attempts": 2, **NO_EXPANSIONS})],
    ids=["ok", "retried"],
)
def test_every_instance_is_placed_once_per_attempt(source: str, config: dict) -> None:
    run = cached(source, **config)
    trace = run.trace
    cells = {c["name"]: c for c in trace["design"]["cells"]}
    instances = trace["design"]["instances"]
    blocks = attempts_of(trace)
    assert len(blocks) == run.result.attempts
    for attempt, block in enumerate(blocks):
        (begin,) = [e for e in block if e["type"] == "placement_begin"]
        assert begin["attempt"] == attempt
        column_of = {i: c["column"] for c in begin["columns"] for i in c["instances"]}
        assert sorted(column_of) == list(range(len(instances)))
        placed = [e for e in block if e["type"] == "component_placed"]
        assert sorted(e["instance"] for e in placed) == list(range(len(instances)))
        for event in placed:
            cell = cells[instances[event["instance"]]["cell"]]
            assert event["column"] == column_of[event["instance"]]
            assert event["orientation"] in cell["orientations"]
            here = cell_blocks(cell, event["origin"], event["orientation"])
            assert event["bounds"] == bounds_of(here["voxels"] + here["keepout"] + list(here["pins"].values()))
        (complete,) = [e for e in block if e["type"] == "placement_complete"]
        assert complete["attempt"] == attempt and complete["component_count"] == len(instances)
        assert complete["probes"] >= len(instances)
        assert complete["component_voxels"] == sum(len(cells[i["cell"]]["voxels"]) for i in instances)
        keepout = set()
        for event in placed:
            cell = cells[instances[event["instance"]]["cell"]]
            keepout.update(map(tuple, cell_blocks(cell, event["origin"], event["orientation"])["keepout"]))
        assert complete["keepout_voxels"] == len(keepout)
        changes = [e for e in block if e["type"] == "design_bounds_changed"]
        assert changes and payload(changes[-1]) == complete["bounds"]
        everything = [c for e in placed for c in (e["bounds"]["min"], e["bounds"]["max"])]
        assert complete["bounds"] == bounds_of(everything)
    final = {p["instance"]: p for p in trace["final"]["placement"]}
    last = {e["instance"]: e for e in blocks[-1] if e["type"] == "component_placed"}
    assert {i: (p["origin"], p["orientation"], p["bounds"]) for i, p in final.items()} == {
        i: (e["origin"], e["orientation"], e["bounds"]) for i, e in last.items()
    }


def test_detailed_placement_traces_every_probe() -> None:
    trace = cached(FULL_ADDER, "detailed").trace
    cells = {c["name"]: c for c in trace["design"]["cells"]}
    instances = trace["design"]["instances"]
    placement = [e for e in trace["events"] if e["phase"] == "placement" and e["type"] != "design_bounds_changed"]
    probes = 0
    for event, after in pairwise(placement):
        if event["type"] != "component_place_attempt":
            continue
        probes += 1
        cell = cells[instances[event["instance"]]["cell"]]
        here = cell_blocks(cell, event["origin"], event["orientation"])
        assert event["voxels"] == here["voxels"]
        assert event["keepout"] == len(cell["keepout"])
        assert event["bounds"] == bounds_of(here["voxels"] + here["keepout"] + list(here["pins"].values()))
        # A probe is resolved right away: rejected, or placed at that very origin.
        assert after["type"] in ("component_place_rejected", "component_placed")
        assert (after["instance"], after["origin"], after["orientation"]) == (
            event["instance"], event["origin"], event["orientation"]
        )  # fmt: skip
    (complete,) = of_type(trace, "placement_complete")
    assert probes == complete["probes"] == len(of_type(trace, "component_place_attempt")) > 0
    assert len(of_type(trace, "component_placed")) + len(of_type(trace, "component_place_rejected")) == probes


def test_a_rejected_placement_probe_records_its_reason() -> None:
    mapped = map_primitives_to_minecraft(synthesize_to_primitives(compile_source(NOT), interface=PADS))
    geometry = PrimitivePnRConfig().attempt_geometry(0)
    scout = PrimitiveTraceRecorder("detailed")
    place_design(mapped, BlockGrid(max_y=geometry.max_y), geometry, scout)
    first = next(e for e in scout.events if e["type"] == "component_place_attempt")
    # Same placement again, with a foreign block sitting in the first probe's slot.
    for inst in mapped.instances:
        inst.unplace()
    grid = BlockGrid(max_y=geometry.max_y)
    grid.place(999, [tuple(first["voxels"][0])], [], [])
    trace = PrimitiveTraceRecorder("detailed")
    placed = place_design(mapped, grid, geometry, trace)
    assert "component_place_rejected" in assert_event_fields(trace.events)
    mine = [e for e in trace.events if e.get("instance") == first["instance"]]
    assert [e["type"] for e in mine[:2]] == ["component_place_attempt", "component_place_rejected"]
    rejected = mine[1]
    assert (rejected["origin"], rejected["orientation"]) == (first["origin"], first["orientation"])
    assert isinstance(rejected["reason"], str) and rejected["reason"]
    assert mine[-1]["type"] == "component_placed" and mine[-1]["origin"] != first["origin"]
    assert mapped.instances[first["instance"]].origin == tuple(mine[-1]["origin"])
    attempts = [e for e in trace.events if e["type"] == "component_place_attempt"]
    (complete,) = [e for e in trace.events if e["type"] == "placement_complete"]
    assert placed.probes == complete["probes"] == len(attempts) > len(mapped.instances)


# -- routing ---------------------------------------------------------------------------


def assert_route_tree(route: dict[str, Any], net: dict[str, Any], blocks: dict[int, dict[str, Any]]) -> None:
    """A route tree record against the doc: rooted at the driver pin block, one
    ordered staircase path per sink, attached to the tree built so far."""
    driver = net["driver"]
    assert route["net"] == net["id"] and route["driver"] == driver
    assert route["root"] == blocks[driver["instance"]]["pins"][driver["pin"]]
    assert route["fanout"] == len(route["branches"]) == net["fanout"]
    assert sorted((b["sink"]["instance"], b["sink"]["pin"]) for b in route["branches"]) == sorted(
        (s["instance"], s["pin"]) for s in net["sinks"]
    )
    tree = [route["root"]]
    for branch in route["branches"]:
        path, sink = branch["path"], branch["sink"]
        assert path[0] == branch["start"] and path[-1] == branch["goal"]
        assert branch["start"] in tree  # attaches to the tree built so far
        assert branch["goal"] == blocks[sink["instance"]]["pins"][sink["pin"]]
        for a, b in pairwise(path):
            assert abs(a[0] - b[0]) + abs(a[2] - b[2]) == 1 and abs(a[1] - b[1]) <= 1, (a, b)
        tree.extend(c for c in path if c not in tree)
    assert route["cells"] == tree and route["length"] == len(tree)
    # Routes leave the driver and enter every sink along the pin's facing.
    parent = route_parents(route)
    children = [c for c, up in parent.items() if up == tuple(route["root"])]
    out = DIRECTIONS[blocks[driver["instance"]]["facing"][driver["pin"]]]
    assert all((c[0] - route["root"][0], c[2] - route["root"][2]) == (out[0], out[2]) for c in children)
    for branch in route["branches"]:
        sink, goal = branch["sink"], branch["goal"]
        facing = DIRECTIONS[blocks[sink["instance"]]["facing"][sink["pin"]]]
        up = parent[tuple(goal)]
        assert (up[0] - goal[0], up[2] - goal[2]) == (facing[0], facing[2])


def test_routing_events_build_a_tree_for_every_net() -> None:
    run = cached(FULL_ADDER)
    trace, config = run.trace, PrimitivePnRConfig()
    nets = {n["id"]: n for n in trace["design"]["nets"]}
    blocks = placed_cells(trace)
    (begin,) = of_type(trace, "routing_begin")
    assert begin["nets"] == len(nets) and sorted(begin["order"]) == sorted(nets)
    assert begin["present_factor"] == config.present_factor_initial
    assert begin["search_bounds"] == trace["final"]["search_bounds"]
    starts, ends = of_type(trace, "routing_iteration_begin"), of_type(trace, "routing_iteration_end")
    assert [e["iteration"] for e in starts] == [e["iteration"] for e in ends] == list(range(len(starts)))
    assert starts[0]["nets"] == begin["order"]
    snapshots = of_type(trace, "congestion_snapshot")
    assert [s["iteration"] for s in snapshots] == [e["iteration"] for e in ends]
    for end, snapshot in zip(ends, snapshots, strict=True):
        assert end["conflicts"] == len(snapshot["conflicts"])
        assert all(c["kind"] in CONFLICT_KINDS for c in snapshot["conflicts"])
        assert {tuple(c["coord"]) for c in snapshot["cells"]} == {
            tuple(c) for conflict in snapshot["conflicts"] for c in conflict["cells"]
        }
    assert ends[-1]["conflicts"] == 0 and snapshots[-1]["conflicts"] == [] == snapshots[-1]["cells"]
    # Iteration k > 0 reroutes exactly the nets of iteration k-1's conflicts,
    # ripping each up first; routed_cells is the replayed routing at that point.
    for k, (start, end) in enumerate(zip(starts, ends, strict=True)):
        assert end["rerouted"] == len(start["nets"]) and 0 <= end["changed"] <= end["rerouted"]
        routes_now = replay(trace["events"][: end["seq"]])["routes"]
        assert end["routed_cells"] == sum(r["length"] for r in routes_now.values())
        if k:
            assert set(start["nets"]) == {n for c in snapshots[k - 1]["conflicts"] for n in c["nets"]}
            ripped = [e["net"] for e in of_type(trace, "net_rip_up") if e["iteration"] == k]
            assert ripped == start["nets"]
    committed = of_type(trace, "net_route_committed")
    assert {e["net"] for e in committed} == set(nets)
    for event in committed:  # every committed tree of the (single) attempt is well formed
        assert_route_tree(payload(event, "iteration"), nets[event["net"]], blocks)
    (complete,) = of_type(trace, "routing_complete")
    assert complete["iterations"] == len(starts) == run.result.metrics["routing"]["iterations"]
    assert complete["routed_nets"] == len(nets) and complete["failure"] is None
    assert complete["rip_ups"] == len(of_type(trace, "net_rip_up")) == run.result.metrics["routing"]["rip_ups"]
    assert all(e["reason"] == "congestion" for e in of_type(trace, "net_rip_up"))


@pytest.mark.parametrize("source", [HALF_ADDER, FULL_ADDER], ids=["half-adder", "full-adder"])
def test_detailed_routing_events_build_the_committed_tree(source: str) -> None:
    trace = cached(source, "detailed").trace
    blocks = placed_cells(trace)
    pin_block = {(i, name): position for i, here in blocks.items() for name, position in here["pins"].items()}
    partial: dict[int, list[dict[str, Any]]] = {}
    rejected: dict[tuple, set[tuple[int, ...]]] = {}
    snapshots = {e["iteration"]: e for e in of_type(trace, "congestion_snapshot")}

    def branch_key(event: dict[str, Any]) -> tuple:
        return (event["net"], event["iteration"], event["sink"]["instance"], event["sink"]["pin"])

    for event in trace["events"]:
        kind = event["type"]
        if kind == "net_route_begin":
            assert event["driver"]["coord"] == pin_block[(event["driver"]["instance"], event["driver"]["pin"])]
            assert all(s["coord"] == pin_block[(s["instance"], s["pin"])] for s in event["sinks"])
            assert event["fanout"] == len(event["sinks"])
            partial[event["net"]] = []
        elif kind == "branch_route_begin":
            assert event["net"] in partial  # inside its net's route
            assert event["goal"] == pin_block[(event["sink"]["instance"], event["sink"]["pin"])]
        elif kind == "branch_path_rejected":
            assert event["coord"] in event["path"] and event["reason"]
            rejected.setdefault(branch_key(event), set()).add(tuple(event["coord"]))
        elif kind == "branch_route_found":
            assert event["length"] == len(event["path"]) - 1
            # A rejected block is avoided by every later search of that branch.
            assert not rejected.get(branch_key(event), set()) & {tuple(c) for c in event["path"]}
            partial[event["net"]].append({k: event[k] for k in ("sink", "start", "goal", "path")})
        elif kind == "net_route_committed":
            assert partial.pop(event["net"]) == event["branches"]
        elif kind == "physical_conflict":
            assert payload(event, "iteration") in snapshots[event["iteration"]]["conflicts"]
    assert partial == {}  # every started net was committed


def pad(grid: BlockGrid, site: PinSite) -> None:
    """A one-block cell under ``site``: the pin's endpoint dust rests on it."""
    x, y, z = site.cell
    grid.place(site.instance, [(x, y - 1, z)], [], [site])


def bottleneck() -> tuple[BlockGrid, list[RouteRequest], Bounds]:
    """Two nets heading east through a wall at x = 5 whose only near opening is a
    ONE-block tunnel (z = 0, y = 1); the other opening (z = 9) is a long detour.
    Both shortest routes need the tunnel, so the first pass must conflict."""
    grid = BlockGrid(max_y=4)
    wall = [(5, y, z) for y in range(5) for z in range(-12, 13) if z != 9 and not (z == 0 and y < 2)]
    grid.place(99, wall, [], [])
    requests = []
    for net, z in ((0, -1), (1, 1)):
        driver = PinSite((0, 1, z), 2 * net, "y", "out", Direction.EAST, 15, net)
        sink = PinSite((10, 1, z), 2 * net + 1, "a", "in", Direction.WEST, 1, net)
        pad(grid, driver)
        pad(grid, sink)
        requests.append(RouteRequest(net, "data", driver, (sink,)))
    return grid, requests, Bounds((-2, 1, -14), (12, 3, 14))


def test_negotiation_rips_up_congested_nets() -> None:
    grid, requests, bounds = bottleneck()
    trace = PrimitiveTraceRecorder("basic")
    # A mild first-pass congestion price: the second net still squeezes into the tunnel.
    config = PrimitivePnRConfig(present_factor_initial=0.5, present_factor_growth=1.6, history_increment=1.0)
    router = NegotiatedRedstoneRouter(grid, requests, bounds=bounds, config=config, trace=trace)
    outcome = router.run()
    assert outcome.success and outcome.iterations > 1
    assert grid.conflicts() == []
    events = trace.events
    rips = [e for e in events if e["type"] == "net_rip_up"]
    assert rips and all(e["reason"] == "congestion" and e["iteration"] >= 1 for e in rips)
    # Each rip-up removes exactly the route last committed for that net.
    last: dict[int, dict[str, Any]] = {}
    for event in events:
        if event["type"] == "net_route_committed":
            last[event["net"]] = event
        elif event["type"] == "net_rip_up":
            assert event["cells"] == last[event["net"]]["cells"]
            assert event["length"] == last[event["net"]]["length"]
    snapshots = [e for e in events if e["type"] == "congestion_snapshot"]
    first = snapshots[0]
    assert first["iteration"] == 0 and first["conflicts"]
    assert {n for c in first["conflicts"] for n in c["nets"]} == {0, 1}
    assert [5, 1, 0] in [c["coord"] for c in first["cells"]]  # the contested tunnel block
    assert all(c["history"] > 0 for c in first["cells"])
    assert snapshots[-1]["conflicts"] == [] and snapshots[-1]["cells"] == []
    # The tunnel holds one net at most: the other one was negotiated out of it.
    tunnel = [n for n, r in outcome.routes.items() if (5, 1, 0) in r.cell_set]
    assert len(tunnel) <= 1
    assert replay(events)["routes"] == {n: r.to_dict() for n, r in outcome.routes.items()}


@pytest.mark.parametrize(
    ("source", "config"),
    [(HALF_ADDER, {}), (NOT, {"max_pnr_attempts": 1, **NO_EXPANSIONS})],
    ids=["half-adder", "astar-budget-exhausted"],
)
def test_search_level_records_expansions_and_capped_blocked_moves(source: str, config: dict) -> None:
    cap = 3
    trace = cached(source, "search", max_blocked_events_per_branch=cap, **config).trace
    expands = of_type(trace, "route_search_expand")
    blocked = of_type(trace, "route_transition_blocked")
    assert expands and (blocked or source != HALF_ADDER)
    for event in expands:
        assert {"net", "iteration", "sink", "coord", "g", "h", "f", "frontier"} <= set(event)
        assert event["f"] == pytest.approx(event["g"] + event["h"])
        assert event["g"] >= 0 and event["h"] >= 0 and type(event["frontier"]) is int
    for event in blocked:
        a, b = event["from"], event["to"]
        assert abs(a[0] - b[0]) + abs(a[2] - b[2]) == 1 and abs(a[1] - b[1]) <= 1
        assert isinstance(event["reason"], str) and event["reason"]

    def key(event: dict[str, Any]) -> tuple:
        return (event["net"], event["iteration"], event["sink"]["instance"], event["sink"]["pin"])

    per_branch = Counter(key(e) for e in blocked)
    assert max(per_branch.values(), default=0) <= cap
    if source == HALF_ADDER:
        assert cap in per_branch.values()  # the cap really bites on this design
    # Every expansion is accounted for by that branch's search statistics.
    stats = Counter()
    for event in of_type(trace, "branch_search_stats"):
        stats[key(event)] += event["expansions"]
    assert Counter(key(e) for e in expands) == stats


def test_trace_level_filters_events_but_never_changes_the_design() -> None:
    traces = {level: pnr(FULL_ADDER, level).trace for level in LEVELS}
    table = documented_events()
    finals = [t["final"] for t in traces.values()]
    assert all(final == finals[0] for final in finals) and finals[0]["success"] is True
    types = {level: {e["type"] for e in t["events"]} for level, t in traces.items()}
    assert types["none"] == set()
    detailed_only = {"primitive_emitted", "primitive_mapped", "component_place_attempt", "net_route_begin",
                     "branch_route_begin", "branch_search_stats", "branch_route_found",
                     "route_legalization_begin", "signal_strength_scan"}  # fmt: skip
    search_only = {"route_search_expand", "route_transition_blocked"}
    assert not types["basic"] & (detailed_only | search_only)
    assert detailed_only <= types["detailed"] and not types["detailed"] & search_only
    assert detailed_only | search_only <= types["search"]
    # Each level records exactly the events of the level below, plus its own
    # (undocumented types are left to test_events_are_sequenced_documented...).
    for low, high in (("basic", "detailed"), ("detailed", "search")):
        rank = LEVELS.index(low)

        def documented(events: list[dict[str, Any]], at: int) -> list[dict[str, Any]]:
            return [payload(e) for e in events if e["type"] in table and LEVELS.index(table[e["type"]][1]) <= at]

        assert documented(traces[high]["events"], rank) == documented(traces[low]["events"], rank), (low, high)


def test_none_level_has_header_and_final_only(tmp_path: Path) -> None:
    run = cached(HALF_ADDER, "none")
    trace = run.trace
    assert trace["events"] == [] and trace["trace_level"] == "none"
    assert trace["config"]["trace_level"] == "none"
    assert trace["design"]["instances"] and trace["final"]["success"] is True
    assert trace["final"]["metrics"] == run.result.metrics
    path = run.result.trace.write(tmp_path / "half.primitive.pnr.jsonl")
    assert [json.loads(line)["record"] for line in path.read_text(encoding="utf-8").splitlines()] == [
        "header", "final"
    ]  # fmt: skip


# -- legalization ------------------------------------------------------------------------


def test_legalization_events_on_a_design_that_needs_repeaters() -> None:
    # 20-block routing channels: every net is longer than dust can carry a signal.
    run = cached(NOT, channel_width=20)
    trace, result = run.trace, run.result
    nets = trace["design"]["nets"]
    assert result.success
    (begin,) = of_type(trace, "legalization_begin")
    assert begin["round"] == 0 and begin["nets"] == len(nets)
    realized = {e["net"]: e for e in of_type(trace, "route_realized")}
    assert sorted(realized) == [n["id"] for n in nets]
    inserted = of_type(trace, "repeater_inserted")
    assert {e["net"] for e in inserted} == set(realized)  # every net needed a repeater
    for event in inserted:
        assert event["round"] == 0 and event["facing"] in ORIENTATIONS
        (element,) = [r for r in realized[event["net"]]["repeaters"] if r["coord"] == event["coord"]]
        assert (element["facing"], element["strength"], element["delay"]) == (
            event["facing"], event["input_strength"], event["delay"]
        )  # fmt: skip
    assert sum(len(r["repeaters"]) for r in realized.values()) == len(inserted)
    (complete,) = of_type(trace, "legalization_complete")
    assert (complete["round"], complete["realized"], complete["failures"], complete["repeaters"]) == (
        0, len(nets), 0, len(inserted)
    )  # fmt: skip
    assert result.metrics["routing"]["repeaters"] == result.metrics["legalization"]["repeaters"] == len(inserted)
    types = [e["type"] for e in trace["events"]]
    assert types.index("routing_complete") < types.index("legalization_begin")
    assert types.index("legalization_complete") < types.index("design_finalized") < types.index("pnr_attempt_end")


@pytest.mark.parametrize("source", [NOT, FULL_ADDER, COUNTER], ids=["not", "full-adder", "counter"])
def test_realized_routes_are_electrically_consistent_records(source: str) -> None:
    """Recomputes signal strength, repeaters, supports and delays from the trace
    JSON alone, exactly as the doc defines them."""
    run = cached(source, channel_width=20) if source == NOT else cached(source)  # NOT: every net repeated
    trace, final = run.trace, run.trace["final"]
    pins = pin_index(trace)
    nets = {n["id"]: n for n in trace["design"]["nets"]}
    routes = {r["net"]: r for r in final["routes"]}
    assert sorted(routes) == sorted(nets) == sorted(r["net"] for r in final["realized"])
    for realized in final["realized"]:
        route, net = routes[realized["net"]], nets[realized["net"]]
        elements = {tuple(e["coord"]): e for e in realized["elements"]}
        assert [e["coord"] for e in realized["elements"]] == route["cells"]
        parent = route_parents(route)
        for cell, element in elements.items():
            assert (None if element["parent"] is None else tuple(element["parent"])) == parent[cell]
        drive = pins[net["driver"]["instance"]][net["driver"]["pin"]]["strength"]
        assert realized["powered"] is (drive > 0)
        children = Counter(up for up in parent.values() if up is not None)
        for cell, element in elements.items():
            up = parent[cell]
            if not realized["powered"]:
                expected = 0
            elif up is None:
                expected = drive
            elif elements[up]["kind"] == "repeater":
                expected = 15
            elif element["kind"] == "repeater":
                expected = elements[up]["strength"]
            else:
                expected = max(0, elements[up]["strength"] - 1)
            assert element["strength"] == expected, (realized["net"], cell)
            ticks = 0 if up is None else elements[up]["delay"]
            if up is not None and elements[up]["kind"] == "repeater":
                ticks += REPEATER_DELAY_TICKS
            assert element["delay"] == ticks
            if realized["powered"]:
                assert element["strength"] >= 1
            if element["kind"] == "repeater":
                (child,) = [c for c, p in parent.items() if p == cell]
                assert up is not None and children[cell] == 1 and up[1] == cell[1] == child[1]
                step = [child[i] - cell[i] for i in range(3)]
                assert [cell[i] - up[i] for i in range(3)] == step == list(DIRECTIONS[element["facing"]])
        assert realized["repeaters"] == [e for e in realized["elements"] if e["kind"] == "repeater"]
        pin_blocks = {tuple(route["root"]), *(tuple(b["goal"]) for b in route["branches"])}
        below = [[c[0], c[1] - 1, c[2]] for c in elements if c not in pin_blocks]
        assert sorted(realized["supports"]) == sorted(below)
        # A staircase step keeps the block above its LOWER end as air.
        stairs = {
            (low[0], low[1] + 1, low[2])
            for c, up in parent.items()
            if up is not None and up[1] != c[1]
            for low in [min(c, up, key=lambda b: b[1])]
        }
        assert {tuple(c) for c in realized["clearances"]} == stairs
        dust = [e["strength"] for e in realized["elements"] if e["kind"] == "dust"]
        assert realized["min_strength"] == (min(dust) if realized["powered"] else 0)
        for report in realized["sinks"]:
            assert report["required"] == pins[report["instance"]][report["pin"]]["strength"]
            element = elements[tuple(report["coord"])]
            assert report["strength"] == element["strength"] and report["delay_ticks"] == element["delay"]
            if realized["powered"]:
                assert report["strength"] >= report["required"]
            depth, cursor, repeaters = 0, parent[tuple(report["coord"])], 0
            while cursor is not None:
                depth, repeaters = depth + 1, repeaters + (elements[cursor]["kind"] == "repeater")
                cursor = parent[cursor]
            assert (report["distance"], report["repeaters"]) == (depth, repeaters)
        assert realized["max_delay_ticks"] == max(s["delay_ticks"] for s in realized["sinks"])


@pytest.mark.parametrize(
    ("source", "config"),
    [(FULL_ADDER, {}), (COUNTER, {}), (NOT, {"channel_width": 20})],
    ids=["full-adder", "counter", "repeaters"],
)
def test_final_geometry_is_legal_from_the_trace_alone(source: str, config: dict) -> None:
    """The trace is self-contained: cell definitions + final placement + final
    realized routes are enough to re-check the redstone model's block rules."""
    trace = cached(source, **config).trace
    final = trace["final"]
    blocks = placed_cells(trace)
    nets = {n["id"]: n for n in trace["design"]["nets"]}
    body: dict[tuple[int, ...], int] = {}
    for inst, here in blocks.items():
        for voxel in map(tuple, here["voxels"]):
            assert voxel not in body, f"instances {body.get(voxel)} and {inst} overlap at {voxel}"
            body[voxel] = inst
    keepout = {tuple(c) for here in blocks.values() for c in here["keepout"]}
    assert not keepout & set(body)  # no cell sits in another cell's keep-out
    pin_owner = {
        tuple(position): (inst, name) for inst, here in blocks.items() for name, position in here["pins"].items()
    }
    signal: dict[tuple[int, ...], int] = {}
    support: dict[tuple[int, ...], int] = {}
    for realized in final["realized"]:
        net = nets[realized["net"]]
        own_pins = {(t["instance"], t["pin"]) for t in (net["driver"], *net["sinks"])}
        for element in realized["elements"]:
            cell = tuple(element["coord"])
            assert cell not in signal, f"nets {signal.get(cell)} and {net['id']} share {cell}"
            signal[cell] = net["id"]
            assert cell not in body and cell not in keepout, (net["id"], cell)
            if cell in pin_owner:
                assert pin_owner[cell] in own_pins, f"net {net['id']} uses pin {pin_owner[cell]}"
                assert (cell[0], cell[1] - 1, cell[2]) in body  # a pin's dust rests on its cell
        for cell in map(tuple, realized["supports"]):
            assert cell not in support and cell not in body and cell not in keepout and cell not in pin_owner
            support[cell] = net["id"]
    assert not set(signal) & set(support)
    for cell, net in signal.items():  # every non-pin signal block rests on its own support
        assert cell in pin_owner or support.get((cell[0], cell[1] - 1, cell[2])) == net
    # Rule 1: no signal block of ANOTHER net in a block's 12-block neighbourhood.
    neighbourhood = [(dx, dy, dz) for dy in (0, 1, -1) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    for (x, y, z), net in signal.items():
        for dx, dy, dz in neighbourhood:
            other = signal.get((x + dx, y + dy, z + dz))
            assert other in (None, net), f"nets {net} and {other} touch at {(x, y, z)}"


# -- failure ---------------------------------------------------------------------------


def test_failed_pnr_still_has_a_complete_trace() -> None:
    run = cached(NOT, max_pnr_attempts=2, **NO_EXPANSIONS)
    trace, result = run.trace, run.result
    final = trace["final"]
    assert final["success"] is False and result.success is False
    failure = final["failure"]
    assert set(failure) == {"stage", "reason", "message", "attempt", "attempts", "net", "iteration", "details"}
    assert (failure["stage"], failure["reason"], failure["attempt"], failure["attempts"]) == (
        "routing", "unroutable", 1, 2
    )  # fmt: skip
    assert failure["net"] in {n["id"] for n in trace["design"]["nets"]} and failure["iteration"] == 0
    assert failure["details"] and failure["details"][0]["search"] == "expansion_limit"
    # The last attempt's placement is complete; nothing was legalized.
    assert all(p["origin"] is not None for p in final["placement"])
    assert final["realized"] == [] and isinstance(final["conflicts"], list)
    assert final["search_bounds"] is not None and final["geometry"]["attempt"] == 1
    assert final["metrics"] == result.metrics and final["metrics"]["pnr"] == {"attempts": 2, "attempt": 1}
    (routing_failed,) = [e for e in attempts_of(trace)[-1] if e["type"] == "routing_failed"]
    assert routing_failed["failure"]["reason"] == "unroutable"
    assert of_type(trace, "pnr_end")[0]["success"] is False
    with pytest.raises(PrimitivePnRError, match="after 2 attempt"):
        result.raise_for_failure()
    with pytest.raises(PrimitivePnRError):
        result.to_design_dict()


def test_a_placement_failure_on_every_attempt_is_traced() -> None:
    run = pnr(DISPLAY, interface=DefaultInterfacePolicy(), max_y=3, retry_height_growth=0, max_pnr_attempts=2)
    trace = run.trace
    assert not run.result.success and run.result.failure.stage == "placement"
    assert [e["attempt"] for e in of_type(trace, "placement_failed")] == [0, 1]
    assert [e["status"] for e in of_type(trace, "pnr_attempt_end")] == ["failed", "failed"]
    final = trace["final"]
    assert final["failure"]["stage"] == "placement" and final["routes"] == [] and final["search_bounds"] is None
    display = [i["id"] for i in trace["design"]["instances"] if i["kind"] == "peripheral"]
    assert display and all(p["origin"] is None for p in final["placement"] if p["instance"] in display)
    state = replay(trace["events"])
    assert state["placement"] == {
        p["instance"]: (p["origin"], p["orientation"]) for p in final["placement"] if p["origin"] is not None
    }


def test_recorder_fail_closes_an_interrupted_trace(tmp_path: Path) -> None:
    recorder = PrimitiveTraceRecorder("detailed")
    recorder.begin_source(path="x.redc", top="main")
    recorder.emit("synthesis", "synthesis_begin", live_nodes=0, ops={}, sequential=False)
    recorder.fail("boom", stage="routing")
    expected = {"success": False, "failure": {"stage": "routing", "reason": "routing", "message": "boom"}}
    assert recorder.final == expected
    recorder.fail("a later problem")  # the first failure wins
    assert recorder.final == expected
    for suffix in (".json", ".jsonl"):
        loaded = load_trace(recorder.write(tmp_path / f"x{suffix}"))
        assert loaded["final"] == expected and [e["type"] for e in loaded["events"]] == ["synthesis_begin"]
        assert loaded["design"] is None and loaded["source"] == {"path": "x.redc", "top": "main"}
    finished = PrimitiveTraceRecorder()
    finished.finish({"success": True})
    finished.fail("too late")
    assert finished.final == {"success": True}
    default = PrimitiveTraceRecorder()
    default.fail("crash")
    assert default.final["failure"]["stage"] == "internal"
    with pytest.raises(CompileError, match="phase"):
        default.emit("wiring", "anything")
    assert PrimitiveTraceRecorder("none").wants(PrimitiveTraceRecorder("search").level) is False


def test_synthesis_failure_closes_the_trace_and_reraises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # place_and_route_graph has no registry argument: swap in a registry with no recipes.
    monkeypatch.setattr(lower, "default_registry", SynthesisRegistry)
    recorder = PrimitiveTraceRecorder("basic")
    with pytest.raises(CompileError, match="no primitive synthesis recipe"):
        place_and_route_graph(compile_source(HALF_ADDER), trace=recorder, source="half.redc", top="main")
    final = recorder.final
    assert final["success"] is False and final["failure"]["stage"] == "synthesis"
    assert "no primitive synthesis recipe" in final["failure"]["message"]
    types = [e["type"] for e in recorder.events]
    assert types[0] == "synthesis_begin" and set(types[1:]) <= {"ir_node_synthesized"}
    assert {e["synthesis_pass"] for e in recorder.events[1:]} == {"A"}  # inputs made it, the add did not
    loaded = load_trace(recorder.write(tmp_path / "half.primitive.pnr.json"))
    assert loaded["final"] == final and loaded["source"] == {"path": "half.redc", "top": "main"}


def test_techmap_failure_closes_the_trace_and_reraises() -> None:
    recorder = PrimitiveTraceRecorder("basic")
    with pytest.raises(CompileError, match="has no cell"):
        place_and_route_graph(
            compile_source(NOT), interface=PADS, library=PrimitiveTechnologyLibrary("empty", ()), trace=recorder
        )
    assert recorder.final["success"] is False and recorder.final["failure"]["stage"] == "techmap"
    types = [e["type"] for e in recorder.events]
    assert "synthesis_complete" in types and "techmap_complete" not in types and "pnr_begin" not in types


def test_a_net_legalization_keeps_rejecting_fails_with_a_full_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    force_legalization_failure(monkeypatch, net=0)
    run = pnr(NOT, max_legalization_rounds=2, max_pnr_attempts=1)
    trace, result = run.trace, run.result
    assert not result.success
    failure = trace["final"]["failure"]
    assert (failure["stage"], failure["reason"], failure["net"]) == ("legalization", "signal_strength", 0)
    assert_event_fields(trace["events"])
    assert [e["round"] for e in of_type(trace, "legalization_begin")] == [0, 1, 2]
    rejected = of_type(trace, "legalization_failed")
    assert [(e["net"], e["round"]) for e in rejected] == [(0, 0), (0, 1), (0, 2)]
    assert all(e["message"] and e["sink"] is None for e in rejected)
    assert [e["failures"] for e in of_type(trace, "legalization_complete")] == [1, 1, 1]
    # Between rounds the rejected net is ripped up for legalization and rerouted.
    rips = [e for e in of_type(trace, "net_rip_up") if e["reason"] == "legalization"]
    assert [e["net"] for e in rips] == [0, 0] and rips[0]["iteration"] < rips[1]["iteration"]
    repaired = {r["iteration"] for r in rips}
    repairs = [e for e in of_type(trace, "routing_iteration_begin") if e["iteration"] in repaired]
    assert [e["nets"] for e in repairs] == [[0], [0]]
    assert "design_finalized" not in {e["type"] for e in trace["events"]}
    final = trace["final"]
    assert sorted(r["net"] for r in final["realized"]) == [1]  # the rejected net has no realized route
    assert sorted(r["net"] for r in final["routes"]) == [0, 1]
    state = replay(trace["events"])
    assert state["routes"] == {r["net"]: r for r in final["routes"]}
    assert state["realized"] == {r["net"]: r for r in final["realized"]}


def test_a_verification_failure_is_traced(monkeypatch: pytest.MonkeyPatch) -> None:
    violation = Violation("short", "nets 0 and 1 touch (forced)", ((0, 1, 0), (1, 1, 0)), (0, 1))
    monkeypatch.setattr(pnr_design, "verify_design", lambda *args, **kwargs: [violation])
    run = pnr(NOT, max_pnr_attempts=2)
    trace, result = run.trace, run.result
    assert not result.success and result.failure.stage == "verification"
    illegal = of_type(trace, "illegal_transition")
    assert [payload(e) for e in illegal] == [violation.to_dict()] * 2  # once per attempt
    assert all(e["phase"] == "legalization" for e in illegal)
    assert_event_fields(trace["events"])
    ends = of_type(trace, "pnr_attempt_end")
    assert [(e["status"], e["failure"]["stage"], e["failure"]["reason"]) for e in ends] == [
        ("failed", "verification", "short")
    ] * 2
    final = trace["final"]
    assert final["violations"] == [violation.to_dict()] and final["failure"]["message"] == violation.message
    assert "design_finalized" not in {e["type"] for e in trace["events"]}
    # Legalization itself succeeded: every net is realized, exactly as replayed.
    state = replay(trace["events"])
    assert sorted(state["realized"]) == [n["id"] for n in trace["design"]["nets"]]
    assert state["realized"] == {r["net"]: r for r in final["realized"]}
    with pytest.raises(PrimitivePnRError, match="forced"):
        result.raise_for_failure()


def test_a_failed_legalization_repair_leaves_no_stale_realized_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round 0 rejects net 0; its repair renegotiates, rips up net 1 -- which WAS
    realized in round 0 -- for congestion that never clears, and routing fails.
    ``final`` must then describe that end state: net 1's round-0 realized route
    is gone (``net_rip_up`` deletes it on replay) and must not survive in
    ``final.realized`` next to net 1's NEW route tree."""
    forced = force_legalization_failure(monkeypatch, net=0, times=1)
    real_conflicts = BlockGrid.conflicts

    def conflicts(grid: BlockGrid) -> list[Conflict]:
        found = real_conflicts(grid)
        if forced["forced"]:  # after the rejection: congestion on net 1's blocks that never resolves
            claims = grid.claims.get(1)
            cells = tuple(sorted(claims.signals)) if claims else ((0, 1, 99),)
            found = [*found, Conflict("shared_signal", cells, (0, 1))]
        return found

    monkeypatch.setattr(BlockGrid, "conflicts", conflicts)
    run = pnr(NOT, max_routing_iterations=1, max_pnr_attempts=1)
    trace = run.trace
    final = trace["final"]
    # The scenario happened: net 1 was realized, then ripped up by the repair, then routing failed.
    assert (final["failure"]["stage"], final["failure"]["reason"]) == ("routing", "congestion")
    types = [(e["type"], e.get("net"), e.get("reason")) for e in trace["events"]]
    realized_at = types.index(("route_realized", 1, None))
    assert ("net_rip_up", 1, "congestion") in types[realized_at:]
    state = replay(trace["events"])
    assert state["routes"] == {r["net"]: r for r in final["routes"]}
    assert state["realized"] == {r["net"]: r for r in final["realized"]}, "final.realized keeps a ripped-up route"
    routes = {r["net"]: r for r in final["routes"]}
    for realized in final["realized"]:
        assert [e["coord"] for e in realized["elements"]] == routes[realized["net"]]["cells"]


# -- replay and final ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "level", "config"),
    [
        (HALF_ADDER, "basic", {}),
        (FULL_ADDER, "detailed", {}),
        (COUNTER, "basic", {}),
        (NOT, "basic", {"channel_width": 20}),
        (NOT, "basic", {"max_pnr_attempts": 2, **NO_EXPANSIONS}),
        (FULL_ADDER, "basic", {"keyframe_interval": 1}),
    ],
    ids=["half-adder", "detailed", "sequential", "repeaters", "failed", "keyframes"],
)
def test_replaying_the_events_reproduces_final(source: str, level: str, config: dict) -> None:
    run = cached(source, level, **config)
    final = run.trace["final"]
    state = replay(run.trace["events"])
    assert state["attempt"] == final["attempt"]
    assert state["placement"] == {
        p["instance"]: (p["origin"], p["orientation"]) for p in final["placement"] if p["origin"] is not None
    }
    assert state["routes"] == {r["net"]: r for r in final["routes"]}
    assert state["realized"] == {r["net"]: r for r in final["realized"]}
    if final["success"]:
        assert state["congestion"] == final["conflicts"] == []


def test_keyframes_snapshot_the_replayed_routes() -> None:
    trace = cached(FULL_ADDER, "basic", keyframe_interval=1).trace
    frames = of_type(trace, "keyframe")
    assert [f["iteration"] for f in frames] == [e["iteration"] for e in of_type(trace, "routing_iteration_end")]
    for frame in frames:
        before = replay(trace["events"][: frame["seq"]])
        assert {r["net"]: r for r in frame["routes"]} == before["routes"]


def test_final_matches_the_result() -> None:
    run = cached(COUNTER)
    final, result = run.trace["final"], run.result
    assert final["metrics"] == result.metrics
    assert final["geometry"] == result.geometry.to_dict()
    assert final["design_bounds"] == result.design_bounds.to_dict()
    assert {p["instance"]: p["origin"] for p in final["placement"]} == {
        i.id: list(i.origin) for i in run.mapped.instances
    }
    for entry, inst in zip(final["placement"], run.mapped.instances, strict=True):
        assert entry["orientation"] == inst.orientation.name
    assert final["routes"] == [result.routes[n].to_dict() for n in sorted(result.routes)]
    assert final["realized"] == [result.realized[n].to_dict() for n in sorted(result.realized)]
    assert final["conflicts"] == [] and final["violations"] == []
    routing = result.metrics["routing"]
    assert routing["routed_nets"] == len(final["routes"]) == len(run.mapped.nets)
    assert routing["dust_blocks"] == sum(e["kind"] == "dust" for r in final["realized"] for e in r["elements"])
    assert routing["repeaters"] == sum(len(r["repeaters"]) for r in final["realized"])
    assert routing["total_routed_length"] == sum(r["length"] for r in final["routes"])
    (finalized,) = of_type(run.trace, "design_finalized")
    assert (finalized["dust"], finalized["repeaters"], finalized["bounds"]) == (
        routing["dust_blocks"], routing["repeaters"], final["design_bounds"]
    )  # fmt: skip
    (end,) = of_type(run.trace, "pnr_attempt_end")
    assert end["metrics"] == result.metrics


# -- determinism -------------------------------------------------------------------------


def test_trace_is_deterministic() -> None:
    first = pnr(FULL_ADDER, "detailed").result.trace.to_json()
    assert pnr(FULL_ADDER, "detailed").result.trace.to_json() == first


def test_trace_is_identical_across_processes(tmp_path: Path) -> None:
    """Hash randomization (``PYTHONHASHSEED``) must not leak into the trace."""
    script = (
        "import sys\n"
        "from redc import compile_source\n"
        "from redc.physical_primitive import PadInterfacePolicy, PrimitivePnRConfig, "
        "PrimitiveTraceRecorder, place_and_route_graph\n"
        "recorder = PrimitiveTraceRecorder('detailed')\n"
        "place_and_route_graph(compile_source(sys.argv[1]), PrimitivePnRConfig(trace_level='detailed'), "
        "interface=PadInterfacePolicy(), trace=recorder, source='examples/test.redc', top='main')\n"
        "sys.stdout.write(recorder.to_json())\n"
    )
    outputs = []
    for seed in ("0", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"}
        done = subprocess.run(
            [sys.executable, "-c", script, FULL_ADDER], capture_output=True, text=True, env=env, cwd=tmp_path,
            check=False, timeout=120,
        )  # fmt: skip
        assert done.returncode == 0, done.stderr
        outputs.append(done.stdout)
    assert outputs[0] == outputs[1] == pnr(FULL_ADDER, "detailed").result.trace.to_json()
