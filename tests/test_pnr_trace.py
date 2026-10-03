"""The redc.pnr.trace.v1 replay trace, the physical-design file and the viewer."""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from redc import CompileError, compile_source
from redc.cli import app
from redc.physical import lower_to_physical
from redc.physical.pnr import (
    PHYSICAL_SCHEMA,
    TRACE_SCHEMA,
    PnRConfig,
    TraceLevel,
    TraceRecorder,
    load_trace,
    place_and_route,
)
from redc.viewer import check_trace, render_pnr_html

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
FIB = (EXAMPLES / "uint8_fib.redc").read_text()
ADD = (EXAMPLES / "uint8_add.redc").read_text()

COORD_KEYS = {"coord", "origin", "root", "start", "goal", "min", "max"}
PATH_KEYS = {"path", "cells"}


def run(source: str = FIB, top: str = "fib", **config):
    netlist = lower_to_physical(compile_source(source, top=top))
    return netlist, place_and_route(netlist, PnRConfig(**config))


def walk(value, key=None):
    """Yield ``(key, value)`` for every nested field."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield k, v
            yield from walk(v, k)
    elif isinstance(value, list):
        for v in value:
            yield from walk(v, key)


def is_coord(value) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(type(c) is int for c in value)


def assert_plain_json(value) -> None:
    if isinstance(value, dict):
        assert all(isinstance(k, str) for k in value)
        for v in value.values():
            assert_plain_json(v)
    elif isinstance(value, list):
        for v in value:
            assert_plain_json(v)
    else:
        assert value is None or type(value) in (bool, int, float, str), repr(value)


# -- schema ---------------------------------------------------------------------


def test_trace_header_and_sequence() -> None:
    _netlist, result = run()
    trace = result.trace.to_dict()
    assert trace["schema"] == TRACE_SCHEMA == "redc.pnr.trace.v1"
    assert trace["coordinate_system"]["units"] == "grid_cells"
    assert trace["coordinate_system"]["y"] == "up"
    assert [e["seq"] for e in trace["events"]] == list(range(len(trace["events"])))
    assert all({"seq", "phase", "type"} <= set(e) for e in trace["events"])
    assert {e["phase"] for e in trace["events"]} <= {"pnr", "placement", "routing", "congestion", "finalize"}
    # The header shows the UNPLACED design.
    assert all(i["origin"] is None for i in trace["design"]["instances"])
    assert trace["config"]["trace_level"] == "basic"
    assert trace["final"]["success"] is True


def test_trace_is_plain_json_and_round_trips() -> None:
    _netlist, result = run()
    trace = result.trace.to_dict()
    assert_plain_json(trace)
    assert json.loads(result.trace.to_json()) == trace


def test_every_coordinate_is_an_xyz_triple() -> None:
    _netlist, result = run(trace_level="detailed")
    for key, value in walk(result.trace.to_dict()):
        if key in COORD_KEYS and value is not None:
            assert is_coord(value), (key, value)
        if key in PATH_KEYS and isinstance(value, list):
            # `cells` also names lists of cell RECORDS (congestion, grid
            # snapshots); those carry their coordinate under "coord".
            for c in value:
                assert is_coord(c) or (isinstance(c, dict) and is_coord(c["coord"])), key


def test_every_referenced_id_exists() -> None:
    _netlist, result = run(trace_level="detailed")
    trace = result.trace.to_dict()
    instances = {i["id"] for i in trace["design"]["instances"]}
    nets = {n["id"] for n in trace["design"]["nets"]}
    components = {c["id"] for c in trace["design"]["components"]}
    assert {i["component"] for i in trace["design"]["instances"]} <= components
    for net in trace["design"]["nets"]:
        assert net["driver"]["instance"] in instances
        assert all(s["instance"] in instances for s in net["sinks"])
    for event in trace["events"]:
        if "instance" in event:
            assert event["instance"] in instances
        if "net" in event:
            assert event["net"] in nets
        if isinstance(event.get("sink"), dict):
            assert event["sink"]["instance"] in instances
        for net in event.get("nets", []) if isinstance(event.get("nets"), list) else []:
            assert net in nets
        for cell in event.get("cells", []) if event["type"] == "congestion_snapshot" else []:
            assert set(cell["nets"]) <= nets


def test_component_definitions_are_complete() -> None:
    netlist, result = run()
    trace = result.trace.to_dict()
    defs = {c["id"]: c for c in trace["design"]["components"]}
    for inst in netlist.instances.values():
        record = defs[next(i["component"] for i in trace["design"]["instances"] if i["id"] == inst.id)]
        assert record["dim"] == list(inst.component.dim)
        assert record["latency"] == inst.component.latency
        assert record["stateful"] == inst.component.is_stateful
        assert [p["name"] for p in record["ports"]] == [p.name for p in inst.component.ports]
        for port, p in zip(record["ports"], inst.component.ports):
            assert port["face"] == p.face.name.lower()
            assert port["offset"] == list(p.offset)
            assert port["direction"] in ("in", "out")
            assert port["type"]["name"] == p.dtype.name
            assert {"encoding", "lane_width", "lane_count"} <= set(port["layout"])
    lever = next(c for c in defs.values() if c["peripheral"] and c["peripheral"]["kind"] == "lever")
    assert lever["peripheral"]["direction"] == "input"
    assert lever["category"] == "peripheral" and lever["latency"] is None and lever["nbt"] is None
    for event in trace["events"]:
        if event["type"] == "component_placed":
            inst = next(i for i in trace["design"]["instances"] if i["id"] == event["instance"])
            assert event["dim"] == defs[inst["component"]]["dim"]


def test_final_state_matches_the_result() -> None:
    netlist, result = run()
    final = result.trace.to_dict()["final"]
    origins = {p["instance"]: p["origin"] for p in final["placement"]}
    assert origins == {i.id: list(i.origin) for i in netlist.instances.values()}
    routes = {r["net"]: r for r in final["routes"]}
    assert set(routes) == set(result.routes)
    for net_id, routed in result.routes.items():
        assert routes[net_id] == routed.to_dict()
        assert routes[net_id]["branches"][0]["path"][0] in routes[net_id]["cells"]
    assert final["congestion"] == []
    wires = [c for c in final["grid"]["cells"] if c["kind"] == "wire"]
    assert len(wires) == result.metrics["routing"]["wire_cells"]
    assert final["metrics"] == result.metrics
    # Replaying committed routes and rip-ups reproduces the final routes.
    replay: dict[int, list] = {}
    for event in result.trace.events:
        if event["type"] == "pnr_attempt_begin":
            replay = {}
        elif event["type"] == "net_route_committed":
            replay[event["net"]] = event["branches"]
        elif event["type"] == "net_rip_up":
            replay.pop(event["net"])
    assert replay == {n: r["branches"] for n, r in routes.items()}


def test_trace_is_deterministic() -> None:
    first = run(trace_level="detailed")[1].trace.to_json()
    second = run(trace_level="detailed")[1].trace.to_json()
    assert first == second


def test_trace_levels() -> None:
    def types(level: str) -> set[str]:
        return {e["type"] for e in run(ADD, "add", trace_level=level)[1].trace.events}

    assert types("none") == set()
    basic = types("basic")
    assert {"pnr_attempt_begin", "component_placed", "design_bounds_changed", "routing_begin",
            "routing_iteration_begin", "net_route_committed", "routing_iteration_end",
            "congestion_snapshot", "routes_finalized", "pnr_attempt_end"} <= basic  # fmt: skip
    assert "branch_route_found" not in basic and "route_search_expand" not in basic
    detailed = types("detailed")
    assert {"component_place_attempt", "net_route_begin", "branch_route_begin", "branch_route_found"} <= detailed
    search = types("search")
    assert "route_search_expand" in search
    _netlist, result = run(ADD, "add", trace_level="search")
    expand = next(e for e in result.trace.events if e["type"] == "route_search_expand")
    assert {"net", "sink", "coord", "g", "h", "f", "frontier"} <= set(expand)
    assert expand["f"] == pytest.approx(expand["g"] + expand["h"])
    none_trace = run(ADD, "add", trace_level="none")[1].trace.to_dict()
    assert none_trace["final"]["success"] is True and none_trace["events"] == []


def test_keyframes_are_optional_snapshots() -> None:
    _netlist, result = run(keyframe_interval=1)
    frames = [e for e in result.trace.events if e["type"] == "keyframe"]
    assert frames
    assert all(len(f["placement"]) == len(result.netlist.instances) for f in frames)
    assert {r["net"] for r in frames[-1]["routes"]} == set(result.routes)


def test_failure_trace_keeps_attempts_and_congestion() -> None:
    _netlist, result = run(ADD, "add", max_astar_expansions=1, max_pnr_attempts=2)
    trace = result.trace.to_dict()
    assert trace["final"]["success"] is False
    assert trace["final"]["attempts"] == 2
    assert [e["attempt"] for e in trace["events"] if e["type"] == "pnr_attempt_begin"] == [0, 1]
    assert any(e["type"] == "branch_route_failed" for e in trace["events"])


def test_jsonl_round_trip(tmp_path: Path) -> None:
    _netlist, result = run(ADD, "add")
    path = result.trace.write(tmp_path / "add.pnr.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["record"] == "header"
    assert json.loads(lines[-1])["record"] == "final"
    assert load_trace(path) == result.trace.to_dict()
    json_path = result.trace.write(tmp_path / "add.pnr.json")
    assert load_trace(json_path) == result.trace.to_dict()
    (tmp_path / "bogus.json").write_text('{"schema": "nope"}', encoding="utf-8")
    with pytest.raises(CompileError, match="not a redc.pnr.trace.v1"):
        load_trace(tmp_path / "bogus.json")


def test_recorder_rejects_unknown_phase() -> None:
    with pytest.raises(CompileError, match="phase"):
        TraceRecorder().emit("wiring", "anything")
    assert TraceRecorder(TraceLevel.NONE).wants(TraceLevel.BASIC) is False


# -- final physical design -----------------------------------------------------


def test_physical_design_file() -> None:
    netlist, result = run()
    design = result.to_physical_dict()
    assert design["schema"] == PHYSICAL_SCHEMA
    assert_plain_json(design)
    assert {i["id"] for i in design["instances"]} == set(netlist.instances)
    defs = {c["id"]: c for c in design["components"]}
    for inst in design["instances"]:
        assert is_coord(inst["origin"])
        dims = defs[inst["component"]]["dim"]
        assert inst["bounds"]["dims"] == dims
    for net in design["nets"]:
        assert net["route"]["net"] == net["id"]
        assert {"encoding", "lane_count", "lane_width"} <= set(net["layout"])
        assert all(is_coord(c) for b in net["route"]["branches"] for c in b["path"])
    assert design["metrics"]["routing"]["overused_cells"] == 0
    # Only final geometry: no transient search / rip-up data.
    text = json.dumps(design)
    assert "rip_up" not in text and "search" not in text


# -- viewer -----------------------------------------------------------------------


def embedded_trace(page: str) -> dict:
    match = re.search(r'<script type="application/json" id="redc-trace">(.*?)</script>', page, re.DOTALL)
    assert match
    return json.loads(match.group(1))


def test_viewer_page_embeds_the_trace_and_controls() -> None:
    _netlist, result = run()
    page = render_pnr_html(result.trace.to_dict(), title="fib <test>")
    assert embedded_trace(page) == result.trace.to_dict()
    for element in ('id="timeline"', 'id="btn-play"', 'id="btn-back"', 'id="btn-fwd"',
                    'id="speed"', 'id="attempt-select"', 'id="iteration-select"',
                    'id="btn-final"', 'id="selection"', 'id="file-input"', 'id="error"'):  # fmt: skip
        assert element in page, element
    for layer in ("components", "labels", "ports", "wires", "voxels", "search", "congestion", "grid", "bounds"):
        assert f'data-layer="{layer}"' in page
    for view in ("fit", "iso", "top", "front", "side"):
        assert f'data-view="{view}"' in page
    assert "three/addons/" in page and "OrbitControls" in page
    assert "validateTrace" in page and "__REDC_" not in page
    assert "<title>fib &lt;test&gt;</title>" in page


def test_viewer_escapes_script_breaking_text() -> None:
    _netlist, result = run(ADD, "add")
    trace = result.trace.to_dict()
    trace["design"]["instances"][0]["label"] = "</script><!-- x"
    page = render_pnr_html(trace)
    block = page.split('id="redc-trace">', 1)[1].split("</script>", 1)[0]
    assert "</script" not in block and "<!--" not in block
    assert embedded_trace(page)["design"]["instances"][0]["label"] == "</script><!-- x"


def test_viewer_rejects_malformed_traces() -> None:
    with pytest.raises(CompileError, match="schema"):
        render_pnr_html({"schema": "something.else"})
    with pytest.raises(CompileError, match="design"):
        check_trace({"schema": TRACE_SCHEMA, "events": []})
    with pytest.raises(CompileError, match="events"):
        check_trace({"schema": TRACE_SCHEMA, "design": {"components": [], "instances": [], "nets": []}})


# -- CLI ----------------------------------------------------------------------------


def test_cli_pnr_writes_trace_design_and_viewer(tmp_path: Path) -> None:
    runner = CliRunner()
    trace = tmp_path / "fib.pnr.json"
    design = tmp_path / "fib.physical.json"
    result = runner.invoke(app, ["pnr", str(EXAMPLES / "uint8_fib.redc"), "--top", "fib",
                                 "--trace", str(trace), "--output", str(design)])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert load_trace(trace)["final"]["success"] is True
    assert json.loads(design.read_text(encoding="utf-8"))["schema"] == PHYSICAL_SCHEMA
    page = (tmp_path / "fib.pnr.html").read_text(encoding="utf-8")
    assert embedded_trace(page)["schema"] == TRACE_SCHEMA
    rendered = tmp_path / "again.html"
    again = runner.invoke(app, ["render-pnr", str(trace), "--output", str(rendered)])
    assert again.exit_code == 0, again.output
    assert rendered.exists()


def test_cli_failure_still_writes_the_trace(tmp_path: Path) -> None:
    runner = CliRunner()
    trace = tmp_path / "fib.pnr.json"
    design = tmp_path / "fib.physical.json"
    result = runner.invoke(app, ["pnr", str(EXAMPLES / "uint8_fib.redc"), "--top", "fib",
                                 "--grid-height", "1", "--max-attempts", "2", "--no-viewer",
                                 "--trace", str(trace), "--output", str(design)])  # fmt: skip
    assert result.exit_code == 1
    assert "place-and-route failed after 2 attempt" in result.output
    data = load_trace(trace)
    assert data["final"]["success"] is False
    assert data["final"]["failure"]["stage"] == "placement"
    assert [e["status"] for e in data["events"] if e["type"] == "pnr_attempt_end"] == ["failed", "failed"]
    assert not design.exists()  # no final design for a failed run


def test_cli_render_rejects_non_traces(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.pnr.json"
    bogus.write_text('{"schema": "redc.physical.v1"}', encoding="utf-8")
    result = CliRunner().invoke(app, ["render-pnr", str(bogus)])
    assert result.exit_code == 1
    assert "redc.pnr.trace.v1" in result.output
