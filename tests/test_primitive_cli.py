"""The CLI with both physical backends: ``redc pnr --backend``,
``dump-netlist`` / ``dump-primitive-netlist``, ``physical-backends`` and
``backends``.

``--backend physical-primitive`` must write its own artifacts
(``build/<stem>.primitive.pnr.json`` trace, ``build/<stem>.primitive.physical.json``
design, ``build/<stem>.primitive.pnr.html`` viewer), reject options that belong
to the other backend, and still write the trace when place-and-route fails.
The coarse default (``--backend physical``) must behave exactly as before.

Every test runs in its own ``tmp_path`` working directory, so the default
``build/`` paths land there.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from redc import compile_source
from redc.backends import BACKENDS, DEFAULT_BACKEND, available_backends
from redc.cli import app
from redc.physical import lower_to_physical
from redc.physical.pnr import PnRConfig, TraceLevel, TraceRecorder, place_and_route
from redc.physical_primitive import (
    PHYSICAL_SCHEMA,
    TRACE_SCHEMA,
    synthesize_to_primitives,
)
from redc.physical_primitive.pnr import load_trace
from redc.physical_primitive.synthesis import PadInterfacePolicy

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
ADD = EXAMPLES / "uint8_add.redc"
ALU = EXAMPLES / "uint8_alu.redc"

NOT = "bool main(bool a) { return !a; }"
#: ``result: uint8`` is ONE 2-dig-7-seg display (five blocks tall) under the
#: default interface policy: it cannot be placed with ``--max-height 3``.
DISPLAY = "uint8 main(uint8 a) { return a; }"

Invoke = Callable[..., Result]


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Invoke:
    """``cli("pnr", path, ...)`` runs ``redc`` with ``tmp_path`` as the working directory."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    def invoke(*args: object) -> Result:
        return runner.invoke(app, [str(a) for a in args])

    return invoke


def program(directory: Path, name: str, text: str) -> Path:
    path = directory / "src" / f"{name}.redc"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def written(root: Path) -> set[str]:
    """Every file under ``root`` except the test's own sources, as posix paths."""
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.parts[-2:-1] != ("src",)}


def absolute(local: list[int], origin: list[int], orientation: str) -> list[int]:
    """``origin + rotate(local)``; ``south`` turns local +x to +z: ``(x, y, z) -> (-z, y, x)``."""
    x, y, z = local
    for _ in range(("east", "south", "west", "north").index(orientation)):
        x, z = -z, x
    return [origin[0] + x, origin[1] + y, origin[2] + z]


def embedded_trace(page: str) -> dict:
    match = re.search(r'<script type="application/json" id="redc-trace">(.*?)</script>', page, re.DOTALL)
    assert match, "no embedded trace"
    return json.loads(match.group(1))


# -- redc pnr --backend physical-primitive -------------------------------------------


def test_primitive_pnr_writes_its_own_trace_and_design(cli: Invoke, tmp_path: Path) -> None:
    result = cli("pnr", ADD, "--top", "add", "--backend", "physical-primitive", "--no-viewer")
    assert result.exit_code == 0, result.output
    assert written(tmp_path) == {"build/uint8_add.primitive.pnr.json", "build/uint8_add.primitive.physical.json"}
    trace = load_trace(tmp_path / "build" / "uint8_add.primitive.pnr.json")
    assert trace["schema"] == TRACE_SCHEMA and trace["backend"] == "physical-primitive"
    assert trace["source"] == {"path": ADD.resolve().as_posix(), "top": "add"}
    assert trace["trace_level"] == "basic" and trace["final"]["success"] is True
    design = json.loads((tmp_path / "build" / "uint8_add.primitive.physical.json").read_text(encoding="utf-8"))
    assert design["schema"] == PHYSICAL_SCHEMA == "redc.physical-primitive.v1"
    assert design["backend"] == "physical-primitive" and design["coordinate_system"]["units"] == "blocks"
    assert design["source"] == trace["source"] and design["placeholder_geometry"] is True
    assert design["metrics"] == trace["final"]["metrics"]
    assert design["bounds"] == trace["final"]["design_bounds"] == design["metrics"]["final"]["bounds"]
    # Default interface: in_a / in_b are pads, the uint8 result one display.
    ports = {p["name"]: p["realization"] for p in design["logical"]["ports"]}
    assert ports == {"in_a": "pads", "in_b": "pads", "result": "2-dig-7-seg"}
    cells = {c["name"]: c for c in design["cells"]}
    placement = {p["instance"]: p for p in trace["final"]["placement"]}
    assert [i["id"] for i in design["instances"]] == list(range(len(placement)))
    net_of = {}
    for net in design["nets"]:
        for terminal in (net["driver"], *net["sinks"]):
            net_of[(terminal["instance"], terminal["pin"])] = net["id"]
    for inst in design["instances"]:
        cell, where = cells[inst["cell"]], placement[inst["id"]]
        assert (inst["origin"], inst["orientation"]) == (where["origin"], where["orientation"])
        # Absolute blocks = origin + rotate(local), for voxels, keep-out and pins alike.
        place = functools.partial(absolute, origin=inst["origin"], orientation=inst["orientation"])
        assert sorted(v["coord"] for v in inst["voxels"]) == sorted(place(v["coord"]) for v in cell["voxels"])
        assert sorted(inst["keepout"]) == sorted(place(c) for c in cell["keepout"])
        assert [(p["name"], p["direction"], p["position"]) for p in inst["pins"]] == [
            (p["name"], p["direction"], place(p["position"])) for p in cell["pins"]
        ]
        assert all(p["net"] == net_of[(inst["id"], p["name"])] for p in inst["pins"])  # every pin is connected
    routes = {r["net"]: r for r in trace["final"]["routes"]}
    realized = {r["net"]: r for r in trace["final"]["realized"]}
    assert [n["id"] for n in design["nets"]] == sorted(routes) == sorted(realized)
    for net in design["nets"]:
        assert net["width"] == 1
        assert net["route"] == routes[net["id"]] and net["realized"] == realized[net["id"]]
    metrics = design["metrics"]
    assert metrics["routing"]["routed_nets"] == len(design["nets"])
    assert f"{metrics['routing']['routed_nets']} one-bit nets" in result.output
    assert f"{metrics['routing']['repeaters']} repeaters" in result.output
    # Final geometry only: no replay events (rip-ups, searches) in the design file.
    text = json.dumps(design)
    assert '"net_rip_up"' not in text and '"route_search_expand"' not in text and '"events"' not in text


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_trace_and_output_paths_are_honored(cli: Invoke, tmp_path: Path, suffix: str) -> None:
    source = program(tmp_path, "inv", NOT)
    trace_path = Path("out") / f"custom{suffix}"
    result = cli("pnr", source, "--backend", "physical-primitive", "--trace", trace_path,
                 "--output", "out/design.json", "--no-viewer")  # fmt: skip
    assert result.exit_code == 0, result.output
    assert written(tmp_path) == {trace_path.as_posix(), "out/design.json"}
    if suffix == ".jsonl":
        lines = (tmp_path / trace_path).read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["record"] == "header" and json.loads(lines[-1])["record"] == "final"
    trace = load_trace(tmp_path / trace_path)
    assert trace["final"]["success"] is True and trace["source"]["top"] == "main"
    design = json.loads((tmp_path / "out" / "design.json").read_text(encoding="utf-8"))
    assert design["schema"] == PHYSICAL_SCHEMA and design["metrics"] == trace["final"]["metrics"]


@pytest.mark.parametrize("level", ["none", "detailed", "search"])
def test_trace_level_reaches_the_primitive_trace(cli: Invoke, tmp_path: Path, level: str) -> None:
    source = program(tmp_path, "inv", NOT)
    result = cli("pnr", source, "--backend", "physical-primitive", "--trace-level", level, "--no-viewer")
    assert result.exit_code == 0, result.output
    trace = load_trace(tmp_path / "build" / "inv.primitive.pnr.json")
    assert trace["trace_level"] == trace["config"]["trace_level"] == level
    types = {e["type"] for e in trace["events"]}
    if level == "none":
        assert types == set() and trace["final"]["success"] is True
    else:
        assert "component_place_attempt" in types
        assert ("route_search_expand" in types) is (level == "search")
    assert f"replay trace, {len(trace['events'])} events" in result.output


@pytest.mark.parametrize("stem", ["inv", "inv.v2"])
def test_primitive_viewer_is_written_by_default(cli: Invoke, tmp_path: Path, stem: str) -> None:
    source = program(tmp_path, stem, NOT)
    result = cli("pnr", source, "--backend", "physical-primitive")
    assert result.exit_code == 0, result.output
    assert written(tmp_path) == {
        f"build/{stem}.primitive.pnr.json",
        f"build/{stem}.primitive.physical.json",
        f"build/{stem}.primitive.pnr.html",
    }
    page = (tmp_path / "build" / f"{stem}.primitive.pnr.html").read_text(encoding="utf-8")
    assert embedded_trace(page) == load_trace(tmp_path / "build" / f"{stem}.primitive.pnr.json")
    assert f"{stem}.redc" in re.search(r"<title>(.*?)</title>", page).group(1)


def test_custom_trace_path_puts_the_viewer_next_to_it(cli: Invoke, tmp_path: Path) -> None:
    source = program(tmp_path, "inv", NOT)
    result = cli("pnr", source, "--backend", "physical-primitive", "--trace", "elsewhere/run.pnr.jsonl")
    assert result.exit_code == 0, result.output
    assert written(tmp_path) == {
        "elsewhere/run.pnr.jsonl", "elsewhere/run.primitive.pnr.html", "build/inv.primitive.physical.json"
    }  # fmt: skip


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_render_pnr_picks_the_primitive_viewer(cli: Invoke, tmp_path: Path, suffix: str) -> None:
    source = program(tmp_path, "inv", NOT)
    trace = f"build/inv.primitive.pnr{suffix}"
    assert cli("pnr", source, "--backend", "physical-primitive", "--trace", trace, "--no-viewer").exit_code == 0
    result = cli("render-pnr", trace)
    assert result.exit_code == 0, result.output
    page = (tmp_path / "build" / "inv.primitive.pnr.html").read_text(encoding="utf-8")
    assert embedded_trace(page) == load_trace(tmp_path / trace)


def test_failed_primitive_pnr_still_writes_the_trace(cli: Invoke, tmp_path: Path) -> None:
    source = program(tmp_path, "show", DISPLAY)
    result = cli("pnr", source, "--backend", "physical-primitive", "--max-height", "3", "--max-attempts", "1",
                 "--no-viewer")  # fmt: skip
    assert result.exit_code == 1
    assert "place-and-route failed after 1 attempt(s)" in result.output
    assert written(tmp_path) == {"build/show.primitive.pnr.json"}  # no design for a failed run
    trace = load_trace(tmp_path / "build" / "show.primitive.pnr.json")
    final = trace["final"]
    assert final["success"] is False and final["attempts"] == 1
    assert final["failure"]["stage"] == "placement" and final["failure"]["message"] in result.output
    assert [e["status"] for e in trace["events"] if e["type"] == "pnr_attempt_end"] == ["failed"]
    assert trace["config"]["max_y"] == 3 and trace["config"]["max_pnr_attempts"] == 1


def test_a_failed_primitive_pnr_with_the_viewer_still_writes_both(cli: Invoke, tmp_path: Path) -> None:
    source = program(tmp_path, "show", DISPLAY)
    result = cli("pnr", source, "--backend", "physical-primitive", "--max-height", "3", "--max-attempts", "1")
    assert result.exit_code == 1, result.output
    assert written(tmp_path) == {"build/show.primitive.pnr.json", "build/show.primitive.pnr.html"}
    page = (tmp_path / "build" / "show.primitive.pnr.html").read_text(encoding="utf-8")
    assert embedded_trace(page)["final"]["success"] is False


def test_synthesis_failure_still_writes_the_trace(
    cli: Invoke, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from redc.physical_primitive.synthesis import SynthesisRegistry, lower

    monkeypatch.setattr(lower, "default_registry", SynthesisRegistry)  # no recipe resolves
    source = program(tmp_path, "inv", NOT)
    result = cli("pnr", source, "--backend", "physical-primitive", "--no-viewer")
    assert result.exit_code == 1
    assert "no primitive synthesis recipe" in result.output
    assert written(tmp_path) == {"build/inv.primitive.pnr.json"}
    final = load_trace(tmp_path / "build" / "inv.primitive.pnr.json")["final"]
    assert final["success"] is False and final["failure"]["stage"] == "synthesis"


def test_synthesis_failure_with_the_viewer_reports_the_real_error(
    cli: Invoke, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A synthesis-stage failure trace has no design yet (``design: null``); the
    CLI must still report the synthesis error and must not claim the trace --
    which it DID write -- could not be written."""
    from redc.physical_primitive.synthesis import SynthesisRegistry, lower

    monkeypatch.setattr(lower, "default_registry", SynthesisRegistry)
    source = program(tmp_path, "inv", NOT)
    result = cli("pnr", source, "--backend", "physical-primitive")
    assert result.exit_code == 1
    assert "no primitive synthesis recipe" in result.output
    assert "build/inv.primitive.pnr.json" in written(tmp_path)
    assert load_trace(tmp_path / "build" / "inv.primitive.pnr.json")["final"]["failure"]["stage"] == "synthesis"
    assert "could not write the trace" not in result.output, result.output


def test_interface_option_selects_the_port_realization(cli: Invoke, tmp_path: Path) -> None:
    source = program(tmp_path, "show", DISPLAY)
    kinds = {}
    for interface in ("default", "pads"):
        result = cli("pnr", source, "--backend", "physical-primitive", "--interface", interface,
                     "-o", f"{interface}.json", "--trace", f"{interface}.pnr.json", "--no-viewer")  # fmt: skip
        assert result.exit_code == 0, result.output
        design = json.loads((tmp_path / f"{interface}.json").read_text(encoding="utf-8"))
        kinds[interface] = sorted({i["kind"] for i in design["instances"]})
    assert kinds == {"default": ["input_bit", "peripheral"], "pads": ["input_bit", "output_bit"]}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--interface", "bogus"], "unknown interface policy 'bogus'"),
        (["--trace-level", "loud"], "unknown trace level 'loud'"),
        (["--max-height", "400"], "max_y must be between 3 and"),
    ],
    ids=["interface", "trace-level", "max-height"],
)
def test_invalid_primitive_settings_fail_before_any_trace(
    cli: Invoke, tmp_path: Path, args: list[str], message: str
) -> None:
    source = program(tmp_path, "inv", NOT)
    result = cli("pnr", source, "--backend", "physical-primitive", *args)
    assert result.exit_code == 1
    assert message in result.output
    assert written(tmp_path) == set()


@pytest.mark.parametrize(
    ("backend", "args", "expected"),
    [
        (
            "physical-primitive",
            ["--spacing", "2", "--routing-margin", "4", "--max-route-iterations", "5", "--max-attempts", "2",
             "--channel-width", "7", "--max-height", "11", "--trace-level", "detailed", "--clock-period", "40",
             "--clock-margin", "2", "--max-clock-skew", "1"],
            {"component_spacing": 2, "routing_margin": 4, "max_routing_iterations": 5, "max_pnr_attempts": 2,
             "channel_width": 7, "max_y": 11, "trace_level": "detailed", "clock_period_rt": 40, "clock_margin_rt": 2,
             "max_clock_skew_rt": 1},
        ),
        (
            "physical",
            ["--spacing", "2", "--routing-margin", "3", "--max-route-iterations", "9", "--max-attempts", "2",
             "--grid-height", "20", "--layer-gap", "3"],
            {"component_spacing": 2, "routing_margin": 3, "max_routing_iterations": 9, "max_pnr_attempts": 2,
             "grid_height": 20, "layer_gap": 3, "trace_level": "basic"},
        ),
    ],
    ids=["primitive", "coarse"],
)  # fmt: skip
def test_backend_options_reach_the_backend_config(
    cli: Invoke, tmp_path: Path, backend: str, args: list[str], expected: dict
) -> None:
    source = program(tmp_path, "inv", NOT)
    result = cli("pnr", source, "--backend", backend, *args, "--trace", "run.json", "--no-viewer")
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))["config"]
    assert {key: config[key] for key in expected} == expected


# -- backend selection and option ownership -------------------------------------------


@pytest.mark.parametrize("command", ["pnr", "dump-netlist"])
def test_unknown_backend_lists_the_choices(cli: Invoke, tmp_path: Path, command: str) -> None:
    source = program(tmp_path, "inv", NOT)
    result = cli(command, source, "--backend", "bananas")
    assert result.exit_code == 1
    assert "unknown physical backend 'bananas'" in result.output
    assert "physical, physical-primitive" in result.output
    assert written(tmp_path) == set()


@pytest.mark.parametrize(
    ("backend", "option", "owner"),
    [
        ("physical-primitive", ["--grid-height", "10"], "physical"),
        ("physical-primitive", ["--layer-gap", "2"], "physical"),
        ("physical", ["--max-height", "5"], "physical-primitive"),
        ("physical", ["--channel-width", "3"], "physical-primitive"),
        ("physical", ["--interface", "pads"], "physical-primitive"),
        ("physical", ["--clock-period", "10"], "physical-primitive"),
        ("physical", ["--max-clock-skew", "1"], "physical-primitive"),
        (None, ["--max-height", "5"], "physical-primitive"),
    ],
    ids=["grid-height", "layer-gap", "max-height", "channel-width", "interface", "clock-period", "max-clock-skew",
         "default-backend"],
)
def test_options_of_the_other_backend_are_rejected(
    cli: Invoke, tmp_path: Path, backend: str | None, option: list[str], owner: str
) -> None:
    source = program(tmp_path, "inv", NOT)
    selected = [] if backend is None else ["--backend", backend]
    result = cli("pnr", source, *selected, *option)
    assert result.exit_code == 1
    flag = option[0]
    assert f"the {backend or 'physical'} backend does not support {flag}" in result.output
    assert f"{flag} belongs to {owner}" in result.output
    assert written(tmp_path) == set()


# -- dump-netlist ------------------------------------------------------------------------


def test_dump_netlist_prints_the_primitive_netlist(cli: Invoke, tmp_path: Path) -> None:
    source = program(tmp_path, "show", DISPLAY)
    result = cli("dump-netlist", source, "--backend", "physical-primitive")
    assert result.exit_code == 0, result.output
    netlist = json.loads(result.stdout)
    assert netlist["schema"] == "redc.primitive-netlist.v1" and netlist["stage"] == "primitive"
    expected = synthesize_to_primitives(compile_source(DISPLAY)).to_dict()  # default interface
    assert netlist == expected
    pads = json.loads(cli("dump-netlist", source, "--backend", "physical-primitive", "--interface", "pads").stdout)
    assert {p["realization"] for p in pads["ports"]} == {"pads"}
    assert pads == synthesize_to_primitives(compile_source(DISPLAY), interface=PadInterfacePolicy()).to_dict()
    assert written(tmp_path) == set()


def test_dump_netlist_handles_types_the_coarse_backend_rejects(cli: Invoke) -> None:
    coarse = cli("dump-netlist", ALU)
    assert coarse.exit_code == 1 and "uint3" in coarse.output
    result = cli("dump-netlist", ALU, "--backend", "physical-primitive")
    assert result.exit_code == 0, result.output
    netlist = json.loads(result.stdout)
    assert netlist["schema"] == "redc.primitive-netlist.v1"
    op = next(p for p in netlist["ports"] if p["name"] == "in_op")
    assert op["type"]["name"] == "uint3" and len(op["bits"]) == 3
    assert netlist["summary"]["gates"] > 0
    mapped = cli("dump-netlist", ALU, "--backend", "physical-primitive", "--stage", "mapped")
    assert mapped.exit_code == 0, mapped.output
    assert json.loads(mapped.stdout)["logical"] == netlist
    short = cli("dump-primitive-netlist", ALU)
    assert short.exit_code == 0 and json.loads(short.stdout) == netlist


def test_dump_netlist_mapped_stage(cli: Invoke, tmp_path: Path) -> None:
    source = program(tmp_path, "inv", NOT)
    result = cli("dump-netlist", source, "--backend", "physical-primitive", "--stage", "mapped")
    assert result.exit_code == 0, result.output
    mapped = json.loads(result.stdout)
    assert mapped["schema"] == "redc.primitive-mapped-netlist.v1" and mapped["stage"] == "mapped"
    assert all(i["origin"] is None and i["orientation"] is None for i in mapped["instances"])
    assert mapped["logical"]["schema"] == "redc.primitive-netlist.v1"
    assert {i["cell"] for i in mapped["instances"]} <= {c["name"] for c in mapped["cells"]}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--backend", "physical-primitive", "--stage", "bogus"], "no netlist stage 'bogus'"),
        (["--backend", "physical", "--stage", "mapped"], "no netlist stage 'mapped'"),
        (["--backend", "physical", "--interface", "pads"], "--interface"),
        (["--backend", "physical-primitive", "--interface", "bogus"], "unknown interface policy 'bogus'"),
    ],
    ids=["primitive-stage", "coarse-stage", "coarse-interface", "primitive-interface"],
)
def test_dump_netlist_rejects_bad_requests(cli: Invoke, tmp_path: Path, args: list[str], message: str) -> None:
    source = program(tmp_path, "inv", NOT)
    result = cli("dump-netlist", source, *args)
    assert result.exit_code == 1
    assert message in result.output


@pytest.mark.parametrize("backend", [None, "physical"])
def test_dump_netlist_still_prints_the_coarse_netlist(cli: Invoke, backend: str | None) -> None:
    selected = [] if backend is None else ["--backend", backend]
    result = cli("dump-netlist", ADD, "--top", "add", *selected)
    assert result.exit_code == 0, result.output
    graph = compile_source(ADD.read_text(encoding="utf-8"), top="add")
    assert result.stdout == json.dumps(lower_to_physical(graph).to_dict(), indent=2) + "\n"


def test_dump_primitive_netlist_is_a_shorthand(cli: Invoke, tmp_path: Path) -> None:
    source = program(tmp_path, "inv", NOT)
    long = cli("dump-netlist", source, "--backend", "physical-primitive")
    short = cli("dump-primitive-netlist", source)
    assert short.exit_code == 0, short.output
    assert short.stdout == long.stdout
    mapped = cli("dump-primitive-netlist", source, "--stage", "mapped")
    assert mapped.exit_code == 0 and json.loads(mapped.stdout)["schema"] == "redc.primitive-mapped-netlist.v1"
    saved = cli("dump-primitive-netlist", source, "-o", "out/inv.primitive.json")
    assert saved.exit_code == 0, saved.output
    assert (tmp_path / "out" / "inv.primitive.json").read_text(encoding="utf-8") == long.stdout
    assert cli("dump-primitive-netlist", source, "--stage", "nope").exit_code == 1


# -- registries ----------------------------------------------------------------------------


def test_physical_backends_lists_both_with_their_schemas(cli: Invoke) -> None:
    result = cli("physical-backends")
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert [line.split("\t")[:3] for line in lines] == [
        ["physical (default)", "redc.pnr.trace.v1", "redc.physical.v1"],
        ["physical-primitive", "redc.physical-primitive.pnr.v1", "redc.physical-primitive.v1"],
    ]
    assert all(len(line.split("\t")) == 4 and line.split("\t")[3] for line in lines)


def test_output_backends_list_is_unchanged(cli: Invoke) -> None:
    result = cli("backends")
    assert result.exit_code == 0, result.output
    expected = "".join(
        f"{name}{' (default)' if name == DEFAULT_BACKEND else ''}\t{BACKENDS[name].extension}\t"
        f"{BACKENDS[name].summary}\n"
        for name in available_backends()
    )
    assert result.stdout == expected
    assert sorted(line.split("\t")[0] for line in result.stdout.splitlines()) == [
        "ir-json", "systemverilog (default)"
    ]  # fmt: skip
    assert "physical" not in result.stdout


# -- the coarse default is unchanged ---------------------------------------------------


def coarse_reference(source: Path, top: str) -> tuple[str, str, str]:
    """``(trace, design, summary line)`` exactly as the pre-registry ``redc pnr`` made them."""
    graph = compile_source(source.read_text(encoding="utf-8"), top=top)
    config = PnRConfig(
        grid_height=24, base_y=min(PnRConfig.base_y, 23), component_spacing=3, layer_gap=4,
        routing_margin=4, max_routing_iterations=40, max_pnr_attempts=4, trace_level=TraceLevel.parse("basic"),
    )  # fmt: skip
    recorder = TraceRecorder(config.trace_level)
    result = place_and_route(lower_to_physical(graph), config, trace=recorder)
    assert result.success
    metrics = result.metrics
    bounds = metrics["final"]["bounds"]
    summary = (
        f"P&R: {metrics['placement']['component_count']} components, "
        f"{metrics['routing']['routed_nets']} nets, {metrics['routing']['wire_cells']} wire cells, "
        f"{metrics['routing']['iterations']} routing iteration(s), attempt {result.geometry.attempt}, "
        f"bounds {bounds['dims'] if bounds else '-'}"
    )
    return recorder.to_json(), json.dumps(result.to_physical_dict(), indent=1) + "\n", summary


def test_default_pnr_is_the_unchanged_coarse_flow(cli: Invoke, tmp_path: Path) -> None:
    trace, design, summary = coarse_reference(ADD, "add")
    result = cli("pnr", ADD, "--top", "add", "--no-viewer")
    assert result.exit_code == 0, result.output
    assert written(tmp_path) == {"build/uint8_add.pnr.json", "build/uint8_add.physical.json"}
    assert (tmp_path / "build" / "uint8_add.pnr.json").read_text(encoding="utf-8") == trace
    assert (tmp_path / "build" / "uint8_add.physical.json").read_text(encoding="utf-8") == design
    assert result.stdout.splitlines()[-1] == summary
    explicit = cli("pnr", ADD, "--top", "add", "--backend", "physical", "--no-viewer")
    assert explicit.exit_code == 0, explicit.output
    assert (tmp_path / "build" / "uint8_add.physical.json").read_text(encoding="utf-8") == design


def test_default_pnr_still_writes_the_coarse_viewer(cli: Invoke, tmp_path: Path) -> None:
    result = cli("pnr", ADD, "--top", "add")
    assert result.exit_code == 0, result.output
    assert written(tmp_path) == {
        "build/uint8_add.pnr.json", "build/uint8_add.physical.json", "build/uint8_add.pnr.html"
    }  # fmt: skip
    page = (tmp_path / "build" / "uint8_add.pnr.html").read_text(encoding="utf-8")
    assert embedded_trace(page)["schema"] == "redc.pnr.trace.v1"
    assert "<title>RedC P&amp;R - uint8_add.redc</title>" in page


# -- timing closure ---------------------------------------------------------------------

COUNTER = "uint2 main(uint2 n) { uint2 x = 0; for (uint2 i = 0; i < n; i++) { x = x + 1; } return x; }"


def test_sequential_pnr_reports_timing_closure_and_writes_the_reports(cli: Invoke, tmp_path: Path) -> None:
    source = program(tmp_path, "count", COUNTER)
    result = cli("pnr", source, "--backend", "physical-primitive", "--interface", "pads", "--no-viewer")
    assert result.exit_code == 0, result.output
    assert "Timing: clock" in result.output and "(auto), skew 0 rt" in result.output
    assert "redstone simulation (abstract-components) validated" in result.output
    design = json.loads((tmp_path / "build" / "count.primitive.physical.json").read_text(encoding="utf-8"))
    assert design["timing"]["closure"]["passed"] and design["timing"]["clock"]["skew"]["gt"] == 0
    assert design["simulation"]["validated"] and design["world"]["schema"] == "redc.minecraft-design.v1"


def test_a_too_short_clock_period_fails_with_a_trace_and_no_design(cli: Invoke, tmp_path: Path) -> None:
    source = program(tmp_path, "count", COUNTER)
    result = cli("pnr", source, "--backend", "physical-primitive", "--interface", "pads", "--clock-period", "4")
    assert result.exit_code == 1
    assert "the requested clock period of 4 rt is shorter than" in result.output
    assert "after 1 attempt(s)" in result.output  # deterministic: never retried
    trace = load_trace(tmp_path / "build" / "count.primitive.pnr.json")
    assert trace["final"]["failure"]["code"] == "timing/setup"
    assert trace["final"]["timing"]["clock"]["period"]["rt"] == 4  # the user's period, not raised
    assert not (tmp_path / "build" / "count.primitive.physical.json").exists()
