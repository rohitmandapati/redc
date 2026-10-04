"""The ``redc.physical-primitive.pnr.v1`` replay viewer page and the viewer API
that serves both physical backends (``redc.viewer``)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

from redc import CompileError, compile_source
from redc.physical import lower_to_physical
from redc.physical.pnr import TRACE_SCHEMA as COARSE_SCHEMA
from redc.physical.pnr import PnRConfig, place_and_route
from redc.physical_primitive import PadInterfacePolicy, synthesize_to_primitives
from redc.physical_primitive.pnr import TRACE_SCHEMA as PRIMITIVE_SCHEMA
from redc.physical_primitive.pnr import (
    PrimitivePnRConfig,
    PrimitiveTraceRecorder,
    place_and_route_primitive,
)
from redc.physical_primitive.techmap import map_primitives_to_minecraft
from redc.viewer import (
    PRIMITIVE_TRACE_SCHEMA,
    TRACE_SCHEMA,
    check_primitive_trace,
    check_trace,
    load_any_trace,
    render_pnr_html,
    render_primitive_pnr_html,
    render_trace_html,
    trace_backend,
    write_primitive_pnr_html,
    write_trace_html,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
HALF_ADDER = "uint2 main(bool a, bool b) { return (uint2)a + (uint2)b; }"


@cache
def _primitive_recorder(level: str) -> PrimitiveTraceRecorder:
    graph = compile_source(HALF_ADDER, top="main")
    recorder = PrimitiveTraceRecorder(level)
    recorder.begin_source(path="half_adder.redc", top="main")
    netlist = synthesize_to_primitives(graph, interface=PadInterfacePolicy(), trace=recorder)
    mapped = map_primitives_to_minecraft(netlist, trace=recorder)
    place_and_route_primitive(mapped, PrimitivePnRConfig(), trace=recorder)
    return recorder


def primitive_trace(level: str = "basic") -> dict:
    return json.loads(_primitive_recorder(level).to_json())


@cache
def _coarse_recorder():
    netlist = lower_to_physical(compile_source((EXAMPLES / "uint8_add.redc").read_text(), top="add"))
    return place_and_route(netlist, PnRConfig()).trace


def coarse_trace() -> dict:
    return json.loads(_coarse_recorder().to_json())


def embedded_trace(page: str) -> dict:
    match = re.search(r'<script type="application/json" id="redc-trace">(.*?)</script>', page, re.DOTALL)
    assert match
    return json.loads(match.group(1))


# -- the primitive page -----------------------------------------------------------------


def test_schema_constants_come_from_the_backends() -> None:
    assert PRIMITIVE_TRACE_SCHEMA == PRIMITIVE_SCHEMA == "redc.physical-primitive.pnr.v1"
    assert TRACE_SCHEMA == COARSE_SCHEMA == "redc.pnr.trace.v1"


def test_importing_the_viewer_does_not_import_the_coarse_backend() -> None:
    code = "import sys, redc.viewer as v; print('redc.physical' in sys.modules); v.load_trace; print('redc.physical' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "True"]  # redc.viewer.load_trace still resolves (lazily) to the coarse loader


def test_page_embeds_the_trace_and_controls() -> None:
    trace = primitive_trace("detailed")
    page = render_primitive_pnr_html(trace, title="half <adder> & co")
    assert embedded_trace(page) == trace
    for element in ('id="timeline"', 'id="btn-play"', 'id="btn-back"', 'id="btn-fwd"', 'id="btn-start"',
                    'id="btn-end"', 'id="speed"', 'id="attempt-select"', 'id="phase-select"',
                    'id="iteration-select"', 'id="btn-final"', 'id="hl-mode"', 'id="hl-item"',
                    'id="btn-hl-clear"', 'id="congestion-mode"', 'id="legend"', 'id="selection"',
                    'id="event"', 'id="file-input"', 'id="error"', 'id="status"', 'id="meta"'):  # fmt: skip
        assert element in page, element
    for layer in ("components", "pins", "keepout", "dust", "repeaters", "supports", "clearances",
                  "congestion", "search", "probe", "bounds", "grid"):  # fmt: skip
        assert f'data-layer="{layer}"' in page, layer
    for view in ("fit", "iso", "top", "front", "side"):
        assert f'data-view="{view}"' in page
    assert "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js" in page
    assert "three/addons/" in page and "OrbitControls" in page and "InstancedMesh" in page
    assert "validateTrace" in page and "did not start" in page and "__REDC_" not in page
    assert "<title>half &lt;adder&gt; &amp; co</title>" in page
    # The coarse viewer is a different page.
    assert "pnr_viewer" not in page and 'data-layer="voxels"' not in page


def test_default_titles() -> None:
    assert "<title>RedC primitive P&amp;R replay</title>" in render_primitive_pnr_html(primitive_trace())


def test_page_from_a_trace_file(tmp_path: Path) -> None:
    for name in ("adder.primitive.pnr.json", "adder.primitive.pnr.jsonl"):
        path = _primitive_recorder("basic").write(tmp_path / name)
        page = render_primitive_pnr_html(path)
        assert embedded_trace(page) == primitive_trace()
        assert f"<title>RedC primitive P&amp;R replay - {name}</title>" in page
        assert "<title>custom</title>" in render_primitive_pnr_html(str(path), title="custom")


def test_script_breaking_text_is_escaped() -> None:
    trace = primitive_trace()
    trace["source"]["path"] = "</script><!-- x"
    trace["design"]["instances"][0]["role"] = "<b>role</b>"
    page = render_primitive_pnr_html(trace, title="</title><script>alert(1)</script>")
    block = page.split('id="redc-trace">', 1)[1].split("</script>", 1)[0]
    assert "</script" not in block and "<!--" not in block and "<" not in block
    back = embedded_trace(page)
    assert back["source"]["path"] == "</script><!-- x" and back["design"]["instances"][0]["role"] == "<b>role</b>"
    assert "<title>&lt;/title&gt;&lt;script&gt;alert(1)&lt;/script&gt;</title>" in page


def test_write_primitive_page(tmp_path: Path) -> None:
    out = write_primitive_pnr_html(primitive_trace(), tmp_path / "nested" / "dir" / "adder.primitive.pnr.html", title="t")
    assert out == tmp_path / "nested" / "dir" / "adder.primitive.pnr.html"
    assert out.read_text(encoding="utf-8") == render_primitive_pnr_html(primitive_trace(), title="t")


def _drop(key):
    def mutate(trace):
        del trace[key]

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda t: t.update(schema="something.else"), "schema is 'something.else'"),
        (lambda t: t.update(schema=COARSE_SCHEMA), "schema is 'redc.pnr.trace.v1'"),
        (_drop("design"), "no cells / instances / nets"),
        (lambda t: t["design"].update(cells=None), "no cells / instances / nets"),
        (lambda t: t["design"]["instances"][1].update(cell="nope"), "instance 1 uses unknown cell 'nope'"),
        (lambda t: t["design"]["instances"].append(7), "design instance 10 is not an object"),
        (_drop("events"), "no events list"),
        (lambda t: t["events"][5].update(seq=50), "event 5 is malformed or out of sequence"),
        (lambda t: t["events"].insert(0, "bogus"), "event 0 is malformed"),
        (lambda t: t.update(final=[1]), "final must be an object"),
    ],
)
def test_malformed_primitive_traces_are_rejected(mutate, message: str) -> None:
    trace = primitive_trace()
    mutate(trace)
    with pytest.raises(CompileError, match=re.escape(message)) as error:
        render_primitive_pnr_html(trace)
    assert PRIMITIVE_TRACE_SCHEMA in str(error.value)  # every message names the schema
    with pytest.raises(CompileError):
        check_primitive_trace(trace)


def test_non_objects_are_rejected() -> None:
    for bad in ([], "trace", None, 3):
        with pytest.raises(CompileError, match=re.escape(PRIMITIVE_TRACE_SCHEMA)):
            check_primitive_trace(bad)


def test_a_failed_or_interrupted_trace_still_renders() -> None:
    trace = primitive_trace()
    trace["final"] = {"success": False, "failure": {"stage": "synthesis", "reason": "synthesis", "message": "boom"}}
    assert embedded_trace(render_primitive_pnr_html(trace))["final"]["failure"]["message"] == "boom"
    trace["final"] = None  # a crashed run that never closed its trace
    assert check_primitive_trace(trace) is trace


# -- one renderer per backend, and the dispatcher -------------------------------------------


def test_each_renderer_rejects_the_other_backends_trace() -> None:
    with pytest.raises(CompileError, match=re.escape(f"expected {COARSE_SCHEMA!r}")):
        render_pnr_html(primitive_trace())
    with pytest.raises(CompileError, match=re.escape(f"expected {COARSE_SCHEMA!r}")):
        check_trace(primitive_trace())
    with pytest.raises(CompileError, match=re.escape(f"expected {PRIMITIVE_TRACE_SCHEMA!r}")):
        render_primitive_pnr_html(coarse_trace())


def test_dispatcher_picks_the_viewer_from_the_schema(tmp_path: Path) -> None:
    primitive, coarse = primitive_trace(), coarse_trace()
    assert trace_backend(primitive) == "physical-primitive"
    assert trace_backend(coarse) == "physical"
    assert render_trace_html(primitive, title="x") == render_primitive_pnr_html(primitive, title="x")
    assert render_trace_html(coarse, title="x") == render_pnr_html(coarse, title="x")
    assert 'data-layer="repeaters"' in render_trace_html(primitive)
    assert 'data-layer="voxels"' in render_trace_html(coarse)
    primitive_path = _primitive_recorder("basic").write(tmp_path / "p.primitive.pnr.json")
    coarse_path = _coarse_recorder().write(tmp_path / "c.pnr.jsonl")
    assert trace_backend(primitive_path) == "physical-primitive" and trace_backend(str(coarse_path)) == "physical"
    assert render_trace_html(primitive_path) == render_primitive_pnr_html(primitive_path)
    assert render_trace_html(coarse_path) == render_pnr_html(coarse_path)
    out = write_trace_html(primitive, tmp_path / "a" / "b.html", title="t")
    assert out.read_text(encoding="utf-8") == render_primitive_pnr_html(primitive, title="t")
    out = write_trace_html(coarse_path, tmp_path / "c" / "d.html")
    assert embedded_trace(out.read_text(encoding="utf-8")) == coarse


def test_dispatcher_names_both_schemas_for_anything_else(tmp_path: Path) -> None:
    both = f"expected {COARSE_SCHEMA!r} or {PRIMITIVE_TRACE_SCHEMA!r}"
    for bad in ({"schema": "redc.physical.v1"}, {}, {"schema": 3}):
        with pytest.raises(CompileError, match=re.escape(both)):
            trace_backend(bad)
        with pytest.raises(CompileError, match=re.escape(both)):
            render_trace_html(bad)
        with pytest.raises(CompileError, match=re.escape(both)):
            write_trace_html(bad, tmp_path / "never" / "x.html")
    assert not (tmp_path / "never").exists()
    with pytest.raises(CompileError, match="not a P&R replay trace: schema is 'redc.physical.v1'"):
        trace_backend({"schema": "redc.physical.v1"})
    with pytest.raises(CompileError):
        render_trace_html(42)  # type: ignore[arg-type]


# -- reading trace files of either backend -------------------------------------------------


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_load_any_trace_reads_both_backends(tmp_path: Path, suffix: str) -> None:
    primitive = _primitive_recorder("detailed").write(tmp_path / f"p.primitive.pnr{suffix}")
    coarse = _coarse_recorder().write(tmp_path / f"c.pnr{suffix}")
    loaded = load_any_trace(primitive)
    assert loaded == primitive_trace("detailed") and loaded["schema"] == PRIMITIVE_TRACE_SCHEMA
    loaded = load_any_trace(str(coarse))
    assert loaded == coarse_trace() and loaded["schema"] == COARSE_SCHEMA


def test_load_any_trace_does_not_need_schema_first(tmp_path: Path) -> None:
    trace = primitive_trace()
    reordered = {"events": trace["events"], **{k: v for k, v in trace.items() if k != "events"}}
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(reordered, indent=2), encoding="utf-8")
    assert load_any_trace(path) == trace


def test_load_any_trace_rejects_other_files(tmp_path: Path) -> None:
    both = f"expected {COARSE_SCHEMA!r} or {PRIMITIVE_TRACE_SCHEMA!r}"
    design = tmp_path / "adder.physical.json"
    design.write_text('{"schema": "redc.physical.v1", "instances": []}', encoding="utf-8")
    with pytest.raises(CompileError, match=re.escape(both)) as error:
        load_any_trace(design)
    assert "adder.physical.json: not a P&R replay trace: schema is 'redc.physical.v1'" in str(error.value)
    headless = tmp_path / "events.jsonl"
    headless.write_text('{"record": "event", "seq": 0}\n', encoding="utf-8")
    with pytest.raises(CompileError, match=re.escape(both)):
        load_any_trace(headless)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(CompileError, match="schema is None"):
        load_any_trace(empty)
    listed = tmp_path / "list.json"
    listed.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(CompileError, match=re.escape(both)):
        load_any_trace(listed)
    broken = tmp_path / "broken.json"
    broken.write_text('{"schema": "redc.physical-primitive.pnr.v1", "design": ', encoding="utf-8")
    with pytest.raises(ValueError):
        load_any_trace(broken)
    with pytest.raises(OSError):
        load_any_trace(tmp_path / "missing.json")
