"""Independent review checks of the primitive replay viewer (``primitive_viewer.js``).

Optional, like the viewer's own suite: runs only when ``quickjs`` is installed,

    uv run --with quickjs pytest tests/test_primitive_viewer_review_js.py

The shipped script runs headlessly in QuickJS with the Three.js / DOM stubs of
``tests/test_primitive_viewer_js.py`` (reused, not copied).  These tests probe
what that suite leaves open: orientation math for every quarter turn (keep-outs,
rejected probes drawn from the cell, the selection box), repeater arrows for all
four facings, checkpointed seeking against a pure from-scratch replay (frontier
chunking, failed attempts, a keyframe after legalization), drawn layers against
the replay state under random seeks and layer toggles, the viewer's own route
support / clearance derivation, loading a ``.jsonl`` trace through the file
input, and mesh counts that do not grow with the design.

The last three tests are REVIEW FINDINGS: they fail against the viewer as
reviewed and are kept as evidence until the viewer is fixed.
"""

from __future__ import annotations

import copy
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest

quickjs = pytest.importorskip("quickjs")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # pytest's default (prepend) import mode already adds it
    sys.path.insert(0, str(HERE))

from test_primitive_viewer_js import (  # helpers only: importing a test_* name would collect it twice
    ADD4,
    COUNTER,
    HALF_ADDER,
    HARNESS,
    STUBS,
    failing_config,
    final_maps,
    js,
    page_spec,
    renumber,
    run,
    trace_for,
    with_late_rip_up,
)

from redc.physical_primitive.geometry import Direction, Orientation
from redc.viewer.primitive import PRIMITIVE_SCRIPT

NAMES = ["east", "south", "west", "north"]


def _rotated(trace: dict) -> dict:
    """Every instance turned to ``NAMES[id % 4]`` (v1 placement only uses east)."""
    for event in trace["events"]:
        if event["type"] in ("component_placed", "component_place_attempt"):
            event["orientation"] = NAMES[event["instance"] % 4]
            event.pop("voxels", None)
    for entry in trace["final"]["placement"]:
        entry["orientation"] = NAMES[entry["instance"] % 4]
    return trace


def _cells(trace: dict) -> dict[str, dict]:
    return {c["name"]: c for c in trace["design"]["cells"]}


def _blocks(rows: list) -> list[list[int]]:
    """[id, x, y, z] of drawn unit-ish boxes centred in their block."""
    return sorted([r[1], round(r[2] - 0.5), round(r[3] - 0.5), round(r[4] - 0.5)] for r in rows)


# -- orientation -------------------------------------------------------------------


def test_rotation_helpers_match_orientation_apply_for_all_quarter_turns() -> None:
    trace = trace_for(HALF_ADDER, "basic")
    coords = [list(c) for c in itertools.product(range(-3, 4), range(3), range(-3, 4))]
    names = sorted(_cells(trace))
    out = run(trace, js("""
      var r = { local: [], facing: [], turns: NAMES.map(quarterTurns), bounds: {} };
      __COORDS__.forEach(function (c) { for (var q = 0; q < 4; q++) r.local.push(rotateLocal(c, q)); });
      NAMES.forEach(function (d) { for (var q = 0; q < 4; q++) r.facing.push(rotateFacing(d, q)); });
      __CELLS__.forEach(function (n) {
        for (var q = 0; q < 4; q++) { var oc = viewer.orientedCell(n, q); r.bounds[n + '|' + q] = [oc.lo, oc.hi]; }
      });
      return r;
    """.replace("NAMES", json.dumps(NAMES)), coords=coords, cells=names))
    assert "error" not in out, out
    assert out["turns"] == [0, 1, 2, 3]
    assert out["local"] == [list(Orientation(q).apply(tuple(c))) for c in coords for q in range(4)]
    assert out["facing"] == [Direction.parse(d).rotated(q).label for d in NAMES for q in range(4)]
    # The selection box is the backend's OrientedCell.bounds (voxels, keep-outs and pins).
    cells = _cells(trace)
    for name in names:
        for q in range(4):
            turn = Orientation(q)
            points = [turn.apply(tuple(v["coord"])) for v in cells[name]["voxels"]]
            points += [turn.apply(tuple(c)) for c in cells[name]["keepout"]]
            points += [turn.apply(tuple(p["position"])) for p in cells[name]["pins"]]
            lo = [min(p[i] for p in points) for i in range(3)]
            hi = [max(p[i] for p in points) for i in range(3)]
            assert out["bounds"][f"{name}|{q}"] == [lo, hi], (name, q)


def test_rotated_keepouts_and_rejected_probes_follow_the_backend_rotation() -> None:
    trace = _rotated(trace_for(HALF_ADDER, "detailed"))
    cells = _cells(trace)
    # A rejected probe carries no voxels: the viewer draws it from the rotated cell.
    k = next(i for i, e in enumerate(trace["events"]) if e["type"] == "component_place_attempt" and e["instance"] % 4 == 1)
    probe = trace["events"][k]
    origin = [probe["origin"][0], probe["origin"][1], probe["origin"][2] - 3]
    rejected = {"phase": "placement", "type": "component_place_rejected", "instance": probe["instance"],
                "origin": origin, "orientation": probe["orientation"], "reason": "review probe"}
    trace["events"] = renumber(trace["events"][:k] + [rejected] + trace["events"][k:])
    out = run(trace, js("""
      setLayer('keepout', true); frame();
      var keepout = drawn('keepout');
      viewer.seek(__AT__); frame();
      return { keepout: keepout, probe: drawn('probe'), status: elements.status.textContent };
    """, at=k + 1))
    assert "error" not in out, out
    want = []
    for entry in trace["final"]["placement"]:
        turn = Orientation.parse(entry["orientation"])
        cell = cells[trace["design"]["instances"][entry["instance"]]["cell"]]
        for c in cell["keepout"]:
            x, y, z = turn.apply(tuple(c))
            want.append([entry["instance"], entry["origin"][0] + x, entry["origin"][1] + y, entry["origin"][2] + z])
    assert _blocks(out["keepout"]) == sorted(want)
    turn = Orientation.parse(probe["orientation"])
    assert probe["orientation"] == "south"
    cell = cells[trace["design"]["instances"][probe["instance"]]["cell"]]
    want_probe = sorted([probe["instance"], *(o + d for o, d in zip(origin, turn.apply(tuple(v["coord"]))))]
                        for v in cell["voxels"])
    assert _blocks(out["probe"]) == want_probe
    assert "REJECTED: review probe" in out["status"]


def test_repeater_arrows_point_along_all_four_facings() -> None:
    trace = trace_for(HALF_ADDER, "basic")
    realized = next(e for e in trace["events"] if e["type"] == "route_realized")
    elements = [{"coord": [100, 3, 100], "kind": "dust", "parent": None, "strength": 15, "delay": 0}]
    for i, facing in enumerate(NAMES):
        elements.append({"coord": [100 + 10 * (i + 1), 3, 100], "kind": "repeater", "parent": None,
                         "strength": 3, "delay": 2, "facing": facing})
    synthetic = dict(realized, elements=elements, repeaters=elements[1:], supports=[], clearances=[], sinks=[])
    trace["events"] = renumber(trace["events"][:-1] + [synthetic] + trace["events"][-1:])
    out = run(trace, "frame(); return drawn('routes').filter(function (r) { return r[0] === 'arrow' && r[2] > 105; });")
    assert "error" not in out, out
    got = {round((r[2] - 100.5) / 10): (round(r[5] / 0.66), round(r[6] / 0.66), round(r[8] / 0.5), round(r[9] / 0.5))
           for r in out}
    # Local +x (the wedge tip) maps to the facing; local +z to the facing turned clockwise.
    for i, facing in enumerate(NAMES):
        fx, _, fz = Direction.parse(facing).vector
        assert got[i + 1] == (fx, fz, -fz, fx), facing


# -- seeking -----------------------------------------------------------------------


def _repair_with_keyframe(trace: dict) -> dict:
    """After legalization round 0: a repair iteration that rips up net 0,
    recommits it and ends in a keyframe, then round 1 realizes it again."""
    ev = trace["events"]
    net = 0
    commit = max(k for k, e in enumerate(ev) if e["type"] == "net_route_committed" and e["net"] == net)
    real = max(k for k, e in enumerate(ev) if e["type"] == "route_realized" and e["net"] == net)
    done = next(k for k, e in enumerate(ev) if e["type"] == "legalization_complete")
    routes: dict[int, dict] = {}
    for e in ev[:done]:
        if e["type"] == "net_route_committed":
            routes[e["net"]] = {k: v for k, v in e.items() if k not in ("seq", "phase", "type", "iteration")}
        elif e["type"] == "net_rip_up":
            routes.pop(e["net"], None)
    extra = [
        {"phase": "routing", "type": "routing_iteration_begin", "iteration": 50, "present_factor": 9.0, "nets": [net]},
        {"phase": "routing", "type": "net_rip_up", "net": net, "iteration": 50, "reason": "legalization",
         "cells": ev[commit]["cells"], "length": ev[commit]["length"]},
        dict(ev[commit], iteration=50),
        {"phase": "routing", "type": "routing_iteration_end", "iteration": 50, "present_factor": 9.0, "rerouted": 1,
         "changed": 0, "conflicts": 0, "conflict_cells": 0, "routed_cells": 1, "history_total": 0.0},
        {"phase": "congestion", "type": "congestion_snapshot", "iteration": 50, "present_factor": 9.0,
         "conflicts": [], "cells": [], "history_cells": 0},
        {"phase": "routing", "type": "keyframe", "iteration": 50, "routes": [routes[n] for n in sorted(routes)]},
        {"phase": "routing", "type": "routing_complete", "iteration": 50, "iterations": 51,
         "routed_nets": len(routes), "rip_ups": 1, "failure": None},
        {"phase": "legalization", "type": "legalization_begin", "round": 1, "nets": len(routes)},
        dict(ev[real], round=1),
        {"phase": "legalization", "type": "legalization_complete", "round": 1, "realized": len(routes),
         "failures": 0, "repeaters": 0},
    ]
    trace["events"] = renumber(ev[: done + 1] + json.loads(json.dumps(extra)) + ev[done + 1:])
    return trace


SEEK_CASES: dict[str, tuple[Any, dict[str, int]]] = {
    "search, frontier drawn in chunks of 7": (
        lambda: trace_for(HALF_ADDER, "search", keyframe_interval=1), {"MAX_SEARCH_CELLS": 7, "MAX_BLOCKED_MARKS": 3}),
    "unroutable, two attempts": (lambda: trace_for(HALF_ADDER, "detailed", **failing_config("unroutable")), {}),
    "congestion, two attempts": (lambda: trace_for(HALF_ADDER, "basic", **failing_config("congestion")), {}),
    "rip-up after legalization": (lambda: with_late_rip_up(trace_for(HALF_ADDER, "basic")), {}),
    "keyframe after legalization": (lambda: _repair_with_keyframe(trace_for(HALF_ADDER, "basic", keyframe_interval=1)), {}),
}

CHECKPOINTED_SEEKS = r"""
  var v = viewer, take = v.takeCheckpoint, restarts = v.restarts, plain = {}, fast = {}, back = {}, plainBack = {};
  // A pure replay: events 0..t-1 from an empty state, no checkpoints, no attempt shortcut.
  function plainAt(t) {
    v.checkpoints = new Map(); v.checkpointList = []; v.restarts = []; v.takeCheckpoint = function () {};
    v.state = v.emptyState(); v.position = 0; v.seek(t);
    var d = fullDump(v);
    v.restarts = restarts; v.takeCheckpoint = take;
    return d;
  }
  __SAMPLE__.forEach(function (t) { plain[t] = plainAt(t); });
  v.checkpointInterval = 13; v.checkpoints = new Map(); v.checkpointList = [];
  v.seek(0); v.seek(v.events.length);
  __ORDER__.forEach(function (t) { v.seek(t); frame(); fast[t] = fullDump(v); });
  v.seek(v.events.length);
  for (var i = v.events.length; i >= Math.max(0, v.events.length - 40); i--) { v.seek(i); frame(); back[i] = fullDump(v); }
  var made = v.checkpointList.length;
  Object.keys(back).forEach(function (t) { plainBack[t] = plainAt(Number(t)); });
  return { plain: plain, fast: fast, back: back, plainBack: plainBack, made: made };
"""


@pytest.mark.parametrize("name", list(SEEK_CASES))
def test_checkpointed_seeks_equal_a_plain_replay(name: str) -> None:
    make, constants = SEEK_CASES[name]
    trace = make()
    n = len(trace["events"])
    rng = random.Random(n)
    sample = sorted(set(rng.sample(range(n + 1), min(n + 1, 50))) | {0, 1, n - 1, n})
    order = sample[:]
    rng.shuffle(order)
    out = run(trace, js(CHECKPOINTED_SEEKS, sample=sample, order=order), constants=constants)
    assert "error" not in out, out
    assert out["made"] > 0
    for t in sample:
        assert out["fast"][str(t)] == out["plain"][str(t)], ("seek", t)
    for t, state in out["back"].items():
        assert state == out["plainBack"][t], ("step back", t)
    expected = final_maps(trace)
    for part in ("placed", "routes", "realized"):
        assert out["plain"][str(n)][part] == expected[part], part


# -- what is drawn -------------------------------------------------------------------


def test_drawn_layers_track_the_replay_state_through_seeks_and_toggles() -> None:
    for trace, constants in ((trace_for(HALF_ADDER, "search", keyframe_interval=1), {"MAX_SEARCH_CELLS": 9}),
                             (trace_for(ADD4, "detailed", keyframe_interval=2), {}),
                             (trace_for(HALF_ADDER, "detailed", **failing_config("unroutable")), {})):
        n = len(trace["events"])
        rng = random.Random(n)
        targets = [rng.randint(0, n) for _ in range(90)]
        flips = [rng.choice([None, None, "components", "dust", "repeaters", "supports", "search"]) for _ in targets]
        out = run(trace, js("""
          var v = viewer, bad = [], on = { components: true, dust: true, repeaters: true, supports: false, search: true };
          setLayer('supports', false);
          v.checkpointInterval = 17; v.checkpoints = new Map(); v.checkpointList = []; v.seek(0); v.seek(v.events.length);
          function keys(list) { return list.sort().join(' '); }
          __T__.forEach(function (t, k) {
            var f = __F__[k];
            if (f) { on[f] = !on[f]; setLayer(f, on[f]); }
            v.seek(t); frame();
            var s = v.state;
            if (on.components) {
              var voxels = 0;
              s.placed.forEach(function (p, id) { voxels += v.orientedCell(v.instances[id].cell, p.q).voxels.length / 4; });
              if (drawn('components').length !== voxels) bad.push(['components', t]);
            }
            if (on.dust) {
              var want = [];
              s.realized.forEach(function (r, id) { r.elements.forEach(function (e) { if (e.kind === 'dust') want.push(id + ':' + e.coord.join(',')); }); });
              s.routes.forEach(function (r, id) { if (!s.realized.has(id)) r.cells.forEach(function (c) { want.push(id + ':' + c.join(',')); }); });
              var got = drawn('routes').filter(function (r) { return r[0] === 'dust:pad'; })
                .map(function (r) { return r[1] + ':' + [r[2] - 0.5, Math.round(r[3] - 0.04), r[4] - 0.5].join(','); });
              if (keys(want) !== keys(got)) bad.push(['dust', t]);
            }
            if (on.repeaters) {
              var reps = 0; s.realized.forEach(function (r) { reps += r.repeaters.length; });
              if (drawn('routes').filter(function (r) { return r[0] === 'arrow'; }).length !== reps) bad.push(['repeaters', t]);
            }
            if (on.supports) {
              var sup = 0;
              s.realized.forEach(function (r) { sup += r.supports.length; });
              s.routes.forEach(function (r, id) { if (!s.realized.has(id)) sup += v.routeGeometry(r, false).supports.length; });
              if (drawn('supports').length !== sup) bad.push(['supports', t]);
            }
            if (on.search && v.hasSearch) {
              var cells = s.search.cells.map(function (e) { return e.coord.join(','); });
              var dc = drawn('search').filter(function (r) { return r[0] === 'search'; })
                .map(function (r) { return [r[2] - 0.5, r[3] - 0.5, r[4] - 0.5].join(','); });
              var blocked = drawn('search').filter(function (r) { return r[0] === 'search:blocked'; }).length;
              if (keys(cells) !== keys(dc) || blocked !== s.search.blocked.length) bad.push(['search', t]);
            }
          });
          return bad;
        """, t=targets, f=flips), constants=constants)
        assert out == [], out


def test_derived_tree_supports_clearances_and_parents_match_the_backend() -> None:
    """A committed (not yet legalized) tree is drawn from the viewer's own
    RouteTree.parent / .supports / .clearances derivation."""
    for trace in (trace_for(ADD4, "basic"), trace_for(COUNTER, "basic")):
        out = run(trace, """
          var v = viewer, bad = [];
          function key(a) { return a.map(function (c) { return c.join(','); }).sort().join(' '); }
          v.final.routes.forEach(function (r) {
            var real = v.final.realized.find(function (x) { return x.net === r.net; });
            var g = v.routeGeometry(r, false);
            if (key(g.supports) !== key(real.supports)) bad.push(['supports', r.net]);
            if (key(g.clearances) !== key(real.clearances)) bad.push(['clearances', r.net]);
            var parent = {};
            real.elements.forEach(function (e) { parent[e.coord.join(',')] = e.parent ? e.parent.join(',') : null; });
            for (var i = 0; i < g.n; i++) {
              var c = [g.xyz[i * 3], g.xyz[i * 3 + 1], g.xyz[i * 3 + 2]].join(','), p = g.parent[i];
              var pc = p >= 0 ? [g.xyz[p * 3], g.xyz[p * 3 + 1], g.xyz[p * 3 + 2]].join(',') : null;
              if (parent[c] !== pc) { bad.push(['parent', r.net, c]); break; }
            }
          });
          return bad;
        """)
        assert out == [], out


def test_loading_a_jsonl_trace_through_the_file_input_replaces_the_viewer() -> None:
    first, second = trace_for(HALF_ADDER, "basic"), trace_for(ADD4, "detailed")
    header = {k: v for k, v in second.items() if k not in ("events", "final")}
    records = [{"record": "header", **header}, *({"record": "event", **e} for e in second["events"]),
               {"record": "final", "final": second["final"]}]
    jsonl = "\r\n".join(json.dumps(r) for r in records) + "\r\n"  # CRLF, as a Windows editor may save it
    source = PRIMITIVE_SCRIPT.read_text(encoding="utf-8")
    body = "\n".join(line for line in source.splitlines() if not line.startswith("import "))
    context = quickjs.Context()
    context.eval(STUBS.replace("__SPEC__", json.dumps(page_spec())))
    context.eval("document.getElementById('redc-trace').textContent = " + json.dumps(json.dumps(first)) + ";")
    context.eval(body.replace("export function", "function"))
    context.eval(HARNESS)

    def load(name: str, text: str) -> Any:
        context.eval("elements['file-input']._l.change[0]({ target: { files: [{ name: " + json.dumps(name) +
                     ", text: function () { return Promise.resolve(" + json.dumps(text) + "); } }] } });")
        while context.execute_pending_job():
            pass
        return json.loads(context.eval("""JSON.stringify({
          error: elements.error.hidden ? '' : elements.error.textContent, events: viewer.events.length,
          alive: viewer.alive, canvases: elements.scene.children.filter(function (c) { return c.tagName === 'CANVAS'; }).length,
          state: dump(viewer.state) })"""))

    context.eval("var old = viewer;")
    out = load("add4.primitive.pnr.jsonl", jsonl)
    assert out["error"] == "" and out["events"] == len(second["events"]) and out["canvases"] == 1
    assert context.eval("viewer !== old && old.alive === false")
    expected = final_maps(second)
    for part in ("placed", "routes", "realized"):
        assert out["state"][part] == expected[part], part
    coarse = json.dumps({"schema": "redc.pnr.trace.v1", "design": {"components": [], "instances": [], "nets": []}, "events": []})
    out = load("c.pnr.json", coarse)
    assert "coarse" in out["error"] and out["alive"] and out["events"] == len(second["events"])
    out = load("broken.json", "{oops")
    assert "could not parse broken.json" in out["error"]


def _chain_trace(n: int) -> dict:
    """``n`` AND gates in a grid, each driving the next through a six-block
    route with one repeater: a design of any size, built without P&R."""
    base = trace_for(HALF_ADDER, "basic")
    cell = next(c for c in base["design"]["cells"] if c["kind"] == "and")
    side = int(n ** 0.5) + 1
    instances = [{"id": i, "kind": "and", "category": "gate", "cell": cell["name"], "realizes": [i], "ir_node": i % 5,
                  "ir_op": "and", "ir_type": "uint8", "role": "and", "bit": i % 8, "group": 0, "port": None, "init": None,
                  "peripheral": None, "origin": None, "orientation": None} for i in range(n)]
    nets, events = [], [{"phase": "pnr", "type": "pnr_attempt_begin", "attempt": 0, "component_spacing": 1,
                         "channel_width": 6, "routing_margin": 6, "max_y": 9}]
    for i in range(n):
        events.append({"phase": "placement", "type": "component_placed", "instance": i, "column": 0, "block": "n0",
                       "origin": [10 * (i % side), 0, 6 * (i // side)], "orientation": "east", "bounds": None})
    for i in range(n - 1):
        a = [10 * (i % side) + 4, 1, 6 * (i // side) + 1]
        cells = [[a[0] + k, 1, a[2]] for k in range(6)]
        nets.append({"id": i, "role": "data", "width": 1, "fanout": 1, "driver": {"instance": i, "pin": "y"},
                     "sinks": [{"instance": i + 1, "pin": "a"}], "logical": [{"ir_node": i % 5, "bit": i % 8}], "ports": []})
        events.append({"phase": "routing", "type": "net_route_committed", "iteration": 0, "net": i,
                       "driver": {"instance": i, "pin": "y"}, "root": a, "cells": cells, "length": 6, "fanout": 1,
                       "branches": [{"sink": {"instance": i + 1, "pin": "a"}, "start": a, "goal": cells[-1], "path": cells}]})
        elements = [{"coord": c, "kind": "repeater" if k == 3 else "dust", "parent": None if k == 0 else cells[k - 1],
                     "strength": 15 - k, "delay": 0, **({"facing": "east"} if k == 3 else {})} for k, c in enumerate(cells)]
        events.append({"phase": "legalization", "type": "route_realized", "round": 0, "net": i, "powered": True,
                       "elements": elements, "repeaters": [elements[3]], "supports": [[c[0], 0, c[2]] for c in cells[1:-1]],
                       "clearances": [], "sinks": [], "min_strength": 10, "max_delay_ticks": 1})
    trace = copy.deepcopy(base)
    trace["design"] = dict(trace["design"], cells=[cell], instances=instances, nets=nets,
                           groups=[{"id": 0, "name": "design", "kind": "root", "parent": None, "ir_node": None}],
                           ports=[], buses=[], ir_nodes=[])
    trace["events"] = renumber(events)
    trace["final"] = {"success": True, "attempt": 0, "attempts": 1, "placement": [], "routes": [], "realized": []}
    return trace


def test_mesh_count_does_not_grow_with_design_size() -> None:
    counts = {}
    for n in (200, 800):
        trace = _chain_trace(n)
        out = run(trace, """
          var v = viewer;
          setLayer('keepout', true); setLayer('supports', true); frame();
          v.setHighlight({ mode: 'ir_node', id: 1 }); frame();
          v.select({ kind: 'net', id: 3 }); frame();
          var meshes = 0, plain = 0;
          v.scene.traverse(function (o) { if (o.isInstancedMesh) meshes++; else if (o.isMesh) plain++; });
          return { meshes: meshes, plain: plain, voxels: drawn('components').length, pads: drawn('routes').length };
        """)
        assert "error" not in out, out
        voxels = len(trace["design"]["cells"][0]["voxels"])
        assert out["voxels"] == n * voxels
        counts[n] = (out["meshes"], out["plain"])
    assert counts[200] == counts[800], counts  # one InstancedMesh per style, never one Mesh per block


# -- REVIEW FINDINGS (fail against the reviewed viewer) ---------------------------------------


def test_grid_lines_fall_on_block_boundaries() -> None:
    """REVIEW FINDING: buildGrid centres a square GridHelper of side
    max(dx, dz) + 2 on the bounds' centre, so along the axis whose extent has
    the other parity every grid line sits half a block off the block
    boundaries (the same code as pnr_viewer.js, where cells are tiles)."""
    trace = trace_for(HALF_ADDER, "basic")
    out = run(trace, """
      var v = viewer, made = [];
      var Real = THREE.GridHelper;
      THREE.GridHelper = class extends Real {
        constructor(size, divisions) { super(size, divisions); made.push(this); this.size = size; this.divisions = divisions; }
      };
      var r = [];
      [[10, 5], [5, 10], [84, 53]].forEach(function (d) {
        v.state.searchBounds = { min: [0, 1, 0], max: [d[0] - 1, 9, d[1] - 1] };
        v.bump('bounds'); v.sync();
        var g = made[made.length - 1];
        r.push({ dims: d, first: [g.position.x - g.size / 2, g.position.z - g.size / 2], step: g.size / g.divisions });
      });
      return r;
    """)
    assert "error" not in out, out
    for grid in out:
        assert grid["step"] == 1, grid
        # Lines at first + k: block boundaries are the integers.
        assert all(float(c).is_integer() for c in grid["first"]), grid


def _legalization_repair(trace: dict) -> tuple[dict, int]:
    """Round 0 fails one net's legalization; a repair iteration rips it up and
    reroutes it; round 1 realizes every net -- the run succeeds."""
    ev = trace["events"]
    begin = next(k for k, e in enumerate(ev) if e["type"] == "legalization_begin")
    real = next(k for k in range(begin, len(ev)) if ev[k]["type"] == "route_realized")
    net = ev[real]["net"]
    commit = max(k for k in range(begin) if ev[k]["type"] == "net_route_committed" and ev[k]["net"] == net)
    done = next(k for k in range(begin, len(ev)) if ev[k]["type"] == "legalization_complete")
    coord = ev[real]["elements"][-1]["coord"]
    failed = {"phase": "legalization", "type": "legalization_failed", "net": net, "round": 0,
              "reason": "signal too weak and no repeater site upstream",
              "message": f"net {net}: signal too weak at {coord}", "coord": coord, "sink": None}
    round0 = [e for e in ev[begin + 1:done] if not (e.get("net") == net and e["type"] in ("route_realized", "repeater_inserted"))]
    repair = [
        dict(ev[done], failures=1),
        {"phase": "routing", "type": "routing_iteration_begin", "iteration": 9, "present_factor": 4.0, "nets": [net]},
        {"phase": "routing", "type": "net_rip_up", "net": net, "iteration": 9, "reason": "legalization",
         "cells": ev[commit]["cells"], "length": ev[commit]["length"]},
        dict(ev[commit], iteration=9),
        {"phase": "routing", "type": "routing_iteration_end", "iteration": 9, "present_factor": 4.0, "rerouted": 1,
         "changed": 0, "conflicts": 0, "conflict_cells": 0, "routed_cells": 0, "history_total": 0.0},
        {"phase": "congestion", "type": "congestion_snapshot", "iteration": 9, "present_factor": 4.0,
         "conflicts": [], "cells": [], "history_cells": 0},
        {"phase": "routing", "type": "routing_complete", "iteration": 9, "iterations": 10, "routed_nets": 0,
         "rip_ups": 1, "failure": None},
        {"phase": "legalization", "type": "legalization_begin", "round": 1, "nets": 0},
    ]
    round1 = [dict(e, round=1) for e in ev[begin + 1:done]]
    trace["events"] = renumber(ev[: begin + 1] + json.loads(json.dumps([failed, *round0, *repair, *round1])) + ev[done:])
    return trace, net


def test_legalization_failure_marker_clears_once_the_net_is_realized() -> None:
    """REVIEW FINDING: a branch failure marker is dropped once its net routes
    after all (dropRoutingFailures), but a legalization failure marker is
    never dropped -- after a successful repair round the finished, successful
    design still reports "1 failure(s)" and draws a red failure box where the
    ripped-up route used to be."""
    trace, net = _legalization_repair(trace_for(HALF_ADDER, "basic"))
    assert trace["final"]["success"]
    out = run(trace, """
      var v = viewer; v.seek(v.events.length); frame();
      return { failures: v.state.failures, status: elements.status.textContent, realized: v.state.realized.has(__NET__),
               markers: drawn('congestion').filter(function (r) { return r[0] === 'marker'; }).length, state: dump(v.state) };
    """.replace("__NET__", str(net)))
    assert "error" not in out, out
    expected = final_maps(trace)
    for part in ("placed", "routes", "realized"):
        assert out["state"][part] == expected[part], part  # the replay itself is right
    assert out["realized"] and "P&R succeeded" in out["status"]
    assert [f for f in out["failures"] if f["net"] == net] == [], out["failures"]
    assert out["markers"] == 0 and "failure(s)" not in out["status"], out["status"]


def test_relaxed_search_status_names_the_actual_mode() -> None:
    """REVIEW FINDING: branch_search_relaxed always says "searching again
    ignoring congestion", but its first relaxation is mode "greedy", which
    still prices congestion (docs/physical-primitive-trace.md); and
    branch_search_stats reads a non-existent ``relaxed`` field instead of the
    documented ``mode``, so the status never names the mode."""
    trace = trace_for(HALF_ADDER, "basic")
    k = next(i for i, e in enumerate(trace["events"]) if e["type"] == "routing_iteration_begin") + 1
    sink = trace["design"]["nets"][0]["sinks"][0]
    extra = [
        {"phase": "routing", "type": "branch_search_relaxed", "net": 0, "iteration": 0, "sink": sink, "mode": "greedy",
         "expansions": 7},
        {"phase": "routing", "type": "branch_search_stats", "net": 0, "iteration": 0, "sink": sink, "attempt": 1,
         "mode": "greedy", "expansions": 14, "found": True, "blocked": {}},
        {"phase": "routing", "type": "branch_search_relaxed", "net": 0, "iteration": 0, "sink": sink,
         "mode": "ignore_congestion", "expansions": 14},
    ]
    trace["events"] = renumber(trace["events"][:k] + extra + trace["events"][k:])
    out = run(trace, js("""
      var r = [];
      for (var i = 1; i <= 3; i++) { viewer.seek(__AT__ + i); r.push(viewer.state.status); }
      return r;
    """, at=k))
    assert "error" not in out, out
    greedy, stats, ignore = out
    assert "ignoring congestion" in ignore  # the last resort is described correctly
    assert "greedy" in greedy and "ignoring congestion" not in greedy, greedy
    assert "greedy" in stats, stats
