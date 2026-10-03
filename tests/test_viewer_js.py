"""Execute the real browser viewer script against real traces, headlessly.

Optional: runs only when the ``quickjs`` package is available, e.g.

    uv run --with quickjs pytest tests/test_viewer_js.py

The module's Three.js imports are replaced by small stubs that keep the parts
of the API the viewer depends on honest (scene-graph add/remove, instance
counts, DOM parent/child bookkeeping), so replay, seeking, selection, layer
toggles and playback all execute the shipped JavaScript.
"""

import json
from pathlib import Path

import pytest

from redc import compile_source
from redc.physical import lower_to_physical
from redc.physical.pnr import PnRConfig, place_and_route
from redc.viewer import SCRIPT

quickjs = pytest.importorskip("quickjs")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

STUBS = r"""
var rafs = [];
function makeElement(tag) {
  return {
    tagName: (tag || 'div').toUpperCase(), children: [], style: {}, dataset: {},
    className: '', textContent: '', _inner: '', hidden: false, value: '', checked: false,
    clientWidth: 800, clientHeight: 600, parentNode: null, files: [],
    set innerHTML(v) { this._inner = v; this.children.forEach(function (c) { c.parentNode = null; }); this.children = []; },
    get innerHTML() { return this._inner; },
    appendChild: function (c) { this.children.push(c); c.parentNode = this; return c; },
    removeChild: function (c) {
      var i = this.children.indexOf(c);
      if (i < 0) throw new Error('removeChild: not a child');
      this.children.splice(i, 1); c.parentNode = null; return c;
    },
    addEventListener: function (t, f) { (this._l = this._l || {})[t] = f; },
    getBoundingClientRect: function () { return { left: 0, top: 0, width: 800, height: 600 }; },
  };
}
var elements = {};
var layerBoxes = ['components', 'labels', 'ports', 'wires', 'voxels', 'search', 'congestion', 'grid', 'bounds']
  .map(function (n) { var b = makeElement('input'); b.dataset.layer = n; b.checked = n !== 'labels' && n !== 'voxels'; return b; });
var viewButtons = ['fit', 'iso', 'top', 'front', 'side']
  .map(function (n) { var b = makeElement('button'); b.dataset.view = n; return b; });
var document = {
  getElementById: function (id) { return elements[id] || (elements[id] = makeElement('div')); },
  querySelectorAll: function (sel) { return sel === '[data-layer]' ? layerBoxes : sel === '[data-view]' ? viewButtons : []; },
  createElement: makeElement,
  createTextNode: function (t) { return { textContent: t }; },
};
document.getElementById('speed').value = '20';
var window = { addEventListener: function () {}, removeEventListener: function () {}, devicePixelRatio: 1 };
function requestAnimationFrame(f) { rafs.push(f); }

class Vec3 {
  constructor(x, y, z) { this.x = x || 0; this.y = y || 0; this.z = z || 0; }
  set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
  copy(v) { this.x = v.x; this.y = v.y; this.z = v.z; return this; }
  clone() { return new Vec3(this.x, this.y, this.z); }
  add(v) { this.x += v.x; this.y += v.y; this.z += v.z; return this; }
  sub(v) { this.x -= v.x; this.y -= v.y; this.z -= v.z; return this; }
  multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; return this; }
  length() { return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z); }
  normalize() { return this.multiplyScalar(1 / (this.length() || 1)); }
}
class Obj3D {
  constructor() { this.children = []; this.parent = null; this.position = new Vec3(); this.scale = new Vec3(1, 1, 1); this.visible = true; this.userData = {}; }
  add(o) { this.children.push(o); o.parent = this; return this; }
  remove(o) { var i = this.children.indexOf(o); if (i >= 0) { this.children.splice(i, 1); o.parent = null; } return this; }
  traverse(f) { f(this); this.children.forEach(function (c) { c.traverse(f); }); }
}
class Geometry { constructor() { this.userData = {}; } dispose() {} setAttribute() { return this; } }
class Material { constructor(p) { Object.assign(this, p || {}); } dispose() {} }
class Color { constructor(c) { this.c = c || 0; } setHSL() { return this; } clone() { return new Color(this.c); } multiplyScalar() { return this; } }
class Mesh extends Obj3D { constructor(g, m) { super(); this.geometry = g; this.material = m; } }
class InstancedMesh extends Mesh {
  constructor(g, m, n) { super(g, m); if (!(n > 0)) throw new Error('empty InstancedMesh'); this.count = n; this.filled = 0; this.instanceMatrix = { setUsage: function () {} }; }
  setMatrixAt(i) { if (i < 0 || i >= this.count) throw new Error('instance ' + i + ' >= ' + this.count); this.filled++; }
  setColorAt(i) { if (i < 0 || i >= this.count) throw new Error('color ' + i + ' >= ' + this.count); }
}
var THREE = {
  Vector3: Vec3, Vector2: class { constructor(x, y) { this.x = x; this.y = y; } },
  Group: Obj3D, Scene: Obj3D, Mesh: Mesh, InstancedMesh: InstancedMesh, LineSegments: Mesh,
  Box3Helper: class extends Obj3D { constructor(b) { super(); if (!(b.min instanceof Vec3)) throw new Error('Box3Helper'); } },
  GridHelper: class extends Obj3D {}, AmbientLight: Obj3D, DirectionalLight: Obj3D,
  PerspectiveCamera: class extends Obj3D { updateProjectionMatrix() {} },
  WebGLRenderer: class { constructor() { this.domElement = makeElement('canvas'); } setPixelRatio() {} setSize() {} render() {} dispose() {} },
  BoxGeometry: Geometry, SphereGeometry: Geometry, EdgesGeometry: Geometry, BufferGeometry: Geometry,
  Float32BufferAttribute: class { constructor(a, n) { if (a.length % n) throw new Error('attribute length'); } },
  MeshStandardMaterial: Material, MeshBasicMaterial: Material, LineBasicMaterial: Material, Color: Color,
  Matrix4: class { compose(p, q, s) { if (!(p instanceof Vec3) || !(s instanceof Vec3)) throw new Error('compose'); return this; } },
  Quaternion: class {}, Box3: class { constructor(a, b) { this.min = a; this.max = b; } },
  Raycaster: class { setFromCamera() {} intersectObjects(t) { return t.length ? [{ object: t[0] }] : []; } },
  Clock: class { getDelta() { return 0.5; } }, DynamicDrawUsage: 35048,
};
var OrbitControls = class { constructor() { this.target = new Vec3(); } update() {} };
var CSS2DRenderer = class { constructor() { this.domElement = makeElement('div'); } setSize() {} render() {} };
var CSS2DObject = class extends Obj3D { constructor(el) { super(); this.element = el; this.isCSS2DObject = true; } };
"""

DRIVER = r"""
(function () {
  if (!viewer) {
    return JSON.stringify({ error: elements.error.children.map(function (c) { return c.textContent; }).join(' | ') +
      ' ' + JSON.stringify(elements.error.children.length ? elements.error.children[1].children.map(function (l) { return l.textContent; }) : []) });
  }
  var v = viewer;
  var n = v.events.length;
  [n, 3, n - 1, 0, Math.floor(n / 2), n].forEach(function (t) { v.seek(t); });
  var endState = v.state;
  v.seek(0);
  for (var i = 0; i < n; i++) v.seek(i + 1);
  var stepState = v.state;
  var firstNet = v.trace.design.nets.length ? v.trace.design.nets[0].id : null;
  v.select({ kind: 'instance', id: v.trace.design.instances[0].id });
  if (firstNet !== null) v.select({ kind: 'net', id: firstNet });
  v.pick({ clientX: 5, clientY: 5 });
  layerBoxes.forEach(function (b) { b.checked = !b.checked; b.onchange(); });
  layerBoxes.forEach(function (b) { b.checked = !b.checked; b.onchange(); });
  viewButtons.forEach(function (b) { b.onclick(); });
  document.getElementById('btn-start').onclick();
  document.getElementById('btn-play').onclick();
  for (var k = 0; k < 40 && v.playing; k++) rafs[rafs.length - 1]();
  v.onKey({ key: 'ArrowLeft', target: {}, preventDefault: function () {} });
  v.onKey({ key: 'End', target: {}, preventDefault: function () {} });
  document.getElementById('attempt-select').value = String(v.restarts.length ? v.restarts[0] + 1 : 0);
  document.getElementById('attempt-select').onchange();
  v.seek(n);
  function dump(s) {
    var placed = {};
    s.placed.forEach(function (o, id) { placed[id] = o; });
    var routes = {};
    s.routes.forEach(function (r, id) { routes[id] = r.branches; });
    return { placed: placed, routes: routes, congestion: s.congestion.length, status: s.status };
  }
  return JSON.stringify({
    end: dump(endState), stepped: dump(stepState), final: dump(v.state),
    wires: v.groups.wires.children.length, components: v.groups.components.children.length,
    selection: document.getElementById('selection').children.length,
  });
})()
"""


def run_viewer(trace: dict) -> dict:
    script = SCRIPT.read_text(encoding="utf-8")
    body = "\n".join(line for line in script.splitlines() if not line.startswith("import "))
    body = body.replace("export function", "function")
    context = quickjs.Context()
    context.eval(STUBS)
    context.eval(
        "document.getElementById('redc-trace').textContent = " + json.dumps(json.dumps(trace)) + ";"
    )
    context.eval(body)
    return json.loads(context.eval(DRIVER))


def as_maps(trace: dict) -> tuple[dict, dict]:
    final = trace["final"]
    placed = {str(p["instance"]): p["origin"] for p in final["placement"] if p["origin"]}
    routes = {str(r["net"]): r["branches"] for r in final["routes"]}
    return placed, routes


def trace_for(example: str, top: str, **config) -> dict:
    netlist = lower_to_physical(compile_source((EXAMPLES / example).read_text(), top=top))
    return place_and_route(netlist, PnRConfig(**config)).trace.to_dict()


@pytest.mark.parametrize("level", ["basic", "detailed", "search"])
def test_viewer_replays_to_the_final_state(level: str) -> None:
    trace = trace_for("uint8_fib.redc", "fib", trace_level=level, keyframe_interval=2)
    out = run_viewer(trace)
    assert "error" not in out, out
    placed, routes = as_maps(trace)
    for state in ("end", "stepped", "final"):
        assert out[state]["placed"] == placed, state
        assert out[state]["routes"] == routes, state
        assert out[state]["congestion"] == 0
    assert out["components"] == len(trace["design"]["instances"])
    assert out["wires"] >= len(trace["design"]["nets"])
    assert out["selection"] > 0


def test_viewer_shows_a_failed_multi_attempt_trace() -> None:
    trace = trace_for("uint8_add.redc", "add", max_astar_expansions=1, max_pnr_attempts=2)
    out = run_viewer(trace)
    assert "error" not in out, out
    assert "FAILED" in out["final"]["status"]


def test_viewer_shows_an_eventless_trace_from_its_final_block() -> None:
    trace = trace_for("uint8_add.redc", "add", trace_level="none")
    out = run_viewer(trace)
    assert "error" not in out, out
    placed, routes = as_maps(trace)
    assert out["final"]["placed"] == placed and out["final"]["routes"] == routes


def test_viewer_reports_malformed_traces() -> None:
    out = run_viewer({"schema": "redc.pnr.trace.v1", "design": {"components": [], "instances": [{"id": 1, "component": "missing"}], "nets": []}, "events": []})
    assert "unknown component" in out["error"]
    out = run_viewer({"schema": "something.else"})
    assert "expected" in out["error"]
