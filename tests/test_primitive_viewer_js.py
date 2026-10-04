"""Execute the real primitive viewer script against real primitive traces, headlessly.

Optional: runs only when the ``quickjs`` package is available, e.g.

    uv run --with quickjs pytest tests/test_primitive_viewer_js.py

The module's Three.js imports are replaced by stubs that keep the parts of the
API the viewer depends on honest -- InstancedMesh capacity bounds on
``setMatrixAt`` / ``setColorAt``, never-written or uncoloured instances,
missing ``needsUpdate`` flags, stale cached bounding spheres, disposed
resources, a raycaster answering ``{object, instanceId}`` without looking at
visibility (like three.js) -- and the DOM stub is built from the real HTML
page, so an element id the script uses but the page lacks fails loudly.
Replay, every seek path, checkpoints, selection, the inspector, highlights,
layer toggles, picking and playback all execute the shipped JavaScript.
"""

from __future__ import annotations

import dataclasses
import json
import random
import re
from functools import cache
from typing import Any

import pytest

from redc import compile_source
from redc.physical_primitive import PadInterfacePolicy, synthesize_to_primitives
from redc.physical_primitive.geometry import Direction, Orientation
from redc.physical_primitive.pnr import (
    PrimitivePnRConfig,
    PrimitiveTraceRecorder,
    place_and_route_primitive,
)
from redc.physical_primitive.techmap import map_primitives_to_minecraft
from redc.viewer.primitive import PRIMITIVE_SCRIPT, PRIMITIVE_TEMPLATE

quickjs = pytest.importorskip("quickjs")

HALF_ADDER = "uint2 main(bool a, bool b) { return (uint2)a + (uint2)b; }"
ADD4 = "uint4 main(uint4 a, uint4 b) { return a + b; }"
# Sequential: register bits, the global clock / reset, constants and all four gate kinds.
COUNTER = "uint1 main(uint1 n) { uint1 x = 0; for (uint1 i = 0; i < n; i++) { x = x + 1; } return x; }"


@cache
def _trace_text(source: str, level: str, config: tuple[tuple[str, Any], ...]) -> str:
    graph = compile_source(source, top="main")
    recorder = PrimitiveTraceRecorder(level)
    recorder.begin_source(path="t.redc", top="main")
    netlist = synthesize_to_primitives(graph, interface=PadInterfacePolicy(), trace=recorder)
    mapped = map_primitives_to_minecraft(netlist, trace=recorder)
    place_and_route_primitive(mapped, PrimitivePnRConfig(**dict(config)), trace=recorder)
    return json.dumps(recorder.to_dict())


def trace_for(source: str, level: str = "basic", **config: Any) -> dict:
    return json.loads(_trace_text(source, level, tuple(sorted(config.items()))))


# -- the page as the DOM stub sees it ------------------------------------------------


def page_spec() -> dict:
    """Every element with an id in primitive_viewer.html (tag, type, checked,
    select options), the layer checkboxes and the camera buttons."""
    page = PRIMITIVE_TEMPLATE.read_text(encoding="utf-8")
    body = page[page.index("<body>"):]
    elements = []
    for match in re.finditer(r"<(\w+)((?:\s[^>]*)?)>", body):
        tag, attrs = match.group(1), match.group(2)
        found = re.search(r'\bid="([^"]+)"', attrs)
        if not found:
            continue
        value = re.search(r'\bvalue="([^"]*)"', attrs)
        kind = re.search(r'\btype="([^"]*)"', attrs)
        element = {
            "tag": tag, "id": found.group(1), "type": kind.group(1) if kind else "",
            "checked": bool(re.search(r"\bchecked\b", attrs)), "value": value.group(1) if value else None,
            "hidden": bool(re.search(r"\bhidden\b", attrs)),
        }
        if tag == "select":
            inner = body[match.end():body.index("</select>", match.end())]
            element["options"] = [
                {"value": o.group(1), "selected": bool(o.group(2)), "text": o.group(3)}
                for o in re.finditer(r'<option value="([^"]*)"( selected)?>([^<]*)</option>', inner)
            ]
        elements.append(element)
    layers = [[m.group(1), bool(m.group(2))] for m in re.finditer(r'data-layer="([^"]+)"( checked)?', body)]
    views = re.findall(r'data-view="([^"]+)"', body)
    return {"elements": elements, "layers": layers, "views": views}


STUBS = r"""
var SPEC = __SPEC__;
var rafs = [];
var frames = 0;

// -- DOM ----------------------------------------------------------------------
function makeElement(tag) {
  var el = {
    tagName: (tag || 'div').toUpperCase(), children: [], style: {}, dataset: {}, id: '',
    className: '', hidden: false, checked: false, type: '', title: '', max: '', min: '',
    clientWidth: 800, clientHeight: 600, parentNode: null, files: [], _l: {}, _text: '', _inner: '', _value: '',
    appendChild: function (c) {
      if (!c || typeof c !== 'object') throw new Error('appendChild: not a node');
      if (c.parentNode) c.parentNode.removeChild(c);
      this.children.push(c);
      c.parentNode = this;
      // A select shows its first option until a matching value is chosen.
      if (this.tagName === 'SELECT' && c.tagName === 'OPTION' && !this._options().some(function (o) { return o.value === this._value; }, this)) this._value = c.value;
      return c;
    },
    removeChild: function (c) {
      var i = this.children.indexOf(c);
      if (i < 0) throw new Error('removeChild: not a child');
      this.children.splice(i, 1);
      c.parentNode = null;
      return c;
    },
    _options: function () { return this.children.filter(function (c) { return c.tagName === 'OPTION'; }); },
    addEventListener: function (t, f) { (this._l[t] = this._l[t] || []).push(f); },
    removeEventListener: function (t, f) { var l = this._l[t] || []; var i = l.indexOf(f); if (i >= 0) l.splice(i, 1); },
    getBoundingClientRect: function () { return { left: 0, top: 0, width: 800, height: 600 }; },
  };
  Object.defineProperty(el, 'innerHTML', {
    get: function () { return this._inner; },
    set: function (v) {
      this.children.forEach(function (c) { c.parentNode = null; });
      this.children = []; this._text = ''; this._inner = String(v);
      if (this.tagName === 'SELECT') this._value = '';
    },
  });
  Object.defineProperty(el, 'textContent', {
    get: function () { return this._text + this.children.map(function (c) { return c.textContent; }).join(''); },
    set: function (v) {
      this.children.forEach(function (c) { c.parentNode = null; });
      this.children = []; this._text = String(v);
    },
  });
  Object.defineProperty(el, 'value', {
    get: function () { return this._value; },
    set: function (v) {
      v = String(v);
      // Like the DOM: a select only takes a value one of its options has.
      if (this.tagName === 'SELECT' && !this._options().some(function (o) { return o.value === v; })) v = '';
      this._value = v;
    },
  });
  return el;
}

var elements = {};
SPEC.elements.forEach(function (s) {
  var el = makeElement(s.tag);
  el.id = s.id; el.type = s.type; el.checked = s.checked; el.hidden = s.hidden;
  if (s.options) {
    s.options.forEach(function (o) {
      var opt = makeElement('option'); opt.value = o.value; opt.textContent = o.text; el.appendChild(opt);
      if (o.selected) el.value = o.value;
    });
  } else if (s.value !== null) el.value = s.value;
  elements[s.id] = el;
});
var layerBoxes = SPEC.layers.map(function (l) {
  var b = makeElement('input'); b.type = 'checkbox'; b.dataset.layer = l[0]; b.checked = l[1]; return b;
});
var viewButtons = SPEC.views.map(function (n) { var b = makeElement('button'); b.dataset.view = n; return b; });
var document = {
  getElementById: function (id) {
    if (!Object.prototype.hasOwnProperty.call(elements, id)) throw new Error('the page has no element #' + id);
    return elements[id];
  },
  querySelectorAll: function (sel) {
    if (sel === '[data-layer]') return layerBoxes;
    if (sel === '[data-view]') return viewButtons;
    throw new Error('unexpected selector ' + sel);
  },
  createElement: makeElement,
  createTextNode: function (t) { return { textContent: String(t), parentNode: null }; },
  addEventListener: function () {},
};
var window = { addEventListener: function () {}, removeEventListener: function () {}, devicePixelRatio: 1 };
function requestAnimationFrame(f) { rafs.push(f); }

// -- three.js -------------------------------------------------------------------
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
  constructor() { this.children = []; this.parent = null; this.position = new Vec3(); this.quaternion = { x: 0, y: 0, z: 0, w: 1 }; this.scale = new Vec3(1, 1, 1); this.visible = true; this.userData = {}; this.name = ''; this.renderOrder = 0; }
  add(o) { if (o.parent) o.parent.remove(o); this.children.push(o); o.parent = this; return this; }
  remove(o) { var i = this.children.indexOf(o); if (i >= 0) { this.children.splice(i, 1); o.parent = null; } return this; }
  traverse(f) { f(this); this.children.slice().forEach(function (c) { c.traverse(f); }); }
}
class Geometry {
  constructor() { this.userData = {}; this.disposed = false; this.attributes = {}; }
  dispose() { this.disposed = true; }
  setAttribute(n, a) { this.attributes[n] = a; return this; }
  computeVertexNormals() { if (!this.attributes.position) throw new Error('computeVertexNormals without positions'); }
}
class BoxGeometry extends Geometry {
  constructor(x, y, z) { super(); [x, y, z].forEach(function (v) { if (!(v > 0)) throw new Error('BoxGeometry size'); }); }
}
class Float32BufferAttribute {
  constructor(a, n) { if (!a.length || a.length % n) throw new Error('attribute length'); this.array = a; this.itemSize = n; }
}
class Material {
  constructor(p) { this.userData = {}; this.disposed = false; Object.assign(this, p || {}); }
  dispose() { this.disposed = true; }
}
class Color {
  constructor(c) { this.r = 1; this.g = 1; this.b = 1; if (c !== undefined) this.setHex(c); }
  setHex(h) { this.r = ((h >> 16) & 255) / 255; this.g = ((h >> 8) & 255) / 255; this.b = (h & 255) / 255; return this; }
  setRGB(r, g, b) { this.r = r; this.g = g; this.b = b; return this; }
  setHSL(h, s, l) {
    function f(n) { var k = (n + h * 12) % 12; return l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1)); }
    this.r = f(0); this.g = f(8); this.b = f(4); return this;
  }
  copy(c) { this.r = c.r; this.g = c.g; this.b = c.b; return this; }
  clone() { return new Color().copy(this); }
  multiplyScalar(k) { this.r *= k; this.g *= k; this.b *= k; return this; }
  lerp(c, t) { this.r += (c.r - this.r) * t; this.g += (c.g - this.g) * t; this.b += (c.b - this.b) * t; return this; }
  fromArray(a, o) { o = o || 0; this.r = a[o]; this.g = a[o + 1]; this.b = a[o + 2]; return this; }
  toArray(a, o) { o = o || 0; a[o] = this.r; a[o + 1] = this.g; a[o + 2] = this.b; return a; }
}
class Matrix4 {
  constructor() { this.elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]; }
  fromArray(a, o) { o = o || 0; for (var i = 0; i < 16; i++) this.elements[i] = a[o + i]; return this; }
  toArray(a, o) { o = o || 0; for (var i = 0; i < 16; i++) a[o + i] = this.elements[i]; return a; }
}
class Mesh extends Obj3D { constructor(g, m) { super(); this.geometry = g; this.material = m; this.isMesh = true; } }
function bufferAttribute(array) {
  return { array: array, version: 0, set needsUpdate(v) { if (v) this.version++; }, setUsage: function () {} };
}
var meshesCreated = 0;
class InstancedMesh extends Mesh {
  constructor(g, m, n) {
    super(g, m);
    if (!Number.isInteger(n) || n < 1) throw new Error('InstancedMesh needs a positive integer count, got ' + n);
    this.isInstancedMesh = true;
    this.capacity = n; // three.js sizes the instance buffers once, here
    this.count = n;
    var a = new Float32Array(n * 16);
    for (var i = 0; i < n; i++) { a[i * 16] = 1; a[i * 16 + 5] = 1; a[i * 16 + 10] = 1; a[i * 16 + 15] = 1; }
    this.instanceMatrix = bufferAttribute(a);
    this.instanceColor = null;
    this.boundingSphere = null;
    this.boundingBox = null;
    this.written = new Uint8Array(n);
    this.colored = new Uint8Array(n);
    this.matrixWrites = 0;
    this.colorWrites = 0;
    this.disposed = false;
    this.rendered = null;
    meshesCreated++;
  }
  slot(i, what) { if (!Number.isInteger(i) || i < 0 || i >= this.capacity) throw new Error(what + ' index ' + i + ' outside 0..' + (this.capacity - 1)); }
  setMatrixAt(i, m) {
    this.slot(i, 'setMatrixAt');
    m.elements.forEach(function (v) { if (!isFinite(v)) throw new Error('non-finite matrix'); });
    m.toArray(this.instanceMatrix.array, i * 16);
    this.written[i] = 1;
    this.matrixWrites++;
  }
  setColorAt(i, c) {
    this.slot(i, 'setColorAt');
    if (!isFinite(c.r) || !isFinite(c.g) || !isFinite(c.b)) throw new Error('non-finite colour');
    if (!this.instanceColor) this.instanceColor = bufferAttribute(new Float32Array(this.capacity * 3));
    c.toArray(this.instanceColor.array, i * 3);
    this.colored[i] = 1;
    this.colorWrites++;
  }
  computeBoundingSphere() { this.boundingSphere = { count: this.count, writes: this.matrixWrites }; }
  dispose() { this.disposed = true; }
}
// What WebGLRenderer.render relies on for every visible InstancedMesh.
function checkInstancedMesh(m) {
  if (m.disposed) throw new Error('rendering a disposed InstancedMesh ' + m.userData.key);
  if (m.geometry.disposed || m.material.disposed) throw new Error('mesh ' + m.userData.key + ' uses a disposed geometry or material');
  if (!Number.isInteger(m.count) || m.count < 0 || m.count > m.capacity) throw new Error('count ' + m.count + ' outside 0..' + m.capacity);
  for (var i = 0; i < m.count; i++) {
    if (!m.written[i]) throw new Error(m.userData.key + ': instance ' + i + ' never written (identity box at the origin)');
    if (m.instanceColor && !m.colored[i]) throw new Error(m.userData.key + ': instance ' + i + ' has no colour (renders black)');
  }
  var b = m.boundingSphere; // three.js caches it for frustum culling and raycasting
  if (b && (b.count !== m.count || b.writes !== m.matrixWrites)) throw new Error(m.userData.key + ': stale cached bounding sphere');
  if (!b) m.computeBoundingSphere();
  var r = m.rendered;
  if (r && m.matrixWrites !== r.matrixWrites && m.instanceMatrix.version === r.matrixVersion) throw new Error(m.userData.key + ': matrices changed without needsUpdate');
  if (r && m.instanceColor && m.colorWrites !== r.colorWrites && m.instanceColor.version === r.colorVersion) throw new Error(m.userData.key + ': colours changed without needsUpdate');
  m.rendered = { matrixWrites: m.matrixWrites, colorWrites: m.colorWrites, matrixVersion: m.instanceMatrix.version, colorVersion: m.instanceColor ? m.instanceColor.version : null };
}
function renderScene(scene) {
  (function visit(o) {
    if (!o.visible) return;
    if (o.isInstancedMesh) checkInstancedMesh(o);
    else if (o.geometry && (o.geometry.disposed || (o.material && o.material.disposed))) throw new Error('rendering disposed resources');
    o.children.forEach(visit);
  })(scene);
  frames++;
}
var PICK = { index: null };
var lastHits = [];
class Raycaster {
  setFromCamera(mouse, camera) {
    if (!(Math.abs(mouse.x) <= 1 && Math.abs(mouse.y) <= 1)) throw new Error('mouse outside normalized device coordinates');
    this.ready = true;
  }
  // Like three.js: every target is tested, visible or not; instanced hits carry instanceId.
  intersectObjects(targets) {
    if (!this.ready) throw new Error('setFromCamera first');
    var hits = [];
    targets.forEach(function (t, k) {
      if (!t.isInstancedMesh) { hits.push({ object: t, distance: k }); return; }
      if (t.boundingSphere && (t.boundingSphere.count !== t.count || t.boundingSphere.writes !== t.matrixWrites)) throw new Error('raycast against a stale bounding sphere');
      if (!t.boundingSphere) t.computeBoundingSphere();
      if (t.count === 0) return;
      var i = PICK.index === null ? Math.floor(t.count / 2) : Math.min(t.count - 1, PICK.index);
      hits.push({ object: t, instanceId: i, distance: k });
    });
    lastHits = hits;
    return hits;
  }
}
var THREE = {
  Vector3: Vec3, Vector2: class { constructor(x, y) { this.x = x; this.y = y; } },
  Group: Obj3D, Scene: Obj3D, Mesh: Mesh, InstancedMesh: InstancedMesh, LineSegments: Mesh,
  Box3Helper: class extends Mesh {
    constructor(b, color) {
      super(new Geometry(), new Material({ color: color }));
      if (!(b.min instanceof Vec3) || !(b.max instanceof Vec3)) throw new Error('Box3Helper needs a Box3 of Vector3');
      if (!(b.min.x <= b.max.x && b.min.y <= b.max.y && b.min.z <= b.max.z)) throw new Error('Box3Helper: inverted box');
    }
  },
  GridHelper: class extends Mesh {
    constructor(size, divisions) {
      super(new Geometry(), new Material({}));
      if (!(size > 0) || !Number.isInteger(divisions)) throw new Error('GridHelper size / divisions');
    }
  },
  AmbientLight: Obj3D, DirectionalLight: Obj3D,
  PerspectiveCamera: class extends Obj3D {
    constructor(fov, aspect, near, far) { super(); this.fov = fov; this.aspect = aspect; this.near = near; this.far = far; }
    updateProjectionMatrix() { if (!(this.near > 0 && this.far > this.near)) throw new Error('camera near/far'); } },
  WebGLRenderer: class {
    constructor() { this.domElement = makeElement('canvas'); this.disposed = false; }
    setPixelRatio() {} setSize() {} dispose() { this.disposed = true; }
    render(scene) { if (this.disposed) throw new Error('render after dispose'); renderScene(scene); }
  },
  BoxGeometry: BoxGeometry, BufferGeometry: Geometry, Float32BufferAttribute: Float32BufferAttribute,
  MeshStandardMaterial: Material, MeshBasicMaterial: Material, LineBasicMaterial: Material, Color: Color,
  Matrix4: Matrix4, Box3: class { constructor(a, b) { this.min = a; this.max = b; } },
  Raycaster: Raycaster, Clock: class { getDelta() { return 0.5; } }, DynamicDrawUsage: 35048,
};
var OrbitControls = class {
  constructor() { this.target = new Vec3(); this.enableDamping = false; this._l = {}; }
  addEventListener(t, f) { (this._l[t] = this._l[t] || []).push(f); }
  update() { return false; } // the camera only moves when a test moves it
};
"""

HARNESS = r"""
function frame() { rafs[rafs.length - 1](); }
function dump(s) {
  var placed = {};
  s.placed.forEach(function (p, id) { placed[id] = [p.origin, p.orientation]; });
  var routes = {};
  s.routes.forEach(function (r, id) { routes[id] = { root: r.root, branches: r.branches, cells: r.cells }; });
  var realized = {};
  s.realized.forEach(function (r, id) {
    realized[id] = { powered: r.powered, elements: r.elements, repeaters: r.repeaters, supports: r.supports,
      clearances: r.clearances, sinks: r.sinks };
  });
  return { placed: placed, routes: routes, realized: realized };
}
// Everything a replayed state holds, for comparing two ways of reaching it.
function fullDump(v) {
  var s = v.state;
  var d = dump(s);
  d.position = v.position;
  d.status = s.status;
  d.last = s.lastEvent ? s.lastEvent.seq : null;
  d.attempt = s.attempt; d.iteration = s.iteration; d.round = s.round;
  d.partial = s.partial ? [s.partial.net, s.partial.branches.length, Boolean(s.partial.failed)] : null;
  d.search = [s.search.cells.length, s.search.blocked.length, s.search.goal, s.rejected ? s.rejected.seq : null];
  d.probe = s.probe ? [s.probe.instance, s.probe.origin, s.probe.rejected] : null;
  d.congestion = s.congestion ? (s.congestion.seq === undefined ? 'final' : s.congestion.seq) : null;
  d.failures = s.failures.length; d.violations = s.violations.length;
  d.ripped = s.ripped ? s.ripped.seq : null;
  d.bounds = s.bounds; d.searchBounds = s.searchBounds;
  return d;
}
// A compact signature of the replay state (compared with the Python reference replay).
function sig(v) {
  var s = v.state, placed = [], routes = [], realized = [];
  var byId = function (a, b) { return a[0] - b[0]; };
  s.placed.forEach(function (p, id) { placed.push([id, p.origin[0], p.origin[1], p.origin[2], p.orientation]); });
  s.routes.forEach(function (r, id) { routes.push([id, r.cells.length, r.cells[r.cells.length - 1]]); });
  s.realized.forEach(function (r, id) { realized.push([id, r.elements.length, r.repeaters.length]); });
  return [placed.sort(byId), routes.sort(byId), realized.sort(byId),
    s.partial ? [s.partial.net, s.partial.branches.length] : null,
    s.search.cells.length, s.search.blocked.length,
    s.congestion ? s.congestion.seq : null, s.probe ? [s.probe.instance, s.probe.rejected !== null] : null,
    s.attempt, s.failures.map(function (f) { return [f.kind, f.net]; })];
}
function signatures() {
  var v = viewer, out = [];
  v.seek(0);
  out.push(sig(v));
  for (var i = 0; i < v.events.length; i++) { v.seek(i + 1); out.push(sig(v)); }
  return out;
}
// Every drawn instance of a layer: [key, id, x, y, z, m0, m2, m5, m8, m10, r, g, b].
function drawn(layer) {
  var out = [];
  viewer.layer[layer].group.children.forEach(function (m) {
    if (!m.isInstancedMesh || !m.visible) return;
    for (var i = 0; i < m.count; i++) {
      var a = m.instanceMatrix.array, o = i * 16, c = m.instanceColor ? m.instanceColor.array : null;
      out.push([m.userData.key, m.userData.ids[i], a[o + 12], a[o + 13], a[o + 14], a[o], a[o + 2], a[o + 5], a[o + 8], a[o + 10],
        c ? c[i * 3] : 1, c ? c[i * 3 + 1] : 1, c ? c[i * 3 + 2] : 1]);
    }
  });
  return out;
}
function findButton(root, prefix) {
  if (root.tagName === 'BUTTON' && root.textContent.indexOf(prefix) === 0) return root;
  for (var i = 0; i < (root.children || []).length; i++) {
    var b = findButton(root.children[i], prefix);
    if (b) return b;
  }
  return null;
}
function setLayer(name, on) {
  layerBoxes.forEach(function (b) { if (b.dataset.layer === name) { b.checked = on; b.onchange(); } });
}
function key(k, extra) {
  var ev = { key: k, target: {}, preventDefault: function () {} };
  Object.assign(ev, extra || {});
  viewer.onKey(ev);
}
"""


def js(script: str, **values: Any) -> str:
    """``script`` with every ``__NAME__`` replaced by ``values["name"]`` as JSON."""
    for name, value in values.items():
        script = script.replace(f"__{name.upper()}__", json.dumps(value))
    assert not re.search(r"__[A-Z]+__", script), script
    return script


def run(trace: dict, script: str, constants: dict[str, int] | None = None) -> Any:
    """Boot the viewer on ``trace`` in a fresh QuickJS context, then evaluate
    ``script`` (a function body returning a JSON-serializable value).
    ``constants`` overrides module-level ``const NAME = <number>;`` limits."""
    source = PRIMITIVE_SCRIPT.read_text(encoding="utf-8")
    for name, value in (constants or {}).items():
        source, found = re.subn(rf"^const {name} = \d+;", f"const {name} = {value};", source, flags=re.MULTILINE)
        assert found == 1, name
    body = "\n".join(line for line in source.splitlines() if not line.startswith("import "))
    body = body.replace("export function", "function")
    context = quickjs.Context()
    context.eval(STUBS.replace("__SPEC__", json.dumps(page_spec())))
    context.eval("document.getElementById('redc-trace').textContent = " + json.dumps(json.dumps(trace)) + ";")
    context.eval(body)
    context.eval(HARNESS)
    errors = context.eval("elements.error.hidden ? '' : elements.error.textContent")
    if errors:
        return {"error": errors}
    return json.loads(context.eval("JSON.stringify((function () {" + script + "})())"))


# -- expectations computed in Python ----------------------------------------------------


def final_maps(trace: dict) -> dict:
    final = trace["final"]
    placed = {str(p["instance"]): [p["origin"], p["orientation"]] for p in final.get("placement", []) if p["origin"]}
    routes = {str(r["net"]): {"root": r["root"], "branches": r["branches"], "cells": r["cells"]} for r in final.get("routes", [])}
    keys = ("powered", "elements", "repeaters", "supports", "clearances", "sinks")
    realized = {str(r["net"]): {k: r[k] for k in keys} for r in final.get("realized", [])}
    return {"placed": placed, "routes": routes, "realized": realized}


def absolute_voxels(trace: dict, placement: dict[str, list]) -> list[list]:
    """[instance, x, y, z] of every cell voxel, rotated by the backend's own Orientation."""
    cells = {c["name"]: c for c in trace["design"]["cells"]}
    out = []
    for inst, (origin, orientation) in placement.items():
        cell = cells[trace["design"]["instances"][int(inst)]["cell"]]
        turn = Orientation.parse(orientation)
        for voxel in cell["voxels"]:
            x, y, z = turn.apply(tuple(voxel["coord"]))
            out.append([int(inst), origin[0] + x, origin[1] + y, origin[2] + z])
    return sorted(out)


def pin_tab(row: list) -> list:
    """[instance, x, y, z, fx, fz] of a drawn pin tab: a box half a block long
    along the pin facing, its centre 0.22 blocks from the pin block centre."""

    def axis(centre: float, size: float) -> tuple[int, int]:
        if abs(size - 0.5) > 1e-6:
            return round(centre - 0.5), 0
        facing = 1 if (centre - 0.5) % 1 < 0.5 else -1
        return round(centre - 0.5 - 0.22 * facing), facing

    x, fx = axis(row[2], row[5])
    z, fz = axis(row[4], row[9])
    return [row[1], x, int(row[3]), z, fx, fz]


def absolute_pins(trace: dict, placement: dict[str, list]) -> list[list]:
    """[instance, x, y, z, fx, fz] of every pin block and its rotated facing."""
    cells = {c["name"]: c for c in trace["design"]["cells"]}
    out = []
    for inst, (origin, orientation) in placement.items():
        cell = cells[trace["design"]["instances"][int(inst)]["cell"]]
        turn = Orientation.parse(orientation)
        for pin in cell["pins"]:
            x, y, z = turn.apply(tuple(pin["position"]))
            fx, _, fz = turn.direction(Direction.parse(pin["facing"])).vector
            out.append([int(inst), origin[0] + x, origin[1] + y, origin[2] + z, fx, fz])
    return sorted(out)


def failing_config(kind: str) -> dict:
    """Config knobs that make both of two P&R attempts fail: ``unroutable``
    (every branch search runs out of expansions) or ``congestion`` (no
    negotiation iterations, so conflicts remain)."""
    if kind == "congestion":
        return {"max_routing_iterations": 0, "max_pnr_attempts": 2}
    config = {"max_astar_expansions": 1, "max_pnr_attempts": 2}
    if "expansions_per_block" in {f.name for f in dataclasses.fields(PrimitivePnRConfig)}:
        config["expansions_per_block"] = 0  # else the budget grows with the branch distance
    return config


# -- replay: every seek path --------------------------------------------------


SEEK_PATHS = r"""
  var v = viewer, n = v.events.length, out = {};
  frame();
  out.initial = dump(v.state);
  [0, n, 3, n - 1, Math.floor(n / 2), 1, n].forEach(function (t) { v.seek(t); frame(); });
  out.jumps = dump(v.state);
  v.seek(0);
  for (var i = 0; i < n; i++) { v.seek(i + 1); if (i % 97 === 0) frame(); }
  frame();
  out.stepped = dump(v.state);
  for (var k = 0; k < 25 && v.position > 0; k++) elements['btn-back'].onclick();
  for (var k2 = 0; k2 < 25; k2++) key('ArrowRight');
  out.rewound = dump(v.state);
  __RANDOM__.forEach(function (t) { v.seek(t); });
  v.seek(n); frame();
  out.random = dump(v.state);
  elements.speed.value = '25000'; elements.speed.onchange();
  elements['btn-start'].onclick();
  elements['btn-play'].onclick();
  for (var f = 0; f < 10000 && v.playing; f++) frame();
  out.played = dump(v.state);
  out.playing = v.playing;
  out.status = elements.status.textContent;
  out.frames = frames;
  return out;
"""


def seek_script(trace: dict) -> str:
    n = len(trace["events"])
    rng = random.Random(n)
    return SEEK_PATHS.replace("__RANDOM__", json.dumps([rng.randint(0, n) for _ in range(30)]))


REPLAYS = {
    "half-adder basic": (HALF_ADDER, "basic", {}),
    "half-adder detailed + keyframes": (HALF_ADDER, "detailed", {"keyframe_interval": 2}),
    "half-adder search + keyframes": (HALF_ADDER, "search", {"keyframe_interval": 1}),
    "add4 detailed": (ADD4, "detailed", {}),
    "counter basic (sequential)": (COUNTER, "basic", {}),
}


@pytest.mark.parametrize("name", list(REPLAYS))
def test_every_seek_path_ends_in_the_final_state(name: str) -> None:
    source, level, config = REPLAYS[name]
    trace = trace_for(source, level, **config)
    assert trace["final"]["success"] and trace["events"]
    out = run(trace, seek_script(trace))
    assert "error" not in out, out
    expected = final_maps(trace)
    for path in ("initial", "jumps", "stepped", "rewound", "random", "played"):
        for part in ("placed", "routes", "realized"):
            assert out[path][part] == expected[part], (path, part)
    assert out["playing"] is False
    assert "P&R succeeded" in out["status"]


def test_checkpoints_reproduce_a_plain_replay() -> None:
    trace = trace_for(HALF_ADDER, "search", keyframe_interval=1)
    n = len(trace["events"])
    sample = sorted(set(random.Random(7).sample(range(n + 1), 60)) | set(range(120)) | {n - 1, n})
    out = run(trace, js("""
      var v = viewer, sample = __SAMPLE__, plain = {}, fast = {};
      v.checkpoints = new Map(); v.checkpointList = [];
      v.takeCheckpoint = function () {};
      sample.forEach(function (t) { v.seek(0); v.seek(t); plain[t] = fullDump(v); });
      delete v.takeCheckpoint;
      v.checkpointInterval = 37; v.checkpoints = new Map(); v.checkpointList = [];
      v.seek(0); v.seek(v.events.length);
      var made = v.checkpointList.length;
      sample.slice().reverse().forEach(function (t) { v.seek(t); frame(); fast[t] = fullDump(v); });
      sample.forEach(function (t) { v.seek(t); fast['f' + t] = fullDump(v); });
      return { plain: plain, fast: fast, made: made };
    """, sample=sample))
    assert "error" not in out, out
    assert out["made"] > n // 37  # interval checkpoints plus one per keyframe
    for t in sample:
        assert out["fast"][str(t)] == out["plain"][str(t)], t
        assert out["fast"]["f" + str(t)] == out["plain"][str(t)], t


# -- replay: every event position against a Python reference ------------------


MAX_SEARCH_CELLS = 40000  # primitive_viewer.js caps the drawn frontier at these
MAX_BLOCKED_MARKS = 5000


def reference_signatures(trace: dict, max_cells: int = MAX_SEARCH_CELLS) -> list:
    """The replay state after every prefix of the events, following the
    "Replaying" rules of docs/physical-primitive-trace.md, as ``sig()`` in the
    harness summarizes it.  (The probe is the viewer's own addition: set by
    place attempts / rejections, cleared once the instance is placed.)"""
    placed: dict = {}
    routes: dict = {}
    realized: dict = {}
    partial = None
    frontier = blocked = 0
    congestion = probe = attempt = None
    failures: list = []  # the viewer's failure markers: [kind, net]

    def signature() -> list:
        return [
            [[i, *o, d] for i, (o, d) in sorted(placed.items())],
            [[n, len(r["cells"]), r["cells"][-1]] for n, r in sorted(routes.items())],
            [[n, len(r["elements"]), len(r["repeaters"])] for n, r in sorted(realized.items())],
            partial, frontier, blocked, congestion, probe, attempt, failures,
        ]

    out = [signature()]
    for e in trace["events"]:
        kind = e["type"]
        if kind == "pnr_attempt_begin":  # resets the design
            placed, routes, realized = {}, {}, {}
            partial, frontier, blocked, congestion, probe = None, 0, 0, None, None
            attempt, failures = e["attempt"], []
        elif kind == "component_place_attempt":
            probe = [e["instance"], False]
        elif kind == "component_place_rejected":
            probe = [e["instance"], True]
        elif kind == "component_placed":
            placed[e["instance"]] = (e["origin"], e["orientation"])
            probe = None
        elif kind == "net_route_begin":
            partial = [e["net"], 0]
        elif kind == "branch_route_begin":
            frontier = blocked = 0
        elif kind == "route_search_expand":  # drawn in chunks: a full frontier starts afresh
            frontier = frontier % max_cells + 1
        elif kind == "route_transition_blocked":
            blocked = min(blocked + 1, MAX_BLOCKED_MARKS)
        elif kind == "branch_route_found":
            partial = [e["net"], partial[1] + 1 if partial and partial[0] == e["net"] else 1]
            frontier = blocked = 0
        elif kind == "branch_route_failed":
            frontier = blocked = 0
            failures = [*failures, ["routing", e["net"]]]
        elif kind == "net_route_restarted":  # the tree is regrown; that failure was transient
            partial = [e["net"], 0] if partial and partial[0] == e["net"] else partial
            failures = [f for f in failures if f != ["routing", e["net"]]]
        elif kind == "legalization_failed":
            failures = [*failures, ["legalization", e["net"]]]
        elif kind == "net_route_committed":
            routes[e["net"]] = e
            partial = None
            frontier = blocked = 0
            failures = [f for f in failures if f != ["routing", e["net"]]]
        elif kind == "net_rip_up":  # also removes the realized route
            routes.pop(e["net"], None)
            realized.pop(e["net"], None)
        elif kind == "congestion_snapshot":
            congestion = e["seq"]
        elif kind == "keyframe":  # replaces all routes
            routes = {r["net"]: r for r in e["routes"]}
            partial = None
        elif kind == "route_realized":  # a realized net's legalization failure is superseded
            realized[e["net"]] = e
            failures = [f for f in failures if f != ["legalization", e["net"]]]
        elif kind == "clock_balanced":  # the balanced clock route replaces the legalized one
            realized[e["net"]] = e["realized"]
        out.append(signature())
    return out


def renumber(events: list) -> list:
    for seq, event in enumerate(events):
        event["seq"] = seq
    return events


def with_late_rip_up(trace: dict) -> dict:
    """A net ripped up AFTER legalization realized it, then rerouted and
    realized again (what a legalization repair round does)."""
    events = trace["events"]
    net = 0
    commit = max(k for k, e in enumerate(events) if e["type"] == "net_route_committed" and e["net"] == net)
    real = max(k for k, e in enumerate(events) if e["type"] == "route_realized" and e["net"] == net)
    done = next(k for k, e in enumerate(events) if e["type"] == "legalization_complete")
    rip = {"phase": "routing", "type": "net_rip_up", "net": net, "iteration": 99, "reason": "legalization",
           "cells": events[commit]["cells"], "length": events[commit]["length"]}
    again = [rip, dict(events[commit], iteration=99), dict(events[real], round=1)]
    trace["events"] = renumber(events[: done + 1] + json.loads(json.dumps(again)) + events[done + 1:])
    return trace


def with_keyframe_only_routes(trace: dict) -> dict:
    """Every commit and rip-up before the last keyframe dropped: the keyframe
    alone must bring in (and replace) the routes."""
    events = trace["events"]
    last = max(k for k, e in enumerate(events) if e["type"] == "keyframe")
    kept = [e for k, e in enumerate(events) if k > last or e["type"] not in ("net_route_committed", "net_rip_up")]
    trace["events"] = renumber(kept)
    return trace


def with_unknown_events(trace: dict) -> dict:
    """Forward compatibility: unknown event types and fields are ignored."""
    events = []
    for k, e in enumerate(trace["events"]):
        e = dict(e, future_field={"x": [1, 2]})
        events.append(e)
        if k % 7 == 0:
            events.append({"phase": "routing", "type": "future_event", "net": 0, "coord": [0, 0, 0]})
    trace["events"] = renumber(events)
    return trace


def with_restarted_net(trace: dict, restart: bool = True) -> dict:
    """A fan-out net whose second branch fails, so the router restarts it with
    that sink first -- without a new net_route_begin -- before committing
    (``restart=False``: the net is simply grown on and committed)."""
    events = trace["events"]
    begin = next(k for k, e in enumerate(events) if e["type"] == "net_route_begin" and e["fanout"] >= 2)
    net = events[begin]["net"]
    commit = next(k for k in range(begin, len(events)) if events[k]["type"] == "net_route_committed")
    body = [e for e in events[begin + 1:commit] if e.get("net") == net]
    first = next(k for k, e in enumerate(body) if e["type"] == "branch_route_found")
    second = next(e for e in body[first + 1:] if e["type"] == "branch_route_begin")
    failed = {"phase": "routing", "type": "branch_route_failed", "net": net, "iteration": second["iteration"],
              "sink": second["sink"], "goal": second["goal"], "reason": "unreachable", "expansions": 7,
              "message": f"net {net} is unroutable"}
    restart_event = {"phase": "routing", "type": "net_route_restarted", "net": net, "iteration": second["iteration"],
                     "first_sink": second["sink"], "restart": 1}
    retry = body[: first + 1] + [second, failed] + ([restart_event] if restart else []) + body
    trace["events"] = renumber(events[: begin + 1] + json.loads(json.dumps(retry)) + events[commit:])
    return trace


def with_rejected_probe(trace: dict) -> dict:
    """A placement probe rejected once before the accepted slot."""
    events = trace["events"]
    k = next(k for k, e in enumerate(events) if e["type"] == "component_place_attempt")
    probe = events[k]
    rejected = [dict(probe, origin=[probe["origin"][0], probe["origin"][1], probe["origin"][2] - 1]),
                {"phase": "placement", "type": "component_place_rejected", "instance": probe["instance"],
                 "origin": [probe["origin"][0], probe["origin"][1], probe["origin"][2] - 1],
                 "orientation": probe["orientation"], "reason": "overlaps instance 99 at [0, 0, 0]"}]
    trace["events"] = renumber(events[:k] + json.loads(json.dumps(rejected)) + events[k:])
    return trace


def with_late_failures(trace: dict) -> dict:
    """A legalization failure and a verification violation after routing."""
    events = trace["events"]
    k = next(k for k, e in enumerate(events) if e["type"] == "legalization_begin")
    coord = trace["final"]["realized"][0]["elements"][-1]["coord"]
    extra = [
        {"phase": "legalization", "type": "legalization_failed", "net": 1, "round": 0, "reason": "signal too weak",
         "message": "net 1: signal too weak at " + str(coord), "coord": coord, "sink": None},
        {"phase": "legalization", "type": "illegal_transition", "kind": "adjacent_signals", "message": "nets 0 and 1 touch",
         "cells": [coord, [coord[0] + 1, coord[1], coord[2]]], "nets": [0, 1]},
    ]
    trace["events"] = renumber(events[: k + 1] + extra + events[k + 1:])
    return trace


REFERENCE = {
    "half-adder search + keyframes": lambda: trace_for(HALF_ADDER, "search", keyframe_interval=1),
    "frontier past the drawing cap": lambda: trace_for(HALF_ADDER, "search"),
    "add4 detailed": lambda: trace_for(ADD4, "detailed"),
    "unroutable, two attempts": lambda: trace_for(HALF_ADDER, "detailed", **failing_config("unroutable")),
    "congestion, two attempts": lambda: trace_for(HALF_ADDER, "detailed", **failing_config("congestion")),
    "rip-up after legalization": lambda: with_late_rip_up(trace_for(HALF_ADDER, "basic")),
    "routes only from a keyframe": lambda: with_keyframe_only_routes(trace_for(HALF_ADDER, "detailed", keyframe_interval=1)),
    "unknown events and fields": lambda: with_unknown_events(trace_for(HALF_ADDER, "detailed")),
    "restarted net": lambda: with_restarted_net(trace_for(HALF_ADDER, "search")),
    "failure recovered without a restart": lambda: with_restarted_net(trace_for(HALF_ADDER, "detailed"), restart=False),
    "rejected placement probe": lambda: with_rejected_probe(trace_for(HALF_ADDER, "detailed")),
    "legalization failure and violation": lambda: with_late_failures(trace_for(HALF_ADDER, "basic")),
}


@pytest.mark.parametrize("name", list(REFERENCE))
def test_every_event_position_matches_the_reference_replay(name: str) -> None:
    trace = REFERENCE[name]()
    cap = 50 if "cap" in name else MAX_SEARCH_CELLS
    out = run(trace, "var r = signatures(); viewer.seek(viewer.events.length); frame(); r.push(dump(viewer.state)); return r;",
              constants={"MAX_SEARCH_CELLS": cap})
    assert "error" not in out, out
    end = out.pop()
    expected = reference_signatures(trace, cap)
    if cap < MAX_SEARCH_CELLS:
        assert max(sig[4] for sig in expected) == cap and any(0 < sig[4] < cap for sig in expected)
    assert len(out) == len(expected) == len(trace["events"]) + 1
    for position, (got, want) in enumerate(zip(out, expected)):
        assert got == want, (position, trace["events"][position - 1]["type"] if position else None)
    final = final_maps(trace)
    for part in ("placed", "routes", "realized"):
        assert end[part] == final[part], part


def test_probes_restarts_and_late_failures_are_reported() -> None:
    trace = with_rejected_probe(trace_for(HALF_ADDER, "detailed"))
    at = next(k for k, e in enumerate(trace["events"]) if e["type"] == "component_place_rejected") + 1
    out = run(trace, js("viewer.seek(__AT__); frame(); return [elements.status.textContent, drawn('probe').length];", at=at))
    assert "error" not in out, out
    assert "REJECTED: overlaps instance 99" in out[0] and out[1] == 3  # the input pad cell has three voxels

    trace = with_restarted_net(trace_for(HALF_ADDER, "search"))
    at = next(k for k, e in enumerate(trace["events"]) if e["type"] == "net_route_restarted")
    out = run(trace, js("""
      var v = viewer; v.seek(__AT__); frame();
      var before = [v.state.failures.length, drawn('congestion').length, v.state.partial.branches.length];
      v.seek(__AT__ + 1); frame();
      return [before, [v.state.failures.length, drawn('congestion').length, v.state.partial.branches.length], elements.status.textContent];
    """, at=at))
    assert "error" not in out, out
    assert out[0] == [1, 1, 1] and out[1] == [0, 0, 0] and "restarted" in out[2]

    trace = with_late_failures(trace_for(HALF_ADDER, "basic"))
    at = next(k for k, e in enumerate(trace["events"]) if e["type"] == "illegal_transition") + 1
    out = run(trace, js("""
      viewer.seek(__AT__); frame();
      return [elements.status.textContent, drawn('congestion').map(function (r) { return r[0]; })];
    """, at=at))
    assert "error" not in out, out
    assert "1 failure(s)" in out[0] and "1 violation(s)" in out[0] and "nets 0 and 1 touch" in out[0]
    assert out[1].count("marker") == 3  # one legalization failure box, two violation boxes


def test_the_event_panel_previews_large_events() -> None:
    trace = trace_for(ADD4, "basic", keyframe_interval=1)
    at = next(k for k, e in enumerate(trace["events"]) if e["type"] == "keyframe") + 1
    assert len(trace["events"][at - 1]["routes"]) > 12
    out = run(trace, js("viewer.seek(__AT__); return elements.event.textContent;", at=at))
    assert '"type": "keyframe"' in out and "more" in out and len(out) <= 6004
    assert re.search(r"\[\s*-?\d+,\s*-?\d+,\s*-?\d+\s*\]", out)  # coordinates stay readable


def test_attempt_begin_resets_the_design_when_replayed_forward() -> None:
    trace = trace_for(HALF_ADDER, "basic", **failing_config("congestion"))
    second = [k for k, e in enumerate(trace["events"]) if e["type"] == "pnr_attempt_begin"][1]
    out = run(trace, js("""
      var v = viewer; v.seek(__AT__); var before = sig(v); v.seek(__AT__ + 1); return [before, sig(v)];
    """, at=second))
    assert "error" not in out, out
    before, after = out
    assert before[0] and before[1] and before[6] is not None  # attempt 0 placed, routed, congested
    assert after[:3] == [[], [], []] and after[6] is None and after[8] == 1


# -- what is drawn -------------------------------------------------------------


def test_cells_routes_and_repeaters_are_drawn_block_for_block() -> None:
    trace = trace_for(ADD4, "basic")
    out = run(trace, """
      setLayer('keepout', true); setLayer('supports', true); setLayer('clearances', true); frame();
      return { components: drawn('components'), pins: drawn('pins'), routes: drawn('routes'),
               keepout: drawn('keepout'), supports: drawn('supports'), clearances: drawn('clearances') };
    """)
    assert "error" not in out, out
    final = final_maps(trace)
    # Every body / base voxel of every placed cell, exactly once.
    voxels = sorted([r[1], int(r[2] - 0.5), int(r[3] - 0.5), int(r[4] - 0.5)] for r in out["components"])
    assert voxels == absolute_voxels(trace, final["placed"])
    assert {r[0].split(":")[0] for r in out["components"]} == {"body", "base"}
    # Pin tabs: one per pin, offset from the pin block along its facing.
    assert sorted(pin_tab(r) for r in out["pins"]) == absolute_pins(trace, final["placed"])
    # Dust pads on exactly the realized dust blocks, repeaters with arrows along their facing.
    dust = sorted([r[1], int(r[2] - 0.5), round(r[3] - 0.04), int(r[4] - 0.5)] for r in out["routes"] if r[0] == "dust:pad")
    want_dust = sorted([int(n), *e["coord"]] for n, route in final["realized"].items()
                       for e in route["elements"] if e["kind"] == "dust")
    assert dust == want_dust
    repeaters = sorted([r[1], int(r[2] - 0.5), int(r[3]), int(r[4] - 0.5)] for r in out["routes"] if r[0] == "rep")
    arrows = sorted([r[1], int(r[2] - 0.5), int(r[3]), int(r[4] - 0.5), round(r[5] / 0.66), round(r[6] / 0.66)]
                    for r in out["routes"] if r[0] == "arrow")
    want = [(int(n), e) for n, route in final["realized"].items() for e in route["elements"] if e["kind"] == "repeater"]
    assert want, "ADD4 has repeaters"
    assert repeaters == sorted([n, *e["coord"]] for n, e in want)
    facing = {"east": (1, 0), "south": (0, 1), "west": (-1, 0), "north": (0, -1)}
    assert arrows == sorted([n, *e["coord"], *facing[e["facing"]]] for n, e in want)
    # Every tree edge is drawn: level steps as one bar, staircases as half bar + riser + half bar.
    bars = sum(1 for r in out["routes"] if r[0] == "dust:bar")
    edges = [(e["coord"], e["parent"]) for route in final["realized"].values() for e in route["elements"] if e["parent"]]
    assert bars == sum(1 if c[1] == p[1] else 3 for c, p in edges)
    assert any(c[1] != p[1] for c, p in edges), "staircases are exercised"
    # Optional layers: keep-outs, supports and clearances block for block.
    cells = {c["name"]: c for c in trace["design"]["cells"]}
    keepout = sorted([r[1], int(r[2] - 0.5), int(r[3] - 0.5), int(r[4] - 0.5)] for r in out["keepout"])
    want_keepout = []
    for inst, (origin, orientation) in final["placed"].items():
        turn = Orientation.parse(orientation)
        for c in cells[trace["design"]["instances"][int(inst)]["cell"]]["keepout"]:
            x, y, z = turn.apply(tuple(c))
            want_keepout.append([int(inst), origin[0] + x, origin[1] + y, origin[2] + z])
    assert keepout == sorted(want_keepout)
    supports = sorted([r[1], int(r[2] - 0.5), int(r[3] - 0.5), int(r[4] - 0.5)] for r in out["supports"])
    assert supports == sorted([int(n), *c] for n, route in final["realized"].items() for c in route["supports"])
    clearances = sorted([r[1], int(r[2] - 0.5), int(r[3] - 0.5), int(r[4] - 0.5)] for r in out["clearances"])
    assert clearances == sorted([int(n), *c] for n, route in final["realized"].items() for c in route["clearances"])


def test_cells_are_rotated_exactly_like_the_backend() -> None:
    trace = trace_for(HALF_ADDER, "detailed")
    names = ["east", "south", "west", "north"]
    for event in trace["events"]:
        if event["type"] in ("component_placed", "component_place_attempt"):
            event["orientation"] = names[event["instance"] % 4]
            event.pop("voxels", None)
    for entry in trace["final"]["placement"]:
        entry["orientation"] = names[entry["instance"] % 4]
    out = run(trace, "frame(); return { components: drawn('components'), pins: drawn('pins'), state: dump(viewer.state) };")
    assert "error" not in out, out
    final = final_maps(trace)
    assert out["state"]["placed"] == final["placed"]
    assert {o for _, o in final["placed"].values()} == set(names)
    voxels = sorted([r[1], int(r[2] - 0.5), int(r[3] - 0.5), int(r[4] - 0.5)] for r in out["components"])
    assert voxels == absolute_voxels(trace, final["placed"])
    # Pin tabs point along the rotated facing: the long side and the offset agree.
    assert sorted(pin_tab(r) for r in out["pins"]) == absolute_pins(trace, final["placed"])


# -- inspector, highlights and picking -----------------------------------------


def describe(trace: dict) -> dict:
    """Python-side expectations for the inspector and highlight tests."""
    d = trace["design"]
    xor = next(i for i in d["instances"] if i["kind"] == "xor" and i["role"] == "sum_xor" and i["bit"] == 0)
    groups = {g["id"]: g for g in d["groups"]}
    path, g = [], groups[xor["group"]]
    while g is not None:
        path.append(g["name"])
        g = groups.get(g["parent"]) if g["parent"] is not None else None

    def subtree(root: int) -> set[int]:
        found, stack = set(), [root]
        while stack:
            gid = stack.pop()
            found.add(gid)
            stack.extend(x["id"] for x in d["groups"] if x["parent"] == gid)
        return found

    sub = subtree(xor["group"])
    return {
        "xor": xor,
        "path": list(reversed(path)),
        "ir_instances": sorted(i["id"] for i in d["instances"] if i["ir_node"] == xor["ir_node"]),
        "group_instances": sorted(i["id"] for i in d["instances"] if i["group"] in sub),
        "driven": sorted(n["id"] for n in d["nets"] if n["driver"]["instance"] == xor["id"]),
        "read": sorted(n["id"] for n in d["nets"] if any(s["instance"] == xor["id"] for s in n["sinks"])),
    }


def test_inspector_and_highlight_controls() -> None:
    trace = trace_for(HALF_ADDER, "basic")
    facts = describe(trace)
    xor = facts["xor"]
    out_net = next(n for n in trace["design"]["nets"] if any(p["port"] == "result" for p in n["ports"]))
    in_a = next(p for p in trace["design"]["ports"] if p["name"] == "in_a")
    out = run(trace, js("""
      var v = viewer, r = {};
      function flags(a) { var ids = []; for (var i = 0; a && i < a.length; i++) if (a[i]) ids.push(i); return ids; }
      v.select({ kind: 'instance', id: __XOR__ }); frame();
      r.before = drawn('components');
      var sel = elements.selection;
      r.instanceText = sel.textContent;
      r.selectionBoxes = viewer.layer.selection.group.children.length;
      findButton(sel, 'highlight its IR node').onclick(); frame();
      r.irInst = flags(v.highlight.inst); r.irNet = flags(v.highlight.net); r.irMode = elements['hl-mode'].value;
      r.irItem = elements['hl-item'].value; r.info = elements['hl-info'].textContent;
      r.after = drawn('components');
      v.select({ kind: 'instance', id: __XOR__ });
      findButton(elements.selection, 'highlight its group').onclick(); frame();
      r.groupInst = flags(v.highlight.inst);
      findButton(elements.selection, 'design (root)').onclick();
      r.rootInst = flags(v.highlight.inst).length;
      v.select({ kind: 'net', id: __NET__ }); frame();
      r.netText = elements.selection.textContent;
      r.netOverlay = drawn('selection').length;
      r.wiresBefore = drawn('routes');
      findButton(elements.selection, 'highlight port result').onclick(); frame();
      r.portNets = flags(v.highlight.net);
      r.wiresAfter = drawn('routes');
      elements['hl-mode'].value = 'bus'; elements['hl-mode'].onchange();
      r.busOptions = elements['hl-item'].children.length;
      elements['hl-item'].value = String(__BUS__); elements['hl-item'].onchange(); frame();
      r.busNets = flags(v.highlight.net); r.busInst = flags(v.highlight.inst);
      elements['hl-mode'].value = 'group'; elements['hl-mode'].onchange();
      r.groupOptions = elements['hl-item'].children.map(function (o) { return o.textContent; });
      elements['hl-mode'].value = 'port'; elements['hl-mode'].onchange();
      elements['hl-item'].value = 'in_a'; elements['hl-item'].onchange();
      r.portIn = flags(v.highlight.net); r.portInst = flags(v.highlight.inst);
      elements['btn-hl-clear'].onclick(); frame();
      r.cleared = v.highlight.active; r.clearedMode = elements['hl-mode'].value;
      r.status = elements.status.textContent;
      key('Escape');
      r.afterEscape = v.selection;
      return r;
    """, xor=xor["id"], net=out_net["id"], bus=in_a["ir_node"]))
    assert "error" not in out, out
    text = out["instanceText"]
    for fragment in (f"#{xor['id']}", "xor (gate)", xor["cell"] + " (placeholder)", "PLACEHOLDER 2-input XOR",
                     f"IR node {xor['ir_node']} (add uint2)", "sum_xor", "uint2", "latency", "3 tick(s)", "origin",
                     "east", "drives", "reads a", "reads b"):  # fmt: skip
        assert fragment in text, fragment
    for name in facts["path"]:
        assert name in text, name
    assert out["selectionBoxes"] == 1
    assert out["irInst"] == facts["ir_instances"] and out["irMode"] == "ir_node"
    assert out["irItem"] == str(xor["ir_node"])
    assert set(facts["driven"]) <= set(out["irNet"])
    assert "add uint2" in out["info"]
    # Non-highlighted voxels fade toward the background; highlighted ones keep their colour.
    before = {tuple(x[:5]): sum(x[10:13]) for x in out["before"]}
    after = {tuple(x[:5]): sum(x[10:13]) for x in out["after"]}
    assert before.keys() == after.keys() and len(before) > 20
    for voxel, color in after.items():
        if voxel[1] in facts["ir_instances"]:
            assert color == pytest.approx(before[voxel]), voxel
        else:
            assert color < 0.7 * before[voxel], voxel
    assert out["groupInst"] == facts["group_instances"]
    assert out["rootInst"] == len(trace["design"]["instances"])
    net_text = out["netText"]
    for fragment in (f"net{out_net['id']}", "data", "fanout", "result[", "IR node 6 bit", "(add uint2)",
                     "routed length", "repeaters", "strength", "(needs 1)", "tick(s)"):  # fmt: skip
        assert fragment in net_text, fragment
    assert out["netOverlay"] > 2
    result_nets = sorted(n["id"] for n in trace["design"]["nets"] if any(p["port"] == "result" for p in n["ports"]))
    assert out["portNets"] == result_nets
    wires_before = {tuple(x[:5]): sum(x[10:13]) for x in out["wiresBefore"] if x[0].startswith("dust")}
    wires_after = {tuple(x[:5]): sum(x[10:13]) for x in out["wiresAfter"] if x[0].startswith("dust")}
    assert wires_before.keys() == wires_after.keys() and {k[1] for k in wires_after} > set(result_nets)
    for block, color in wires_after.items():
        if block[1] in result_nets:
            assert color == pytest.approx(wires_before[block]), block
        else:
            assert color < 0.7 * wires_before[block], block
    bus_nets = sorted(n["id"] for n in trace["design"]["nets"] if any(x["ir_node"] == in_a["ir_node"] for x in n["logical"]))
    assert out["busNets"] == bus_nets and out["busOptions"] == len(trace["design"]["buses"]) + 1
    assert out["busInst"] == sorted({trace["design"]["nets"][n]["driver"]["instance"] for n in bus_nets})
    assert any("bit0 (slice)" in o for o in out["groupOptions"])
    assert out["portIn"] == bus_nets and out["portInst"] == [b["instance"] for b in in_a["bits"]]
    assert out["cleared"] is False and out["clearedMode"] == "none"
    assert out["afterEscape"] is None


def test_picking_maps_instance_ids_back_to_entities() -> None:
    trace = trace_for(HALF_ADDER, "basic")
    final = final_maps(trace)
    out = run(trace, """
      var v = viewer, r = {};
      function hit() { var h = lastHits[0]; var a = h.object.instanceMatrix.array, o = h.instanceId * 16; return [a[o + 12], a[o + 13], a[o + 14]]; }
      setLayer('pins', false); setLayer('dust', false); setLayer('repeaters', false); frame();
      PICK.index = null; v.pick({ clientX: 400, clientY: 300 }); r.instance = [v.selection, hit()];
      PICK.index = 0; v.pick({ clientX: 10, clientY: 10 }); r.instance0 = [v.selection, hit()];
      setLayer('components', false); setLayer('dust', true); frame();
      PICK.index = null; v.pick({ clientX: 400, clientY: 300 }); r.net = [v.selection, hit()];
      setLayer('dust', false); setLayer('repeaters', true); frame();
      v.pick({ clientX: 5, clientY: 5 }); r.repeater = [v.selection, hit()];
      setLayer('repeaters', false); frame();
      v.pick({ clientX: 5, clientY: 5 }); r.nothing = v.selection; r.targets = lastHits.length;
      return r;
    """)
    assert "error" not in out, out
    voxels = {(i, x, y, z) for i, x, y, z in absolute_voxels(trace, final["placed"])}
    for name in ("instance", "instance0"):
        selection, (x, y, z) = out[name]
        assert selection["kind"] == "instance"
        assert (selection["id"], int(x - 0.5), int(y - 0.5), int(z - 0.5)) in voxels
    selection, (x, y, z) = out["net"]
    assert selection["kind"] == "net"
    cells = {tuple(e["coord"]) for e in final["realized"][str(selection["id"])]["elements"]}
    near = {(int(x - 0.5 + dx), int(y), int(z - 0.5 + dz)) for dx in (-0.5, 0, 0.5) for dz in (-0.5, 0, 0.5)}
    assert near & cells
    selection, (x, y, z) = out["repeater"]
    repeaters = {tuple(e["coord"]) for e in final["realized"][str(selection["id"])]["repeaters"]}
    assert (int(x - 0.5), int(y), int(z - 0.5)) in repeaters
    assert out["nothing"] is None and out["targets"] == 0  # hidden layers are never offered to the raycaster


# -- failed and eventless traces -----------------------------------------------


@pytest.mark.parametrize("kind", ["unroutable", "congestion"])
@pytest.mark.parametrize("level", ["basic", "detailed"])
def test_a_failed_multi_attempt_trace(kind: str, level: str) -> None:
    trace = trace_for(HALF_ADDER, level, **failing_config(kind))
    final = trace["final"]
    assert final["success"] is False and final["attempts"] == 2 and final["failure"]["reason"] == kind
    out = run(trace, seek_script(trace).replace("return out;", """
      out.failures = viewer.state.failures.length;
      out.conflicts = viewer.state.congestion ? viewer.state.congestion.conflicts.length : 0;
      out.attempts = elements['attempt-select'].children.length;
      elements['attempt-select'].value = elements['attempt-select'].children[1].value;
      elements['attempt-select'].onchange();
      out.attempt0 = [viewer.position, viewer.state.attempt, viewer.state.placed.size, viewer.state.failures.length];
      elements['btn-final'].onclick(); frame();
      out.marks = drawn('congestion').map(function (r) { return r[0]; });
      out.final = dump(viewer.state);
      out.finalStatus = elements.status.textContent;
      out.meta = elements.meta.textContent;
      return out;"""))
    assert "error" not in out, out
    expected = final_maps(trace)
    for path in ("initial", "jumps", "stepped", "rewound", "random", "played", "final"):
        for part in ("placed", "routes", "realized"):
            assert out[path][part] == expected[part], (path, part)
    restarts = [k for k, e in enumerate(trace["events"]) if e["type"] == "pnr_attempt_begin"]
    assert len(restarts) == 2 and out["attempts"] == 3  # "(jump to...)" plus one option per attempt
    assert out["attempt0"] == [restarts[0] + 1, 0, 0, 0]
    if kind == "unroutable":
        assert out["failures"] >= 1 and "marker" in out["marks"]  # a wire box on the failed sink pin
    else:
        assert out["conflicts"] == len(final["conflicts"]) > 0 and "conflict" in out["marks"]
        assert "conflict(s)" in out["finalStatus"]
    assert "P&R FAILED after 2 attempt(s)" in out["finalStatus"] and "FAILED in 2 attempt(s)" in out["meta"]


def test_an_eventless_failed_trace_shows_its_remaining_conflicts() -> None:
    trace = trace_for(HALF_ADDER, "none", **failing_config("congestion"))
    assert trace["events"] == [] and trace["final"]["conflicts"]
    out = run(trace, "frame(); return { state: dump(viewer.state), marks: drawn('congestion').length, status: elements.status.textContent };")
    assert "error" not in out, out
    expected = final_maps(trace)
    for part in ("placed", "routes", "realized"):
        assert out["state"][part] == expected[part], part
    assert out["marks"] > 0 and "FAILED (routing)" in out["status"]


def test_an_eventless_trace_shows_its_final_block() -> None:
    trace = trace_for(HALF_ADDER, "none")
    assert trace["events"] == []
    out = run(trace, """
      frame();
      var r = { state: dump(viewer.state), position: elements.position.textContent, components: drawn('components').length,
                routes: drawn('routes').length };
      elements['btn-start'].onclick(); elements['btn-play'].onclick(); frame(); key('ArrowLeft'); key('End');
      r.after = dump(viewer.state);
      return r;
    """)
    assert "error" not in out, out
    expected = final_maps(trace)
    for state in ("state", "after"):
        for part in ("placed", "routes", "realized"):
            assert out[state][part] == expected[part], (state, part)
    assert "final block" in out["position"]
    assert out["components"] > 0 and out["routes"] > 0


# -- controls ------------------------------------------------------------------


def test_layer_toggles_controls_and_keys_never_throw() -> None:
    trace = trace_for(COUNTER, "basic")
    out = run(trace, """
      var v = viewer, n = v.events.length, r = {};
      layerBoxes.forEach(function (b) { b.checked = !b.checked; b.onchange(); frame(); });
      layerBoxes.forEach(function (b) { b.checked = !b.checked; b.onchange(); frame(); });
      v.kindBoxes.forEach(function (b) { b.checked = false; b.onchange(); });
      frame();
      r.hiddenKinds = drawn('components').length + drawn('pins').length;
      v.kindBoxes.forEach(function (b) { b.checked = true; b.onchange(); });
      frame();
      r.kinds = v.kindBoxes.map(function (b) { return b.dataset.kind; });
      viewButtons.forEach(function (b) { b.onclick(); frame(); });
      var snap = v.events.findIndex(function (e) { return e.type === 'congestion_snapshot' && e.cells.length; });
      v.seek(snap + 1);
      elements['congestion-mode'].value = 'history'; elements['congestion-mode'].onchange(); frame();
      r.history = drawn('congestion').length;
      elements['congestion-mode'].value = 'kind'; elements['congestion-mode'].onchange(); frame();
      r.kind = drawn('congestion').length;
      ['phase-select', 'iteration-select', 'attempt-select'].forEach(function (id) {
        var s = elements[id];
        s.children.forEach(function (o) { s.value = o.value; s.onchange(); frame(); });
      });
      r.phases = elements['phase-select'].children.map(function (o) { return o.textContent; });
      ['Home', 'PageDown', 'PageDown', 'PageUp', 'ArrowRight', 'ArrowLeft', 'f', 'Escape', 'End'].forEach(function (k) { key(k); frame(); });
      key('ArrowLeft', { shiftKey: true }); key(' '); frame(); key(' ');
      v.select({ kind: 'instance', id: v.instances.findIndex(function (i) { return i.kind === 'register_bit'; }) });
      r.register = elements.selection.textContent;
      v.select({ kind: 'net', id: v.nets.findIndex(function (x) { return x.role === 'clock'; }) });
      r.clock = elements.selection.textContent;
      elements.timeline.value = String(Math.floor(n / 3)); elements.timeline.oninput(); frame();
      r.timeline = v.position;
      v.seek(n); frame();
      r.final = dump(v.state);
      r.legend = elements.legend.textContent;
      r.meta = elements.meta.textContent;
      r.meshes = meshesCreated;
      v.dispose();
      r.disposed = v.alive;
      return r;
    """)
    assert "error" not in out, out
    assert out["hiddenKinds"] == 0
    kinds = {i["kind"] for i in trace["design"]["instances"]}
    assert set(out["kinds"]) == kinds and {"register_bit", "clock_source", "reset_source", "and", "or", "xor", "not"} <= kinds
    assert out["history"] > 0 and out["kind"] > 0
    assert any("routing" in p for p in out["phases"]) and any("legalization round 0" in p for p in out["phases"])
    assert "register_bit" in out["register"] and "init" in out["register"] and "yes" in out["register"]
    assert "clock" in out["clock"] and "#" in out["clock"]
    assert out["timeline"] == len(trace["events"]) // 3
    expected = final_maps(trace)
    for part in ("placed", "routes", "realized"):
        assert out["final"][part] == expected[part], part
    for fragment in ("AND gate", "register bit", "clock source", "repeater", "shared signal", "strength 15"):
        assert fragment in out["legend"], fragment
    for fragment in ("physical-primitive", "redc-primitive-placeholder-v1", "one-bit nets", "(basic)", "succeeded in 1 attempt(s)"):
        assert fragment in out["meta"], fragment
    assert out["disposed"] is False


# -- malformed traces ----------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda t: t.update(schema="redc.pnr.trace.v1"), "coarse"),
        (lambda t: t.update(schema="something.else"), 'expected "redc.physical-primitive.pnr.v1"'),
        (lambda t: t["design"].pop("cells"), "design.cells, design.instances and design.nets must be arrays"),
        (lambda t: t["design"]["instances"][0].update(cell="missing_cell"), 'unknown cell "missing_cell"'),
        (lambda t: t["events"][3].update(seq=99), "out of sequence"),
        (lambda t: next(e for e in t["events"] if e["type"] == "component_placed").update(origin=[0.5, 0, 0]),
         "non-integer origin"),
        (lambda t: next(e for e in t["events"] if e["type"] == "component_placed").update(instance=10**6),
         "unknown instance"),
        (lambda t: t["design"]["nets"][0]["driver"].update(instance=10**6), "unknown instance"),
        (lambda t: t.update(events={}), "events must be an array"),
    ],
)
def test_malformed_traces_are_reported(mutate, message: str) -> None:
    trace = trace_for(HALF_ADDER, "basic")
    mutate(trace)
    out = run(trace, "return {};")
    assert message in out.get("error", ""), out


def test_non_object_traces_are_reported() -> None:
    out = run([1, 2, 3], "return {};")  # type: ignore[arg-type]
    assert "not a JSON object" in out["error"]
