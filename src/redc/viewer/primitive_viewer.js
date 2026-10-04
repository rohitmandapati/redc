// RedC primitive place-and-route replay viewer for redc.physical-primitive.pnr.v1.
//
// The `physical-primitive` backend bit-blasts every value into one-bit AND /
// OR / XOR / NOT gates and one-bit register bits, uses ONE coordinate = ONE
// Minecraft block, and routes every one-bit net as a tree of redstone dust
// with repeaters inserted by electrical legalization.  Its replay trace is
// specified in docs/physical-primitive-trace.md.
//
// Everything drawn comes from the trace itself -- technology cell definitions
// (sparse voxels, pins, keep-outs), instance provenance, nets, events and the
// final block -- exactly as a future Minecraft mod would consume it.  The state
// after N events is rebuilt by applying events in order; seeking backwards
// restarts from the latest `pnr_attempt_begin` (which resets the design) or
// from one of the viewer's own replay checkpoints (taken every few thousand
// events and at every `keyframe`).
//
// Scalability: primitive designs reach 10^4..10^5 voxels and thousands of
// nets, so nothing here is one Mesh per block.  Each layer draws one
// THREE.InstancedMesh per style (all AND body voxels, all dust pads, all
// repeater arrows, ...) with per-instance colours; meshes are pooled and
// rewritten in place, and a layer is rebuilt only while it is visible and
// only when a state version it depends on changed.  Frames are rendered only
// when the scene or the camera changed.  Picking maps a raycast hit's
// instanceId back to its instance or net through per-mesh id arrays.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

window.__redcViewerStarted = true;

const SCHEMA = 'redc.physical-primitive.pnr.v1';
const COARSE_SCHEMA = 'redc.pnr.trace.v1';
const MAX_SEARCH_CELLS = 40000; // A* expansions drawn at once (then a fresh chunk starts)
const MAX_BLOCKED_MARKS = 5000; // blocked transitions drawn for one branch search
const MAX_CONGESTION_CELLS = 20000;
const MAX_MARKERS = 4000; // failure / violation markers
const MAX_CHECKPOINTS = 200; // replay snapshots kept per trace (memory bound)
const MIN_CHECKPOINT_INTERVAL = 2000; // events between replay snapshots
const BACKGROUND = 0x101418;
const DIM = 0.84; // how far non-highlighted items fade toward the background

// Component kinds in legend order: [kind, colour, label].  AND / OR / XOR use
// the three hues that stay pairwise distinct under colour-vision deficiency
// on this background; NOT is separated by lightness (a pale tint) and by its
// smaller footprint.  The other kinds also differ in shape and position
// (inputs in the first column, outputs in the last), and the legend and the
// inspector always name them, so colour never carries identity alone.
const KINDS = [
  ['and', 0x3987e5, 'AND gate'],
  ['or', 0xd95926, 'OR gate'],
  ['xor', 0x199e70, 'XOR gate'],
  ['not', 0xd6c8ff, 'NOT gate'],
  ['register_bit', 0x9085e9, 'register bit'],
  ['input_bit', 0xc98500, 'input bit'],
  ['output_bit', 0xd55181, 'output bit'],
  ['const0', 0x5b6270, 'constant 0'],
  ['const1', 0xe66767, 'constant 1'],
  ['clock_source', 0xffd166, 'clock source'],
  ['reset_source', 0xef476f, 'reset source'],
  ['peripheral', 0x2fbf71, 'peripheral'],
];
const OTHER_KIND = 0x8a96a6;
const PIN_IN = 0x6fd3ff;
const PIN_OUT = 0xffe08a;
const CLOCK_NET = 0xffd166;
const RESET_NET = 0xef476f;
const REPEATER_BODY = 0xb9c0c9;
const REPEATER_ARROW = 0xff3b3b;
const RIPPED = 0xff3b3b;
const SUPPORT = 0x8a8f98;
const CLEARANCE = 0x6fd3ff;
const KEEPOUT = 0xff6b6b;
const VIOLATION = 0xff00ff;
const FAILURE = 0xff2020;
const SEARCH = 0x3fd8ff;
const BLOCKED = 0xff4d4d;
const GOAL = 0x2fbf71;
const PROBE = 0xffe066;
const PROBE_REJECTED = 0xff4d4d;
const SELECT = 0xffffff;
// Conflict kinds, highest priority first (a block in several conflicts shows the first).
const CONFLICT_KINDS = [
  ['shared_signal', 0xff3b3b, 'two nets share a signal block'],
  ['adjacent_signals', 0xff9a3c, 'signals of two nets touch (would short)'],
  ['signal_on_support', 0xff5fd2, 'signal block on another route support'],
  ['shared_support', 0xffd166, 'two nets share a support block'],
  ['clearance_blocked', 0x6fd3ff, 'staircase clearance not air'],
];

// Dust geometry (block fractions): every dust block is a flat pad, every tree
// edge a thin bar; a staircase edge is a half bar, a riser up the face of the
// upper block's support, and another half bar -- the step stays visible.
const DUST_PAD = 0.3;
const DUST_WIDTH = 0.16;
const DUST_THICK = 0.05;
const DUST_Y = 0.03;

const CLOCKWISE = ['east', 'south', 'west', 'north'];
const FACING = { east: [1, 0, 0], south: [0, 0, 1], west: [-1, 0, 0], north: [0, 0, -1] };

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Trace loading and validation
// ---------------------------------------------------------------------------

function isCoord(v) {
  return Array.isArray(v) && v.length === 3 && v.every(Number.isInteger);
}

export function validateTrace(trace) {
  if (!trace || typeof trace !== 'object' || Array.isArray(trace)) {
    return ['the file is not a JSON object'];
  }
  const errors = [];
  if (trace.schema !== SCHEMA) {
    let message = 'schema is ' + JSON.stringify(trace.schema) + ', expected "' + SCHEMA + '"';
    if (trace.schema === COARSE_SCHEMA) {
      message += ' (this is a coarse "physical" backend trace: `redc render-pnr` writes the coarse viewer for it)';
    }
    errors.push(message);
  }
  const design = trace.design;
  if (!design || typeof design !== 'object' || !Array.isArray(design.cells) ||
      !Array.isArray(design.instances) || !Array.isArray(design.nets)) {
    errors.push('design.cells, design.instances and design.nets must be arrays');
  }
  if (!Array.isArray(trace.events)) errors.push('events must be an array');
  const f = trace.final;
  if (f !== undefined && f !== null && (typeof f !== 'object' || Array.isArray(f))) {
    errors.push('final must be an object');
  }
  if (errors.length) return errors;

  const cells = new Set();
  design.cells.forEach((c, k) => {
    if (!c || typeof c.name !== 'string') {
      errors.push('cell ' + k + ' has no name');
      return;
    }
    cells.add(c.name);
    if (!Array.isArray(c.voxels) || !c.voxels.every((v) => v && isCoord(v.coord))) {
      errors.push('cell ' + JSON.stringify(c.name) + ' has malformed voxels (each needs an integer [x, y, z] coord)');
    }
    if (c.keepout !== undefined && (!Array.isArray(c.keepout) || !c.keepout.every(isCoord))) {
      errors.push('cell ' + JSON.stringify(c.name) + ' has a malformed keepout list');
    }
    if (c.pins !== undefined && (!Array.isArray(c.pins) || !c.pins.every((p) => p && isCoord(p.position)))) {
      errors.push('cell ' + JSON.stringify(c.name) + ' has malformed pins (each needs an integer position)');
    }
  });
  const count = design.instances.length;
  for (let k = 0; k < count && errors.length < 20; k++) {
    const inst = design.instances[k];
    if (!inst || inst.id !== k) {
      errors.push('instance ids must be dense 0..N-1: entry ' + k + ' has id ' + JSON.stringify(inst ? inst.id : inst));
      break;
    }
    if (!cells.has(inst.cell)) errors.push('instance ' + k + ' uses unknown cell ' + JSON.stringify(inst.cell));
  }
  const known = (t) => t && Number.isInteger(t.instance) && t.instance >= 0 && t.instance < count;
  for (let k = 0; k < design.nets.length; k++) {
    const net = design.nets[k];
    if (!net || net.id !== k) {
      errors.push('net ids must be dense 0..M-1: entry ' + k + ' has id ' + JSON.stringify(net ? net.id : net));
      break;
    }
    if (!known(net.driver) || !Array.isArray(net.sinks) || !net.sinks.every(known)) {
      errors.push('net ' + k + ' has a malformed driver / sinks or references an unknown instance');
      break;
    }
  }
  for (let k = 0; k < trace.events.length; k++) {
    const e = trace.events[k];
    if (!e || typeof e !== 'object' || e.seq !== k || typeof e.type !== 'string' || typeof e.phase !== 'string') {
      errors.push('event ' + k + ' is malformed or out of sequence (seq must count 0, 1, 2, ...)');
      break;
    }
    if (e.type === 'component_placed' && (!known(e) || !isCoord(e.origin))) {
      errors.push('event ' + k + ' (component_placed) places an unknown instance or has a non-integer origin');
      break;
    }
  }
  return errors.slice(0, 20);
}

function parseTraceText(text, name) {
  if (name && name.endsWith('.jsonl')) {
    const trace = { events: [] };
    text.split(/\r?\n/).forEach((line) => {
      if (!line.trim()) return;
      const record = JSON.parse(line);
      const kind = record.record;
      delete record.record;
      if (kind === 'header') Object.assign(trace, record);
      else if (kind === 'event') trace.events.push(record);
      else if (kind === 'final') trace.final = record.final;
    });
    return trace;
  }
  return JSON.parse(text);
}

function showError(messages) {
  const box = $('error');
  box.innerHTML = '';
  const title = document.createElement('strong');
  title.textContent = 'Cannot display this trace';
  box.appendChild(title);
  const list = document.createElement('ul');
  messages.forEach((m) => {
    const li = document.createElement('li');
    li.textContent = m;
    list.appendChild(li);
  });
  box.appendChild(list);
  box.hidden = false;
}

function hideError() {
  $('error').hidden = true;
}

// ---------------------------------------------------------------------------
// Block geometry: orientations exactly as redc.physical_primitive.geometry
// ---------------------------------------------------------------------------

// Quarter turns of an orientation name (east = identity, south, west, north).
export function quarterTurns(orientation) {
  const q = CLOCKWISE.indexOf(orientation);
  return q < 0 ? 0 : q;
}

// Orientation.apply: each clockwise quarter turn (seen from above; x east,
// z south) maps local (x, y, z) to (-z, y, x).  `east` is the identity and
// `south` turns local +x into +z.
export function rotateLocal(c, q) {
  let x = c[0];
  let z = c[2];
  for (let i = 0; i < q; i++) {
    const t = x;
    x = -z;
    z = t;
  }
  return [x + 0, c[1], z + 0]; // "+ 0" folds -0 into 0
}

// A pin facing turned clockwise by `q` quarter turns (Direction.rotated).
export function rotateFacing(facing, q) {
  const k = CLOCKWISE.indexOf(facing);
  return k < 0 ? facing : CLOCKWISE[(k + q) % 4];
}

function coordKey(c) {
  return c[0] + ',' + c[1] + ',' + c[2];
}

// A flat arrowhead pointing along local +x over a unit footprint, unit height;
// each repeater yaws it to its facing.
function wedgeGeometry() {
  const tip = [0.5, 0];
  const left = [-0.5, -0.5];
  const right = [-0.5, 0.5];
  const lo = -0.5;
  const hi = 0.5;
  const v = [];
  const vertex = (p, y) => v.push(p[0], y, p[1]);
  const tri = (a, ya, b, yb, c, yc) => { vertex(a, ya); vertex(b, yb); vertex(c, yc); };
  tri(tip, hi, left, hi, right, hi); // top, counter-clockwise seen from above
  tri(tip, lo, right, lo, left, lo); // bottom
  [[tip, left], [left, right], [right, tip]].forEach((edge) => {
    const p = edge[0];
    const q = edge[1];
    tri(p, lo, q, lo, q, hi);
    tri(p, lo, q, hi, p, hi);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
  geometry.computeVertexNormals();
  return geometry;
}

// ---------------------------------------------------------------------------
// Instance batches: per-style accumulators reused across rebuilds
// ---------------------------------------------------------------------------

// Collects matrices (16 floats), linear RGB colours and pick ids for one
// InstancedMesh while a layer is rebuilt -- no per-instance allocation.
class Batch {
  constructor() {
    this.n = 0;
    this.cap = 0;
    this.m = new Float32Array(0);
    this.c = new Float32Array(0);
    this.ids = new Int32Array(0);
  }

  reserve() {
    if (this.n < this.cap) return;
    const cap = Math.max(64, this.cap * 2);
    const m = new Float32Array(cap * 16);
    const c = new Float32Array(cap * 3);
    const ids = new Int32Array(cap);
    m.set(this.m);
    c.set(this.c);
    ids.set(this.ids);
    this.m = m;
    this.c = c;
    this.ids = ids;
    this.cap = cap;
  }

  // An axis-aligned box centred at (x, y, z) of size (sx, sy, sz).
  box(x, y, z, sx, sy, sz, rgb, id) {
    this.yawed(x, y, z, 1, 0, sx, sy, sz, rgb, id);
  }

  // A box whose local +x points along the horizontal unit vector (fx, 0, fz):
  // a rotation about +y, written directly as matrix columns (no trigonometry).
  yawed(x, y, z, fx, fz, sx, sy, sz, rgb, id) {
    this.reserve();
    const m = this.m;
    const o = this.n * 16;
    m[o] = fx * sx; m[o + 1] = 0; m[o + 2] = fz * sx; m[o + 3] = 0;
    m[o + 4] = 0; m[o + 5] = sy; m[o + 6] = 0; m[o + 7] = 0;
    m[o + 8] = -fz * sz; m[o + 9] = 0; m[o + 10] = fx * sz; m[o + 11] = 0;
    m[o + 12] = x; m[o + 13] = y; m[o + 14] = z; m[o + 15] = 1;
    const k = this.n * 3;
    this.c[k] = rgb[0];
    this.c[k + 1] = rgb[1];
    this.c[k + 2] = rgb[2];
    this.ids[this.n] = id;
    this.n++;
  }
}

// ---------------------------------------------------------------------------
// Small DOM helpers
// ---------------------------------------------------------------------------

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function linkButton(text, onclick, title) {
  const b = element('button', 'link', text);
  b.onclick = onclick;
  if (title) b.title = title;
  return b;
}

// rows: [label, value] where value is text, a DOM node, or an array of nodes.
function table(rows) {
  const t = document.createElement('table');
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    const k = element('td', '', row[0]);
    const v = document.createElement('td');
    const value = row[1];
    if (Array.isArray(value)) value.forEach((node) => v.appendChild(node));
    else if (value && typeof value === 'object' && value.tagName) v.appendChild(value);
    else v.textContent = value === null || value === undefined || value === '' ? '-' : String(value);
    tr.appendChild(k);
    tr.appendChild(v);
    t.appendChild(tr);
  });
  return t;
}

function heading(text) {
  return element('h2', '', text);
}

function fmtCoord(c) {
  return Array.isArray(c) ? '[' + c.join(', ') + ']' : '-';
}

function hexCss(hex) {
  return '#' + hex.toString(16).padStart(6, '0');
}

function push(map, key, value) {
  const list = map.get(key);
  if (list) list.push(value);
  else map.set(key, [value]);
}

function typeName(t) {
  return t && typeof t === 'object' ? t.name : t;
}

function disposeObject(o) {
  o.traverse((x) => {
    if (x.geometry && !(x.geometry.userData && x.geometry.userData.shared)) x.geometry.dispose();
    if (x.material && !(x.material.userData && x.material.userData.shared)) {
      (Array.isArray(x.material) ? x.material : [x.material]).forEach((m) => m.dispose());
    }
  });
}

// A bounded copy of an event for the side panel: long arrays keep their first
// items (a keyframe holds every route of the design), deep nesting is elided.
function preview(value, depth) {
  if (Array.isArray(value)) {
    if (depth > 8) return '[' + value.length + ' items]';
    const head = value.slice(0, 12).map((v) => preview(v, depth + 1));
    if (value.length > 12) head.push('... ' + (value.length - 12) + ' more');
    return head;
  }
  if (value && typeof value === 'object') {
    if (depth > 8) return '{...}';
    const out = {};
    Object.keys(value).forEach((k) => { out[k] = preview(value[k], depth + 1); });
    return out;
  }
  return value;
}

const EMPTY_SEARCH = Object.freeze({ cells: Object.freeze([]), blocked: Object.freeze([]), goal: null, net: null, sink: null, h0: 0 });

function newSearch(net, sink, goal) {
  return { cells: [], blocked: [], goal: goal, net: net, sink: sink, h0: 0 };
}

const LAYERS = ['components', 'pins', 'keepout', 'routes', 'partial', 'supports', 'clearances',
  'congestion', 'search', 'probe', 'bounds', 'grid', 'signals', 'selection'];
// The state versions each layer is drawn from.
const LAYER_DEPS = {
  components: ['placed', 'highlight'],
  pins: ['placed', 'highlight', 'kinds'],
  keepout: ['placed', 'kinds'],
  routes: ['routes', 'highlight'],
  partial: ['partial'],
  supports: ['routes'],
  clearances: ['routes'],
  congestion: ['congestion', 'mode'],
  probe: ['probe'],
  bounds: ['bounds'],
  grid: ['bounds'],
  signals: ['signals'],
  selection: ['selection', 'placed', 'routes', 'partial'],
};
// Layers whose meshes answer clicks.
const PICKABLE = ['components', 'pins', 'routes', 'partial'];

// ---------------------------------------------------------------------------
// The viewer
// ---------------------------------------------------------------------------

class Viewer {
  constructor(trace) {
    this.trace = trace;
    this.events = trace.events;
    this.final = trace.final && typeof trace.final === 'object' ? trace.final : {};
    const d = trace.design;
    this.design = d;
    this.cells = new Map(d.cells.map((c) => [c.name, c]));
    this.instances = d.instances;
    this.nets = d.nets;
    this.groups = Array.isArray(d.groups) ? d.groups : [];
    this.ports = Array.isArray(d.ports) ? d.ports : [];
    this.buses = Array.isArray(d.buses) ? d.buses : [];
    this.irNodes = new Map((Array.isArray(d.ir_nodes) ? d.ir_nodes : []).map((n) => [n.id, n]));
    this.cellCache = new Map();
    this.geomCache = new WeakMap();
    this.indexDesign();
    this.indexEvents();
    this.indexTiming();
    this.layers = {};
    document.querySelectorAll('[data-layer]').forEach((box) => {
      this.layers[box.dataset.layer] = box.checked;
    });
    this.kindVisible = {};
    this.versions = {
      placed: 0, routes: 0, partial: 0, search: 0, congestion: 0, probe: 0, bounds: 0,
      highlight: 0, selection: 0, kinds: 0, mode: 0, signals: 0,
    };
    this.highlight = { active: false, inst: null, net: null, spec: null, label: '' };
    this.selection = null;
    this.congestionMode = 'kind';
    this.state = this.emptyState();
    this.position = 0;
    this.checkpoints = new Map();
    this.checkpointList = [];
    this.checkpointInterval = Math.max(MIN_CHECKPOINT_INTERVAL, Math.ceil(this.events.length / MAX_CHECKPOINTS));
    this.statsKey = null;
    this.stats = null;
    this.playing = false;
    this.speed = Number($('speed').value) || 20;
    this.accum = 0;
    this.alive = true;
    this.initColors();
    this.initThree();
    this.initUI();
    // Open on the finished design (or the failure); Home / Play replays it.
    if (this.events.length) this.seek(this.events.length);
    else this.loadFinal();
    this.fit('iso');
  }

  // -- design indexes ----------------------------------------------------------

  indexDesign() {
    this.instByIrNode = new Map(); // IR node -> instance ids
    this.instByGroup = new Map(); // group -> instance ids
    this.kindCounts = {};
    this.instances.forEach((inst) => {
      if (inst.ir_node !== null && inst.ir_node !== undefined) push(this.instByIrNode, inst.ir_node, inst.id);
      if (inst.group !== null && inst.group !== undefined) push(this.instByGroup, inst.group, inst.id);
      this.kindCounts[inst.kind] = (this.kindCounts[inst.kind] || 0) + 1;
    });
    this.netsOfDriver = new Map(); // instance -> nets it drives
    this.netsOfSink = new Map(); // instance -> [{net, pin}] it reads
    this.netsOfBus = new Map(); // IR node -> nets carrying one of its bits
    this.netsOfPort = new Map(); // port name -> nets carrying one of its bits
    this.nets.forEach((net) => {
      push(this.netsOfDriver, net.driver.instance, net.id);
      net.sinks.forEach((s) => push(this.netsOfSink, s.instance, { net: net.id, pin: s.pin }));
      const buses = new Set((net.logical || []).map((l) => l.ir_node));
      buses.forEach((n) => push(this.netsOfBus, n, net.id));
      const ports = new Set((net.ports || []).map((p) => p.port));
      ports.forEach((p) => push(this.netsOfPort, p, net.id));
    });
    this.groupById = new Map(this.groups.map((g) => [g.id, g]));
    this.groupChildren = new Map();
    this.groups.forEach((g) => {
      if (g.parent !== null && g.parent !== undefined) push(this.groupChildren, g.parent, g.id);
    });
    // Primitives per group subtree (for the highlight list).
    this.groupSize = new Map();
    const size = (id, depth) => {
      if (depth > 10000) return 0;
      let n = (this.instByGroup.get(id) || []).length;
      (this.groupChildren.get(id) || []).forEach((c) => { n += size(c, depth + 1); });
      this.groupSize.set(id, n);
      return n;
    };
    this.groups.forEach((g) => {
      if (g.parent === null || g.parent === undefined || !this.groupById.has(g.parent)) size(g.id, 0);
    });
  }

  indexEvents() {
    this.restarts = [];
    this.iterations = [];
    this.milestones = [];
    let attempt = null;
    let search = false;
    let detailed = false;
    this.events.forEach((e, k) => {
      switch (e.type) {
        case 'pnr_attempt_begin':
          attempt = e.attempt;
          this.restarts.push(k);
          this.milestones.push({ index: k, label: 'attempt ' + e.attempt + ' begins' });
          break;
        case 'routing_iteration_begin':
          this.iterations.push({ index: k, attempt: attempt, iteration: e.iteration, nets: (e.nets || []).length });
          break;
        case 'synthesis_begin':
          this.milestones.push({ index: k, label: 'synthesis' });
          break;
        case 'techmap_complete':
          this.milestones.push({ index: k, label: 'technology mapping done' });
          break;
        case 'placement_begin':
          this.milestones.push({ index: k, label: 'attempt ' + attempt + ' · placement' });
          break;
        case 'routing_begin':
          this.milestones.push({ index: k, label: 'attempt ' + attempt + ' · routing' });
          break;
        case 'legalization_begin':
          this.milestones.push({ index: k, label: 'attempt ' + attempt + ' · legalization round ' + e.round });
          break;
        case 'design_finalized':
          this.milestones.push({ index: k, label: 'attempt ' + attempt + ' · design finalized' });
          break;
        case 'pnr_attempt_end':
          this.milestones.push({ index: k, label: 'attempt ' + e.attempt + ' · ' + e.status });
          break;
        case 'clock_tree_analyzed':
          this.milestones.push({ index: k, label: 'attempt ' + attempt + ' · clock tree' });
          break;
        case 'timing_analyzed':
          this.milestones.push({ index: k, label: 'attempt ' + attempt + ' · timing analysis' });
          break;
        case 'simulation_validated':
        case 'simulation_failed':
          this.milestones.push({ index: k, label: 'attempt ' + attempt + ' · redstone simulation' });
          break;
        case 'route_search_expand':
        case 'route_transition_blocked':
          search = true;
          break;
        case 'branch_route_begin':
        case 'branch_path_rejected':
          detailed = true;
          break;
        default:
          break;
      }
    });
    this.hasSearch = search || detailed; // anything for the search layer: frontier, goals, rejected paths
    this.hasProbe = this.events.some((e) => e.type === 'component_place_attempt');
  }

  // The timing / simulation reports of the final block, and the recorded
  // signal playback (a short redstone simulation of the final design).
  indexTiming() {
    const f = this.final;
    this.timing = f.timing && typeof f.timing === 'object' ? f.timing : null;
    this.simulation = f.simulation && typeof f.simulation === 'object' ? f.simulation : null;
    this.clockTree = f.clock_tree && typeof f.clock_tree === 'object' ? f.clock_tree : null;
    const pb = f.simulation_playback;
    this.playback = null;
    if (pb && typeof pb === 'object' && Array.isArray(pb.dust) && Array.isArray(pb.devices)) {
      const dust = pb.dust.filter((r) => Array.isArray(r) && r.length === 5);
      const devices = pb.devices.filter((r) => Array.isArray(r) && r.length === 5);
      dust.sort((a, b) => a[0] - b[0]);
      devices.sort((a, b) => a[0] - b[0]);
      this.playback = {
        start: Number(pb.start_gt) || 0, end: Number(pb.end_gt) || 0, mode: pb.mode || '?',
        initialDust: Array.isArray(pb.initial_dust) ? pb.initial_dust : [],
        initialDevices: Array.isArray(pb.initial_devices) ? pb.initial_devices : [],
        dust: dust, devices: devices, marks: Array.isArray(pb.marks) ? pb.marks : [], truncated: !!pb.truncated,
      };
    }
    this.simTick = this.playback ? this.playback.start : 0;
    this.simPlaying = false;
    this.simSpeed = 20; // game ticks per second (Minecraft's own rate)
    this.simAccum = 0;
    this.signalCache = null;
  }

  // Dust strength and device on/off at game tick `t` of the playback:
  // the initial snapshot plus every recorded change up to and including `t`.
  signalState(t) {
    const pb = this.playback;
    if (!pb) return { dust: new Map(), devices: new Map(), marks: [] };
    if (this.signalCache && this.signalCache.t === t) return this.signalCache;
    const dust = new Map();
    const devices = new Map();
    pb.initialDust.forEach((r) => dust.set(coordKey(r), r[3]));
    pb.initialDevices.forEach((r) => devices.set(coordKey(r), !!r[3]));
    for (let i = 0; i < pb.dust.length && pb.dust[i][0] <= t; i++) {
      const r = pb.dust[i];
      dust.set(coordKey([r[1], r[2], r[3]]), r[4]);
    }
    for (let i = 0; i < pb.devices.length && pb.devices[i][0] <= t; i++) {
      const r = pb.devices[i];
      devices.set(coordKey([r[1], r[2], r[3]]), !!r[4]);
    }
    const marks = pb.marks.filter((m) => m && m.t === t);
    this.signalCache = { t: t, dust: dust, devices: devices, marks: marks };
    return this.signalCache;
  }

  setSimTick(t) {
    const pb = this.playback;
    if (!pb) return;
    this.simTick = Math.max(pb.start, Math.min(pb.end, Math.round(Number(t) || 0)));
    this.bump('signals');
    this.sync();
    this.updateSimUI();
  }

  // -- colours -----------------------------------------------------------------

  initColors() {
    const c = new THREE.Color();
    const rgb = (hex) => {
      c.setHex(hex);
      return new Float32Array([c.r, c.g, c.b]);
    };
    this.rgb = rgb;
    this.bg = rgb(BACKGROUND);
    const shade = (a, k) => new Float32Array([a[0] * k, a[1] * k, a[2] * k]);
    this.fade = (a) => new Float32Array([
      a[0] + (this.bg[0] - a[0]) * DIM, a[1] + (this.bg[1] - a[1]) * DIM, a[2] + (this.bg[2] - a[2]) * DIM,
    ]);
    this.kindColors = {};
    const addKind = (kind, hex) => {
      const body = rgb(hex);
      const base = shade(body, 0.42);
      this.kindColors[kind] = { hex: hex, body: body, base: base, bodyDim: this.fade(body), baseDim: this.fade(base) };
    };
    KINDS.forEach((k) => addKind(k[0], k[1]));
    Object.keys(this.kindCounts).forEach((kind) => {
      if (!this.kindColors[kind]) addKind(kind, OTHER_KIND);
    });
    this.colors = {
      pinIn: rgb(PIN_IN), pinOut: rgb(PIN_OUT), repeater: rgb(REPEATER_BODY), arrow: rgb(REPEATER_ARROW),
      ripped: rgb(RIPPED), support: rgb(SUPPORT), clearance: rgb(CLEARANCE), keepout: rgb(KEEPOUT),
      violation: rgb(VIOLATION), failure: rgb(FAILURE), search: rgb(SEARCH), blocked: rgb(BLOCKED),
      goal: rgb(GOAL), head: rgb(0xffffff), probe: rgb(PROBE), probeRejected: rgb(PROBE_REJECTED),
      select: rgb(SELECT), driver: rgb(0x2fbf71), sink: rgb(0xff4d4d),
    };
    this.colors.pinInDim = this.fade(this.colors.pinIn);
    this.colors.pinOutDim = this.fade(this.colors.pinOut);
    this.colors.repeaterDim = this.fade(this.colors.repeater);
    this.colors.arrowDim = this.fade(this.colors.arrow);
    this.conflictColors = {};
    CONFLICT_KINDS.forEach((k) => { this.conflictColors[k[0]] = rgb(k[1]); });
    // One colour per net: golden-angle hues; the global clock and reset fixed.
    this.netRGB = new Float32Array(this.nets.length * 3);
    this.nets.forEach((net) => {
      if (net.role === 'clock') c.setHex(CLOCK_NET);
      else if (net.role === 'reset') c.setHex(RESET_NET);
      else c.setHSL((net.id * 0.618033988749895) % 1, 0.72, 0.56);
      this.netRGB[net.id * 3] = c.r;
      this.netRGB[net.id * 3 + 1] = c.g;
      this.netRGB[net.id * 3 + 2] = c.b;
    });
    this.scratch = new Float32Array(3);
  }

  // The colour of net `id` scaled by `k` (signal strength), faded if dimmed.
  netColor(id, k, dim, out) {
    const o = id * 3;
    for (let i = 0; i < 3; i++) {
      const v = this.netRGB[o + i] * k;
      out[i] = dim ? v + (this.bg[i] - v) * DIM : v;
    }
    return out;
  }

  // -- replay state ------------------------------------------------------------

  emptyState() {
    return {
      attempt: null, geometry: null, iteration: null, presentFactor: null, round: null,
      placed: new Map(), // instance -> {origin, orientation, q (quarter turns), column, block}
      probe: null, // {instance, origin, orientation, voxels, rejected}
      routes: new Map(), // net -> committed route tree (the trace record)
      realized: new Map(), // net -> realized route (the trace record)
      partial: null, // the tree being built: {net, root, branches, failed}
      ripped: null, // the route just ripped up (shown for one event)
      search: EMPTY_SEARCH, // the current A* frontier: {cells, blocked, goal, net, sink, h0}, append-only
      rejected: null, // the last branch_path_rejected event
      congestion: null, // the last congestion_snapshot
      violations: [], // illegal_transition events
      failures: [], // routing / legalization failure markers
      bounds: null, searchBounds: null,
      status: 'unplaced design', lastEvent: null,
    };
  }

  bump() {
    for (let i = 0; i < arguments.length; i++) this.versions[arguments[i]]++;
  }

  bumpAll() {
    Object.keys(this.versions).forEach((k) => { this.versions[k]++; });
  }

  // A net that routed (or was realized) after all no longer shows its earlier
  // failures of that kind.
  dropFailures(s, net, kind) {
    if (s.failures.some((f) => f.kind === kind && f.net === net)) {
      s.failures = s.failures.filter((f) => !(f.kind === kind && f.net === net));
      this.bump('congestion');
    }
  }

  dropRoutingFailures(s, net) {
    this.dropFailures(s, net, 'routing');
  }

  clearSearch(s) {
    if (s.search !== EMPTY_SEARCH || s.rejected) {
      s.search = EMPTY_SEARCH;
      s.rejected = null;
      this.bump('search');
    }
  }

  apply(e) {
    let s = this.state;
    if (s.ripped) {
      s.ripped = null;
      this.bump('routes');
    }
    s.lastEvent = e;
    switch (e.type) {
      case 'synthesis_begin':
        s.status = 'synthesis: ' + e.live_nodes + ' live IR nodes' + (e.sequential ? ' (sequential)' : '');
        break;
      case 'ir_node_synthesized':
        s.status = 'IR node ' + e.ir_node + ' (' + e.op + ' ' + e.ir_type + ') -> ' + e.primitives +
          ' primitive(s)' + (e.recipe ? ' via ' + e.recipe : '') + ', pass ' + e.synthesis_pass;
        break;
      case 'primitive_emitted':
        s.status = 'primitive ' + e.instance + ': ' + e.kind + ' (' + e.role + (e.bit !== null && e.bit !== undefined ? ' bit ' + e.bit : '') + ')';
        break;
      case 'synthesis_complete':
        s.status = 'synthesis complete: ' + e.gates + ' gates, ' + e.register_bits + ' register bits, ' + e.nets + ' one-bit nets';
        break;
      case 'primitive_mapped':
        s.status = 'primitive ' + e.instance + ' -> cell ' + e.cell;
        break;
      case 'techmap_complete':
        s.status = 'mapped ' + e.instances + ' instances onto ' + Object.keys(e.cells || {}).length +
          ' cells (' + e.component_voxels + ' voxels)';
        break;
      case 'pnr_begin':
        s.status = 'P&R of ' + e.instances + ' instances, ' + e.nets + ' nets (up to ' + e.max_attempts + ' attempts)';
        break;
      case 'pnr_attempt_begin':
        this.state = s = this.emptyState();
        s.lastEvent = e;
        s.attempt = e.attempt;
        s.geometry = e;
        s.status = 'attempt ' + e.attempt + ': spacing ' + e.component_spacing + ', channel ' +
          e.channel_width + ', margin ' + e.routing_margin + ', max y ' + e.max_y;
        this.bumpAll();
        break;
      case 'placement_begin':
        s.status = 'placing ' + (e.columns || []).reduce((n, c) => n + ((c && c.instances) || []).length, 0) +
          ' cells in ' + (e.columns || []).length + ' columns' +
          (e.blocks && e.blocks.length ? ', ' + e.blocks.length + ' blocks' : '');
        break;
      case 'component_place_attempt':
        s.probe = { instance: e.instance, origin: e.origin, orientation: e.orientation, voxels: e.voxels, rejected: null };
        this.bump('probe');
        break;
      case 'component_place_rejected': {
        const prior = s.probe;
        const same = prior && prior.instance === e.instance && coordKey(prior.origin) === coordKey(e.origin);
        s.probe = { instance: e.instance, origin: e.origin, orientation: e.orientation, voxels: same ? prior.voxels : null, rejected: e.reason };
        s.status = 'probe #' + e.instance + ' at ' + fmtCoord(e.origin) + ' rejected: ' + e.reason;
        this.bump('probe');
        break;
      }
      case 'component_placed':
        s.placed.set(e.instance, {
          origin: e.origin, orientation: e.orientation, q: quarterTurns(e.orientation), column: e.column, block: e.block,
        });
        s.probe = null;
        s.status = 'placed #' + e.instance + ' at ' + fmtCoord(e.origin) + ' (' + e.orientation + ', column ' + e.column +
          (e.block !== undefined ? ', block ' + e.block : '') + ')';
        this.bump('placed', 'probe');
        break;
      case 'design_bounds_changed':
        s.bounds = { min: e.min, max: e.max };
        this.bump('bounds');
        break;
      case 'placement_complete':
        s.status = 'placed ' + e.component_count + ' cells: ' + e.component_voxels + ' voxels, ' +
          e.keepout_voxels + ' keep-out blocks, ' + e.probes + ' probes';
        break;
      case 'placement_failed':
        s.status = 'placement FAILED: ' + e.reason;
        break;
      case 'routing_begin':
        s.searchBounds = e.search_bounds;
        s.presentFactor = e.present_factor;
        s.status = 'routing ' + e.nets + ' one-bit nets';
        this.bump('bounds');
        break;
      case 'routing_iteration_begin':
        s.iteration = e.iteration;
        s.presentFactor = e.present_factor;
        s.status = 'iteration ' + e.iteration + ': ' + (e.nets || []).length + ' net(s) to (re)route';
        break;
      case 'net_route_begin':
        s.partial = { net: e.net, root: e.driver ? e.driver.coord : null, branches: [], failed: null };
        s.status = 'routing net ' + e.net + ' (fanout ' + e.fanout + ')';
        this.bump('partial');
        break;
      case 'branch_route_begin':
        s.search = newSearch(e.net, e.sink, e.goal);
        s.rejected = null;
        this.bump('search');
        break;
      case 'route_search_expand':
        if (s.search === EMPTY_SEARCH) s.search = newSearch(e.net, e.sink, null);
        if (!s.search.h0) s.search.h0 = e.h || 1;
        // Drawing is capped: a full frontier starts a fresh chunk (same search,
        // same blocked marks) so the newest expansions are always the ones shown.
        if (s.search.cells.length >= MAX_SEARCH_CELLS) s.search = Object.assign({}, s.search, { cells: [] });
        s.search.cells.push(e);
        this.bump('search');
        break;
      case 'route_transition_blocked':
        if (s.search === EMPTY_SEARCH) s.search = newSearch(e.net, e.sink, null);
        if (s.search.blocked.length < MAX_BLOCKED_MARKS) s.search.blocked.push(e);
        this.bump('search');
        break;
      case 'branch_search_stats':
        s.status = 'net ' + e.net + ' -> #' + e.sink.instance + '.' + e.sink.pin + ': ' + e.expansions +
          ' expansions' + (e.mode && e.mode !== 'normal' ? ' (' + e.mode + ' search)' : '') + ', ' +
          (e.found ? 'found' : 'not found');
        break;
      case 'branch_search_relaxed':
        s.status = 'net ' + e.net + ' -> #' + e.sink.instance + '.' + e.sink.pin + ': budget spent after ' +
          e.expansions + ' expansions; ' + (e.mode === 'greedy'
            ? 'searching again in greedy mode (higher A* weight, congestion still priced)'
            : 'searching again ignoring congestion (last resort)');
        break;
      case 'branch_path_rejected':
        s.rejected = e;
        s.status = 'net ' + e.net + ': path rejected (' + e.reason + ' at ' + fmtCoord(e.coord) + '), searching again';
        this.bump('search');
        break;
      case 'branch_route_found': {
        const prior = s.partial && s.partial.net === e.net ? s.partial : { net: e.net, root: e.start, branches: [], failed: null };
        s.partial = {
          net: e.net, root: prior.root || e.start, failed: null,
          branches: prior.branches.concat([{ sink: e.sink, start: e.start, goal: e.goal, path: e.path }]),
        };
        this.clearSearch(s);
        this.bump('partial');
        break;
      }
      case 'branch_route_failed':
        if (s.failures.length < MAX_MARKERS) {
          s.failures.push({ kind: 'routing', net: e.net, coord: e.goal, reason: e.reason, message: e.message });
        }
        if (s.partial && s.partial.net === e.net) s.partial = Object.assign({}, s.partial, { failed: e });
        s.status = 'net ' + e.net + ' FAILED: ' + (e.message || e.reason);
        this.clearSearch(s);
        this.bump('partial', 'congestion');
        break;
      case 'net_route_restarted':
        // The net walled in its own sink: its tree is discarded and grown again
        // (no new net_route_begin follows), so the branch failure was transient.
        if (s.partial && s.partial.net === e.net) s.partial = Object.assign({}, s.partial, { branches: [], failed: null });
        this.dropRoutingFailures(s, e.net);
        s.status = 'net ' + e.net + ' restarted with #' + e.first_sink.instance + '.' + e.first_sink.pin +
          ' first (restart ' + e.restart + ')';
        this.bump('partial', 'congestion');
        break;
      case 'net_route_committed':
        s.routes.set(e.net, e);
        s.partial = null;
        this.dropRoutingFailures(s, e.net);
        s.status = 'net ' + e.net + ' committed: ' + e.length + ' blocks, ' + (e.branches || []).length + ' branch(es)';
        this.clearSearch(s);
        this.bump('routes', 'partial');
        break;
      case 'net_rip_up':
        s.routes.delete(e.net);
        s.realized.delete(e.net);
        s.ripped = e;
        s.status = 'ripped up net ' + e.net + ' (' + e.reason + ', ' + e.length + ' blocks)';
        this.bump('routes');
        break;
      case 'routing_iteration_end':
        s.status = 'iteration ' + e.iteration + ' done: ' + e.conflicts + ' conflict(s) on ' +
          e.conflict_cells + ' block(s), ' + e.routed_cells + ' signal blocks';
        break;
      case 'congestion_snapshot':
        s.congestion = e;
        this.bump('congestion');
        break;
      case 'physical_conflict':
        s.status = 'conflict ' + e.kind + ' at ' + (e.cells || []).map(fmtCoord).join(' ') + ' (nets ' + (e.nets || []).join(', ') + ')';
        break;
      case 'keyframe':
        s.routes = new Map((e.routes || []).map((r) => [r.net, r]));
        s.partial = null;
        this.bump('routes', 'partial');
        break;
      case 'routing_complete':
        s.status = 'routing converged after ' + e.iterations + ' iteration(s), ' + e.rip_ups + ' rip-up(s)';
        break;
      case 'routing_failed':
        s.status = 'routing FAILED: ' + (e.failure ? e.failure.message : '');
        break;
      case 'legalization_begin':
        s.round = e.round;
        s.status = 'legalization round ' + e.round + ': ' + e.nets + ' nets';
        break;
      case 'route_legalization_begin':
        s.status = 'legalizing net ' + e.net + ' (' + e.length + ' blocks, drive ' + e.drive + ')';
        break;
      case 'signal_strength_scan':
        s.status = 'net ' + e.net + ': signal strength scan (' + (e.powered ? 'powered' : 'never powered') + ')';
        break;
      case 'repeater_inserted':
        s.status = 'net ' + e.net + ': repeater at ' + fmtCoord(e.coord) + ' facing ' + e.facing + ' (input strength ' + e.input_strength + ')';
        break;
      case 'legalization_failed':
        if (s.failures.length < MAX_MARKERS) {
          s.failures.push({ kind: 'legalization', net: e.net, coord: e.coord, reason: e.reason, message: e.message });
        }
        s.status = 'legalization FAILED: ' + (e.message || e.reason);
        this.bump('congestion');
        break;
      case 'route_realized':
        s.realized.set(e.net, e);
        this.dropFailures(s, e.net, 'legalization');
        s.status = 'net ' + e.net + ' realized: ' + (e.elements || []).length + ' blocks, ' + (e.repeaters || []).length + ' repeater(s)';
        this.bump('routes');
        break;
      case 'legalization_complete':
        s.status = 'legalization round ' + e.round + ': ' + e.realized + ' realized, ' + e.failures + ' failed, ' + e.repeaters + ' repeaters';
        break;
      case 'illegal_transition':
        if (s.violations.length < MAX_MARKERS) s.violations.push(e);
        s.status = 'verification: ' + e.kind + ' -- ' + e.message;
        this.bump('congestion');
        break;
      case 'design_finalized':
        if (e.bounds) s.bounds = { min: e.bounds.min, max: e.bounds.max };
        s.status = 'final: ' + e.nets + ' nets, ' + e.dust + ' dust, ' + e.repeaters + ' repeaters, ' + e.supports + ' supports';
        this.bump('bounds');
        break;
      case 'clock_tree_analyzed':
        s.status = 'clock tree: ' + (e.sinks || []).length + ' register clock pins, skew ' +
          (e.skew ? e.skew.rt : '?') + ' rt before balancing';
        break;
      case 'clock_balanced':
        // The balanced clock route replaces the legalized one.
        if (e.realized) s.realized.set(e.net, e.realized);
        s.status = 'clock balanced: skew ' + e.skew_before.rt + ' -> ' + e.skew_after.rt + ' rt (' +
          e.repeaters_added + ' repeater(s) added, ' + e.repeaters_raised + ' raised, +' + e.delay_added_rt + ' rt)';
        this.bump('routes');
        break;
      case 'clock_balance_failed':
        s.status = 'clock balancing FAILED: ' + e.message;
        break;
      case 'timing_analyzed':
      case 'timing_closed':
        s.status = (e.type === 'timing_closed' ? 'timing closed: ' : 'timing analysed: ') + (e.sequential
          ? 'clock period ' + (e.period ? e.period.rt : '?') + ' rt (' + e.mode + '), skew ' + (e.skew ? e.skew.rt : '?') +
            ' rt, worst setup slack ' + (e.worst_setup_slack ? e.worst_setup_slack.gt : '-') + ' gt, worst hold slack ' +
            (e.worst_hold_slack ? e.worst_hold_slack.gt : '-') + ' gt'
          : 'combinational settle ' + (e.settle ? e.settle.rt : '?') + ' rt');
        break;
      case 'timing_failed':
        s.status = 'timing FAILED: ' + (e.failures || []).map((x) => x.code + ' ' + x.message).join('; ');
        break;
      case 'simulation_validated':
      case 'simulation_failed':
        s.status = 'redstone simulation (' + e.mode + '): ' + (e.validated ? 'agrees with the primitive simulator' : 'FAILED') +
          (e.cycles ? ', ' + e.cycles + ' logical cycles' : '') + (e.vectors ? ', ' + e.vectors + ' vectors' : '') +
          (e.sim_events !== undefined ? ', ' + e.sim_events + ' events' : '');
        break;
      case 'pnr_attempt_end':
        s.status = 'attempt ' + e.attempt + ' ' + e.status + (e.failure ? ': ' + e.failure.message : '');
        break;
      case 'pnr_end':
        s.status = e.success ? 'P&R succeeded' : 'P&R FAILED after ' + e.attempts + ' attempt(s)';
        break;
      default:
        break; // unknown event types are ignored (forward compatibility)
    }
  }

  // The final block (an eventless trace starts here).
  loadFinal() {
    const f = this.final;
    const s = this.emptyState();
    (f.placement || []).forEach((p) => {
      if (p && isCoord(p.origin)) s.placed.set(p.instance, { origin: p.origin, orientation: p.orientation, q: quarterTurns(p.orientation) });
    });
    (f.routes || []).forEach((r) => s.routes.set(r.net, r));
    (f.realized || []).forEach((r) => s.realized.set(r.net, r));
    if (f.conflicts && f.conflicts.length) s.congestion = { conflicts: f.conflicts, cells: [] };
    s.violations = (f.violations || []).slice(0, MAX_MARKERS);
    const failure = f.failure;
    if (failure && Array.isArray(failure.details)) {
      failure.details.forEach((d) => {
        const where = d && (d.sink && d.sink.coord ? d.sink.coord : d.coord);
        if (isCoord(where) && s.failures.length < MAX_MARKERS) {
          s.failures.push({ kind: failure.stage, net: d.net, coord: where, reason: d.reason, message: d.message });
        }
      });
    }
    if (f.design_bounds) s.bounds = { min: f.design_bounds.min, max: f.design_bounds.max };
    if (f.search_bounds) s.searchBounds = f.search_bounds;
    s.attempt = f.attempt === undefined ? null : f.attempt;
    s.status = f.success ? 'final design' : 'FAILED' + (failure ? ' (' + failure.stage + '): ' + failure.message : '');
    this.state = s;
    this.bumpAll();
    this.sync();
    this.updateUI();
  }

  // -- seeking -----------------------------------------------------------------

  seek(target) {
    const n = this.events.length;
    target = Math.max(0, Math.min(n, Math.round(Number(target) || 0)));
    if (n) {
      const start = this.restartPoint(target);
      if (target < this.position || start.position > this.position) {
        if (start.checkpoint) this.restoreCheckpoint(start.position);
        else {
          this.state = this.emptyState();
          this.position = start.position;
        }
        this.bumpAll();
      }
      while (this.position < target) {
        const e = this.events[this.position];
        this.apply(e);
        this.position++;
        if (this.position % this.checkpointInterval === 0 || e.type === 'keyframe') this.takeCheckpoint();
      }
    }
    this.sync();
    this.updateUI();
  }

  // Where to start replaying to reach `target`: the latest checkpoint at or
  // before it, or the latest attempt start before it (which resets the state).
  restartPoint(target) {
    let best = { position: 0, checkpoint: false };
    for (let i = 0; i < this.restarts.length && this.restarts[i] < target; i++) {
      best = { position: this.restarts[i], checkpoint: false };
    }
    const list = this.checkpointList;
    let lo = 0;
    let hi = list.length - 1;
    let found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (list[mid] <= target) { found = list[mid]; lo = mid + 1; } else hi = mid - 1;
    }
    if (found >= best.position && found > 0) best = { position: found, checkpoint: true };
    return best;
  }

  // Snapshot the replay state.  Maps are copied; route, probe and congestion
  // records are immutable and shared; the append-only search frontier is
  // captured by length.
  takeCheckpoint() {
    if (this.checkpoints.has(this.position) || this.checkpoints.size >= MAX_CHECKPOINTS * 2) return;
    const s = this.state;
    const snap = Object.assign({}, s);
    snap.placed = new Map(s.placed);
    snap.routes = new Map(s.routes);
    snap.realized = new Map(s.realized);
    snap.violations = s.violations.slice();
    snap.failures = s.failures.slice();
    snap.searchCells = s.search.cells.length;
    snap.searchBlocked = s.search.blocked.length;
    this.checkpoints.set(this.position, snap);
    const list = this.checkpointList;
    let i = list.length;
    while (i > 0 && list[i - 1] > this.position) i--;
    list.splice(i, 0, this.position);
  }

  restoreCheckpoint(position) {
    const snap = this.checkpoints.get(position);
    const s = Object.assign({}, snap);
    s.placed = new Map(snap.placed);
    s.routes = new Map(snap.routes);
    s.realized = new Map(snap.realized);
    s.violations = snap.violations.slice();
    s.failures = snap.failures.slice();
    const src = snap.search;
    s.search = src === EMPTY_SEARCH ? EMPTY_SEARCH : Object.assign({}, src, {
      cells: src.cells.slice(0, snap.searchCells), blocked: src.blocked.slice(0, snap.searchBlocked),
    });
    delete s.searchCells;
    delete s.searchBlocked;
    this.state = s;
    this.position = position;
  }

  // -- cell and route geometry ---------------------------------------------------

  // A technology cell rotated by `q` quarter turns (cached): flat voxel arrays
  // [x, y, z, isBase], keep-out [x, y, z], pins and the local bounds.
  orientedCell(name, q) {
    const key = name + '|' + q;
    let oc = this.cellCache.get(key);
    if (oc) return oc;
    const cell = this.cells.get(name);
    if (!cell) return null;
    const voxels = new Int32Array(cell.voxels.length * 4);
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    const grow = (r) => {
      for (let i = 0; i < 3; i++) {
        if (r[i] < lo[i]) lo[i] = r[i];
        if (r[i] > hi[i]) hi[i] = r[i];
      }
    };
    cell.voxels.forEach((v, k) => {
      const r = rotateLocal(v.coord, q);
      voxels[k * 4] = r[0];
      voxels[k * 4 + 1] = r[1];
      voxels[k * 4 + 2] = r[2];
      voxels[k * 4 + 3] = v.role === 'base' ? 1 : 0;
      grow(r);
    });
    const keepoutList = cell.keepout || [];
    const keepout = new Int32Array(keepoutList.length * 3);
    keepoutList.forEach((c, k) => {
      const r = rotateLocal(c, q);
      keepout.set(r, k * 3);
      grow(r);
    });
    const pins = (cell.pins || []).map((p) => {
      const r = rotateLocal(p.position, q);
      grow(r);
      const facing = rotateFacing(p.facing, q);
      const f = FACING[facing] || [0, 0, 0];
      return {
        name: p.name, direction: p.direction, position: r, facing: facing, fx: f[0], fz: f[2], strength: p.strength,
      };
    });
    if (lo[0] === Infinity) { lo.fill(0); hi.fill(0); }
    oc = { cell: cell, q: q, voxels: voxels, keepout: keepout, pins: pins, lo: lo, hi: hi };
    this.cellCache.set(key, oc);
    return oc;
  }

  // Absolute block of pin `pin` of a placed instance (or null).
  pinBlock(instance, pin) {
    const p = this.state.placed.get(instance);
    const inst = this.instances[instance];
    if (!p || !inst) return null;
    const oc = this.orientedCell(inst.cell, p.q);
    const found = oc ? oc.pins.find((x) => x.name === pin) : null;
    if (!found) return null;
    return [p.origin[0] + found.position[0], p.origin[1] + found.position[1], p.origin[2] + found.position[2]];
  }

  // Flat element arrays for one route (cached per immutable route record):
  // xyz, parent index (-1 at the root), kind (0 dust, 1 repeater), strength
  // (-1 unknown), facing index (-1 none), plus derived supports / clearances.
  routeGeometry(route, realized) {
    let g = this.geomCache.get(route);
    if (g) return g;
    const coords = [];
    const parents = [];
    const kinds = [];
    const strengths = [];
    const facings = [];
    const index = new Map();
    const pins = new Set();
    if (realized) {
      (route.elements || []).forEach((el) => {
        if (!isCoord(el.coord)) return;
        index.set(coordKey(el.coord), coords.length);
        coords.push(el.coord);
        parents.push(el.parent);
        kinds.push(el.kind === 'repeater' ? 1 : 0);
        strengths.push(typeof el.strength === 'number' ? el.strength : -1);
        facings.push(el.facing ? CLOCKWISE.indexOf(el.facing) : -1);
      });
    } else {
      // RouteTree.parent: the root, then every branch path block keeps the
      // first upstream block it was reached from.
      const add = (c, parent) => {
        if (!isCoord(c)) return;
        const k = coordKey(c);
        if (index.has(k)) return;
        index.set(k, coords.length);
        coords.push(c);
        parents.push(parent);
        kinds.push(0);
        strengths.push(-1);
        facings.push(-1);
      };
      if (route.root) {
        add(route.root, null);
        pins.add(coordKey(route.root));
      }
      (route.branches || []).forEach((b) => {
        const path = b.path || [];
        if (path.length) add(path[0], null);
        for (let i = 1; i < path.length; i++) add(path[i], path[i - 1]);
        if (b.goal) pins.add(coordKey(b.goal));
      });
    }
    const n = coords.length;
    g = {
      n: n, xyz: new Int32Array(n * 3), parent: new Int32Array(n), kind: new Uint8Array(kinds),
      strength: new Int8Array(strengths), facing: new Int8Array(facings), supports: null, clearances: null,
    };
    for (let i = 0; i < n; i++) {
      g.xyz.set(coords[i], i * 3);
      const p = parents[i];
      g.parent[i] = p && index.has(coordKey(p)) ? index.get(coordKey(p)) : -1;
    }
    if (realized) {
      g.supports = (route.supports || []).filter(isCoord);
      g.clearances = (route.clearances || []).filter(isCoord);
    } else {
      // Derived like RouteTree.supports / .clearances.
      g.supports = [];
      const clear = new Map();
      for (let i = 0; i < n; i++) {
        const c = coords[i];
        if (!pins.has(coordKey(c))) g.supports.push([c[0], c[1] - 1, c[2]]);
        const p = g.parent[i];
        if (p >= 0 && coords[p][1] !== c[1]) {
          const lower = coords[p][1] < c[1] ? coords[p] : c;
          const above = [lower[0], lower[1] + 1, lower[2]];
          clear.set(coordKey(above), above);
        }
      }
      g.clearances = Array.from(clear.values());
    }
    this.geomCache.set(route, g);
    return g;
  }

  // -- three.js ----------------------------------------------------------------

  initThree() {
    const host = $('scene');
    this.host = host;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(host.clientWidth, host.clientHeight);
    host.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BACKGROUND);
    this.camera = new THREE.PerspectiveCamera(50, host.clientWidth / Math.max(1, host.clientHeight), 0.1, 50000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    // Render on demand: only after the scene changed (sync, fit, resize) or
    // the camera moved.  Not from OrbitControls.update()'s return value: r160
    // re-clamps the target every call, so it reports a change every frame.
    this.dirty = true;
    this.viewKey = new Float64Array(10);
    this.viewNow = new Float64Array(10);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(0.6, 2, 1.2);
    this.scene.add(sun);
    const fill = new THREE.DirectionalLight(0xffffff, 0.3);
    fill.position.set(-1, 0.5, -0.8);
    this.scene.add(fill);

    this.layer = {};
    LAYERS.forEach((name) => {
      const group = new THREE.Group();
      group.name = name;
      this.scene.add(group);
      this.layer[name] = { name: name, group: group, meshes: new Map(), batches: new Map(), key: null, extras: [] };
    });
    this.initStyles();
    this.searchDrawn = { obj: null, cells: 0, blocked: 0, rejected: null };
    this.tmpMatrix = new THREE.Matrix4();
    this.tmpColor = new THREE.Color();
    this.raycaster = new THREE.Raycaster();

    this.onResize = () => {
      const w = host.clientWidth;
      const h = Math.max(1, host.clientHeight);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
      this.dirty = true;
    };
    window.addEventListener('resize', this.onResize);

    let down = null;
    this.renderer.domElement.addEventListener('pointerdown', (ev) => { down = [ev.clientX, ev.clientY]; });
    this.renderer.domElement.addEventListener('pointerup', (ev) => {
      if (down && Math.abs(ev.clientX - down[0]) + Math.abs(ev.clientY - down[1]) < 5) this.pick(ev);
      down = null;
    });

    this.clock = new THREE.Clock();
    const loop = () => {
      if (!this.alive) return;
      requestAnimationFrame(loop);
      const dt = this.clock.getDelta();
      if (this.playing) {
        this.accum += dt * this.speed;
        const steps = Math.floor(this.accum);
        if (steps > 0) {
          this.accum -= steps;
          const target = Math.min(this.events.length, this.position + steps);
          this.seek(target);
          if (target >= this.events.length) this.setPlaying(false);
        }
      }
      if (this.simPlaying && this.playback) {
        this.simAccum += dt * this.simSpeed;
        const ticks = Math.floor(this.simAccum);
        if (ticks > 0) {
          this.simAccum -= ticks;
          const next = Math.min(this.playback.end, this.simTick + ticks);
          if (next >= this.playback.end) this.simPlaying = false;
          this.setSimTick(next);
        }
      }
      this.controls.update(); // applies damping
      if (this.cameraMoved() || this.dirty) {
        this.dirty = false;
        this.renderer.render(this.scene, this.camera);
      }
    };
    loop();
  }

  // Whether the camera moved noticeably since the last check (no allocation).
  cameraMoved() {
    const p = this.camera.position;
    const q = this.camera.quaternion;
    const t = this.controls.target;
    const v = this.viewNow;
    const k = this.viewKey;
    v[0] = p.x; v[1] = p.y; v[2] = p.z; v[3] = q.x; v[4] = q.y; v[5] = q.z; v[6] = q.w;
    v[7] = t.x; v[8] = t.y; v[9] = t.z;
    for (let i = 0; i < 10; i++) {
      if (Math.abs(v[i] - k[i]) > 1e-7 * (1 + Math.abs(k[i]))) {
        k.set(v);
        return true;
      }
    }
    return false;
  }

  // Geometry + material per style family; every mesh of a family shares them
  // and carries its colours per instance.
  initStyles() {
    const box = new THREE.BoxGeometry(1, 1, 1);
    const wedge = wedgeGeometry();
    box.userData.shared = true;
    wedge.userData.shared = true;
    this.geometries = [box, wedge];
    this.materials = [];
    const make = (Kind, params) => {
      const material = new Kind(Object.assign({ color: 0xffffff }, params));
      material.userData.shared = true;
      this.materials.push(material);
      return material;
    };
    const lit = (params) => make(THREE.MeshStandardMaterial, Object.assign({ roughness: 0.8, metalness: 0 }, params));
    const flat = (params) => make(THREE.MeshBasicMaterial, params);
    const ghost = (opacity) => flat({ transparent: true, opacity: opacity, depthWrite: false });
    this.styles = {
      body: { geometry: box, material: lit({}), pick: 'instance' },
      base: { geometry: box, material: lit({ roughness: 0.95 }), pick: 'instance' },
      pin: { geometry: box, material: flat({}), pick: 'instance' },
      keepout: { geometry: box, material: ghost(0.13) },
      dust: { geometry: box, material: flat({}), pick: 'net' },
      rep: { geometry: box, material: lit({ roughness: 0.6 }), pick: 'net' },
      arrow: { geometry: wedge, material: flat({}), pick: 'net' },
      ripped: { geometry: box, material: ghost(0.5) },
      partial: { geometry: box, material: ghost(0.6), pick: 'net' },
      support: { geometry: box, material: lit({ transparent: true, opacity: 0.25, depthWrite: false }) },
      clearance: { geometry: box, material: ghost(0.16) },
      conflict: { geometry: box, material: ghost(0.55) },
      marker: { geometry: box, material: flat({ wireframe: true }) },
      search: { geometry: box, material: ghost(0.42) },
      solid: { geometry: box, material: flat({}) },
      probe: { geometry: box, material: ghost(0.4) },
      sel: { geometry: box, material: flat({ transparent: true, opacity: 0.7, depthTest: false }), renderOrder: 10 },
      glow: { geometry: box, material: flat({ transparent: true, opacity: 0.9, depthWrite: false }), renderOrder: 5 },
    };
  }

  dispose() {
    this.alive = false;
    window.removeEventListener('resize', this.onResize);
    window.removeEventListener('keydown', this.onKey);
    LAYERS.forEach((name) => {
      const layer = this.layer[name];
      this.clearExtras(layer);
      layer.meshes.forEach((mesh) => {
        layer.group.remove(mesh);
        mesh.dispose();
      });
      layer.meshes.clear();
      this.scene.remove(layer.group);
    });
    this.geometries.forEach((g) => g.dispose());
    this.materials.forEach((m) => m.dispose());
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode) this.host.removeChild(this.renderer.domElement);
  }

  // -- layer machinery -----------------------------------------------------------

  layerVisible(name) {
    const L = this.layers;
    switch (name) {
      case 'routes': return L.dust !== false || L.repeaters !== false;
      case 'partial': return L.dust !== false;
      case 'search': return L.search !== false && this.hasSearch;
      case 'selection': return true;
      default: return L[name] !== false;
    }
  }

  // Per-style visibility inside a visible layer (kind toggles, dust / repeaters).
  styleVisible(key) {
    const sep = key.indexOf(':');
    const family = sep < 0 ? key : key.slice(0, sep);
    if (family === 'body' || family === 'base') return this.kindVisible[key.slice(sep + 1)] !== false;
    if (family === 'dust') return this.layers.dust !== false;
    if (family === 'rep' || family === 'arrow') return this.layers.repeaters !== false;
    return true;
  }

  batch(layer, key) {
    let b = layer.batches.get(key);
    if (!b) {
      b = new Batch();
      layer.batches.set(key, b);
    }
    return b;
  }

  begin(layer) {
    layer.batches.forEach((b) => { b.n = 0; });
    this.clearExtras(layer);
  }

  clearExtras(layer) {
    layer.extras.forEach((o) => {
      layer.group.remove(o);
      disposeObject(o);
    });
    layer.extras = [];
  }

  addExtra(layer, object) {
    layer.group.add(object);
    layer.extras.push(object);
  }

  // A pooled InstancedMesh for `key` holding at least `n` instances (a new
  // one starts with `minimum` slots, for meshes that grow event by event).
  meshFor(layer, key, n, minimum) {
    let mesh = layer.meshes.get(key);
    if (mesh && mesh.userData.capacity >= n) return mesh;
    const capacity = Math.max(n, minimum || 0, mesh ? Math.ceil(mesh.userData.capacity * 1.5) : 0);
    if (mesh) {
      layer.group.remove(mesh);
      mesh.dispose();
    }
    const sep = key.indexOf(':');
    const style = this.styles[sep < 0 ? key : key.slice(0, sep)];
    mesh = new THREE.InstancedMesh(style.geometry, style.material, capacity);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.renderOrder = style.renderOrder || 0;
    mesh.userData = { capacity: capacity, key: key, pick: style.pick || null, ids: new Int32Array(capacity), layer: layer.name };
    layer.group.add(mesh);
    layer.meshes.set(key, mesh);
    return mesh;
  }

  // Write instances [from, to) of a batch into its mesh and show `to` of them
  // (instances before `from` are already in the mesh from an earlier write).
  writeMesh(layer, key, batch, from, to, minimum) {
    const prior = layer.meshes.get(key);
    if (to === 0) {
      if (prior) {
        prior.count = 0;
        prior.visible = false;
      }
      return;
    }
    const mesh = this.meshFor(layer, key, to, minimum);
    if (mesh !== prior) from = 0; // a new (bigger) mesh holds nothing yet
    else if (from >= to && mesh.count === to) return; // nothing new
    const m = this.tmpMatrix;
    const c = this.tmpColor;
    for (let i = from; i < to; i++) {
      m.fromArray(batch.m, i * 16);
      mesh.setMatrixAt(i, m);
      c.fromArray(batch.c, i * 3);
      mesh.setColorAt(i, c);
    }
    mesh.userData.ids.set(batch.ids.subarray(from, to), from);
    mesh.count = to;
    mesh.visible = this.styleVisible(key);
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    mesh.boundingSphere = null; // three.js caches these per InstancedMesh
    mesh.boundingBox = null;
  }

  flush(layer) {
    layer.batches.forEach((batch, key) => this.writeMesh(layer, key, batch, 0, batch.n));
  }

  // Rebuild every visible layer whose inputs changed.
  sync() {
    const builders = this.builders || (this.builders = {
      components: (l) => this.buildComponents(l),
      pins: (l) => this.buildPins(l),
      keepout: (l) => this.buildKeepout(l),
      routes: (l) => this.buildRoutes(l),
      partial: (l) => this.buildPartial(l),
      supports: (l) => this.buildSupports(l),
      clearances: (l) => this.buildClearances(l),
      congestion: (l) => this.buildCongestion(l),
      probe: (l) => this.buildProbe(l),
      bounds: (l) => this.buildBounds(l),
      grid: (l) => this.buildGrid(l),
      signals: (l) => this.buildSignals(l),
      selection: (l) => this.buildSelection(l),
    });
    Object.keys(builders).forEach((name) => {
      const layer = this.layer[name];
      if (!this.layerVisible(name)) return; // rebuilt when shown again
      const key = LAYER_DEPS[name].map((d) => this.versions[d]).join(':');
      if (layer.key === key) return;
      layer.key = key;
      this.begin(layer);
      builders[name](layer);
      this.flush(layer);
    });
    if (this.layerVisible('search')) this.renderSearch();
    this.applyVisibility();
    this.dirty = true;
  }

  applyVisibility() {
    LAYERS.forEach((name) => {
      const layer = this.layer[name];
      layer.group.visible = this.layerVisible(name);
      layer.meshes.forEach((mesh, key) => { mesh.visible = mesh.count > 0 && this.styleVisible(key); });
    });
  }

  // -- layers --------------------------------------------------------------------

  buildComponents(layer) {
    const hl = this.highlight;
    this.state.placed.forEach((p, id) => {
      const inst = this.instances[id];
      const oc = inst ? this.orientedCell(inst.cell, p.q) : null;
      if (!oc) return;
      const colors = this.kindColors[inst.kind] || this.kindColors.and;
      const dim = hl.active && !hl.inst[id];
      const body = this.batch(layer, 'body:' + inst.kind);
      const base = this.batch(layer, 'base:' + inst.kind);
      const ox = p.origin[0] + 0.5;
      const oy = p.origin[1] + 0.5;
      const oz = p.origin[2] + 0.5;
      const v = oc.voxels;
      for (let k = 0; k < v.length; k += 4) {
        if (v[k + 3]) base.box(ox + v[k], oy + v[k + 1], oz + v[k + 2], 1, 1, 1, dim ? colors.baseDim : colors.base, id);
        else body.box(ox + v[k], oy + v[k + 1], oz + v[k + 2], 0.92, 0.92, 0.92, dim ? colors.bodyDim : colors.body, id);
      }
    });
  }

  // Pin endpoint blocks: a short tab pointing along the pin facing (the one
  // direction its route enters or leaves); inputs and outputs coloured apart.
  buildPins(layer) {
    const hl = this.highlight;
    const C = this.colors;
    const pins = this.batch(layer, 'pin');
    this.state.placed.forEach((p, id) => {
      const inst = this.instances[id];
      if (!inst || this.kindVisible[inst.kind] === false) return;
      const oc = this.orientedCell(inst.cell, p.q);
      if (!oc) return;
      const dim = hl.active && !hl.inst[id];
      oc.pins.forEach((pin) => {
        const x = p.origin[0] + pin.position[0] + 0.5 + pin.fx * 0.22;
        const y = p.origin[1] + pin.position[1] + 0.1;
        const z = p.origin[2] + pin.position[2] + 0.5 + pin.fz * 0.22;
        const color = pin.direction === 'in' ? (dim ? C.pinInDim : C.pinIn) : (dim ? C.pinOutDim : C.pinOut);
        pins.box(x, y, z, pin.fx ? 0.5 : 0.24, 0.2, pin.fz ? 0.5 : 0.24, color, id);
      });
    });
  }

  buildKeepout(layer) {
    const b = this.batch(layer, 'keepout');
    this.state.placed.forEach((p, id) => {
      const inst = this.instances[id];
      if (!inst || this.kindVisible[inst.kind] === false) return;
      const oc = this.orientedCell(inst.cell, p.q);
      if (!oc) return;
      const k = oc.keepout;
      for (let i = 0; i < k.length; i += 3) {
        b.box(p.origin[0] + k[i] + 0.5, p.origin[1] + k[i + 1] + 0.5, p.origin[2] + k[i + 2] + 0.5, 0.86, 0.86, 0.86, this.colors.keepout, id);
      }
    });
  }

  // One route: dust pads, edge bars / staircase risers, repeaters (slab plus
  // an arrow along its facing).  Dust brightness follows the signal strength
  // of a realized route, so a repeater's refresh to 15 is visible.
  emitWire(layer, prefix, net, g, realized, dim, override) {
    const pads = this.batch(layer, prefix + ':pad');
    const bars = this.batch(layer, prefix + ':bar');
    const reps = prefix === 'dust' ? this.batch(layer, 'rep') : null;
    const arrows = prefix === 'dust' ? this.batch(layer, 'arrow') : null;
    const C = this.colors;
    const xyz = g.xyz;
    const powered = !realized || realized.powered !== false;
    const color = this.scratch;
    for (let i = 0; i < g.n; i++) {
      const x = xyz[i * 3];
      const y = xyz[i * 3 + 1];
      const z = xyz[i * 3 + 2];
      let k = 0.85;
      if (realized) k = powered && g.strength[i] >= 0 ? 0.35 + 0.65 * Math.min(15, g.strength[i]) / 15 : 0.3;
      if (override) color.set(override);
      else this.netColor(net, k, dim, color);
      if (g.kind[i] === 1 && reps) {
        reps.box(x + 0.5, y + 0.0625, z + 0.5, 0.86, 0.125, 0.86, dim ? C.repeaterDim : C.repeater, net);
        const f = FACING[CLOCKWISE[g.facing[i]]] || FACING.east;
        arrows.yawed(x + 0.5, y + 0.16, z + 0.5, f[0], f[2], 0.66, 0.07, 0.5, dim ? C.arrowDim : C.arrow, net);
      } else {
        pads.box(x + 0.5, y + DUST_Y + 0.005, z + 0.5, DUST_PAD, DUST_THICK + 0.02, DUST_PAD, color, net);
      }
      const p = g.parent[i];
      if (p >= 0) this.edge(bars, x, y, z, xyz[p * 3], xyz[p * 3 + 1], xyz[p * 3 + 2], color, net);
    }
  }

  // The dust connecting block (x, y, z) to its parent block (px, py, pz).
  edge(bars, x, y, z, px, py, pz, color, net) {
    const dx = px - x;
    const dy = py - y;
    const dz = pz - z;
    const W = DUST_WIDTH;
    const T = DUST_THICK;
    if (Math.abs(dx) + Math.abs(dz) !== 1 || Math.abs(dy) > 1) {
      // Not one redstone step (never in a valid trace): a thin box spanning both centres.
      bars.box((x + px) / 2 + 0.5, (y + py) / 2 + DUST_Y, (z + pz) / 2 + 0.5,
        Math.abs(dx) + W, Math.abs(dy) + T, Math.abs(dz) + W, color, net);
      return;
    }
    if (dy === 0) {
      bars.box(x + 0.5 + dx / 2, y + DUST_Y, z + 0.5 + dz / 2, dx ? 1 : W, T, dz ? 1 : W, color, net);
      return;
    }
    // Staircase: the lower block's half bar, the riser up the face of the
    // upper block's support, then the upper block's half bar.
    const lx = dy > 0 ? x : px;
    const ly = dy > 0 ? y : py;
    const lz = dy > 0 ? z : pz;
    const hx = dy > 0 ? dx : -dx;
    const hz = dy > 0 ? dz : -dz;
    bars.box(lx + 0.5 + hx * 0.25, ly + DUST_Y, lz + 0.5 + hz * 0.25, hx ? 0.5 : W, T, hz ? 0.5 : W, color, net);
    bars.box(lx + 0.5 + hx * 0.47, ly + 0.53, lz + 0.5 + hz * 0.47, hx ? T : W, 1.05, hz ? T : W, color, net);
    bars.box(lx + hx + 0.5 - hx * 0.25, ly + 1 + DUST_Y, lz + hz + 0.5 - hz * 0.25, hx ? 0.5 : W, T, hz ? 0.5 : W, color, net);
  }

  buildRoutes(layer) {
    const s = this.state;
    const hl = this.highlight;
    s.realized.forEach((r, net) => {
      this.emitWire(layer, 'dust', net, this.routeGeometry(r, true), r, hl.active && !hl.net[net]);
    });
    s.routes.forEach((r, net) => {
      if (!s.realized.has(net)) this.emitWire(layer, 'dust', net, this.routeGeometry(r, false), null, hl.active && !hl.net[net]);
    });
    if (s.ripped && s.ripped.cells) {
      const b = this.batch(layer, 'ripped');
      s.ripped.cells.forEach((c) => b.box(c[0] + 0.5, c[1] + 0.2, c[2] + 0.5, 0.62, 0.4, 0.62, this.colors.ripped, s.ripped.net));
    }
  }

  buildPartial(layer) {
    const p = this.state.partial;
    if (!p) return;
    // A tree whose next branch failed is drawn in the failure colour.
    this.emitWire(layer, 'partial', p.net, this.routeGeometry(p, false), null, false, p.failed ? this.colors.failure : null);
  }

  // Powered dust (brightness = signal strength) and active devices at the
  // playback tick: a short redstone simulation of the final design.
  buildSignals(layer) {
    if (!this.playback) return;
    const st = this.signalState(this.simTick);
    const glow = this.batch(layer, 'glow');
    st.dust.forEach((strength, key) => {
      if (!(strength > 0)) return;
      const c = key.split(',').map(Number);
      const k = 0.35 + 0.65 * Math.min(15, strength) / 15;
      glow.box(c[0] + 0.5, c[1] + 0.1, c[2] + 0.5, 0.44, 0.12, 0.44, [k, 0.07 * k, 0.04 * k], -1);
    });
    st.devices.forEach((on, key) => {
      if (!on) return;
      const c = key.split(',').map(Number);
      glow.box(c[0] + 0.5, c[1] + 0.35, c[2] + 0.5, 0.36, 0.36, 0.36, [1, 0.86, 0.25], -1);
    });
  }

  buildSupports(layer) {
    const s = this.state;
    const b = this.batch(layer, 'support');
    const draw = (cells, net) => cells.forEach((c) => b.box(c[0] + 0.5, c[1] + 0.5, c[2] + 0.5, 0.97, 0.97, 0.97, this.colors.support, net));
    s.realized.forEach((r, net) => draw(this.routeGeometry(r, true).supports, net));
    s.routes.forEach((r, net) => { if (!s.realized.has(net)) draw(this.routeGeometry(r, false).supports, net); });
  }

  buildClearances(layer) {
    const s = this.state;
    const b = this.batch(layer, 'clearance');
    const draw = (cells, net) => cells.forEach((c) => b.box(c[0] + 0.5, c[1] + 0.5, c[2] + 0.5, 0.8, 0.8, 0.8, this.colors.clearance, net));
    s.realized.forEach((r, net) => draw(this.routeGeometry(r, true).clearances, net));
    s.routes.forEach((r, net) => { if (!s.realized.has(net)) draw(this.routeGeometry(r, false).clearances, net); });
  }

  // Conflict blocks coloured by conflict kind or by accumulated history, plus
  // verification violations and failure markers.
  buildCongestion(layer) {
    const s = this.state;
    const c = s.congestion;
    const b = this.batch(layer, 'conflict');
    if (c) {
      if (this.congestionMode === 'history' && c.cells && c.cells.length) {
        const cells = c.cells;
        let max = 0;
        cells.forEach((x) => { if ((Number(x.history) || 0) > max) max = Number(x.history); });
        const color = this.scratch;
        const lo = this.rgb(0x5a2410);
        const hi = this.rgb(0xffc04d);
        for (let i = 0; i < cells.length && i < MAX_CONGESTION_CELLS; i++) {
          const t = max > 0 ? Math.min(1, (Number(cells[i].history) || 0) / max) : 1;
          for (let j = 0; j < 3; j++) color[j] = lo[j] + (hi[j] - lo[j]) * t;
          const co = cells[i].coord;
          if (isCoord(co)) b.box(co[0] + 0.5, co[1] + 0.5, co[2] + 0.5, 0.82, 0.82, 0.82, color, -1);
        }
      } else {
        const rank = {};
        CONFLICT_KINDS.forEach((k, i) => { rank[k[0]] = i; });
        const blocks = new Map();
        (c.conflicts || []).forEach((conflict) => {
          const r = rank[conflict.kind] === undefined ? CONFLICT_KINDS.length : rank[conflict.kind];
          (conflict.cells || []).forEach((co) => {
            if (!isCoord(co)) return;
            const k = coordKey(co);
            const prior = blocks.get(k);
            if (!prior || prior.rank > r) blocks.set(k, { coord: co, rank: r, kind: conflict.kind });
          });
        });
        let n = 0;
        blocks.forEach((x) => {
          if (n++ >= MAX_CONGESTION_CELLS) return;
          const color = this.conflictColors[x.kind] || this.colors.failure;
          b.box(x.coord[0] + 0.5, x.coord[1] + 0.5, x.coord[2] + 0.5, 0.82, 0.82, 0.82, color, -1);
        });
      }
    }
    const markers = this.batch(layer, 'marker');
    s.violations.forEach((v) => (v.cells || []).forEach((co) => {
      if (isCoord(co)) markers.box(co[0] + 0.5, co[1] + 0.5, co[2] + 0.5, 1.06, 1.06, 1.06, this.colors.violation, -1);
    }));
    s.failures.forEach((f) => {
      if (isCoord(f.coord)) markers.box(f.coord[0] + 0.5, f.coord[1] + 0.5, f.coord[2] + 0.5, 1.12, 1.12, 1.12, this.colors.failure, -1);
    });
  }

  // The search layer updates incrementally: expansions are append-only until
  // the next branch starts, so its pooled meshes just grow.
  renderSearch() {
    const layer = this.layer.search;
    const s = this.state;
    const cur = s.search;
    const drawn = this.searchDrawn;
    const fresh = drawn.obj !== cur || drawn.cells > cur.cells.length || drawn.blocked > cur.blocked.length;
    if (!fresh && drawn.cells === cur.cells.length && drawn.blocked === cur.blocked.length && drawn.rejected === s.rejected) return;
    const cells = this.batch(layer, 'search');
    const blocked = this.batch(layer, 'search:blocked');
    if (fresh) {
      cells.n = 0;
      blocked.n = 0;
      drawn.cells = 0;
      drawn.blocked = 0;
    }
    const C = this.colors;
    const color = this.scratch;
    const h0 = cur.h0 || 1;
    const cellsFrom = cells.n;
    for (let i = drawn.cells; i < cur.cells.length; i++) {
      const e = cur.cells[i];
      if (!isCoord(e.coord)) continue;
      const t = Math.max(0, Math.min(1, 1 - (Number(e.h) || 0) / h0));
      for (let j = 0; j < 3; j++) color[j] = C.search[j] * (0.3 + 0.7 * t);
      cells.box(e.coord[0] + 0.5, e.coord[1] + 0.5, e.coord[2] + 0.5, 0.36, 0.36, 0.36, color, cur.net);
    }
    const blockedFrom = blocked.n;
    for (let i = drawn.blocked; i < cur.blocked.length; i++) {
      const to = cur.blocked[i].to;
      if (isCoord(to)) blocked.box(to[0] + 0.5, to[1] + 0.5, to[2] + 0.5, 0.18, 0.18, 0.18, C.blocked, cur.net);
    }
    this.writeMesh(layer, 'search', cells, cellsFrom, cells.n, 1024);
    this.writeMesh(layer, 'search:blocked', blocked, blockedFrom, blocked.n, 256);
    // Small and rewritten whole: the newest expansion, the goal, a rejected path.
    const solid = this.batch(layer, 'solid');
    solid.n = 0;
    const last = cur.cells.length ? cur.cells[cur.cells.length - 1].coord : null;
    if (isCoord(last)) solid.box(last[0] + 0.5, last[1] + 0.5, last[2] + 0.5, 0.5, 0.5, 0.5, C.head, cur.net);
    const rejected = s.rejected;
    if (rejected) {
      (rejected.path || []).forEach((co) => {
        if (isCoord(co)) solid.box(co[0] + 0.5, co[1] + 0.08, co[2] + 0.5, 0.3, 0.12, 0.3, C.probe, rejected.net);
      });
      const co = rejected.coord;
      if (isCoord(co)) solid.box(co[0] + 0.5, co[1] + 0.5, co[2] + 0.5, 0.6, 0.6, 0.6, C.blocked, rejected.net);
    }
    this.writeMesh(layer, 'solid', solid, 0, solid.n);
    const goal = this.batch(layer, 'marker');
    goal.n = 0;
    if (isCoord(cur.goal)) goal.box(cur.goal[0] + 0.5, cur.goal[1] + 0.5, cur.goal[2] + 0.5, 1.04, 1.04, 1.04, C.goal, cur.net);
    this.writeMesh(layer, 'marker', goal, 0, goal.n);
    drawn.obj = cur;
    drawn.cells = cur.cells.length;
    drawn.blocked = cur.blocked.length;
    drawn.rejected = rejected;
  }

  buildProbe(layer) {
    const probe = this.state.probe;
    if (!probe) return;
    let voxels = probe.voxels;
    if (!voxels) {
      const inst = this.instances[probe.instance];
      const oc = inst ? this.orientedCell(inst.cell, quarterTurns(probe.orientation)) : null;
      voxels = [];
      for (let k = 0; oc && k < oc.voxels.length; k += 4) {
        voxels.push([probe.origin[0] + oc.voxels[k], probe.origin[1] + oc.voxels[k + 1], probe.origin[2] + oc.voxels[k + 2]]);
      }
    }
    const b = this.batch(layer, 'probe');
    const color = probe.rejected ? this.colors.probeRejected : this.colors.probe;
    voxels.forEach((c) => {
      if (isCoord(c)) b.box(c[0] + 0.5, c[1] + 0.5, c[2] + 0.5, 1.02, 1.02, 1.02, color, probe.instance);
    });
  }

  box3(b) {
    return new THREE.Box3(
      new THREE.Vector3(b.min[0], b.min[1], b.min[2]),
      new THREE.Vector3(b.max[0] + 1, b.max[1] + 1, b.max[2] + 1),
    );
  }

  buildBounds(layer) {
    const s = this.state;
    if (s.bounds) this.addExtra(layer, new THREE.Box3Helper(this.box3(s.bounds), 0xffffff));
    if (s.searchBounds) this.addExtra(layer, new THREE.Box3Helper(this.box3(s.searchBounds), 0x2fbf71));
  }

  buildGrid(layer) {
    const s = this.state;
    const ref = s.searchBounds || s.bounds || this.finalBounds();
    if (!ref) return;
    const sx = ref.max[0] - ref.min[0] + 1;
    const sz = ref.max[2] - ref.min[2] + 1;
    // A square helper whose low corner sits on a block boundary one block
    // outside the bounds, so every line falls on an integer coordinate.
    const size = Math.max(sx, sz) + 2;
    const grid = new THREE.GridHelper(size, size, 0x3a4756, 0x232c36);
    grid.position.set(ref.min[0] - 1 + size / 2, 0, ref.min[2] - 1 + size / 2);
    this.addExtra(layer, grid);
  }

  buildSelection(layer) {
    const sel = this.selection;
    if (!sel) return;
    const s = this.state;
    const C = this.colors;
    if (sel.kind === 'instance') {
      const p = s.placed.get(sel.id);
      const inst = this.instances[sel.id];
      const oc = p && inst ? this.orientedCell(inst.cell, p.q) : null;
      if (!oc) return;
      const o = p.origin;
      this.addExtra(layer, new THREE.Box3Helper(this.box3({
        min: [o[0] + oc.lo[0], o[1] + oc.lo[1], o[2] + oc.lo[2]], max: [o[0] + oc.hi[0], o[1] + oc.hi[1], o[2] + oc.hi[2]],
      }), 0xffffff));
      return;
    }
    const route = s.realized.get(sel.id) || s.routes.get(sel.id) || (s.partial && s.partial.net === sel.id ? s.partial : null);
    const b = this.batch(layer, 'sel');
    if (route) {
      const g = this.routeGeometry(route, route === s.realized.get(sel.id));
      for (let i = 0; i < g.n; i++) {
        b.box(g.xyz[i * 3] + 0.5, g.xyz[i * 3 + 1] + 0.12, g.xyz[i * 3 + 2] + 0.5, 0.42, 0.1, 0.42, C.select, sel.id);
      }
    }
    const net = this.nets[sel.id];
    if (!net) return;
    const mark = (t, color) => {
      const c = this.pinBlock(t.instance, t.pin);
      if (c) b.box(c[0] + 0.5, c[1] + 0.35, c[2] + 0.5, 0.3, 0.7, 0.3, color, sel.id);
    };
    mark(net.driver, C.driver);
    net.sinks.forEach((t) => mark(t, C.sink));
  }

  finalBounds() {
    const f = this.final;
    return f.design_bounds || (f.metrics && f.metrics.placement && f.metrics.placement.bounds) || null;
  }

  // Frame the whole design: `fit` keeps the current viewing direction, the
  // presets look from fixed ones.  The distance fits the bounding sphere into
  // the narrower of the vertical and horizontal fields of view.
  fit(view) {
    const b = this.finalBounds() || this.state.bounds || { min: [0, 0, 0], max: [8, 8, 8] };
    const lo = new THREE.Vector3(b.min[0], b.min[1], b.min[2]);
    const hi = new THREE.Vector3(b.max[0] + 1, b.max[1] + 1, b.max[2] + 1);
    const mid = lo.clone().add(hi).multiplyScalar(0.5);
    const radius = Math.max(2, hi.clone().sub(lo).length() / 2);
    const dirs = { iso: [0.9, 0.8, 1.1], top: [0, 1, 0.0001], front: [0, 0.2, 1], side: [1, 0.2, 0] };
    let dir = null;
    if (view === 'fit') {
      dir = this.camera.position.clone().sub(this.controls.target);
      if (dir.length() < 1e-6) dir = null;
    }
    if (!dir) {
      const v = dirs[view] || dirs.iso;
      dir = new THREE.Vector3(v[0], v[1], v[2]);
    }
    dir.normalize();
    const vfov = ((this.camera.fov || 50) * Math.PI) / 180;
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * (this.camera.aspect || 1));
    const distance = (radius / Math.sin(Math.min(vfov, hfov) / 2)) * 1.04;
    this.camera.position.copy(mid.clone().add(dir.multiplyScalar(distance)));
    this.controls.target.copy(mid);
    this.camera.near = Math.max(0.05, distance / 2000);
    this.camera.far = distance * 20 + radius * 4;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.dirty = true;
  }

  // -- picking / inspector ---------------------------------------------------------

  pickTargets() {
    const targets = [];
    PICKABLE.forEach((name) => {
      if (!this.layerVisible(name)) return;
      this.layer[name].meshes.forEach((mesh) => {
        if (mesh.visible && mesh.count > 0 && mesh.userData.pick) targets.push(mesh);
      });
    });
    return targets;
  }

  pick(ev) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((ev.clientX - rect.left) / rect.width) * 2 - 1,
      -((ev.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(mouse, this.camera);
    // three.js raycasts invisible objects too, so only visible meshes are offered.
    const hits = this.raycaster.intersectObjects(this.pickTargets(), false);
    for (let i = 0; i < hits.length; i++) {
      const hit = hits[i];
      const info = hit.object.userData;
      if (!info || !info.pick || hit.instanceId === undefined || hit.instanceId >= hit.object.count) continue;
      const id = info.ids[hit.instanceId];
      if (id >= 0) {
        this.select({ kind: info.pick, id: id });
        return;
      }
    }
    this.select(null);
  }

  select(sel) {
    this.selection = sel;
    this.bump('selection');
    this.sync();
    this.showSelection();
  }

  describeTerminal(t) {
    const inst = this.instances[t.instance];
    if (!inst) return '#' + t.instance + '.' + t.pin;
    return '#' + inst.id + ' ' + inst.kind + '.' + t.pin + ' (' + inst.role + (inst.bit !== null && inst.bit !== undefined ? ' bit ' + inst.bit : '') + ')';
  }

  instanceLink(t) {
    return linkButton(this.describeTerminal(t), () => this.select({ kind: 'instance', id: t.instance }), 'select this primitive');
  }

  netLink(id, text) {
    return linkButton(text || 'net ' + id, () => this.select({ kind: 'net', id: id }), 'select this net');
  }

  irLabel(id) {
    const node = this.irNodes.get(id);
    const bus = this.buses.find((b) => b.ir_node === id);
    const op = node ? node.op : bus ? bus.op : '?';
    const type = typeName(node ? node.type : bus ? bus.type : null);
    return 'IR node ' + id + ' (' + op + (type ? ' ' + type : '') + (node && node.name ? ' "' + node.name + '"' : '') + ')';
  }

  groupPath(id) {
    const path = [];
    let g = this.groupById.get(id);
    for (let guard = 0; g && guard < 10000; guard++) {
      path.push(g);
      g = g.parent === null || g.parent === undefined ? null : this.groupById.get(g.parent);
    }
    return path.reverse();
  }

  showSelection() {
    const box = $('selection');
    box.innerHTML = '';
    const sel = this.selection;
    if (!sel) {
      box.textContent = 'Click a component or a wire.';
      box.className = 'muted';
      return;
    }
    box.className = '';
    if (sel.kind === 'instance') this.showInstance(box, sel.id);
    else this.showNet(box, sel.id);
  }

  showInstance(box, id) {
    const inst = this.instances[id];
    if (!inst) return;
    const cell = this.cells.get(inst.cell) || {};
    const placed = this.state.placed.get(id);
    const oc = this.orientedCell(inst.cell, placed ? placed.q : 0);
    const has = (v) => v !== null && v !== undefined;
    const rows = [
      ['instance', '#' + inst.id],
      ['kind', inst.kind + (inst.category ? ' (' + inst.category + ')' : '')],
      ['cell', inst.cell + (cell.placeholder ? ' (placeholder)' : '')],
      ['description', cell.description],
      ['IR node', has(inst.ir_node) ? this.irLabel(inst.ir_node) : 'none (global control)'],
      ['IR op', inst.ir_op],
      ['IR type', inst.ir_type],
      ['bit', inst.bit],
      ['role', inst.role],
      ['port', inst.port],
      ['init', inst.init],
      ['origin', placed ? fmtCoord(placed.origin) : 'unplaced'],
      ['orientation', placed ? placed.orientation : '-'],
      ['placement', placed && placed.column !== undefined ? 'column ' + placed.column +
        (placed.block !== undefined && placed.block !== null ? ', block ' + placed.block : '') : '-'],
      ['latency', has(cell.latency) ? cell.latency + ' tick(s)' : 'unknown'],
      ['stateful', cell.stateful ? 'yes' : 'no'],
      ['footprint', oc ? (oc.voxels.length / 4) + ' voxels, ' + (oc.keepout.length / 3) + ' keep-out' : '-'],
    ];
    if (inst.peripheral) rows.push(['peripheral', inst.peripheral.kind + ' (' + inst.peripheral.direction + ', ' + inst.peripheral.width + ' bits)']);
    if (inst.attrs) rows.push(['attrs', JSON.stringify(inst.attrs)]);
    box.appendChild(table(rows));

    const actions = element('div', 'row');
    if (has(inst.ir_node)) actions.appendChild(linkButton('highlight its IR node', () => this.setHighlight({ mode: 'ir_node', id: inst.ir_node })));
    if (has(inst.group) && this.groupById.has(inst.group)) actions.appendChild(linkButton('highlight its group', () => this.setHighlight({ mode: 'group', id: inst.group })));
    box.appendChild(actions);

    if (has(inst.group) && this.groupById.has(inst.group)) {
      box.appendChild(heading('hierarchy'));
      const list = element('div', 'path');
      this.groupPath(inst.group).forEach((g, depth) => {
        const row = element('div', '');
        row.style.paddingLeft = (depth * 10) + 'px';
        row.appendChild(linkButton(g.name + ' (' + g.kind + ')', () => this.setHighlight({ mode: 'group', id: g.id }), 'highlight this group'));
        list.appendChild(row);
      });
      box.appendChild(list);
    }

    const nets = [];
    (this.netsOfDriver.get(id) || []).forEach((n) => {
      const net = this.nets[n];
      nets.push(['drives', [this.netLink(n, 'net ' + n + ' (' + net.driver.pin + ' -> ' + net.fanout + ' sink(s))')]]);
    });
    (this.netsOfSink.get(id) || []).forEach((r) => {
      const net = this.nets[r.net];
      nets.push(['reads ' + r.pin, [this.netLink(r.net, 'net ' + r.net + ' (from #' + net.driver.instance + ')')]]);
    });
    if (nets.length) {
      box.appendChild(heading('nets'));
      box.appendChild(table(nets));
    }
  }

  showNet(box, id) {
    const net = this.nets[id];
    if (!net) return;
    const s = this.state;
    const route = s.routes.get(id);
    const realized = s.realized.get(id);
    const partial = s.partial && s.partial.net === id ? s.partial : null;
    const aliases = (net.logical || []).map((l) => {
      const wrap = element('div', '');
      wrap.appendChild(linkButton(this.irLabel(l.ir_node).replace('IR node ' + l.ir_node, 'IR node ' + l.ir_node + ' bit ' + l.bit),
        () => this.setHighlight({ mode: 'bus', id: l.ir_node }), 'highlight every bit of this bus'));
      return wrap;
    });
    const ports = (net.ports || []).map((p) => {
      const wrap = element('div', '');
      wrap.appendChild(linkButton(p.port + '[' + p.bit + ']', () => this.setHighlight({ mode: 'port', name: p.port }), 'highlight every bit of this port'));
      return wrap;
    });
    let length = 'not routed';
    if (route) length = route.length + ' blocks, ' + (route.branches || []).length + ' branch(es)' + (route.iteration !== undefined ? ', iteration ' + route.iteration : '');
    else if (partial) length = 'in progress: ' + partial.branches.length + ' branch(es)' + (partial.failed ? ' (FAILED)' : '');
    const sinks = net.sinks.map((t) => {
      const wrap = element('div', '');
      wrap.appendChild(this.instanceLink(t));
      return wrap;
    });
    const rows = [
      ['net', id], ['role', net.role], ['width', (net.width || 1) + ' bit'],
      ['driver', [this.instanceLink(net.driver)]], ['sinks', sinks], ['fanout', net.fanout],
      ['logical', aliases.length ? aliases : 'internal signal'], ['ports', ports.length ? ports : '-'],
      ['routed length', length],
    ];
    if (realized) {
      const dust = (realized.elements || []).length - (realized.repeaters || []).length;
      rows.push(['realized', dust + ' dust, ' + (realized.repeaters || []).length + ' repeater(s), ' +
        (realized.supports || []).length + ' supports, ' + (realized.clearances || []).length + ' clearances']);
      rows.push(['repeaters', (realized.repeaters || []).length]);
      rows.push(['powered', realized.powered ? 'yes (min strength ' + realized.min_strength + ')' : 'never (constant 0)']);
      rows.push(['max delay', realized.max_delay_ticks + ' repeater tick(s)']);
    } else {
      rows.push(['repeaters', route ? 'not legalized yet' : '-']);
    }
    box.appendChild(table(rows));
    const actions = element('div', 'row');
    const busIds = Array.from(new Set((net.logical || []).map((l) => l.ir_node)));
    busIds.slice(0, 4).forEach((n) => actions.appendChild(linkButton('highlight bus n' + n, () => this.setHighlight({ mode: 'bus', id: n }))));
    Array.from(new Set((net.ports || []).map((p) => p.port))).forEach((p) => actions.appendChild(linkButton('highlight port ' + p, () => this.setHighlight({ mode: 'port', name: p }))));
    box.appendChild(actions);
    if (realized && (realized.sinks || []).length) {
      box.appendChild(heading('sinks (electrical)'));
      box.appendChild(table(realized.sinks.map((r) => [
        '#' + r.instance + '.' + r.pin,
        'strength ' + r.strength + ' (needs ' + r.required + '), ' + r.repeaters + ' repeater(s), ' +
        r.delay_ticks + ' tick(s), ' + r.distance + ' blocks',
      ])));
    }
  }

  // -- highlight -------------------------------------------------------------------

  // spec: {mode: 'ir_node' | 'group' | 'bus', id} or {mode: 'port', name}; null clears.
  setHighlight(spec) {
    const N = this.instances.length;
    const M = this.nets.length;
    if (!spec || !spec.mode || spec.mode === 'none') {
      this.highlight = { active: false, inst: null, net: null, spec: null, label: '' };
    } else {
      const inst = new Uint8Array(N);
      const net = new Uint8Array(M);
      const markDriven = (id) => (this.netsOfDriver.get(id) || []).forEach((n) => { net[n] = 1; });
      let label = '';
      if (spec.mode === 'ir_node') {
        (this.instByIrNode.get(spec.id) || []).forEach((id) => { inst[id] = 1; markDriven(id); });
        (this.netsOfBus.get(spec.id) || []).forEach((n) => { net[n] = 1; });
        label = this.irLabel(spec.id);
      } else if (spec.mode === 'group') {
        const stack = [spec.id];
        for (let guard = 0; stack.length && guard < 1000000; guard++) {
          const g = stack.pop();
          (this.instByGroup.get(g) || []).forEach((id) => { inst[id] = 1; markDriven(id); });
          (this.groupChildren.get(g) || []).forEach((c) => stack.push(c));
        }
        const g = this.groupById.get(spec.id);
        label = 'group ' + (g ? this.groupPath(g.id).map((x) => x.name).join(' / ') : spec.id);
      } else if (spec.mode === 'bus') {
        (this.netsOfBus.get(spec.id) || []).forEach((n) => { net[n] = 1; inst[this.nets[n].driver.instance] = 1; });
        label = 'bus of ' + this.irLabel(spec.id);
      } else if (spec.mode === 'timing') {
        label = this.markTiming(spec.name, inst, net);
      } else if (spec.mode === 'port') {
        (this.netsOfPort.get(spec.name) || []).forEach((n) => { net[n] = 1; });
        const port = this.ports.find((p) => p.name === spec.name);
        (port ? port.bits : []).forEach((b) => { if (b.instance >= 0 && b.instance < N) inst[b.instance] = 1; });
        label = 'port ' + spec.name + (port ? ' (' + port.direction + ' ' + typeName(port.type) + ')' : '');
      }
      let ni = 0;
      let nn = 0;
      for (let i = 0; i < N; i++) ni += inst[i];
      for (let i = 0; i < M; i++) nn += net[i];
      this.highlight = { active: true, inst: inst, net: net, spec: spec, label: label + ': ' + ni + ' primitive(s), ' + nn + ' net(s)' };
    }
    this.bump('highlight');
    this.syncHighlightControls();
    this.sync();
    this.updateUI();
  }

  // Mark the clock tree or the worst setup / hold path (nets and cells named
  // by the timing report's path steps).
  markTiming(which, inst, net) {
    const N = this.instances.length;
    const M = this.nets.length;
    const mark = (steps) => (steps || []).forEach((st) => {
      (st.nets || []).forEach((n) => { if (n >= 0 && n < M) net[n] = 1; });
      const id = st.labels ? st.labels.instance : undefined;
      if (id !== undefined && id >= 0 && id < N) inst[id] = 1;
    });
    const markLabels = (labels) => {
      if (labels && labels.instance !== undefined && labels.instance >= 0 && labels.instance < N) inst[labels.instance] = 1;
    };
    if (which === 'clock') {
      this.nets.forEach((n) => { if (n.role === 'clock') { net[n.id] = 1; inst[n.driver.instance] = 1; } });
      this.instances.forEach((i) => { if (i.kind === 'register_bit') inst[i.id] = 1; });
      return 'clock tree';
    }
    const t = this.timing;
    const path = t ? (which === 'setup' ? (t.setup && t.setup.critical_path) : (t.hold && t.hold.critical_path)) ||
      (which === 'setup' && t.combinational ? t.combinational.critical_path : null) : null;
    if (!path) return 'no ' + which + ' path in this trace';
    mark(path.steps);
    mark(path.launch_clock_path);
    mark(path.capture_clock_path);
    markLabels(path.launch_register_labels);
    markLabels(path.capture_register_labels);
    return 'worst ' + which + ' path to ' + path.endpoint + ' (slack ' + (path.slack ? path.slack.gt + ' gt' : '-') + ')';
  }

  highlightItems(mode) {
    const items = [];
    if (mode === 'timing') {
      items.push(['clock', 'clock tree (clock net + every register)']);
      if (this.timing && (this.timing.setup || this.timing.combinational)) items.push(['setup', 'worst setup (critical) path']);
      if (this.timing && this.timing.hold) items.push(['hold', 'worst hold path']);
      return items;
    }
    if (mode === 'ir_node') {
      const ids = Array.from(this.instByIrNode.keys()).sort((a, b) => a - b);
      ids.forEach((id) => items.push([String(id), 'n' + id + ' ' + this.irLabel(id).replace('IR node ' + id + ' ', '') +
        ' · ' + this.instByIrNode.get(id).length + ' primitive(s)']));
    } else if (mode === 'group') {
      const walk = (id, depth) => {
        const g = this.groupById.get(id);
        if (!g || depth > 200) return;
        items.push([String(id), '\u00a0'.repeat(depth * 2) + g.name + ' (' + g.kind + ') · ' + (this.groupSize.get(id) || 0)]);
        (this.groupChildren.get(id) || []).forEach((c) => walk(c, depth + 1));
      };
      this.groups.forEach((g) => {
        if (g.parent === null || g.parent === undefined || !this.groupById.has(g.parent)) walk(g.id, 0);
      });
    } else if (mode === 'bus') {
      this.buses.forEach((b) => items.push([String(b.ir_node), 'n' + b.ir_node + ' ' + b.op + ' ' + typeName(b.type) +
        ' · ' + (this.netsOfBus.get(b.ir_node) || []).length + ' net(s)']));
    } else if (mode === 'port') {
      this.ports.forEach((p) => items.push([p.name, p.name + ' (' + p.direction + ' ' + typeName(p.type) + ', ' + p.realization + ') · ' +
        (this.netsOfPort.get(p.name) || []).length + ' net(s)']));
    }
    return items;
  }

  fillHighlightItems(mode) {
    const select = $('hl-item');
    select.innerHTML = '';
    const items = this.highlightItems(mode);
    const none = element('option', '', items.length ? '(choose)' : '(none)');
    none.value = '';
    select.appendChild(none);
    items.forEach((it) => {
      const o = element('option', '', it[1]);
      o.value = it[0];
      select.appendChild(o);
    });
    select.value = '';
  }

  syncHighlightControls() {
    const spec = this.highlight.spec;
    const mode = spec ? spec.mode : 'none';
    if ($('hl-mode').value !== mode || this.hlFilled !== mode) {
      $('hl-mode').value = mode;
      this.fillHighlightItems(mode);
      this.hlFilled = mode;
    }
    $('hl-item').value = spec ? String(spec.mode === 'port' || spec.mode === 'timing' ? spec.name : spec.id) : '';
    $('hl-info').textContent = this.highlight.active ? this.highlight.label : 'nothing highlighted';
  }

  // -- UI --------------------------------------------------------------------------

  initUI() {
    const t = this.trace;
    const f = this.final;
    const summary = this.design.summary || {};
    const gates = summary.gates !== undefined ? summary.gates : this.instances.filter((i) => i.category === 'gate').length;
    const regs = summary.register_bits !== undefined ? summary.register_bits : this.kindCounts.register_bit || 0;
    const source = t.source && (t.source.path || t.source.top) ? (t.source.path || '?') + (t.source.top ? ' · top ' + t.source.top : '') + '\n' : '';
    $('meta').textContent = source + (t.backend || 'physical-primitive') + ' · ' + (this.design.library || '') + '\n' +
      this.instances.length + ' instances (' + gates + ' gates, ' + regs + ' register bits) · ' +
      this.nets.length + ' one-bit nets\n' + this.events.length + ' events (' + t.trace_level + ') · ' +
      (f.success ? 'succeeded' : 'FAILED') + ' in ' + (f.attempts || '?') + ' attempt(s)';

    const timeline = $('timeline');
    timeline.max = String(this.events.length);
    timeline.oninput = () => { this.setPlaying(false); this.seek(Number(timeline.value)); };
    $('btn-start').onclick = () => { this.setPlaying(false); this.seek(0); };
    $('btn-back').onclick = () => { this.setPlaying(false); this.seek(this.position - 1); };
    $('btn-fwd').onclick = () => { this.setPlaying(false); this.seek(this.position + 1); };
    $('btn-end').onclick = () => { this.setPlaying(false); this.seek(this.events.length); };
    $('btn-final').onclick = $('btn-end').onclick;
    $('btn-play').onclick = () => {
      if (this.position >= this.events.length) this.seek(0);
      this.setPlaying(!this.playing);
    };
    // Default speed: a full replay in about a minute and a half (at least 20 events/s).
    const want = Math.max(20, this.events.length / 90);
    const speeds = Array.from($('speed').children).map((o) => Number(o.value)).filter((x) => x > 0);
    const pick = speeds.filter((x) => x <= want).pop();
    if (pick) $('speed').value = String(pick);
    this.speed = Number($('speed').value) || 20;
    $('speed').onchange = () => { this.speed = Number($('speed').value) || 20; };

    const fill = (id, entries) => {
      const select = $(id);
      select.innerHTML = '';
      const head = element('option', '', entries.length ? '(jump to...)' : '(none in this trace)');
      head.value = '';
      select.appendChild(head);
      entries.forEach((en) => {
        const o = element('option', '', en[1]);
        o.value = String(en[0]);
        select.appendChild(o);
      });
      select.onchange = () => {
        if (select.value === '') return;
        this.setPlaying(false);
        this.seek(Number(select.value));
      };
    };
    fill('attempt-select', this.restarts.map((k) => [k + 1, 'attempt ' + this.events[k].attempt + ' (event ' + k + ')']));
    fill('phase-select', this.milestones.map((m) => [m.index + 1, m.label + ' (event ' + m.index + ')']));
    fill('iteration-select', this.iterations.map((it) => [it.index + 1, 'attempt ' + it.attempt + ' · iteration ' + it.iteration + ' (' + it.nets + ' nets)']));

    $('hl-mode').onchange = () => {
      this.fillHighlightItems($('hl-mode').value);
      this.hlFilled = $('hl-mode').value;
      if ($('hl-mode').value === 'none') this.setHighlight(null);
    };
    $('hl-item').onchange = () => {
      const mode = $('hl-mode').value;
      const value = $('hl-item').value;
      if (value === '') return;
      this.setHighlight(mode === 'port' || mode === 'timing' ? { mode: mode, name: value } : { mode: mode, id: Number(value) });
    };
    $('btn-hl-clear').onclick = () => this.setHighlight(null);
    this.syncHighlightControls();

    document.querySelectorAll('[data-layer]').forEach((box) => {
      box.onchange = () => {
        this.layers[box.dataset.layer] = box.checked;
        this.sync();
      };
    });
    $('layer-search').className = this.hasSearch ? '' : 'absent';
    $('layer-probe').className = this.hasProbe ? '' : 'absent';
    $('congestion-mode').onchange = () => {
      this.congestionMode = $('congestion-mode').value === 'history' ? 'history' : 'kind';
      this.bump('mode');
      this.sync();
    };
    document.querySelectorAll('[data-view]').forEach((b) => {
      b.onclick = () => this.fit(b.dataset.view);
    });
    this.buildLegend();
    this.buildTimingPanel();

    this.onKey = (ev) => {
      if (ev.target && (ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT')) return;
      const n = ev.shiftKey ? 100 : 1;
      if (ev.key === ' ') { ev.preventDefault(); $('btn-play').onclick(); }
      else if (ev.key === 'ArrowRight') { this.setPlaying(false); this.seek(this.position + n); }
      else if (ev.key === 'ArrowLeft') { this.setPlaying(false); this.seek(this.position - n); }
      else if (ev.key === 'Home') $('btn-start').onclick();
      else if (ev.key === 'End') $('btn-end').onclick();
      else if (ev.key === 'PageDown' || ev.key === 'PageUp') this.jumpMilestone(ev.key === 'PageDown' ? 1 : -1);
      else if (ev.key === 'f' || ev.key === 'F') this.fit('fit');
      else if (ev.key === 'Escape') { this.select(null); this.setHighlight(null); }
      else return;
      if (ev.preventDefault) ev.preventDefault();
    };
    window.addEventListener('keydown', this.onKey);
  }

  // The physical timing / simulation summary (from the final block).
  buildTimingPanel() {
    const box = $('timing-panel');
    box.innerHTML = '';
    const t = this.timing;
    const sim = this.simulation;
    const ticks = (r) => (r && r.gt !== null && r.gt !== undefined ? r.rt + ' rt (' + r.gt + ' gt)' : '-');
    if (!t && !sim) {
      box.appendChild(element('div', 'muted small', 'no timing report (the run stopped before timing closure)'));
    }
    if (t) {
      const rows = [];
      if (t.sequential && t.clock) {
        rows.push(['clock period', ticks(t.clock.period) + ' · ' + t.clock.mode]);
        rows.push(['required', ticks(t.clock.required_period)]);
        rows.push(['clock arrival', ticks(t.clock.arrival_min) + ' .. ' + ticks(t.clock.arrival_max)]);
        rows.push(['clock skew', ticks(t.clock.skew)]);
        rows.push(['worst setup slack', ticks(t.setup && t.setup.worst_slack)]);
        rows.push(['worst hold slack', ticks(t.hold && t.hold.worst_slack)]);
        rows.push(['input offset', ticks(t.input_offset)]);
      } else if (t.combinational) {
        rows.push(['combinational settle', ticks(t.combinational.settle)]);
      }
      if (this.clockTree) {
        rows.push(['clock balancing', this.clockTree.repeaters_added + ' repeater(s) added, ' + this.clockTree.repeaters_raised +
          ' raised (+' + this.clockTree.delay_added_rt + ' rt); skew ' + this.clockTree.skew_before.rt + ' -> ' +
          this.clockTree.skew_after.rt + ' rt']);
      }
      rows.push(['closure', t.closure && t.closure.passed ? 'passed' : 'FAILED ' +
        ((t.closure && t.closure.failures) || []).map((x) => x.code).join(', ')]);
      box.appendChild(table(rows));
      const path = t.setup ? t.setup.critical_path : t.combinational ? t.combinational.critical_path : null;
      if (path && Array.isArray(path.steps)) {
        box.appendChild(element('div', 'muted small', 'critical path to ' + path.endpoint + ' (' + path.steps.length + ' steps)'));
        const list = element('div', 'small');
        path.steps.slice(0, 40).forEach((st) => {
          const line = element('div', '', '+' + (st.edge_delay ? st.edge_delay.gt : 0) + ' gt → ' + (st.arrival ? st.arrival.gt : '?') + ' gt · ' + st.kind + ' ');
          (st.nets || []).slice(0, 2).forEach((n) => line.appendChild(this.netLink(n)));
          if (st.labels && st.labels.instance !== undefined) {
            line.appendChild(linkButton('#' + st.labels.instance + ' ' + (st.labels.kind || ''), () => this.select({ kind: 'instance', id: st.labels.instance })));
          }
          list.appendChild(line);
        });
        box.appendChild(list);
      }
      const row = element('div', 'row');
      [['clock', 'clock tree'], ['setup', 'setup path'], ['hold', 'hold path']].forEach((pair) => {
        row.appendChild(linkButton('highlight ' + pair[1], () => this.setHighlight({ mode: 'timing', name: pair[0] })));
      });
      box.appendChild(row);
    }
    if (sim) {
      box.appendChild(table([
        ['simulation', (sim.mode || '?') + (sim.validated ? ' · validated' : sim.skipped ? ' · skipped' : ' · FAILED')],
        ['model', (sim.model || '') + ' · ' + (sim.minecraft_version || '')],
        ['checked', sim.cycles ? sim.cycles + ' logical cycles' : sim.vectors ? sim.vectors + ' input vectors' : '-'],
      ]));
    }
    const pb = this.playback;
    $('sim-controls').hidden = !pb;
    if (pb) {
      const slider = $('sim-tick');
      slider.min = String(pb.start);
      slider.max = String(pb.end);
      slider.value = String(this.simTick);
      slider.oninput = () => { this.simPlaying = false; this.setSimTick(Number(slider.value)); };
      $('btn-sim-play').onclick = () => {
        if (this.simTick >= pb.end) this.setSimTick(pb.start);
        this.simPlaying = !this.simPlaying;
        this.simAccum = 0;
        this.updateSimUI();
      };
      this.updateSimUI();
    }
  }

  updateSimUI() {
    const pb = this.playback;
    if (!pb) return;
    $('sim-tick').value = String(this.simTick);
    const st = this.signalState(this.simTick);
    let powered = 0;
    st.dust.forEach((v) => { if (v > 0) powered++; });
    const marks = st.marks.map((m) => m.type + (m.component ? ' ' + m.component : '') + (m.port ? ' ' + m.port : '')).join(', ');
    $('sim-info').textContent = 'game tick ' + this.simTick + ' (' + (this.simTick / 2) + ' rt) of ' + pb.start + '..' + pb.end +
      ' · ' + powered + ' powered dust' + (marks ? ' · ' + marks : '') + (pb.truncated ? ' · (recording truncated)' : '');
    $('btn-sim-play').textContent = this.simPlaying ? '\u23F8 Pause' : '\u25B6 Play signals';
  }

  jumpMilestone(direction) {
    const stops = this.milestones.map((m) => m.index + 1).concat(this.iterations.map((it) => it.index + 1)).sort((a, b) => a - b);
    let target = direction > 0 ? this.events.length : 0;
    if (direction > 0) {
      for (let i = 0; i < stops.length; i++) if (stops[i] > this.position) { target = stops[i]; break; }
    } else {
      for (let i = stops.length - 1; i >= 0; i--) if (stops[i] < this.position) { target = stops[i]; break; }
    }
    this.setPlaying(false);
    this.seek(target);
  }

  buildLegend() {
    const legend = $('legend');
    legend.innerHTML = '';
    const swatch = (hex) => {
      const sw = element('span', 'swatch');
      sw.style.background = hexCss(hex);
      return sw;
    };
    const row = (hex, text) => {
      const r = element('div', '');
      r.appendChild(swatch(hex));
      r.appendChild(document.createTextNode(text));
      return r;
    };
    legend.appendChild(element('div', 'legend-title', 'components (click to show / hide)'));
    this.kindBoxes = [];
    const kinds = KINDS.filter((k) => this.kindCounts[k[0]]).map((k) => k.slice());
    Object.keys(this.kindCounts).forEach((kind) => {
      if (!KINDS.some((k) => k[0] === kind)) kinds.push([kind, OTHER_KIND, kind]);
    });
    kinds.forEach((k) => {
      const label = element('label', 'kind');
      const box = element('input', '');
      box.type = 'checkbox';
      box.checked = this.kindVisible[k[0]] !== false;
      box.dataset.kind = k[0];
      box.onchange = () => {
        this.kindVisible[k[0]] = box.checked;
        this.bump('kinds');
        this.sync();
      };
      this.kindBoxes.push(box);
      label.appendChild(box);
      label.appendChild(swatch(k[1]));
      label.appendChild(document.createTextNode(k[2] + ' · ' + this.kindCounts[k[0]]));
      legend.appendChild(label);
    });
    legend.appendChild(element('div', 'muted small', 'darker blocks: cell base layer'));
    legend.appendChild(row(PIN_IN, 'input pin (tab points along facing)'));
    legend.appendChild(row(PIN_OUT, 'output pin'));
    legend.appendChild(element('div', 'legend-title', 'routes'));
    const ramp = element('div', '');
    [15, 11, 7, 3, 1].forEach((st) => {
      const sw = element('span', 'swatch');
      const k = 0.35 + 0.65 * st / 15;
      sw.style.background = 'hsl(200, 70%, ' + Math.round(18 + 42 * k) + '%)';
      sw.title = 'strength ' + st;
      ramp.appendChild(sw);
    });
    ramp.appendChild(document.createTextNode('dust, bright = strength 15 ... dim = 1 (each net its own hue)'));
    legend.appendChild(ramp);
    legend.appendChild(row(REPEATER_ARROW, 'repeater: grey slab + arrow along its facing'));
    legend.appendChild(row(CLOCK_NET, 'clock net'));
    legend.appendChild(row(RESET_NET, 'reset net'));
    legend.appendChild(row(RIPPED, 'route just ripped up'));
    legend.appendChild(row(SUPPORT, 'support block (layer)'));
    legend.appendChild(row(CLEARANCE, 'staircase clearance, kept air (layer)'));
    legend.appendChild(row(KEEPOUT, 'cell keep-out (layer)'));
    legend.appendChild(element('div', 'legend-title', 'congestion and failures'));
    CONFLICT_KINDS.forEach((k) => legend.appendChild(row(k[1], k[0].replace(/_/g, ' ') + ': ' + k[2])));
    legend.appendChild(row(0xffc04d, 'history mode: dark = little, bright = most history'));
    legend.appendChild(row(FAILURE, 'routing / legalization failure (wire box)'));
    legend.appendChild(row(VIOLATION, 'verification violation (wire box)'));
    if (this.hasSearch || this.hasProbe) legend.appendChild(element('div', 'legend-title', 'search and placement'));
    if (this.hasSearch) {
      legend.appendChild(row(SEARCH, 'A* expansion (brighter = nearer the goal)'));
      legend.appendChild(row(0xffffff, 'newest expansion'));
      legend.appendChild(row(BLOCKED, 'blocked transition / rejected block'));
      legend.appendChild(row(GOAL, 'branch goal (sink pin)'));
      legend.appendChild(row(PROBE, 'rejected path'));
    }
    if (this.hasProbe) {
      legend.appendChild(row(PROBE, 'placement probe'));
      legend.appendChild(row(PROBE_REJECTED, 'rejected probe (reason in the status)'));
    }
  }

  setPlaying(playing) {
    this.playing = playing;
    this.accum = 0;
    $('btn-play').textContent = playing ? '⏸ Pause' : '▶ Play';
  }

  routeStats() {
    const key = this.versions.routes + ':' + this.versions.placed;
    if (this.statsKey === key) return this.stats;
    const s = this.state;
    let repeaters = 0;
    let blocks = 0;
    s.realized.forEach((r) => {
      repeaters += (r.repeaters || []).length;
      blocks += (r.elements || []).length;
    });
    s.routes.forEach((r, net) => { if (!s.realized.has(net)) blocks += r.length || 0; });
    this.stats = { repeaters: repeaters, blocks: blocks };
    this.statsKey = key;
    return this.stats;
  }

  updateUI() {
    const n = this.events.length;
    $('timeline').value = String(this.position);
    $('position').textContent = n ? this.position + ' / ' + n + ' events' : 'no events (trace level none): final block';
    const s = this.state;
    const lines = [s.status];
    const where = [];
    if (s.attempt !== null) where.push('attempt ' + s.attempt);
    if (s.iteration !== null) where.push('iteration ' + s.iteration);
    if (s.presentFactor !== null) where.push('present factor ' + Number(s.presentFactor).toFixed(2));
    if (s.round !== null) where.push('legalization round ' + s.round);
    if (where.length) lines.push(where.join(' · '));
    const stats = this.routeStats();
    lines.push(s.placed.size + '/' + this.instances.length + ' placed · ' + s.routes.size + '/' + this.nets.length +
      ' routed · ' + s.realized.size + ' realized · ' + stats.blocks + ' signal blocks · ' + stats.repeaters + ' repeaters');
    const conflicts = s.congestion && s.congestion.conflicts ? s.congestion.conflicts.length : 0;
    if (conflicts || s.failures.length || s.violations.length) {
      lines.push(conflicts + ' conflict(s) · ' + s.failures.length + ' failure(s) · ' + s.violations.length + ' violation(s)');
    }
    if (s.probe && s.probe.rejected) lines.push('probe #' + s.probe.instance + ' REJECTED: ' + s.probe.rejected);
    if (this.highlight.active) lines.push('highlight: ' + this.highlight.label);
    $('status').textContent = lines.join('\n');
    const e = s.lastEvent;
    if (e !== this.shownEvent || !n) {
      this.shownEvent = e;
      let text = e ? JSON.stringify(preview(e, 0), null, 1) : (n ? '-' : 'final block (no events recorded)');
      if (text.length > 6000) text = text.slice(0, 6000) + '\n...';
      $('event').textContent = text;
    }
    // The inspector only changes with the selection or the placed / routed state.
    const v = this.versions;
    const key = v.selection + ':' + v.placed + ':' + v.routes + ':' + v.partial + ':' + v.highlight;
    if (this.selection && key !== this.shownKey) this.showSelection();
    this.shownKey = key;
  }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

let viewer = null;

function start(trace) {
  const errors = validateTrace(trace);
  if (errors.length) {
    showError(errors);
    return;
  }
  hideError();
  if (viewer) viewer.dispose();
  viewer = null;
  try {
    viewer = new Viewer(trace);
    window.redcViewer = viewer; // for poking at the replay state from the browser console
  } catch (err) {
    showError(['viewer error: ' + err.message]);
    throw err;
  }
}

function loadFile(file) {
  if (!file) return;
  file.text().then((text) => {
    let trace;
    try {
      trace = parseTraceText(text, file.name);
    } catch (err) {
      showError(['could not parse ' + file.name + ': ' + err.message]);
      return;
    }
    start(trace);
  });
}

$('file-input').addEventListener('change', (ev) => loadFile(ev.target.files[0]));
document.addEventListener('dragover', (ev) => ev.preventDefault());
document.addEventListener('drop', (ev) => {
  ev.preventDefault();
  if (ev.dataTransfer && ev.dataTransfer.files.length) loadFile(ev.dataTransfer.files[0]);
});

const embedded = $('redc-trace').textContent.trim();
if (embedded && embedded !== 'null') {
  let trace = null;
  try {
    trace = JSON.parse(embedded);
  } catch (err) {
    showError(['embedded trace is not valid JSON: ' + err.message]);
  }
  if (trace) start(trace);
} else {
  showError(['No trace embedded. Use "Trace file" to load a .primitive.pnr.json or .jsonl trace.']);
}
