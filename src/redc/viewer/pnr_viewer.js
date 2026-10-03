// RedC place-and-route replay viewer for redc.pnr.trace.v1 traces.
//
// Everything rendered comes from the trace itself (component dims, ports,
// events, final state) -- never from the YAML cell library -- exactly as a
// future Minecraft mod would consume it.  The viewer replays the event list:
// the state after N events is rebuilt by applying events in order, restarting
// from the latest `pnr_attempt_begin` (which resets the design) when seeking
// backwards.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

window.__redcViewerStarted = true;

const SCHEMA = 'redc.pnr.trace.v1';
const MAX_SEARCH_CELLS = 40000;
const CATEGORY_COLORS = {
  compute: 0x4f8fdf,
  register: 0xb05be0,
  peripheral: 0x2fbf71,
  boundary: 0x9aa4b2,
  control: 0xf2b134,
  routing: 0x7a7a7a,
};
const PORT_IN = 0x5ab0ff;
const PORT_OUT = 0xff9a3c;

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
    errors.push('schema is ' + JSON.stringify(trace.schema) + ', expected "' + SCHEMA + '"');
  }
  const design = trace.design;
  if (!design || !Array.isArray(design.components) || !Array.isArray(design.instances) ||
      !Array.isArray(design.nets)) {
    errors.push('design.components, design.instances and design.nets must be arrays');
  }
  if (!Array.isArray(trace.events)) errors.push('events must be an array');
  if (errors.length) return errors;
  const components = new Set();
  design.components.forEach((c) => {
    components.add(c.id);
    if (!Array.isArray(c.dim) || c.dim.length !== 3 || !c.dim.every((d) => d > 0)) {
      errors.push('component ' + c.id + ' has no positive [dx, dy, dz] dim');
    }
  });
  const instances = new Set();
  design.instances.forEach((i) => {
    instances.add(i.id);
    if (!components.has(i.component)) {
      errors.push('instance ' + i.id + ' uses unknown component ' + JSON.stringify(i.component));
    }
  });
  for (let k = 0; k < trace.events.length; k++) {
    const e = trace.events[k];
    if (!e || e.seq !== k || typeof e.type !== 'string' || typeof e.phase !== 'string') {
      errors.push('event ' + k + ' is malformed or out of sequence');
      break;
    }
    if (e.type === 'component_placed' && (!instances.has(e.instance) || !isCoord(e.origin))) {
      errors.push('event ' + k + ' places an unknown instance or has a bad origin');
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
// Small DOM / geometry helpers
// ---------------------------------------------------------------------------

function table(rows) {
  const t = document.createElement('table');
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    const k = document.createElement('td');
    k.textContent = row[0];
    const v = document.createElement('td');
    v.textContent = row[1] === null || row[1] === undefined ? '-' : String(row[1]);
    tr.appendChild(k);
    tr.appendChild(v);
    t.appendChild(tr);
  });
  return t;
}

function fmtCoord(c) {
  return c ? '[' + c.join(', ') + ']' : '-';
}

function netColor(net) {
  if (net && net.role === 'clock') return new THREE.Color(0xffd166);
  if (net && net.role === 'reset') return new THREE.Color(0xef476f);
  const hue = ((net ? net.id : 0) * 0.618033988749895) % 1;
  return new THREE.Color().setHSL(hue, 0.75, 0.58);
}

function wireThickness(net) {
  const lanes = net && net.layout ? net.layout.lane_count : 1;
  return 0.12 + 0.05 * Math.log2(Math.max(1, lanes));
}

function center(c) {
  return new THREE.Vector3(c[0] + 0.5, c[1] + 0.5, c[2] + 0.5);
}

function disposeGroup(group) {
  while (group.children.length) {
    const child = group.children[group.children.length - 1];
    group.remove(child);
    child.traverse((o) => {
      if (o.geometry && !o.geometry.userData.shared) o.geometry.dispose();
      if (o.material) {
        (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
      }
      if (o.isCSS2DObject && o.element && o.element.parentNode) {
        o.element.parentNode.removeChild(o.element);
      }
    });
  }
}

// ---------------------------------------------------------------------------
// The viewer
// ---------------------------------------------------------------------------

class Viewer {
  constructor(trace) {
    this.trace = trace;
    this.events = trace.events;
    this.components = new Map(trace.design.components.map((c) => [c.id, c]));
    this.instances = new Map(trace.design.instances.map((i) => [i.id, i]));
    this.nets = new Map(trace.design.nets.map((n) => [n.id, n]));
    this.restarts = [];
    this.iterations = [];
    let attempt = 0;
    this.events.forEach((e, k) => {
      if (e.type === 'pnr_attempt_begin') {
        attempt = e.attempt;
        this.restarts.push(k);
      } else if (e.type === 'routing_iteration_begin') {
        this.iterations.push({ index: k, attempt: attempt, iteration: e.iteration });
      }
    });
    this.hasSearch = this.events.some((e) => e.type === 'route_search_expand');
    this.hasCongestion = this.events.some((e) => e.type === 'congestion_snapshot' && e.cells.length) ||
      Boolean(trace.final && trace.final.congestion && trace.final.congestion.length);
    this.layers = {};
    document.querySelectorAll('[data-layer]').forEach((box) => {
      this.layers[box.dataset.layer] = box.checked;
    });
    this.versions = { placed: 0, routes: 0, search: 0, congestion: 0, bounds: 0, probe: 0 };
    this.rendered = {};
    this.state = this.emptyState();
    this.position = 0;
    this.playing = false;
    this.speed = Number($('speed').value);
    this.accum = 0;
    this.selection = null;
    this.alive = true;
    this.initThree();
    this.initUI();
    this.seek(0);
    if (!this.events.length) this.loadFinal();
    this.fit('iso');
  }

  emptyState() {
    return {
      attempt: null, geometry: null, placed: new Map(), probe: null, routes: new Map(),
      search: [], congestion: [], ripped: null, failed: [], bounds: null, searchBounds: null,
      iteration: null, presentFactor: null, status: 'unplaced design', lastEvent: null,
    };
  }

  bump() {
    for (let i = 0; i < arguments.length; i++) this.versions[arguments[i]]++;
  }

  bumpAll() {
    Object.keys(this.versions).forEach((k) => { this.versions[k]++; });
  }

  // -- replay ---------------------------------------------------------------

  apply(e) {
    let s = this.state;
    if (s.ripped) {
      s.ripped = null;
      this.bump('routes');
    }
    s.lastEvent = e;
    switch (e.type) {
      case 'pnr_attempt_begin':
        this.state = s = this.emptyState();
        s.lastEvent = e;
        s.attempt = e.attempt;
        s.geometry = e;
        s.status = 'attempt ' + e.attempt + ': spacing ' + e.component_spacing +
          ', layer gap ' + e.layer_gap + ', margin ' + e.routing_margin;
        this.bumpAll();
        break;
      case 'placement_begin':
        s.status = 'placing ' + e.layers.reduce((n, l) => n + l.instances.length, 0) +
          ' components in ' + e.layers.length + ' layers';
        break;
      case 'component_place_attempt':
        s.probe = { instance: e.instance, origin: e.origin, rejected: null };
        this.bump('probe');
        break;
      case 'component_place_rejected':
        s.probe = { instance: e.instance, origin: e.origin, rejected: e.reason };
        this.bump('probe');
        break;
      case 'component_placed':
        s.placed.set(e.instance, e.origin);
        s.probe = null;
        this.bump('placed', 'probe');
        break;
      case 'design_bounds_changed':
        s.bounds = { min: e.min, max: e.max };
        this.bump('bounds');
        break;
      case 'placement_complete':
        s.status = 'placed ' + e.component_count + ' components';
        break;
      case 'placement_failed':
        s.status = 'placement failed';
        break;
      case 'routing_begin':
        s.searchBounds = e.search_bounds;
        s.status = 'routing ' + e.nets + ' nets';
        this.bump('bounds');
        break;
      case 'routing_iteration_begin':
        s.iteration = e.iteration;
        s.presentFactor = e.present_factor;
        s.status = 'iteration ' + e.iteration + ': ' + e.nets.length + ' nets to route';
        break;
      case 'net_route_begin':
        s.routes.set(e.net, { branches: [], root: e.driver.coord, partial: true });
        this.bump('routes');
        break;
      case 'branch_route_begin':
        s.search = [];
        this.bump('search');
        break;
      case 'route_search_expand':
        if (s.search.length < MAX_SEARCH_CELLS) s.search.push(e.coord);
        this.bump('search');
        break;
      case 'branch_route_found': {
        let route = s.routes.get(e.net);
        if (!route) {
          route = { branches: [], root: e.start, partial: true };
          s.routes.set(e.net, route);
        }
        route.branches = route.branches.concat([{ sink: e.sink, start: e.start, goal: e.goal, path: e.path }]);
        s.search = [];
        this.bump('routes', 'search');
        break;
      }
      case 'branch_route_failed':
        s.failed = s.failed.concat([{ net: e.net, goal: e.goal, reason: e.reason }]);
        s.status = 'net ' + e.net + ' FAILED: ' + e.reason;
        this.bump('routes');
        break;
      case 'net_route_committed':
        s.routes.set(e.net, { branches: e.branches, root: e.root, cells: e.cells, partial: false });
        s.search = [];
        this.bump('routes', 'search');
        break;
      case 'net_rip_up':
        s.routes.delete(e.net);
        s.ripped = { net: e.net, cells: e.cells };
        this.bump('routes');
        break;
      case 'routing_iteration_end':
        s.status = 'iteration ' + e.iteration + ' done: ' + e.overused + ' overused cells, ' +
          e.routed_cells + ' wire cells';
        break;
      case 'congestion_snapshot':
        s.congestion = e.cells;
        this.bump('congestion');
        break;
      case 'keyframe':
        s.placed = new Map(e.placement.filter((p) => p.origin).map((p) => [p.instance, p.origin]));
        s.routes = new Map(e.routes.map((r) => [r.net, { branches: r.branches, root: r.root, cells: r.cells, partial: false }]));
        this.bump('placed', 'routes');
        break;
      case 'routing_complete':
        s.status = 'routing converged after ' + e.iterations + ' iteration(s)';
        break;
      case 'routing_failed':
        s.status = 'routing failed: ' + (e.failure ? e.failure.message : '');
        break;
      case 'routes_finalized':
        s.congestion = [];
        s.status = 'final: ' + e.nets + ' nets, ' + e.wire_cells + ' wire cells';
        this.bump('congestion');
        break;
      case 'pnr_attempt_end':
        s.status = 'attempt ' + e.attempt + ' ' + e.status +
          (e.failure ? ': ' + e.failure.message : '');
        break;
      case 'pnr_end':
        s.status = e.success ? 'P&R succeeded' : 'P&R FAILED after ' + e.attempts + ' attempt(s)';
        break;
      default:
        break; // unknown event types are ignored (forward compatibility)
    }
  }

  loadFinal() {
    const f = this.trace.final;
    if (!f) return;
    const s = this.state;
    s.placed = new Map((f.placement || []).filter((p) => p.origin).map((p) => [p.instance, p.origin]));
    s.routes = new Map((f.routes || []).map((r) => [r.net, { branches: r.branches, root: r.root, cells: r.cells, partial: false }]));
    s.congestion = f.congestion || [];
    if (f.design_bounds) s.bounds = { min: f.design_bounds.min, max: f.design_bounds.max };
    if (f.search_bounds) s.searchBounds = f.search_bounds;
    s.status = f.success ? 'final design' : 'FAILED: ' + (f.failure ? f.failure.message : '');
    this.bumpAll();
    this.sync();
    this.updateUI();
  }

  seek(target) {
    const n = this.events.length;
    target = Math.max(0, Math.min(n, Math.round(target)));
    if (target < this.position) {
      let start = 0;
      for (let i = 0; i < this.restarts.length && this.restarts[i] < target; i++) start = this.restarts[i];
      this.state = this.emptyState();
      this.bumpAll();
      this.position = start;
    }
    while (this.position < target) {
      this.apply(this.events[this.position]);
      this.position++;
    }
    this.sync();
    this.updateUI();
  }

  // -- three.js -------------------------------------------------------------

  initThree() {
    const host = $('scene');
    this.host = host;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(host.clientWidth, host.clientHeight);
    host.appendChild(this.renderer.domElement);
    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.setSize(host.clientWidth, host.clientHeight);
    const ls = this.labelRenderer.domElement.style;
    ls.position = 'absolute';
    ls.top = '0';
    ls.pointerEvents = 'none';
    host.appendChild(this.labelRenderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x101418);
    this.camera = new THREE.PerspectiveCamera(50, host.clientWidth / Math.max(1, host.clientHeight), 0.1, 20000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(1, 2, 1.4);
    this.scene.add(sun);

    this.groups = {};
    ['components', 'labels', 'ports', 'wires', 'voxels', 'search', 'congestion', 'grid', 'bounds', 'probe']
      .forEach((name) => {
        const g = new THREE.Group();
        g.name = name;
        this.groups[name] = g;
        this.scene.add(g);
      });
    this.unitBox = new THREE.BoxGeometry(1, 1, 1);
    this.portGeometry = new THREE.SphereGeometry(0.13, 10, 8);
    this.unitBox.userData.shared = true; // reused by many meshes: never disposed per rebuild
    this.portGeometry.userData.shared = true;
    this.raycaster = new THREE.Raycaster();

    this.onResize = () => {
      const w = host.clientWidth;
      const h = Math.max(1, host.clientHeight);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
      this.labelRenderer.setSize(w, h);
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
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this.labelRenderer.render(this.scene, this.camera);
    };
    loop();
  }

  dispose() {
    this.alive = false;
    window.removeEventListener('resize', this.onResize);
    window.removeEventListener('keydown', this.onKey);
    Object.keys(this.groups).forEach((k) => disposeGroup(this.groups[k]));
    this.renderer.dispose();
    this.host.removeChild(this.renderer.domElement);
    this.host.removeChild(this.labelRenderer.domElement);
  }

  sync() {
    const v = this.versions;
    const r = this.rendered;
    if (r.placed !== v.placed) { this.renderComponents(); r.placed = v.placed; }
    if (r.probe !== v.probe) { this.renderProbe(); r.probe = v.probe; }
    if (r.routes !== v.routes) { this.renderWires(); r.routes = v.routes; }
    if (r.search !== v.search) { this.renderSearch(); r.search = v.search; }
    if (r.congestion !== v.congestion) { this.renderCongestion(); r.congestion = v.congestion; }
    if (r.bounds !== v.bounds) { this.renderBounds(); r.bounds = v.bounds; }
    this.applyLayerVisibility();
  }

  applyLayerVisibility() {
    const L = this.layers;
    this.groups.components.visible = L.components;
    this.groups.probe.visible = L.components;
    this.groups.ports.visible = L.ports;
    this.groups.wires.visible = L.wires;
    this.groups.voxels.visible = L.voxels;
    this.groups.search.visible = L.search && this.hasSearch;
    this.groups.congestion.visible = L.congestion;
    this.groups.grid.visible = L.grid;
    this.groups.bounds.visible = L.bounds;
    this.groups.labels.visible = L.labels;
    this.groups.labels.children.forEach((o) => { o.visible = L.labels; });
  }

  renderComponents() {
    disposeGroup(this.groups.components);
    disposeGroup(this.groups.labels);
    disposeGroup(this.groups.ports);
    const selectedId = this.selection && this.selection.kind === 'instance' ? this.selection.id : null;
    const portIn = new THREE.MeshBasicMaterial({ color: PORT_IN });
    const portOut = new THREE.MeshBasicMaterial({ color: PORT_OUT });
    const stubs = [];
    this.state.placed.forEach((origin, id) => {
      const inst = this.instances.get(id);
      const comp = inst ? this.components.get(inst.component) : null;
      if (!comp) return;
      const d = comp.dim;
      const geometry = new THREE.BoxGeometry(d[0] * 0.96, d[1] * 0.96, d[2] * 0.96);
      const color = CATEGORY_COLORS[comp.category] || 0x888888;
      const material = new THREE.MeshStandardMaterial({
        color: color, transparent: true, opacity: 0.82,
        emissive: id === selectedId ? 0x335577 : 0x000000,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(origin[0] + d[0] / 2, origin[1] + d[1] / 2, origin[2] + d[2] / 2);
      mesh.userData = { kind: 'instance', id: id };
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        new THREE.LineBasicMaterial({ color: id === selectedId ? 0xffffff : 0xdfe7f0, transparent: true, opacity: id === selectedId ? 1 : 0.35 }),
      );
      mesh.add(edges);
      this.groups.components.add(mesh);

      const div = document.createElement('div');
      div.className = 'label';
      div.textContent = '#' + id + (inst.label ? ' ' + inst.label : '') + ' · ' + comp.name;
      const label = new CSS2DObject(div);
      label.position.set(origin[0] + d[0] / 2, origin[1] + d[1] + 0.35, origin[2] + d[2] / 2);
      this.groups.labels.add(label);

      comp.ports.forEach((p) => {
        const pin = [origin[0] + p.offset[0], origin[1] + p.offset[1], origin[2] + p.offset[2]];
        const n = p.normal;
        const face = center(pin).add(new THREE.Vector3(n[0] * 0.5, n[1] * 0.5, n[2] * 0.5));
        const sphere = new THREE.Mesh(this.portGeometry, p.direction === 'in' ? portIn : portOut);
        sphere.position.copy(face);
        sphere.userData = { kind: 'instance', id: id };
        this.groups.ports.add(sphere);
        const outward = center([pin[0] + n[0], pin[1] + n[1], pin[2] + n[2]]);
        stubs.push(face.x, face.y, face.z, outward.x, outward.y, outward.z);
      });
    });
    if (stubs.length) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(stubs, 3));
      this.groups.ports.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xc8d2dc })));
    }
  }

  renderProbe() {
    disposeGroup(this.groups.probe);
    const probe = this.state.probe;
    if (!probe) return;
    const inst = this.instances.get(probe.instance);
    const comp = inst ? this.components.get(inst.component) : null;
    if (!comp) return;
    const d = comp.dim;
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(d[0], d[1], d[2])),
      new THREE.LineBasicMaterial({ color: probe.rejected ? 0xff4d4d : 0xffe066 }),
    );
    box.position.set(probe.origin[0] + d[0] / 2, probe.origin[1] + d[1] / 2, probe.origin[2] + d[2] / 2);
    this.groups.probe.add(box);
  }

  instanced(count, material) {
    const mesh = new THREE.InstancedMesh(this.unitBox, material, count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    return mesh;
  }

  renderWires() {
    disposeGroup(this.groups.wires);
    disposeGroup(this.groups.voxels);
    const selectedNet = this.selection && this.selection.kind === 'net' ? this.selection.id : null;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    this.state.routes.forEach((route, netId) => {
      const net = this.nets.get(netId);
      const color = netColor(net);
      const t = wireThickness(net);
      const segments = [];
      const dots = [];
      const cells = new Map();
      if (route.root) cells.set(route.root.join(','), route.root);
      route.branches.forEach((b) => {
        if (b.path.length === 1) dots.push(b.path[0]);
        b.path.forEach((c, k) => {
          cells.set(c.join(','), c);
          if (k > 0) segments.push([b.path[k - 1], c]);
        });
      });
      if (!segments.length && !dots.length && route.root) dots.push(route.root);
      const count = segments.length + dots.length;
      if (count) {
        const material = new THREE.MeshStandardMaterial({
          color: color, emissive: color.clone().multiplyScalar(netId === selectedNet ? 0.8 : 0.25),
          transparent: route.partial, opacity: route.partial ? 0.6 : 1,
        });
        const mesh = this.instanced(count, material);
        let i = 0;
        segments.forEach((s) => {
          const a = center(s[0]);
          const b = center(s[1]);
          const mid = a.clone().add(b).multiplyScalar(0.5);
          const scale = new THREE.Vector3(
            s[0][0] !== s[1][0] ? 1 + t : t,
            s[0][1] !== s[1][1] ? 1 + t : t,
            s[0][2] !== s[1][2] ? 1 + t : t,
          );
          m.compose(mid, q, scale);
          mesh.setMatrixAt(i++, m);
        });
        dots.forEach((c) => {
          m.compose(center(c), q, new THREE.Vector3(t * 1.8, t * 1.8, t * 1.8));
          mesh.setMatrixAt(i++, m);
        });
        mesh.userData = { kind: 'net', id: netId };
        this.groups.wires.add(mesh);
      }
      if (this.layers.voxels && cells.size) {
        const vm = this.instanced(cells.size, new THREE.MeshStandardMaterial({ color: color, transparent: true, opacity: 0.18, depthWrite: false }));
        let i = 0;
        cells.forEach((c) => {
          m.compose(center(c), q, new THREE.Vector3(0.94, 0.94, 0.94));
          vm.setMatrixAt(i++, m);
        });
        vm.userData = { kind: 'net', id: netId };
        this.groups.voxels.add(vm);
      }
    });
    const ripped = this.state.ripped;
    if (ripped && ripped.cells.length) {
      const mesh = this.instanced(ripped.cells.length, new THREE.MeshBasicMaterial({ color: 0xff3b3b, transparent: true, opacity: 0.45, depthWrite: false }));
      ripped.cells.forEach((c, i) => {
        m.compose(center(c), q, new THREE.Vector3(0.5, 0.5, 0.5));
        mesh.setMatrixAt(i, m);
      });
      this.groups.wires.add(mesh);
    }
    this.state.failed.forEach((f) => {
      const marker = new THREE.Mesh(this.unitBox, new THREE.MeshBasicMaterial({ color: 0xff0000, wireframe: true }));
      marker.position.copy(center(f.goal));
      this.groups.wires.add(marker);
    });
  }

  renderSearch() {
    disposeGroup(this.groups.search);
    const cells = this.state.search;
    if (!cells.length) return;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3(0.42, 0.42, 0.42);
    const mesh = this.instanced(cells.length, new THREE.MeshBasicMaterial({ color: 0x3fd8ff, transparent: true, opacity: 0.28, depthWrite: false }));
    cells.forEach((c, i) => {
      m.compose(center(c), q, s);
      mesh.setMatrixAt(i, m);
    });
    this.groups.search.add(mesh);
    const head = new THREE.Mesh(this.unitBox, new THREE.MeshBasicMaterial({ color: 0xffffff }));
    head.scale.set(0.5, 0.5, 0.5);
    head.position.copy(center(cells[cells.length - 1]));
    this.groups.search.add(head);
  }

  renderCongestion() {
    disposeGroup(this.groups.congestion);
    const cells = this.state.congestion;
    if (!cells.length) return;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const maxHistory = cells.reduce((a, c) => Math.max(a, c.history), 0) || 1;
    const mesh = this.instanced(cells.length, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55, depthWrite: false }));
    const color = new THREE.Color();
    cells.forEach((c, i) => {
      const k = Math.min(1, 0.55 + 0.15 * c.occupancy);
      m.compose(center(c.coord), q, new THREE.Vector3(k, k, k));
      mesh.setMatrixAt(i, m);
      color.setHSL(0.08 * (1 - c.history / maxHistory), 1, 0.5);
      mesh.setColorAt(i, color);
    });
    mesh.userData = { kind: 'congestion' };
    this.groups.congestion.add(mesh);
  }

  renderBounds() {
    disposeGroup(this.groups.bounds);
    disposeGroup(this.groups.grid);
    const s = this.state;
    const addBox = (b, color) => {
      const box = new THREE.Box3(
        new THREE.Vector3(b.min[0], b.min[1], b.min[2]),
        new THREE.Vector3(b.max[0] + 1, b.max[1] + 1, b.max[2] + 1),
      );
      this.groups.bounds.add(new THREE.Box3Helper(box, color));
    };
    if (s.bounds) addBox(s.bounds, 0xffffff);
    if (s.searchBounds) addBox(s.searchBounds, 0x2fbf71);
    const ref = s.searchBounds || s.bounds || this.finalBounds();
    if (ref) {
      const sx = ref.max[0] - ref.min[0] + 1;
      const sz = ref.max[2] - ref.min[2] + 1;
      const size = Math.max(sx, sz) + 2;
      const grid = new THREE.GridHelper(size, size, 0x3a4756, 0x232c36);
      grid.position.set(ref.min[0] + sx / 2, 0, ref.min[2] + sz / 2);
      this.groups.grid.add(grid);
    }
  }

  finalBounds() {
    const f = this.trace.final;
    return f && f.design_bounds ? f.design_bounds : null;
  }

  fit(view) {
    const b = this.finalBounds() || this.state.bounds || { min: [0, 0, 0], max: [8, 8, 8] };
    const lo = new THREE.Vector3(b.min[0], b.min[1], b.min[2]);
    const hi = new THREE.Vector3(b.max[0] + 1, b.max[1] + 1, b.max[2] + 1);
    const mid = lo.clone().add(hi).multiplyScalar(0.5);
    const size = Math.max(4, hi.clone().sub(lo).length());
    const d = size * 1.1;
    const dirs = {
      fit: [0.9, 0.8, 1.1], iso: [0.9, 0.8, 1.1], top: [0, 1, 0.0001], front: [0, 0.15, 1], side: [1, 0.15, 0],
    };
    const v = dirs[view] || dirs.iso;
    const dir = new THREE.Vector3(v[0], v[1], v[2]).normalize();
    this.camera.position.copy(mid.clone().add(dir.multiplyScalar(d)));
    this.controls.target.copy(mid);
    this.camera.near = Math.max(0.05, size / 500);
    this.camera.far = size * 50;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  // -- picking / info ---------------------------------------------------------

  pick(ev) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((ev.clientX - rect.left) / rect.width) * 2 - 1,
      -((ev.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(mouse, this.camera);
    const targets = [];
    if (this.layers.wires) targets.push.apply(targets, this.groups.wires.children);
    if (this.layers.voxels) targets.push.apply(targets, this.groups.voxels.children);
    if (this.layers.components) targets.push.apply(targets, this.groups.components.children);
    if (this.layers.ports) targets.push.apply(targets, this.groups.ports.children);
    const hits = this.raycaster.intersectObjects(targets, false);
    const hit = hits.find((h) => h.object.userData && (h.object.userData.kind === 'instance' || h.object.userData.kind === 'net'));
    this.select(hit ? hit.object.userData : null);
  }

  select(sel) {
    this.selection = sel;
    this.bump('placed', 'routes');
    this.sync();
    this.showSelection();
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
    if (sel.kind === 'instance') {
      const inst = this.instances.get(sel.id);
      const comp = this.components.get(inst.component);
      const origin = this.state.placed.get(sel.id);
      const rows = [
        ['instance', '#' + inst.id], ['label', inst.label], ['component', comp.name],
        ['family', comp.family + ' (' + comp.category + ')'], ['op', comp.op],
        ['dim', comp.dim.join(' × ')], ['origin', origin ? fmtCoord(origin) : 'unplaced'],
        ['latency', comp.latency === null ? 'unknown' : comp.latency + ' ticks'],
        ['stateful', comp.stateful ? 'yes' : 'no'], ['init', inst.init],
      ];
      if (comp.peripheral) rows.push(['peripheral', comp.peripheral.kind + ' (' + comp.peripheral.direction + ')']);
      if (comp.signatures.length) rows.push(['implements', comp.signatures.join('; ')]);
      box.appendChild(table(rows));
      const ports = comp.ports.map((p) => [
        p.name, p.direction + ' ' + p.type.name + ' ' + p.face + ' @' + fmtCoord(p.offset),
      ]);
      const h = document.createElement('h2');
      h.textContent = 'ports';
      box.appendChild(h);
      box.appendChild(table(ports));
      const nets = [];
      this.nets.forEach((n) => {
        if (n.driver.instance === inst.id) nets.push(['drives', 'net ' + n.id + ' (' + n.type.name + ')']);
        n.sinks.forEach((s) => {
          if (s.instance === inst.id) nets.push([s.port, 'net ' + n.id + ' (' + n.type.name + ')']);
        });
      });
      if (nets.length) {
        const h2 = document.createElement('h2');
        h2.textContent = 'nets';
        box.appendChild(h2);
        box.appendChild(table(nets));
      }
    } else {
      const net = this.nets.get(sel.id);
      const route = this.state.routes.get(sel.id);
      const describe = (t) => {
        const inst = this.instances.get(t.instance);
        return '#' + t.instance + (inst && inst.label ? ' ' + inst.label : '') + '.' + t.port;
      };
      let length = null;
      if (route) {
        const cells = new Set();
        if (route.root) cells.add(route.root.join(','));
        route.branches.forEach((b) => b.path.forEach((c) => cells.add(c.join(','))));
        length = cells.size + ' cells' + (route.partial ? ' (partial)' : '');
      }
      box.appendChild(table([
        ['net', net.id], ['role', net.role], ['type', net.type.name],
        ['encoding', net.layout.encoding + ' (' + net.layout.lane_count + ' × ' + net.layout.lane_width + '-bit lanes)'],
        ['fanout', net.fanout], ['route', length || 'not routed'], ['driver', describe(net.driver)],
        ['sinks', net.sinks.map(describe).join(', ')],
      ]));
    }
  }

  // -- UI -------------------------------------------------------------------

  initUI() {
    const t = this.trace;
    const f = t.final || {};
    $('meta').textContent = t.design.instances.length + ' instances, ' + t.design.nets.length + ' nets, ' +
      this.events.length + ' events (' + t.trace_level + '); ' +
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
    $('speed').onchange = () => { this.speed = Number($('speed').value); };

    const attempts = $('attempt-select');
    attempts.innerHTML = '';
    this.restarts.forEach((index) => {
      const o = document.createElement('option');
      o.value = String(index + 1);
      o.textContent = 'attempt ' + this.events[index].attempt + ' (event ' + index + ')';
      attempts.appendChild(o);
    });
    attempts.onchange = () => { this.setPlaying(false); this.seek(Number(attempts.value)); };
    const iterations = $('iteration-select');
    iterations.innerHTML = '';
    this.iterations.forEach((it) => {
      const o = document.createElement('option');
      o.value = String(it.index + 1);
      o.textContent = 'attempt ' + it.attempt + ' · iteration ' + it.iteration;
      iterations.appendChild(o);
    });
    iterations.onchange = () => { this.setPlaying(false); this.seek(Number(iterations.value)); };

    document.querySelectorAll('[data-layer]').forEach((box) => {
      box.onchange = () => {
        this.layers[box.dataset.layer] = box.checked;
        if (box.dataset.layer === 'voxels') this.bump('routes');
        this.sync();
      };
    });
    $('layer-search').className = this.hasSearch ? '' : 'absent';
    $('layer-congestion').className = this.hasCongestion ? '' : 'absent';
    document.querySelectorAll('[data-view]').forEach((b) => {
      b.onclick = () => this.fit(b.dataset.view);
    });

    const legend = $('legend');
    legend.innerHTML = '';
    Object.keys(CATEGORY_COLORS).forEach((k) => {
      const row = document.createElement('div');
      const sw = document.createElement('span');
      sw.className = 'swatch';
      sw.style.background = '#' + CATEGORY_COLORS[k].toString(16).padStart(6, '0');
      row.appendChild(sw);
      row.appendChild(document.createTextNode(k));
      legend.appendChild(row);
    });

    this.onKey = (ev) => {
      if (ev.target && (ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT')) return;
      if (ev.key === ' ') { ev.preventDefault(); $('btn-play').onclick(); }
      else if (ev.key === 'ArrowRight') $('btn-fwd').onclick();
      else if (ev.key === 'ArrowLeft') $('btn-back').onclick();
      else if (ev.key === 'Home') $('btn-start').onclick();
      else if (ev.key === 'End') $('btn-end').onclick();
    };
    window.addEventListener('keydown', this.onKey);
  }

  setPlaying(playing) {
    this.playing = playing;
    this.accum = 0;
    $('btn-play').textContent = playing ? '⏸ Pause' : '▶ Play';
  }

  updateUI() {
    const n = this.events.length;
    $('timeline').value = String(this.position);
    $('position').textContent = this.position + ' / ' + n + ' events';
    const s = this.state;
    const lines = [s.status];
    if (s.attempt !== null) lines.push('attempt ' + s.attempt + (s.iteration !== null ? ' · iteration ' + s.iteration : '') +
      (s.presentFactor !== null ? ' · present factor ' + Number(s.presentFactor).toFixed(2) : ''));
    lines.push(s.placed.size + ' placed · ' + s.routes.size + ' routed' +
      (s.congestion.length ? ' · ' + s.congestion.length + ' congested cells' : ''));
    $('status').textContent = lines.join('\n');
    const e = s.lastEvent;
    let text = e ? JSON.stringify(e, null, 1) : '-';
    if (text.length > 4000) text = text.slice(0, 4000) + '\n...';
    $('event').textContent = text;
    if (this.selection) this.showSelection();
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
  try {
    viewer = new Viewer(trace);
  } catch (err) {
    showError(['viewer error: ' + err.message]);
    throw err;
  }
}

$('file-input').addEventListener('change', (ev) => {
  const file = ev.target.files[0];
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
  showError(['No trace embedded. Use "Trace file" to load a .pnr.json or .pnr.jsonl file.']);
}
