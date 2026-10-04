# Primitive P&R replay trace — `redc.physical-primitive.pnr.v1`

`redc pnr --backend physical-primitive` writes a replay trace of the whole
compile: bit-level synthesis, technology mapping, placement, single-bit
routing (every rip-up and congestion snapshot), and redstone electrical
legalization. It drives the bundled browser viewer today and can drive a
Minecraft mod later. It is distinct from the final design file
(`redc.physical-primitive.v1`): the design file is the physical truth, and the
trace is how the compiler got there. Never rebuild a design by replaying the
trace.

The trace is self-contained. It describes every technology cell (sparse
voxels, pins, keep-outs), every instance with its provenance, and every
one-bit net with its logical grouping, so a consumer needs neither Python nor
the technology library.

## Files

Same two forms as the coarse trace (`docs/pnr-trace.md`): `.json` (one object:
header fields + `events` + `final`) or `.jsonl` (`{"record": "header", ...}`,
then one `{"record": "event", ...}` per event, then `{"record": "final",
"final": {...}}`). Default names: `build/<name>.primitive.pnr.json` and the
viewer `build/<name>.primitive.pnr.html`.

## Conventions

- **Coordinates** are `[x, y, z]` **Minecraft block positions**: x east, y up,
  z south. One coordinate is one block. There is no tile or cell scale.
- **Bounds**: `{"min": [x,y,z], "max": [x,y,z], "dims": [dx,dy,dz], "volume": n}`, inclusive.
- **Ids** are dense per kind: instances `0..N-1`, nets `0..M-1`, hierarchy
  groups `0..G-1`. An instance id and a net id may be equal; every field says
  which it references (`"instance"` vs `"net"`). Mapped instance ids equal the
  logical primitive ids they realize (v1 technology mapping is one-to-one).
- **Bits** are LSB first everywhere (`bits[0]` is bit 0).
- **Forward compatibility**: ignore unknown event types and fields. A breaking
  change bumps the schema version.

## Header

```jsonc
{
  "schema": "redc.physical-primitive.pnr.v1",
  "generator": "redc",
  "backend": "physical-primitive",
  "trace_level": "basic",                      // none | basic | detailed | search
  "coordinate_system": {"units": "blocks", "x": "east", "y": "up", "z": "south", "cell": "..."},
  "source": {"path": "examples/uint8_add.redc", "top": "add"},
  "ir": {"live_nodes": 4, "ops": {"add": 1, "input": 2}, "sequential": false, "inputs": [...], "outputs": [...]},
  "design": {...},                             // the UNPLACED mapped design, below
  "config": {...},                             // PrimitivePnRConfig (units: blocks)
  "events": [...],
  "final": {...}
}
```

### `design`

| Field | Content |
| --- | --- |
| `library` | technology library name (`redc-primitive-placeholder-v1`) |
| `cells[]` | one entry per technology cell used (below) |
| `instances[]` | `{id, kind, category, cell, realizes, ir_node, ir_op, ir_type, role, bit, group, port, init, peripheral, origin: null, orientation: null, attrs?}` |
| `nets[]` | `{id, role, width: 1, fanout, driver: {instance, pin}, sinks: [{instance, pin}], logical: [{ir_node, bit}], ports: [{port, bit}]}` |
| `groups[]` | synthesis hierarchy `{id, name, kind, parent, ir_node, attrs?}`; `kind` is `root`, `ir_node`, `port`, `helper`, `slice`, `stage`, `iteration`, `control` |
| `ports[]` | logical ports `{name, direction, type, source_name, ir_node, realization, bits: [{instance, pin}]}` (`realization` is `pads` or a peripheral kind) |
| `buses[]` | the result of every live IR node as LSB-first one-bit signals `{ir_node, op, type, bits: [{instance, pin}]}` |
| `ir_nodes[]` | `{id, op, type, args, name?/value?/init?}` |
| `summary` | primitive counts (`gates`, `gates_by_kind`, `register_bits`, `nets`, `fanout`, ...) |

`kind` is one of `and`, `or`, `xor`, `not` (the only combinational
primitives), `register_bit`, `const0`, `const1`, `input_bit`, `output_bit`,
`clock_source`, `reset_source`, `peripheral`. `category` groups them: `gate`,
`state`, `constant`, `boundary`, `control`, `peripheral`. Provenance (`ir_node`,
`ir_op`, `ir_type`, `role`, `bit`, `group`) answers "this AND gate came from IR
node 42, a uint8 add, bit 5, carry-generation logic"; walk `groups` from
`group` up to the root for the full hierarchy path. Logical buses and ports are
**metadata only**: every net is one bit and is routed on its own.

### `design.cells[]` — technology cell definitions (local coordinates)

```jsonc
{
  "name": "xor_gate_2x3_placeholder", "kind": "xor",
  "placeholder": true, "structure": null,      // EVERY v1 cell is a placeholder
  "latency": 3, "stateful": false, "description": "...",
  "orientations": ["east", "south", "west", "north"],
  "peripheral": null,                          // or {kind, direction, width}
  "voxels": [{"coord": [0,0,0], "role": "base"}, {"coord": [1,1,0], "role": "body"}, ...],
  "keepout": [[0,1,1], ...],
  "pins": [{"name": "a", "direction": "in", "position": [0,1,0], "facing": "west", "strength": 1}, ...]
}
```

Local coordinates have the signal flowing along +x. An orientation rotates
them clockwise (seen from above) about y: `east` = identity, `south` turns local
+x to +z (`(x, y, z) -> (-z, y, x)`), and so on. A placed instance's absolute
block is `origin + rotate(local)`. `voxels` are sparse (never assume a solid
box). `keepout` blocks may hold no route signal or support block. A pin's
`position` is where the route's endpoint dust sits (resting on the cell's own
base); routes enter/leave along `facing`. Output pins state the strength they
drive there (0 = never powered, e.g. constant 0); inputs the minimum they need.

## Events

Every event has a contiguous `seq`, a `phase` (`synthesis`, `techmap`, `pnr`,
`placement`, `routing`, `congestion`, `legalization`, `finalize`) and a `type`.
**Level** is the lowest trace level that records it (`none` records no events).

| Type | Phase | Level | Payload |
| --- | --- | --- | --- |
| `synthesis_begin` | synthesis | basic | `live_nodes`, `ops`, `sequential` |
| `ir_node_synthesized` | synthesis | basic | `ir_node`, `op`, `ir_type`, `synthesis_pass` (`A` inputs/registers, `B` logic, `D` register next-state), `recipe`, `group`, `primitives`, `first_instance`, `last_instance`, `by_kind` |
| `primitive_emitted` | synthesis | detailed | `instance`, `kind`, `ir_node`, `role`, `bit`, `group`, `port?`, `attrs?` |
| `synthesis_complete` | synthesis | basic | the primitive netlist summary |
| `primitive_mapped` | techmap | detailed | `instance`, `kind`, `cell`, `candidates` |
| `techmap_complete` | techmap | basic | `library`, `instances`, `nets`, `cells: {name: count}`, `placeholder_cells`, `component_voxels` |
| `pnr_begin` | pnr | basic | `instances`, `nets`, `max_attempts` |
| `pnr_attempt_begin` | pnr | basic | `attempt`, `component_spacing`, `channel_width`, `routing_margin`, `max_y` — **resets the design**: fresh grid, nothing placed or routed |
| `placement_begin` | placement | basic | `attempt`, `component_spacing`, `channel_width`, `columns: [{column, x, instances}]`, `blocks: [{block, super_column, instances}]` (one block per IR node, plus `inputs` / `outputs`) |
| `component_place_attempt` | placement | detailed | `instance`, `origin`, `orientation`, `voxels` (absolute occupied blocks), `keepout` (count), `bounds` |
| `component_place_rejected` | placement | detailed | `instance`, `origin`, `orientation`, `reason` |
| `component_placed` | placement | basic | `instance`, `column`, `block`, `origin`, `orientation`, `bounds` |
| `design_bounds_changed` | placement | basic | a bounds object (bodies, keep-outs, pins) |
| `placement_complete` | placement | basic | `attempt`, `component_count`, `component_voxels`, `keepout_voxels`, `probes`, `bounds` |
| `placement_failed` | placement | basic | `attempt`, `reason` |
| `routing_begin` | routing | basic | `nets`, `order: [net]`, `search_bounds`, `present_factor` |
| `routing_iteration_begin` | routing | basic | `iteration`, `present_factor`, `nets: [net]` to (re)route |
| `net_route_begin` | routing | detailed | `net`, `iteration`, `driver`, `sinks` (with `coord` = pin block), `fanout` |
| `branch_route_begin` | routing | detailed | `net`, `iteration`, `sink: {instance, pin}`, `goal` |
| `route_search_expand` | routing | search | `net`, `iteration`, `sink`, `coord`, `g`, `h`, `f`, `frontier` |
| `route_transition_blocked` | routing | search | `net`, `iteration`, `sink`, `from`, `to`, `reason` (capped per branch) |
| `branch_search_stats` | routing | detailed | `net`, `iteration`, `sink`, `attempt`, `mode` (`normal`, `greedy`, `ignore_congestion`), `expansions`, `found`, `blocked: {reason: count}` — one per search attempt |
| `branch_search_relaxed` | routing | basic | `net`, `iteration`, `sink`, `mode`, `expansions` — the branch ran out of its expansion budget and is searched again in a more relaxed `mode`: `greedy` (higher A* weight, larger budget, congestion still priced), then as a last resort `ignore_congestion` (temporary sharing that later iterations push apart). A net keeps its mode for the rest of the iteration |
| `branch_path_rejected` | routing | detailed | `net`, `iteration`, `sink`, `reason`, `coord`, `path` — a found path broke an intra-net rule; the search retries avoiding `coord` |
| `branch_route_found` | routing | detailed | `net`, `iteration`, `sink`, `start`, `goal`, ordered `path`, `length` |
| `branch_route_failed` | routing | basic | `net`, `iteration`, `sink`, `goal`, `reason` (`unreachable`, `expansion_limit`, `goal_out_of_bounds`, `intra_net_conflict`, `effort_limit`), `expansions`, `message` |
| `net_route_restarted` | routing | basic | `net`, `iteration`, `first_sink`, `restart` — a branch could not be connected (the net walled in its own sink); the net is routed again with that sink first |
| `net_route_committed` | routing | basic | the net's complete route tree (below) |
| `net_rip_up` | routing | basic | `net`, `iteration`, `reason` (`congestion` or `legalization`), `cells`, `length` — remove the tree |
| `routing_iteration_end` | routing | basic | `iteration`, `present_factor`, `rerouted`, `changed`, `conflicts`, `conflict_cells`, `routed_cells`, `history_total` |
| `congestion_snapshot` | congestion | basic | `iteration`, `present_factor`, `conflicts: [{kind, cells, nets}]`, `cells: [{coord, history, nets}]`, `history_cells` — replaces the congestion layer |
| `physical_conflict` | congestion | detailed | `iteration`, `kind`, `cells`, `nets` |
| `keyframe` | routing | basic, if `keyframe_interval > 0` | `iteration`, `placement: [{instance, origin, orientation, bounds}]`, `routes: [route]`, `realized: [realized route]` (only nets whose tree survived since the last legalization), `congestion` (the cells of this iteration's snapshot) — enough to restart a replay here |
| `routing_complete` / `routing_failed` | routing | basic | `iteration`, `iterations`, `routed_nets`, `rip_ups`, `failure` |
| `legalization_begin` | legalization | basic | `round`, `nets` |
| `route_legalization_begin` | legalization | detailed | `net`, `round`, `length`, `drive` |
| `signal_strength_scan` | legalization | detailed | `net`, `round`, `powered`, `cells: [[x, y, z, strength], ...]` |
| `repeater_inserted` | legalization | basic | `net`, `round`, `coord`, `facing`, `input_strength`, `delay` |
| `legalization_failed` | legalization | basic | `net`, `round`, `reason`, `message`, `coord`, `sink` |
| `route_realized` | legalization | basic | a realized route (below) |
| `legalization_complete` | legalization | basic | `round`, `realized`, `failures`, `repeaters` |
| `illegal_transition` | legalization | basic | independent verification found a violation: `kind`, `message`, `cells`, `nets` |
| `design_finalized` | finalize | basic | `attempt`, `nets`, `dust`, `repeaters`, `supports`, `bounds` |
| `pnr_attempt_end` | pnr | basic | `attempt`, `status` (`success`/`failed`), `failure`, `metrics` |
| `pnr_end` | pnr | basic | `success`, `attempts` |

Conflict `kind`s: `shared_signal`, `shared_support`, `signal_on_support`,
`clearance_blocked`, `adjacent_signals` (two nets' signal blocks within one
another's 12-block signal neighbourhood — they would short).

### Route tree

`{net, driver: {instance, pin}, root, branches: [{sink, start, goal, path}], cells, length, fanout}`.
`root` is the driver's pin block. Each branch `path` is ordered root-to-sink:
it starts at `start` (a block already on the tree — the attachment point) and
ends at `goal` (the sink's pin block). Consecutive blocks differ by one
horizontal step and at most one block of height (a staircase). Signal flows
from `root` outward; `cells` is the union in construction order.

### Realized route

`{net, powered, elements: [{coord, kind: "dust"|"repeater", parent, strength, delay, facing?, blockstate_facing?}], repeaters, supports, clearances, sinks: [{instance, pin, coord, strength, required, repeaters, delay_ticks, distance}], min_strength, max_delay_ticks}`.
A dust element's `strength` is its signal strength (15 next to the driver,
minus one per dust block); a repeater's is its input strength and its `facing`
is the direction it outputs to — Minecraft's `minecraft:repeater[facing=...]` blockstate names the opposite (input) side, given as `blockstate_facing`. `parent` gives direction. `supports` are the
blocks under the route's dust/repeaters (pins rest on their cell) and must be full opaque
redstone conductors such as `minecraft:stone` (on glass a staircase only carries signal up);
`clearances` must stay air so staircases connect. `delay` / `delay_ticks`
count repeater redstone ticks from the root.

### Replaying

Apply events `0 .. N-1` in order:

- `pnr_attempt_begin` clears everything (placement, routes, realized routes,
  congestion). Earlier attempts stay in the trace; seeking back restarts from
  the latest `pnr_attempt_begin` (or `keyframe`) before the target.
- `component_place_attempt` / `component_place_rejected` set the current probe;
  `component_placed` sets an instance's origin and orientation.
- `net_route_begin` starts a partial tree; `branch_route_found` appends a
  branch; `net_route_committed` replaces it with the complete tree;
  `net_rip_up` deletes it (and any realized route for that net).
- `net_route_restarted` discards the net's partial tree: its branches are
  searched again from the root, and the `branch_route_failed` that caused the
  restart is not final.
- `route_search_expand` adds to the current search frontier.  Every event that
  starts or ends one search clears it: `branch_route_begin`,
  `branch_search_relaxed`, `branch_path_rejected`, `branch_route_found`,
  `branch_route_failed`, `net_route_restarted` and `net_route_committed`.
- `congestion_snapshot` replaces the congestion layer.
- `route_realized` replaces a net's realized (dust/repeater) route.
- `keyframe` replaces placement, all routes, all realized routes and the
  congestion layer, so seeking can restart from it.

After all events the state matches `final`.

## `final`

```jsonc
{
  "success": true, "attempt": 0, "attempts": 1,
  "geometry": {"attempt": 0, "component_spacing": 1, "channel_width": 6, "routing_margin": 6, "max_y": 9},
  "failure": null,      // or {stage, reason, message, attempt, attempts, net, iteration, details}
  "placement": [{"instance": 0, "origin": [x,y,z], "orientation": "east", "bounds": {...}}],
  "routes": [route, ...], "realized": [realized route, ...],
  "conflicts": [],      // remaining conflicts (failure only)
  "violations": [],     // verifier findings (verification failure only)
  "design_bounds": {...}, "search_bounds": {...},
  "metrics": {...}
}
```

A failed run still writes a complete trace with `success: false` — every
attempt that ran, its last placement, routes and remaining congestion. If
synthesis or technology mapping itself fails, the trace holds the events so far
and `final = {"success": false, "failure": {"stage": ..., "message": ...}}`.
