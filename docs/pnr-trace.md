# P&R replay trace — `redc.pnr.trace.v1`

`redc pnr` writes a replay trace of the whole place-and-route process. It
records every placement, every routing iteration, every rip-up and the
congestion between them, and it ends with the final design. The trace is meant
to drive animations: the bundled browser viewer reads it today, and the same
data can later drive a Minecraft mod that shows the circuit being built.

A trace is self-contained. It describes every component box and pin, so a
consumer never needs the RedC YAML cell library, Python, or the compiler.

## Files

| Form | Extension | Layout |
| --- | --- | --- |
| JSON | `.pnr.json` | one object: header fields + `events` + `final` |
| JSON Lines | `.pnr.jsonl` | one record per line: `{"record": "header", ...}`, then `{"record": "event", ...}` per event, then `{"record": "final", "final": {...}}` |

Both forms hold the same data. A JSONL header record carries the same fields as
the JSON form's top level, and each event record is an event with an extra
`"record": "event"`. Use JSONL for large `search`-level traces that you want to
stream.

All values are plain JSON: objects, arrays, numbers, strings, booleans and
`null`. Enumerations are lowercase strings. There are no language-specific
encodings.

## Conventions

- **Coordinates** are integer grid cells written `[x, y, z]`: x points east,
  y up, z south (`coordinate_system` in the header). Cell `[x, y, z]` is the
  unit cube `[x, x+1) × [y, y+1) × [z, z+1)`. These are abstract cells, not
  Minecraft blocks.
- **Bounds** objects look like `{"min": [x,y,z], "max": [x,y,z], "dims": [dx,dy,dz], "volume": n}`.
  `min` and `max` are inclusive.
- **Ids.** Instance and net ids are integers drawn from one shared counter, so
  they never collide. Component definitions are identified by a string `id`.
- **Forward compatibility.** Consumers must ignore event types and fields they
  don't recognise. A breaking change bumps the schema version.

## Top level

```jsonc
{
  "schema": "redc.pnr.trace.v1",
  "generator": "redc",
  "trace_level": "basic",            // none | basic | detailed | search
  "coordinate_system": {"units": "grid_cells", "x": "east", "y": "up", "z": "south", "cell": "..."},
  "design": {"components": [...], "instances": [...], "nets": [...]},
  "config": {...},                   // the PnRConfig used (grid_height, spacing, ...)
  "events": [...],
  "final": {...}
}
```

### `design.components[]` — one entry per distinct cell definition

| Field | Meaning |
| --- | --- |
| `id`, `name` | definition id (the component name, plus `#n` only if two distinct definitions share a name) and cell name |
| `family` | `operation`, `primitive_gate`, `type_cast`, `register`, `wiring`, `input_pad`, `output_pad`, `constant`, `clock_source`, `reset_source`, `peripheral` |
| `category` | display grouping: `compute`, `register`, `routing`, `boundary`, `control`, `peripheral` |
| `op` | IR operation for compute cells, else `null` |
| `dim` | `[dx, dy, dz]`: the box size in cells, taken directly from the YAML |
| `latency` | propagation delay in ticks, or `null` if unknown |
| `stateful`, `combinational` | booleans |
| `nbt` | NBT structure file name, or `null` |
| `signatures` | operation signatures implemented, as strings, e.g. `"add(uint8, uint8) -> uint8"` |
| `ports[]` | `{name, direction: "in"/"out", type: {name, width, signed, boolean}, layout: {encoding, lane_width, lane_count}, face: "east"/"west"/"top"/"bottom"/"south"/"north", normal: [nx,ny,nz], offset: [ox,oy,oz]}` |
| `peripheral` | `{kind, direction: "input"/"output"}` for peripherals, else `null` |
| `value` | constant cells only: the raw bit value |

A placed instance with origin `o` covers the cells `o + [0..dx) × [0..dy) × [0..dz)`.
A port's pin cell is `o + offset`. Its **escape cell**, where a wire attaches,
is `o + offset + normal`, one step outside the face.

### `design.instances[]`

`{id, component, label, category, init, origin}`. `component` is a
component-definition `id`. `label` is the module port name for boundary
instances (`"in_a"`, `"start"`, `"result"`), `"n<id>"` for the IR node that a
compute cell realises, and `"clk"`/`"rst"` for the global sources. `init` is a
register's reset value (raw bits), otherwise `null`. `origin` is always `null`
in the header, because the header shows the design before placement.

### `design.nets[]`

`{id, role, type, layout, fanout, driver: {instance, port}, sinks: [{instance, port}]}`.
`role` is `clock`, `reset` or `data`. Each net is one logical signal that is
routed as one path tree, even when it is a wide bus. `layout` gives its
physical lanes: bool → 1 lane; 4- and 8-bit values → binary, one 1-bit lane per
bit; 16-, 32- and 64-bit values → hex, one 4-bit lane per nibble.

## Events

Every event carries:

- `seq`: its position in the list (0, 1, 2, …, with no gaps);
- `phase`: `pnr`, `placement`, `routing`, `congestion` or `finalize`;
- `type`: the event type, listed below.

Order events by `seq`. The trace has no timestamps.

The **Level** column gives the lowest trace level that records an event. `none`
records no events. `basic` is the default. `detailed` adds placement probes and
per-branch events. `search` adds every A* expansion.

| Type | Phase | Level | Payload |
| --- | --- | --- | --- |
| `pnr_begin` | pnr | basic | `instances`, `nets`, `max_attempts` |
| `pnr_attempt_begin` | pnr | basic | `attempt`, `component_spacing`, `layer_gap`, `routing_margin` (**resets the design**: fresh grid, everything unplaced and unrouted) |
| `placement_begin` | placement | basic | `attempt`, spacing, `base_y`, `layers: [{layer, instances: [id]}]` |
| `component_place_attempt` | placement | detailed | `instance`, `origin`, `dim`, `bounds` |
| `component_place_rejected` | placement | detailed | `instance`, `origin`, `reason` |
| `component_placed` | placement | basic | `instance`, `layer`, `origin`, `dim`, `bounds` |
| `design_bounds_changed` | placement / routing | basic | a bounds object: bodies, plus routed wires once routing starts |
| `placement_complete` | placement | basic | `attempt`, `component_count`, `component_cells`, `bounds` |
| `placement_failed` | placement | basic | `attempt`, `reason` |
| `routing_begin` | routing | basic | `nets`, `order: [net]`, `search_bounds`, `present_factor` |
| `routing_iteration_begin` | routing | basic | `iteration`, `present_factor`, `nets: [net]` to (re)route |
| `net_route_begin` | routing | detailed | `net`, `iteration`, `driver`, `sinks` (endpoints carry `coord` = escape cell), `fanout` |
| `branch_route_begin` | routing | detailed | `net`, `iteration`, `sink: {instance, port}`, `goal` |
| `route_search_expand` | routing | search | `net`, `iteration`, `sink`, `coord`, `g`, `h`, `f`, `frontier` |
| `branch_route_found` | routing | detailed | `net`, `iteration`, `sink`, `start`, `goal`, ordered `path`, `length`, `expansions` |
| `branch_route_failed` | routing | basic | `net`, `iteration`, `sink`, `goal`, `reason` (`unreachable`, `expansion_limit`, `goal_blocked`, `goal_out_of_bounds`, `root_blocked`), `expansions`, `message` |
| `net_route_committed` | routing | basic | `net`, `iteration`, `root`, `branches`, `cells`, `length`, `fanout`: the net's complete tree |
| `net_rip_up` | routing | basic | `net`, `iteration`, `reason`, `cells`, `length`: remove this net's tree |
| `routing_iteration_end` | routing | basic | `iteration`, `present_factor`, `rerouted`, `changed`, `overused`, `max_occupancy`, `routed_cells`, `history_total`, `history_max` |
| `congestion_snapshot` | congestion | basic | `iteration`, `present_factor`, `stats`, `cells: [{coord, occupancy, history, nets}]`, listing only overused cells |
| `keyframe` | routing | basic, if `keyframe_interval > 0` | `iteration`, `placement: [{instance, origin}]`, `routes: [route]`: a full snapshot to seek from |
| `routing_complete` / `routing_failed` | routing | basic | `iteration`, `iterations`, `routed_nets`, `failure` |
| `routes_finalized` | finalize | basic | `attempt`, `nets`, `wire_cells`, `bounds`: wires are now permanent |
| `pnr_attempt_end` | pnr | basic | `attempt`, `status: "success"/"failed"`, `failure`, `metrics` |
| `pnr_end` | pnr | basic | `success`, `attempts` |

A **route** (in `net_route_committed`, `keyframe` and `final.routes`) has this
shape: `{net, root, branches: [{sink: {instance, port}, start, goal, path: [[x,y,z], ...]}], cells, length, fanout}`.
`root` is the driver's escape cell. Each branch `path` is ordered and
six-connected. It starts at `start`, a cell already on the tree, and ends at
`goal`, the sink's escape cell. `cells` is the union of all cells, each listed
once in construction order. The ordered paths are the canonical geometry;
`cells` exists as a convenience.

### Replaying the trace

To rebuild the state after N events, apply events `0 .. N-1` in order:

- `pnr_attempt_begin` clears everything: nothing is placed, routed or congested.
- `component_placed` sets an instance's origin.
- `net_route_begin` starts an empty, partial tree for that net, and
  `branch_route_found` appends a branch to it. These two are `detailed`-level
  only.
- `net_route_committed` replaces the net's tree with the complete one.
- `net_rip_up` deletes the net's tree.
- `congestion_snapshot` replaces the congestion layer.
- `route_search_expand` adds to the current search frontier. The frontier
  clears at the next `branch_route_*` or `net_route_committed`.
- `keyframe`, when present, replaces placement and routes wholesale.

To seek backwards, restart from the latest `pnr_attempt_begin` (or `keyframe`)
before the target event. After all events have been applied, the state matches
`final`.

## `final`

```jsonc
{
  "success": true,
  "attempt": 0, "attempts": 1,          // index of the last attempt; number of attempts made
  "geometry": {"attempt": 0, "component_spacing": 3, "layer_gap": 4, "routing_margin": 4},
  "failure": null,                       // or {stage, reason, message, attempt, attempts, net, sink, iteration, overused}
  "placement": [{"instance": 0, "origin": [x,y,z], "bounds": {...}}],
  "routes": [route, ...],                // final routes (working routes of the last attempt on failure)
  "congestion": [],                      // remaining overused cells (failure only)
  "grid": {"height": 24, "cells": [{"coord": [x,y,z], "owner": id, "kind": "component"|"pin"|"wire"|...}]},
  "design_bounds": {...}, "search_bounds": {...},
  "metrics": {"placement": {...}, "routing": {...}, "final": {...}}
}
```

A failed run still writes a full trace with `success: false`. It covers every
attempt that ran, every routing iteration, the last placed state and the
remaining congestion.

## The final design file — `redc.physical.v1`

`redc pnr` also writes `<name>.physical.json`, but only when P&R succeeds. It
is the concise final design that future NBT realization will consume:
`schema`, `coordinate_system`, `grid: {height, bounds}`, and `components` in
the same format as above. It also contains `instances` (each with its `origin`
and `bounds`), `nets` (each with its final `route`) and `metrics`. It contains
no transient search, rip-up or congestion data.

## Viewer

```bash
uv run redc pnr examples/uint8_fib.redc --top fib     # writes build/uint8_fib.pnr.{json,html} + .physical.json
uv run redc render-pnr build/uint8_fib.pnr.json        # (re)build the HTML viewer for any trace
```

Open the `.pnr.html` file in a browser. It embeds the trace and the viewer
script, and loads Three.js from the jsDelivr CDN, so it needs network access.
You can also load any other `.pnr.json` or `.pnr.jsonl` file into it with its
file picker. Controls:

- **Replay:** play/pause, step forward and back, a scrubber, and speed.
- **Jump to:** an attempt, a routing iteration, or the final result.
- **Layers:** components, labels, ports, wires (with wire voxels), search,
  congestion, grid, and bounds.
- **Camera:** orbit, pan and zoom, plus fit/iso/top/front/side presets.
- **Inspect:** click a box or a wire to see its details.
