# The `physical-primitive` backend

`physical-primitive` is a second, independent Minecraft physical backend. Where
the coarse `physical` backend maps each IR operation onto one hand-made wide
component (an 8-bit adder cell, a bus-valued net), this backend lowers
**everything** to one-bit primitive redstone logic:

```text
RedC source
  -> redc.ir.Graph                        (unchanged, target-neutral, validated)
  -> PrimitiveNetlist                      primitive synthesis: bit blasting
  -> PrimitivePhysicalNetlist              Minecraft primitive technology mapping
  -> PlacedPrimitiveDesign                 placement
  -> RoutedPrimitiveDesign                 single-bit routing
  -> LegalizedPrimitiveDesign              redstone electrical legalization
  -> MinecraftPhysicalDesign               materialized blocks (redc.minecraft, backend-neutral)
  -> clock-tree balancing, static timing analysis, automatic clock period
  -> redstone simulation at that clock     (validated against the PrimitiveSimulator)
  -> redc.physical-primitive.v1 JSON       (+ redc.physical-primitive.pnr.v1 replay trace)
```

A design only succeeds when routing, legalization, independent verification,
**timing closure** (zero clock skew, setup and hold) and the **redstone
simulation check** all pass: one logical cycle is one physical clock period.
See [minecraft-timing.md](minecraft-timing.md) and
[minecraft-simulator.md](minecraft-simulator.md).

No handcrafted `uint8_adder`, `uint32_comparator` or `uint16_multiplier` cell is
needed: an adder is AND/OR/XOR gates, a divider is comparators, subtractors
and muxes made of AND/OR/XOR/NOT gates. Circuits get enormous. That is the
point. Priorities: correctness, clean compiler seams, universal lowering,
determinism, traceability, physical legality — compactness later.

## Usage

```bash
uv run redc pnr examples/uint8_add.redc --top add --backend physical-primitive
# build/uint8_add.primitive.pnr.json       replay trace (redc.physical-primitive.pnr.v1)
# build/uint8_add.primitive.pnr.html       3D replay viewer
# build/uint8_add.primitive.physical.json  final design (redc.physical-primitive.v1)

uv run redc pnr examples/uint8_add.redc --top add                    # coarse backend (default)
uv run redc physical-backends                                         # list physical backends

uv run redc dump-netlist examples/uint8_alu.redc --backend physical-primitive               # one-bit netlist
uv run redc dump-netlist examples/uint8_alu.redc --backend physical-primitive --stage mapped # tech-mapped
uv run redc dump-primitive-netlist examples/uint8_alu.redc                                   # shorthand
```

`--trace PATH` (`.json` or `.jsonl`), `--trace-level none|basic|detailed|search`,
`--no-viewer`, `--spacing`, `--routing-margin`, `--max-route-iterations` and
`--max-attempts` work with both backends. `--channel-width` (free blocks
between placement columns), `--max-height` (highest block y routes may use) and
`--interface default|pads` are primitive-only, as are the timing options
`--clock-period RT` (default: automatic; a given period is checked, never
raised), `--clock-margin RT` (added to the automatic period, default 1) and
`--max-clock-skew RT` (default 0); `--grid-height` and
`--layer-gap` are coarse-only. Passing an option the selected backend does not
understand is an error, as is an unknown `--backend` (never a silent
fallback). The trace is written even when P&R fails; the command then exits 1
and writes no design file.

Unlike the coarse backend, every IR width 1–64 is supported (`uint3`, `int24`,
...): everything becomes bits.

## Python API

```python
from redc import compile_source
from redc.physical_primitive import (
    synthesize_to_primitives, PrimitiveSimulator,       # seam 2-4: logic
    map_primitives_to_minecraft,                         # seam 5: technology
    place_and_route_primitive, PrimitivePnRConfig,       # seams 6-8: physical
    PrimitiveTraceRecorder, place_and_route_graph,       # the whole pipeline
)

graph = compile_source("uint8 add(uint8 a, uint8 b) { return a + b; }", top="add")
netlist = synthesize_to_primitives(graph)            # PrimitiveNetlist
assert PrimitiveSimulator(netlist).evaluate(in_a=200, in_b=100) == graph.evaluate(in_a=200, in_b=100)
mapped = map_primitives_to_minecraft(netlist)        # PrimitivePhysicalNetlist (unplaced)
result = place_and_route_primitive(mapped, PrimitivePnRConfig())
design = result.to_design_dict()                     # the final physical truth
```

## Package map (`src/redc/physical_primitive/`)

| Module | Phase | Knows about |
| --- | --- | --- |
| `netlist.py` | data model | one-bit primitives, nets, ports, buses, provenance, hierarchy |
| `synthesis/builder.py` | synthesis | the only way to emit primitives (ids, nets, provenance scopes) |
| `synthesis/logic.py` | synthesis | bitwise maps, balanced reductions, the 2:1 mux structure |
| `synthesis/arithmetic.py`, `compare.py`, `shift.py`, `cast.py` | synthesis | ripple adders, array multiplier, restoring divider, comparators, barrel shifters, casts |
| `synthesis/registry.py`, `recipes.py` | synthesis | recipes keyed by exact `OperationSignature` |
| `synthesis/interface.py` | synthesis | ports as pads or one peripheral with one-bit pins |
| `synthesis/lower.py` | synthesis | the four-pass `Graph -> PrimitiveNetlist` driver |
| `simulate.py` | oracle | bit-parallel simulation vs `Graph.evaluate/step/run` |
| `technology/` | tech mapping | placeholder Minecraft cells: sparse voxels, pins, keep-outs |
| `techmap.py`, `physical.py` | tech mapping | which cell realizes each primitive (never where) |
| `geometry.py`, `redstone.py` | physical | block coordinates, rotations, the redstone legality model |
| `grid.py` | physical | sparse block occupancy, negotiated routing claims, conflicts |
| `pnr/placement.py` | placement | instance -> origin + orientation |
| `pnr/search.py`, `pnr/routing.py` | routing | redstone-aware A*, directed trees, PathFinder negotiation |
| `pnr/legalize.py` | legalization | signal strength, repeater insertion |
| `pnr/verify.py` | verification | independent re-check of every geometric/electrical rule |
| `materialize.py` | physical | the legalized design as a backend-neutral `MinecraftPhysicalDesign` |
| `pnr/clock.py` | timing | clock-tree arrivals and tree-aware physical balancing |
| `pnr/timing.py` | timing | balancing -> materialize -> STA / closure -> redstone simulation validation |
| `technology/structures.py` | tech mapping | reference block-level cells (NOT, OR, pads, ...), characterized by simulation |
| `pnr/design.py`, `pnr/trace.py`, `pnr/records.py` | output | attempts, metrics, final JSON, replay trace |
| `backend.py` | CLI | the pipeline behind `--backend physical-primitive` |

Shared, target-neutral abstractions moved out of the coarse backend (re-exported
there unchanged): `redc.signature.OperationSignature` and
`redc.tracing.TraceLevel`. Importing `redc.physical_primitive` never imports
`redc.physical`. The physical-backend registry is `redc.physical_backends`,
deliberately separate from the `redc build` output-backend registry
(`redc.backends`): an output backend is one `Graph -> text` function, a
physical backend is a multi-phase, multi-artifact pipeline.

## Synthesis

The fundamental invariant: after synthesis every runtime computation is one-bit
AND/OR/XOR/NOT gates, one-bit `REGISTER_BIT`s, constants and boundaries, joined
by one-bit nets. `PrimitiveNetlist.validate()` enforces it, and there is no
width field anywhere — a multi-bit net cannot even be expressed. Wide values
exist only as `LogicalPort` / `LogicalBus` metadata (LSB first). Fanout is one
net with many sinks.

Lowering consumes `graph.live_nodes()` of the validated graph and runs four
passes because register `next`/`enable` back-edges may point forward:

* **A** — every input becomes `width` source bits (pads, or one peripheral);
  every register becomes `width` `REGISTER_BIT`s whose `q` is readable at once.
* **B** — constants are bit-blasted (one `CONST0`/`CONST1` per IR node and
  value) and every operation is synthesized in IR order by the recipe its
  exact `OperationSignature` resolves to.
* **C** — output ports observe their node's bits.
* **D** — each register bit gets `d = MUX(enable, next[i], q[i])` (the IR's
  enable expressed as ordinary gates; the state primitive stays minimal:
  `d`, `clk`, `rst`, `q`, `init`) and joins the single clock and reset nets.

Recipes are v1-obvious structures: ripple-carry add; subtract as
`a + ~b + 1` through the same adder; negate as `~a + 1`; partial-product
multiply (correct for signed and unsigned: the low W bits of the product are
identical); one shared unrolled restoring `unsigned_divmod` for `div` and `mod`
with sign handling via conditional two's complement (divide-by-zero forced to
0, remainder follows the dividend); MSB-first comparator chains (signed via a
`signs_differ` mux); staged barrel shifters with a 64-bit `amount >= W`
comparison against the constant W (the shift amount is always `uint64`);
OR-reduction for `-> bool` casts and pure rewiring for other casts. There is
deliberately no Boolean simplification, structural hashing or CSE in v1; those
are later `PrimitiveNetlist -> PrimitiveNetlist` passes.

Every primitive carries provenance — source IR node (whose op and type are in
`ir_nodes`), role (`propagate_xor`, `carry_generate`, `restoring_compare`, ...),
bit index and hierarchy group — so a viewer can say "this AND gate came from IR
node 42, a uint8 add, bit 5, carry-generation logic" and highlight every gate
of an IR node, of one adder bit slice, or every wire of logical bus `in_a`.

`PrimitiveSimulator` is the correctness oracle: bit-parallel (every signal is a
Python integer carrying one bit per test vector), it checks synthesis against
`Graph.evaluate` / `step` / `run` exhaustively for small widths before any
geometry exists.

## Technology mapping

`map_primitives_to_minecraft` picks a `PrimitiveCell` for every primitive
(v1: the first candidate; the `CellSelector` hook allows smarter choices) and
assigns no coordinates. Mapping is one-to-one, so mapped instance ids equal
logical primitive ids; `realizes` is the seam for future many-to-one fusion.

**Every cell in the default library is a placeholder** (`placeholder: true`,
`structure: null`): honest, deterministic footprints sized like compact
redstone gates — sparse occupied voxels, pin endpoints with facing and drive
strength, keep-out voxels — but not verified in-game circuits. All cells share
conventions: a base layer at local y=0 (pins rest on it), signal level y=1,
inputs on the local west face, outputs east, different pins at least two blocks
apart.

## Placement, routing, legalization

One coordinate is one Minecraft block. The grid is sparse (dicts keyed by used
blocks), never a world-sized array.

**Placement** is hierarchical and uses provenance only as a locality hint.
Every primitive belongs to the block of the IR node that generated it (all
sources form one `inputs` block, all outputs one `outputs` block). Blocks are
ordered along +x by IR depth (super-columns; register outputs cut loops,
constants sit just before their first consumer, right-aligned), and inside a
block primitives are levelized by their in-block depth (local columns). Every
column is separated by a routing channel. Along z, blocks stack by the
barycentre of their drivers, local columns order cells by barycentre and then
bit index (so bit slices stay together and sibling slices stay ordered), and
the `inputs` block interleaves ports bit by bit (`a[0], b[0], a[1], ...`).
Slots are probed along z with a legality check that respects bodies,
keep-outs, pins and pin approaches. Keeping each IR operation compact is what
stops a wide design from degenerating into a few enormous columns of
unrelated gates.

**Routing** treats every one-bit net separately. A net is a rooted directed
tree grown branch by branch by a multi-source weighted A* (`f = g + 1.3·h`,
like VPR's `astar_fac`) over redstone moves (four level steps and eight
one-up/one-down staircase steps; no vertical move) and negotiated with
PathFinder: temporary sharing is allowed, priced by present congestion and
history, conflicting nets are ripped up and rerouted until no conflict
remains, and failed attempts retry with wider spacing. While one branch is
searched, the approach corridors of the net's still-unreached sinks are
reserved so its own trunk cannot wall them in; a branch that exhausts its
(distance-scaled) expansion budget under congestion pricing is searched again
ignoring congestion, and a net that still walls in a sink is re-routed with
that sink first. Every branch draws on one effort budget across its retries.

A doomed attempt is given up early and retried with wider spacing, by three
deterministic, configurable rules that never change what counts as legal:
`diverged` (conflicts climb above 4x the fewest seen plus 50 -- typically a
congestion-ignoring fallback cascading), `stagnated` (no new fewest-conflicts
for 8 iterations) and `effort_exhausted` (total A* expansions above
max(5M, 60k per net)). Each abort is a `routing_aborted` trace event, and the
per-attempt effort (searches, expansions, fallbacks, restarts, per-iteration
and per-net counts) is reported in `metrics.routing.effort`.

The **redstone model** (`redstone.py`) is explicit and conservative:

1. no signal block of another net in a signal block's 12-block neighbourhood
   (4 horizontal + 8 diagonal up/down) — and within one net, touching signal
   blocks are exactly the tree's parent/child pairs, so the electrical graph IS
   the routing tree (no shorts, no self-loops, no repeater latches);
2. every signal block rests on a support block owned by the route (pins rest
   on their cell);
3. a staircase step requires the block above the lower dust to stay air;
4. so crossings use at least two blocks of vertical separation;
5. repeaters only on straight level segments;
6. no route signal or support inside a cell's keep-out; pins are entered and
   left only along their facing.

**Legalization** walks each directed tree: 15 at the driver (13 after a
placeholder OR gate), minus one per dust block, 15 again after a repeater, and
inserts repeaters as far downstream as possible on eligible blocks so every
dust block and sink stays powered. A repeater on a shared trunk refreshes every
downstream branch. A net that cannot be legalized is ripped up, penalized and
rerouted. Repeaters are route elements — never netlist components.

**Verification** (`verify.py`) re-derives every rule above from the final
artifacts alone (placed voxels and realized routes), including an independent
signal-strength recomputation, so a design that passes is legal under the
model regardless of the router's internal state.

The legality rules above are geometric. The *behaviour* of the routed blocks
is then checked by the redstone simulator, which derives connectivity from the
block layout alone. Not modelled (documented limitations): quasi-connectivity,
comparators, Java's tick priorities and block-update order inside one game
tick, and the real behaviour of the placeholder gate internals. Placeholder
cells are simulated as ABSTRACT components with declared truth tables and
timing, and every report says `abstract-components`.

## Timing

Three notions stay separate: **state** (only register bits hold it),
**logical depth** (gates on the longest combinational path) and **physical
delay**. Physical delay is the static timing analysis of the *materialized*
world: repeater settings, torch delays and declared cell arcs, from the timing
model the simulator also uses. It decides the clock period. It never decides
which logical cycle something happens on.

After verification, every attempt:

1. measures the clock arrival at every register clock pin from the routed clock
   tree;
2. **balances** it to zero skew by raising repeater delays and turning eligible
   dust into repeaters, so every added delay is a real block state;
3. re-verifies the changed route;
4. materializes the world, then runs STA and closure: the automatic period is
   the smallest safe one plus `clock_margin_rt`, and every setup and hold slack
   must be ≥ 0;
5. drives the world in the redstone simulator at that clock and compares outputs,
   every register bit and the `done` cycle with the `PrimitiveSimulator`.

Every register clock sink gets a reserved straight **clock tap** in front of
its pin (`clock_tap_length`), so balancing always has exclusive repeater sites.
Only `timing/clock_balance` triggers another, roomier attempt. `timing/setup`,
`timing/hold`, `timing/clock_skew` and the `simulation/*` failures are
deterministic and end the run. `metrics.timing` summarizes the result; the full
report is the design's `timing` block. Placeholder cells declare their timing
(`placeholder_estimate: true`). The reference structures' gate timing is
measured by simulation.

## Outputs

* **Final design** `redc.physical-primitive.v1`: schema, backend, source, IR
  summary, library, `placeholder_geometry`, logical metadata (summary, IR
  nodes, hierarchy groups, ports, buses), technology cell definitions,
  instances (provenance, origin, orientation, absolute voxels with roles,
  keep-out, pins with their nets), nets (logical aliases, driver, sinks, the
  route tree, and the realized route: typed dust/repeater elements with
  direction and strength, supports, clearances, per-sink strength and delay),
  bounds, metrics, and the physical sign-off: `clock_tree` (balancing),
  `timing` (clock period, skew, setup/hold slacks and critical paths), `simulation`
  (mode, model, Minecraft version, validated) and `world` — the materialized
  `redc.minecraft-design.v1` block world. It is the physical truth a future
  NBT/schematic exporter consumes — never reconstructed from the trace.
* **Replay trace** `redc.physical-primitive.pnr.v1`: see
  [physical-primitive-trace.md](physical-primitive-trace.md).

## Later (by design, not implemented)

Boolean simplification, structural hashing, CSE, carry-lookahead, Wallace /
Booth multipliers, better dividers, specialized Minecraft cells and gate fusion
(through `realizes`), bus-aware placement hints, timing-driven placement and
routing, dedicated crossover cells, block-level AND / XOR / register cells
(today only NOT, OR, pads, levers, lamps and constants have reference
structures), in-game verification and NBT export.
