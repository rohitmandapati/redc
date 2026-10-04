# Backend-neutral Minecraft circuits and the redstone simulator

`redc.minecraft` is the layer every RedC Minecraft backend lowers into once a
circuit is physical. It knows blocks, coordinates, block states, ports and
explicitly abstract components. It does **not** know RedC IR operations,
primitive synthesis recipes, primitive kinds or any backend's component
classes, and it never imports `redc.physical_primitive` or `redc.physical`.

```text
            redc.ir.Graph
                  |
        +---------+---------+
        |                   |
  coarse physical     primitive physical          (backends)
     backend               backend
        |                   |  materialize_design()
        +---------+---------+
                  v
       MinecraftPhysicalDesign                    redc.minecraft.design
                  |
        +---------+---------+
        v                   v
  RedstoneSimulator   static timing analysis      redc.minecraft.simulator / .sta
        \____ one connectivity derivation (connectivity.py)
              and one timing model (timing.py) shared by both ____/
                  |
           timing closure  ->  clocked validation  redc.minecraft.closure / .testbench
```

The primitive backend is the first producer
(`redc.physical_primitive.materialize`). The coarse backend can adopt the same
representation later without any change here.

| Module | What it holds |
| --- | --- |
| `units.py` | game ticks (gt) vs redstone ticks (rt, 1 rt = 2 gt); explicit conversions only |
| `blocks.py` | `MinecraftBlock(id, state)` plain data, directions, rotation |
| `design.py` | `MinecraftPhysicalDesign`, `AbstractComponent`, `Port`, `PortBit`, `Probe`; JSON schema `redc.minecraft-design.v1`; `fingerprint()` |
| `behaviors.py` | the behaviour registry: one class per supported block id |
| `connectivity.py` | `compile_world`: the element graph derived from geometry |
| `abstract.py` | the ABSTRACT component adapters (combinational, dff) |
| `simulator.py` | `RedstoneSimulator`: the event engine |
| `timing.py` | the shared timing model: delays, arcs, sequential timing |
| `sta.py`, `closure.py` | static timing analysis, period selection, setup and hold |
| `testbench.py` | clocked dynamic validation |
| `characterize.py` | measuring a block-level cell's function and timing by simulation |
| `playback.py` | a compact recording for viewer animation |

## `MinecraftPhysicalDesign` (`redc.minecraft-design.v1`)

```jsonc
{
  "schema": "redc.minecraft-design.v1",
  "source_backend": "physical-primitive",
  "minecraft_version": "java-1.20 (redc-redstone-sim-v1 subset)",
  "timing_model": "redc-redstone-timing-v1",
  "mode": "abstract-components",            // or "block-accurate"
  "palette": [{"id": "minecraft:stone"}, {"id": "minecraft:repeater", "state": {"delay": 2, "facing": "west"}}, ...],
  "blocks": [[x, y, z, palette_index], ...], // sorted; air is implicit
  "block_counts": {"minecraft:redstone_wire": 2697, ...},
  "components": [{                          // ABSTRACT black boxes (no block-level implementation yet)
    "name": "i17", "function": "combinational",          // or "dff"
    "simulation": "abstract-components",
    "pins": [{"name": "a", "direction": "in", "coord": [x,y,z], "strength": 1},   // reads the dust there
             {"name": "y", "direction": "out", "coord": [x,y,z], "strength": 15}], // injects into the dust there
    "truth_tables": {"y": "0110"},          // bit k = output for inputs where input i = bit i of k
    "init": 0,                              // dff reset value
    "timing": {"model": "...", "source": "declared", "arcs": [...], "sequential": null},
    "voxels": [[x,y,z], ...],               // redc:abstract_block markers
    "labels": {"instance": 17, "kind": "xor", ...}   // reports only
  }],
  "ports": [{"name": "in_n", "direction": "in", "role": "data", "width": 2, "signed": false,
             "bits": [{"kind": "source", "coord": [x,y,z], "strength": 15}, ...]},   // LSB first
            {"name": "clk", "direction": "in", "role": "clock", ...},
            {"name": "rst", "direction": "in", "role": "reset", ...}],
  "probes": [{"name": "reg5", "coord": [x,y,z], "threshold": 1}],   // non-intrusive observation points
  "annotations": [[x, y, z, 42], ...],      // block -> producer label (net id): REPORTS ONLY
  "metadata": {...}
}
```

Port bit kinds:

* `lever`: a real lever the test bench flips.
* `source`: abstract; an externally injected power source into the dust at `coord`.
* `open`: an input bit nothing is routed from.
* `dust`: an observed dust block, 1 iff strength ≥ the threshold.
* `lamp`: a real redstone lamp.

**Simulation mode.** A design is `block-accurate` only if it has no abstract
component, no `redc:` block and no `source` port bit. Observing dust is
measurement, not abstraction. Otherwise it is `abstract-components`, and
every report says so. An abstract simulation is never labelled
Minecraft-accurate.

**Fingerprint.** `fingerprint()` is a SHA-256 over blocks, components, ports
and probes, excluding metadata and annotations. Timing and simulation reports
record the fingerprint of the world they analysed, so a stale report can be
detected. The primitive backend refuses to write a design whose world changed
after timing.

## Supported redstone (`redc-redstone-sim-v1`)

| Block | Behaviour (Java Edition semantics modelled) |
| --- | --- |
| `minecraft:air` | nothing |
| `minecraft:stone` | full opaque **conductor**: can be strongly / weakly powered, cuts dust staircases, holds dust |
| `minecraft:glass` | holds dust but is **not** a conductor: a staircase on it carries signal only upward |
| `minecraft:redstone_wire` | strength 0..15, −1 per dust block; links per `calculateTargetStrength`; shape per `getConnectionState` (one connection → a line, none → a `+` cross); weakly powers the block below it and every block it points into; never powered by weak power |
| `minecraft:repeater` | reads only the block behind it (blockstate `facing` = input side); strongly powers the block in front; delay 1..4 rt; a pulse shorter than the delay is extended (Java's second tick) |
| `minecraft:redstone_torch` / `redstone_wall_torch` | lit iff its attachment block is unpowered; toggles 1 rt after its input changes (no pulse extension); powers every neighbour except its attachment, strongly powers the block above |
| `minecraft:lever` | set by a port; powers every neighbour, strongly powers its attachment |
| `minecraft:redstone_block` | always on; powers every neighbour, strongly powers nothing |
| `minecraft:redstone_lamp` | a conductor; lights at once when a neighbour gives it a signal, turns off 2 rt after the last signal goes |
| `redc:abstract_block` | an abstract component's voxel: an inert conductor (forces abstract mode) |

Power rules exactly as in Java:

* **Strong power** comes from a repeater facing in, a torch below, or an attached lever. Adjacent dust reads it.
* **Weak power** comes from dust on top of or pointing into a block. Mechanisms read it (repeater input, torch attachment, lamp), but dust never does.

**Unsupported, with an explicit diagnostic rather than an approximation:**

* every other block id (comparators, observers, pistons, buttons, …) gives `simulation/unsupported_block`;
* repeater locking (a diode pointing into a repeater's side) gives `simulation/unsupported_mechanic`;
* dust, repeaters, torches or levers without valid support give `simulation/unsupported_placement`;
* abstract pins or ports not on dust give `simulation/bad_binding`;
* torch burnout is detected (8 toggles within 60 gt) and reported as `simulation/unstable`, never simulated;
* quasi-connectivity does not apply, since there are no pistons.

## Connectivity is derived from geometry

`compile_world` builds the element graph used by the simulator and by STA.
It uses only the blocks. Producer labels (net ids) are never read, so two
routes that physically touch **are** connected, and the simulator catches
insufficient strength, unintended connections, reversed repeaters and
disconnected wires even when the compiler's metadata claims the route is
fine.

Elements are numbered in sorted-coordinate order:

* **dust** (with power links, shape and weak outputs) and its weakly connected **networks**;
* **sources**: repeater fronts, torches, levers, redstone blocks, abstract outputs and injected port bits;
* **power blocks**: conductors something can power, with strong and weak inputs;
* **readers**: device inputs;
* **devices**: the stateful objects.

## The event engine

Nothing is simulated per tick or per block. A priority queue holds
`(game tick, phase, sequence, event)`:

* phase 0 is external stimulus (`set_input` / `schedule_input`);
* phase 1 is scheduled device activity: repeater, torch and lamp ticks, and abstract evaluations, in scheduling (FIFO) order.

Each event changes some source. Its effects then settle **within the same
game tick**, in a fixed order that is a DAG by construction:

```text
changed sources -> strong block power -> dust networks (recomputed whole, strength-descending
bucket propagation) -> weak block power -> readers -> devices schedule FUTURE events
```

Every device delay is ≥ 1 gt (repeaters ≥ 2, torches 2, abstract arcs ≥ 1),
so a zero-delay loop is impossible. Only affected dust networks are
recomputed. Every set is visited in sorted id order, so the same design,
initial state and stimulus give the same event sequence, with no dependence
on hash order.

Cold start: at tick 0 everything is off. Constants turn on, every device
evaluates its input once, and abstract outputs are first evaluated at their
largest arc delay.

```python
from redc.minecraft import RedstoneSimulator
sim = RedstoneSimulator(design, trace="events")   # "off" | "events" | "full" (+ every dust change)
sim.set_input("start", 1)
sim.run_until_stable()             # raises simulation/unstable past max_ticks
sim.read_output("done")
sim.schedule_input("clk", 1, at_tick=20)   # game ticks
sim.run_until(100)                 # every event BEFORE tick 100 has happened
sim.step(); sim.pending_events; sim.time; sim.is_stable(); sim.probe("reg5"); sim.dust_at((x, y, z))
```

Trace records:

* `input_changed`, `clock_edge`, `wire_strength_changed` (`changes: [[x, y, z, s], ...]`);
* `repeater_scheduled`, `repeater_output_changed`, `torch_scheduled`, `torch_changed`, `lamp_changed`;
* `component_output_changed`, `register_clock_edge`, `register_capture`, `timing_violation`;
* `output_changed`, `probe_changed`, `warning`, `stable`.

Each record carries `t` (game ticks) and a coordinate or component.

### Deterministic model vs Java claims

What is modelled is Java's per-block logic: power rules, dust shape and
attenuation, repeater pulse extension, torch and lamp delays. What is **not**
modelled is Java's tick priorities and the block-update order inside one game
tick. Same-tick events run in scheduling order. RedC technology cells are
designed not to depend on same-tick ordering, and abstract registers use
setup and hold windows of ≥ 1 gt, so a capture never depends on it.

## Abstract components (`abstract-components` mode)

* **combinational**: transport delay per arc. The output at `T` is the truth table of each input's value at `T − delay(input→output)`, using the `max` corner by default. Glitches propagate.
* **dff**: rising edge, asynchronous active-high reset, `init`.
  * A rising clock-pin edge at `e` samples `d` and drives `q` at `e + clk→Q`.
  * `d` must not change in `(e − setup, e + hold)`, and a reset release must precede the edge by `recovery`. Violations are recorded.
  * While reset is high, edges are ignored and `q` goes to `init` after `reset→Q`.
* An input below its pin's required strength (but above 0) is a `simulation/weak_input` warning.

## Characterizing real cells

`characterize_cell(name, blocks, pins)` builds a harness around a block-level
cell: each pin gets its dust plus an approach stub, so the pin dust has its
routed shape. Inputs are driven at the cell's **minimum** required strength,
and output dust is observed. It measures the truth table, every
single-input transition in both directions (first and last output change,
glitches) and output strength. From those it derives conservative arcs:
`min` is the earliest change, `max` the latest settle. `Characterization.check`
compares the result with a declared function, timing and strength, so a
corrupted structure fails even if its declaration says it works.

`redc.physical_primitive.technology.structures.reference_library()` gives the
placeholder footprints of NOT (torch inverter), OR (diode-isolated merge,
strength 13), the pads, clock/reset sources, levers, lamps, the display and the
constants real blocks. Their gate timing is **measured** this way
(`source: "characterized"`). A design built only from them (`!a`, `a | b`,
`!(a | b)`) simulates `block-accurate` end to end. AND, XOR and the register
bit have no reference structure yet.

## Performance

Measured on `uint8_fib` with the default lever/display interface, after P&R
and clock balancing:

| Quantity | Value |
| --- | --- |
| Blocks | 66,287 |
| Dust | 29,049 |
| Repeaters | 1,790 |
| Abstract voxels | 5,544 |
| Dust networks | 2,093 |
| Devices | 2,143 |

One full `fib(10)` transaction (reset, 10 logical cycles at the 117 rt clock)
gave 55 and asserted `done` on cycle 9, as `Graph.run` does:

| Quantity | Value |
| --- | --- |
| Events | 8,120 (6,115 device ticks, 1,982 abstract evaluations) |
| Simulated game ticks | 2,204 (≈ 3.7 events per simulated tick) |
| Peak queue | 286 |
| Dust-network recomputes | 8,955 (87,843 dust changes) |
| Wall time | 2.3 s |

The hotspot is not the event engine (≈ 0.5 s) but the one-time static
`compile_world` (1.7 s: dust shapes and power links for 29k dust blocks).
Callers that simulate a world many times pass the compiled world in, as the
P&R validation does. The element tables are plain integer arrays, so the
engine is a direct candidate for a Rust/C++ port without changing the
representation.
