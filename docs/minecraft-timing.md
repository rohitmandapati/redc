# Physical timing closure (`redc-redstone-timing-v1`)

The invariant: **one logical RedC clock cycle is one physical clock period.**
Physical delay may make the period longer. It never changes what happens on
which logical cycle. RedC never pipelines a program, never adds state and
never fakes delay metadata.

## One timing model, two consumers

`redc.minecraft.timing` defines every delay once. The simulator's block
behaviours schedule events with these functions, and STA builds its edges with
the same functions.

| Element | Delay (game ticks) |
| --- | --- |
| dust | 0 (Java updates dust within the tick) |
| repeater at setting `d` rt | `repeater_delay_gt(d) = 2·d`; latest settle after a *glitching* input `2·(2d) − MIN_PULSE` (pulse extension) |
| torch | `TORCH_DELAY_GT = 2` |
| lamp | on 0, off `LAMP_OFF_DELAY_GT = 4` |
| abstract combinational cell | its declared `CombinationalArc(from_pin, to_pin, min, max)` |
| abstract register | `SequentialTiming(clock, data, q, reset, clk→Q min/max, setup, hold, reset→Q, recovery)` |
| block-level cell | derived from its blocks by the same rules, or characterized by simulation |

Every modelled delay is a whole number of redstone ticks, so with stimuli on
even ticks the narrowest pulse is `MIN_PULSE_GT = 2`. Technology cells carry an
explicit `ComponentTiming`. `cell.latency` is a display number only. The v1
placeholders declare:

* gates: every arc = the placeholder latency (NOT 1, AND 2, OR 1, XOR 3 rt);
* register: clk→Q 2 rt, setup 1 rt, hold 1 rt, reset→Q 2 rt, recovery 1 rt.

Reports set `placeholder_estimate: true` while any of these remain.

## The STA graph

`redc.minecraft.sta` builds its graph from the **same**
`connectivity.compile_world` element graph the simulator runs. Nodes:

* dust networks;
* sources;
* the strong and total power of conductor blocks;
* readers.

Edges:

* wiring edges have delay 0;
* repeater: configured delay; latest uses the pulse-extension bound if its input may glitch;
* torch: 2 gt;
* abstract input pin → output pin: the declared arc;
* a register's q has no incoming edge: it is a launch point.

A node *may glitch* if a predecessor may, or if ≥ 2 of its predecessors can
change. Register outputs and input ports change at most once per cycle.

There are four propagations, each forward in topological order. They take the
latest (max) and earliest (min) arrival, and ties go to the lowest node id, so
equal builds report equal paths:

1. **clock**, from the clock port at 0, giving the clock arrival at every register clock pin;
2. **reset**, from the reset port;
3. **data from registers**, launched at `clock arrival + clk→Q`;
4. **data from inputs**, launched at 0.

Structural errors are reported, never silently analysed:

| Code | Cause |
| --- | --- |
| `timing/clock_unreached` | the clock does not reach a register |
| `timing/clock_leak` | the clock reaches data or reset |
| `timing/clock_glitch` | the clock may glitch |
| `timing/data_reaches_clock` | data reaches a clock pin (gated clock) |
| `timing/data_reaches_reset` | data reaches an asynchronous reset |
| `timing/combinational_loop` | a loop with no register |

Endpoints: register **D** pins (setup and hold), output bits (setup only).
**CLK** pins are clock sinks and **RST** pins are reset sinks, analysed
separately, never as data.

## Setup and hold

The source clock's rising edge `k` is at `E_k`. `clk(r)` is the exact physical
arrival at register `r`; the clock tree never glitches. `D_max(C)` and
`D_min(C)` are the latest and earliest data arrival at capture register `C`,
relative to `E_k`, **with launch clock arrival and clk→Q counted exactly once**:
`D = clk(L) + clk→Q(L) + path(L→C)`.

```text
setup_slack(C) = P + clk(C) − setup(C) − D_max(C)    >= 0
hold_slack(C)  = D_min(C) − clk(C) − hold(C)         >= 0
```

That is the textbook `P + clk_capture − clk_launch − setup − path_max` and
`clk_launch + path_min − clk_capture − hold`. `P` does not appear in hold:
**a slower clock never fixes a hold violation**, and automatic period
selection never looks at hold.

Window semantics, used identically by the simulator's abstract registers: for
an edge at the clock pin at `e`, `d` must not change in the open interval
`(e − setup, e + hold)`. With setup and hold ≥ 1 gt, no capture depends on the
order of two events in the same tick.

Rules for inputs and outputs:

* **Inputs** change at `E_k + input_offset`. The offset is the smallest whole
  number of redstone ticks that keeps every input→register path hold-safe
  (`offset ≥ clk(C) + hold(C) − in_min(C)`). Inputs then add a setup
  constraint `offset + in_max(C) + setup − clk(C) ≤ P`.
* **Outputs** are sampled at `E_{k+1}` before that tick's events, so they must
  settle 1 gt earlier: `D_max(out) + 1 ≤ P`.

## Automatic period

```text
P_required = round_up_to_rt( max( every setup constraint,
                                  2 · max(MIN_PULSE, longest clock-tree repeater) ) )
P_auto     = P_required + clock_margin_rt
```

The second term keeps both clock phases at least as long as any clock
repeater, so no clock pulse is ever distorted. The high phase is
`⌊P_rt / 2⌋` rt.

A **user period** (`--clock-period`, `clock_period_rt`) is used exactly. If it
is shorter than `P_required`, closure fails with `timing/setup`; the period is
never raised silently. `timing/clock_pulse` covers phases that are too short.
`max_clock_skew_rt` (default 0) bounds the accepted skew (`timing/clock_skew`).

## Clock tree: analysis and balancing (`physical_primitive.pnr.clock`)

The clock is a routed one-bit net from CLOCK_SOURCE to every REGISTER_BIT
`clk`. A sink's arrival is the sum of the repeater delay settings on its tree
path. **Balancing** makes every arrival equal to the latest one, using only
physical delay:

* raise an existing repeater's setting (up to 4 rt);
* turn a repeater-eligible dust block into a repeater: a straight, level block with one child that is not a pin, the same rule legalization uses.

The algorithm is deterministic and tree-aware:

1. Target `T = max arrival`, with `deficit(s) = T − arrival(s)`.
2. Walk the tree root-first. At each site `v`, add
   `x(v) = min(capacity(v), min_{s below v} deficit(s) − added_above(v))`.
   Delay goes as **high** as every downstream sink can absorb: a trunk repeater
   delays every sink below it, so a trunk is only delayed when all its sinks
   need it, and the latest sink's path is never touched.
3. Placing the most possible delay highest is optimal. If any sink is left
   short, no assignment balances the tree: `timing/clock_balance`.

The balanced route is re-legalized (strength recomputed), independently
re-verified, and re-materialized. STA then measures skew from the geometry
again (0), and the simulator confirms that one clock edge reaches every
register clock pin in the same game tick.

**Clock taps.** A greedily routed clock net is a comb: one long spine whose
arrival grows about 1 rt per 15 blocks, with each register on a 2–3 block stub
that has no room for balancing delay. So every register clock sink gets a
reserved **tap**: a straight, level run of `clock_tap_length` blocks (default
4, growing by `retry_clock_tap_growth` per retry) in front of its `clk` pin.
The router reaches the tap entry and runs straight down the tap into the pin,
which gives every register an exclusive straight segment with at least 3
repeater sites (≥ 12 rt). Taps stop early at any block that would collide
with or short against another pin or pin approach. `timing/clock_balance` is
the only timing failure that triggers another, roomier P&R attempt.

## Dynamic validation (`redc.minecraft.testbench`)

After closure, the materialized world is driven in the redstone simulator at
the chosen clock. This validates STA; it never replaces it.

1. At tick 0 the clock is low, reset is asserted and cycle-0 inputs are applied. The simulator runs until stable.
2. Reset is released at the next whole rt `R`. The first edge is at `E_1 = R + reset recovery gap`.
3. Edges follow at `E_k = E_1 + (k−1)·P`, and cycle-k inputs change at `E_k + input_offset`.
4. Observation `k` is taken at `E_{k+1}`, before that tick's events: outputs `O_k` and every register probe `S_k`. It corresponds exactly to `Graph.step` number `k`.

The primitive backend compares each observation with the `PrimitiveSimulator`
(outputs, every register bit, the `done` cycle) and checks:

* no setup, hold or recovery violation at any abstract register;
* every register captured **exactly once per edge, at `edge + clk(r)`**, so no missing, extra or intermediate captures;
* no weak inputs.

Failure codes: `simulation/logic_mismatch`, `simulation/cycle_mismatch`,
`simulation/timing_violation`, `simulation/weak_signal`,
`simulation/unstable`, `simulation/unsupported_block`.

A deliberately too-fast clock is caught in one of two ways. If data lands in
the capture tick, it is a window violation. If it lands after the next edge,
the path has silently become two cycles, and that shows up as a logical
mismatch against the one-cycle reference.

## The `timing` report (design file and trace `final`)

```jsonc
"timing": {
  "model": "redc-redstone-timing-v1", "closure_model": "redc-sync-closure-v1",
  "world_fingerprint": "…",                       // the world this analysed
  "sequential": true,
  "constraints": {"period_rt": null, "mode": "auto", "margin_rt": 1, "max_skew_rt": 0},
  "clock": {"period": {"gt": 196, "rt": 98}, "high": …, "low": …, "mode": "auto",
            "required_period": …, "margin": …, "arrival_min": …, "arrival_max": …, "skew": {"gt": 0, "rt": 0},
            "sinks": [{"register": "i8", "labels": {...}, "arrival": …}]},
  "input_offset": …, "reset_recovery_gap": …,
  "setup": {"worst_slack": …, "endpoints": 25, "violations": [],
            "critical_path": {"endpoint": …, "launch": "register", "launch_register": "i10", "launch_clock_arrival": …,
                              "launch_clock_path": [step…], "clk_to_q": …, "steps": [step…],
                              "capture_register": …, "capture_clock_arrival": …, "capture_clock_path": [step…],
                              "data_arrival": …, "required": …, "slack": …, "setup": …}},
  "hold": {"worst_slack": …, "critical_path": {…}, "violations": []},
  "closure": {"passed": true, "failures": []}
}
```

Each path step is `{node, kind (net|source|reader|strong|power), label, coord
| first/last/dust_blocks, device, delay_rt?, component?, pin?, labels?, nets?,
arrival, edge_delay}`. `nets` comes from the producer's annotations and is
for reports only. Combinational designs report
`"combinational": {settle, earliest_change, critical_path}` instead of the
clock, setup and hold blocks.

The design also carries `simulation` (model, mode, Minecraft version,
validated, schedule, events, cycles, failures) and `clock_tree` (skew before
and after, every change). The trace `final` additionally carries
`simulation_playback`, a short recorded run the viewer animates.
