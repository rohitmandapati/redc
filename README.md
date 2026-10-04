# RedC

A small C-like hardware language. You write imperative code; RedC compiles it to
hardware. The first backend is **SystemVerilog**; the long-term goal is compiling
to **Minecraft redstone**.

## Install

```bash
uv sync --all-groups
```

## Usage

```bash
# Validate a program.
uv run redc check examples/uint8_alu.redc

# Compile to SystemVerilog (written under build/).
uv run redc build examples/uint8_alu.redc

# Compile a specific top-level function.
uv run redc build examples/uint8_fib.redc --top fib

# Dump the target-neutral IR as JSON, or list output backends.
uv run redc dump-ir examples/popcount.redc
uv run redc backends
```

Loops with a compile-time bound become plain combinational logic. A loop whose
bound depends on a runtime value (like `uint8_fib`) compiles to a **clocked**
module instead: loop variables become registers and the module gets a
`start`/`done` handshake — pulse `start` while idle, hold inputs stable, read the
result on the one cycle `done` is high. The module then returns to idle and can
be started again without a reset; a `start` while busy is ignored.

## Physical (Minecraft) layer

`redc.physical` lowers the IR to an unplaced `PhysicalNetlist` (technology
mapping; placement and routing come next):

```bash
uv run redc dump-netlist examples/uint8_fib.redc --top fib
```

It then places and routes that netlist on a 3D grid of abstract cells, and
writes a replay trace of the whole process with a browser viewer for it (see
[docs/pnr-trace.md](docs/pnr-trace.md)):

```bash
uv run redc pnr examples/uint8_fib.redc --top fib
# build/uint8_fib.pnr.json      replay trace (redc.pnr.trace.v1)
# build/uint8_fib.pnr.html      3D replay viewer (open in a browser; loads Three.js from a CDN)
# build/uint8_fib.physical.json final placed + routed design
```

- **Types:** the language accepts widths 1–64, but Minecraft supports only
  `bool`, `uint4/int4`, `uint8/int8`, `uint16/int16`, `uint32/int32` and
  `uint64/int64` (`is_supported_physical_type`).
- **Encoding:** `bool` is one line, 4/8-bit values are binary buses, and
  16/32/64-bit values use hex nibble lanes. Every value is still one logical net.
- **Implementations:** cells are indexed by exact `OperationSignature`
  (op + full signed types) in an `ImplementationRegistry`, which can also hold
  composite (one-to-many) implementations.
- **Timing:** `latency` is propagation delay in ticks. It is separate from
  statefulness: only registers are sequential.
- **Clock/reset:** there is one global clock and one global active-high reset
  per design. Each register's reset value belongs to the instance, not the cell.
- **Peripherals:** real Minecraft I/O devices sit at the boundary. Until source
  annotations exist, `start` maps to a lever and a `uint8 result` to a two-digit
  seven-segment display (placeholder geometry); other ports use abstract pads.

### Primitive (bit-level) physical backend

`--backend physical-primitive` is a second, independent physical backend
(`redc.physical_primitive`). Instead of one hand-made cell per wide operation,
it bit-blasts EVERY operation -- add, mul, div, mod, shifts, comparisons,
casts, muxes, registers -- into one-bit AND/OR/XOR/NOT gates and one-bit
register bits, maps each onto a (placeholder) Minecraft cell, and places and
routes every bit as its own redstone net at block resolution (one coordinate
= one block), with repeater insertion and an independent electrical check:

```bash
uv run redc pnr examples/uint8_add.redc --top add --backend physical-primitive
# build/uint8_add.primitive.pnr.json       replay trace (redc.physical-primitive.pnr.v1)
# build/uint8_add.primitive.pnr.html       3D replay viewer
# build/uint8_add.primitive.physical.json  final block-level design (redc.physical-primitive.v1)
uv run redc dump-netlist examples/uint8_alu.redc --backend physical-primitive   # the one-bit netlist
uv run redc physical-backends
```

Every IR width works here (`uint3` included). Circuits get very large; that is
intentional. See [docs/physical-primitive.md](docs/physical-primitive.md) and
[docs/physical-primitive-trace.md](docs/physical-primitive-trace.md).

Every routed design is materialized as backend-neutral Minecraft blocks
(`redc.minecraft`), simulated by an event-driven redstone simulator and closed
for timing. The clock tree is physically balanced, the clock period is chosen
automatically (`--clock-period` checks a given one), and setup and hold are
checked. One logical cycle is one physical clock period, verified by simulating
the design at that clock. See [docs/minecraft-simulator.md](docs/minecraft-simulator.md)
and [docs/minecraft-timing.md](docs/minecraft-timing.md).

See `examples/` for sample programs.

## Roadmap

- Broader sequential support (nested/conditional runtime loops, multi-cycle calls)
- Explicit register/memory declarations and richer module interfaces
- Minecraft: a redstone component IR, then placement, routing, and schematic export
