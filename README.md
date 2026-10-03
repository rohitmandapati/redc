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

See `examples/` for sample programs.

## Roadmap

- Broader sequential support (nested/conditional runtime loops, multi-cycle calls)
- Explicit register/memory declarations and richer module interfaces
- Minecraft: a redstone component IR, then placement, routing, and schematic export
