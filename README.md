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
`start`/`done` handshake — pulse `start`, hold inputs stable, read the result
when `done` is high.

See `examples/` for sample programs.

## Roadmap

- Broader sequential support (nested/conditional runtime loops, multi-cycle calls)
- Explicit register/memory declarations and richer module interfaces
- Minecraft: a redstone component IR, then placement, routing, and schematic export
