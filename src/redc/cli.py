"""Command-line interface for the first RedC compiler pipeline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from .backends import DEFAULT_BACKEND, available_backends, get_backend
from .compiler import Limits, compile_source
from .parser import CompileError

app = typer.Typer(
    name="redc",
    no_args_is_help=True,
    help="Compile RedC programs into combinational or clocked SystemVerilog.",
)

SourcePath = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, readable=True, resolve_path=True),
]


def _limits(max_unroll: int, max_nodes: int, max_steps: int) -> Limits:
    return Limits(loop_iterations=max_unroll, nodes=max_nodes, steps=max_steps)


def _compile(source: Path, top: str, limits: Limits):
    try:
        return compile_source(
            source.read_text(encoding="utf-8"), top=top, limits=limits
        )
    except (CompileError, OSError, UnicodeError) as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command()
def build(
    source: SourcePath,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            "-b",
            help="Output backend (see `redc backends` for the full list).",
        ),
    ] = DEFAULT_BACKEND,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output path (defaults under build/)."),
    ] = None,
    top: Annotated[str, typer.Option(help="Top-level RedC function.")] = "main",
    module: Annotated[
        str | None,
        typer.Option(help="Generated module name (HDL backends)."),
    ] = None,
    emit_ir: Annotated[
        bool,
        typer.Option("--emit-ir", help="Also write the versioned compute IR as JSON."),
    ] = False,
    max_unroll: Annotated[
        int, typer.Option(min=1, help="Maximum loop iterations to unroll.")
    ] = 1024,
    max_nodes: Annotated[
        int, typer.Option(min=1, help="Maximum number of IR nodes.")
    ] = 100_000,
    max_steps: Annotated[
        int, typer.Option(min=1, help="Maximum compile-time lowering steps.")
    ] = 100_000,
) -> None:
    """Compile SOURCE with the chosen backend (SystemVerilog by default)."""
    try:
        target = get_backend(backend)
    except CompileError as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error

    graph = _compile(source, top, _limits(max_unroll, max_nodes, max_steps))
    output = output or Path("build") / f"{source.stem}{target.extension}"
    module = module or f"redc_{top}"
    try:
        artifact = target.emit(graph, module)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(artifact, encoding="utf-8")
        if emit_ir:
            output.with_suffix(".ir.json").write_text(graph.to_json(), encoding="utf-8")
    except (CompileError, OSError) as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error

    operations = Counter(node["op"] for node in graph.live_nodes())
    summary = ", ".join(f"{name}={count}" for name, count in sorted(operations.items()))
    typer.echo(f"Wrote {output} ({target.name})")
    typer.echo(f"IR: {len(graph.live_nodes())} live nodes ({summary})")


@app.command()
def backends() -> None:
    """List the available output backends (``redc build --backend``).

    Place-and-route backends are a separate registry: see
    ``redc physical-backends``."""
    from .backends import BACKENDS

    for name in available_backends():
        target = BACKENDS[name]
        default = " (default)" if name == DEFAULT_BACKEND else ""
        typer.echo(f"{name}{default}\t{target.extension}\t{target.summary}")


@app.command("physical-backends")
def physical_backends_command() -> None:
    """List the physical place-and-route backends (``redc pnr --backend``)."""
    from .physical_backends import (
        DEFAULT_PHYSICAL_BACKEND,
        available_physical_backends,
        get_physical_backend,
    )

    for name in available_physical_backends():
        backend = get_physical_backend(name)
        default = " (default)" if name == DEFAULT_PHYSICAL_BACKEND else ""
        typer.echo(
            f"{name}{default}\t{backend.trace_schema}\t{backend.design_schema}\t{backend.summary}"
        )


@app.command()
def check(
    source: SourcePath,
    top: Annotated[str, typer.Option(help="Top-level RedC function.")] = "main",
    max_unroll: Annotated[int, typer.Option(min=1)] = 1024,
    max_nodes: Annotated[int, typer.Option(min=1)] = 100_000,
    max_steps: Annotated[int, typer.Option(min=1)] = 100_000,
) -> None:
    """Parse, lower, and validate SOURCE without writing output files."""
    graph = _compile(source, top, _limits(max_unroll, max_nodes, max_steps))
    typer.echo(f"OK: {source} ({len(graph.live_nodes())} live IR nodes)")


@app.command("dump-ir")
def dump_ir(
    source: SourcePath,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Write JSON to this path instead of stdout."
        ),
    ] = None,
    top: Annotated[str, typer.Option(help="Top-level RedC function.")] = "main",
) -> None:
    """Print or write the target-neutral compute IR."""
    graph = _compile(source, top, Limits())
    payload = graph.to_json()
    if output is None:
        typer.echo(payload, nl=False)
        return
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    except OSError as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Wrote {output}")


PhysicalBackendOption = Annotated[
    str,
    typer.Option(
        "--backend",
        "-b",
        help="Physical backend: physical (coarse, default) or physical-primitive "
        "(see `redc physical-backends`).",
    ),
]
InterfaceOption = Annotated[
    str | None,
    typer.Option(
        help="physical-primitive only: port realization, 'default' (lever / display "
        "peripherals where applicable) or 'pads' (one pad per bit)."
    ),
]


def _physical_backend(name: str):
    from .physical_backends import get_physical_backend

    try:
        return get_physical_backend(name)
    except CompileError as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error


def _emit_json(payload: str, output: Path | None) -> None:
    if output is None:
        typer.echo(payload, nl=False)
        return
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    except OSError as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Wrote {output}")


@app.command("dump-netlist")
def dump_netlist(
    source: SourcePath,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Write JSON to this path instead of stdout."
        ),
    ] = None,
    top: Annotated[str, typer.Option(help="Top-level RedC function.")] = "main",
    backend: PhysicalBackendOption = "physical",
    stage: Annotated[
        str | None,
        typer.Option(
            help="Netlist stage. physical: 'physical' (default). physical-primitive: "
            "'primitive' (technology-neutral one-bit netlist, default) or 'mapped' "
            "(technology-mapped, unplaced)."
        ),
    ] = None,
    interface: InterfaceOption = None,
) -> None:
    """Print or write a physical backend's unplaced netlist (debug view).

    ``--backend physical`` dumps the coarse PhysicalNetlist (wide components,
    bus nets).  ``--backend physical-primitive`` dumps the technology-neutral
    one-bit PrimitiveNetlist (``redc.primitive-netlist.v1``), or with
    ``--stage mapped`` the Minecraft-technology-mapped version."""
    import json

    target = _physical_backend(backend)
    graph = _compile(source, top, Limits())
    try:
        payload = json.dumps(target.dump_netlist(graph, stage=stage, interface=interface), indent=2) + "\n"
    except CompileError as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error
    _emit_json(payload, output)


@app.command("dump-primitive-netlist")
def dump_primitive_netlist(
    source: SourcePath,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write JSON to this path instead of stdout."),
    ] = None,
    top: Annotated[str, typer.Option(help="Top-level RedC function.")] = "main",
    stage: Annotated[
        str, typer.Option(help="'primitive' (technology-neutral, default) or 'mapped'.")
    ] = "primitive",
    interface: InterfaceOption = None,
) -> None:
    """Shorthand for ``dump-netlist --backend physical-primitive``."""
    dump_netlist(source, output=output, top=top, backend="physical-primitive", stage=stage, interface=interface)


@app.command()
def pnr(
    source: SourcePath,
    top: Annotated[str, typer.Option(help="Top-level RedC function.")] = "main",
    backend: PhysicalBackendOption = "physical",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Final placed/routed design JSON (default build/<name>.physical.json; "
            "physical-primitive: build/<name>.primitive.physical.json).",
        ),
    ] = None,
    trace_path: Annotated[
        Path | None,
        typer.Option(
            "--trace",
            help="Replay trace path, .json or .jsonl (default build/<name>.pnr.json; "
            "physical-primitive: build/<name>.primitive.pnr.json).",
        ),
    ] = None,
    trace_level: Annotated[
        str, typer.Option(help="Trace detail: none, basic, detailed or search.")
    ] = "basic",
    viewer: Annotated[
        bool, typer.Option("--viewer/--no-viewer", help="Also write the 3D replay viewer HTML.")
    ] = True,
    grid_height: Annotated[
        int | None, typer.Option(min=1, help="physical only: grid height in cells [24].")
    ] = None,
    spacing: Annotated[
        int | None,
        typer.Option(
            min=0,
            help="Free space between neighbouring components in a column "
            "[physical: 3 cells; physical-primitive: 1 block beyond keep-outs].",
        ),
    ] = None,
    layer_gap: Annotated[
        int | None, typer.Option(min=0, help="physical only: free cells between layers [4].")
    ] = None,
    channel_width: Annotated[
        int | None,
        typer.Option(min=0, help="physical-primitive only: free blocks between placement columns [6]."),
    ] = None,
    max_height: Annotated[
        int | None,
        typer.Option(min=3, help="physical-primitive only: highest block y routes may use [9]."),
    ] = None,
    routing_margin: Annotated[
        int | None,
        typer.Option(
            min=0, help="Routing search margin around the design [physical: 4 cells; physical-primitive: 6 blocks]."
        ),
    ] = None,
    max_route_iterations: Annotated[
        int | None,
        typer.Option(min=0, help="Negotiated-congestion iterations [physical: 40; physical-primitive: 30]."),
    ] = None,
    max_attempts: Annotated[
        int | None,
        typer.Option(
            min=1, help="Place-and-route attempts, each spreads wider [physical: 4; physical-primitive: 3]."
        ),
    ] = None,
    interface: InterfaceOption = None,
) -> None:
    """Compile SOURCE, lower it with the chosen physical backend, then place
    and route it in 3D.

    ``--backend physical`` (default) uses the coarse component backend;
    ``--backend physical-primitive`` bit-blasts everything into one-bit gates
    and routes every bit as its own redstone net at block resolution.

    Always writes the replay trace -- also when P&R fails (exit code 1)."""
    import json

    from .physical_backends import PnROptions, check_options

    target = _physical_backend(backend)
    options = PnROptions(
        trace_level=trace_level,
        spacing=spacing,
        routing_margin=routing_margin,
        max_route_iterations=max_route_iterations,
        max_attempts=max_attempts,
        grid_height=grid_height,
        layer_gap=layer_gap,
        max_height=max_height,
        channel_width=channel_width,
        interface=interface,
    )
    try:
        check_options(target, options)
    except CompileError as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error
    graph = _compile(source, top, Limits())
    paths = target.default_paths(source.stem)
    trace_path = trace_path or paths.trace
    output = output or paths.output
    try:
        prepared = target.prepare(graph, options, source=source, top=top)
    except CompileError as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error

    recorder = prepared.recorder
    run = None
    compile_error: CompileError | None = None
    try:
        run = target.run(prepared)
    except CompileError as error:
        recorder.fail(str(error))
        compile_error = error
    except Exception as error:  # keep the trace of a crashed run, then re-raise
        recorder.fail(f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            recorder.write(trace_path)
            typer.echo(f"Wrote {trace_path} (replay trace, {len(recorder.events)} events)")
        except (CompileError, OSError) as error:
            typer.echo(f"redc: could not write the trace: {error}", err=True)
        else:
            if viewer:
                html_path = target.viewer_path(trace_path)
                title = f"RedC P&R - {source.name}" if target.name == "physical" else (
                    f"RedC {target.name} P&R - {source.name}"
                )
                try:
                    target.write_viewer(recorder.to_dict(), html_path, title=title)
                    typer.echo(f"Wrote {html_path} (3D replay viewer)")
                except (CompileError, OSError) as error:
                    typer.echo(f"redc: could not write the viewer: {error}", err=True)

    if compile_error is not None:
        typer.echo(f"redc: {compile_error}", err=True)
        raise typer.Exit(code=1) from compile_error
    assert run is not None
    if not run.success:
        typer.echo(
            f"redc: place-and-route failed after {run.attempts} attempt(s): {run.failure}",
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(run.design, indent=1) + "\n", encoding="utf-8")
    except OSError as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Wrote {output} (placed and routed design)")
    typer.echo(run.summary)


@app.command("render-pnr")
def render_pnr(
    trace: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="A coarse (.pnr.json / .pnr.jsonl) or primitive (.primitive.pnr.json) trace.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="HTML path (default: next to the trace)."),
    ] = None,
) -> None:
    """Write a self-contained 3D replay viewer page for a P&R trace of either
    physical backend (chosen from the trace's schema)."""
    from .physical_backends import get_physical_backend
    from .viewer import load_any_trace, trace_backend, write_trace_html

    try:
        data = load_any_trace(trace)
        if output is None:
            output = get_physical_backend(trace_backend(data)).viewer_path(trace)
        write_trace_html(data, output, title=f"RedC P&R replay - {trace.name}")
    except (CompileError, OSError, ValueError) as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Wrote {output} (open it in a browser; Three.js loads from a CDN)")
