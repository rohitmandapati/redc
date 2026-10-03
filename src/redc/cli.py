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
    """List the available output backends."""
    from .backends import BACKENDS

    for name in available_backends():
        target = BACKENDS[name]
        default = " (default)" if name == DEFAULT_BACKEND else ""
        typer.echo(f"{name}{default}\t{target.extension}\t{target.summary}")


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
) -> None:
    """Print or write the unplaced Minecraft PhysicalNetlist (debug view)."""
    import json

    from .physical import lower_to_physical

    graph = _compile(source, top, Limits())
    try:
        payload = json.dumps(lower_to_physical(graph).to_dict(), indent=2) + "\n"
    except CompileError as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error
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


@app.command()
def pnr(
    source: SourcePath,
    top: Annotated[str, typer.Option(help="Top-level RedC function.")] = "main",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Final placed/routed design JSON (default build/<name>.physical.json)."
        ),
    ] = None,
    trace_path: Annotated[
        Path | None,
        typer.Option(
            "--trace", help="Replay trace path, .pnr.json or .pnr.jsonl (default build/<name>.pnr.json)."
        ),
    ] = None,
    trace_level: Annotated[
        str, typer.Option(help="Trace detail: none, basic, detailed or search.")
    ] = "basic",
    viewer: Annotated[
        bool, typer.Option("--viewer/--no-viewer", help="Also write the 3D replay viewer HTML.")
    ] = True,
    grid_height: Annotated[int, typer.Option(min=1, help="Grid height in cells.")] = 24,
    spacing: Annotated[
        int, typer.Option(min=0, help="Free cells between components in a layer.")
    ] = 3,
    layer_gap: Annotated[int, typer.Option(min=0, help="Free cells between layers.")] = 4,
    routing_margin: Annotated[
        int, typer.Option(min=0, help="Routing search margin around the design.")
    ] = 4,
    max_route_iterations: Annotated[
        int, typer.Option(min=0, help="Negotiated-congestion iterations.")
    ] = 40,
    max_attempts: Annotated[
        int, typer.Option(min=1, help="Place-and-route attempts (each spreads wider).")
    ] = 4,
) -> None:
    """Compile SOURCE, tech-map it, then place and route it in 3D.

    Always writes the replay trace -- also when P&R fails (exit code 1)."""
    import json

    from .physical import lower_to_physical
    from .physical.pnr import PnRConfig, TraceLevel, TraceRecorder, place_and_route
    from .viewer import write_pnr_html

    graph = _compile(source, top, Limits())
    stem = source.stem
    trace_path = trace_path or Path("build") / f"{stem}.pnr.json"
    output = output or Path("build") / f"{stem}.physical.json"
    try:
        config = PnRConfig(
            grid_height=grid_height,
            base_y=min(PnRConfig.base_y, grid_height - 1),
            component_spacing=spacing,
            layer_gap=layer_gap,
            routing_margin=routing_margin,
            max_routing_iterations=max_route_iterations,
            max_pnr_attempts=max_attempts,
            trace_level=TraceLevel.parse(trace_level),
        )
        netlist = lower_to_physical(graph)
    except CompileError as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error

    recorder = TraceRecorder(config.trace_level)
    result = None
    compile_error: CompileError | None = None
    try:
        result = place_and_route(netlist, config, trace=recorder)
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
            if viewer:
                html_path = trace_path.with_name(trace_path.name.split(".")[0] + ".pnr.html")
                write_pnr_html(recorder.to_dict(), html_path, title=f"RedC P&R - {source.name}")
                typer.echo(f"Wrote {html_path} (3D replay viewer)")
        except (CompileError, OSError) as error:
            typer.echo(f"redc: could not write the trace: {error}", err=True)

    if compile_error is not None:
        typer.echo(f"redc: {compile_error}", err=True)
        raise typer.Exit(code=1) from compile_error
    assert result is not None
    if not result.success:
        failure = result.failure
        typer.echo(
            f"redc: place-and-route failed after {result.attempts} attempt(s): "
            f"{failure.message if failure else 'unknown failure'}",
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.to_physical_dict(), indent=1) + "\n", encoding="utf-8")
    except OSError as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error
    metrics = result.metrics
    bounds = metrics["final"]["bounds"]
    typer.echo(f"Wrote {output} (placed and routed design)")
    typer.echo(
        f"P&R: {metrics['placement']['component_count']} components, "
        f"{metrics['routing']['routed_nets']} nets, {metrics['routing']['wire_cells']} wire cells, "
        f"{metrics['routing']['iterations']} routing iteration(s), attempt {result.geometry.attempt}, "
        f"bounds {bounds['dims'] if bounds else '-'}"
    )


@app.command("render-pnr")
def render_pnr(
    trace: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="A .pnr.json / .pnr.jsonl trace."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="HTML path (default: next to the trace)."),
    ] = None,
) -> None:
    """Write a self-contained 3D replay viewer page for a P&R trace."""
    from .viewer import write_pnr_html

    output = output or trace.with_name(trace.name.split(".")[0] + ".pnr.html")
    try:
        write_pnr_html(trace, output)
    except (CompileError, OSError, ValueError) as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Wrote {output} (open it in a browser; Three.js loads from a CDN)")
