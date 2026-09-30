"""Command-line interface for the first RedC compiler pipeline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from .backends import emit_systemverilog
from .compiler import Limits, compile_source
from .parser import CompileError

app = typer.Typer(
    name="redc",
    no_args_is_help=True,
    help="Compile stateless RedC programs into combinational hardware.",
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
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="SystemVerilog output path."),
    ] = None,
    top: Annotated[str, typer.Option(help="Top-level RedC function.")] = "main",
    module: Annotated[
        str | None,
        typer.Option(help="Generated SystemVerilog module name."),
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
    """Compile SOURCE into a combinational SystemVerilog module."""
    graph = _compile(source, top, _limits(max_unroll, max_nodes, max_steps))
    output = output or Path("build") / f"{source.stem}.sv"
    module = module or f"redc_{top}"
    try:
        systemverilog = emit_systemverilog(graph, module)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(systemverilog, encoding="utf-8")
        if emit_ir:
            output.with_suffix(".ir.json").write_text(graph.to_json(), encoding="utf-8")
    except (CompileError, OSError) as error:
        typer.echo(f"redc: {error}", err=True)
        raise typer.Exit(code=1) from error

    operations = Counter(node["op"] for node in graph.live_nodes())
    summary = ", ".join(f"{name}={count}" for name, count in sorted(operations.items()))
    typer.echo(f"Wrote {output}")
    typer.echo(f"IR: {len(graph.live_nodes())} live nodes ({summary})")


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
