"""Output backends for the RedC compiler.

A backend turns a validated :class:`~redc.ir.Graph` into a target artifact. They
are registered here so that new targets (a Minecraft component IR, a schematic,
another HDL, ...) can be added without touching the CLI: the CLI simply lets the
user pick a registered backend by name with a flag.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..ir import Graph
from ..parser import CompileError
from .systemverilog import emit_systemverilog


@dataclass(frozen=True)
class Backend:
    """A named way to render a compiled graph into a target artifact."""

    name: str
    summary: str
    extension: str
    emit: Callable[[Graph, str], str]


BACKENDS: dict[str, Backend] = {
    "systemverilog": Backend(
        name="systemverilog",
        summary="Combinational or clocked SystemVerilog module.",
        extension=".sv",
        emit=emit_systemverilog,
    ),
    "ir-json": Backend(
        name="ir-json",
        summary="Target-neutral RedC compute IR as JSON (redc.comb/seq.v1).",
        extension=".ir.json",
        emit=lambda graph, module: graph.to_json(),
    ),
}

DEFAULT_BACKEND = "systemverilog"


def available_backends() -> list[str]:
    """Registered backend names, in a stable order."""
    return sorted(BACKENDS)


def get_backend(name: str) -> Backend:
    """Look up a backend by name, or fail listing the valid choices."""
    try:
        return BACKENDS[name]
    except KeyError:
        choices = ", ".join(available_backends())
        raise CompileError(
            f"unknown backend '{name}'; available backends: {choices}"
        ) from None


__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "Backend",
    "available_backends",
    "emit_systemverilog",
    "get_backend",
]
