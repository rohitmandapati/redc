"""Public package surface for the RedC compiler."""

from .backends import emit_systemverilog
from .compiler import Compiler, Limits, compile_source
from .ir import BOOL, Graph, IRType, Value, type_from_name
from .parser import AST, CompileError, parse


def main() -> None:
    """Console-script entry point (``redc = "redc:main"``)."""
    from .cli import app

    app()


__all__ = [
    "AST",
    "BOOL",
    "CompileError",
    "Compiler",
    "Graph",
    "IRType",
    "Limits",
    "Value",
    "compile_source",
    "emit_systemverilog",
    "main",
    "parse",
    "type_from_name",
]
