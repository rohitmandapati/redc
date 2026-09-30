"""Output backends for RedC compiler IR."""

from .systemverilog import emit_systemverilog

__all__ = ["emit_systemverilog"]
