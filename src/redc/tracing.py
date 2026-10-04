"""Replay-trace verbosity shared by every physical place-and-route backend.

Both ``redc.physical`` (``redc.pnr.trace.v1``) and ``redc.physical_primitive``
(``redc.physical-primitive.pnr.v1``) record their compiler run as an
event-sourced replay trace.  The level semantics are deliberately identical so
``redc pnr --trace-level`` means the same thing whichever backend is selected.
:mod:`redc.physical.pnr.config` re-exports :class:`TraceLevel` unchanged.
"""

from __future__ import annotations

from enum import IntEnum

from .parser import CompileError


class TraceLevel(IntEnum):
    """How much of the P&R process the replay trace records.

    * ``NONE``     -- header and final state only, no events;
    * ``BASIC``    -- attempts, placements, every committed / ripped-up route,
      routing iterations and congestion snapshots (the default, compact);
    * ``DETAILED`` -- also each placement probe and every routed branch;
    * ``SEARCH``   -- also every A* node expansion (large: for visualizing the
      router "thinking").
    """

    NONE = 0
    BASIC = 1
    DETAILED = 2
    SEARCH = 3

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: str | int) -> TraceLevel:
        if isinstance(value, int):
            return cls(value)
        try:
            return cls[value.upper()]
        except KeyError:
            choices = ", ".join(level.label for level in cls)
            raise CompileError(f"unknown trace level {value!r}; use one of {choices}") from None


__all__ = ["TraceLevel"]
