"""The versioned primitive replay trace: ``redc.physical-primitive.pnr.v1``.

The trace is an intentional compiler artifact (not debug logging): an
event-sourced record of how the compiler arrived at the final design, meant to
drive the browser viewer today and a Minecraft mod later.  It is distinct from
the final design file, which is the physical truth; nothing downstream should
rebuild the design by replaying events.

Structure: a header (schema, ``backend: physical-primitive``, coordinate system
in blocks, source, IR summary, the mapped design with every cell definition,
instance provenance, net and logical grouping, and the config), then ordered
events (contiguous ``seq``, a ``phase`` and a ``type``), then a ``final`` block.
Every failed attempt stays in the trace; a crashed or failed run still closes
the trace with :meth:`PrimitiveTraceRecorder.fail` so it can be written.

Levels (see :class:`~redc.tracing.TraceLevel`): ``none`` = header + final only;
``basic`` = synthesis per IR node, attempts, committed placements, routing
iterations, committed / ripped routes, congestion and legalization;
``detailed`` = also each emitted primitive, tech-map choice, placement probe,
route branch and path rejection; ``search`` = also every A* expansion and
(capped) blocked transitions.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ...parser import CompileError
from ...tracing import TraceLevel
from ..physical import PrimitivePhysicalNetlist
from .config import PrimitivePnRConfig
from .records import design_records

TRACE_SCHEMA = "redc.physical-primitive.pnr.v1"
BACKEND = "physical-primitive"

COORDINATE_SYSTEM = {
    "units": "blocks",
    "x": "east",
    "y": "up",
    "z": "south",
    "cell": "integer [x, y, z] names the Minecraft block [x, x+1) x [y, y+1) x [z, z+1)",
}

#: Every event phase, in pipeline order.
PHASES = (
    "synthesis",
    "techmap",
    "pnr",
    "placement",
    "routing",
    "congestion",
    "legalization",
    "finalize",
)


class PrimitiveTraceRecorder:
    """Collects the replay trace of one primitive compile + P&R run."""

    def __init__(self, level: TraceLevel | str | int = TraceLevel.BASIC) -> None:
        self.level = TraceLevel.parse(level)
        self.events: list[dict[str, Any]] = []
        self.source: dict[str, Any] | None = None
        self.ir: dict[str, Any] | None = None
        self.design: dict[str, Any] | None = None
        self.config: dict[str, Any] | None = None
        self.final: dict[str, Any] | None = None

    def wants(self, level: TraceLevel) -> bool:
        """Whether events of ``level`` are being recorded."""
        return self.level != TraceLevel.NONE and self.level >= level

    def emit(self, phase: str, type: str, *, level: TraceLevel = TraceLevel.BASIC, **data: Any) -> None:
        """Record one event if ``level`` is enabled (``data`` must be JSON-native)."""
        if not self.wants(level):
            return
        if phase not in PHASES:
            raise CompileError(f"unknown trace phase {phase!r}")
        self.events.append({"seq": len(self.events), "phase": phase, "type": type, **data})

    def begin_source(self, *, path: str | None = None, top: str | None = None, ir: dict[str, Any] | None = None) -> None:
        """Describe the source program (before synthesis starts)."""
        self.source = {"path": path, "top": top}
        self.ir = ir

    def begin(self, mapped: PrimitivePhysicalNetlist, config: PrimitivePnRConfig) -> None:
        """Snapshot the (unplaced) mapped design and the configuration."""
        self.design = design_records(mapped)
        # The level actually recorded wins over the config's default.
        self.config = {**config.to_dict(), "trace_level": self.level.label}
        if self.ir is None:
            self.ir = mapped.logical.graph_summary

    def finish(self, final: dict[str, Any]) -> None:
        self.final = final

    def fail(self, message: str, *, stage: str = "internal") -> None:
        """Close an interrupted trace so it can still be written and inspected."""
        if self.final is None:
            self.final = {"success": False, "failure": {"stage": stage, "reason": stage, "message": message}}

    def header(self) -> dict[str, Any]:
        return {
            "schema": TRACE_SCHEMA,
            "generator": "redc",
            "backend": BACKEND,
            "trace_level": self.level.label,
            "coordinate_system": COORDINATE_SYSTEM,
            "source": self.source,
            "ir": self.ir,
            "design": self.design,
            "config": self.config,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.header(), "events": self.events, "final": self.final}

    def records(self) -> Iterator[dict[str, Any]]:
        """The JSON Lines form: header, each event, final -- tagged by ``record``."""
        yield {"record": "header", **self.header()}
        for event in self.events:
            yield {"record": "event", **event}
        yield {"record": "final", "final": self.final}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":")) + "\n"

    def write(self, path: str | Path) -> Path:
        """Write ``.jsonl`` as JSON Lines, anything else as one JSON document."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            with path.open("w", encoding="utf-8") as handle:
                for record in self.records():
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        else:
            path.write_text(self.to_json(), encoding="utf-8")
        return path


def load_trace(path: str | Path) -> dict[str, Any]:
    """Read a primitive ``.json`` / ``.jsonl`` trace back into the JSON form."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        trace: dict[str, Any] = {"events": []}
        for line in text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.pop("record", None)
            if kind == "header":
                trace.update(record)
            elif kind == "event":
                trace["events"].append(record)
            elif kind == "final":
                trace["final"] = record["final"]
    else:
        trace = json.loads(text)
    if not isinstance(trace, dict) or trace.get("schema") != TRACE_SCHEMA:
        raise CompileError(f"{path}: not a {TRACE_SCHEMA} trace")
    return trace


__all__ = ["BACKEND", "COORDINATE_SYSTEM", "PHASES", "TRACE_SCHEMA", "PrimitiveTraceRecorder", "load_trace"]
