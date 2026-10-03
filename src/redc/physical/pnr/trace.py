"""The versioned place-and-route replay trace (``redc.pnr.trace.v1``).

The trace is event-sourced: a header describing the unplaced design, then an
ordered list of events (each with a contiguous ``seq``, a ``phase`` and a
``type``) recording the whole process -- placements, design expansion, route
searches, rip-ups, congestion -- and finally the end state.  It contains only
JSON-native values; see ``docs/pnr-trace.md`` for the full schema.

The recorder keeps everything in memory and can write either one JSON document
(``.pnr.json``) or a stream of JSON Lines (``.pnr.jsonl``: header record, one
record per event, final record).  :meth:`TraceRecorder.emit` is the single
choke point, so a streaming backend only needs to override it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ...parser import CompileError
from ..netlist import PhysicalNetlist
from .config import PnRConfig, TraceLevel
from .records import design_records

TRACE_SCHEMA = "redc.pnr.trace.v1"

COORDINATE_SYSTEM = {
    "units": "grid_cells",
    "x": "east",
    "y": "up",
    "z": "south",
    "cell": "integer [x, y, z] names the unit cube [x, x+1) x [y, y+1) x [z, z+1)",
}

#: Every event phase.
PHASES = ("pnr", "placement", "routing", "congestion", "finalize")


class TraceRecorder:
    """Collects the replay trace of one :func:`place_and_route` run."""

    def __init__(self, level: TraceLevel | str | int = TraceLevel.BASIC) -> None:
        self.level = TraceLevel.parse(level)
        self.events: list[dict[str, Any]] = []
        self.design: dict[str, Any] | None = None
        self.config: dict[str, Any] | None = None
        self.final: dict[str, Any] | None = None

    def wants(self, level: TraceLevel) -> bool:
        """Whether events of ``level`` are being recorded."""
        return self.level != TraceLevel.NONE and self.level >= level

    def emit(
        self, phase: str, type: str, *, level: TraceLevel = TraceLevel.BASIC, **data: Any
    ) -> None:
        """Record one event if ``level`` is enabled.  ``data`` must be JSON-native."""
        if not self.wants(level):
            return
        if phase not in PHASES:
            raise CompileError(f"unknown trace phase {phase!r}")
        self.events.append({"seq": len(self.events), "phase": phase, "type": type, **data})

    def begin(self, netlist: PhysicalNetlist, config: PnRConfig) -> None:
        """Snapshot the (unplaced) design and the configuration."""
        self.design = design_records(netlist)
        self.config = config.to_dict()

    def finish(self, final: dict[str, Any]) -> None:
        self.final = final

    def fail(self, message: str) -> None:
        """Close an interrupted trace (e.g. after an internal error) so it can
        still be written and inspected."""
        if self.final is None:
            self.final = {"success": False, "failure": {"stage": "internal", "reason": message}}

    def header(self) -> dict[str, Any]:
        return {
            "schema": TRACE_SCHEMA,
            "generator": "redc",
            "trace_level": self.level.label,
            "coordinate_system": COORDINATE_SYSTEM,
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
        """Write ``.pnr.jsonl`` as JSON Lines, anything else as one JSON file."""
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
    """Read a ``.pnr.json`` or ``.pnr.jsonl`` trace back into the JSON form."""
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
