"""Browser 3D replay viewer for ``redc.physical-primitive.pnr.v1`` traces.

The primitive backend's counterpart of :func:`redc.viewer.render_pnr_html`:
a static page (``primitive_viewer.html``) plus one ES module
(``primitive_viewer.js``) using Three.js from the jsDelivr CDN.
:func:`render_primitive_pnr_html` inlines both files and embeds the trace,
producing one portable ``.html`` file -- no server, Python or technology
library needed to view it (the generated page needs network access only to
load Three.js).  The page can also load any other primitive trace through
its file input.

The trace format is specified in ``docs/physical-primitive-trace.md``; the
viewer re-validates it in JavaScript (``validateTrace``).
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..parser import CompileError
from ..physical_primitive.pnr.trace import TRACE_SCHEMA as PRIMITIVE_TRACE_SCHEMA
from ..physical_primitive.pnr.trace import load_trace as load_primitive_trace

VIEWER_DIR = Path(__file__).resolve().parent
PRIMITIVE_TEMPLATE = VIEWER_DIR / "primitive_viewer.html"
PRIMITIVE_SCRIPT = VIEWER_DIR / "primitive_viewer.js"

_JS_SLOT = "/*__REDC_VIEWER_JS__*/"
_TRACE_SLOT = "__REDC_TRACE_JSON__"
_TITLE_SLOT = "__REDC_TITLE__"
_DEFAULT_TITLE = "RedC primitive P&R replay"


def _fail(message: str) -> CompileError:
    return CompileError(f"{PRIMITIVE_TRACE_SCHEMA} trace: {message}")


def check_primitive_trace(trace: Any) -> Mapping[str, Any]:
    """Structural validation of a primitive replay trace (the viewer re-checks in JS).

    Checks the schema, that the design lists its cells, instances and nets,
    that every instance uses a defined cell, and that the events carry a
    contiguous ``seq``.  Every error names the schema."""
    if not isinstance(trace, Mapping):
        raise _fail("must be a JSON object")
    schema = trace.get("schema")
    if schema != PRIMITIVE_TRACE_SCHEMA:
        raise CompileError(
            f"not a primitive P&R replay trace: schema is {schema!r}, expected {PRIMITIVE_TRACE_SCHEMA!r}"
        )
    design = trace.get("design")
    if not isinstance(design, Mapping) or not all(
        isinstance(design.get(key), list) for key in ("cells", "instances", "nets")
    ):
        raise _fail("the design has no cells / instances / nets lists")
    names = {cell.get("name") for cell in design["cells"] if isinstance(cell, Mapping)}
    for index, inst in enumerate(design["instances"]):
        if not isinstance(inst, Mapping):
            raise _fail(f"design instance {index} is not an object")
        if inst.get("cell") not in names:
            raise _fail(f"instance {inst.get('id', index)} uses unknown cell {inst.get('cell')!r}")
    events = trace.get("events")
    if not isinstance(events, list):
        raise _fail("no events list")
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or event.get("seq") != index or not isinstance(event.get("type"), str):
            raise _fail(f"event {index} is malformed or out of sequence (seq must count 0, 1, 2, ...)")
    final = trace.get("final")
    if final is not None and not isinstance(final, Mapping):
        raise _fail("final must be an object")
    return trace


def render_primitive_pnr_html(trace: Mapping[str, Any] | str | Path, *, title: str | None = None) -> str:
    """A self-contained primitive viewer page with ``trace`` embedded.

    ``trace`` is the trace dict or a path to a ``.json`` / ``.jsonl`` trace."""
    if isinstance(trace, (str, Path)):
        source = Path(trace)
        data = check_primitive_trace(load_primitive_trace(source))
        title = title or f"{_DEFAULT_TITLE} - {source.name}"
    else:
        data = check_primitive_trace(trace)
    # "<" can only occur inside JSON strings, where < is an equivalent
    # escape -- this keeps "</script>" and "<!--" out of the embedded block.
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    page = PRIMITIVE_TEMPLATE.read_text(encoding="utf-8")
    page = page.replace(_TITLE_SLOT, html.escape(title or _DEFAULT_TITLE))
    page = page.replace(_JS_SLOT, PRIMITIVE_SCRIPT.read_text(encoding="utf-8"))
    return page.replace(_TRACE_SLOT, payload)


def write_primitive_pnr_html(
    trace: Mapping[str, Any] | str | Path, output: str | Path, *, title: str | None = None
) -> Path:
    """Write :func:`render_primitive_pnr_html` to ``output`` (parents created)."""
    output = Path(output)
    page = render_primitive_pnr_html(trace, title=title)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


__all__ = [
    "PRIMITIVE_SCRIPT",
    "PRIMITIVE_TEMPLATE",
    "PRIMITIVE_TRACE_SCHEMA",
    "check_primitive_trace",
    "load_primitive_trace",
    "render_primitive_pnr_html",
    "write_primitive_pnr_html",
]
