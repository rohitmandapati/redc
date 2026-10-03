"""Browser 3D replay viewer for ``redc.pnr.trace.v1`` place-and-route traces.

Kept apart from the compiler: it only reads the trace JSON.  The viewer is a
static HTML page (``pnr_viewer.html``) plus one ES module (``pnr_viewer.js``)
using Three.js from the jsDelivr CDN, so the generated page needs network
access to load Three.js.  :func:`render_pnr_html` inlines both files and embeds
the trace, producing one portable ``.html`` file -- no server, Python or YAML
needed to view it.  The page can also load any other trace via its file input.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..parser import CompileError
from ..physical.pnr.trace import TRACE_SCHEMA, load_trace

VIEWER_DIR = Path(__file__).resolve().parent
TEMPLATE = VIEWER_DIR / "pnr_viewer.html"
SCRIPT = VIEWER_DIR / "pnr_viewer.js"

_JS_SLOT = "/*__REDC_VIEWER_JS__*/"
_TRACE_SLOT = "__REDC_TRACE_JSON__"
_TITLE_SLOT = "__REDC_TITLE__"


def check_trace(trace: Any) -> Mapping[str, Any]:
    """Minimal structural validation of a trace (the viewer re-checks in JS)."""
    if not isinstance(trace, Mapping):
        raise CompileError("P&R trace must be a JSON object")
    if trace.get("schema") != TRACE_SCHEMA:
        raise CompileError(
            f"not a P&R replay trace: schema is {trace.get('schema')!r}, expected {TRACE_SCHEMA!r}"
        )
    design = trace.get("design")
    if not isinstance(design, Mapping) or not all(
        isinstance(design.get(key), list) for key in ("components", "instances", "nets")
    ):
        raise CompileError("P&R trace has no design components / instances / nets")
    if not isinstance(trace.get("events"), list):
        raise CompileError("P&R trace has no events list")
    return trace


def render_pnr_html(trace: Mapping[str, Any] | str | Path, *, title: str | None = None) -> str:
    """A self-contained viewer page with ``trace`` embedded.

    ``trace`` is the trace dict or a path to a ``.pnr.json`` / ``.pnr.jsonl``."""
    if isinstance(trace, (str, Path)):
        source = Path(trace)
        data = check_trace(load_trace(source))
        title = title or f"RedC P&R replay - {source.name}"
    else:
        data = check_trace(trace)
    # "<" can only occur inside JSON strings, where < is an equivalent
    # escape -- this keeps "</script>" and "<!--" out of the embedded block.
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    page = TEMPLATE.read_text(encoding="utf-8")
    page = page.replace(_TITLE_SLOT, html.escape(title or "RedC P&R replay"))
    page = page.replace(_JS_SLOT, SCRIPT.read_text(encoding="utf-8"))
    return page.replace(_TRACE_SLOT, payload)


def write_pnr_html(
    trace: Mapping[str, Any] | str | Path, output: str | Path, *, title: str | None = None
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_pnr_html(trace, title=title), encoding="utf-8")
    return output


__all__ = ["check_trace", "render_pnr_html", "write_pnr_html"]
