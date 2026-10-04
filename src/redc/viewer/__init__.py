"""Browser 3D replay viewers for place-and-route traces of both physical backends.

Kept apart from the compiler: they only read the trace JSON.

* ``redc.pnr.trace.v1`` -- the coarse ``physical`` backend.  A static HTML
  page (``pnr_viewer.html``) plus one ES module (``pnr_viewer.js``);
  :func:`render_pnr_html` / :func:`write_pnr_html`.
* ``redc.physical-primitive.pnr.v1`` -- the ``physical-primitive`` backend
  (one-bit gates, one block per coordinate, one-bit redstone routes).
  ``primitive_viewer.html`` plus ``primitive_viewer.js``;
  :func:`render_primitive_pnr_html` / :func:`write_primitive_pnr_html`
  (see :mod:`redc.viewer.primitive`).

:func:`render_trace_html` / :func:`write_trace_html` pick the right viewer
from the trace's schema, :func:`load_any_trace` reads a ``.json`` /
``.jsonl`` trace of either backend and :func:`trace_backend` names the
backend that wrote one.

Both viewers use Three.js from the jsDelivr CDN, so a generated page needs
network access to load Three.js.  The renderers inline the page, the script
and the trace into one portable ``.html`` file -- no server, Python or YAML
needed to view it.  Each page can also load another trace of its backend via
its file input.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..parser import CompileError
from .primitive import (
    PRIMITIVE_SCRIPT,
    PRIMITIVE_TEMPLATE,
    PRIMITIVE_TRACE_SCHEMA,
    check_primitive_trace,
    load_primitive_trace,
    render_primitive_pnr_html,
    write_primitive_pnr_html,
)

#: The coarse backend's trace schema (``redc.physical.pnr.trace.TRACE_SCHEMA``),
#: spelled out so importing the viewer does not import the coarse backend.
TRACE_SCHEMA = "redc.pnr.trace.v1"

#: Every replay-trace schema a viewer here can display.
TRACE_SCHEMAS = (TRACE_SCHEMA, PRIMITIVE_TRACE_SCHEMA)

VIEWER_DIR = Path(__file__).resolve().parent
TEMPLATE = VIEWER_DIR / "pnr_viewer.html"
SCRIPT = VIEWER_DIR / "pnr_viewer.js"

_JS_SLOT = "/*__REDC_VIEWER_JS__*/"
_TRACE_SLOT = "__REDC_TRACE_JSON__"
_TITLE_SLOT = "__REDC_TITLE__"


def _load_coarse_trace(path: str | Path) -> dict[str, Any]:
    from ..physical.pnr.trace import load_trace

    return load_trace(path)


def __getattr__(name: str) -> Any:
    # ``redc.viewer.load_trace`` (the coarse loader) used to be imported eagerly;
    # it stays available without importing the coarse backend at import time.
    if name == "load_trace":
        from ..physical.pnr.trace import load_trace

        return load_trace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        data = check_trace(_load_coarse_trace(source))
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


# -- either backend ------------------------------------------------------------------

_BACKENDS = {TRACE_SCHEMA: "physical", PRIMITIVE_TRACE_SCHEMA: "physical-primitive"}

# A ``.json`` trace whose first key is "schema" (both recorders write it first).
_LEADING_SCHEMA = re.compile(rb'\A\s*\{\s*"schema"\s*:\s*("(?:[^"\\]|\\.)*")')


def _unknown_schema(schema: Any, where: str | Path | None = None) -> CompileError:
    prefix = f"{where}: " if where else ""
    return CompileError(
        f"{prefix}not a P&R replay trace: schema is {schema!r}, "
        f"expected {TRACE_SCHEMA!r} or {PRIMITIVE_TRACE_SCHEMA!r}"
    )


def _file_schema(path: Path) -> Any:
    """The schema a trace file declares, without parsing a large file twice:
    the ``.jsonl`` header record, else a leading ``"schema"`` key, else (any
    other key order) the parsed document's ``schema``."""
    if path.suffix == ".jsonl":
        with path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if isinstance(record, dict) and record.get("record") == "header":
                        return record.get("schema")
                    return None
        return None
    with path.open("rb") as handle:
        head = handle.read(4096)
    match = _LEADING_SCHEMA.match(head)
    if match:
        return json.loads(match.group(1))
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("schema") if isinstance(data, dict) else None


def load_any_trace(path: str | Path) -> dict[str, Any]:
    """Read a ``.json`` / ``.jsonl`` replay trace written by either backend.

    The schema is detected from the file (a ``.jsonl`` header record, or the
    document's ``schema``) and the file is read by that backend's loader:
    :func:`redc.physical.pnr.trace.load_trace` for ``redc.pnr.trace.v1``,
    :func:`redc.physical_primitive.pnr.trace.load_trace` for
    ``redc.physical-primitive.pnr.v1``.  Any other schema raises
    :class:`CompileError` naming both; unreadable files raise ``OSError`` and
    invalid JSON ``ValueError``."""
    path = Path(path)
    schema = _file_schema(path)
    if schema == TRACE_SCHEMA:
        return _load_coarse_trace(path)
    if schema == PRIMITIVE_TRACE_SCHEMA:
        return load_primitive_trace(path)
    raise _unknown_schema(schema, path)


def trace_backend(trace: Mapping[str, Any] | str | Path) -> str:
    """``"physical"`` or ``"physical-primitive"``: the backend that wrote
    ``trace`` (a trace dict, or a path to a trace file)."""
    if isinstance(trace, (str, Path)):
        schema = _file_schema(Path(trace))
        where: str | Path | None = trace
    else:
        schema = trace.get("schema") if isinstance(trace, Mapping) else None
        where = None
    backend = _BACKENDS.get(schema) if isinstance(schema, str) else None
    if backend is None:
        raise _unknown_schema(schema, where)
    return backend


def render_trace_html(trace: Mapping[str, Any] | str | Path, *, title: str | None = None) -> str:
    """The replay viewer page for a trace of either backend: the coarse viewer
    for ``redc.pnr.trace.v1``, the primitive viewer for
    ``redc.physical-primitive.pnr.v1`` (``trace`` is a dict or a path)."""
    if not isinstance(trace, (str, Path, Mapping)):
        raise CompileError("P&R trace must be a JSON object or a path to a trace file")
    if trace_backend(trace) == "physical":
        return render_pnr_html(trace, title=title)
    return render_primitive_pnr_html(trace, title=title)


def write_trace_html(
    trace: Mapping[str, Any] | str | Path, output: str | Path, *, title: str | None = None
) -> Path:
    """Write :func:`render_trace_html` to ``output`` (parent directories created)."""
    output = Path(output)
    page = render_trace_html(trace, title=title)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


__all__ = [
    "PRIMITIVE_SCRIPT",
    "PRIMITIVE_TEMPLATE",
    "PRIMITIVE_TRACE_SCHEMA",
    "TRACE_SCHEMA",
    "TRACE_SCHEMAS",
    "check_primitive_trace",
    "check_trace",
    "load_any_trace",
    "load_primitive_trace",
    "render_pnr_html",
    "render_primitive_pnr_html",
    "render_trace_html",
    "trace_backend",
    "write_pnr_html",
    "write_primitive_pnr_html",
    "write_trace_html",
]
