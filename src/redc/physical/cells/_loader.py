"""Turn a family YAML file into a library of :class:`Component` definitions.

Each family module (``operation.py``, ``primitive_gate.py``, ``wiring.py``)
calls :func:`load_family` with its YAML sibling and the family class.  The YAML
is the *contract* the placer and router read; the ``nbt`` field only names the
structure file that realises the cell, and is never opened here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ...ir import type_from_name
from ...parser import CompileError
from ..components import Component, Face, Port, PortDir


def load_family(yaml_name: str, cls: type[Component]) -> dict[str, Component]:
    """Load every variant in ``yaml_name`` as an instance of ``cls``.

    Returns a mapping from the variant's convention name to its component.
    """
    path = Path(__file__).with_name(yaml_name)
    data = yaml.safe_load(path.read_text()) or {}
    default_dtype = data.get("datatype")

    library: dict[str, Component] = {}
    for name, spec in (data.get("variants") or {}).items():
        library[name] = _build(name, spec, cls, default_dtype)
    return library


def _build(
    name: str, spec: dict[str, Any], cls: type[Component], default_dtype: str | None
) -> Component:
    variant_dtype = spec.get("datatype", default_dtype)

    def ports(entries: list[dict] | None, direction: PortDir) -> tuple[Port, ...]:
        built = []
        for entry in entries or []:
            dtype_name = entry.get("datatype", variant_dtype)
            if dtype_name is None:
                raise CompileError(f"{name}: pin {entry.get('name')!r} has no datatype")
            built.append(
                Port(
                    name=entry["name"],
                    dtype=type_from_name(dtype_name),
                    face=Face[entry["face"]],
                    offset=tuple(entry["offset"]),
                    direction=direction,
                )
            )
        return tuple(built)

    kwargs: dict[str, Any] = {
        "name": name,
        "latency": spec["latency"],
        "dim": tuple(spec["dim"]),
        "inputs": ports(spec.get("inputs"), PortDir.IN),
        "outputs": ports(spec.get("outputs"), PortDir.OUT),
        "nbt": spec.get("nbt"),
    }
    # Family-specific extra fields, passed through only when present.
    for extra in ("op", "init"):  # op: Operation / PrimitiveGate;  init: Register
        if extra in spec:
            kwargs[extra] = spec[extra]

    try:
        return cls(**kwargs)
    except TypeError as exc:  # unexpected field for this family
        raise CompileError(f"{name}: {exc}") from exc
