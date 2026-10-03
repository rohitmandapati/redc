"""Turn a family YAML file into a library of :class:`Component` definitions.

Each family module (``operation.py``, ``primitive_gate.py``, ...) calls
:func:`load_family` with its YAML sibling and the family class.  The YAML is the
*contract* the placer and router read; the ``nbt`` field only names the
structure file that realises the cell (``null`` until one exists), and is never
opened here.

A variant is either written out explicitly or, to avoid hand-copying one layout
per datatype, *parametric*: an ``expand`` mapping of placeholder -> list of
values produces one concrete, explicitly-named variant per combination, with
``{placeholder}`` substituted into the key and every string field.  For example::

    "{t}_shl_value-0-0-0_amount-0-0-1_out-0-0-2":
      expand: {t: [uint8, int8]}
      datatype: "{t}"
      ...

yields ``uint8_shl_...`` and ``int8_shl_...``.  ``unless_equal: [a, b]`` skips
combinations where placeholders ``a`` and ``b`` coincide (identity casts).

Pins are kept in exactly their YAML order: input declaration order is the
canonical IR operand order, so it is never sorted.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml

from ...ir import IRType, type_from_name
from ...parser import CompileError
from ..components import Component, Face, Port, PortDir

#: Every key a variant may carry; anything else (e.g. a per-cell ``init``) is a
#: contract violation, not something to silently ignore.
_VARIANT_KEYS = frozenset(
    {"op", "latency", "dim", "nbt", "inputs", "outputs", "datatype", "expand", "unless_equal"}
)
_PIN_KEYS = frozenset({"name", "face", "offset", "datatype"})


def load_family[C: Component](yaml_name: str, cls: type[C]) -> dict[str, C]:
    """Load every variant in ``yaml_name`` as an instance of ``cls``.

    Returns a mapping from the variant's convention name to its component, in
    YAML order (parametric variants expand in placeholder-value order).
    """
    path = Path(__file__).with_name(yaml_name)
    data = yaml.safe_load(path.read_text()) or {}
    default_dtype = data.get("datatype")

    library: dict[str, C] = {}
    for key, spec in (data.get("variants") or {}).items():
        for name, concrete in _expand(key, spec):
            if name in library:
                raise CompileError(f"{yaml_name}: duplicate variant {name!r}")
            library[name] = _build(name, concrete, cls, default_dtype)
    return library


def _expand(key: str, spec: dict[str, Any]):
    """Yield ``(name, spec)`` for each concrete variant ``key`` stands for."""
    unknown = set(spec) - _VARIANT_KEYS
    if unknown:
        raise CompileError(f"{key}: unknown variant field(s) {sorted(unknown)}")
    params: dict[str, list[str]] = spec.get("expand") or {}
    body = {k: v for k, v in spec.items() if k not in {"expand", "unless_equal"}}
    if not params:
        yield key, body
        return
    skip = spec.get("unless_equal") or []
    names = list(params)
    for values in itertools.product(*(params[n] for n in names)):
        binding = dict(zip(names, values))
        if skip and len({binding[n] for n in skip}) == 1:
            continue
        yield key.format(**binding), _substitute(body, binding)


def _substitute(value: Any, binding: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format(**binding)
    if isinstance(value, list):
        return [_substitute(v, binding) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, binding) for k, v in value.items()}
    return value


def _dtype(name: str, where: str) -> IRType:
    typ = type_from_name(name)
    if typ is None:
        raise CompileError(f"{where}: datatype {name!r} is not a value type")
    return typ


def _build[C: Component](
    name: str, spec: dict[str, Any], cls: type[C], default_dtype: str | None
) -> C:
    variant_dtype = spec.get("datatype", default_dtype)

    def ports(entries: list[dict] | None, direction: PortDir) -> tuple[Port, ...]:
        built = []
        for entry in entries or []:
            unknown = set(entry) - _PIN_KEYS
            if unknown:
                raise CompileError(
                    f"{name}: pin {entry.get('name')!r} has unknown field(s) {sorted(unknown)}"
                )
            dtype_name = entry.get("datatype", variant_dtype)
            if dtype_name is None:
                raise CompileError(f"{name}: pin {entry.get('name')!r} has no datatype")
            built.append(
                Port(
                    name=entry["name"],
                    dtype=_dtype(dtype_name, name),
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
    if "op" in spec:  # Operation / PrimitiveGate
        kwargs["op"] = spec["op"]

    try:
        return cls(**kwargs)
    except TypeError as exc:  # unexpected field for this family
        raise CompileError(f"{name}: {exc}") from exc
