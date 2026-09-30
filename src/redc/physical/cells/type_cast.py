"""Type-cast cells, loaded from ``type_cast.yaml`` at import."""

from __future__ import annotations

from ..components import TypeCast
from ._loader import load_family

#: Convention-name -> :class:`TypeCast` for every variant in the YAML.
TYPE_CASTS: dict[str, TypeCast] = load_family("type_cast.yaml", TypeCast)
