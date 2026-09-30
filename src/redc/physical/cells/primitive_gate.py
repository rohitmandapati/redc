"""Boolean gate cells, loaded from ``primitive_gate.yaml`` at import."""

from __future__ import annotations

from ..components import PrimitiveGate
from ._loader import load_family

#: Convention-name -> :class:`PrimitiveGate` for every variant in the YAML.
GATES: dict[str, PrimitiveGate] = load_family("primitive_gate.yaml", PrimitiveGate)
