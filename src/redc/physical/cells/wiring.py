"""Signal-carrying wiring cells, loaded from ``wiring.yaml`` at import."""

from __future__ import annotations

from ..components import Wiring
from ._loader import load_family

#: Convention-name -> :class:`Wiring` for every variant in the YAML.
WIRING: dict[str, Wiring] = load_family("wiring.yaml", Wiring)
