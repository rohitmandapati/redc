"""Datapath operation cells, loaded from ``operation.yaml`` at import."""

from __future__ import annotations

from ..components import Operation
from ._loader import load_family

#: Convention-name -> :class:`Operation` for every variant in the YAML.
OPERATIONS: dict[str, Operation] = load_family("operation.yaml", Operation)
