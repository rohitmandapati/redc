"""State-element cells, loaded from ``register.yaml`` at import."""

from __future__ import annotations

from ..components import Register
from ._loader import load_family

#: Convention-name -> :class:`Register` for every variant in the YAML.
REGISTERS: dict[str, Register] = load_family("register.yaml", Register)
