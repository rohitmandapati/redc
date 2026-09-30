"""The cell library: all component variations, loaded from YAML at import.

Each family lives in its own module beside a YAML file of variants.  ``LIBRARY``
is the merged convention-name -> :class:`~redc.physical.components.Component`
lookup the placer and router resolve against.
"""

from __future__ import annotations

from ..components import Component
from .operation import OPERATIONS
from .primitive_gate import GATES
from .wiring import WIRING

#: Every cell variation, keyed by its convention name.
LIBRARY: dict[str, Component] = {**OPERATIONS, **GATES, **WIRING}

__all__ = ["GATES", "LIBRARY", "OPERATIONS", "WIRING"]
