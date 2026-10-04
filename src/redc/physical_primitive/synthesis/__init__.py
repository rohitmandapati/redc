"""Primitive synthesis: typed IR operations -> one-bit Boolean/state primitives.

* :mod:`.builder`   -- :class:`PrimitiveBuilder`, the only way primitives are made;
* :mod:`.logic`     -- bitwise maps, balanced reductions, the 2:1 mux structure;
* :mod:`.registry`  -- recipes keyed by exact :class:`~redc.signature.OperationSignature`;
* :mod:`.recipes`   -- one family recipe per IR op (:data:`DEFAULT_SYNTHESIS`);
* :mod:`.interface` -- pads vs. peripheral realization of module ports;
* :mod:`.lower`     -- :func:`synthesize_to_primitives`, the four-pass driver.

Nothing in this package knows about Minecraft.
"""

from .builder import PrimitiveBuilder
from .interface import (
    DEFAULT_INTERFACE_POLICY,
    DefaultInterfacePolicy,
    InterfacePolicy,
    PadInterfacePolicy,
    PortRealization,
)
from .lower import synthesize_to_primitives
from .registry import RecipeEntry, SynthesisRecipe, SynthesisRegistry

__all__ = [
    "DEFAULT_INTERFACE_POLICY",
    "DefaultInterfacePolicy",
    "InterfacePolicy",
    "PadInterfacePolicy",
    "PortRealization",
    "PrimitiveBuilder",
    "RecipeEntry",
    "SynthesisRecipe",
    "SynthesisRegistry",
    "synthesize_to_primitives",
]
