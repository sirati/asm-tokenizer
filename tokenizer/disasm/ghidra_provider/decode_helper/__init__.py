"""Provider-owned decode-helper package.

Public surface (unchanged from the former single-module ``decode_helper.py``):
``_GhidraDecodeHelper`` is the facade injected into the view wrappers.
"""

from __future__ import annotations

from tokenizer.disasm.ghidra_provider.decode_helper.helper import (
    _GhidraDecodeHelper,
)

__all__ = ["_GhidraDecodeHelper"]
