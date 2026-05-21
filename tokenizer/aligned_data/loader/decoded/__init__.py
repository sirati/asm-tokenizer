"""Out-of-band decoded view of v2 token streams.

Public surface for the splice-and-decode pipeline. Individual submodules
own their concerns (run-length masks, custom-float encoding, vocab
introspection, decoded-function dataclass, single-function extractor,
recursive splicer); this ``__init__`` re-exports only the symbols
external callers are expected to use.

Re-exports are added by their owning phase. Phase 1.3 contributes the
two ``category_tokens`` resolvers — single source of truth for
TokenType-to-uint16-vocab-id mappings consumed everywhere else in
``decoded/``.
"""

from __future__ import annotations

from .category_tokens import (
    resolve_category_token_ids,
    resolve_number_token_ids,
)

__all__ = [
    "resolve_category_token_ids",
    "resolve_number_token_ids",
]
