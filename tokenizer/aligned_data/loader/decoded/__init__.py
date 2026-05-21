"""Out-of-band decoded view of v2 token streams.

Public surface for the splice-and-decode pipeline. Individual submodules
own their concerns (run-length masks, custom-float encoding, vocab
introspection, decoded-function dataclass, single-function extractor,
recursive splicer); this ``__init__`` re-exports only the symbols
external callers are expected to use.

Re-exports are added by their owning phase. Phase 1.2 contributed the
custom-float constants; Phase 1.3 contributed the two ``category_tokens``
resolvers — single source of truth for TokenType-to-uint16-vocab-id
mappings consumed everywhere else in ``decoded/``. Phase 4.2 surfaces
the ``DecodedFunction`` dataclass, the single-function extractor, the
recursive splicer, and re-exports ``Category`` from ``tokenizer.tokens``
so consumers do not have to reach into the deep tokens module.
"""

from __future__ import annotations

from tokenizer.tokens import Category

from .category_tokens import (
    resolve_category_token_ids,
    resolve_number_token_ids,
)
from .custom_float import (
    INFNAN_EXPONENT_UNBIASED,
    TARGET_EXPONENT_BIAS,
    TARGET_EXPONENT_BITS,
    TARGET_SIGNIFICAND_BITS,
)
from .decoded_function import DecodedFunction
from .extract import decode_raw_tokens
from .splice import IDENTITY_SENTINEL, splice_with_callees

__all__ = [
    "Category",
    "DecodedFunction",
    "IDENTITY_SENTINEL",
    "INFNAN_EXPONENT_UNBIASED",
    "TARGET_EXPONENT_BIAS",
    "TARGET_EXPONENT_BITS",
    "TARGET_SIGNIFICAND_BITS",
    "decode_raw_tokens",
    "resolve_category_token_ids",
    "resolve_number_token_ids",
    "splice_with_callees",
]
