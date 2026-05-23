"""Out-of-band decoded view of v2 token streams.

Public surface for the decoded helpers. Individual submodules own their
concerns (run-length masks, custom-float encoding, vocab introspection);
this ``__init__`` re-exports only the symbols external callers are
expected to use.

The Phase 4.2 ``DecodedFunction`` dataclass + single-function extractor
+ recursive splicer were removed in Phase 5; the production decode path
is the batch-vectorized pipeline in
:mod:`tokenizer.aligned_data.loader.batch_decode`.
"""

from __future__ import annotations

from tokenizer.tokens import Category

from .category_tokens import (
    resolve_category_token_ids,
    resolve_number_token_ids,
    resolve_value_negative_token_id,
)
from .custom_float import (
    INFNAN_EXPONENT_UNBIASED,
    TARGET_EXPONENT_BIAS,
    TARGET_EXPONENT_BITS,
    TARGET_SIGNIFICAND_BITS,
)

__all__ = [
    "Category",
    "INFNAN_EXPONENT_UNBIASED",
    "TARGET_EXPONENT_BIAS",
    "TARGET_EXPONENT_BITS",
    "TARGET_SIGNIFICAND_BITS",
    "resolve_category_token_ids",
    "resolve_number_token_ids",
    "resolve_value_negative_token_id",
]
