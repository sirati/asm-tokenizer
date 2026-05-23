"""Shared NUMBER-band vocab anchors for the 3c per-emitter package.

Single concern: derive band boundaries + canonical TokenType ordering
from :class:`VocabularyManager`'s source of truth so a vocab-layout shift
surfaces in one place instead of N. Module-local widths that don't cross
the package boundary (e.g. ``_emit_fixed_fp``'s payload widths) stay in
their owner module.
"""

from __future__ import annotations

from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType


__all__ = [
    "_V2_RESERVED_DIGIT_COUNT",
    "_V2_NUMBER_BLOCK_START",
    "_V2_NUMBER_BLOCK_COUNT",
    "_NUMBER_BAND_LO_SHIFTED",
    "_NUMBER_BAND_HI_SHIFTED",
    "_NUMBER_BLOCK_TOKEN_TYPES",
    "_FIXED_ROW_WIDTH",
]


_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT  # 256
_V2_NUMBER_BLOCK_START = VocabularyManager._V2_NUMBER_BLOCK_START  # 257
_V2_NUMBER_BLOCK_COUNT = VocabularyManager._V2_NUMBER_BLOCK_COUNT  # 7

# Post-strip / post-shift band: surviving expanded ids in ``[1, 8)``
# select a NUMBER-block carrier.
_NUMBER_BAND_LO_SHIFTED = (
    _V2_NUMBER_BLOCK_START - _V2_RESERVED_DIGIT_COUNT
)  # 1
_NUMBER_BAND_HI_SHIFTED = (
    _V2_NUMBER_BLOCK_START
    + _V2_NUMBER_BLOCK_COUNT
    - _V2_RESERVED_DIGIT_COUNT
)  # 8 (exclusive)


# Canonical NUMBER block ordering (plan vocab table + token_manager.py
# class docstring lines 94-97): VC2, F16, BF16, F32, F64, F80, F128.
# Indexed by ``shifted_id - 1`` (so shifted id 1 -> VC2 at index 0).
_NUMBER_BLOCK_TOKEN_TYPES: tuple[TokenType, ...] = (
    TokenType.VALUED_CONST_V2,
    TokenType.FLOAT16,
    TokenType.BFLOAT16,
    TokenType.FLOAT32,
    TokenType.FLOAT64,
    TokenType.FLOAT80,
    TokenType.FLOAT128,
)
assert len(_NUMBER_BLOCK_TOKEN_TYPES) == _V2_NUMBER_BLOCK_COUNT, (
    "_NUMBER_BLOCK_TOKEN_TYPES length must match VocabularyManager"
    "._V2_NUMBER_BLOCK_COUNT; a vocab-layout change touched one without "
    "the other."
)


# Per-TokenType row width (columns in idx_2d). F16/BF16/F32/F64/F80 emit
# 1 row covering the full payload; F128 emits ``chunk_count`` rows of
# 8 bytes each; VC2 emits ``K_visible`` rows of 8 bytes each.
_FIXED_ROW_WIDTH: dict[TokenType, int] = {
    TokenType.FLOAT16: 2,
    TokenType.BFLOAT16: 2,
    TokenType.FLOAT32: 4,
    TokenType.FLOAT64: 8,
    TokenType.FLOAT80: 10,
    TokenType.FLOAT128: 8,
    TokenType.VALUED_CONST_V2: 8,
}
