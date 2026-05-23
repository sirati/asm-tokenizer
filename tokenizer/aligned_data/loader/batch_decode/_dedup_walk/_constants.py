"""Shared constants for the per-row dedup walk submodules.

Single concern: pin the unified-vocab IDENTITY-block layout +
``Category``-to-shifted-id mapping + Category partition (FUNCTION vs
COUNTER) used by every submodule under :mod:`._dedup_walk`. Constants
only; no algorithmic code (helpers live in :mod:`._helpers`).

Plan reference: ``batch_decode_plan.md`` ``## Vocab + wire format
reference`` + D4 (Category partition + per-Category counter spaces).
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category


__all__ = [
    "COUNTER_CATEGORIES",
    "FUNCTION_CATEGORIES",
    "NOT_FOUND_U16",
    "_CALL_TARGET_TYPE_TO_CATEGORY",
    "_CATEGORY_TO_SHIFTED_ID",
]


# ---------------------------------------------------------------------------
# Category partition (plan D4).
#
# The unified vocab's IDENTITY block has 8 Categories. The dedup-walk
# treats them in two disjoint groups:
#
# * FUNCTION categories: identity values dedupe across functions within a
#   row by ``function_name_ptr`` (FID) equality.
# * COUNTER categories: identity values renumber by a running per-row
#   offset; no dedup lookup.
#
# The two tuples below are the single source of truth for the partition;
# every dispatch path in :mod:`._dedup_walk` routes off them.
# ---------------------------------------------------------------------------
FUNCTION_CATEGORIES: tuple[Category, ...] = (
    Category.LOCAL_FUNC,
    Category.PLT_FUNC,
    Category.EXT_FUNC,
)

COUNTER_CATEGORIES: tuple[Category, ...] = (
    Category.BLOCK,
    Category.STRING_PTR,
    Category.RO_DATA_PTR,
    Category.RW_DATA_PTR,
    Category.JUMP_TABLE,
)


# ---------------------------------------------------------------------------
# Shifted vocab ids for the IDENTITY block (post strip+shift; ``id - 256``).
#
# The unified vocab pins the IDENTITY block at slots 264..271 in the
# pre-shift space (``_V2_IDENTITY_BLOCK_START`` = 264, count = 8). After
# stage 2's strip+shift, the model-facing token ids in
# ``expanded_token_ids`` are these vocab ids minus 256.
#
# The block layout is "user-canonical, then alphabetical" (plan vocab
# table + ALG-6):
#
#   offset 0 -> BLOCK_V2      -> shifted = 8
#   offset 1 -> LOCAL_FUNC    -> shifted = 9
#   offset 2 -> PLT_FUNC      -> shifted = 10
#   offset 3 -> EXT_FUNC      -> shifted = 11
#   offset 4 -> STRING_PTR    -> shifted = 12
#   offset 5 -> JUMP_TABLE    -> shifted = 13
#   offset 6 -> RO_DATA_PTR   -> shifted = 14
#   offset 7 -> RW_DATA_PTR   -> shifted = 15
#
# Resolving these once at import time keeps the per-row walk free of
# attribute lookups. The plan pins the unified vocab and every consumer
# asserts ``format_version=1``.
# ---------------------------------------------------------------------------
_V2_IDENTITY_BLOCK_START = VocabularyManager._V2_IDENTITY_BLOCK_START
_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT


def _shifted(category_offset: int) -> int:
    return _V2_IDENTITY_BLOCK_START + category_offset - _V2_RESERVED_DIGIT_COUNT


_CATEGORY_TO_SHIFTED_ID: dict[Category, int] = {
    Category.BLOCK: _shifted(0),
    Category.LOCAL_FUNC: _shifted(1),
    Category.PLT_FUNC: _shifted(2),
    Category.EXT_FUNC: _shifted(3),
    Category.STRING_PTR: _shifted(4),
    Category.JUMP_TABLE: _shifted(5),
    Category.RO_DATA_PTR: _shifted(6),
    Category.RW_DATA_PTR: _shifted(7),
}


# Map ``CallTargetType`` to the FUNCTION Category it produces. The
# call_target table's ``type`` field uses ``CallTargetType`` (LOCAL /
# PLT / EXTERN), while the dedup dispatch is keyed on ``Category``
# (LOCAL_FUNC / PLT_FUNC / EXT_FUNC). This is the single mapping site.
_CALL_TARGET_TYPE_TO_CATEGORY: dict[CallTargetType, Category] = {
    CallTargetType.LOCAL: Category.LOCAL_FUNC,
    CallTargetType.PLT: Category.PLT_FUNC,
    CallTargetType.EXTERN: Category.EXT_FUNC,
}


# Sentinel for ``HashMapU32U16.lookup_ndarray`` misses (plan ALG-3 +
# ``dedup_hashmap/src/lib.rs`` miss-sentinel table for unsigned ints =
# ``<dtype>::MAX``).
NOT_FOUND_U16: np.uint16 = np.uint16(0xFFFF)
