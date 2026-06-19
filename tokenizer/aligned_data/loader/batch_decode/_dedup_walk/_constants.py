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
    "ROOT_FUNC_SLOT",
    "_CALL_TARGET_TYPE_TO_CATEGORY",
    "_CATEGORY_TO_SHIFTED_ID",
    "_COUNTER_CATEGORY_TO_SLOT",
    "_FUNCTION_CATEGORY_TO_SLOT",
    "_SHIFTED_ID_TO_CATEGORY",
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


# Inverse of ``_CATEGORY_TO_SHIFTED_ID``. The IDENTITY-band per-row walker
# (and the inspector's BatchDecode rendering backend) reads a shifted id off
# the token stream and needs to dispatch to the owning ``Category``; pinning
# both directions in the same module keeps the forward + inverse maps
# automatically consistent (a forward-map change reshapes the inverse on
# import). Built once at module load; injection-safe via the explicit
# eight-Category total in the forward map.
_SHIFTED_ID_TO_CATEGORY: dict[int, Category] = {
    shifted_id: cat for cat, shifted_id in _CATEGORY_TO_SHIFTED_ID.items()
}
assert len(_SHIFTED_ID_TO_CATEGORY) == len(_CATEGORY_TO_SHIFTED_ID), (
    "_CATEGORY_TO_SHIFTED_ID values must be unique; got a collision while "
    "building the inverse map"
)


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


# ---------------------------------------------------------------------------
# Dense per-partition Category -> slot codes for the Rust remap kernel.
#
# The kernel (``dedup_hashmap.apply_remap_walk``) addresses FUNCTION and
# COUNTER categories by a dense int slot, NOT the ``Category`` enum -- it
# only needs the partition + a per-category counter space. These two maps
# are the SINGLE site that assigns the slots; the order MUST match
# ``FUNCTION_CATEGORIES`` / ``COUNTER_CATEGORIES`` so the kernel's per-slot
# FID-inverse output lines up with the pass-2 ``[per_variant[cat] for cat
# in FUNCTION_CATEGORIES]`` concatenation byte-for-byte. The maps are
# total + collision-checked below.
# ---------------------------------------------------------------------------
_FUNCTION_CATEGORY_TO_SLOT: dict[Category, int] = {
    cat: slot for slot, cat in enumerate(FUNCTION_CATEGORIES)
}
_COUNTER_CATEGORY_TO_SLOT: dict[Category, int] = {
    cat: slot for slot, cat in enumerate(COUNTER_CATEGORIES)
}
assert len(_FUNCTION_CATEGORY_TO_SLOT) == len(FUNCTION_CATEGORIES), (
    "FUNCTION_CATEGORIES has a duplicate Category; the kernel slot map "
    "must be total + collision-free"
)
assert len(_COUNTER_CATEGORY_TO_SLOT) == len(COUNTER_CATEGORIES), (
    "COUNTER_CATEGORIES has a duplicate Category; the kernel slot map "
    "must be total + collision-free"
)
# The LOCAL_FUNC root-seed slot (ALG-3 + ALG-9). LOCAL_FUNC is
# ``FUNCTION_CATEGORIES[0]`` by the canonical layout.
ROOT_FUNC_SLOT: int = _FUNCTION_CATEGORY_TO_SLOT[Category.LOCAL_FUNC]
