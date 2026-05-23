"""FID-resolution branch of the identity decode (plan Decision 22).

When the splice walker calls ``_decode_to_staging(fids_per_category=...)``,
the per-position payload decode for categories in
:data:`FID_KEYED_CATEGORIES` switches from "inline payload IS the
identity value" (legacy) to "inline payload is a caller-local id; the
identity value is ``fids_per_category[c][local_id]``".  These tests
pin both the resolution path and the bounds-check sentinel behaviour
on synthetic single-function streams.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader.decoded.category_tokens import (
    FID_KEYED_CATEGORIES,
)
from tokenizer.aligned_data.loader.decoded.extract import _decode_to_staging
from tokenizer.tokens import Category, TokenType


# Pick vocab ids well above the reserved [0, 256) inline-digit range.
_LOCAL_FUNC_ID = 300
_PLT_FUNC_ID = 301
_EXT_FUNC_ID = 302
_BLOCK_ID = 303


def _id_token_ids():
    return {
        Category.LOCAL_FUNC: _LOCAL_FUNC_ID,
        Category.PLT_FUNC: _PLT_FUNC_ID,
        Category.EXT_FUNC: _EXT_FUNC_ID,
        Category.BLOCK: _BLOCK_ID,
    }


def _number_token_ids():
    return {}  # synthetic streams carry no number tokens here


def _stream(*tokens: int) -> np.ndarray:
    """Hand-build a uint16 stream where each ``LOCAL_FUNC`` (etc.) is
    immediately followed by an inline-digit byte holding the local id."""
    return np.array(tokens, dtype=np.uint16)


def test_fid_resolution_indexes_into_lookup():
    """Caller-local ids decode to the matching FIDs in the lookup."""
    stream = _stream(
        _BLOCK_ID,
        _LOCAL_FUNC_ID, 0,   # local id 0 -> FID 42
        _LOCAL_FUNC_ID, 1,   # local id 1 -> FID 99
        _LOCAL_FUNC_ID, 0,   # local id 0 -> FID 42 again
        _LOCAL_FUNC_ID, 2,   # local id 2 -> FID 7
    )
    fids = {
        Category.LOCAL_FUNC: np.array([42, 99, 7], dtype=np.uint32),
        Category.PLT_FUNC: np.empty(0, dtype=np.uint32),
        Category.EXT_FUNC: np.empty(0, dtype=np.uint32),
    }

    staging = _decode_to_staging(
        stream,
        id_token_ids=_id_token_ids(),
        number_token_ids=_number_token_ids(),
        fids_per_category=fids,
        value_negative_token_id=256,
        format_version=1,
    )
    arr = staging.identities[Category.LOCAL_FUNC]
    assert arr.dtype == np.uint32
    np.testing.assert_array_equal(arr, [42, 99, 42, 7])


def test_fid_resolution_out_of_range_emits_u32_sentinel():
    """Caller-local id beyond the lookup length resolves to ``0xFFFFFFFF``
    in the u32 staging; compaction downstream folds it to the public u16
    sentinel.
    """
    stream = _stream(
        _BLOCK_ID,
        _LOCAL_FUNC_ID, 0,   # local id 0 -> FID 42 (in range)
        _LOCAL_FUNC_ID, 5,   # local id 5 -> out of range -> sentinel
    )
    fids = {
        Category.LOCAL_FUNC: np.array([42, 99], dtype=np.uint32),
        Category.PLT_FUNC: np.empty(0, dtype=np.uint32),
        Category.EXT_FUNC: np.empty(0, dtype=np.uint32),
    }
    staging = _decode_to_staging(
        stream,
        id_token_ids=_id_token_ids(),
        number_token_ids=_number_token_ids(),
        fids_per_category=fids,
        value_negative_token_id=256,
        format_version=1,
    )
    arr = staging.identities[Category.LOCAL_FUNC]
    assert arr.dtype == np.uint32
    np.testing.assert_array_equal(arr, [42, 0xFFFFFFFF])


def test_fid_resolution_legacy_fallback_when_kwarg_absent():
    """``fids_per_category=None`` -> identity values are the decoded
    inline payloads directly (u16, plan decision 7 sentinel on overflow).
    """
    stream = _stream(
        _BLOCK_ID,
        _LOCAL_FUNC_ID, 42,
        _LOCAL_FUNC_ID, 99,
    )
    staging = _decode_to_staging(
        stream,
        id_token_ids=_id_token_ids(),
        number_token_ids=_number_token_ids(),
        fids_per_category=None,
        value_negative_token_id=256,
        format_version=1,
    )
    arr = staging.identities[Category.LOCAL_FUNC]
    assert arr.dtype == np.uint16
    np.testing.assert_array_equal(arr, [42, 99])


def test_non_fid_keyed_category_unaffected_by_kwarg():
    """``BLOCK`` (NOT in ``FID_KEYED_CATEGORIES``) keeps the legacy u16
    decode regardless of whether ``fids_per_category`` is supplied.
    """
    stream = _stream(
        _BLOCK_ID, 0,
        _BLOCK_ID, 3,
        _BLOCK_ID, 1,
    )
    fids = {
        Category.LOCAL_FUNC: np.array([42], dtype=np.uint32),
        Category.PLT_FUNC: np.empty(0, dtype=np.uint32),
        Category.EXT_FUNC: np.empty(0, dtype=np.uint32),
    }
    staging = _decode_to_staging(
        stream,
        id_token_ids=_id_token_ids(),
        number_token_ids=_number_token_ids(),
        fids_per_category=fids,
        value_negative_token_id=256,
        format_version=1,
    )
    block = staging.identities[Category.BLOCK]
    assert block.dtype == np.uint16
    np.testing.assert_array_equal(block, [0, 3, 1])


def test_fid_keyed_categories_membership_pinned():
    """Pin which categories the resolver branches on (plan Decision 26).

    A regression that adds or drops a FID-keyed category without updating
    the test would surface here loudly.
    """
    assert FID_KEYED_CATEGORIES == frozenset(
        {Category.LOCAL_FUNC, Category.PLT_FUNC, Category.EXT_FUNC}
    )
