"""Tests for :attr:`BatchDecodeResult.fid_per_category_counts`.

Single concern: pin the per-row per-FUNCTION-Category deduped counter
cardinality sidecar that consumers (notably ``BatchDecodeBackend``) use
to slice :attr:`BatchDecodeResult.fid_sidecar` per Category without
re-scanning the identity stream (which would over-count under recursive
or repeated FID references -- multiple in-stream occurrences of the
same counter id collapse to ONE sidecar entry per Category).

Reuses the synthetic-fixture helpers from :mod:`test_assemble` to drive
:func:`assemble_batch` end-to-end at stage 4. The new field lives on
:class:`BatchDecodeResult` so the assemble boundary is the natural test
plane.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._assemble import assemble_batch
from tokenizer.aligned_data.loader.batch_decode._batch_layout import UINT32_MAX
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    FUNCTION_CATEGORIES,
)
from tokenizer.tokens import Category

from .test_assemble import (
    _LOCAL_FUNC_TOKEN,
    _PLT_FUNC_TOKEN,
    _build_batch,
    _build_call_target,
)


# Shifted token id for EXT_FUNC (post strip+shift; mirrors the
# ``_LOCAL_FUNC_TOKEN`` + ``_PLT_FUNC_TOKEN`` derivation in
# :mod:`test_assemble`). Computed locally to keep this test focused.
from tokenizer.token_manager import VocabularyManager  # noqa: E402

_RESERVED = VocabularyManager._V2_RESERVED_DIGIT_COUNT
_IDENT_BASE = VocabularyManager._V2_IDENTITY_BLOCK_START
_EXT_FUNC_TOKEN = _IDENT_BASE + 3 - _RESERVED  # = 11


def test_single_variant_single_category_shape_and_value() -> None:
    """A 1-row batch with a single LOCAL CT (root + one LOCAL callee in
    the call_targets table). Expected counts:

    * LOCAL_FUNC = 2 (root seed counter 0 + minted callee counter 1)
    * PLT_FUNC   = 0
    * EXT_FUNC   = 0

    Pins the array shape (``u32[batch_size, 3]``) + dtype + column
    ordering (mirrors :data:`FUNCTION_CATEGORIES`).
    """

    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=np.asarray(
            [_LOCAL_FUNC_TOKEN, _LOCAL_FUNC_TOKEN],  # prepend + one in-stream
            dtype=np.uint16,
        ),
        in_stream_caller_local_ids=[0],
        section_call_targets=[(200, CallTargetType.LOCAL)],
    )
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray([[0, 0]], dtype=np.uint32),
    )

    result = assemble_batch(batch, context_len=4, include_fid_sidecar=True)

    assert result.fid_per_category_counts is not None
    assert result.fid_per_category_counts.dtype == np.uint32
    assert result.fid_per_category_counts.shape == (1, 3)
    # Column order matches FUNCTION_CATEGORIES = (LOCAL, PLT, EXT).
    assert FUNCTION_CATEGORIES == (
        Category.LOCAL_FUNC,
        Category.PLT_FUNC,
        Category.EXT_FUNC,
    )
    np.testing.assert_array_equal(
        result.fid_per_category_counts,
        np.array([[2, 0, 0]], dtype=np.uint32),
    )
    # Sanity-cross-check with fid_sidecar slicing: row 0's LOCAL segment
    # is the first 2 entries; PLT segment empty; EXT segment empty.
    counts = result.fid_per_category_counts[0]
    assert counts.sum() == int(
        result.fid_row_offsets[1] - result.fid_row_offsets[0]
    )


def test_multi_category_per_row_counts_match_segment_lengths() -> None:
    """One CT references one LOCAL callee + one PLT callee + one EXT
    callee. Per-row counts pin to the per-Category sidecar segment
    lengths (LOCAL=2 because the root takes counter 0; PLT=1; EXT=1)."""

    expanded = np.asarray(
        [
            _LOCAL_FUNC_TOKEN,  # prepend (root self)
            _LOCAL_FUNC_TOKEN,  # in-stream LOCAL callee
            _PLT_FUNC_TOKEN,  # in-stream PLT callee
            _EXT_FUNC_TOKEN,  # in-stream EXT callee
        ],
        dtype=np.uint16,
    )
    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=expanded,
        in_stream_caller_local_ids=[0, 0, 0],
        section_call_targets=[
            (200, CallTargetType.LOCAL),
            (300, CallTargetType.PLT),
            (400, CallTargetType.EXTERN),
        ],
    )
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray([[0, 0]], dtype=np.uint32),
    )

    result = assemble_batch(batch, context_len=6, include_fid_sidecar=True)

    assert result.fid_per_category_counts is not None
    np.testing.assert_array_equal(
        result.fid_per_category_counts,
        np.array([[2, 1, 1]], dtype=np.uint32),
    )
    # Sidecar segments derivable from the counts: LOCAL[0:2], PLT[2:3],
    # EXT[3:4] within this row's slice.
    assert result.fid_sidecar is not None
    row_lo, row_hi = (
        int(result.fid_row_offsets[0]),
        int(result.fid_row_offsets[1]),
    )
    row_slice = result.fid_sidecar[row_lo:row_hi]
    counts = result.fid_per_category_counts[0]
    cum = np.cumsum(counts)
    # Root seed at LOCAL[0] = 100; LOCAL callee at LOCAL[1] = 200.
    assert row_slice[: cum[0]].tolist() == [100, 200]
    # PLT[0] = 300.
    assert row_slice[cum[0] : cum[1]].tolist() == [300]
    # EXT[0] = 400.
    assert row_slice[cum[1] : cum[2]].tolist() == [400]


def test_recursive_local_call_counts_dedup_to_one() -> None:
    """Root calls itself: two in-stream LOCAL_FUNC slots both point at
    the root's FID (caller-local 0 = self). After dedup the LOCAL
    sidecar segment has ONE entry (the root) and the per-row count is 1,
    NOT 2 (the in-stream occurrence count).

    Pins decision #16 / D1: ``fid_per_category_counts`` is the deduped
    counter cardinality, not the stream occurrence count -- so
    consumers cannot derive it via ``np.bincount`` over the identity
    stream.
    """

    expanded = np.asarray(
        [
            _LOCAL_FUNC_TOKEN,  # prepend (root self)
            _LOCAL_FUNC_TOKEN,  # in-stream slot 1 -> root self again
            _LOCAL_FUNC_TOKEN,  # in-stream slot 2 -> root self again
        ],
        dtype=np.uint16,
    )
    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=expanded,
        in_stream_caller_local_ids=[0, 0],
        section_call_targets=[
            (100, CallTargetType.LOCAL),  # self-reference
        ],
    )
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray([[0, 0]], dtype=np.uint32),
    )

    result = assemble_batch(batch, context_len=4, include_fid_sidecar=True)

    assert result.fid_per_category_counts is not None
    # LOCAL count = 1 (root only -- the two in-stream self-refs collapse
    # to the same counter id 0); PLT = 0; EXT = 0.
    np.testing.assert_array_equal(
        result.fid_per_category_counts,
        np.array([[1, 0, 0]], dtype=np.uint32),
    )
    # Cross-check: the row's sidecar slice is exactly the root FID.
    assert result.fid_sidecar is not None
    assert result.fid_row_offsets is not None
    row_lo, row_hi = (
        int(result.fid_row_offsets[0]),
        int(result.fid_row_offsets[1]),
    )
    assert result.fid_sidecar[row_lo:row_hi].tolist() == [100]


def test_include_fid_sidecar_false_returns_none() -> None:
    """When ``include_fid_sidecar=False`` (default), the sidecar fields
    AND ``fid_per_category_counts`` are all ``None`` -- the same
    ``Optional`` semantics mirror the other two sidecars."""

    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=np.asarray(
            [_LOCAL_FUNC_TOKEN, _LOCAL_FUNC_TOKEN], dtype=np.uint16
        ),
        in_stream_caller_local_ids=[0],
        section_call_targets=[(200, CallTargetType.LOCAL)],
    )
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray([[0, 0]], dtype=np.uint32),
    )

    result = assemble_batch(batch, context_len=4)

    assert result.fid_sidecar is None
    assert result.fid_row_offsets is None
    assert result.fid_per_category_counts is None


def test_padding_row_counts_are_zero() -> None:
    """Padding rows (sentinel ``(UINT32_MAX, UINT32_MAX)``) contribute
    ``(0, 0, 0)`` per the shared padding-row semantics."""

    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=np.asarray(
            [_LOCAL_FUNC_TOKEN, _LOCAL_FUNC_TOKEN], dtype=np.uint16
        ),
        in_stream_caller_local_ids=[0],
        section_call_targets=[(200, CallTargetType.LOCAL)],
    )
    sentinel = int(UINT32_MAX)
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray(
            [[0, 0], [sentinel, sentinel]], dtype=np.uint32
        ),
    )

    result = assemble_batch(batch, context_len=4, include_fid_sidecar=True)

    assert result.fid_per_category_counts is not None
    np.testing.assert_array_equal(
        result.fid_per_category_counts,
        np.array([[2, 0, 0], [0, 0, 0]], dtype=np.uint32),
    )


def test_resample_multi_mapped_rows_replicate_counts() -> None:
    """RESAMPLE: one variant referenced by two batch rows. Both rows get
    the SAME per-Category counts -- ``fid_per_category_counts`` is
    per-row dense (not cumulative), so the per-variant value is simply
    repeated, matching the per-row replication done by the FID sidecar.
    """

    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=np.asarray(
            [_LOCAL_FUNC_TOKEN, _LOCAL_FUNC_TOKEN, _PLT_FUNC_TOKEN],
            dtype=np.uint16,
        ),
        in_stream_caller_local_ids=[0, 0],
        section_call_targets=[
            (200, CallTargetType.LOCAL),
            (300, CallTargetType.PLT),
        ],
    )
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray(
            [[0, 0], [0, 0]], dtype=np.uint32
        ),
    )

    result = assemble_batch(batch, context_len=4, include_fid_sidecar=True)

    assert result.fid_per_category_counts is not None
    # LOCAL = 2 (root + callee), PLT = 1, EXT = 0; same for both rows.
    np.testing.assert_array_equal(
        result.fid_per_category_counts,
        np.array([[2, 1, 0], [2, 1, 0]], dtype=np.uint32),
    )
