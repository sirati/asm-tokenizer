"""Stage 2c unit tests -- surviving identity + number-chunk count masking.

Single concern: validate that
:func:`tokenizer.aligned_data.loader.batch_decode._surviving_counts.count_surviving`
counts post-shifted token positions in the IDENTITY and NUMBER bands
*exactly*, on the slice ``expanded_token_ids[:partial_cut_length]``, per
plan D8 + Stage 2 step 5.

The vocab band boundaries are derived from
:class:`tokenizer.token_manager.VocabularyManager` at import time -- one
test below re-derives them locally and pins the expected values so that
a future canonical-block extension surfaces here too, not only in the
production module.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._surviving_counts import (
    SurvivingCounts,
    _IDENTITY_BAND_HI_SHIFTED,
    _IDENTITY_BAND_LO_SHIFTED,
    _NUMBER_BAND_HI_SHIFTED,
    _NUMBER_BAND_LO_SHIFTED,
    count_surviving,
)
from tokenizer.token_manager import VocabularyManager


def _u16(values) -> np.ndarray:
    return np.asarray(values, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Empty / null-content / partial_cut_length boundary semantics
# ---------------------------------------------------------------------------


def test_empty_array_returns_zeros() -> None:
    out = count_surviving(_u16([]), partial_cut_length=0)
    assert out == SurvivingCounts(
        surviving_identity_count=0,
        surviving_number_chunk_count=0,
    )


def test_pure_null_content_contributes_zero() -> None:
    """Post-shift id 0 is the reserved null-content slot per plan D5; it
    must NOT land in either band."""
    out = count_surviving(_u16([0, 0, 0, 0, 0]), partial_cut_length=5)
    assert out.surviving_identity_count == 0
    assert out.surviving_number_chunk_count == 0


def test_partial_cut_length_zero_returns_zeros_regardless_of_content() -> None:
    """Per plan Stage 2 step 5: a zero-length surviving prefix yields
    zero counts even when the underlying array is fully in-band."""
    fully_in_band = _u16([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    out = count_surviving(fully_in_band, partial_cut_length=0)
    assert out == SurvivingCounts(
        surviving_identity_count=0,
        surviving_number_chunk_count=0,
    )


def test_partial_cut_length_negative_returns_zeros() -> None:
    """Defensive: any non-positive cut length yields zeros (the fast
    path's predicate ``<= 0`` covers the unlikely negative case)."""
    out = count_surviving(_u16([1, 2, 8, 9]), partial_cut_length=-1)
    assert out == SurvivingCounts(0, 0)


def test_partial_cut_length_exceeds_array_clamps_to_full_length() -> None:
    """numpy slicing already clamps ``stop`` to the array length; the
    caller (2b) may pass the full length verbatim for fully-included
    call_targets, so we document + assert that contract here."""
    arr = _u16([1, 8, 2, 9])  # 2 numbers + 2 identities
    out = count_surviving(arr, partial_cut_length=999)
    assert out.surviving_number_chunk_count == 2
    assert out.surviving_identity_count == 2


# ---------------------------------------------------------------------------
# Pure-stream cardinality
# ---------------------------------------------------------------------------


def test_pure_identity_stream_counts_full_length() -> None:
    """Every post-shift IDENTITY band id (8..15) must contribute to
    ``surviving_identity_count`` and never to the number count."""
    identity_ids = _u16([8, 9, 10, 11, 12, 13, 14, 15])
    out = count_surviving(identity_ids, partial_cut_length=len(identity_ids))
    assert out.surviving_identity_count == len(identity_ids)
    assert out.surviving_number_chunk_count == 0


def test_pure_number_stream_counts_full_length() -> None:
    """Every post-shift NUMBER band id (1..7) must contribute to
    ``surviving_number_chunk_count`` and never to the identity count."""
    number_ids = _u16([1, 2, 3, 4, 5, 6, 7])
    out = count_surviving(number_ids, partial_cut_length=len(number_ids))
    assert out.surviving_number_chunk_count == len(number_ids)
    assert out.surviving_identity_count == 0


# ---------------------------------------------------------------------------
# Boundary semantics around the band edges
# ---------------------------------------------------------------------------


def test_boundary_ids_0_and_16_excluded_from_both_bands() -> None:
    """Post-shift id 0 (null-content) and id 16 (first instruction-rep
    slot, outside the IDENTITY block) must be excluded from both
    bands."""
    out = count_surviving(_u16([0, 16, 0, 16]), partial_cut_length=4)
    assert out.surviving_identity_count == 0
    assert out.surviving_number_chunk_count == 0


def test_boundary_ids_7_and_8_route_to_correct_bands() -> None:
    """Id 7 is the last NUMBER band id (F128); id 8 is the first
    IDENTITY band id (BLOCK_V2). They sit on the band-split boundary
    and must each route to the correct counter."""
    out = count_surviving(_u16([7, 8]), partial_cut_length=2)
    assert out.surviving_number_chunk_count == 1  # id 7 -> NUMBER (F128)
    assert out.surviving_identity_count == 1  # id 8 -> IDENTITY (BLOCK_V2)


def test_boundary_ids_15_just_inside_identity_band() -> None:
    """Id 15 is the last IDENTITY band id (RW_DATA_PTR); must count."""
    out = count_surviving(_u16([15, 15, 15]), partial_cut_length=3)
    assert out.surviving_identity_count == 3
    assert out.surviving_number_chunk_count == 0


# ---------------------------------------------------------------------------
# Per-id cardinality -- every NUMBER + IDENTITY id contributes exactly 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tid", list(range(1, 8)))
def test_each_number_id_contributes_one_to_number_count(tid: int) -> None:
    out = count_surviving(_u16([tid]), partial_cut_length=1)
    assert out.surviving_number_chunk_count == 1
    assert out.surviving_identity_count == 0


@pytest.mark.parametrize("tid", list(range(8, 16)))
def test_each_identity_id_contributes_one_to_identity_count(tid: int) -> None:
    out = count_surviving(_u16([tid]), partial_cut_length=1)
    assert out.surviving_identity_count == 1
    assert out.surviving_number_chunk_count == 0


# ---------------------------------------------------------------------------
# Mixed-content spot check
# ---------------------------------------------------------------------------


def test_mixed_stream_spot_check() -> None:
    """Hand-crafted mixed stream:

    * 3 IDENTITY tokens (ids 9, 12, 15)
    * 4 NUMBER tokens (ids 1, 5, 7, 3)
    * 2 null-content slots (id 0)
    * 1 out-of-band token (id 16; should be excluded)

    Slicing the full 10-element stream must count 3 identity + 4 number.
    """
    stream = _u16([9, 1, 0, 12, 5, 7, 15, 0, 3, 16])
    out = count_surviving(stream, partial_cut_length=len(stream))
    assert out.surviving_identity_count == 3
    assert out.surviving_number_chunk_count == 4


def test_mixed_stream_partial_slice_only_counts_prefix() -> None:
    """Slice semantics: ``partial_cut_length`` cuts off the *tail*; only
    positions ``[0, cut)`` participate in the masking."""
    # Identities at positions 0, 3 (in-prefix) and 6 (post-prefix).
    # Numbers at positions 1, 4 (in-prefix) and 5, 8 (5 in-prefix, 8 not).
    stream = _u16([9, 1, 0, 12, 5, 7, 15, 0, 3])
    out = count_surviving(stream, partial_cut_length=5)
    # Prefix is [9, 1, 0, 12, 5]: identities = {9, 12} (count 2);
    # numbers = {1, 5} (count 2).
    assert out.surviving_identity_count == 2
    assert out.surviving_number_chunk_count == 2


def test_slice_semantics_uses_front_not_tail() -> None:
    """Hammer the prefix-slice contract: a cut of 3 over a stream whose
    only in-band tokens live in the LAST 3 positions returns 0."""
    # Prefix [0,0,0] gets 0; tail [9,9,9] would have been all identities.
    stream = _u16([0, 0, 0, 9, 9, 9])
    out = count_surviving(stream, partial_cut_length=3)
    assert out.surviving_identity_count == 0
    assert out.surviving_number_chunk_count == 0
    # Sanity: the same stream with full length counts the identities.
    out_full = count_surviving(stream, partial_cut_length=6)
    assert out_full.surviving_identity_count == 3


# ---------------------------------------------------------------------------
# Constants pin -- ensures band edges stay aligned with VocabularyManager
# ---------------------------------------------------------------------------


def test_band_constants_pin_to_expected_post_shift_values() -> None:
    """Plan D5 + vocab layout: post-shift bands are NUMBER=[1,8) and
    IDENTITY=[8,16). Pin them so the module's compile-time derivation
    from :class:`VocabularyManager` stays correct."""
    assert _NUMBER_BAND_LO_SHIFTED == 1
    assert _NUMBER_BAND_HI_SHIFTED == 8
    assert _IDENTITY_BAND_LO_SHIFTED == 8
    assert _IDENTITY_BAND_HI_SHIFTED == 16


def test_band_constants_derived_from_vocabulary_manager_anchors() -> None:
    """Re-derive the constants locally from
    :class:`VocabularyManager` and confirm the module agrees -- this
    pins the *derivation*, not just the literal value."""
    shift = VocabularyManager._V2_RESERVED_DIGIT_COUNT
    assert _NUMBER_BAND_LO_SHIFTED == VocabularyManager._V2_NUMBER_BLOCK_START - shift
    assert (
        _NUMBER_BAND_HI_SHIFTED
        == VocabularyManager._V2_IDENTITY_BLOCK_START - shift
    )
    assert _IDENTITY_BAND_LO_SHIFTED == _NUMBER_BAND_HI_SHIFTED
    assert _IDENTITY_BAND_HI_SHIFTED == VocabularyManager._V2_EAGER_BLOCK_END - shift


# ---------------------------------------------------------------------------
# Return-type contract
# ---------------------------------------------------------------------------


def test_surviving_counts_is_frozen_dataclass() -> None:
    """Plan + feedback_lazy_view_no_materialization.md: handoff structs
    are frozen so consumers never accidentally rewrite them."""
    from dataclasses import FrozenInstanceError

    out = count_surviving(_u16([1, 8]), partial_cut_length=2)
    assert isinstance(out, SurvivingCounts)
    with pytest.raises(FrozenInstanceError):
        out.surviving_identity_count = 99  # type: ignore[misc]


def test_counts_are_python_ints_not_numpy_scalars() -> None:
    """Caller (2d) populates a frozen dataclass typed ``int``; numpy
    scalar leakage causes type-mismatch surprises downstream."""
    out = count_surviving(_u16([1, 8]), partial_cut_length=2)
    assert type(out.surviving_identity_count) is int
    assert type(out.surviving_number_chunk_count) is int
