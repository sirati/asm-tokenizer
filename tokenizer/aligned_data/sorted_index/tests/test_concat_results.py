"""Unit tests for sorted_index._sampler._concat_results (plan ALG-6).

The cross-binary concat helper is exercised against hand-crafted
:class:`BatchDecodeResult` instances so the test does NOT depend on
the full batch_decode pipeline. Each test builds 1-2 synthetic results
and asserts on the stitched output's invariants (shape, row-offset
re-base, ``binary_id_per_row`` numbering, fid all-or-none rule,
shape-precondition).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
)
from tokenizer.aligned_data.sorted_index._sampler import _concat_results


# ---------------------------------------------------------------------------
# Synthetic-result builder
# ---------------------------------------------------------------------------


def _make_result(
    *,
    batch_size: int,
    context_len: int,
    per_row_identity: list[int],
    per_row_numbers: list[int],
    tokens_fill: int,
    btv_fill_offset: int = 0,
    include_fid: bool = False,
) -> BatchDecodeResult:
    """Build a synthetic :class:`BatchDecodeResult` with controlled shapes.

    ``per_row_identity[i]`` is the identity-token count on row ``i``;
    ``per_row_numbers[i]`` is the number-chunk count on row ``i``. The
    flat identities + numbers arrays are filled with deterministic
    sentinel values (``tokens_fill`` + position) so the concat output's
    contents can be cross-checked.
    """
    assert len(per_row_identity) == batch_size
    assert len(per_row_numbers) == batch_size

    tokens = np.full(
        (batch_size, context_len), tokens_fill, dtype=np.uint16,
    )

    identity_total = int(sum(per_row_identity))
    identities = np.arange(identity_total, dtype=np.uint16) + tokens_fill
    identity_row_offsets = np.concatenate(
        ([np.uint32(0)], np.cumsum(per_row_identity, dtype=np.uint32)),
    )

    number_total = int(sum(per_row_numbers))
    numbers_significant = (
        np.arange(number_total, dtype=np.uint64) + tokens_fill
    )
    numbers_sign_exponent = (
        np.arange(number_total, dtype=np.uint32) + tokens_fill
    )
    number_row_offsets = np.concatenate(
        ([np.uint32(0)], np.cumsum(per_row_numbers, dtype=np.uint32)),
    )

    btv = np.array(
        [
            [btv_fill_offset + i, btv_fill_offset + i + 100]
            for i in range(batch_size)
        ],
        dtype=np.uint32,
    )

    fid_sidecar: Optional[np.ndarray] = None
    fid_row_offsets: Optional[np.ndarray] = None
    if include_fid:
        fid_total = batch_size * 2
        fid_sidecar = np.arange(fid_total, dtype=np.uint32) + tokens_fill
        fid_row_offsets = np.concatenate(
            (
                [np.uint32(0)],
                np.cumsum(np.full(batch_size, 2, dtype=np.uint32)),
            ),
        )

    return BatchDecodeResult(
        tokens=tokens,
        identities=identities,
        identity_row_offsets=identity_row_offsets,
        numbers_significant=numbers_significant,
        numbers_sign_exponent=numbers_sign_exponent,
        number_row_offsets=number_row_offsets,
        batch_idx_to_section_variant=btv,
        fid_sidecar=fid_sidecar,
        fid_row_offsets=fid_row_offsets,
        intermediate=None,
    )


# ---------------------------------------------------------------------------
# Shape sums + binary_id stamping
# ---------------------------------------------------------------------------


def test_concat_two_binaries_shape_sums_correctly() -> None:
    a = _make_result(
        batch_size=2, context_len=8,
        per_row_identity=[3, 1],
        per_row_numbers=[2, 0],
        tokens_fill=10,
    )
    b = _make_result(
        batch_size=3, context_len=8,
        per_row_identity=[0, 2, 4],
        per_row_numbers=[1, 5, 0],
        tokens_fill=100,
    )
    out = _concat_results([("alpha", a), ("beta", b)])
    inner = out.inner
    # tokens: rows stacked, context_len preserved.
    assert inner.tokens.shape == (5, 8)
    np.testing.assert_array_equal(inner.tokens[:2], a.tokens)
    np.testing.assert_array_equal(inner.tokens[2:], b.tokens)
    # identities: flat sum across both batches.
    assert inner.identities.size == a.identities.size + b.identities.size
    # numbers: flat sum across both batches (both significand + sign_exp).
    assert (
        inner.numbers_significant.size
        == a.numbers_significant.size + b.numbers_significant.size
    )
    assert (
        inner.numbers_sign_exponent.size
        == a.numbers_sign_exponent.size + b.numbers_sign_exponent.size
    )
    # btv: stacked along axis=0.
    assert inner.batch_idx_to_section_variant.shape == (5, 2)


def test_concat_binary_id_per_row_matches_input_order() -> None:
    a = _make_result(
        batch_size=2, context_len=4,
        per_row_identity=[1, 1], per_row_numbers=[0, 0], tokens_fill=0,
    )
    b = _make_result(
        batch_size=3, context_len=4,
        per_row_identity=[0, 0, 0], per_row_numbers=[0, 0, 0], tokens_fill=0,
    )
    out = _concat_results([("alpha", a), ("beta", b)])
    assert out.binary_names == ["alpha", "beta"]
    np.testing.assert_array_equal(
        out.binary_id_per_row,
        np.array([0, 0, 1, 1, 1], dtype=np.uint32),
    )


def test_concat_row_offsets_rebase_to_global_cumsum() -> None:
    a = _make_result(
        batch_size=2, context_len=4,
        per_row_identity=[3, 1], per_row_numbers=[2, 0], tokens_fill=0,
    )
    b = _make_result(
        batch_size=2, context_len=4,
        per_row_identity=[0, 4], per_row_numbers=[1, 5], tokens_fill=0,
    )
    out = _concat_results([("alpha", a), ("beta", b)])
    inner = out.inner
    # identity_row_offsets: prefix is a's [0, 3, 4]; then b's [0, 4]
    # appended re-based by 4 -> [4, 8]. Result [0, 3, 4, 4, 8].
    np.testing.assert_array_equal(
        inner.identity_row_offsets,
        np.array([0, 3, 4, 4, 8], dtype=np.uint32),
    )
    # number_row_offsets: a's [0, 2, 2]; b's [1, 6] re-based by 2 -> [3, 8].
    # Result [0, 2, 2, 3, 8].
    np.testing.assert_array_equal(
        inner.number_row_offsets,
        np.array([0, 2, 2, 3, 8], dtype=np.uint32),
    )


def test_concat_intermediate_dropped() -> None:
    a = _make_result(
        batch_size=1, context_len=2,
        per_row_identity=[0], per_row_numbers=[0], tokens_fill=0,
    )
    out = _concat_results([("alpha", a)])
    assert out.inner.intermediate is None


def test_concat_btv_stacked_without_renumbering() -> None:
    """Cross-binary identity is carried by ``binary_id_per_row``; the
    btv array itself is concatenated as-is (the same idx may legitimately
    appear in two different binaries)."""
    a = _make_result(
        batch_size=2, context_len=2,
        per_row_identity=[0, 0], per_row_numbers=[0, 0], tokens_fill=0,
        btv_fill_offset=0,
    )
    b = _make_result(
        batch_size=1, context_len=2,
        per_row_identity=[0], per_row_numbers=[0], tokens_fill=0,
        btv_fill_offset=0,    # same starting offset as `a`: collisions in btv.
    )
    out = _concat_results([("alpha", a), ("beta", b)])
    inner = out.inner
    np.testing.assert_array_equal(
        inner.batch_idx_to_section_variant,
        np.concatenate(
            [a.batch_idx_to_section_variant, b.batch_idx_to_section_variant],
            axis=0,
        ),
    )


# ---------------------------------------------------------------------------
# fid all-or-none
# ---------------------------------------------------------------------------


def test_concat_fid_all_present(tmp_path) -> None:
    a = _make_result(
        batch_size=2, context_len=2,
        per_row_identity=[0, 0], per_row_numbers=[0, 0], tokens_fill=0,
        include_fid=True,
    )
    b = _make_result(
        batch_size=1, context_len=2,
        per_row_identity=[0], per_row_numbers=[0], tokens_fill=10,
        include_fid=True,
    )
    out = _concat_results([("alpha", a), ("beta", b)])
    inner = out.inner
    assert inner.fid_sidecar is not None
    assert inner.fid_row_offsets is not None
    # both sources have 2 fid entries per row -> totals 4 + 2.
    assert inner.fid_sidecar.size == a.fid_sidecar.size + b.fid_sidecar.size
    # row-offsets: a is [0, 2, 4]; b is [0, 2] -> drop leading 0,
    # rebase by 4 -> append [6]. Result [0, 2, 4, 6].
    np.testing.assert_array_equal(
        inner.fid_row_offsets,
        np.array([0, 2, 4, 6], dtype=np.uint32),
    )


def test_concat_fid_all_absent() -> None:
    a = _make_result(
        batch_size=1, context_len=2,
        per_row_identity=[0], per_row_numbers=[0], tokens_fill=0,
        include_fid=False,
    )
    b = _make_result(
        batch_size=1, context_len=2,
        per_row_identity=[0], per_row_numbers=[0], tokens_fill=0,
        include_fid=False,
    )
    out = _concat_results([("alpha", a), ("beta", b)])
    assert out.inner.fid_sidecar is None
    assert out.inner.fid_row_offsets is None


def test_concat_fid_mixed_inputs_raises() -> None:
    a = _make_result(
        batch_size=1, context_len=2,
        per_row_identity=[0], per_row_numbers=[0], tokens_fill=0,
        include_fid=True,
    )
    b = _make_result(
        batch_size=1, context_len=2,
        per_row_identity=[0], per_row_numbers=[0], tokens_fill=0,
        include_fid=False,
    )
    with pytest.raises(ValueError, match="include_fid_sidecar inconsistent"):
        _concat_results([("alpha", a), ("beta", b)])


# ---------------------------------------------------------------------------
# Shape precondition (D-2.1) + empty input
# ---------------------------------------------------------------------------


def test_concat_shape_precondition_raises_on_context_len_mismatch() -> None:
    a = _make_result(
        batch_size=1, context_len=8,
        per_row_identity=[0], per_row_numbers=[0], tokens_fill=0,
    )
    b = _make_result(
        batch_size=1, context_len=16,
        per_row_identity=[0], per_row_numbers=[0], tokens_fill=0,
    )
    with pytest.raises(ValueError, match="tokens.shape\\[1\\] mismatch"):
        _concat_results([("alpha", a), ("beta", b)])


def test_concat_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="empty input"):
        _concat_results([])


# ---------------------------------------------------------------------------
# Single-binary degenerate case
# ---------------------------------------------------------------------------


def test_concat_single_binary_passthrough_shapes() -> None:
    """One-binary input still gets wrapped; row offsets unchanged."""
    a = _make_result(
        batch_size=2, context_len=4,
        per_row_identity=[2, 3], per_row_numbers=[1, 0], tokens_fill=0,
    )
    out = _concat_results([("alpha", a)])
    inner = out.inner
    np.testing.assert_array_equal(inner.tokens, a.tokens)
    np.testing.assert_array_equal(
        inner.identity_row_offsets, a.identity_row_offsets,
    )
    np.testing.assert_array_equal(
        inner.number_row_offsets, a.number_row_offsets,
    )
    assert out.binary_names == ["alpha"]
    np.testing.assert_array_equal(
        out.binary_id_per_row, np.array([0, 0], dtype=np.uint32),
    )
