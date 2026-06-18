"""Per-row ``max_depth`` equivalence + byte-identity gate (piece 3).

``vector_batch_tokens`` accepts ``max_depth`` as a scalar OR a per-row
``int`` array. These tests pin the two correctness invariants of the
per-row-depth path:

1. BYTE-IDENTITY: a constant per-row array == the scalar path, array-for-
   array on the token tensor + every dense sidecar. The single-depth
   contract must not shift by even a byte.
2. EQUIVALENCE: a MIXED-depth batch == the row-wise stitch of the
   per-depth single-depth decodes. Because sampling + the ``batch_idx``
   layout are depth-AGNOSTIC (they run before dispatch on the SAME seed),
   the row at batch position ``r`` decoded at its mixed depth ``d_r``
   must equal that SAME row in a pure-depth-``d_r`` batch drawn with the
   same seed -- verified per row over both the token tensor and the
   CSR-segmented dense sidecars.

The corpus is the combined fixture (multi-variant section + an
asymmetric splice edge), so d0 (root-only) and d3 (fully spliced) rows
have genuinely different bodies -- the mixed batch is non-trivial.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.vector_batch._entry import (
    vector_batch_tokens,
)
from tokenizer.aligned_data.loader.vector_batch.session_handles import (
    open_vector_batch_handles,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture,
    make_test_vocab_manager,
)

from ._byte_identity_harness import _nonempty_matched_idxs, _prepare
from ._rich_corpus import build_rich_splice_fixture

_BINARY_NAME = "sortbin"


def _decode(base, *, section_idxs, max_depth, seed, context_len, n_var):
    """Run ``vector_batch_tokens`` for ``section_idxs`` at ``max_depth``.

    ``max_depth`` is threaded through verbatim (scalar OR per-row array);
    a fresh seeded RNG makes the depth-agnostic sample reproducible.
    """
    pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=int(i))
        for i in section_idxs
    ]
    dataset = BinaryDataset(
        base, _BINARY_NAME, vocab_manager=make_test_vocab_manager()
    )
    with dataset.open_session() as session:
        with open_vector_batch_handles(base, _BINARY_NAME) as handles:
            return vector_batch_tokens(
                session,
                pointers,
                handles=handles,
                num_variants_per_section=n_var,
                context_len=context_len,
                max_depth=max_depth,
                variant_padding=VariantPadding.PAD_NULL,
                include_fid_sidecar=True,
                rng=np.random.default_rng(seed),
            )


def _row_segment(flat, offsets, r):
    """The CSR segment for batch row ``r``."""
    off = np.asarray(offsets)
    return np.asarray(flat)[int(off[r]):int(off[r + 1])]


def _assert_row_equal(a, b, r, *, what):
    """Assert row ``r`` of two results matches: tokens + every dense CSR."""
    assert np.array_equal(a.tokens[r], b.tokens[r]), (
        f"{what}: tokens row {r} differ\n  a={a.tokens[r].tolist()}\n"
        f"  b={b.tokens[r].tolist()}"
    )
    assert np.array_equal(
        _row_segment(a.identities, a.identity_row_offsets, r),
        _row_segment(b.identities, b.identity_row_offsets, r),
    ), f"{what}: identities row {r} differ"
    assert np.array_equal(
        _row_segment(a.numbers_significant, a.number_row_offsets, r),
        _row_segment(b.numbers_significant, b.number_row_offsets, r),
    ), f"{what}: numbers_significant row {r} differ"
    assert np.array_equal(
        _row_segment(a.numbers_sign_exponent, a.number_row_offsets, r),
        _row_segment(b.numbers_sign_exponent, b.number_row_offsets, r),
    ), f"{what}: numbers_sign_exponent row {r} differ"
    assert np.array_equal(
        _row_segment(a.fid_sidecar, a.fid_row_offsets, r),
        _row_segment(b.fid_sidecar, b.fid_row_offsets, r),
    ), f"{what}: fid_sidecar row {r} differ"
    assert np.array_equal(
        np.asarray(a.fid_per_category_counts)[r],
        np.asarray(b.fid_per_category_counts)[r],
    ), f"{what}: fid_per_category_counts row {r} differ"


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_constant_array_is_byte_identical_to_scalar(seed, depth, tmp_path):
    """A constant per-pointer depth array == the scalar path, byte-for-byte."""
    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    common = dict(
        section_idxs=idxs, seed=seed, context_len=4096, n_var=4,
    )
    scalar = _decode(base, max_depth=depth, **common)
    # One depth per section pointer (the per-pointer cross-depth unit).
    const_arr = np.full(len(idxs), depth, dtype=np.int64)
    arrayed = _decode(base, max_depth=const_arr, **common)

    assert np.array_equal(arrayed.tokens, scalar.tokens)
    assert np.array_equal(
        arrayed.batch_idx_to_section_variant,
        scalar.batch_idx_to_section_variant,
    )
    for name in (
        "identities",
        "identity_row_offsets",
        "numbers_significant",
        "numbers_sign_exponent",
        "number_row_offsets",
        "fid_sidecar",
        "fid_row_offsets",
        "fid_per_category_counts",
    ):
        assert np.array_equal(
            np.asarray(getattr(arrayed, name)),
            np.asarray(getattr(scalar, name)),
        ), f"{name} differs between constant-array and scalar paths"


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_mixed_depth_equals_per_depth_decodes(seed, tmp_path):
    """A mixed-depth batch == the row-wise stitch of single-depth decodes.

    Rows are assigned alternating depths (d0 / d3). Each row of the mixed
    batch must equal that row in a PURE-depth batch drawn with the same
    seed (sampling is depth-agnostic, so the per-row geometry is the only
    thing that changes with depth).

    The rich-splice fixture is used so d0 (root-only) and d3 (``root``
    inlines ``leaf``) genuinely diverge on the spliced row -- a trivial
    no-contrast batch would not exercise the per-depth grouping.
    """
    base = _prepare(build_rich_splice_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    common = dict(
        section_idxs=idxs, seed=seed, context_len=4096, n_var=4,
    )
    assert len(idxs) >= 2, "fixture must yield multiple section pointers"

    depth_a, depth_b = 0, 3
    # One depth PER SECTION POINTER, alternating a/b.
    per_pointer = np.where(
        np.arange(len(idxs)) % 2 == 0, depth_a, depth_b
    ).astype(np.int64)

    mixed = _decode(base, max_depth=per_pointer, **common)
    pure_a = _decode(base, max_depth=depth_a, **common)
    pure_b = _decode(base, max_depth=depth_b, **common)

    # The depth-agnostic sample/layout must be identical across all runs.
    assert np.array_equal(
        mixed.batch_idx_to_section_variant,
        pure_a.batch_idx_to_section_variant,
    )
    assert np.array_equal(
        pure_a.batch_idx_to_section_variant,
        pure_b.batch_idx_to_section_variant,
    )

    # Each row's depth follows its section pointer (mapping column 0);
    # padding rows are inert.
    mapping = np.asarray(mixed.batch_idx_to_section_variant)
    batch_size = mixed.tokens.shape[0]
    sentinel = np.iinfo(np.uint32).max
    for r in range(batch_size):
        ptr_idx = int(mapping[r, 0])
        if ptr_idx == sentinel:
            continue  # padding row -- both paths leave it null
        ref = pure_a if per_pointer[ptr_idx] == depth_a else pure_b
        _assert_row_equal(
            mixed, ref, r,
            what=f"mixed-row {r} (ptr {ptr_idx}, depth {int(per_pointer[ptr_idx])})",
        )

    # The two depths must actually produce DIFFERENT bodies on at least
    # one row, else the test would pass trivially (no depth contrast).
    differs = any(
        not np.array_equal(pure_a.tokens[r], pure_b.tokens[r])
        for r in range(batch_size)
    )
    assert differs, (
        "d0 and d3 produced identical bodies on every row; the fixture "
        "no longer exercises depth contrast"
    )


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_single_depth_depth_per_row_is_all_that_depth(seed, depth, tmp_path):
    """A single-depth batch labels every non-padding row with that depth.

    ``depth_per_row`` is the NEW per-row source-depth identifier; for a
    scalar ``max_depth`` every decoded row holds that one depth and padding
    rows (mapping sentinel) hold 0. The token tensor + dense sidecars are
    unchanged (the byte-identity gate above covers that); this only pins
    the additional field.
    """
    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    result = _decode(
        base, section_idxs=idxs, max_depth=depth, seed=seed,
        context_len=4096, n_var=4,
    )
    mapping = np.asarray(result.batch_idx_to_section_variant)
    sentinel = np.iinfo(np.uint32).max
    is_padding = mapping[:, 0] == sentinel
    depth_per_row = np.asarray(result.depth_per_row)
    assert depth_per_row.shape == (result.tokens.shape[0],)
    assert depth_per_row.dtype == np.int64
    assert np.all(depth_per_row[~is_padding] == depth)
    assert np.all(depth_per_row[is_padding] == 0)


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_cross_depth_depth_per_row_labels_each_row_source_depth(seed, tmp_path):
    """A mixed-depth batch's depth_per_row == each row's section depth.

    Each row's source depth is the depth of the section pointer it was
    drawn from (mapping column 0 -> per_pointer[ptr_idx]); padding rows
    hold 0. Cross-checked against the SAME per-pointer depth vector the
    sampler/loader used to split the depth groups.
    """
    base = _prepare(build_rich_splice_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    assert len(idxs) >= 2, "fixture must yield multiple section pointers"

    depth_a, depth_b = 0, 3
    per_pointer = np.where(
        np.arange(len(idxs)) % 2 == 0, depth_a, depth_b
    ).astype(np.int64)
    result = _decode(
        base, section_idxs=idxs, max_depth=per_pointer, seed=seed,
        context_len=4096, n_var=4,
    )

    mapping = np.asarray(result.batch_idx_to_section_variant)
    depth_per_row = np.asarray(result.depth_per_row)
    sentinel = np.iinfo(np.uint32).max
    expected = np.zeros(result.tokens.shape[0], dtype=np.int64)
    real = mapping[:, 0] != sentinel
    expected[real] = per_pointer[mapping[real, 0].astype(np.int64)]
    np.testing.assert_array_equal(depth_per_row, expected)
    # The batch genuinely mixes depths (else the label test is trivial).
    assert set(depth_per_row[real].tolist()) == {depth_a, depth_b}


def test_per_pointer_length_mismatch_raises(tmp_path):
    """A per-pointer array whose length != #pointers is a hard caller error."""
    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    bad = np.zeros(len(idxs) + 3, dtype=np.int64)
    with pytest.raises(ValueError, match="per-pointer max_depth has length"):
        _decode(
            base,
            section_idxs=idxs,
            max_depth=bad,
            seed=0,
            context_len=256,
            n_var=4,
        )
