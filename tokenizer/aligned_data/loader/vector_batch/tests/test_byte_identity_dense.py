"""Byte-identity gate: rich-body dense-sidecar splice fixture.

Makes the DENSE sidecar assertions load-bearing -- NUMBER significands +
cross-call_target FUNCTION remap + COUNTER offset bump + a real inlined
callee. The decode-agnostic matched fixtures produce trivially-empty
dense arrays; this corpus does not, so the per-array byte-identity holds
over real content (each test guards against a vacuous gate). Both paths
assert full byte-identity via the shared :mod:`._byte_identity_harness`.
"""

from __future__ import annotations

import pytest

from ._byte_identity_harness import (
    _assert_token_identity,
    _nonempty_matched_idxs,
    _prepare,
    _run_both,
)
from ._rich_corpus import build_rich_splice_fixture


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_dense_identity_rich_splice_full_depth(seed, tmp_path):
    """Rich carriers + a REAL inlined callee (root -> leaf), wide context.

    Exercises the full dense decode: VC2 + F16 significands/signs, the
    BLOCK COUNTER offset bump across call_targets, and the LOCAL_FUNC
    dedup remap minting counters across the root + inlined leaf. Every
    dense array must be byte-identical."""
    base = _prepare(build_rich_splice_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=2,
        context_len=4096,
        max_depth=3,
        seed=seed,
    )
    _assert_token_identity(ref, new)
    # Guard against a vacuous gate: the fixture MUST carry real dense
    # content (else the array-equal assertions are over empty arrays).
    assert ref.numbers_significant.size > 0, "fixture lost its NUMBER carriers"
    assert int(ref.identity_row_offsets[-1]) > 0, "fixture lost identities"
    assert int(ref.fid_per_category_counts.sum()) > 0, "fixture lost FID dedup"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_dense_identity_rich_splice_straddler_cut(seed, tmp_path):
    """A tight context_len cuts the straddler mid-body -- the dense
    surviving id / number counts (and the post-cut remap) must match."""
    base = _prepare(build_rich_splice_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=2,
        context_len=7,
        max_depth=3,
        seed=seed,
    )
    _assert_token_identity(ref, new)


def test_dense_identity_rich_depth_zero(tmp_path):
    """depth 0 over the rich corpus: root-only rows, but the root bodies
    still carry NUMBER + IDENTITY + COUNTER carriers, so the dense decode
    + per-row remap run with no callee."""
    base = _prepare(build_rich_splice_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=2,
        context_len=256,
        max_depth=0,
        seed=0,
    )
    _assert_token_identity(ref, new)
    assert ref.numbers_significant.size > 0, "fixture lost its NUMBER carriers"
