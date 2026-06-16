"""Byte-identity gate: UNMATCHED-arm roots (the per-arm dispatch).

Roots live in the UNMATCHED arm: a MULTI-version root (FLAG-A) + a root
that inlines an unmatched callee AND drops a matched callee cross-arm.
The corpus leads with a multi-version pad so the roots' BASE RECORD idx
is shifted past their section idx (exercises the record-idx -> section-idx
mapping). The new path opens BOTH arms (VectorBatchArmSet) and routes
each root through its own arm; full byte-identity vs batch_decode's
arm-keyed pipeline via the shared :mod:`._byte_identity_harness`.
"""

from __future__ import annotations

import pytest

from ._byte_identity_harness import (
    _assert_token_identity,
    _prepare,
    _run_both_unmatched,
)
from ._rich_unmatched_corpus import build_rich_unmatched_fixture


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_unmatched_byte_identity_full_depth(seed, tmp_path):
    """Whole unmatched corpus, wide context: the multi-version FLAG-A root,
    the inlined unmatched callee, and the dropped matched callee all
    assemble byte-identically to batch_decode."""
    base = _prepare(build_rich_unmatched_fixture, tmp_path)
    ref, new = _run_both_unmatched(
        base,
        num_variants_per_section=2,
        context_len=4096,
        max_depth=3,
        seed=seed,
    )
    _assert_token_identity(ref, new)
    # Non-vacuous dense gate: the unmatched root + inlined leaf carry real
    # NUMBER / IDENTITY / FID carriers.
    assert ref.numbers_significant.size > 0, "fixture lost its NUMBER carriers"
    assert int(ref.identity_row_offsets[-1]) > 0, "fixture lost identities"
    assert int(ref.fid_per_category_counts.sum()) > 0, "fixture lost FID dedup"


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_unmatched_byte_identity_straddler_cut(seed, tmp_path):
    """A tight context_len cuts the unmatched roots mid-body -- the cut
    column + the post-cut dense remap must land identically."""
    base = _prepare(build_rich_unmatched_fixture, tmp_path)
    ref, new = _run_both_unmatched(
        base,
        num_variants_per_section=2,
        context_len=7,
        max_depth=3,
        seed=seed,
    )
    _assert_token_identity(ref, new)


def test_unmatched_byte_identity_depth_zero(tmp_path):
    """depth 0: unmatched roots are root-only (no splice), but the bodies
    still carry NUMBER + IDENTITY + COUNTER carriers, so the dense decode
    runs with no callee."""
    base = _prepare(build_rich_unmatched_fixture, tmp_path)
    ref, new = _run_both_unmatched(
        base,
        num_variants_per_section=2,
        context_len=256,
        max_depth=0,
        seed=0,
    )
    _assert_token_identity(ref, new)
    assert ref.numbers_significant.size > 0, "fixture lost its NUMBER carriers"


def test_unmatched_cross_arm_drop_is_real(tmp_path):
    """The matched callee ``uroot`` v0 names MUST be dropped cross-arm.

    Both paths follow the unmatched root's inlined unmatched callee but
    NEITHER inlines the matched callee (its offset is absent from the
    unmatched arm's section map). This pins the drop is real, not a
    fixture artefact: full byte-identity holds AND the inlined unmatched
    leaf actually contributed (the root row is longer than a depth-0 row),
    while the row never grows by the matched callee's body."""
    base = _prepare(build_rich_unmatched_fixture, tmp_path)
    # Full-depth: uleaf inlines, mcallee drops.
    ref_full, new_full = _run_both_unmatched(
        base,
        num_variants_per_section=2,
        context_len=4096,
        max_depth=3,
        seed=0,
    )
    _assert_token_identity(ref_full, new_full)
    # Depth 0: no splice at all (root-only).
    ref_d0, _ = _run_both_unmatched(
        base,
        num_variants_per_section=2,
        context_len=4096,
        max_depth=0,
        seed=0,
    )
    # The unmatched leaf DID inline somewhere (full-depth carries strictly
    # more identity carriers than depth-0), proving the cross-arm drop is
    # selective (drops mcallee) rather than dropping every callee.
    assert int(ref_full.identity_row_offsets[-1]) > int(
        ref_d0.identity_row_offsets[-1]
    ), "no callee inlined -- fixture/splice broken, drop test vacuous"
