"""Byte-identity harness: the vectorized path vs ``batch_decode``.

THE GATE for the vectorized dataloader. Runs the NEW path
(:func:`...vector_batch._entry.vector_batch_tokens`) and the CURRENT
:func:`...batch_decode.batch_decode` with IDENTICAL settings (same RNG
seed, same section pointers, same depth, BACKFILL OFF) on a real small
binary and asserts FULL byte-identity of the token tensor + the
``batch_idx_to_section_variant`` mapping.

Both paths now do sampled-subset outline detection (T0 merged), so the
included set + emission order converge by construction; this test pins
that the geometry + scatter assemble the SAME ``tokens[B, L]`` the
staged ``batch_decode`` pipeline produces.

Highest-risk fixture (TC1 author's flag): ``build_combined_fixture``
carries a MULTI-VARIANT section (``multi_fn``, 4 variants) AND an
ASYMMETRIC call graph (``caller_fn`` -> ``callee_fn`` with disjoint vkey
sets -> a MISSING_VARIANT_INDEX splice edge), so subset-vs-full FLAG-A
inclusion differs across rows. The straddler cut is exercised by a tight
``context_len``; a multi-section batch by pointing at several sections.

REUSE: the on-disk corpus + RLG3 sidecars + unified-vocab session come
from the shared sorted-index fixtures + realized-geometry generators (the
same wiring the realized-geometry oracle test uses); the new path's
handles come from :func:`...vector_batch.session_handles.
open_vector_batch_handles`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode import (
    SectionPointerSpec,
    VariantPadding,
    batch_decode,
)
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.vector_batch._entry import (
    vector_batch_tokens,
)
from tokenizer.aligned_data.loader.vector_batch.session_handles import (
    open_vector_batch_arm_set,
    open_vector_batch_handles,
)
from tokenizer.aligned_data.realized_lengths import (
    generate_realized_geometry,
    generate_realized_lengths,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture,
    build_many_variant_section_fixture,
    make_test_vocab_manager,
)

from ._rich_corpus import build_rich_splice_fixture
from ._rich_unmatched_corpus import build_rich_unmatched_fixture


_BINARY_NAME = "sortbin"


def _prepare(builder, tmp_path: Path):
    """Build the corpus + RLG3 sidecars; return ``base_path``."""
    base = builder(tmp_path)
    generate_realized_lengths(base, _BINARY_NAME)
    generate_realized_geometry(base, _BINARY_NAME)
    return base


def _nonempty_matched_idxs(base: Path) -> list[int]:
    """Matched section indices with at least one variant.

    ``batch_decode`` rejects a 0-variant section at sampling time (the
    combined fixture deliberately carries one as an edge case), so the
    harness batches only the sample-able sections.
    """
    dataset = BinaryDataset(base, _BINARY_NAME, vocab_manager=None)
    out: list[int] = []
    n = 0
    with dataset.open_session() as session:
        while True:
            try:
                matched = session.load_matched(n)
            except IndexError:
                break
            if len(matched.variants) > 0:
                out.append(n)
            n += 1
    return out


def _unmatched_section_base_idxs(base: Path) -> list[int]:
    """Unmatched section FIRST-RECORD idxs with at least one version.

    A ``SectionPointerSpec(arm=UNMATCHED, idx=...)`` must carry a
    section's BASE record idx (loading a non-base record raises), so the
    harness probes every record idx and keeps the ones the session
    accepts as a section base -- exactly the pointers ``batch_decode``
    samples. Because the corpus leads with a MULTI-version ``upad``, the
    later sections' base record idx is SHIFTED past their section idx, so
    this set exercises ``base_record_idx != section_idx``.
    """
    dataset = BinaryDataset(base, _BINARY_NAME, vocab_manager=None)
    out: list[int] = []
    n = 0
    with dataset.open_session() as session:
        while True:
            try:
                section, _off, variants = (
                    session._load_unmatched_section_and_all_variants(n)
                )
            except IndexError:
                break
            except ValueError:
                # Non-base record idx of a multi-version section -- skip.
                n += 1
                continue
            if len(variants) > 0:
                out.append(n)
            n += 1
    return out


def _run_both(
    base: Path,
    *,
    section_idxs,
    num_variants_per_section: int,
    context_len: int,
    max_depth: int,
    seed: int,
    variant_padding: VariantPadding = VariantPadding.PAD_NULL,
    arm: SectionKind = SectionKind.MATCHED,
):
    """Run ``batch_decode`` + the vectorized path with the SAME draw.

    Both paths request the FID sidecar (``include_fid_sidecar=True``) so
    the dedup walk's inverse map + per-Category counts are exercised AND
    compared in full -- the strongest test of the ALG-3/4/9 remap.

    ``arm`` selects which arm the root pointers target. MATCHED roots use
    a single-arm ``VectorBatchHandles`` (the historical contract);
    UNMATCHED roots use a both-arms ``VectorBatchArmSet`` so the per-arm
    dispatch routes them through the unmatched catalog + geometry (and a
    cross-arm callee misses the unmatched ``_sec_map`` -> dropped, the
    same arm-keyed DROP ``batch_decode`` produces).
    """
    pointers = [
        SectionPointerSpec(arm=arm, idx=int(i)) for i in section_idxs
    ]
    dataset = BinaryDataset(
        base, _BINARY_NAME, vocab_manager=make_test_vocab_manager()
    )
    if arm is SectionKind.MATCHED:
        handles_cm = open_vector_batch_handles(base, _BINARY_NAME)
    else:
        handles_cm = open_vector_batch_arm_set(base, _BINARY_NAME)
    with dataset.open_session() as session:
        # Reference: the current staged pipeline.
        ref = batch_decode(
            session,
            pointers,
            num_variants_per_section=num_variants_per_section,
            context_len=context_len,
            max_depth=max_depth,
            variant_padding=variant_padding,
            include_fid_sidecar=True,
            rng=np.random.default_rng(seed),
        )
        # New path: same inputs, fresh rng with the SAME seed -> same draw.
        with handles_cm as handles:
            new = vector_batch_tokens(
                session,
                pointers,
                handles=handles,
                num_variants_per_section=num_variants_per_section,
                context_len=context_len,
                max_depth=max_depth,
                variant_padding=variant_padding,
                include_fid_sidecar=True,
                rng=np.random.default_rng(seed),
            )
    return ref, new


def _assert_token_identity(ref, new) -> None:
    """FULL byte-identity: tokens + mapping + EVERY dense sidecar.

    Asserts ``np.array_equal`` IN FULL on every returned array -- the
    token tensor, the ``batch_idx_to_section_variant`` mapping, AND all
    the dense sidecars (identity data + offsets; numeric significand +
    sign-exp + offsets; per-Category FID sidecar + offsets + counts).
    Any mismatch is a debug-to-root-cause failure; the assertion is never
    weakened to a content-slice compare.
    """
    assert new.tokens.dtype == ref.tokens.dtype == np.uint16
    assert new.tokens.shape == ref.tokens.shape, (
        f"shape mismatch: new {new.tokens.shape} vs ref {ref.tokens.shape}"
    )
    assert np.array_equal(
        new.batch_idx_to_section_variant, ref.batch_idx_to_section_variant
    ), "batch_idx_to_section_variant differs"
    if not np.array_equal(new.tokens, ref.tokens):
        diff_rows = np.nonzero(
            (new.tokens != ref.tokens).any(axis=1)
        )[0]
        r = int(diff_rows[0])
        raise AssertionError(
            f"tokens differ in {diff_rows.size} row(s); first row {r}:\n"
            f"  ref={ref.tokens[r].tolist()}\n"
            f"  new={new.tokens[r].tolist()}"
        )
    _assert_dense_identity(ref, new)


def _assert_array(name: str, ref_arr, new_arr) -> None:
    """``np.array_equal`` on one named array; rich diff on mismatch."""
    ref_arr = np.asarray(ref_arr)
    new_arr = np.asarray(new_arr)
    assert new_arr.dtype == ref_arr.dtype, (
        f"{name} dtype: new {new_arr.dtype} vs ref {ref_arr.dtype}"
    )
    assert new_arr.shape == ref_arr.shape, (
        f"{name} shape: new {new_arr.shape} vs ref {ref_arr.shape}"
    )
    if not np.array_equal(new_arr, ref_arr):
        diff = np.nonzero(new_arr.reshape(-1) != ref_arr.reshape(-1))[0]
        k = int(diff[0])
        raise AssertionError(
            f"{name} differs at {diff.size} position(s); first flat idx "
            f"{k}: ref={ref_arr.reshape(-1)[k]!r} new={new_arr.reshape(-1)[k]!r}"
        )


def _assert_dense_identity(ref, new) -> None:
    """Assert FULL byte-identity of every dense sidecar array."""
    _assert_array("identities", ref.identities, new.identities)
    _assert_array(
        "identity_row_offsets",
        ref.identity_row_offsets,
        new.identity_row_offsets,
    )
    _assert_array(
        "numbers_significant", ref.numbers_significant, new.numbers_significant
    )
    _assert_array(
        "numbers_sign_exponent",
        ref.numbers_sign_exponent,
        new.numbers_sign_exponent,
    )
    _assert_array(
        "number_row_offsets", ref.number_row_offsets, new.number_row_offsets
    )
    _assert_array("fid_sidecar", ref.fid_sidecar, new.fid_sidecar)
    _assert_array(
        "fid_row_offsets", ref.fid_row_offsets, new.fid_row_offsets
    )
    _assert_array(
        "fid_per_category_counts",
        ref.fid_per_category_counts,
        new.fid_per_category_counts,
    )


# ---------------------------------------------------------------------------
# Multi-variant + asymmetric call graph (the highest-risk case)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_byte_identity_combined_fixture_full_depth(seed, tmp_path):
    """Whole combined corpus (multi-variant + asymmetric splice), a wide
    context so no straddler -- every row's full body must match."""
    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=4,
        context_len=4096,
        max_depth=3,
        seed=seed,
    )
    _assert_token_identity(ref, new)


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_byte_identity_combined_straddler_cut(seed, tmp_path):
    """A tight context_len forces a mid-function straddler cut on the
    longer rows -- the cut column must land identically."""
    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=4,
        context_len=12,
        max_depth=3,
        seed=seed,
    )
    _assert_token_identity(ref, new)


@pytest.mark.parametrize("seed", [0, 5])
def test_byte_identity_many_variant_section(seed, tmp_path):
    """The many-variant single section -- subset sampling of <4 of the
    variants exercises FLAG-A subset-vs-full divergence per row."""
    base = _prepare(build_many_variant_section_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=2,
        context_len=512,
        max_depth=2,
        seed=seed,
    )
    _assert_token_identity(ref, new)


def test_byte_identity_depth_zero_roots_only(tmp_path):
    """depth 0: every row is root-only (no splice) -- the prefix + root
    body + self-token must match with no callee assembly."""
    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=4,
        context_len=256,
        max_depth=0,
        seed=0,
    )
    _assert_token_identity(ref, new)


# ---------------------------------------------------------------------------
# Rich-body splice fixture -- makes the DENSE sidecar assertions
# load-bearing (NUMBER significands + cross-call_target FUNCTION remap +
# COUNTER offset bump + a real inlined callee). The decode-agnostic
# fixtures above produce trivially-empty dense arrays; these do not.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# UNMATCHED-arm roots -- the per-arm dispatch gate. Roots live in the
# UNMATCHED arm: a MULTI-version root (FLAG-A) + a root that inlines an
# unmatched callee AND drops a matched callee cross-arm. The corpus leads
# with a multi-version pad so the roots' BASE RECORD idx is shifted past
# their section idx (exercises the record-idx -> section-idx mapping). The
# new path opens BOTH arms (VectorBatchArmSet) and routes each root through
# its own arm; full byte-identity vs batch_decode's arm-keyed pipeline.
# ---------------------------------------------------------------------------


def _run_both_unmatched(base, **kwargs):
    """``_run_both`` driving UNMATCHED-arm root pointers (the arm set)."""
    idxs = _unmatched_section_base_idxs(base)
    return _run_both(
        base, section_idxs=idxs, arm=SectionKind.UNMATCHED, **kwargs
    )


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
