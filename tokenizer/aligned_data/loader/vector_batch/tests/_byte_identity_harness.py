"""Shared byte-identity harness: drive both paths + assert identity.

THE GATE plumbing for the vectorized dataloader. Builds the corpus +
RLG3 sidecars, runs the NEW path
(:func:`...vector_batch._entry.vector_batch_tokens`) and the CURRENT
:func:`...batch_decode.batch_decode` with IDENTICAL settings (same RNG
seed, same section pointers, same depth, BACKFILL OFF) on a real small
binary, and asserts FULL byte-identity of the token tensor + the
``batch_idx_to_section_variant`` mapping + every dense sidecar.

The per-concern test cases that drive this harness live in the sibling
``test_byte_identity_*`` modules (matched-arm fixtures, the rich dense
sidecar corpus, the unmatched-arm dispatch).

REUSE: the on-disk corpus + RLG3 sidecars + unified-vocab session come
from the shared sorted-index fixtures + realized-geometry generators (the
same wiring the realized-geometry oracle test uses); the new path's
handles come from :func:`...vector_batch.session_handles.
open_vector_batch_handles`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
    make_test_vocab_manager,
)


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


def _run_both_unmatched(base, **kwargs):
    """``_run_both`` driving UNMATCHED-arm root pointers (the arm set)."""
    idxs = _unmatched_section_base_idxs(base)
    return _run_both(
        base, section_idxs=idxs, arm=SectionKind.UNMATCHED, **kwargs
    )
