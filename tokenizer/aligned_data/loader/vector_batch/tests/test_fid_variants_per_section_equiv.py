"""Equivalence gate: columnar FID pass-2 == the tree-read pass-2.

The object-tree-elimination plan (step 5b) re-points the FID-sidecar
pass-2 of ``apply_per_row_remap`` on the vector dense path to read the
per-section variant counts from a COLUMNAR ``variants_per_section ==
np.ones(n_rows)`` instead of ``[len(s.variants) for s in
stage3_batch.sections]`` -- so pass-2 stops reaching into the object tree.

This module pins that re-point two ways on the live rich-splice binary:

1. INPUT contract: from inside the real vector path, the columnar
   ``variants_per_section`` the re-point threads IS all-ones AND equals the
   tree's ``[len(s.variants) for s in stage3.sections]`` per batch (so the
   two pass-2 reductions read identical counts -> identical output).

2. FULL-PATH byte-identity: a whole vector run with the re-point ACTIVE
   (columnar counts) vs a whole vector run forced onto the tree-read path
   (``variants_per_section=None``) produces a byte-identical FID sidecar
   (each run builds a fresh ``stage3``, so the in-place dedup walk is not
   double-applied).

The teeth case proves a perturbed (zeroed) count diverges the FID sidecar.
"""

from __future__ import annotations

import unittest.mock as _mock

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode import (
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.vector_batch._entry import (
    vector_batch_tokens,
)
from tokenizer.aligned_data.loader.vector_batch._scatter import (
    _dense as _dense_mod,
)
from tokenizer.aligned_data.loader.vector_batch.session_handles import (
    open_vector_batch_handles,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    make_test_vocab_manager,
)

from ._byte_identity_harness import _nonempty_matched_idxs, _prepare
from ._rich_corpus import build_rich_splice_fixture


def _run_vector(base, *, context_len, max_depth, remap_override=None):
    """One full vector run; optional patch wraps ``apply_per_row_remap``.

    Returns the :class:`BatchDecodeResult`-like output. ``remap_override``
    is a callable ``(real_remap) -> wrapper`` used to force / perturb the
    ``variants_per_section`` the re-point threads.
    """
    pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=int(i))
        for i in _nonempty_matched_idxs(base)
    ]
    dataset = BinaryDataset(
        base, "sortbin", vocab_manager=make_test_vocab_manager()
    )

    def _go():
        with dataset.open_session() as session:
            with open_vector_batch_handles(base, "sortbin") as handles:
                return vector_batch_tokens(
                    session,
                    pointers,
                    handles=handles,
                    num_variants_per_section=2,
                    context_len=context_len,
                    max_depth=max_depth,
                    variant_padding=VariantPadding.PAD_NULL,
                    include_fid_sidecar=True,
                    rng=np.random.default_rng(0),
                )

    if remap_override is None:
        return _go()
    wrapper = remap_override(_dense_mod.apply_per_row_remap)
    with _mock.patch.object(_dense_mod, "apply_per_row_remap", wrapper):
        return _go()


def _force_tree_read(real_remap):
    """Drop the threaded ``variants_per_section`` -> pass-2 reads the tree."""

    def _wrapper(stage3, *, collect_fid_sidecar=False, flat=None,
                 variants_per_section=None):
        return real_remap(
            stage3,
            collect_fid_sidecar=collect_fid_sidecar,
            flat=flat,
            variants_per_section=None,
        )

    return _wrapper


@pytest.mark.parametrize(
    "context_len,max_depth", [(4096, 3), (7, 3), (256, 0)]
)
def test_fid_variants_per_section_input_contract(
    context_len, max_depth, tmp_path
):
    """Threaded ``variants_per_section`` is all-ones == the tree's counts."""
    base = _prepare(build_rich_splice_fixture, tmp_path)
    real_remap = _dense_mod.apply_per_row_remap
    seen: list = []

    def _override(_real):
        def _wrapper(stage3, *, collect_fid_sidecar=False, flat=None,
                     variants_per_section=None):
            if variants_per_section is not None:
                threaded = np.asarray(variants_per_section)
                tree_counts = np.asarray(
                    [len(s.variants) for s in stage3.sections], dtype=np.int64
                )
                seen.append((threaded, tree_counts))
            return real_remap(
                stage3,
                collect_fid_sidecar=collect_fid_sidecar,
                flat=flat,
                variants_per_section=variants_per_section,
            )

        return _wrapper

    _run_vector(
        base, context_len=context_len, max_depth=max_depth,
        remap_override=_override,
    )

    assert seen, "the columnar FID pass-2 was never exercised"
    for threaded, tree_counts in seen:
        assert np.array_equal(threaded, np.ones_like(threaded)), (
            "threaded variants_per_section is not all-ones"
        )
        assert np.array_equal(threaded, tree_counts), (
            "threaded counts diverge from the tree's per-section counts"
        )


@pytest.mark.parametrize(
    "context_len,max_depth", [(4096, 3), (7, 3), (256, 0)]
)
def test_fid_variants_per_section_full_path_identical(
    context_len, max_depth, tmp_path
):
    """Columnar pass-2 FID == tree-read pass-2 FID over a full vector run."""
    base = _prepare(build_rich_splice_fixture, tmp_path)
    columnar = _run_vector(base, context_len=context_len, max_depth=max_depth)
    tree = _run_vector(
        base, context_len=context_len, max_depth=max_depth,
        remap_override=_force_tree_read,
    )

    assert np.array_equal(columnar.fid_sidecar, tree.fid_sidecar), (
        "fid_sidecar diverged"
    )
    assert np.array_equal(columnar.fid_row_offsets, tree.fid_row_offsets), (
        "fid_row_offsets diverged"
    )
    assert np.array_equal(
        columnar.fid_per_category_counts, tree.fid_per_category_counts
    ), "fid_per_category_counts diverged"
    # Non-vacuous on the full-fit / deep grid: real FID rows present.
    if context_len >= 256 and max_depth >= 3:
        assert int(columnar.fid_row_offsets[-1]) > 0, (
            "no FID rows on the full-fit deep grid -- vacuous gate"
        )


def test_fid_variants_per_section_gate_has_teeth(tmp_path):
    """A perturbed (all-zero) count MUST diverge the FID sidecar.

    Zeroing EVERY section's variant count empties the per-unique-variant
    sidecar list, so every row resolves to an empty slice -- the FID
    sidecar collapses to length 0, diverging from the real (non-empty)
    output. Proves the threaded count is load-bearing for pass-2.
    """
    base = _prepare(build_rich_splice_fixture, tmp_path)

    def _zero_all(real_remap):
        def _wrapper(stage3, *, collect_fid_sidecar=False, flat=None,
                     variants_per_section=None):
            if variants_per_section is not None:
                variants_per_section = np.zeros_like(
                    np.asarray(variants_per_section)
                )
            return real_remap(
                stage3,
                collect_fid_sidecar=collect_fid_sidecar,
                flat=flat,
                variants_per_section=variants_per_section,
            )

        return _wrapper

    ref = _run_vector(base, context_len=4096, max_depth=3)
    assert int(ref.fid_row_offsets[-1]) > 0, "no FID rows -- cannot perturb"
    bad = _run_vector(
        base, context_len=4096, max_depth=3, remap_override=_zero_all
    )
    assert not (
        np.array_equal(bad.fid_sidecar, ref.fid_sidecar)
        and np.array_equal(bad.fid_row_offsets, ref.fid_row_offsets)
    ), "zeroed variant count did NOT diverge -- vacuous gate"
