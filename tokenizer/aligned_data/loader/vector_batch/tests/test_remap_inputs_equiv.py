"""Equivalence gate: columnar ``FlatRemapInputs`` == the tree-walk's.

The object-tree-elimination plan (step 4) re-points ``apply_per_row_remap``
on the vector dense path to feed the Rust kernel from a COLUMNAR
:class:`FlatRemapInputs` (:func:`...vector_batch._scatter._remap_inputs.
build_flat_remap_inputs`, built from the dense + catalog columns) instead
of the GIL-bound per-call-target object-tree walk
(:func:`...batch_decode._dedup_walk._flat_extract.extract_flat_remap_inputs`).

This module pins that re-point DIRECTLY: on the live rich-splice binary it
captures, from inside the real vector path, the columnar ``flat`` the
re-point threads to the kernel AND the ``stage3`` tree the staged extractor
walks, then asserts EVERY ``FlatRemapInputs`` field is ``np.array_equal``
(and ``row_keys`` / ``n_rows`` identical). The full-FID-sidecar byte-
identity gate in :mod:`.test_byte_identity_dense` already proves the kernel
output matches end to end; this gate isolates the input contract so a
field-level drift surfaces here with a precise message.

Driven over the same context_len x depth grid the byte-identity harness
uses (full fit, mid-body straddler cut, depth-0 root-only), so the surviving
clip + the in-stream prepend-drop + the fully-dropped node case are all
exercised on real content.
"""

from __future__ import annotations

import unittest.mock as _mock

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode import (
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._flat_extract import (
    extract_flat_remap_inputs,
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
from ._full_tree_oracle import (
    build_full_tree_stage3,
    capture_columnar_inputs,
)
from ._rich_corpus import build_rich_splice_fixture


_FLAT_ARRAY_FIELDS = (
    "node_row",
    "node_skip",
    "node_prepend_pos",
    "node_fid",
    "node_enc_func_slot",
    "ct_off",
    "ct_fid",
    "ct_func_slot",
    "instream_off",
    "instream_func_slot",
    "instream_counter_slot",
    "counter_counts",
)


def _assert_flat_equal(oracle, columnar) -> None:
    """Assert every ``FlatRemapInputs`` field equal (tree vs columnar)."""
    assert columnar.n_rows == oracle.n_rows, (
        f"n_rows: columnar {columnar.n_rows} vs tree {oracle.n_rows}"
    )
    assert columnar.row_keys == oracle.row_keys, (
        f"row_keys diverge:\n  tree={oracle.row_keys}\n"
        f"  columnar={columnar.row_keys}"
    )
    for name in _FLAT_ARRAY_FIELDS:
        o = np.asarray(getattr(oracle, name))
        c = np.asarray(getattr(columnar, name))
        assert c.shape == o.shape, (
            f"{name} shape: columnar {c.shape} vs tree {o.shape}"
        )
        if not np.array_equal(c, o):
            diff = np.nonzero(c.reshape(-1) != o.reshape(-1))[0]
            k = int(diff[0])
            raise AssertionError(
                f"{name} differs at {diff.size} pos; first flat idx {k}: "
                f"tree={o.reshape(-1)[k]!r} columnar={c.reshape(-1)[k]!r}"
            )


@pytest.mark.parametrize(
    "context_len,max_depth", [(4096, 3), (7, 3), (256, 0)]
)
def test_flat_remap_inputs_equiv_live_binary(
    context_len, max_depth, tmp_path
):
    """Columnar ``FlatRemapInputs`` == the staged tree-walk's, per batch."""
    base = _prepare(build_rich_splice_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)

    # Capture the production columnar ``flat`` (the re-point's threaded
    # kernel input). The tree-walk ORACLE is rebuilt independently from the
    # captured ``(geometry, dense, catalog)`` (the production tree is now
    # SLIM -- step-5 object-tree elimination -- so we cannot read it off the
    # production ``stage3`` any more).
    flats: list = []
    real_remap = _dense_mod.apply_per_row_remap

    def _capturing(
        stage3, *, collect_fid_sidecar=False, flat=None,
        variants_per_section=None,
    ):
        if flat is not None:
            flats.append(flat)
        return real_remap(
            stage3,
            collect_fid_sidecar=collect_fid_sidecar,
            flat=flat,
            variants_per_section=variants_per_section,
        )

    pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=int(i)) for i in idxs
    ]
    dataset = BinaryDataset(
        base, "sortbin", vocab_manager=make_test_vocab_manager()
    )
    with capture_columnar_inputs() as columnar_inputs:
        with _mock.patch.object(
            _dense_mod, "apply_per_row_remap", _capturing
        ):
            with dataset.open_session() as session:
                with open_vector_batch_handles(base, "sortbin") as handles:
                    vector_batch_tokens(
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

    assert flats, "the columnar remap path was never exercised"
    assert len(flats) == len(columnar_inputs), (
        "columnar-input capture / flat capture lengths diverge: "
        f"{len(columnar_inputs)} vs {len(flats)}"
    )
    any_nodes = False
    for (geometry, dense, catalog), columnar in zip(columnar_inputs, flats):
        oracle = extract_flat_remap_inputs(
            build_full_tree_stage3(geometry, dense, catalog)
        )
        _assert_flat_equal(oracle, columnar)
        if oracle.node_row.shape[0] > 0:
            any_nodes = True
    assert any_nodes, "live capture carried no nodes -- vacuous gate"
