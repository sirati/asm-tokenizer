"""Equivalence gate: columnar number-sidecar stream == the tree-walk's.

The object-tree-elimination plan (step 5a) re-points the number-sidecar
concat on the vector dense path to feed the per-chunk stream from a
COLUMNAR :class:`NumberChunkColumns` (:func:`...vector_batch._scatter.
_number_chunk_columns.build_number_chunk_columns`, built from the dense
columns + the kernel-built per-call_target chunk-slice ``.start`` arrays +
the emission row CSR) instead of the GIL-bound ``sections -> variants ->
call_targets`` object-tree walk (:func:`...batch_decode._sidecar_concat.
_tree_chunk_columns`).

This module pins that re-point DIRECTLY: on the live rich-splice binary it
captures, from inside the real vector path, the columnar ``numbers`` the
re-point threads AND the ``stage3`` tree the staged walk consumes, then
asserts:

1. The per-chunk :class:`NumberChunkColumns` triple (``out_block``,
   ``slice_start``, ``ct_ordinal``) + the variant CSR are ``np.array_equal``
   to the tree walk's, AND
2. the FULL ``assemble_number_sidecars`` output (``numbers_significant`` /
   ``numbers_sign_exponent``) is byte-identical between the two sources.

Driven over the same context_len x depth grid the byte-identity harness
uses (full fit, mid-body straddler cut, depth-0 root-only) so the
surviving clip + the VC2/F128 multi-chunk per-(ct, block) rank + the
mid-cut F128 invisible-MSB drop are all exercised on real content. The
teeth case proves a deliberate slice-start / rank perturbation FAILS.
"""

from __future__ import annotations

import unittest.mock as _mock

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode import (
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.batch_decode._sidecar_concat import (
    NumberChunkColumns,
    _NUMBER_BLOCK_TOKEN_TYPES,
    _build_global_chunk_stream,
    _tree_chunk_columns,
    assemble_number_sidecars,
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


_TRIPLE_FIELDS = ("out_block", "slice_start", "ct_ordinal")


def _assert_chunk_columns_equal(
    tree: NumberChunkColumns, columnar: NumberChunkColumns
) -> None:
    """Assert the per-chunk triple + the variant CSR equal (tree vs columnar)."""
    for name in _TRIPLE_FIELDS + ("variant_chunk_offsets",):
        t = np.asarray(getattr(tree, name))
        c = np.asarray(getattr(columnar, name))
        assert c.shape == t.shape, (
            f"{name} shape: columnar {c.shape} vs tree {t.shape}"
        )
        if not np.array_equal(c, t):
            diff = np.nonzero(c.reshape(-1) != t.reshape(-1))[0]
            k = int(diff[0])
            raise AssertionError(
                f"{name} differs at {diff.size} pos; first flat idx {k}: "
                f"tree={t.reshape(-1)[k]!r} columnar={c.reshape(-1)[k]!r}"
            )


@pytest.mark.parametrize(
    "context_len,max_depth", [(4096, 3), (7, 3), (256, 0)]
)
def test_number_chunk_columns_equiv_live_binary(
    context_len, max_depth, tmp_path
):
    """Columnar number-sidecar stream == the staged tree-walk's, per batch."""
    base = _prepare(build_rich_splice_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)

    captured: list = []
    real_assemble = _dense_mod.assemble_number_sidecars

    def _capturing(stage3, numbers=None):
        # The re-point ALWAYS supplies columnar ``numbers`` on a non-empty
        # batch; capture it alongside the tree-walk oracle built from the
        # same ``stage3``.
        if numbers is not None:
            tree = _tree_chunk_columns(stage3)
            captured.append((stage3, tree, numbers))
        return real_assemble(stage3, numbers)

    pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=int(i)) for i in idxs
    ]
    dataset = BinaryDataset(
        base, "sortbin", vocab_manager=make_test_vocab_manager()
    )
    with _mock.patch.object(
        _dense_mod, "assemble_number_sidecars", _capturing
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

    assert captured, "the columnar number-sidecar path was never exercised"
    any_chunks = False
    for stage3, tree, columnar in captured:
        _assert_chunk_columns_equal(tree, columnar)

        # Full sidecar byte-identity: the shared rank + gather tail must
        # produce the same (sig, sex) from either source.
        tree_sig, tree_sex = assemble_number_sidecars(stage3)
        col_sig, col_sex = assemble_number_sidecars(stage3, columnar)
        assert np.array_equal(tree_sig, col_sig), "numbers_significant diverged"
        assert np.array_equal(
            tree_sex, col_sex
        ), "numbers_sign_exponent diverged"

        if tree.out_block.shape[0] > 0:
            any_chunks = True
    assert any_chunks, "live capture carried no number chunks -- vacuous gate"


# ---------------------------------------------------------------------------
# Teeth: a SYNTHETIC multi-same-block-per-ct stream where the per-(ct, block)
# rank reset is load-bearing -- a rolled ``ct_ordinal`` MUST mis-route.
# ---------------------------------------------------------------------------


def _synthetic_two_ct_vc2_stream():
    """Two call_targets, two VC2 chunks each, interleaved with an F32 chunk.

    The shared rank + gather tail reads ``numbers_per_TokenType[VC2]`` at
    ``slice_start + rank``; with two VC2 chunks per ct the rank goes 0, 1
    WITHIN each ct and RESETS at the ct boundary. A rolled ``ct_ordinal``
    breaks that reset, so the gathered significands diverge. Returns a
    ``(stage3_stub, columnar)`` pair sharing one ``numbers_per_TokenType``.
    """

    class _Stage3Stub:
        # ``_build_global_chunk_stream`` reads only ``numbers_per_TokenType``
        # when fed a columnar ``numbers`` (no tree walk), so a stub carrying
        # just that mapping is sufficient + decoupled from the object tree.
        def __init__(self, numbers_per_TokenType):
            self.numbers_per_TokenType = numbers_per_TokenType

    from tokenizer.tokens import TokenType

    # Per-type source tables: distinct values so a mis-routed gather is
    # observable. VC2 has 4 sources (ct0: rows 0,1; ct1: rows 2,3).
    vc2_sig = np.array([10, 11, 12, 13], dtype=np.uint64)
    vc2_sex = np.array([100, 101, 102, 103], dtype=np.uint32)
    f32_sig = np.array([20, 21], dtype=np.uint64)
    f32_sex = np.array([200, 201], dtype=np.uint32)
    numbers_per_TokenType = {T: (np.zeros(0, np.uint64), np.zeros(0, np.uint32))
                             for T in _NUMBER_BLOCK_TOKEN_TYPES}
    numbers_per_TokenType[TokenType.VALUED_CONST_V2] = (vc2_sig, vc2_sex)
    numbers_per_TokenType[TokenType.FLOAT32] = (f32_sig, f32_sex)

    # Stream (DFS-then-stream): ct0 [VC2, VC2, F32], ct1 [VC2, VC2, F32].
    out_block = np.array([0, 0, 3, 0, 0, 3], dtype=np.int64)
    ct_ordinal = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    # ct0's VC2 slice starts at 0, F32 at 0; ct1's VC2 at 2, F32 at 1.
    slice_start = np.array([0, 0, 0, 2, 2, 1], dtype=np.int64)
    variant_chunk_offsets = np.array([0, 3, 6], dtype=np.int64)
    columnar = NumberChunkColumns(
        out_block=out_block,
        slice_start=slice_start,
        ct_ordinal=ct_ordinal,
        variant_chunk_offsets=variant_chunk_offsets,
    )
    return _Stage3Stub(numbers_per_TokenType), columnar


def test_number_chunk_columns_gate_has_teeth():
    """Slice-start + rank perturbations both mis-route the synthetic stream."""
    import dataclasses

    stage3, columnar = _synthetic_two_ct_vc2_stream()
    ref_sig, ref_sex = _build_global_chunk_stream(stage3, columnar)[:2]
    # Non-vacuous: the reference gather actually consumes per-(ct, block)
    # ranks > 0 (the second VC2 chunk of each ct).
    assert ref_sig.tolist() == [10, 11, 20, 12, 13, 21], (
        f"unexpected reference gather: {ref_sig.tolist()}"
    )

    # Rolled ct_ordinal breaks the per-(ct, block) rank reset.
    bad_ct = dataclasses.replace(
        columnar, ct_ordinal=np.roll(columnar.ct_ordinal, 1)
    )
    bad_sig = _build_global_chunk_stream(stage3, bad_ct)[0]
    assert not np.array_equal(bad_sig, ref_sig), (
        "rolled ct_ordinal did NOT diverge -- vacuous gate"
    )

    # Shifted slice_start mis-routes every chunk's source base.
    bad_start = dataclasses.replace(
        columnar, slice_start=columnar.slice_start * 0 + 1
    )
    bad_sig2 = _build_global_chunk_stream(stage3, bad_start)[0]
    assert not np.array_equal(bad_sig2, ref_sig), (
        "shifted slice_start did NOT diverge -- vacuous gate"
    )
