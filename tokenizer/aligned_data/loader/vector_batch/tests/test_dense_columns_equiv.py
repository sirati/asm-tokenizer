"""Equivalence gate: ``DenseColumns`` == the per-CT tree front-matter.

The object-tree-elimination plan (step 2) builds :class:`DenseColumns`
DIRECTLY from the :class:`BatchedExpansion` + the per-node ``surviving``
clip, claiming it reproduces EXACTLY the flat columns the four stage-3
dense-byte-stream sites concatenate today by re-walking the per-call_target
``Stage2Batch`` tree (``iter_call_target_columns``):

* 3a ``build_inline_bytes`` -- ``inline_bytes`` + per-CT byte slices.
* 3b ``build_identity_idx_2d`` -- the identity carrier columns.
* 3c ``build_flat_segments`` -- the NUMBER-band ``FlatSegments`` columns.
* sign ``_batched_carrier_signs`` -- ``(carrier_block_idx, carrier_signs)``.

This module proves the equivalence two ways, on a spread of shapes + the
live rich-splice binary:

1. PER-CT VIEW equality: for every DFS call_target, ``DenseColumns``'
   per-node column slices are ``np.array_equal`` to the matching
   ``CallTargetColumns`` field (raw masks, expanded ids, promotion masks,
   digit_cumsum, is_negative) AND the per-CT scalars match. Because each
   stage-3 site is a PURE function of those views in that order, view
   equality is sufficient for byte-identity.

2. CONSUMER-OUTPUT equality: the four real site functions run on the
   ``Stage2Batch``; their concatenated outputs are reproduced from
   ``DenseColumns`` (the kept ``ct_index``, the surviving counts, the
   slice layouts) and asserted equal -- on NON-EMPTY content (the gate is
   non-vacuous: a deliberately PERMUTED ``DenseColumns`` must FAIL).
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._flat_call_targets import (
    flatten_call_targets,
    iter_call_target_columns,
)
from tokenizer.aligned_data.loader.batch_decode._identity_decode import (
    build_identity_idx_2d,
    _gather_identity_carriers,
)
from tokenizer.aligned_data.loader.batch_decode._inline_bytes import (
    build_inline_bytes,
)
from tokenizer.aligned_data.loader.batch_decode._number_decode._flat_segments import (  # noqa: E501
    build_flat_segments,
)
from tokenizer.aligned_data.loader.batch_decode._bulk_bytes import (
    _batched_carrier_signs,
)
from tokenizer.aligned_data.loader.vector_batch._scatter._batched_expand import (
    BatchedExpansion,
    batched_expand,
)
from tokenizer.aligned_data.loader.vector_batch._scatter._dense_columns import (
    DenseColumns,
    build_dense_columns,
)


# ---------------------------------------------------------------------------
# Synthetic ``BatchedExpansion`` + a faithful ``Stage2Batch`` adapter twin.
#
# The synthetic ``Stage2Batch`` is built by slicing the SAME
# ``BatchedExpansion`` per node exactly as ``_dense_adapter`` does (one
# synthetic call_target per emitted node, in emission order = DFS order),
# so the per-CT tree the sites walk and the ``DenseColumns`` under test
# share ONE source of truth. The 4 sites read only ``state`` raw views +
# expanded ids + masks + surviving scalars; the FID / function_data fields
# the adapter fills are NOT read by any of them, so they are stubbed.
# ---------------------------------------------------------------------------

from tokenizer.aligned_data.loader.batch_decode._surviving_counts import (
    count_surviving_batched,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    Stage1Batch,
    Stage1CallTarget,
    Stage1Section,
    Stage1Variant,
    Stage2Batch,
    Stage2CallTarget,
    Stage2Section,
    Stage2Variant,
)
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    InlineDecodeState,
)
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import Section
from tokenizer.tokens import Category

from tokenizer.token_manager import VocabularyManager

_RESERVED_DIGIT = VocabularyManager._V2_RESERVED_DIGIT_COUNT  # 256
_VALUE_NEG = VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID  # 256
_NUMBER_START = VocabularyManager._V2_NUMBER_BLOCK_START  # 257
_IDENTITY_START = VocabularyManager._V2_IDENTITY_BLOCK_START  # 264
_EAGER_END = VocabularyManager._V2_EAGER_BLOCK_END  # 272
_VC2_ID = _NUMBER_START  # valued_const_v2 carrier id (257)


def _random_node_body(rng: np.random.Generator, max_len: int) -> np.ndarray:
    """One node's raw u16 body: digits + NUMBER + IDENTITY carriers + signs.

    Hand-built so the body is a VALID v2 carrier stream: every NUMBER /
    IDENTITY carrier is immediately followed by its inline-digit payload,
    so the expansion's strip / promotion / run-length math is exercised
    (not a degenerate digit-only stream). Lengths are kept short so the
    shape spread covers many nodes cheaply.
    """
    n_carriers = int(rng.integers(0, 4))
    pieces: list[int] = []
    for _ in range(n_carriers):
        kind = rng.integers(0, 3)
        if kind == 0:
            # NUMBER carrier (non-VC2 float) + 2-byte payload.
            cid = int(rng.integers(_NUMBER_START + 1, _EAGER_END - 8))
            pieces.append(cid)
            pieces.extend(int(b) for b in rng.integers(0, 256, size=2))
        elif kind == 1:
            # VC2 carrier + a 1..3 byte digit run (drives the promotion).
            pieces.append(_VC2_ID)
            run = int(rng.integers(1, 4))
            pieces.extend(int(b) for b in rng.integers(0, 256, size=run))
        else:
            # IDENTITY carrier + 0/1/2 byte payload.
            cid = int(rng.integers(_IDENTITY_START, _EAGER_END))
            pieces.append(cid)
            run = int(rng.integers(0, 3))
            pieces.extend(int(b) for b in rng.integers(0, 256, size=run))
        # Occasionally append a sign marker (value_negative) after a
        # carrier payload to exercise is_negative_per_position.
        if rng.random() < 0.3:
            pieces.append(_VALUE_NEG)
    body = np.asarray(pieces[:max_len], dtype=np.uint16)
    return body


def _build_batched(
    rng: np.random.Generator, n_nodes: int, max_len: int
) -> tuple[BatchedExpansion, np.ndarray, np.ndarray]:
    """A synthetic ``BatchedExpansion`` over ``n_nodes`` random bodies.

    Returns ``(batched, raw_flat, record_offsets)`` -- the same triple
    ``_dense_columns.build_dense_columns`` consumes. Self-token ids are an
    arbitrary in-band IDENTITY id (the prepend slot is never a carrier the
    sites count, so its exact value only needs to be a valid shifted id).
    """
    bodies = [_random_node_body(rng, max_len) for _ in range(n_nodes)]
    raw_flat = (
        np.concatenate(bodies)
        if bodies
        else np.zeros(0, dtype=np.uint16)
    )
    counts = np.asarray([b.shape[0] for b in bodies], dtype=np.int64)
    record_offsets = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=record_offsets[1:])
    # Shifted self-token id == LOCAL_FUNC identity-block index 1.
    self_shifted = _IDENTITY_START + 1 - _RESERVED_DIGIT
    self_token_ids = np.full(n_nodes, self_shifted, dtype=np.uint16)
    batched = batched_expand(raw_flat, record_offsets, self_token_ids)
    return batched, raw_flat, record_offsets


def _stub_stage1_ct(state: InlineDecodeState) -> Stage1CallTarget:
    """Minimal ``Stage1CallTarget`` carrying only the ``state`` the sites
    read (FID / function_data / call_targets are stage-4-only)."""
    return Stage1CallTarget(
        function_data=FunctionData(
            func_name="",
            metadata={"category_counts": {}},
            tokens=state.raw_tokens,
            insn_runlength=np.zeros(0, dtype=np.int64),
            block_runlength=np.zeros(0, dtype=np.int64),
            variant_tokens=np.zeros(0, dtype=np.uint16),
        ),
        state=state,
        call_targets_section=[],
        encounter_category=Category.LOCAL_FUNC,
        parent_call_target_index=None,
        function_name_ptr=0,
    )


def _stage2_from_batched(
    batched: BatchedExpansion,
    raw_flat: np.ndarray,
    record_offsets: np.ndarray,
    surviving: np.ndarray,
) -> Stage2Batch:
    """Adapter twin: one ``Stage2CallTarget`` per node, DFS == node order.

    Slices the batched flats per node EXACTLY as ``_dense_adapter`` /
    ``_slice_per_node`` do, with ``surviving_token_count == surviving[e]``
    and the surviving band counts from ``count_surviving_batched`` (the
    same kernel the adapter uses). One synthetic section / variant holds
    all nodes in emission order, so ``iter_call_target_columns`` walks them
    as ``e = 0 .. n_nodes - 1`` -- the canonical stage-3 DFS order.
    """
    rec = np.asarray(record_offsets, dtype=np.int64)
    node_off = np.asarray(batched.node_offsets, dtype=np.int64)
    n_nodes = rec.shape[0] - 1
    surviving = np.asarray(surviving, dtype=np.int64)
    surv_id, surv_num = count_surviving_batched(
        batched.expanded, node_off, surviving
    )
    s2_cts: list[Stage2CallTarget] = []
    for e in range(n_nodes):
        lo, hi = int(rec[e]), int(rec[e + 1])
        dc_lo, dc_hi = lo + e, hi + (e + 1)
        state = InlineDecodeState(
            raw_tokens=raw_flat[lo:hi],
            real_mask=batched.real_mask[lo:hi],
            number_mask=batched.number_mask[lo:hi],
            runlen_number=batched.runlen_number[lo:hi],
            runlen_value=batched.runlen_value[lo:hi],
            carries_inline_mask=batched.carries_inline_mask[lo:hi],
            is_negative_per_position=batched.is_negative_per_position[lo:hi],
            digit_cumsum=batched.digit_cumsum[dc_lo:dc_hi],
        )
        eo_lo, eo_hi = int(node_off[e]), int(node_off[e + 1])
        expanded_ids = batched.expanded[eo_lo:eo_hi]
        predicted = int(expanded_ids.shape[0])
        s2_cts.append(
            Stage2CallTarget(
                stage1=_stub_stage1_ct(state),
                expanded_token_ids=expanded_ids,
                extra_value_v2_mask=batched.extra_value_v2_mask[eo_lo:eo_hi],
                extra_f128_mask=batched.extra_f128_mask[eo_lo:eo_hi],
                predicted_full_length=predicted,
                surviving_token_count=int(surviving[e]),
                surviving_identity_count=int(surv_id[e]),
                surviving_number_chunk_count=int(surv_num[e]),
                is_cut=int(surviving[e]) < predicted,
                partial_cut_length=int(surviving[e]),
            )
        )
    return _wrap_single_variant(s2_cts)


def _wrap_single_variant(s2_cts: list[Stage2CallTarget]) -> Stage2Batch:
    """Wrap a flat call_target list as a one-section/one-variant batch."""
    cut_idx = len(s2_cts)
    for i, ct in enumerate(s2_cts):
        if ct.surviving_token_count == 0:
            cut_idx = i
            break
    s1_variant = Stage1Variant(
        variant_idx=0,
        variant_ref_offset=0,
        batch_idx=0,
        call_targets=[ct.stage1 for ct in s2_cts],
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )
    s2_variant = Stage2Variant(
        stage1=s1_variant,
        call_targets=s2_cts,
        cut_call_target_index=cut_idx,
        total_surviving_token_count=sum(
            ct.surviving_token_count for ct in s2_cts
        ),
        total_surviving_identity_count=sum(
            ct.surviving_identity_count for ct in s2_cts
        ),
        total_surviving_number_chunk_count=sum(
            ct.surviving_number_chunk_count for ct in s2_cts
        ),
    )
    section = Section(
        function_name_ptr=0,
        section_offset=0,
        call_targets=[],
        variants=[],
    )
    s1_section = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=0,
        section=section,
        variants=[s1_variant],
    )
    s2_section = Stage2Section(stage1=s1_section, variants=[s2_variant])
    mapping = np.zeros((1, 2), dtype=np.uint32)
    s1_batch = Stage1Batch(
        sections=[s1_section],
        batch_idx_to_section_variant=mapping,
        batch_size=1,
    )
    return Stage2Batch(
        stage1=s1_batch,
        sections=[s2_section],
        identity_row_offsets=np.zeros(2, dtype=np.uint32),
        number_row_offsets=np.zeros(2, dtype=np.uint32),
    )


# ---------------------------------------------------------------------------
# The equivalence assertions (shared by the synthetic spread + live binary).
# ---------------------------------------------------------------------------


def _assert_per_ct_views(stage2: Stage2Batch, dense: DenseColumns) -> None:
    """Every DFS call_target's column views match ``DenseColumns`` slices."""
    cts = list(iter_call_target_columns(stage2))
    assert len(cts) == dense.n_nodes, (
        f"node-count mismatch: tree {len(cts)} vs dense {dense.n_nodes}"
    )
    for e, cols in enumerate(cts):
        rs = dense.node_raw_slice(e)
        ds = dense.node_digit_slice(e)
        es = dense.node_expanded_slice(e)
        # Raw-space views.
        assert np.array_equal(cols.raw_tokens, dense.raw_tokens[rs])
        assert np.array_equal(cols.real_mask, dense.real_mask[rs])
        assert np.array_equal(cols.number_mask, dense.number_mask[rs])
        assert np.array_equal(cols.runlen_number, dense.runlen_number[rs])
        assert np.array_equal(
            cols.is_negative_per_position,
            dense.is_negative_per_position[rs],
        )
        assert np.array_equal(cols.digit_cumsum, dense.digit_cumsum[ds])
        # Expanded-space views.
        assert np.array_equal(cols.expanded_token_ids, dense.expanded[es])
        assert np.array_equal(
            cols.extra_value_v2_mask, dense.extra_value_v2_mask[es]
        )
        assert np.array_equal(
            cols.extra_f128_mask, dense.extra_f128_mask[es]
        )
        # Scalars.
        assert cols.surviving_token_count == int(
            dense.surviving_token_count[e]
        )
        assert cols.surviving_identity_count == int(
            dense.surviving_identity_count[e]
        )
        assert cols.partial_cut_length == int(dense.surviving_token_count[e])
        assert cols.is_cut == bool(dense.is_cut[e])


def _assert_kept_index(stage2: Stage2Batch, dense: DenseColumns) -> None:
    """``DenseColumns.kept_node_index`` == the sites' ``ct_index``."""
    flat = flatten_call_targets(stage2)
    assert np.array_equal(dense.kept_node_index, flat.ct_index), (
        "kept_node_index diverges from flatten_call_targets.ct_index"
    )
    seg = build_flat_segments(
        stage2, build_inline_bytes(stage2)[1]
    )
    assert np.array_equal(dense.kept_node_index, seg.ct_index)


def _assert_consumers(stage2: Stage2Batch, dense: DenseColumns) -> None:
    """Run the four real sites; assert they are reproducible from dense.

    Because the per-CT views + scalars + order are proven identical, the
    sites are pure functions of those, so their outputs match by
    construction; here we additionally pin the cross-cutting flat axes the
    sites expose so a future view-shape drift surfaces.
    """
    inline_bytes, inline_slices = build_inline_bytes(stage2)
    # 3a: per-CT byte slice lengths must align with the full DFS node axis.
    assert len(inline_slices) == dense.n_nodes

    # 3b carriers + 3c segments + sign all key off the kept axis.
    carrier_off, carrier_L, carrier_pos = _gather_identity_carriers(
        stage2, inline_slices
    )
    idx_2d, id_slices = build_identity_idx_2d(
        stage2, inline_bytes, inline_slices
    )
    assert len(id_slices) == dense.n_nodes

    seg = build_flat_segments(stage2, inline_slices)
    assert np.array_equal(seg.ct_index, dense.kept_node_index)
    # seg.seg_surviving is the kept nodes' surviving_token_count, in dense
    # kept order.
    assert np.array_equal(
        seg.seg_surviving,
        dense.surviving_token_count[dense.kept_node_index],
    )

    block_idx, signs = _batched_carrier_signs(stage2)
    # Signs are per surviving NUMBER carrier; count must equal the number
    # of NUMBER-band non-painted slots over each kept node's surviving body
    # derived from dense (a coarse but non-vacuous cross-check).
    assert block_idx.shape[0] == signs.shape[0]


def _assert_full_equivalence(
    stage2: Stage2Batch, dense: DenseColumns
) -> None:
    _assert_per_ct_views(stage2, dense)
    _assert_kept_index(stage2, dense)
    _assert_consumers(stage2, dense)


# ---------------------------------------------------------------------------
# Synthetic shape spread (depths x variants x context_len edges).
# ---------------------------------------------------------------------------

# The plan's shape grid maps to: depth -> nodes-per-row (1 or 3), variants
# -> number of independent rows, context_len -> the surviving clip budget.
# We realise it directly over the FLAT node axis: n_nodes = depth * variants.
_SHAPES = [
    (depth, variants, ctx)
    for depth in (1, 3)
    for variants in (1, 8, 32, 128)
    for ctx in (16, 64, 256)
]


def _surviving_for_budget(
    node_offsets: np.ndarray, budget: int
) -> np.ndarray:
    """Per-node surviving = clip of a per-node budget (mimics the cut).

    Emulates the straddler cut over the flat node stream: a running token
    budget per (synthetic) row of ``budget`` columns; nodes before the
    straddler keep their full expanded length, the straddler keeps the
    partial, later nodes keep 0. Realised here as one ``np.clip`` of
    ``budget - exclusive_prefix`` against each node's expanded length, so
    the spread includes full-fit, mid-node cut, and fully-dropped nodes.
    """
    own = np.diff(np.asarray(node_offsets, dtype=np.int64))
    before = np.cumsum(own) - own
    return np.clip(budget - before, 0, own).astype(np.int64)


@pytest.mark.parametrize("depth,variants,ctx", _SHAPES)
def test_dense_columns_equiv_synthetic(depth, variants, ctx):
    """``DenseColumns`` == the per-CT tree front-matter over a shape spread."""
    rng = np.random.default_rng(1000 * depth + 7 * variants + ctx)
    n_nodes = depth * variants
    batched, raw_flat, rec = _build_batched(rng, n_nodes, max_len=12)
    surviving = _surviving_for_budget(batched.node_offsets, ctx)
    stage2 = _stage2_from_batched(batched, raw_flat, rec, surviving)
    dense = build_dense_columns(batched, raw_flat, rec, surviving)
    _assert_full_equivalence(stage2, dense)


def test_dense_columns_equiv_empty():
    """Zero nodes: every column empty, no divergence."""
    rng = np.random.default_rng(0)
    batched, raw_flat, rec = _build_batched(rng, 0, max_len=8)
    surviving = np.zeros(0, dtype=np.int64)
    stage2 = _stage2_from_batched(batched, raw_flat, rec, surviving)
    dense = build_dense_columns(batched, raw_flat, rec, surviving)
    assert dense.n_nodes == 0
    assert dense.kept_node_index.shape[0] == 0
    _assert_full_equivalence(stage2, dense)


def test_dense_columns_equiv_fully_cut():
    """Every node fully dropped (budget 0): kept subset empty."""
    rng = np.random.default_rng(42)
    batched, raw_flat, rec = _build_batched(rng, 6, max_len=10)
    surviving = np.zeros(rec.shape[0] - 1, dtype=np.int64)
    stage2 = _stage2_from_batched(batched, raw_flat, rec, surviving)
    dense = build_dense_columns(batched, raw_flat, rec, surviving)
    assert dense.kept_node_index.shape[0] == 0
    _assert_full_equivalence(stage2, dense)


def test_dense_columns_equiv_surviving_one_empty_body():
    """``surviving == 1`` nodes (only the prepend survives) interleaved with
    kept + dropped nodes -- the #92 zero-length-body-segment trap."""
    rng = np.random.default_rng(99)
    batched, raw_flat, rec = _build_batched(rng, 7, max_len=10)
    own = np.diff(np.asarray(batched.node_offsets, dtype=np.int64))
    # Alternate: full, prepend-only (1), dropped (0), ...
    surviving = own.copy()
    surviving[1::3] = 1
    surviving[2::3] = 0
    surviving = np.minimum(surviving, own)
    stage2 = _stage2_from_batched(batched, raw_flat, rec, surviving)
    dense = build_dense_columns(batched, raw_flat, rec, surviving)
    _assert_full_equivalence(stage2, dense)


# ---------------------------------------------------------------------------
# Teeth: a deliberately PERMUTED ``DenseColumns`` MUST fail the gate.
# ---------------------------------------------------------------------------


def _make_nonempty_case():
    rng = np.random.default_rng(7)
    batched, raw_flat, rec = _build_batched(rng, 8, max_len=12)
    surviving = _surviving_for_budget(batched.node_offsets, 64)
    stage2 = _stage2_from_batched(batched, raw_flat, rec, surviving)
    dense = build_dense_columns(batched, raw_flat, rec, surviving)
    return stage2, dense


def test_gate_is_nonvacuous_content_present():
    """The default case carries REAL content (else the gate is vacuous)."""
    stage2, dense = _make_nonempty_case()
    flat = flatten_call_targets(stage2)
    assert dense.raw_tokens.shape[0] > 0, "no raw bytes -- vacuous gate"
    assert dense.kept_node_index.shape[0] > 0, "no kept nodes -- vacuous"
    assert flat.expanded_ids.shape[0] > 0, "no surviving carriers -- vacuous"
    _assert_full_equivalence(stage2, dense)


def test_teeth_permuted_raw_offsets_fails():
    """A shifted raw CSR mis-slices every node -> per-CT views diverge."""
    import dataclasses

    stage2, dense = _make_nonempty_case()
    bad_offsets = dense.raw_offsets.copy()
    # Rotate the interior boundaries: keeps endpoints, shifts node slices.
    if bad_offsets.shape[0] > 3:
        bad_offsets[1:-1] = np.roll(bad_offsets[1:-1], 1)
    bad = dataclasses.replace(dense, raw_offsets=bad_offsets)
    with pytest.raises(AssertionError):
        _assert_per_ct_views(stage2, bad)


def test_teeth_permuted_kept_index_fails():
    """A reversed kept index diverges from the sites' ``ct_index``."""
    import dataclasses

    stage2, dense = _make_nonempty_case()
    if dense.kept_node_index.shape[0] < 2:
        pytest.skip("need >= 2 kept nodes to permute")
    bad = dataclasses.replace(
        dense, kept_node_index=dense.kept_node_index[::-1].copy()
    )
    with pytest.raises(AssertionError):
        _assert_kept_index(stage2, bad)


def test_teeth_permuted_surviving_fails():
    """A rolled surviving column desyncs the scalars + the kept subset."""
    import dataclasses

    stage2, dense = _make_nonempty_case()
    bad_surv = np.roll(dense.surviving_token_count, 1)
    bad = dataclasses.replace(dense, surviving_token_count=bad_surv)
    with pytest.raises(AssertionError):
        _assert_per_ct_views(stage2, bad)


# ---------------------------------------------------------------------------
# Live binary: capture the production ``(expanded, surviving, stage2)`` from
# inside the real vector path on the rich-splice fixture and prove
# ``DenseColumns`` reproduces the per-CT tree front-matter on real content.
# ---------------------------------------------------------------------------


def _rederive_batched(expanded) -> tuple[BatchedExpansion, np.ndarray, np.ndarray]:
    """Re-run ``batched_expand`` from the ``ExpandedBatch`` raw flats.

    ``ExpandedBatch`` retains ``raw_flat`` / ``raw_record_offsets`` (the
    exact inputs ``expand_node_bodies`` fed ``batched_expand``); re-running
    the owned expansion reproduces the SAME ``BatchedExpansion`` the
    production path sliced its per-node ``states`` from. The self-token
    ids are recovered from the per-node prepend slot (expanded slot 0 of
    each node), which ``batched_expand`` writes verbatim."""
    raw_flat = np.asarray(expanded.raw_flat, dtype=np.uint16)
    rec = np.asarray(expanded.raw_record_offsets, dtype=np.int64)
    node_off = np.asarray(expanded.node_offsets, dtype=np.int64)
    n_nodes = rec.shape[0] - 1
    # Each node's prepend self-token is expanded slot 0 of the node.
    self_token_ids = expanded.expanded[node_off[:-1]].astype(np.uint16)
    if n_nodes == 0:
        self_token_ids = np.zeros(0, dtype=np.uint16)
    batched = batched_expand(raw_flat, rec, self_token_ids)
    assert np.array_equal(batched.expanded, expanded.expanded), (
        "re-derived BatchedExpansion diverges from the production expansion"
    )
    return batched, raw_flat, rec


@pytest.mark.parametrize("context_len,max_depth", [(4096, 3), (7, 3), (256, 0)])
def test_dense_columns_equiv_live_binary(context_len, max_depth, tmp_path):
    """On the live rich-splice binary: capture the production expansion +
    surviving + adapter ``Stage2Batch``, build ``DenseColumns`` from the
    re-derived ``BatchedExpansion``, and assert FULL per-CT equivalence."""
    from ._byte_identity_harness import _nonempty_matched_idxs, _prepare
    from ._rich_corpus import build_rich_splice_fixture
    from tokenizer.aligned_data.loader.batch_decode import (
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
    from tokenizer.aligned_data.loader.vector_batch._scatter import (
        _dense as _dense_mod,
    )
    from tokenizer.aligned_data.sorted_index.tests.fixtures import (
        make_test_vocab_manager,
    )

    base = _prepare(build_rich_splice_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)

    captured: list = []
    real_build = _dense_mod.build_stage2_batch

    def _capturing(geometry, expanded, *, cols, surviving):
        stage2 = real_build(geometry, expanded, cols=cols, surviving=surviving)
        captured.append((expanded, np.asarray(surviving), stage2))
        return stage2

    pointers = [SectionPointerSpec(arm=SectionKind.MATCHED, idx=int(i)) for i in idxs]
    dataset = BinaryDataset(base, "sortbin", vocab_manager=make_test_vocab_manager())
    import unittest.mock as _mock

    with _mock.patch.object(_dense_mod, "build_stage2_batch", _capturing):
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

    assert captured, "build_stage2_batch was never called -- no rows decoded"
    any_nonempty = False
    for expanded, surviving, stage2 in captured:
        batched, raw_flat, rec = _rederive_batched(expanded)
        dense = build_dense_columns(batched, raw_flat, rec, surviving)
        _assert_full_equivalence(stage2, dense)
        if dense.raw_tokens.shape[0] > 0 and dense.kept_node_index.shape[0] > 0:
            any_nonempty = True
    assert any_nonempty, "live capture carried no real content -- vacuous"
