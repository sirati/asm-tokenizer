"""Flat vector emission -> staged ``Stage2Batch`` adapter (dense pass).

Single concern: build the ``Stage2Batch`` hierarchy the ``batch_decode``
dense kernels consume, sourcing EVERY per-call-target field from the
vector path's already-computed state -- never a fresh BIN parse and never
a re-expansion of ``_data.bin``.

Adapter shape: one level-2 "section" per NON-PADDING batch row; its
single level-3 variant's ``call_targets`` are that row's emitted nodes in
BFS emission order. Each level-4 ``Stage2CallTarget`` carries the vector
path's threaded per-node ``state`` + promotion masks + surviving counts,
and a ``Stage1CallTarget`` whose ``call_targets_section`` /
``function_name_ptr`` / ``encounter_category`` come from the columnar
catalog (the same fields the decode walker loads, body-free) and whose
``function_data.metadata["category_counts"]`` is decoded from the
already-gathered body (no BIN parse).

The synthetic ``batch_idx_to_section_variant`` is the IDENTITY over
non-padding rows (section ``r`` == non-padding row ``r``, variant 0) so
the kernels' per-row offset cumsums enumerate the non-padding rows in
order; the orchestrator (:mod:`._dense`) re-expands onto the true batch.
"""

from __future__ import annotations

from typing import List

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    _CALL_TARGET_TYPE_TO_CATEGORY,
)
from tokenizer.aligned_data.loader.batch_decode._surviving_counts import (
    count_surviving,
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
from tokenizer.aligned_data.loader.category_counts import (
    category_counts_from_runlen,
)
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import CallTarget, Section
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections

from .._types import BatchGeometry
from ._expand import ExpandedBatch


__all__ = ["build_stage2_batch"]


_PADDING_SENTINEL = int(np.iinfo(np.uint32).max)


def build_stage2_batch(
    geometry: BatchGeometry,
    expanded: ExpandedBatch,
    *,
    cols: ColumnarSections,
    surviving: np.ndarray,
) -> Stage2Batch:
    """Adapt the flat vector emission into a ``Stage2Batch``.

    Parameters
    ----------
    geometry / expanded / cols:
        See :func:`._dense.build_dense_sidecars`.
    surviving:
        ``int64[n_emitted]`` -- the per-node surviving token count from
        the shared straddler cut (:func:`._surviving.
        surviving_token_counts`). This IS each call_target's
        ``partial_cut_length`` (= its ``surviving_token_count``).
    """
    emission = geometry.emission
    roff = np.asarray(emission.row_offsets, dtype=np.int64)
    edge_type = np.asarray(emission.edge_type, dtype=np.uint8)
    node_off = np.asarray(expanded.node_offsets, dtype=np.int64)
    expanded_flat = expanded.expanded
    n_rows = int(geometry.n_rows)

    section_of_node = _section_of_node(geometry, cols)
    ct_offsets = np.asarray(cols.ct_offsets, dtype=np.int64)
    section_fid = np.asarray(cols.function_name_ptr)

    sections: List[Stage2Section] = []
    for r in range(n_rows):
        lo = int(roff[r])
        hi = int(roff[r + 1])
        s2_cts: List[Stage2CallTarget] = []
        for e in range(lo, hi):
            sec = int(section_of_node[e])
            state = expanded.states[e]
            raw_tokens = state.raw_tokens
            ct_section = _call_targets_section(cols, ct_offsets, sec)
            s1_ct = Stage1CallTarget(
                function_data=_function_data_for(
                    raw_tokens, state.runlen_number
                ),
                state=expanded.states[e],
                call_targets_section=ct_section,
                encounter_category=_CALL_TARGET_TYPE_TO_CATEGORY[
                    CallTargetType(int(edge_type[e]))
                ],
                parent_call_target_index=None if e == lo else 0,
                function_name_ptr=int(section_fid[sec]),
                path_depth=0 if e == lo else 1,
            )
            s2_cts.append(
                _stage2_call_target(
                    s1_ct,
                    expanded_token_ids=(
                        expanded_flat[node_off[e] : node_off[e + 1]]
                    ),
                    extra_value_v2_mask=expanded.extra_value_v2_masks[e],
                    extra_f128_mask=expanded.extra_f128_masks[e],
                    surviving_token_count=int(surviving[e]),
                )
            )
        sections.append(_stage2_section(r, s2_cts))

    mapping = _identity_mapping(n_rows)
    stage1_batch = Stage1Batch(
        sections=[s.stage1 for s in sections],
        batch_idx_to_section_variant=mapping,
        batch_size=n_rows,
    )
    identity_row_offsets, number_row_offsets = _row_offset_cumsums(sections)
    return Stage2Batch(
        stage1=stage1_batch,
        sections=sections,
        identity_row_offsets=identity_row_offsets,
        number_row_offsets=number_row_offsets,
    )


def _section_of_node(
    geometry: BatchGeometry, cols: ColumnarSections
) -> np.ndarray:
    """``int64[n_emitted]`` -- the catalog section index owning each node.

    The emission ``node`` axis is ``var_offsets``-major; the owning
    section is the CSR bucket of ``cols.var_offsets`` the node falls into.
    """
    nodes = np.asarray(geometry.emission.node, dtype=np.int64)
    var_offsets = np.asarray(cols.var_offsets, dtype=np.int64)
    return np.searchsorted(var_offsets, nodes, side="right") - 1


def _call_targets_section(
    cols: ColumnarSections, ct_offsets: np.ndarray, sec: int
) -> List[CallTarget]:
    """The section's parsed ``call_targets`` table from the catalog.

    Rebuilds the ``list[CallTarget]`` the decode walker passes from the
    columnar ``ct_*`` slices -- the encoder's LOCAL->PLT->EXTERN grouping
    (and within-group order) is preserved by the catalog, so the per-
    Category caller-local id density the dedup walk relies on holds.
    """
    lo = int(ct_offsets[sec])
    hi = int(ct_offsets[sec + 1])
    fids = cols.ct_function_name_ptr[lo:hi]
    secptrs = cols.ct_function_section_ptr[lo:hi]
    types = cols.ct_type[lo:hi]
    matched = cols.ct_is_matched[lo:hi]
    return [
        CallTarget(
            function_name_ptr=int(fids[i]),
            function_section_ptr=int(secptrs[i]),
            type=CallTargetType(int(types[i])),
            is_matched=bool(matched[i]),
        )
        for i in range(hi - lo)
    ]


def _function_data_for(
    raw_tokens: np.ndarray, runlen_number: np.ndarray
) -> FunctionData:
    """A minimal :class:`FunctionData` carrying ``category_counts``.

    The dense kernels read ONLY ``function_data.metadata['category_counts']``
    (the ALG-4 COUNTER offset bump) off the level-4 ``function_data``;
    the runlength sidecar (off by default) would also read ``tokens`` /
    ``*_runlength``. The COUNTER counts are decoded from the SAME raw body
    the expansion already gathered, reusing the per-node ``runlen_number``
    the batched expansion already computed (no BIN re-parse, no second
    ``InlineDecodeState`` rebuild), exactly as the loader's
    :func:`compute_category_counts` would.
    """
    return FunctionData(
        func_name="",
        metadata={
            "category_counts": category_counts_from_runlen(
                raw_tokens, runlen_number
            )
        },
        tokens=raw_tokens,
        insn_runlength=np.zeros(0, dtype=np.int64),
        block_runlength=np.zeros(0, dtype=np.int64),
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )


def _stage2_call_target(
    s1_ct: Stage1CallTarget,
    *,
    expanded_token_ids: np.ndarray,
    extra_value_v2_mask: np.ndarray,
    extra_f128_mask: np.ndarray,
    surviving_token_count: int,
) -> Stage2CallTarget:
    """Wrap one node's expansion as a ``Stage2CallTarget``.

    ``partial_cut_length == surviving_token_count`` (the row-level cut at
    the token level is already encoded in the per-node surviving count);
    ``is_cut`` is True iff the node was truncated below its full length.
    The surviving identity / number-chunk counts come from the SHARED
    band-mask kernel (:func:`count_surviving`) over the surviving prefix
    -- the same kernel Stage-2c uses.
    """
    predicted = int(expanded_token_ids.shape[0])
    counts = count_surviving(expanded_token_ids, surviving_token_count)
    return Stage2CallTarget(
        stage1=s1_ct,
        expanded_token_ids=expanded_token_ids,
        extra_value_v2_mask=extra_value_v2_mask,
        extra_f128_mask=extra_f128_mask,
        predicted_full_length=predicted,
        surviving_token_count=surviving_token_count,
        surviving_identity_count=counts.surviving_identity_count,
        surviving_number_chunk_count=counts.surviving_number_chunk_count,
        is_cut=surviving_token_count < predicted,
        partial_cut_length=surviving_token_count,
    )


def _stage2_variant(s2_cts: List[Stage2CallTarget], *, batch_idx: int):
    """Build the level-3 ``Stage2Variant`` (+ its ``Stage1Variant``).

    ``cut_call_target_index`` is the first node with zero surviving
    tokens (the cut boundary; ``len`` when the row fits entirely),
    mirroring Stage-2b's ``CutoffResult``. The per-variant aggregate
    surviving counts are the sums over the call_targets -- the same
    aggregation Stage-2 performs.
    """
    cut_idx = len(s2_cts)
    for i, ct in enumerate(s2_cts):
        if ct.surviving_token_count == 0:
            cut_idx = i
            break
    total_id = sum(ct.surviving_identity_count for ct in s2_cts)
    total_num = sum(ct.surviving_number_chunk_count for ct in s2_cts)
    total_tok = sum(ct.surviving_token_count for ct in s2_cts)
    s1_variant = Stage1Variant(
        variant_idx=0,
        variant_ref_offset=0,
        batch_idx=batch_idx,
        call_targets=[ct.stage1 for ct in s2_cts],
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )
    return Stage2Variant(
        stage1=s1_variant,
        call_targets=s2_cts,
        cut_call_target_index=cut_idx,
        total_surviving_token_count=total_tok,
        total_surviving_identity_count=total_id,
        total_surviving_number_chunk_count=total_num,
    )


def _stage2_section(r: int, s2_cts: List[Stage2CallTarget]) -> Stage2Section:
    """Build the level-2 ``Stage2Section`` for non-padding batch row ``r``.

    The level-2 ``section`` is a minimal :class:`Section` carrying the
    row's union ``call_targets`` -- read ONLY by the dedup-map capacity
    estimate (:func:`...batch_decode._assemble._build_dedup_maps`), an
    upper bound, so the per-row union is a faithful (and tight) source.
    """
    s2_variant = _stage2_variant(s2_cts, batch_idx=r)
    union_call_targets: List[CallTarget] = []
    for ct in s2_cts:
        union_call_targets.extend(ct.stage1.call_targets_section)
    section = Section(
        function_name_ptr=0,
        section_offset=r,
        call_targets=union_call_targets,
        variants=[],
    )
    s1_section = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=r,
        section=section,
        variants=[s2_variant.stage1],
    )
    return Stage2Section(stage1=s1_section, variants=[s2_variant])


def _identity_mapping(n_rows: int) -> np.ndarray:
    """``u32[n_rows, 2]`` identity mapping (row r -> section r, variant 0)."""
    mapping = np.zeros((n_rows, 2), dtype=np.uint32)
    mapping[:, 0] = np.arange(n_rows, dtype=np.uint32)
    return mapping


def _row_offset_cumsums(
    sections: List[Stage2Section],
) -> tuple[np.ndarray, np.ndarray]:
    """``(identity_row_offsets, number_row_offsets)`` -- u32[n_rows + 1].

    Per-row cumsum of the variant's aggregate surviving identity / number
    counts (one variant per synthetic section), matching Stage-2's D8
    sizing contract.
    """
    n = len(sections)
    id_len = np.empty(n, dtype=np.int64)
    num_len = np.empty(n, dtype=np.int64)
    for r, s in enumerate(sections):
        v = s.variants[0]
        id_len[r] = v.total_surviving_identity_count
        num_len[r] = v.total_surviving_number_chunk_count
    identity_row_offsets = np.zeros(n + 1, dtype=np.uint32)
    number_row_offsets = np.zeros(n + 1, dtype=np.uint32)
    np.cumsum(id_len, out=identity_row_offsets[1:])
    np.cumsum(num_len, out=number_row_offsets[1:])
    return identity_row_offsets, number_row_offsets
