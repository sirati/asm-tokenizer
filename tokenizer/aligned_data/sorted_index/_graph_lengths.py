"""Splice-graph depth-DP over the columnar section catalog.

Single concern: per-(matched section, variant) depth-``d`` spliced
token lengths, computed WITHOUT loading a single token body per call
target. The semantics replicated here are the stage-1 callee walk +
stage-2 length predict under the sorted-index build's no-cutoff budget
(see :mod:`._length_compute`):

* a node is one (section, variant) pair; its OWN length is
  ``1 + contributing-body-length(record)`` -- the prepended self-token
  plus the record's post-promotion post-strip stream length, computed
  in bulk by :func:`...loader.batch_decode._bulk_expand_lengths.
  bulk_contributing_body_lengths`;
* node edges mirror the walker's gates one-for-one: one edge per
  call_target (NOT per call site), EXTERN rows skipped, unresolved
  pointers (``function_section_ptr == 0``) skipped, cross-arm /
  unknown section offsets skipped, callee variant picked by
  :func:`...loader.decoded._variant_selection.choose_callee_variant`'s
  fallback chain (primary's first per-call entry if usable, else the
  lowest sibling variant with a usable first entry; the
  ``MISSING_VARIANT_INDEX`` sentinel is never usable);
* the depth recurrence ``total_d(node) = own(node) + sum(
  total_{d-1}(child))`` equals the walk's path_depth <= d call-target
  sum on every revisit-free tree; roots that could trip the walker's
  active-path cycle skip are flagged via :mod:`._cycle_exact` and
  replayed exactly.

``MISSING_VARIANT_INDEX`` per-call entries are reported at ERROR level
(they silently drop a splice edge -- a data-quality defect worth
surfacing on every build).
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from dedup_hashmap import HashMapU32U32, HashMapU64U32

from tokenizer.aligned_data.binary_format._bulk_geometry import (
    bulk_token_spans,
)
from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._bulk_expand_lengths import (
    bulk_contributing_body_lengths,
)
from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections

from ._cycle_exact import exact_lengths_for_root, flag_cycle_roots


__all__ = ["compute_node_lengths", "LARGE_CONTEXT_LEN"]


logger = logging.getLogger(__name__)

#: Same role as the historical walk budget: the build ASSERTS every
#: spliced length stays under this bound, because beyond it the legacy
#: stage-2 cutoff would have fired and the sorted index would silently
#: under-report (plan D-2.2).
LARGE_CONTEXT_LEN = 2**30

_U32_MISS = np.uint32(0xFFFFFFFF)

#: The record-offset shift packing ``data_offset_shifted`` (see
#: ``_matched_arm_loader``: real offset = ``data_offset_shifted << 4``).
_DATA_OFFSET_SHIFT = 4


def _own_lengths(
    cols: ColumnarSections, data_u8: np.ndarray
) -> np.ndarray:
    """``int64[total_vars]``: 1 self-token + contributing body length."""
    offsets = cols.var_data_offset_shifted.astype(np.int64) << _DATA_OFFSET_SHIFT
    uniq, inverse = np.unique(offsets, return_inverse=True)
    starts, counts = bulk_token_spans(data_u8, uniq)
    body = bulk_contributing_body_lengths(data_u8, starts, counts)
    return body[inverse] + 1


def _first_entries(cols: ColumnarSections):
    """First per-call entry per (variant, called_idx), walker order.

    Returns ``(keys_u64, first_J, first_pv)`` where ``keys_u64 =
    (variant_flat_idx << 16) | called_idx`` for each FIRST occurrence
    -- mirroring ``lookup_callee_variant_for``'s first-match contract.
    """
    pv = np.repeat(
        np.arange(cols.var_n_calls.size, dtype=np.int64), cols.var_n_calls
    )
    keys = (pv.astype(np.uint64) << np.uint64(16)) | cols.pce_called_idx.astype(
        np.uint64
    )
    uniq_keys, first_idx = np.unique(keys, return_index=True)
    return (
        uniq_keys,
        cols.pce_section_variant_index[first_idx].astype(np.int64),
        pv[first_idx],
    )


def _report_missing_sentinels(
    cols: ColumnarSections, first_J: np.ndarray, first_pv: np.ndarray,
    sec_of_var: np.ndarray,
) -> None:
    """ERROR-log MISSING_VARIANT_INDEX per-call entries (user mandate)."""
    raw_missing = int(
        (cols.pce_section_variant_index == MISSING_VARIANT_INDEX).sum()
    )
    if raw_missing == 0:
        return
    miss_mask = first_J == MISSING_VARIANT_INDEX
    secs = np.unique(sec_of_var[first_pv[miss_mask]])
    logger.error(
        "sorted_index: %d per-call entries carry MISSING_VARIANT_INDEX "
        "(%d unique (variant, call_target) slots across %d sections; "
        "sample section indices: %s). Each one silently drops a splice "
        "edge -- the callee's variant set does not cover the caller's "
        "vkey.",
        raw_missing,
        int(miss_mask.sum()),
        int(secs.size),
        secs[:10].tolist(),
    )


def _resolve_edges(
    cols: ColumnarSections,
    section_offsets: np.ndarray,
    sec_of_var: np.ndarray,
):
    """Resolve every splice edge; returns ``(parent_node, child_node)``.

    Order is the walker's: ascending parent node, then per-call-target
    table order within the parent.
    """
    n_sections = cols.n_variants.size
    total_vars = cols.var_n_calls.size

    # --- section offset -> matched section idx ---------------------------
    offs = np.asarray(section_offsets, dtype=np.int64).reshape(-1)
    if offs.size and int(offs.max()) >= 2**32:
        raise ValueError(
            "sections.bin offsets exceed the wire format's u32 "
            "function_section_ptr range; the catalog is corrupt"
        )
    sec_map = HashMapU32U32(capacity=int(n_sections * 2))
    sec_map.insert_ndarray(
        offs.astype(np.uint32), np.arange(n_sections, dtype=np.uint32)
    )

    # --- J-resolution maps ------------------------------------------------
    keys, first_J, first_pv = _first_entries(cols)
    _report_missing_sentinels(cols, first_J, first_pv, sec_of_var)
    usable = first_J != MISSING_VARIANT_INDEX

    primary = HashMapU64U32(capacity=int(keys.size * 2))
    primary.insert_ndarray(
        keys[usable], first_J[usable].astype(np.uint32)
    )

    # Fallback: lowest sibling variant (within the section) with a
    # usable first entry per called_idx. insert_ndarray is last-wins,
    # so inserting in DESCENDING within-section variant order leaves
    # the lowest variant's J in the map.
    u_pv = first_pv[usable]
    u_sec = sec_of_var[u_pv]
    within = u_pv - cols.var_offsets[:-1][u_sec]
    # called_idx is recoverable from the packed key's low 16 bits.
    u_called = (keys[usable] & np.uint64(0xFFFF)).astype(np.int64)
    desc = np.argsort(-within, kind="stable")
    fallback = HashMapU64U32(capacity=int(u_pv.size * 2))
    fallback.insert_ndarray(
        (u_sec[desc].astype(np.uint64) << np.uint64(16))
        | u_called[desc].astype(np.uint64),
        first_J[usable][desc].astype(np.uint32),
    )

    # --- candidate edges: every (variant node, call_target) pair ----------
    nct_of_var = cols.n_call_targets[sec_of_var]
    cand_off = np.zeros(total_vars + 1, dtype=np.int64)
    np.cumsum(nct_of_var, out=cand_off[1:])
    total_cand = int(cand_off[-1])
    parent = np.repeat(np.arange(total_vars, dtype=np.int64), nct_of_var)
    k = np.arange(total_cand, dtype=np.int64) - cand_off[parent]
    ct_idx = cols.ct_offsets[sec_of_var[parent]] + k

    keep = (
        (cols.ct_type[ct_idx] != int(CallTargetType.EXTERN))
        & (cols.ct_function_section_ptr[ct_idx] != 0)
    )
    callee_sec = sec_map.lookup_ndarray(
        cols.ct_function_section_ptr[ct_idx]
    ).astype(np.int64)
    keep &= callee_sec != int(_U32_MISS)

    # J: primary, else fallback, else drop.
    pkeys = (parent.astype(np.uint64) << np.uint64(16)) | k.astype(np.uint64)
    J = primary.lookup_ndarray(pkeys).astype(np.int64)
    fkeys = (
        sec_of_var[parent].astype(np.uint64) << np.uint64(16)
    ) | k.astype(np.uint64)
    J_fb = fallback.lookup_ndarray(fkeys).astype(np.int64)
    J = np.where(J != int(_U32_MISS), J, J_fb)
    keep &= J != int(_U32_MISS)

    parent = parent[keep]
    callee_sec = callee_sec[keep]
    J = J[keep]

    oor = J >= cols.n_variants[callee_sec]
    if bool(oor.any()):
        i = int(np.nonzero(oor)[0][0])
        raise IndexError(
            f"per-call entry resolves variant_index {int(J[i])} on a "
            f"section with {int(cols.n_variants[callee_sec[i]])} variants "
            f"(matched section idx {int(callee_sec[i])}); the catalog is "
            "corrupt"
        )
    child = cols.var_offsets[:-1][callee_sec] + J
    return parent, child


def compute_node_lengths(
    cols: ColumnarSections,
    section_offsets: np.ndarray,
    data_u8: np.ndarray,
    depths: List[int],
) -> Dict[int, np.ndarray]:
    """Depth-``d`` spliced length per (section, variant) node.

    Returns ``{depth -> int64[total_vars]}`` for every requested depth.
    Raises :class:`AssertionError` if any length reaches
    :data:`LARGE_CONTEXT_LEN` (the legacy build's no-cutoff guarantee
    -- plan D-2.2).
    """
    if not depths or any(d < 0 for d in depths):
        raise ValueError(f"depths must be non-empty and >= 0; got {depths!r}")
    max_depth = max(depths)
    total_vars = cols.var_n_calls.size
    sec_of_var = np.repeat(
        np.arange(cols.n_variants.size, dtype=np.int64), cols.n_variants
    )

    if total_vars == 0:
        return {d: np.zeros(0, dtype=np.int64) for d in depths}

    own = _own_lengths(cols, data_u8)

    parent = child = np.zeros(0, dtype=np.int64)
    if max_depth > 0:
        parent, child = _resolve_edges(cols, section_offsets, sec_of_var)

    # --- depth DP ----------------------------------------------------------
    results: Dict[int, np.ndarray] = {}
    if 0 in depths:
        results[0] = own.copy()
    prev = own
    for d in range(1, max_depth + 1):
        sums = np.bincount(
            parent,
            weights=prev[child].astype(np.float64),
            minlength=total_vars,
        ).astype(np.int64)
        prev = own + sums
        if d in depths:
            results[d] = prev.copy()

    # --- active-path cycle fallback -----------------------------------------
    if max_depth > 0 and parent.size:
        flagged = flag_cycle_roots(
            cols.n_variants.size,
            sec_of_var[parent],
            sec_of_var[child],
            max_depth,
        )
        if bool(flagged.any()):
            adj_counts = np.bincount(parent, minlength=total_vars)
            adj_off = np.zeros(total_vars + 1, dtype=np.int64)
            np.cumsum(adj_counts, out=adj_off[1:])
            child_sec = sec_of_var[child]
            for sec in np.nonzero(flagged)[0].tolist():
                for node in range(
                    int(cols.var_offsets[sec]), int(cols.var_offsets[sec + 1])
                ):
                    exact = exact_lengths_for_root(
                        node,
                        sec,
                        own=own,
                        adj_off=adj_off,
                        adj_child=child,
                        child_sec=child_sec,
                        max_depth=max_depth,
                        depths=depths,
                    )
                    for d, value in exact.items():
                        results[d][node] = value

    for d, arr in results.items():
        if arr.size and int(arr.max()) >= LARGE_CONTEXT_LEN:
            node = int(arr.argmax())
            raise AssertionError(
                f"sorted-index length compute: depth-{d} length "
                f"{int(arr.max())} at section_idx={int(sec_of_var[node])} "
                f"variant_idx={int(node - cols.var_offsets[sec_of_var[node]])} "
                f"reaches LARGE_CONTEXT_LEN ({LARGE_CONTEXT_LEN}); the "
                "legacy walk's cutoff would have fired here"
            )
    return results
