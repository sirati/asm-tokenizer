"""Merge per-arm full-batch results into one batch result.

Single concern: combine the per-arm :class:`VectorBatchResult`s the entry
orchestrator produces -- one per arm, each filling its OWN (disjoint)
batch rows and leaving every other row empty -- into the single
full-batch result. Because the arms partition the non-padding rows (each
batch row belongs to exactly one arm), the merge is a per-batch-row
gather: row ``r``'s tokens / dense segments come from whichever arm owns
``r``, reassembled in batch-row order.

Boundary contract (design-first sentence): *given a list of arm-local
full-batch results that fill disjoint rows of the SAME ``[B, L]`` /
``[B + 1]`` CSR layout, return the row-wise union -- tokens summed over
disjoint rows, every CSR sidecar reconcatenated per row.* This module
owns NO decode logic; it only stitches already-decoded per-arm rows.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ._result import VectorBatchResult


__all__ = ["merge_arm_results"]


def merge_arm_results(
    results: Sequence[VectorBatchResult],
    *,
    batch_idx_to_section_variant: np.ndarray,
):
    """Row-wise union of per-arm full-batch results (disjoint rows).

    Each entry is a full-batch :class:`...vector_batch._result.
    VectorBatchResult` whose rows are populated for ONE arm only (the
    arm's batch rows carry that arm's tokens + dense segments; every
    other row is the empty / padding default). The arms partition the
    non-padding rows, so per row at most one arm is non-empty and the
    union is unambiguous.

    ``batch_idx_to_section_variant`` is the SHARED canonical mapping (the
    sampler computed it once, arm-agnostic); each per-arm result instead
    carries an arm-MASKED mapping (other arms' rows blanked to padding),
    so the canonical mapping is threaded in explicitly and stamped onto
    the merged result.
    """
    if not results:
        raise ValueError("merge_arm_results requires at least one result")
    if len(results) == 1:
        return VectorBatchResult(
            tokens=results[0].tokens,
            batch_idx_to_section_variant=batch_idx_to_section_variant,
            identities=results[0].identities,
            identity_row_offsets=results[0].identity_row_offsets,
            numbers_significant=results[0].numbers_significant,
            numbers_sign_exponent=results[0].numbers_sign_exponent,
            number_row_offsets=results[0].number_row_offsets,
            fid_sidecar=results[0].fid_sidecar,
            fid_row_offsets=results[0].fid_row_offsets,
            fid_per_category_counts=results[0].fid_per_category_counts,
        )

    tokens = _sum_disjoint_rows([r.tokens for r in results])
    identities, identity_row_offsets = _merge_csr(
        [r.identities for r in results],
        [r.identity_row_offsets for r in results],
    )
    numbers_significant, number_row_offsets = _merge_csr(
        [r.numbers_significant for r in results],
        [r.number_row_offsets for r in results],
    )
    numbers_sign_exponent, _ = _merge_csr(
        [r.numbers_sign_exponent for r in results],
        [r.number_row_offsets for r in results],
    )
    fid_sidecar, fid_row_offsets, fid_per_category_counts = _merge_fid(results)

    return VectorBatchResult(
        tokens=tokens,
        batch_idx_to_section_variant=batch_idx_to_section_variant,
        identities=identities,
        identity_row_offsets=identity_row_offsets,
        numbers_significant=numbers_significant,
        numbers_sign_exponent=numbers_sign_exponent,
        number_row_offsets=number_row_offsets,
        fid_sidecar=fid_sidecar,
        fid_row_offsets=fid_row_offsets,
        fid_per_category_counts=fid_per_category_counts,
    )


def _sum_disjoint_rows(token_tensors: List[np.ndarray]) -> np.ndarray:
    """Sum ``[B, L]`` token tensors whose non-zero rows are disjoint.

    Each arm filled only its own rows (others are all-zero), so the
    element-wise sum reconstructs the full batch without overlap.
    """
    out = np.zeros_like(token_tensors[0])
    for tensor in token_tensors:
        out += tensor
    return out


def _merge_csr(
    flats: List[np.ndarray], offsets: List[np.ndarray]
):
    """Row-wise concat of per-arm CSR arrays over the SAME batch rows.

    ``offsets[a]`` is arm ``a``'s ``[B + 1]`` CSR (rows it does not own
    are zero-length). The merged per-row length is the SUM across arms
    (at most one arm is non-zero per row), and the merged flat is each
    row's segment taken from whichever arm owns it, in batch-row order.
    """
    offsets = [np.asarray(o, dtype=np.int64) for o in offsets]
    batch_rows = offsets[0].size - 1
    per_row_len = np.zeros(batch_rows, dtype=np.int64)
    for off in offsets:
        per_row_len += np.diff(off)
    merged_offsets = np.zeros(batch_rows + 1, dtype=np.uint32)
    np.cumsum(per_row_len, out=merged_offsets[1:])

    total = int(merged_offsets[-1])
    template = flats[0]
    merged = np.empty(total, dtype=template.dtype)
    # Place each arm's per-row segments at their merged row positions.
    # Disjoint ownership means an arm's contribution for a row it does
    # not own is empty, so the writes never collide.
    for flat, off in zip(flats, offsets):
        flat = np.asarray(flat)
        seg_len = np.diff(off)
        owns = seg_len > 0
        if not owns.any():
            continue
        rows = np.nonzero(owns)[0]
        for r in rows:
            dst = int(merged_offsets[r])
            src = int(off[r])
            n = int(seg_len[r])
            merged[dst : dst + n] = flat[src : src + n]
    return merged, merged_offsets


def _merge_fid(results: Sequence[VectorBatchResult]):
    """Merge the optional per-Category FID sidecars (or ``None``).

    The FID sidecar is opt-in; either every arm carries it or none does
    (the flag is shared across the per-arm runs). The flat FID array +
    its row offsets merge exactly like the other CSR sidecars; the
    ``[B, 3]`` per-category counts sum over arms (disjoint rows).
    """
    if results[0].fid_sidecar is None:
        return None, None, None
    fid_sidecar, fid_row_offsets = _merge_csr(
        [r.fid_sidecar for r in results],
        [r.fid_row_offsets for r in results],
    )
    counts = np.zeros_like(np.asarray(results[0].fid_per_category_counts))
    for r in results:
        counts = counts + np.asarray(r.fid_per_category_counts)
    return fid_sidecar, fid_row_offsets, counts
