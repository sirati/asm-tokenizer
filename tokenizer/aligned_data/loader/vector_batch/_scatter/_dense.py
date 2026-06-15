"""Dense identity + numeric sidecars from the already-expanded bodies.

Single concern: produce the per-batch-row DENSE sidecars -- the
post-remap caller-local->counter identity array (+ offsets), the
``(significand, sign_exp)`` numeric arrays (+ offsets), and the optional
per-Category FID sidecars -- BYTE-IDENTICAL to ``batch_decode`` with
backfill OFF, WITHOUT re-reading ``_data.bin`` or re-running
``expand_tokens``.

REUSE, NOT RE-IMPLEMENTATION (byte-identity contract): the dense decode
SEMANTICS live in ``batch_decode``'s Stage-3 / Stage-4 kernels
(``build_bulk_bytes`` = inline-byte gather + identity idx_2d + number
idx_2d + FP-normalize; ``apply_per_row_remap`` = the ALG-3/4/9 dedup
walk; ``assemble_number_sidecars`` = the per-row chunk concat). This
module DRIVES those owned kernels from the vector path's already-computed
per-node state + masks + surviving counts -- it re-implements none of the
band / promotion / remap rules, so the gate cannot diverge on decode
logic.

Boundary crossed (design-first sentence): *given the geometry + the
already-expanded per-node bodies (state + masks) + the per-node surviving
counts the token scatter produced, produce a per-batch-row
``DenseSidecars`` byte-identical to ``batch_decode``.* The flat-emission
-> staged-hierarchy ADAPTER is owned by :mod:`._dense_adapter`; this
module owns only the kernel orchestration + the batch re-expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._assemble import (
    _build_dedup_maps,
)
from tokenizer.aligned_data.loader.batch_decode._bulk_bytes import (
    build_bulk_bytes,
)
from tokenizer.aligned_data.loader.batch_decode._dedup_walk import (
    apply_per_row_remap,
)
from tokenizer.aligned_data.loader.batch_decode._sidecar_concat import (
    assemble_number_sidecars,
)
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections

from .._types import BatchGeometry
from ._dense_adapter import build_stage2_batch
from ._expand import ExpandedBatch
from ._surviving import surviving_token_counts


__all__ = ["DenseSidecars", "build_dense_sidecars"]


#: The per-row sentinel a PAD_NULL padding row carries in
#: ``batch_idx_to_section_variant``.
_PADDING_SENTINEL = int(np.iinfo(np.uint32).max)


@dataclass(frozen=True)
class DenseSidecars:
    """The per-batch-row dense identity + numeric sidecars.

    Mirrors the ``batch_decode`` ``BatchDecodeResult`` dense fields
    (post-remap identities + numbers + their row offsets + the optional
    per-Category FID arrays). All arrays are indexed by BATCH ROW (padding
    rows contribute zero-length segments), so they drop straight onto the
    final result.
    """

    identities: np.ndarray  # u16[total_surviving_identity_tokens]
    identity_row_offsets: np.ndarray  # u32[batch_size + 1]
    numbers_significant: np.ndarray  # u64[total_surviving_number_chunks]
    numbers_sign_exponent: np.ndarray  # u32[total_surviving_number_chunks]
    number_row_offsets: np.ndarray  # u32[batch_size + 1]
    fid_sidecar: Optional[np.ndarray]
    fid_row_offsets: Optional[np.ndarray]
    fid_per_category_counts: Optional[np.ndarray]


def build_dense_sidecars(
    geometry: BatchGeometry,
    expanded: ExpandedBatch,
    *,
    cols: ColumnarSections,
    batch_idx_to_section_variant: np.ndarray,
    batch_size: int,
    include_fid_sidecar: bool = False,
) -> DenseSidecars:
    """Produce the per-batch-row dense sidecars (backfill OFF).

    Parameters
    ----------
    geometry:
        The body-free prepass result (the emission CSR + the straddler
        cut). Drives the per-node surviving counts + the adapter.
    expanded:
        The vector path's per-node expansion (:class:`._expand.
        ExpandedBatch`): the flat expanded stream + CSR + the threaded
        per-node ``states`` / ``extra_*_masks`` the dense kernels read.
    cols:
        The columnar ``sections.bin`` catalog -- the source of each node's
        ``call_targets_section`` / ``function_name_ptr`` /
        ``encounter_category`` (the same fields the decode walker loads,
        body-free).
    batch_idx_to_section_variant:
        ``u32[batch_size, 2]`` -- the canonical mapping (padding rows hold
        the ``(UINT32_MAX, UINT32_MAX)`` sentinel). The per-non-padding-
        row dense arrays are placed onto the full batch through this.
    batch_size:
        ``B`` -- the full batch row count (incl. padding rows).
    include_fid_sidecar:
        When True, also produce the per-Category FID sidecars (the dedup
        walk's inverse map) per the ``BatchDecodeResult`` contract.

    Returns
    -------
    DenseSidecars
        The per-batch-row dense identity + numeric sidecars.
    """
    surviving = surviving_token_counts(geometry)
    stage2 = build_stage2_batch(
        geometry, expanded, cols=cols, surviving=surviving
    )

    # --- run the OWNED decode kernels (byte-identical by construction) ---
    stage3 = build_bulk_bytes(stage2)
    dedup_maps = _build_dedup_maps(stage3)
    (
        row_identities,
        row_fid_sidecar,
        row_fid_row_offsets,
        row_fid_per_category_counts,
    ) = apply_per_row_remap(
        stage3,
        dedup_maps=dedup_maps,
        collect_fid_sidecar=include_fid_sidecar,
    )
    row_numbers_sig, row_numbers_sex = assemble_number_sidecars(stage3)

    # The adapter's mapping is the IDENTITY over non-padding rows (section
    # i == non-padding row i, variant 0), so the kernels' per-row offsets
    # already enumerate the non-padding batch rows in order. Re-expand
    # onto the full batch (padding rows contribute zero length).
    return _expand_to_batch(
        batch_idx_to_section_variant=batch_idx_to_section_variant,
        batch_size=batch_size,
        row_identities=row_identities,
        row_identity_offsets=np.asarray(stage2.identity_row_offsets),
        row_numbers_sig=row_numbers_sig,
        row_numbers_sex=row_numbers_sex,
        row_number_offsets=np.asarray(stage2.number_row_offsets),
        row_fid_sidecar=row_fid_sidecar,
        row_fid_row_offsets=row_fid_row_offsets,
        row_fid_per_category_counts=row_fid_per_category_counts,
        include_fid_sidecar=include_fid_sidecar,
    )


def _expand_to_batch(
    *,
    batch_idx_to_section_variant: np.ndarray,
    batch_size: int,
    row_identities: np.ndarray,
    row_identity_offsets: np.ndarray,
    row_numbers_sig: np.ndarray,
    row_numbers_sex: np.ndarray,
    row_number_offsets: np.ndarray,
    row_fid_sidecar: Optional[np.ndarray],
    row_fid_row_offsets: Optional[np.ndarray],
    row_fid_per_category_counts: Optional[np.ndarray],
    include_fid_sidecar: bool,
) -> DenseSidecars:
    """Place the per-NON-PADDING-row dense arrays onto the full batch.

    The per-row arrays enumerate non-padding rows in batch order (the
    adapter built one synthetic section per non-padding row in that
    order). This re-expands each CSR-segmented array onto the full
    ``batch_size`` rows, leaving padding rows as zero-length segments --
    matching ``batch_decode``'s per-batch-row output (whose padding rows
    are zero-length too).
    """
    mapping = np.asarray(batch_idx_to_section_variant, dtype=np.int64)
    is_real = mapping[:, 0] != _PADDING_SENTINEL
    real_rows = np.nonzero(is_real)[0]

    identities, identity_row_offsets = _scatter_csr(
        row_identities, row_identity_offsets, real_rows, batch_size
    )
    numbers_sig, number_row_offsets = _scatter_csr(
        row_numbers_sig, row_number_offsets, real_rows, batch_size
    )
    numbers_sex, _ = _scatter_csr(
        row_numbers_sex, row_number_offsets, real_rows, batch_size
    )

    fid_sidecar = fid_row_offsets = fid_per_category_counts = None
    if include_fid_sidecar:
        fid_sidecar, fid_row_offsets = _scatter_csr(
            row_fid_sidecar,
            np.asarray(row_fid_row_offsets),
            real_rows,
            batch_size,
        )
        fid_per_category_counts = _scatter_per_category_counts(
            np.asarray(row_fid_per_category_counts), real_rows, batch_size
        )

    return DenseSidecars(
        identities=identities,
        identity_row_offsets=identity_row_offsets,
        numbers_significant=numbers_sig,
        numbers_sign_exponent=numbers_sex,
        number_row_offsets=number_row_offsets,
        fid_sidecar=fid_sidecar,
        fid_row_offsets=fid_row_offsets,
        fid_per_category_counts=fid_per_category_counts,
    )


def _scatter_csr(
    row_flat: np.ndarray,
    row_offsets: np.ndarray,
    real_rows: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Place a per-NON-PADDING-row CSR array onto the full batch.

    ``row_flat`` + ``row_offsets`` (``len == n_real + 1``) segment the
    non-padding rows in batch order; ``real_rows`` are their batch
    indices. Returns ``(flat, batch_offsets)`` with ``batch_offsets``
    sized ``batch_size + 1`` (padding rows are zero-length). The flat
    data is unchanged (already in batch order); only the CSR is widened
    to the full batch.
    """
    row_offsets = np.asarray(row_offsets, dtype=np.int64)
    n_real = int(real_rows.size)
    if row_offsets.size - 1 != n_real:
        raise AssertionError(
            f"row_offsets covers {row_offsets.size - 1} rows but the batch "
            f"has {n_real} non-padding rows"
        )
    per_real_len = np.diff(row_offsets)
    per_batch_len = np.zeros(batch_size, dtype=np.int64)
    if n_real:
        per_batch_len[real_rows] = per_real_len
    batch_offsets = np.zeros(batch_size + 1, dtype=np.uint32)
    np.cumsum(per_batch_len, out=batch_offsets[1:])
    return np.asarray(row_flat), batch_offsets


def _scatter_per_category_counts(
    row_counts: np.ndarray,
    real_rows: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Place the per-NON-PADDING-row ``[*, 3]`` FID counts onto the batch.

    Padding rows are ``[0, 0, 0]`` (the ``BatchDecodeResult`` contract).
    """
    out = np.zeros((batch_size, 3), dtype=np.uint32)
    if real_rows.size:
        out[real_rows] = row_counts.astype(np.uint32, copy=False)
    return out
