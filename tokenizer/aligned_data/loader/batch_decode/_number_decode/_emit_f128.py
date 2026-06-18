"""Batched per-carrier row emission for the F128 TokenType.

Single concern: emit 1 or 2 rows for EVERY F128 source across the batch
per ALG-2 + ALG-7, in one vectorised pass.

* Finite source (ALG-2 painted continuation present in the FULL
  expanded mask): 2 chunks (LSB then MSB limb) -- INDEPENDENT of the
  per-row cutoff. The MSB chunk's bytes are required for 3d's
  ``actual_exp`` extraction (LSB chunk's exponent base =
  actual_exp - 112), so it MUST be emitted even when the painted MSB
  slot is past ``partial_cut_length``.
* NaN/Inf source (no continuation): 1 chunk = MSB limb (bytes 0..7);
  3d branches on ``f128_is_nan_or_inf`` to call ``_encode_infnan``.

Stage 4's per-row sidecar concat walks the surviving expanded prefix
and naturally drops the trailing invisible MSB chunk for a mid-cut
finite source: chunks are emitted in stream-emission order (LSB then
MSB per finite source), so the stream-visible F128 count slices the
per-CT FLOAT128 chunk array at exactly the right tail.

The cross-call_target batching (B-S2b) replaces the prior per-source
Python loop: the finite signal, the per-source chunk count, and the
LSB/MSB byte rows are all built as vectorised meshgrids over every F128
carrier in the batch.
"""

from __future__ import annotations

import numpy as np


__all__ = ["emit_f128_rows"]


def emit_f128_rows(
    *,
    p_carrier_bytes: np.ndarray,
    expanded_positions: np.ndarray,
    carrier_seg: np.ndarray,
    seg_f128_base: np.ndarray,
    f128_full_mask_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Emit ALG-2 chunks for EVERY F128 carrier, batched.

    All carrier-axis arrays are parallel + in DFS-then-stream order.
    Per carrier ``c`` the finite signal is read from the FULL
    ``extra_f128_mask`` at ``expanded_position[c] + 1`` within the
    carrier's segment (NOT clipped to surviving, per ALG-2).

    Returns ``(rows, rows_per_carrier, is_nan_or_inf)``:

    * ``rows`` -- ``u32[total_rows, 8]`` gather-offset rows; per finite
      carrier 2 rows (LSB limb bytes 8..15, then MSB limb bytes 0..7),
      per NaN/Inf carrier 1 row (MSB limb bytes 0..7). Carriers are laid
      out in carrier order; chunks within a carrier are LSB-first.
    * ``rows_per_carrier`` -- ``int64[n_carriers]`` = 2 for finite, 1 for
      NaN/Inf (the per-call_target slice reconstruction sums these).
    * ``is_nan_or_inf`` -- ``bool[n_carriers]`` per-source NaN/Inf flag
      (one entry per carrier, in carrier order).
    """
    n_carriers = int(p_carrier_bytes.shape[0])
    if n_carriers == 0:
        return (
            np.empty((0, 8), dtype=np.uint32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.bool_),
        )

    # Finite signal: ``extra_f128_mask[expanded_pos + 1]`` against the
    # FULL per-segment mask. ``expanded_pos + 1`` is always in-bounds for
    # a finite carrier (the painting placed a slot there); a carrier at
    # the segment's final FULL slot has no ``+1`` slot and is NaN/Inf.
    seg_base = seg_f128_base[carrier_seg]
    seg_full_len = np.diff(
        np.concatenate(
            [seg_f128_base, np.array([f128_full_mask_flat.shape[0]], dtype=np.int64)]
        )
    )
    lookahead_local = expanded_positions + 1
    in_range = lookahead_local < seg_full_len[carrier_seg]
    is_finite = np.zeros(n_carriers, dtype=np.bool_)
    if in_range.any():
        gather = seg_base[in_range] + lookahead_local[in_range]
        is_finite[in_range] = f128_full_mask_flat[gather]
    is_nan_or_inf = ~is_finite

    rows_per_carrier = np.where(is_finite, 2, 1).astype(np.int64)

    # Flatten carriers' chunks into one ``(total_rows, 8)`` block in
    # carrier order, LSB-first per carrier.
    total_rows = int(rows_per_carrier.sum())
    source_starts = np.empty(n_carriers + 1, dtype=np.int64)
    source_starts[0] = 0
    np.cumsum(rows_per_carrier, out=source_starts[1:])

    row_indices = np.arange(total_rows, dtype=np.int64)
    source_idx_per_row = np.searchsorted(
        source_starts[1:], row_indices, side="right"
    )
    c_within = row_indices - source_starts[source_idx_per_row]

    p_per_row = p_carrier_bytes.astype(np.int64, copy=False)[source_idx_per_row]
    # Chunk 0 = LSB limb (bytes 8..15); chunk 1 = MSB limb (bytes 0..7).
    # NaN/Inf carriers have only chunk 0, which for them is the MSB limb
    # (single chunk). Encode the limb base per row: finite chunk0 -> +8,
    # finite chunk1 -> +0, NaN/Inf chunk0 -> +0.
    finite_per_row = is_finite[source_idx_per_row]
    limb_base = np.where(
        finite_per_row & (c_within == 0), np.int64(8), np.int64(0)
    )
    byte_idx = np.arange(8, dtype=np.int64)
    rows = (
        (p_per_row + limb_base)[:, np.newaxis] + byte_idx[np.newaxis, :]
    ).astype(np.uint32)

    return rows, rows_per_carrier, is_nan_or_inf
