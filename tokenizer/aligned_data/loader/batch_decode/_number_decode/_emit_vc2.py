"""Batched per-carrier row emission for the VC2 (``valued_const_v2``)
TokenType.

Single concern: emit ``K_visible`` rows for EVERY VC2 carrier across the
batch in a single vectorised pass per ALG-8. The MSB chunk may have
fewer than 8 payload bytes when ``L % 8 != 0``; padding slots reference
``inline_bytes[0]`` (3a's leading-zero pad).

Vectorisation strategy
----------------------

* ``K_visible`` per carrier is computed from a SEGMENTED
  trailing-painted-run over the flat ``extra_value_v2_mask[:surviving]``
  concatenation (one segment per call_target). The run is capped at both
  the segment's surviving prefix and ``K_full[c] = max(1, ceil(L / 8))``.
* Per-chunk byte rows are built via a single meshgrid over ALL carriers'
  chunks in one shot; the per-chunk MSB-clipping uses ``np.where`` to
  substitute the leading-zero pad reference for the short MSB chunk's pad
  bytes.

The cross-call_target batching (B-S2b) replaces the prior
per-call_target invocation: the trailing-painted-run that was a
per-call_target reverse-cumsum is now a single segmented barrier scan,
and the byte rows are one meshgrid over every VC2 carrier in the batch.
"""

from __future__ import annotations

import numpy as np

from ._seg_lengths import seg_lengths_from_base


__all__ = ["emit_vc2_rows"]


def _segmented_trailing_painted_run(
    painted_flat: np.ndarray, seg_offsets: np.ndarray
) -> np.ndarray:
    """``run[i]`` = consecutive True at ``i, i+1, ...`` within ``i``'s
    segment, over the flat ``extra_value_v2_mask[:surviving]``
    concatenation.

    Segment-wise equivalent of the prior per-call_target reverse-cumsum
    trailing-run. Barrier model: a True run at ``i`` ends at the first
    index ``>= i`` that is either False or the segment's right edge, so
    ``run[i] = min(first_false_at_or_after[i], seg_end[i]) - i`` for True
    ``i`` (0 for False). ``first_false_at_or_after`` is a reverse running
    minimum of "index where False else n"; the segment-end cap prevents a
    run bleeding into the next call_target. Every kept call_target has
    ``surviving >= 1`` so no zero-length segments occur.
    """
    n = int(painted_flat.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.int64)
    is_true = painted_flat.astype(bool)
    idx = np.arange(n, dtype=np.int64)
    false_pos = np.where(~is_true, idx, n)
    first_false = np.minimum.accumulate(false_pos[::-1])[::-1]
    # seg_id[i] = segment owning flat index i; seg_end (exclusive) =
    # seg_offsets[seg_id + 1]. Built without a Python loop via the CSR
    # start-mark + cumsum (same expansion as segment_ids_from_offsets,
    # inlined here to keep the seg_end gather local to this kernel).
    seg_id = np.zeros(n, dtype=np.int64)
    if seg_offsets.shape[0] > 2:
        seg_id[seg_offsets[1:-1]] = 1
        np.cumsum(seg_id, out=seg_id)
    seg_end = seg_offsets[1:][seg_id]
    stop = np.minimum(first_false, seg_end)
    return np.where(is_true, stop - idx, 0).astype(np.int64)


def emit_vc2_rows(
    *,
    p_carriers: np.ndarray,
    p_carrier_bytes: np.ndarray,
    expanded_positions: np.ndarray,
    carrier_seg: np.ndarray,
    seg_painted_offsets: np.ndarray,
    seg_painted_vc2_flat: np.ndarray,
    seg_surviving: np.ndarray,
    seg_runlen_base: np.ndarray,
    runlen_number_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Emit per-chunk byte rows for EVERY VC2 carrier per ALG-8, batched.

    All carrier-axis arrays are parallel + in DFS-then-stream order.

    Per carrier ``c``:

    * ``L[c] = runlen_number[p_carrier[c] + 1]`` (read from the owning
      segment's flat runlength).
    * ``K_full[c] = max(1, ceil(L[c] / 8))``.
    * ``K_visible[c]`` = ``1 +`` count of consecutive
      ``extra_value_v2_mask`` True positions immediately after the
      carrier in expanded space (segmented trailing-painted-run), capped
      by both the surviving prefix and ``K_full[c]``.

    Returns ``(rows, rows_per_carrier, chunk_indices)``:

    * ``rows`` -- ``u32[total_rows, 8]`` gather-offset rows (all
      carriers' chunks concatenated in carrier order, LSB-first per
      carrier).
    * ``rows_per_carrier`` -- ``int64[n_carriers]`` = ``K_visible`` per
      carrier (for the per-call_target slice reconstruction).
    * ``chunk_indices`` -- ``int64[total_rows]`` per-chunk index within
      source (``0 = LSB``); feeds the VC2 chunk-exponent sidecar.

    Short MSB chunks left-pad with ``inline_bytes[0]`` references
    (zeros): for ``L=17`` the MSB chunk yields ``[0]*7 + [p_carrier_byte]``.
    """
    n_carriers = int(p_carriers.shape[0])
    if n_carriers == 0:
        return (
            np.empty((0, 8), dtype=np.uint32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    # ALG-8: ``L = runlen_number[p_carrier + 1]`` from the owning segment.
    # ``runlen_number`` is length ``N`` (only ``digit_cumsum`` is ``N+1``),
    # so the ``+1`` lookahead must be bounds-guarded against the carrier's
    # OWN segment -- mirrors the F128 emitter's per-segment guard. A VC2
    # carrier at the segment's final raw position has no ``+1`` slot; its
    # value is zero-length, giving ``L = 0`` (then ``K_full = max(1, 0) =
    # 1``, the single LSB chunk -- the correct ALG-8 zero-payload result).
    # Without the guard this either reads off the end of the flat array
    # (terminal carrier in the LAST segment -> IndexError) or silently
    # misreads the NEIGHBOUR segment's value (terminal in a non-last
    # segment).
    seg_runlen_len = seg_lengths_from_base(
        seg_runlen_base, int(runlen_number_flat.shape[0])
    )
    lookahead_raw = p_carriers + 1
    in_seg = lookahead_raw < seg_runlen_len[carrier_seg]
    L = np.zeros(n_carriers, dtype=np.int64)
    L[in_seg] = runlen_number_flat[
        seg_runlen_base[carrier_seg[in_seg]] + lookahead_raw[in_seg]
    ]
    K_full = np.maximum(1, (L + 7) // 8)

    # Segmented trailing-painted-run, then per-carrier lookahead at
    # ``expanded_position + 1`` within the carrier's segment.
    trailing_run = _segmented_trailing_painted_run(
        seg_painted_vc2_flat, seg_painted_offsets
    )
    lookahead = expanded_positions + 1
    surviving_per_carrier = seg_surviving[carrier_seg]
    in_range = lookahead < surviving_per_carrier
    run_lengths = np.zeros(n_carriers, dtype=np.int64)
    if in_range.any():
        # Flat index into the per-segment painted prefix.
        flat_lookahead = (
            seg_painted_offsets[carrier_seg[in_range]]
            + lookahead[in_range]
        )
        run_lengths[in_range] = trailing_run[flat_lookahead]
    K_visible = 1 + np.minimum(run_lengths, K_full - 1)

    # Flatten every carrier's ``K_visible[c]`` rows into one
    # ``(total_rows, 8)`` block (carrier order, LSB-first per carrier).
    total_rows = int(K_visible.sum())
    source_starts = np.empty(n_carriers + 1, dtype=np.int64)
    source_starts[0] = 0
    np.cumsum(K_visible, out=source_starts[1:])

    row_indices = np.arange(total_rows, dtype=np.int64)
    source_idx_per_row = np.searchsorted(
        source_starts[1:], row_indices, side="right"
    )
    c_within = row_indices - source_starts[source_idx_per_row]

    p_carrier_bytes_i64 = p_carrier_bytes.astype(np.int64, copy=False)
    L_per_row = L[source_idx_per_row]
    p_per_row = p_carrier_bytes_i64[source_idx_per_row]
    unclipped_starts = p_per_row + L_per_row - 8 * (c_within + 1)
    byte_idx = np.arange(8, dtype=np.int64)
    cols = unclipped_starts[:, np.newaxis] + byte_idx[np.newaxis, :]
    rows = np.where(
        cols < p_per_row[:, np.newaxis],
        np.uint32(0),
        cols.astype(np.uint32),
    )

    return rows, K_visible, c_within
