"""Cross-call_target NUMBER-band flat-segment extraction (batched B-S2).

Single concern: given a :class:`Stage2Batch` + the per-DFS-call_target
``inline_byte_slices``, walk the shared Step-1 call_target columns ONCE
and concatenate the per-segment context arrays the GIL-released number
emission kernel (``dedup_hashmap.build_number_idx_2d_kernel``) consumes.

This is the GIL-bound front matter that touches Python-object column
views; the carrier identification (segmented expanded->raw recovery +
byte-offset arithmetic) and the per-:class:`TokenType` ALG-2/7/8 row
emission run inside the kernel over these flat arrays.

Boundary crossed (design-first sentence): *given the Stage2 DFS
call_target hierarchy + per-call_target byte slices, produce the flat
per-segment NUMBER-band context arrays (body expanded/painted axis,
per-segment CSR bases over the real-position / digit_cumsum /
runlen_number / painted-prefix / f128-full concatenations) the emission
kernel consumes in one batched pass.* The carrier recovery + per-type
row layout are the kernel's concern; this module only walks once and
concatenates the columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from dedup_hashmap import build_flat_segments_kernel

from .._dense_columns import DenseColumns


__all__ = [
    "FlatSegments",
    "build_flat_segments",
]


@dataclass(frozen=True)
class FlatSegments:
    """Flat per-segment NUMBER-band context for the emission kernel.

    All ``*_flat`` arrays are concatenations over the SURVIVING-prefix of
    each kept call_target, laid out in DFS encounter order; the ``seg_*``
    / ``*_base`` arrays are the matching per-segment CSR bases / scalars.
    The kernel recovers each carrier's raw position + byte offset from
    these (reproducing :func:`expanded_to_raw_position_map` segment-wise)
    and emits each :class:`TokenType`'s rows.

    Fields
    ------
    expanded_body / painted_body:
        ``int64`` / ``bool`` body axis = ``expanded[1:surviving]`` (the
        prepend slot dropped) + ``extra_value_v2_mask | extra_f128_mask``
        over the same body positions.
    body_seg_len:
        ``int64[n_kept]`` -- per-segment body length (``max(surviving-1,
        0)``); the kernel builds the body CSR from it.
    real_pos_flat / real_seg_base:
        ``int64`` -- concatenated ``nonzero(real_mask)`` raw positions +
        per-segment CSR base. The carrier's raw position is
        ``real_pos_flat[real_seg_base[seg] + within_seg_real_idx]``.
    digit_flat / digit_base:
        ``int64`` -- concatenated ``digit_cumsum`` (N+1 per segment) +
        per-segment CSR base. ``carrier_byte = seg_slice_start[seg] +
        digit_flat[digit_base[seg] + raw + 1]``.
    seg_slice_start:
        ``int64[n_kept]`` -- each segment's ``inline_byte_slices.start``.
    seg_painted_vc2_flat / seg_painted_offsets / seg_surviving:
        ``int64`` -- VC2 trailing-painted-run context: the per-segment
        ``extra_value_v2_mask[:surviving]`` concatenation, its CSR
        offsets, and the per-segment ``surviving_token_count``.
    seg_runlen_base / runlen_number_flat:
        ``int64`` -- the ALG-8 ``L`` lookahead context (per-segment
        ``state.runlen_number`` concatenation + CSR base).
    seg_f128_base / f128_full_mask_flat:
        ``int64`` / ``bool`` -- the ALG-2 F128 finite-signal context
        (per-segment FULL ``extra_f128_mask`` concatenation + CSR base).
    ct_index:
        ``int64[n_kept]`` -- DFS index of each kept call_target (the
        ordinal the per-DFS-call_target slice lists are keyed by).
    n_total_cts:
        Number of call_targets in the FULL DFS enumeration (slice axis).
    """

    expanded_body: np.ndarray
    painted_body: np.ndarray
    body_seg_len: np.ndarray
    real_pos_flat: np.ndarray
    real_seg_base: np.ndarray
    digit_flat: np.ndarray
    digit_base: np.ndarray
    seg_slice_start: np.ndarray
    seg_painted_vc2_flat: np.ndarray
    seg_painted_offsets: np.ndarray
    seg_surviving: np.ndarray
    seg_runlen_base: np.ndarray
    runlen_number_flat: np.ndarray
    seg_f128_base: np.ndarray
    f128_full_mask_flat: np.ndarray
    ct_index: np.ndarray
    n_total_cts: int


def build_flat_segments(
    dense: DenseColumns,
    inline_byte_slices: List[slice],
) -> FlatSegments:
    """Concatenate the per-segment NUMBER-band context, batched.

    Walks every kept (``surviving_token_count > 0``) node's shared
    :class:`DenseColumns` columns once and concatenates the body axis +
    per-segment CSR context the emission kernel reads. No decode rule runs
    here -- the carrier mask, segmented expanded->raw recovery, byte-offset
    arithmetic, and per-type emission all run in the kernel over these
    arrays.
    """
    n_total_cts = len(inline_byte_slices)
    ct_index = np.asarray(dense.kept_node_index, dtype=np.int64)

    # Per-FULL-DFS-node ``inline_byte_slices.start`` column the kernel reads
    # as ``seg_slice_start[i] = slice_start_per_node[kept_node_index[i]]``.
    slice_start_per_node = np.fromiter(
        (s.start for s in inline_byte_slices),
        dtype=np.int64,
        count=n_total_cts,
    )

    # GIL-released Rust kernel: the per-kept-node ``DenseColumns`` slice +
    # concat (body axis / real-position / digit_cumsum / runlen / painted
    # prefix / FULL f128) in one ``py.detach`` pass. The native-dtype
    # source columns (expanded / runlen uint16, digit_cumsum uint32) cross
    # the boundary un-widened -- the kernel widens to i64 internally where
    # it accumulates; the masks stay bool.
    (
        expanded_body,
        painted_body,
        body_seg_len,
        real_pos_flat,
        real_seg_base,
        digit_flat,
        digit_base,
        seg_slice_start,
        seg_painted_vc2_flat,
        seg_painted_offsets,
        seg_surviving,
        seg_runlen_base,
        runlen_number_flat,
        seg_f128_base,
        f128_full_mask_flat,
    ) = build_flat_segments_kernel(
        np.ascontiguousarray(dense.surviving_token_count, dtype=np.int64),
        np.ascontiguousarray(dense.expanded, dtype=np.uint16),
        np.ascontiguousarray(dense.extra_value_v2_mask, dtype=np.bool_),
        np.ascontiguousarray(dense.extra_f128_mask, dtype=np.bool_),
        np.ascontiguousarray(dense.real_mask, dtype=np.bool_),
        np.ascontiguousarray(dense.runlen_number, dtype=np.uint16),
        np.ascontiguousarray(dense.digit_cumsum, dtype=np.uint32),
        np.ascontiguousarray(dense.raw_offsets, dtype=np.int64),
        np.ascontiguousarray(dense.digit_offsets, dtype=np.int64),
        np.ascontiguousarray(dense.node_offsets, dtype=np.int64),
        ct_index,
        slice_start_per_node,
    )

    return FlatSegments(
        expanded_body=expanded_body,
        painted_body=painted_body,
        body_seg_len=body_seg_len,
        real_pos_flat=real_pos_flat,
        real_seg_base=real_seg_base,
        digit_flat=digit_flat,
        digit_base=digit_base,
        seg_slice_start=seg_slice_start,
        seg_painted_vc2_flat=seg_painted_vc2_flat,
        seg_painted_offsets=seg_painted_offsets,
        seg_surviving=seg_surviving,
        seg_runlen_base=seg_runlen_base,
        runlen_number_flat=runlen_number_flat,
        seg_f128_base=seg_f128_base,
        f128_full_mask_flat=f128_full_mask_flat,
        ct_index=ct_index,
        n_total_cts=n_total_cts,
    )
