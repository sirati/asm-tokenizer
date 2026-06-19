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
from typing import TYPE_CHECKING, List

import numpy as np

from .._flat_call_targets import iter_call_target_columns

if TYPE_CHECKING:
    from .._types import Stage2Batch


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
    stage2: "Stage2Batch",
    inline_byte_slices: List[slice],
) -> FlatSegments:
    """Concatenate the per-segment NUMBER-band context, batched.

    Walks every kept (``surviving_token_count > 0``) call_target's shared
    Step-1 columns once and concatenates the body axis + per-segment CSR
    context the emission kernel reads. No decode rule runs here -- the
    carrier mask, segmented expanded->raw recovery, byte-offset
    arithmetic, and per-type emission all run in the kernel over these
    arrays.
    """
    n_total_cts = len(inline_byte_slices)
    kept = [
        cols
        for cols in iter_call_target_columns(stage2)
        if cols.surviving_token_count > 0
    ]
    ct_index = np.asarray(
        [cols.dfs_index for cols in kept], dtype=np.int64
    )
    n_kept = len(kept)

    expanded_body_chunks: List[np.ndarray] = []
    painted_body_chunks: List[np.ndarray] = []
    painted_vc2_prefix_chunks: List[np.ndarray] = []
    real_pos_chunks: List[np.ndarray] = []
    digit_chunks: List[np.ndarray] = []
    runlen_chunks: List[np.ndarray] = []
    f128_full_chunks: List[np.ndarray] = []
    body_seg_len = np.empty(n_kept, dtype=np.int64)
    painted_prefix_seg_len = np.empty(n_kept, dtype=np.int64)
    real_seg_base = np.empty(n_kept, dtype=np.int64)
    digit_base = np.empty(n_kept, dtype=np.int64)
    seg_runlen_base = np.empty(n_kept, dtype=np.int64)
    seg_f128_base = np.empty(n_kept, dtype=np.int64)
    seg_surviving = np.empty(n_kept, dtype=np.int64)
    seg_slice_start = np.empty(n_kept, dtype=np.int64)

    real_running = 0
    digit_running = 0
    runlen_running = 0
    f128_running = 0
    for i, cols in enumerate(kept):
        surviving = cols.surviving_token_count
        seg_surviving[i] = surviving
        seg_slice_start[i] = int(inline_byte_slices[cols.dfs_index].start)
        body = max(surviving - 1, 0)
        body_seg_len[i] = body
        # Body axis: ``expanded[1:surviving]`` + the two extra masks over
        # the same body positions (slot j == ``expanded[j + 1]``).
        expanded_body_chunks.append(
            cols.expanded_token_ids[1:surviving].astype(np.int64, copy=False)
        )
        painted_body_chunks.append(
            cols.extra_value_v2_mask[1:surviving]
            | cols.extra_f128_mask[1:surviving]
        )
        # Surviving-prefix VC2 painted mask (axis ``[:surviving]``); the
        # VC2 emitter's trailing-run lookahead indexes carrier expanded
        # positions directly into this prefix.
        painted_vc2_prefix_chunks.append(
            cols.extra_value_v2_mask[:surviving].astype(np.int64, copy=False)
        )
        painted_prefix_seg_len[i] = surviving
        real_positions = np.nonzero(cols.real_mask)[0]
        real_pos_chunks.append(real_positions.astype(np.int64, copy=False))
        real_seg_base[i] = real_running
        real_running += int(real_positions.shape[0])
        dc = cols.digit_cumsum.astype(np.int64, copy=False)
        digit_chunks.append(dc)
        digit_base[i] = digit_running
        digit_running += int(dc.shape[0])
        rl = cols.runlen_number.astype(np.int64, copy=False)
        runlen_chunks.append(rl)
        seg_runlen_base[i] = runlen_running
        runlen_running += int(rl.shape[0])
        # FULL ``extra_f128_mask`` (not surviving-clipped): the F128 finite
        # signal reads ``extra_f128_mask[expanded_pos + 1]`` against the
        # full mask per ALG-2 (a mid-cut finite source still reports
        # finite even when its painted MSB slot is past the cut).
        ef = cols.extra_f128_mask
        f128_full_chunks.append(ef)
        seg_f128_base[i] = f128_running
        f128_running += int(ef.shape[0])

    seg_painted_offsets = np.zeros(n_kept + 1, dtype=np.int64)
    if n_kept > 0:
        np.cumsum(painted_prefix_seg_len, out=seg_painted_offsets[1:])

    def _concat(chunks: List[np.ndarray], dtype) -> np.ndarray:
        return (
            np.concatenate(chunks)
            if chunks
            else np.empty(0, dtype=dtype)
        )

    return FlatSegments(
        expanded_body=_concat(expanded_body_chunks, np.int64),
        painted_body=_concat(painted_body_chunks, np.bool_),
        body_seg_len=body_seg_len,
        real_pos_flat=_concat(real_pos_chunks, np.int64),
        real_seg_base=real_seg_base,
        digit_flat=_concat(digit_chunks, np.int64),
        digit_base=digit_base,
        seg_slice_start=seg_slice_start,
        seg_painted_vc2_flat=_concat(painted_vc2_prefix_chunks, np.int64),
        seg_painted_offsets=seg_painted_offsets,
        seg_surviving=seg_surviving,
        seg_runlen_base=seg_runlen_base,
        runlen_number_flat=_concat(runlen_chunks, np.int64),
        seg_f128_base=seg_f128_base,
        f128_full_mask_flat=_concat(f128_full_chunks, np.bool_),
        ct_index=ct_index,
        n_total_cts=n_total_cts,
    )
