"""Cross-call_target NUMBER-band carrier identification (batched B-S2).

Single concern: given a :class:`Stage2Batch` + the per-DFS-call_target
``inline_byte_slices``, produce the FLAT cross-call_target carrier table
the per-:class:`TokenType` emitters batch over -- one entry per surviving
NUMBER-band carrier across EVERY call_target, in DFS-then-stream
encounter order, carrying the per-carrier fields the emitters read
(block index, byte offset into ``inline_bytes``, raw-stream position,
expanded-stream position, owning call_target's segment + per-segment
context).

This is the segmented form of
:func:`._per_call_target._emit_call_target_rows`'s per-call_target front
matter: the per-call_target ``expanded_to_raw_position_map`` walk +
carrier-mask gather + byte-offset lookup all run as ONE vectorised pass
over the flat concatenation of every surviving call_target's
``expanded[:surviving]`` prefix, instead of one pass per call_target.

The module re-implements no decode rule. The per-carrier byte-offset
arithmetic is identical to the scalar walk -- it is laced over flat
carriers via per-segment CSR bases + a segmented cumsum (the same
``cumsum(is_real) - 1`` -> ``real_positions`` mapping the scalar walk
does per call_target, recovered here without a Python loop).

Boundary crossed (design-first sentence): *given the Stage2 DFS
call_target hierarchy + per-call_target byte slices, produce the flat
per-carrier table (block, byte-offset, raw/expanded position, owning
segment + per-segment context) the per-TokenType row emitters consume in
one batched pass.* The per-TokenType row layout is the emitters'
concern; this module only identifies + locates the carriers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np

from .._flat_call_targets import iter_call_target_columns
from ._band_constants import (
    _NUMBER_BAND_HI_SHIFTED,
    _NUMBER_BAND_LO_SHIFTED,
)

if TYPE_CHECKING:
    from .._types import Stage2Batch


__all__ = [
    "BatchedCarriers",
    "build_batched_carriers",
]


@dataclass(frozen=True)
class BatchedCarriers:
    """Flat cross-call_target NUMBER-band carrier table.

    All per-carrier arrays are parallel and ordered DFS-then-stream
    (the canonical stage-3 linearisation): call_targets in DFS order,
    carriers within a call_target in ascending expanded-stream position.

    The per-segment context arrays (``seg_*``) carry the per-call_target
    state the VC2 / F128 emitters need (their ``surviving`` prefix
    length, the CSR over the flat painted-mask concatenation). A carrier
    indexes its owning segment via ``carrier_seg``.

    Fields
    ------
    carrier_block_idx:
        ``int64[n_carriers]`` -- NUMBER-block index (0 = VC2, 1 = F16,
        ..., 6 = F128). Equal to ``shifted_id - 1``.
    carrier_byte_offsets:
        ``int64[n_carriers]`` -- first inline-byte offset of each
        carrier's payload in ``inline_bytes`` (after the per-call_target
        leading pad).
    carrier_raw_positions:
        ``int64[n_carriers]`` -- the carrier's raw-stream position
        (``state.runlen_number`` index; ALG-8 reads ``L`` at ``+1``).
    carrier_expanded_positions:
        ``int64[n_carriers]`` -- the carrier's expanded-stream position
        within its owning call_target's prefix.
    carrier_seg:
        ``int64[n_carriers]`` -- owning call_target's index into the
        kept-call_target enumeration (segment id).
    n_kept:
        Number of surviving call_targets (segments).
    seg_painted_offsets:
        ``int64[n_kept + 1]`` -- CSR over the flat
        ``extra_value_v2_mask[:surviving]`` concatenation (one segment
        per kept call_target). Used by the VC2 emitter's segmented
        trailing-painted-run.
    seg_painted_vc2_flat:
        ``int64[total_surviving]`` -- flat ``extra_value_v2_mask
        [:surviving]`` concatenation (int64) over kept call_targets.
    seg_surviving:
        ``int64[n_kept]`` -- per-segment ``surviving_token_count``.
    seg_runlen_base:
        ``int64[n_kept]`` -- CSR base into ``runlen_number_flat`` per
        segment (the VC2 emitter reads ``L`` from it).
    runlen_number_flat:
        ``int64[sum_runlen]`` -- flat per-segment ``state.runlen_number``
        concatenation; ``runlen_number_flat[seg_runlen_base[s] + p]`` is
        segment ``s``'s ``runlen_number[p]``.
    seg_f128_base:
        ``int64[n_kept]`` -- CSR base into ``f128_full_mask_flat`` per
        segment (the F128 emitter reads the finite signal from it).
    f128_full_mask_flat:
        ``bool[sum_full_len]`` -- flat per-segment ``extra_f128_mask``
        (FULL, not surviving-clipped) concatenation; the F128 finite
        signal reads ``extra_f128_mask[expanded_pos + 1]`` against the
        full mask per ALG-2.
    """

    carrier_block_idx: np.ndarray
    carrier_byte_offsets: np.ndarray
    carrier_raw_positions: np.ndarray
    carrier_expanded_positions: np.ndarray
    carrier_seg: np.ndarray
    n_kept: int
    seg_painted_offsets: np.ndarray
    seg_painted_vc2_flat: np.ndarray
    seg_surviving: np.ndarray
    seg_runlen_base: np.ndarray
    runlen_number_flat: np.ndarray
    seg_f128_base: np.ndarray
    f128_full_mask_flat: np.ndarray


def _empty_carriers(
    n_kept: int,
    seg_painted_offsets: np.ndarray,
    seg_painted_vc2_flat: np.ndarray,
    seg_surviving: np.ndarray,
    seg_runlen_base: np.ndarray,
    runlen_number_flat: np.ndarray,
    seg_f128_base: np.ndarray,
    f128_full_mask_flat: np.ndarray,
) -> BatchedCarriers:
    """A carrier table with zero carriers (still carries kept segments)."""
    empty_i = np.empty(0, dtype=np.int64)
    return BatchedCarriers(
        carrier_block_idx=empty_i,
        carrier_byte_offsets=empty_i,
        carrier_raw_positions=empty_i,
        carrier_expanded_positions=empty_i,
        carrier_seg=empty_i,
        n_kept=n_kept,
        seg_painted_offsets=seg_painted_offsets,
        seg_painted_vc2_flat=seg_painted_vc2_flat,
        seg_surviving=seg_surviving,
        seg_runlen_base=seg_runlen_base,
        runlen_number_flat=runlen_number_flat,
        seg_f128_base=seg_f128_base,
        f128_full_mask_flat=f128_full_mask_flat,
    )


def build_batched_carriers(
    stage2: "Stage2Batch",
    inline_byte_slices: List[slice],
) -> tuple[BatchedCarriers, np.ndarray]:
    """Identify + locate every surviving NUMBER-band carrier, batched.

    Mirrors :func:`._per_call_target._emit_call_target_rows`'s per-call_target
    front matter (the lines that recover ``keep_raw_positions``, build
    ``carrier_*`` arrays, and compute ``carrier_byte_offsets``) but over
    the flat concatenation of every surviving call_target's prefix, in
    one vectorised pass.

    The recovery of each carrier's raw position reproduces
    :func:`expanded_to_raw_position_map` SEGMENT-WISE: the per-call_target
    ``carrier_idx_per_slot = cumsum(~is_extra) - 1`` becomes a global
    cumsum with the per-segment carry-in subtracted (the same trick the
    sign-collection path uses), and the carrier (always a non-painted
    slot) maps directly to ``real_positions[carrier_idx_per_slot]``.

    Returns the carrier table + ``ct_index`` (the kept call_targets' DFS
    indices, for the entry's per-DFS-call_target slice reconstruction).
    """
    kept = [
        cols
        for cols in iter_call_target_columns(stage2)
        if cols.surviving_token_count > 0
    ]
    ct_index = np.asarray(
        [cols.dfs_index for cols in kept], dtype=np.int64
    )
    n_kept = len(kept)

    # --- per-segment concatenations -----------------------------------
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
    seg_painted_vc2_flat = (
        np.concatenate(painted_vc2_prefix_chunks)
        if painted_vc2_prefix_chunks
        else np.empty(0, dtype=np.int64)
    )
    runlen_number_flat = (
        np.concatenate(runlen_chunks)
        if runlen_chunks
        else np.empty(0, dtype=np.int64)
    )
    f128_full_mask_flat = (
        np.concatenate(f128_full_chunks)
        if f128_full_chunks
        else np.empty(0, dtype=np.bool_)
    )

    if n_kept == 0:
        return (
            _empty_carriers(
                0,
                seg_painted_offsets,
                seg_painted_vc2_flat,
                seg_surviving,
                seg_runlen_base,
                runlen_number_flat,
                seg_f128_base,
                f128_full_mask_flat,
            ),
            ct_index,
        )

    expanded_body = np.concatenate(expanded_body_chunks)
    painted_body = np.concatenate(painted_body_chunks)
    real_pos_flat = np.concatenate(real_pos_chunks)
    digit_flat = np.concatenate(digit_chunks)

    total_body = int(expanded_body.shape[0])
    empty = _empty_carriers(
        n_kept,
        seg_painted_offsets,
        seg_painted_vc2_flat,
        seg_surviving,
        seg_runlen_base,
        runlen_number_flat,
        seg_f128_base,
        f128_full_mask_flat,
    )
    if total_body == 0:
        return empty, ct_index

    # CSR over the body axis (segment i = call_target i). A kept
    # call_target with ``surviving == 1`` has a ZERO-LENGTH body segment
    # (its only surviving slot is the prepend, which the body axis drops);
    # ``np.repeat`` over the per-segment lengths yields the per-slot
    # segment id correctly even for those empty segments (the
    # mark-and-cumsum CSR expansion would silently merge consecutive
    # zero-length boundaries).
    body_seg_offsets = np.zeros(n_kept + 1, dtype=np.int64)
    np.cumsum(body_seg_len, out=body_seg_offsets[1:])
    body_seg_id = np.repeat(
        np.arange(n_kept, dtype=np.int64), body_seg_len
    )

    in_number_band = (expanded_body >= _NUMBER_BAND_LO_SHIFTED) & (
        expanded_body < _NUMBER_BAND_HI_SHIFTED
    )
    is_real_body = ~painted_body
    carrier_mask = in_number_band & is_real_body
    if not carrier_mask.any():
        return empty, ct_index

    # --- segmented expanded->raw map (carrier_idx_per_slot) -----------
    is_real_i64 = is_real_body.astype(np.int64)
    global_cum = np.cumsum(is_real_i64)
    global_cum_excl = global_cum - is_real_i64
    first_idx = np.minimum(body_seg_offsets[:-1], total_body - 1)
    seg_carry_in = global_cum_excl[first_idx]
    real_idx_per_slot = global_cum - 1 - seg_carry_in[body_seg_id]

    carrier_body_idx = np.nonzero(carrier_mask)[0]
    carrier_seg = body_seg_id[carrier_body_idx]
    carrier_real_global = (
        real_seg_base[carrier_seg] + real_idx_per_slot[carrier_body_idx]
    )
    carrier_raw_positions = real_pos_flat[carrier_real_global]

    # Body slot j corresponds to ``expanded[j + 1]``; per-segment-local
    # expanded position = (j - segment_start_body) + 1.
    carrier_expanded_positions = (
        carrier_body_idx - body_seg_offsets[carrier_seg]
    ) + 1

    carrier_shifted = expanded_body[carrier_body_idx]
    carrier_block_idx = carrier_shifted - _NUMBER_BAND_LO_SHIFTED

    # ``carrier_byte = inline_slice_start[seg] + digit_cumsum[seg][raw + 1]``.
    gather_idx = digit_base[carrier_seg] + carrier_raw_positions + 1
    carrier_byte_offsets = (
        seg_slice_start[carrier_seg] + digit_flat[gather_idx]
    )

    return (
        BatchedCarriers(
            carrier_block_idx=carrier_block_idx,
            carrier_byte_offsets=carrier_byte_offsets,
            carrier_raw_positions=carrier_raw_positions,
            carrier_expanded_positions=carrier_expanded_positions,
            carrier_seg=carrier_seg,
            n_kept=n_kept,
            seg_painted_offsets=seg_painted_offsets,
            seg_painted_vc2_flat=seg_painted_vc2_flat,
            seg_surviving=seg_surviving,
            seg_runlen_base=seg_runlen_base,
            runlen_number_flat=runlen_number_flat,
            seg_f128_base=seg_f128_base,
            f128_full_mask_flat=f128_full_mask_flat,
        ),
        ct_index,
    )
