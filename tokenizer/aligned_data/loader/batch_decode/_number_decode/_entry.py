"""Stage 3c -- per-:class:`TokenType` number ``idx_2d`` construction.

Single concern: lay out 2D gather-offset arrays into ``inline_bytes``
for the number-arm view-cast step (ALG-7 byte layouts + ALG-8 VC2
multi-chunk packing). The vectorised FP normalisation (denormal +
NaN/Inf branches that turn the u64 bit patterns into the ``(significand,
sign_exp)`` f96 sidecar shape) is 3d's concern.

Batched (B-S2b): the prior per-call_target DFS loop is replaced by a
single batched pass. :func:`._batched_carriers.build_batched_carriers`
identifies + locates every surviving NUMBER-band carrier across the
batch; the per-:class:`TokenType` emitters
(:mod:`._emit_vc2` / :mod:`._emit_f128` / :mod:`._emit_fixed_fp`) build
each type's whole ``idx_2d`` block in one meshgrid. The per-call_target
chunk slices are reconstructed from per-DFS-call_target per-type ROW
counts (a single segmented sum + cumsum), not a per-call_target Python
walk.

Byte layouts (ALG-7)
--------------------

* F16 / BF16 -- 1 row of 2 bytes.
* F32        -- 1 row of 4 bytes.
* F64        -- 1 row of 8 bytes.
* F80        -- 1 row of 10 bytes (3d reshapes into 5 big-endian u16
  limbs for the explicit-leading-bit reassembly path).
* F128       -- ``(chunk_count, 8)`` rows. Finite sources (ALG-2)
  emit 2 chunks INDEPENDENT of the per-row cutoff: LSB limb (bytes
  8..15) then MSB limb (bytes 0..7). NaN/Inf sources emit 1 chunk =
  MSB limb (bytes 0..7); 3d uses the ``f128_is_nan_or_inf`` sidecar
  to branch into the ``_encode_infnan(sign, mantissa_is_zero)`` path
  that needs only the high limb's sign + exponent + high-mantissa
  bits. The dataloader's ".rodata-robustness" policy
  (custom_float.py:336) sanctions this approximation -- canonical
  NaN/Inf is fully determined by the high 8 bytes. The MSB chunk is
  emitted even when the painted MSB slot is past
  ``partial_cut_length`` so 3d can read ``actual_exp`` from the high
  limb (LSB chunk's exponent base = actual_exp - 112); stage 4's
  per-row sidecar concat drops the invisible MSB chunk via the
  stream-walk's surviving-prefix count, not via chunk-emission
  suppression here.
* VC2 -- variable-length payload per ALG-8; ``(K_visible, 8)`` rows.
  ``K_full = max(1, ceil(L / 8))`` and ``K_visible`` is the count of
  chunks whose expanded-stream slot survived the cut. The MSB chunk
  may have fewer than 8 payload bytes when ``L % 8 != 0``; padding
  slots reference ``inline_bytes[0]`` (3a's leading-zero pad).

ALG-8 VC2 byte layout (verbatim from the plan)
----------------------------------------------

::

    # Per VC2 source at carrier position p_carrier (raw_tokens position):
    #   p_carrier_byte: source's first inline-byte offset in inline_bytes
    #                   (after the leading pad).
    #   L            : inline_len = state.runlen_number[p_carrier + 1].
    #   K            : max(1, ceil(L / 8))  # chunk count.
    #
    # Per chunk c (0 = LSB chunk, K-1 = MSB chunk):
    #   Payload bytes: [p_carrier_byte + L - 8*(c+1), p_carrier_byte + L - 8*c)
    #     intersected with [p_carrier_byte, p_carrier_byte + L).

Mid-cut sources: chunks emit in LSB-first stream-emission order. A cut
inside a multi-chunk source drops the trailing (MSB-end) chunks. The
carrier slot is chunk 0; continuation slots are chunks 1, 2, ...

Iteration order: sections -> variants -> call_targets in DFS encounter
order (the same linearisation stage 1 + 3a use); within each
call_target's surviving prefix the carriers are walked in expanded-stream
order. Per-type row order is DFS-then-stream over carriers; the f128 /
vc2 sidecars are populated in the same per-type stream order.

ALG-8 reads ``L = state.runlen_number[p_carrier + 1]``, so we still
need the raw-stream carrier position. The expanded stream alone
doesn't tell us ``L`` (only ``K``); two distinct ``L`` values can map
to the same ``K``. The carrier table rebuilds the expanded->raw
position map (segment-wise) so ``L`` is recoverable.

Plan refs: ``batch_decode_plan.md`` ALG-7 + ALG-8 + Stage 3 step 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

from tokenizer.tokens import TokenType

from ._band_constants import (
    _FIXED_ROW_WIDTH,
    _NUMBER_BLOCK_TOKEN_TYPES,
)
from ._batched_carriers import build_batched_carriers
from ._emit_f128 import emit_f128_rows
from ._emit_fixed_fp import emit_fixed_fp_rows
from ._emit_vc2 import emit_vc2_rows

if TYPE_CHECKING:
    from .._types import Stage2Batch


__all__ = ["build_number_idx_2d"]


# Block index of each NUMBER-block TokenType (0 = VC2, ..., 6 = F128).
_VC2_BLOCK = 0
_F128_BLOCK = len(_NUMBER_BLOCK_TOKEN_TYPES) - 1


def build_number_idx_2d(
    stage2_batch: "Stage2Batch",
    inline_bytes: np.ndarray,
    inline_byte_slices: list[slice],
) -> tuple[
    dict[TokenType, np.ndarray],
    dict[TokenType, list[slice]],
    np.ndarray,
    np.ndarray,
]:
    """Build per-:class:`TokenType` ``idx_2d`` arrays per ALG-7 + ALG-8.

    Identifies every surviving number-source carrier across the batch in
    one pass, emits each :class:`TokenType`'s ``idx_2d`` block as a
    vectorised meshgrid, and reconstructs the per-call_target chunk
    slices from per-call_target per-type ROW counts.

    Parameters
    ----------
    stage2_batch
        Per-call_target reads pull ``stage1.state`` (raw_tokens, masks,
        runlen_number) -- the byte-width / payload-length data lives
        in the ORIGINAL pre-promotion stream, while expansion identifies
        which positions are chunk-carriers.
    inline_bytes
        3a's flat ``u8`` buffer (index 0 = leading-zero pad). Not
        mutated here -- the rows we emit are gather offsets into it.
    inline_byte_slices
        Parallel to the DFS-flat call_target enumeration: entry ``i``
        is the range in ``inline_bytes`` owned by call_target ``i``.

    Returns
    -------
    idx_2d_per_type
        ``dict[TokenType, np.ndarray]`` with every NUMBER-block
        TokenType always present (empty types get zero-row arrays of
        the canonical row width).
    number_chunk_slices_per_type
        ``dict[TokenType, list[slice]]``; entry ``[T][i]`` is the
        slice into ``idx_2d_per_type[T]`` owned by call_target ``i``.
    f128_is_nan_or_inf
        ``bool[n_f128_sources]``. One entry per F128 SOURCE (not
        chunk); routes 3d's per-chunk dispatch (NaN/Inf path vs
        finite path) AND drives ``chunks_per_source = where(
        is_nan_or_inf, 1, 2)``. 3c always emits the full ALG-2 chunk
        set per finite source (2 chunks: LSB + MSB) so 3d can read
        ``actual_exp`` from the MSB limb. Stage 4's per-row sidecar
        concat drops the trailing invisible MSB chunk for a mid-cut
        finite source via the stream-walk's surviving-prefix count.
    vc2_chunk_exponent_sidecar
        ``u32[total_vc2_chunks]``. Per-chunk index within source
        (``0 = LSB``, ``K-1 = MSB``); stage 4 multiplies by 64 for
        ``exponent_base``.
    """

    n_total_cts = len(inline_byte_slices)

    carriers, ct_index = build_batched_carriers(
        stage2_batch, inline_byte_slices
    )

    # Per-type accumulators. Each entry holds the type's row block, the
    # per-CARRIER row count (carrier order), and the per-CARRIER owning
    # DFS-call_target index (for the slice reconstruction).
    idx_2d_per_type: dict[TokenType, np.ndarray] = {}
    rows_per_carrier_per_type: dict[TokenType, np.ndarray] = {}
    carrier_dfs_ct_per_type: dict[TokenType, np.ndarray] = {}

    block_idx = carriers.carrier_block_idx
    byte_offsets = carriers.carrier_byte_offsets
    # DFS-call_target index for each carrier (the slice axis is over the
    # FULL DFS enumeration, including dropped call_targets).
    carrier_dfs_ct = (
        ct_index[carriers.carrier_seg]
        if block_idx.shape[0]
        else np.empty(0, dtype=np.int64)
    )

    f128_is_nan_or_inf = np.empty(0, dtype=np.bool_)
    vc2_chunk_indices = np.empty(0, dtype=np.int64)

    for b, token_type in enumerate(_NUMBER_BLOCK_TOKEN_TYPES):
        type_mask = block_idx == b
        type_byte_offsets = byte_offsets[type_mask]
        type_dfs_ct = carrier_dfs_ct[type_mask]

        if b == _VC2_BLOCK:
            rows, rows_per_carrier, chunk_idx = emit_vc2_rows(
                p_carriers=carriers.carrier_raw_positions[type_mask],
                p_carrier_bytes=type_byte_offsets,
                expanded_positions=(
                    carriers.carrier_expanded_positions[type_mask]
                ),
                carrier_seg=carriers.carrier_seg[type_mask],
                seg_painted_offsets=carriers.seg_painted_offsets,
                seg_painted_vc2_flat=carriers.seg_painted_vc2_flat,
                seg_surviving=carriers.seg_surviving,
                seg_runlen_base=carriers.seg_runlen_base,
                runlen_number_flat=carriers.runlen_number_flat,
            )
            vc2_chunk_indices = chunk_idx
        elif b == _F128_BLOCK:
            rows, rows_per_carrier, is_nan_or_inf = emit_f128_rows(
                p_carrier_bytes=type_byte_offsets,
                expanded_positions=(
                    carriers.carrier_expanded_positions[type_mask]
                ),
                carrier_seg=carriers.carrier_seg[type_mask],
                seg_f128_base=carriers.seg_f128_base,
                f128_full_mask_flat=carriers.f128_full_mask_flat,
            )
            f128_is_nan_or_inf = is_nan_or_inf
        else:
            rows = emit_fixed_fp_rows(
                p_carrier_bytes=type_byte_offsets,
                token_type=token_type,
            )
            rows_per_carrier = np.ones(
                int(type_byte_offsets.shape[0]), dtype=np.int64
            )

        idx_2d_per_type[token_type] = rows
        rows_per_carrier_per_type[token_type] = rows_per_carrier
        carrier_dfs_ct_per_type[token_type] = type_dfs_ct

    # Ensure every type has a canonical-width empty array when no carrier
    # of that type appears.
    for token_type in _NUMBER_BLOCK_TOKEN_TYPES:
        if idx_2d_per_type[token_type].shape[0] == 0:
            idx_2d_per_type[token_type] = np.empty(
                (0, _FIXED_ROW_WIDTH[token_type]), dtype=np.uint32
            )

    chunk_slices_per_type = _reconstruct_per_ct_slices(
        n_total_cts=n_total_cts,
        rows_per_carrier_per_type=rows_per_carrier_per_type,
        carrier_dfs_ct_per_type=carrier_dfs_ct_per_type,
    )

    return (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_chunk_indices.astype(np.uint32, copy=False),
    )


def _reconstruct_per_ct_slices(
    *,
    n_total_cts: int,
    rows_per_carrier_per_type: dict[TokenType, np.ndarray],
    carrier_dfs_ct_per_type: dict[TokenType, np.ndarray],
) -> dict[TokenType, list[slice]]:
    """Rebuild per-DFS-call_target chunk slices from per-carrier ROW counts.

    The scalar entry produced ``chunk_slices_per_type[T][i] = slice(
    pre_counts[T], running_counts[T])`` -- the per-DFS-call_target run of
    type ``T``'s rows. Here we segment-sum each type's per-carrier row
    counts into per-DFS-call_target totals (``np.add.at`` over the
    carrier's owning DFS index), then ``cumsum`` to the abutting slice
    boundaries over the FULL DFS enumeration (dropped call_targets get a
    zero-length slice ``slice(c, c)`` at their cursor, exactly as the
    scalar walk emitted for a ``surviving_token_count == 0`` or
    no-carrier call_target).
    """
    chunk_slices_per_type: dict[TokenType, list[slice]] = {}
    for token_type in _NUMBER_BLOCK_TOKEN_TYPES:
        per_ct_rows = np.zeros(n_total_cts, dtype=np.int64)
        rows_per_carrier = rows_per_carrier_per_type[token_type]
        dfs_ct = carrier_dfs_ct_per_type[token_type]
        if rows_per_carrier.shape[0]:
            np.add.at(per_ct_rows, dfs_ct, rows_per_carrier)
        boundaries = np.zeros(n_total_cts + 1, dtype=np.int64)
        np.cumsum(per_ct_rows, out=boundaries[1:])
        slices: List[slice] = [
            slice(int(boundaries[i]), int(boundaries[i + 1]))
            for i in range(n_total_cts)
        ]
        chunk_slices_per_type[token_type] = slices
    return chunk_slices_per_type
