"""Stage 3c -- per-:class:`TokenType` number ``idx_2d`` construction.

Single concern: lay out 2D gather-offset arrays into ``inline_bytes``
for the number-arm view-cast step (ALG-7 byte layouts + ALG-8 VC2
multi-chunk packing). The vectorised FP normalisation (denormal +
NaN/Inf branches that turn the u64 bit patterns into the ``(significand,
sign_exp)`` f96 sidecar shape) is 3d's concern.

Byte layouts (ALG-7)
--------------------

* F16 / BF16 -- 1 row of 2 bytes.
* F32        -- 1 row of 4 bytes.
* F64        -- 1 row of 8 bytes.
* F80        -- 1 row of 10 bytes (3d reshapes into 5 big-endian u16
  limbs for the explicit-leading-bit reassembly path).
* F128       -- ``(chunk_count, 8)`` rows. Finite sources (ALG-2)
  emit 2 chunks: LSB limb (bytes 8..15) then MSB limb (bytes 0..7).
  NaN/Inf sources emit 1 chunk = MSB limb (bytes 0..7); 3d uses the
  ``f128_is_nan_or_inf`` sidecar to branch into the
  ``_encode_infnan(sign, mantissa_is_zero)`` path that needs only the
  high limb's sign + exponent + high-mantissa bits. The dataloader's
  ".rodata-robustness" policy (custom_float.py:336) sanctions this
  approximation -- canonical NaN/Inf is fully determined by the high 8
  bytes.
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
order (the same linearisation stage 1 + 3a use). Within each
call_target's surviving prefix we walk the expanded-stream positions
and emit per-source rows in stream order. The per-call-target
``inline_byte_slice`` produced by 3a anchors the offsets we emit.

ALG-8 reads ``L = state.runlen_number[p_carrier + 1]``, so we still
need the raw-stream carrier position. The expanded stream alone
doesn't tell us ``L`` (only ``K``); two distinct ``L`` values can map
to the same ``K``. We rebuild the expanded->raw position map locally
per call_target.

Plan refs: ``batch_decode_plan.md`` ALG-7 + ALG-8 + Stage 3 step 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tokenizer.tokens import TokenType

from ._band_constants import (
    _FIXED_ROW_WIDTH,
    _NUMBER_BLOCK_TOKEN_TYPES,
)
from ._per_call_target import _emit_call_target_rows

if TYPE_CHECKING:
    from .._types import Stage2Batch


__all__ = ["build_number_idx_2d"]


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

    Walks every call_target in DFS encounter order, iterates surviving
    expanded-stream positions, and emits ``u32`` gather-offset rows per
    surviving number-source carrier into the per-:class:`TokenType`
    ``idx_2d`` array.

    Parameters
    ----------
    stage2_batch
        Per-call-target reads pull ``stage1.state`` (raw_tokens, masks,
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
        chunk); NaN/Inf sources contribute 1 row, finite contribute 2.
    vc2_chunk_exponent_sidecar
        ``u32[total_vc2_chunks]``. Per-chunk index within source
        (``0 = LSB``, ``K-1 = MSB``); stage 4 multiplies by 64 for
        ``exponent_base``.
    """

    # Output accumulators. Every NUMBER-block TokenType key is always
    # present so callers can index without a KeyError guard.
    row_lists_per_type: dict[TokenType, list[np.ndarray]] = {
        T: [] for T in _NUMBER_BLOCK_TOKEN_TYPES
    }
    chunk_slices_per_type: dict[TokenType, list[slice]] = {
        T: [] for T in _NUMBER_BLOCK_TOKEN_TYPES
    }
    running_counts: dict[TokenType, int] = {
        T: 0 for T in _NUMBER_BLOCK_TOKEN_TYPES
    }
    f128_nan_or_inf_flags: list[bool] = []
    vc2_chunk_indices: list[int] = []

    # DFS encounter order matches 3a's ``inline_byte_slices`` so a
    # single positional index lines up byte-slice + row-slice handoffs.
    ct_dfs_idx = 0
    for section in stage2_batch.sections:
        for variant in section.variants:
            for ct in variant.call_targets:
                # Per-TokenType row counts BEFORE processing this ct, to
                # produce the per-ct slice into the final concatenation.
                pre_counts = {
                    T: running_counts[T] for T in _NUMBER_BLOCK_TOKEN_TYPES
                }

                if ct.surviving_token_count > 0:
                    _emit_call_target_rows(
                        ct=ct,
                        inline_byte_slice=inline_byte_slices[ct_dfs_idx],
                        row_lists_per_type=row_lists_per_type,
                        running_counts=running_counts,
                        f128_nan_or_inf_flags=f128_nan_or_inf_flags,
                        vc2_chunk_indices=vc2_chunk_indices,
                    )

                for T in _NUMBER_BLOCK_TOKEN_TYPES:
                    chunk_slices_per_type[T].append(
                        slice(pre_counts[T], running_counts[T])
                    )

                ct_dfs_idx += 1

    idx_2d_per_type: dict[TokenType, np.ndarray] = {}
    for T in _NUMBER_BLOCK_TOKEN_TYPES:
        if row_lists_per_type[T]:
            idx_2d_per_type[T] = np.concatenate(
                row_lists_per_type[T], axis=0
            )
        else:
            idx_2d_per_type[T] = np.empty(
                (0, _FIXED_ROW_WIDTH[T]), dtype=np.uint32
            )

    f128_is_nan_or_inf = np.asarray(f128_nan_or_inf_flags, dtype=np.bool_)
    vc2_chunk_exponent_sidecar = np.asarray(
        vc2_chunk_indices, dtype=np.uint32
    )

    return (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_chunk_exponent_sidecar,
    )
