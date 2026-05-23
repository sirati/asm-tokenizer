"""Per-call-target expanded-stream walk and per-source dispatch.

Single concern: walk one call_target's surviving expanded prefix,
recover the expanded->raw position map via the shared
:func:`tokenizer.aligned_data.loader.decoded._inline_decode_state.expanded_to_raw_position_map`
helper, and dispatch each surviving NUMBER-band carrier to the right
per-:class:`TokenType` emitter. The emitters are owned by ``_emit_*``
siblings -- this module only does the routing.

Carrier identification is a single vectorised boolean-mask gather
over ``expanded[:surviving]``; emission is dispatched by
:class:`TokenType` derived from each carrier's shifted vocab id and
runs as a batched meshgrid per TokenType (one block per VC2 source
population, one per fixed-width FP population). F128 emission still
iterates per carrier (its per-source helper is the F128-sidecar
plumbing concern); the outer dispatch only touches F128 carriers of
the current call_target, which is bounded by source count rather than
expanded-position count.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    expanded_to_raw_position_map,
)
from tokenizer.tokens import TokenType

from ._band_constants import (
    _NUMBER_BAND_HI_SHIFTED,
    _NUMBER_BAND_LO_SHIFTED,
    _NUMBER_BLOCK_TOKEN_TYPES,
)
from ._emit_f128 import _emit_f128_source
from ._emit_fixed_fp import _emit_fixed_fp_sources
from ._emit_vc2 import _emit_vc2_sources


__all__ = ["_emit_call_target_rows"]


def _emit_call_target_rows(
    *,
    ct,  # Stage2CallTarget -- forward-ref to avoid a runtime import cycle.
    inline_byte_slice: slice,
    row_lists_per_type: dict[TokenType, list[np.ndarray]],
    running_counts: dict[TokenType, int],
    f128_nan_or_inf_flags: list[bool],
    vc2_chunk_indices: list[int],
) -> None:
    """Walk one call_target's surviving expanded stream and emit rows.

    Carrier identification is a single vectorised boolean-mask gather
    over ``expanded[:surviving]``; emission is dispatched by
    :class:`TokenType` derived from each carrier's shifted vocab id
    and runs as a batched meshgrid per TokenType (one block per VC2
    source population, one per fixed-width FP population). F128
    emission still iterates per carrier (the per-source helper is the
    F128 sidecar plumbing concern); the outer dispatch only touches
    F128 carriers of the current call_target.
    """

    state = ct.stage1.state
    runlen_number = state.runlen_number

    extra_value_v2_mask = ct.extra_value_v2_mask
    extra_f128_mask = ct.extra_f128_mask
    expanded_token_ids = ct.expanded_token_ids
    surviving = int(ct.surviving_token_count)
    if surviving == 0:
        return

    # Recover the expanded->raw position map.  Painted VC2 / F128
    # continuation slots are NOT in ``state.real_mask`` but ARE in the
    # kept set from ``expand_tokens``'s strip step.  The shared helper
    # vectorises the prior per-call-target Python walk and reads only
    # ``state.real_mask`` plus the two extra masks.
    n_expanded_real = int(extra_value_v2_mask.shape[0]) - 1
    keep_raw_positions = expanded_to_raw_position_map(
        state=state,
        n_expanded_real=n_expanded_real,
        extra_value_v2_mask=extra_value_v2_mask,
        extra_f128_mask=extra_f128_mask,
    )

    # Exclusive-prefix inline-digit count per raw position lives on the
    # shared ``InlineDecodeState`` (``digit_cumsum[p + 1]`` = inline
    # byte count strictly before raw position ``p + 1``).  Reading the
    # per-stream cumsum here keeps the per-call-target Python work to
    # the outer dispatch only.
    inline_cumsum = state.digit_cumsum
    inline_slice_start = int(inline_byte_slice.start)

    # Vectorised carrier identification over expanded[:surviving]. A
    # position is a NUMBER-band carrier iff its shifted id is in
    # ``[1, 8)`` AND neither painted mask flags it as a continuation
    # slot. The prepend slot (expanded[0]) holds an IDENTITY-band id,
    # so the band predicate filters it out naturally.
    shifted_ids = expanded_token_ids[:surviving]
    is_painted = (
        extra_value_v2_mask[:surviving] | extra_f128_mask[:surviving]
    )
    in_number_band = (
        (shifted_ids >= _NUMBER_BAND_LO_SHIFTED)
        & (shifted_ids < _NUMBER_BAND_HI_SHIFTED)
    )
    carrier_mask = in_number_band & ~is_painted
    carrier_expanded_positions = np.nonzero(carrier_mask)[0].astype(np.int64)

    if carrier_expanded_positions.size == 0:
        return

    carrier_shifted_ids = shifted_ids[carrier_expanded_positions]
    # Block index 0 -> VC2, 1 -> F16, ..., 6 -> F128. Indexing into
    # ``_NUMBER_BLOCK_TOKEN_TYPES`` resolves each carrier's TokenType.
    carrier_block_idx = (
        carrier_shifted_ids - _NUMBER_BAND_LO_SHIFTED
    ).astype(np.int64)

    # p_carrier (raw-stream position) per carrier, then p_carrier_byte
    # (offset into ``inline_bytes``). ``expanded[0] = prepend`` has no
    # raw counterpart, so the lookup uses ``carrier_expanded_positions
    # - 1`` (carriers never appear at slot 0 because the prepend lives
    # in the IDENTITY band).
    carrier_raw_positions = keep_raw_positions[
        carrier_expanded_positions - 1
    ].astype(np.int64)
    carrier_byte_offsets = (
        inline_slice_start
        + inline_cumsum[carrier_raw_positions + 1].astype(np.int64)
    )

    # Per-block carrier groups. The within-group order is
    # stream-encounter order (np.nonzero returns ascending indices).
    # Each per-type idx_2d array's row order matches that stream
    # order; the f128 / vc2 sidecars are populated in the same per-
    # type stream order.
    for block_idx, token_type in enumerate(_NUMBER_BLOCK_TOKEN_TYPES):
        type_mask = carrier_block_idx == block_idx
        if not type_mask.any():
            continue
        type_expanded_positions = carrier_expanded_positions[type_mask]
        type_byte_offsets = carrier_byte_offsets[type_mask]

        if token_type is TokenType.VALUED_CONST_V2:
            type_raw_positions = carrier_raw_positions[type_mask]
            _emit_vc2_sources(
                state_runlen_number=runlen_number,
                p_carriers=type_raw_positions,
                p_carrier_bytes=type_byte_offsets,
                expanded_positions=type_expanded_positions,
                extra_value_v2_mask=extra_value_v2_mask,
                surviving=surviving,
                row_lists_per_type=row_lists_per_type,
                running_counts=running_counts,
                vc2_chunk_indices=vc2_chunk_indices,
            )
        elif token_type is TokenType.FLOAT128:
            # F128 emission is per-source (the per-source helper owns
            # the ALG-2 painted-continuation handling). The outer
            # dispatch is bounded by F128 source count rather than
            # expanded-position count.
            for ei_int, byte_int in zip(
                type_expanded_positions.tolist(),
                type_byte_offsets.tolist(),
            ):
                _emit_f128_source(
                    p_carrier_byte=int(byte_int),
                    expanded_idx=int(ei_int),
                    extra_f128_mask=extra_f128_mask,
                    surviving=surviving,
                    row_lists_per_type=row_lists_per_type,
                    running_counts=running_counts,
                    f128_nan_or_inf_flags=f128_nan_or_inf_flags,
                )
        else:
            # Fixed-width FP types (F16 / BF16 / F32 / F64 / F80) emit
            # exactly 1 row of ``width`` bytes per carrier; build the
            # whole per-type block as a single (n_carriers, width)
            # meshgrid.
            _emit_fixed_fp_sources(
                p_carrier_bytes=type_byte_offsets,
                token_type=token_type,
                row_lists_per_type=row_lists_per_type,
                running_counts=running_counts,
            )
