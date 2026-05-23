"""Per-call-target expanded-stream walk and per-source dispatch.

Single concern: walk one call_target's surviving expanded prefix,
recover the expanded->raw position map via the shared
:func:`tokenizer.aligned_data.loader.decoded._inline_decode_state.expanded_to_raw_position_map`
helper, and dispatch each surviving NUMBER-band carrier to the right
per-:class:`TokenType` emitter. The emitters are owned by ``_emit_*``
siblings -- this module only does the routing.
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
from ._emit_fixed_fp import _emit_fixed_fp_source
from ._emit_vc2 import _emit_vc2_source


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

    Per-source emission is dispatched by :class:`TokenType` derived
    from the carrier's shifted vocab id. The helper mutates the
    accumulators in place.

    Per-source row construction uses ``np.arange`` over byte offsets;
    the only Python-level loop is the per-source dispatch. Per
    call_target the source count is small (single-digit to low
    hundreds), so the hot path is 3d's cross-source vectorisation, NOT
    this layout step.
    """

    state = ct.stage1.state
    runlen_number = state.runlen_number

    # Recover the expanded->raw position map.  Painted VC2 / F128
    # continuation slots are NOT in ``state.real_mask`` but ARE in the
    # kept set from ``expand_tokens``'s strip step.  The shared helper
    # vectorises the prior per-call-target Python walk and reads only
    # ``state.real_mask`` plus the two extra masks.
    n_expanded_real = int(ct.extra_value_v2_mask.shape[0]) - 1
    keep_raw_positions = expanded_to_raw_position_map(
        state=state,
        n_expanded_real=n_expanded_real,
        extra_value_v2_mask=ct.extra_value_v2_mask,
        extra_f128_mask=ct.extra_f128_mask,
    )

    # Exclusive-prefix inline-digit count per raw position lives on the
    # shared ``InlineDecodeState`` (``digit_cumsum[p + 1]`` = inline
    # byte count strictly before raw position ``p + 1``).  Reading the
    # per-stream cumsum here keeps the per-call-target Python work to
    # the carrier-emission loop only.
    inline_cumsum = state.digit_cumsum

    inline_slice_start = int(inline_byte_slice.start)

    expanded_token_ids = ct.expanded_token_ids
    extra_value_v2_mask = ct.extra_value_v2_mask
    extra_f128_mask = ct.extra_f128_mask
    surviving = int(ct.surviving_token_count)

    # The prepend slot (expanded[0]) holds an IDENTITY-band id; the
    # NUMBER-band predicate below filters it out naturally.
    expanded_idx = 0
    while expanded_idx < surviving:
        tok_id = int(expanded_token_ids[expanded_idx])
        is_number_carrier = (
            _NUMBER_BAND_LO_SHIFTED <= tok_id < _NUMBER_BAND_HI_SHIFTED
            and not bool(extra_value_v2_mask[expanded_idx])
            and not bool(extra_f128_mask[expanded_idx])
        )
        if not is_number_carrier:
            expanded_idx += 1
            continue

        token_type = _NUMBER_BLOCK_TOKEN_TYPES[
            tok_id - _NUMBER_BAND_LO_SHIFTED
        ]
        # expanded[0] = prepend (no raw counterpart); subtract 1 for
        # the raw-position lookup.
        p_carrier = int(keep_raw_positions[expanded_idx - 1])
        # First inline-digit byte of this source = cumulative
        # inline-digit count in raw[0..p_carrier], shifted by the
        # call_target's slice start.
        p_carrier_byte = inline_slice_start + int(inline_cumsum[p_carrier + 1])

        if token_type is TokenType.VALUED_CONST_V2:
            chunks_consumed = _emit_vc2_source(
                state_runlen_number=runlen_number,
                p_carrier=p_carrier,
                p_carrier_byte=p_carrier_byte,
                expanded_idx=expanded_idx,
                expanded_token_ids=expanded_token_ids,
                extra_value_v2_mask=extra_value_v2_mask,
                surviving=surviving,
                row_lists_per_type=row_lists_per_type,
                running_counts=running_counts,
                vc2_chunk_indices=vc2_chunk_indices,
            )
        elif token_type is TokenType.FLOAT128:
            chunks_consumed = _emit_f128_source(
                p_carrier_byte=p_carrier_byte,
                expanded_idx=expanded_idx,
                extra_f128_mask=extra_f128_mask,
                surviving=surviving,
                row_lists_per_type=row_lists_per_type,
                running_counts=running_counts,
                f128_nan_or_inf_flags=f128_nan_or_inf_flags,
            )
        else:
            # Fixed-width FP types (F16 / BF16 / F32 / F64 / F80).
            _emit_fixed_fp_source(
                p_carrier_byte=p_carrier_byte,
                token_type=token_type,
                row_lists_per_type=row_lists_per_type,
                running_counts=running_counts,
            )
            chunks_consumed = 1

        expanded_idx += chunks_consumed
