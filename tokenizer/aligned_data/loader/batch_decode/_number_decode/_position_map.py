"""Cross-call-target raw/expanded position helpers for 3c.

Single concern: turn ``state.real_mask`` + the ``extra_*_mask`` flags
into the per-call-target expanded->raw position map and the inline-
digit cumulative-count helper. Both are pure-functional and free of
any per-emitter knowledge -- the per-emitter callers consume the arrays
they return.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "_expanded_to_raw_position_map",
    "_build_inline_cumsum",
]


def _expanded_to_raw_position_map(
    *,
    state,
    extra_value_v2_mask: np.ndarray,
    extra_f128_mask: np.ndarray,
) -> np.ndarray:
    """Recover the raw-stream position for each expanded[1:] slot.

    Painted VC2 / F128 continuation slots in 2a are contiguous in
    raw-space immediately after their carrier. We walk
    ``state.real_mask``'s nonzero positions for carriers (and other
    non-promoted real tokens); when the extra_*_mask flags a painted
    continuation, the painted slot's raw position is the prior slot's
    raw position + 1.

    Returns ``u32[predicted_full_length - 1]`` (i.e. one entry per
    expanded[1:] slot; expanded[0] = synthetic prepend has no raw
    counterpart).
    """
    real_positions = np.nonzero(state.real_mask)[0].astype(np.uint32)
    n_expanded_real = int(extra_value_v2_mask.shape[0]) - 1  # subtract prepend

    if n_expanded_real == 0:
        return np.empty(0, dtype=np.uint32)

    out = np.empty(n_expanded_real, dtype=np.uint32)

    # ``real_idx`` cursors over real_positions (raw-stream carriers +
    # other non-painted real tokens). When an extra_*_mask is True at
    # the current expanded slot, the painted continuation's raw
    # position = prior expanded slot's raw position + 1 (painted slots
    # are contiguous after their carrier).
    real_idx = 0
    for expanded_real_idx in range(n_expanded_real):
        is_extra = bool(
            extra_value_v2_mask[expanded_real_idx + 1]
            | extra_f128_mask[expanded_real_idx + 1]
        )
        if is_extra:
            out[expanded_real_idx] = out[expanded_real_idx - 1] + 1
        else:
            out[expanded_real_idx] = real_positions[real_idx]
            real_idx += 1

    return out


def _build_inline_cumsum(number_mask: np.ndarray) -> np.ndarray:
    """Cumulative inline-digit count: ``cumsum[p] = #digits in raw[0..p)``.

    Length ``len(number_mask) + 1``. The caller reads ``cumsum[p + 1]``
    to get the count of inline-digit bytes preceding raw position
    ``p + 1`` -- which is exactly the offset of the inline-digit byte
    at ``p + 1`` within the call_target's slice of ``inline_bytes``
    (3a guarantees no per-byte cut inside the slice; chunk-granularity
    cuts are handled by the caller).
    """
    n = int(number_mask.shape[0])
    cumsum = np.empty(n + 1, dtype=np.uint32)
    cumsum[0] = 0
    if n > 0:
        np.cumsum(number_mask.astype(np.uint32), out=cumsum[1:])
    return cumsum
