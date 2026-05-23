"""Per-source row emission for the VC2 (``valued_const_v2``) TokenType.

Single concern: emit ``K_visible`` rows for one VC2 source per ALG-8.
The MSB chunk may have fewer than 8 payload bytes when ``L % 8 != 0``;
padding slots reference ``inline_bytes[0]`` (3a's leading-zero pad).
"""

from __future__ import annotations

import numpy as np

from tokenizer.tokens import TokenType


__all__ = ["_emit_vc2_source"]


def _emit_vc2_source(
    *,
    state_runlen_number: np.ndarray,
    p_carrier: int,
    p_carrier_byte: int,
    expanded_idx: int,
    expanded_token_ids: np.ndarray,
    extra_value_v2_mask: np.ndarray,
    surviving: int,
    row_lists_per_type: dict[TokenType, list[np.ndarray]],
    running_counts: dict[TokenType, int],
    vc2_chunk_indices: list[int],
) -> int:
    """Emit ``K_visible`` rows for one VC2 source per ALG-8.

    See :mod:`._entry` module docstring (verbatim ALG-8 block) for the
    per-chunk byte-range formula. ``K_visible`` = 1 carrier + count of
    consecutive ``extra_value_v2_mask`` True positions immediately
    following, capped by both the surviving prefix and ``K_full``.

    Short MSB chunks left-pad with ``inline_bytes[0]`` references
    (zeros): for ``L=17`` the MSB chunk yields ``[0]*7 + [p_carrier_byte]``.

    Returns the number of expanded positions consumed (``K_visible``).
    """
    # ALG-8: ``L = state.runlen_number[p_carrier + 1]``. The carrier
    # always has a p+1 slot per _promote_vc2's tail assertion.
    L = int(state_runlen_number[p_carrier + 1])
    K_full = max(1, (L + 7) // 8)

    K_visible = 1
    while (
        K_visible < K_full
        and expanded_idx + K_visible < surviving
        and bool(extra_value_v2_mask[expanded_idx + K_visible])
    ):
        K_visible += 1

    # ALG-8 per-chunk byte range: ``[p_carrier_byte + L - 8*(c+1),
    # p_carrier_byte + L - 8*c)`` intersected with the payload region.
    # The intersection only matters for the MSB chunk (``c == K_full -
    # 1``, ``L % 8 != 0``); leading slots reference inline_bytes[0].
    for c in range(K_visible):
        unclipped_start = p_carrier_byte + L - 8 * (c + 1)
        unclipped_end = p_carrier_byte + L - 8 * c
        clipped_start = max(unclipped_start, p_carrier_byte)
        clipped_end = unclipped_end  # always <= p_carrier_byte + L
        n_actual_bytes = clipped_end - clipped_start
        n_pad_bytes = 8 - n_actual_bytes

        row = np.empty(8, dtype=np.uint32)
        if n_pad_bytes > 0:
            row[:n_pad_bytes] = 0
        if n_actual_bytes > 0:
            row[n_pad_bytes:] = np.arange(
                clipped_start, clipped_end, dtype=np.uint32
            )

        row_lists_per_type[TokenType.VALUED_CONST_V2].append(
            row[np.newaxis, :]
        )
        running_counts[TokenType.VALUED_CONST_V2] += 1
        vc2_chunk_indices.append(c)

    return K_visible
