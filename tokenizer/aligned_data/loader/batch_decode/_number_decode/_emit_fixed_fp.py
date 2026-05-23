"""Per-source row emission for the fixed-width FP TokenTypes.

Single concern: emit exactly one row for a F16 / BF16 / F32 / F64 / F80
source covering the contiguous byte range ``[p_carrier_byte, +width)``;
3d's view-cast turns that into the big-endian ``u16`` / ``u32`` / ``u64``
/ 5x ``u16``-limb bit pattern per ALG-7.
"""

from __future__ import annotations

import numpy as np

from tokenizer.tokens import TokenType


__all__ = ["_emit_fixed_fp_source"]


# Per-TokenType byte width of a single source's full payload (matches
# the ALG-7 widths). VC2's payload is variable-length so it doesn't
# belong here; F128 emits multiple 8-byte chunks via the F128 emitter.
_FIXED_PAYLOAD_BYTES: dict[TokenType, int] = {
    TokenType.FLOAT16: 2,
    TokenType.BFLOAT16: 2,
    TokenType.FLOAT32: 4,
    TokenType.FLOAT64: 8,
    TokenType.FLOAT80: 10,
}


def _emit_fixed_fp_source(
    *,
    p_carrier_byte: int,
    token_type: TokenType,
    row_lists_per_type: dict[TokenType, list[np.ndarray]],
    running_counts: dict[TokenType, int],
) -> None:
    """Emit 1 row for a fixed-width FP source (F16/BF16/F32/F64/F80).

    Row is the contiguous byte range ``[p_carrier_byte, +width)``;
    3d's view-cast turns that into the big-endian ``u16`` / ``u32`` /
    ``u64`` / 5x``u16``-limb bit pattern per ALG-7.
    """
    width = _FIXED_PAYLOAD_BYTES[token_type]
    row = np.arange(
        p_carrier_byte, p_carrier_byte + width, dtype=np.uint32
    )[np.newaxis, :]
    row_lists_per_type[token_type].append(row)
    running_counts[token_type] += 1
