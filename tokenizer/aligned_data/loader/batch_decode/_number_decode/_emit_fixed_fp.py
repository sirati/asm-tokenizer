"""Batched per-carrier row emission for the fixed-width FP TokenTypes.

Single concern: emit one row per carrier for a F16 / BF16 / F32 / F64 /
F80 carrier population covering the contiguous byte range
``[p_carrier_byte, +width)``; 3d's view-cast turns each row into the
big-endian ``u16`` / ``u32`` / ``u64`` / 5x ``u16``-limb bit pattern
per ALG-7.

Batched: a per-TokenType population of N carriers becomes a single
``(N, width)`` meshgrid -- no per-source Python iteration.
"""

from __future__ import annotations

import numpy as np

from tokenizer.tokens import TokenType


__all__ = ["emit_fixed_fp_rows", "FIXED_PAYLOAD_BYTES"]


# Per-TokenType byte width of a single source's full payload (matches
# the ALG-7 widths). VC2's payload is variable-length so it doesn't
# belong here; F128 emits multiple 8-byte chunks via the F128 emitter.
FIXED_PAYLOAD_BYTES: dict[TokenType, int] = {
    TokenType.FLOAT16: 2,
    TokenType.BFLOAT16: 2,
    TokenType.FLOAT32: 4,
    TokenType.FLOAT64: 8,
    TokenType.FLOAT80: 10,
}


def emit_fixed_fp_rows(
    *,
    p_carrier_bytes: np.ndarray,
    token_type: TokenType,
) -> np.ndarray:
    """Emit one row per carrier for a fixed-width FP TokenType.

    Row ``i`` is the contiguous byte range ``[p_carrier_bytes[i],
    +width)`` where ``width = FIXED_PAYLOAD_BYTES[token_type]``. 3d's
    view-cast turns each row into the big-endian ``u16`` / ``u32`` /
    ``u64`` / 5x``u16``-limb bit pattern per ALG-7.

    Returns ``u32[n_carriers, width]`` -- a single meshgrid, no
    per-source Python iteration. Each fixed-width carrier emits exactly
    one row, so the per-call_target slice reconstruction counts these
    1:1 with carriers.
    """
    width = FIXED_PAYLOAD_BYTES[token_type]
    n_carriers = int(p_carrier_bytes.shape[0])
    if n_carriers == 0:
        return np.empty((0, width), dtype=np.uint32)
    starts = p_carrier_bytes.astype(np.uint32, copy=False)[:, np.newaxis]
    byte_idx = np.arange(width, dtype=np.uint32)[np.newaxis, :]
    return starts + byte_idx
