"""Per-source row emission for the fixed-width FP TokenTypes.

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


__all__ = ["_emit_fixed_fp_sources"]


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


def _emit_fixed_fp_sources(
    *,
    p_carrier_bytes: np.ndarray,
    token_type: TokenType,
    row_lists_per_type: dict[TokenType, list[np.ndarray]],
    running_counts: dict[TokenType, int],
) -> None:
    """Emit one row per carrier for a fixed-width FP TokenType.

    Row ``i`` is the contiguous byte range ``[p_carrier_bytes[i],
    +width)`` where ``width = _FIXED_PAYLOAD_BYTES[token_type]``. 3d's
    view-cast turns each row into the big-endian ``u16`` / ``u32`` /
    ``u64`` / 5x``u16``-limb bit pattern per ALG-7.

    All rows are produced as a single meshgrid -- no per-source Python
    iteration.
    """
    width = _FIXED_PAYLOAD_BYTES[token_type]
    n_carriers = int(p_carrier_bytes.shape[0])
    if n_carriers == 0:
        return
    starts = p_carrier_bytes.astype(np.uint32, copy=False)[:, np.newaxis]
    byte_idx = np.arange(width, dtype=np.uint32)[np.newaxis, :]
    rows = starts + byte_idx
    row_lists_per_type[token_type].append(rows)
    running_counts[token_type] += n_carriers
