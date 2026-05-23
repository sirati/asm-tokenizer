"""Per-source row emission for the F128 TokenType.

Single concern: emit 1 or 2 rows for one F128 source per ALG-2 + ALG-7.

* Finite source (ALG-2 painted continuation): 2 chunks (LSB then MSB
  limb).
* NaN/Inf source (no continuation): 1 chunk = MSB limb (bytes 0..7);
  3d branches on ``f128_is_nan_or_inf`` to call ``_encode_infnan``.

The emitter populates ``f128_nan_or_inf_flags`` in lockstep with each
source it visits: 3d uses the per-source flag to route per-chunk
dispatch (NaN/Inf path vs finite path).
"""

from __future__ import annotations

import numpy as np

from tokenizer.tokens import TokenType


__all__ = ["_emit_f128_source"]


def _emit_f128_source(
    *,
    p_carrier_byte: int,
    expanded_idx: int,
    extra_f128_mask: np.ndarray,
    surviving: int,
    row_lists_per_type: dict[TokenType, list[np.ndarray]],
    running_counts: dict[TokenType, int],
    f128_nan_or_inf_flags: list[bool],
) -> int:
    """Emit 1 or 2 rows for one F128 source.

    Finite source (ALG-2 painted continuation): 2 chunks -- LSB limb
    (bytes 8..15) then MSB limb (bytes 0..7). NaN/Inf source (no
    continuation): 1 chunk = MSB limb (bytes 0..7); 3d branches on
    ``f128_is_nan_or_inf`` to call ``_encode_infnan``.

    Mid-cut: if a finite source's chunk-1 slot is past the cut, emit
    only chunk 0 (LSB). The ``f128_is_nan_or_inf`` flag stays False
    (the source's nature is from ALG-2, not from how many chunks
    survived).

    Returns the number of expanded positions consumed (1 or 2).
    """
    # ALG-2's painted bit at ``expanded_idx + 1`` is the authoritative
    # finite/NaN-Inf signal -- read against the FULL mask, NOT clipped
    # to ``surviving``, so a mid-cut finite source still reports
    # is_finite=True. ``has_continuation`` separately gates whether
    # we ALSO emit chunk 1.
    has_continuation = (
        expanded_idx + 1 < surviving
        and bool(extra_f128_mask[expanded_idx + 1])
    )
    is_finite_source = (
        expanded_idx + 1 < extra_f128_mask.shape[0]
        and bool(extra_f128_mask[expanded_idx + 1])
    )

    if is_finite_source:
        f128_nan_or_inf_flags.append(False)
        row_lsb = np.arange(
            p_carrier_byte + 8, p_carrier_byte + 16, dtype=np.uint32
        )[np.newaxis, :]
        row_lists_per_type[TokenType.FLOAT128].append(row_lsb)
        running_counts[TokenType.FLOAT128] += 1

        if has_continuation:
            row_msb = np.arange(
                p_carrier_byte, p_carrier_byte + 8, dtype=np.uint32
            )[np.newaxis, :]
            row_lists_per_type[TokenType.FLOAT128].append(row_msb)
            running_counts[TokenType.FLOAT128] += 1
            return 2
        return 1

    # NaN/Inf source: 1 row (MSB limb, bytes 0..7).
    f128_nan_or_inf_flags.append(True)
    row_msb = np.arange(
        p_carrier_byte, p_carrier_byte + 8, dtype=np.uint32
    )[np.newaxis, :]
    row_lists_per_type[TokenType.FLOAT128].append(row_msb)
    running_counts[TokenType.FLOAT128] += 1
    return 1
