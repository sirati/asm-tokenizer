"""Per-source row emission for the F128 TokenType.

Single concern: emit 1 or 2 rows for one F128 source per ALG-2 + ALG-7.

* Finite source (ALG-2 painted continuation present in the FULL
  expanded mask): 2 chunks (LSB then MSB limb) -- INDEPENDENT of the
  per-row cutoff. The MSB chunk's bytes are required for 3d's
  ``actual_exp`` extraction (LSB chunk's exponent base =
  actual_exp - 112), so we MUST emit it even when the painted MSB
  slot is past ``partial_cut_length``.
* NaN/Inf source (no continuation): 1 chunk = MSB limb (bytes 0..7);
  3d branches on ``f128_is_nan_or_inf`` to call ``_encode_infnan``.

Stage 4's per-row sidecar concat walks the surviving expanded prefix
and naturally drops the trailing invisible MSB chunk for a mid-cut
finite source: 3c emits chunks in stream emission order (LSB then MSB
per finite source), so the stream-visible F128 count slices the per-CT
FLOAT128 chunk array at exactly the right tail. The mid-cut F128 is
by construction the LAST F128 source in the last surviving CT, so its
MSB chunk lives at the END of that CT's per-type slice.

The emitter populates one side output in lockstep with each source it
visits:

* ``f128_nan_or_inf_flags`` -- per-source NaN/Inf flag from the ALG-2
  painted-continuation signal. Used by 3d to route per-chunk dispatch
  (NaN/Inf path vs finite path) AND to drive ``chunks_per_source =
  where(is_nan_or_inf, 1, 2)``.
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
    """Emit ALG-2 chunks for one F128 source.

    Finite source (ALG-2 painted continuation present in the FULL
    expanded mask): 2 chunks -- LSB limb (bytes 8..15) then MSB limb
    (bytes 0..7). NaN/Inf source (no painted continuation): 1 chunk =
    MSB limb (bytes 0..7); 3d branches on ``f128_is_nan_or_inf`` to
    call ``_encode_infnan``.

    Chunk emission is INDEPENDENT of the per-row cutoff. ALG-7's per-
    chunk normalization (``_fp_normalize._f128``) reads
    ``actual_exp = biased_exp - bias`` from the MSB limb to derive the
    LSB chunk's exponent base (chunk0_base = actual_exp - 112) -- so
    even when the MSB chunk's stream slot is past ``partial_cut_length``
    the MSB BYTES still have to reach 3d. Stage 4's per-row sidecar
    concat drops the trailing invisible MSB chunk via the stream-walk's
    surviving-prefix count.

    Returns the number of expanded positions consumed (1 or 2).
    """
    # ALG-2's painted bit at ``expanded_idx + 1`` is the authoritative
    # finite/NaN-Inf signal -- read against the FULL mask, NOT clipped
    # to ``surviving``, so a mid-cut finite source still reports
    # is_finite=True and contributes both chunks to 3c's output.
    is_finite_source = (
        expanded_idx + 1 < extra_f128_mask.shape[0]
        and bool(extra_f128_mask[expanded_idx + 1])
    )

    # has_continuation = the painted MSB slot is within the surviving
    # prefix. Drives only how many expanded positions this call
    # advances (1 if the cut hides the painted slot, 2 otherwise);
    # row emission is independent of it.
    has_continuation = (
        is_finite_source
        and expanded_idx + 1 < surviving
    )

    if is_finite_source:
        f128_nan_or_inf_flags.append(False)
        row_lsb = np.arange(
            p_carrier_byte + 8, p_carrier_byte + 16, dtype=np.uint32
        )[np.newaxis, :]
        row_msb = np.arange(
            p_carrier_byte, p_carrier_byte + 8, dtype=np.uint32
        )[np.newaxis, :]
        row_lists_per_type[TokenType.FLOAT128].append(row_lsb)
        row_lists_per_type[TokenType.FLOAT128].append(row_msb)
        running_counts[TokenType.FLOAT128] += 2
        # Only one expanded slot consumed when the painted MSB is past
        # the cut; outer walk's surviving-stream traversal handles the
        # stream-visibility of the MSB chunk.
        return 2 if has_continuation else 1

    # NaN/Inf source: 1 row (MSB limb, bytes 0..7).
    f128_nan_or_inf_flags.append(True)
    row_msb = np.arange(
        p_carrier_byte, p_carrier_byte + 8, dtype=np.uint32
    )[np.newaxis, :]
    row_lists_per_type[TokenType.FLOAT128].append(row_msb)
    running_counts[TokenType.FLOAT128] += 1
    return 1
