"""Per-TokenType dispatch for the FP normalizer.

Single concern: gather payload bytes from ``inline_bytes`` via the per-type
``idx_2d`` indexer, view-cast to the type's bit-pattern dtype, and dispatch
to the right per-type normalizer kernel.

The dispatch table covers every number-band :class:`TokenType`
(``VALUED_CONST_V2`` + ``FLOAT16`` / ``BFLOAT16`` / ``FLOAT32`` /
``FLOAT64`` / ``FLOAT80`` / ``FLOAT128``); anything else raises -- identity-
band tokens MUST NOT reach this layer.
"""

from __future__ import annotations

import numpy as np

from tokenizer.tokens import TokenType

from ._f80 import normalize_f80
from ._f128 import normalize_f128
from ._ieee_narrow import normalize_ieee_narrow
from ._vc2 import normalize_vc2, vc2_per_chunk_sign

__all__ = ["normalize_per_token_type"]


def normalize_per_token_type(
    idx_2d_per_type: dict[TokenType, np.ndarray],
    inline_bytes: np.ndarray,
    f128_is_nan_or_inf: np.ndarray,
    vc2_chunk_exponent_sidecar: np.ndarray,
    is_negative_per_source_per_type: dict[TokenType, np.ndarray],
) -> dict[TokenType, tuple[np.ndarray, np.ndarray]]:
    """For each TokenType produce ``(significand: u64, sign_exp: u32)``
    arrays aligned 1:1 with the per-type chunks emitted in stream-position
    order.

    Inputs:

    * ``idx_2d_per_type[T]``: ``u32[n_T_payload_units,
      payload_size_T]`` -- 2D indexer into ``inline_bytes``.
      ``payload_size_T`` per type: F16=2, BF16=2, F32=4, F64=8, F80=10,
      F128=8, VC2=8.
        - F16, BF16, F32, F64, F80: one row per source.
        - F128: one row per CHUNK (2 rows per finite source -- LSB
          then MSB limb -- and 1 row per NaN/Inf source = MSB limb).
          ``f128_is_nan_or_inf`` carries the per-source role; the F128
          normalizer rebuilds the per-source mapping from it.
        - VC2: one row per CHUNK (multi-chunk sources contribute K rows).
    * ``inline_bytes``: u8 array. The 2D indexers gather bytes from here.
    * ``f128_is_nan_or_inf``: ``bool[n_f128_sources]``; one entry per
      F128 SOURCE (not per chunk). Drives the per-source 1-vs-2 chunk
      split when interpreting the F128 idx_2d rows.
    * ``vc2_chunk_exponent_sidecar``: ``u32[n_vc2_chunks]``; per-chunk
      ``chunk_index_within_source``.
    * ``is_negative_per_source_per_type[T]``: ``bool[n_T_sources]``; per
      source in stream-position order. For VC2, expanded internally to
      per-chunk via :func:`vc2_per_chunk_sign`. For IEEE bit-pattern types
      (F16 / BF16 / F32 / F64 / F80 / F128) sign lives in the bit pattern
      -- the parameter is accepted for API uniformity but not consulted
      for those types.

    Output: ``dict[TokenType] -> (significand: u64[n_chunks_of_T],
    sign_exp: u32[n_chunks_of_T])``. TokenTypes with empty input idx_2d
    produce empty arrays.

    Raises ``ValueError`` for non-NUMBER-band TokenTypes.
    """
    out: dict[TokenType, tuple[np.ndarray, np.ndarray]] = {}

    for token_type, idx_2d in idx_2d_per_type.items():
        if idx_2d.shape[0] == 0:
            out[token_type] = (
                np.zeros(0, dtype=np.uint64),
                np.zeros(0, dtype=np.uint32),
            )
            continue

        # Gather payload bytes; result is C-contiguous u8 array shape
        # [n_rows, payload_size].
        gathered = inline_bytes[idx_2d]

        if token_type is TokenType.FLOAT16:
            bits = gathered.view(">u2").reshape(-1).astype(np.uint16)
            out[token_type] = normalize_ieee_narrow(
                bits, mantissa_bits=10, exponent_bits=5, bias=15
            )
        elif token_type is TokenType.BFLOAT16:
            bits = gathered.view(">u2").reshape(-1).astype(np.uint16)
            out[token_type] = normalize_ieee_narrow(
                bits, mantissa_bits=7, exponent_bits=8, bias=127
            )
        elif token_type is TokenType.FLOAT32:
            bits = gathered.view(">u4").reshape(-1).astype(np.uint32)
            out[token_type] = normalize_ieee_narrow(
                bits, mantissa_bits=23, exponent_bits=8, bias=127
            )
        elif token_type is TokenType.FLOAT64:
            bits = gathered.view(">u8").reshape(-1).astype(np.uint64)
            out[token_type] = normalize_ieee_narrow(
                bits, mantissa_bits=52, exponent_bits=11, bias=1023
            )
        elif token_type is TokenType.FLOAT80:
            out[token_type] = normalize_f80(gathered)
        elif token_type is TokenType.FLOAT128:
            out[token_type] = normalize_f128(gathered, f128_is_nan_or_inf)
        elif token_type is TokenType.VALUED_CONST_V2:
            chunk_u64 = gathered.view(">u8").reshape(-1).astype(np.uint64)
            is_negative_per_chunk = vc2_per_chunk_sign(
                vc2_chunk_exponent_sidecar,
                is_negative_per_source_per_type[token_type],
            )
            out[token_type] = normalize_vc2(
                chunk_u64,
                vc2_chunk_exponent_sidecar,
                is_negative_per_chunk,
            )
        else:
            raise ValueError(
                f"normalize_per_token_type does not handle {token_type!r}; "
                "only NUMBER-band TokenTypes are valid here."
            )

    return out
