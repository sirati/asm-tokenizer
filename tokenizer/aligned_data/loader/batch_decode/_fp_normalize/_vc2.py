"""Vectorized VC2 (``valued_const_v2``) integer multi-chunk encoder.

Single concern: 8-byte big-endian payload per chunk -> f96-shape
normalization for integer magnitude limbs. Per source the chunk count is
``ceil(byte_length / 8)`` (from stage-2 sidecar); each chunk carries
``chunk_index_within_source`` in the per-chunk
:attr:`Stage3Batch.vc2_chunk_exponent_sidecar`.

Matches :func:`custom_float.from_int` / ``_split_to_chunks(value, sign,
base_exponent_unbiased=0)``: per-chunk ``chunk_exponent_base =
chunk_index_within_source * 64`` (since the int path's
``base_exponent_unbiased == 0``); zero chunks emit ``pack_sign_exp(sign,
chunk_exponent_base)`` (canonical signed zero at the chunk's stride
exponent).

Sign handling: VC2 chunks of the same source share a single sign sourced
from the postfix ``value_negative`` token (carried by stage 2's
``is_negative_per_position``). This module expands the caller-supplied
per-source array to per-chunk via :func:`vc2_per_chunk_sign` -- source
boundaries are recoverable from ``chunk_exponent_sidecar`` (a 0 entry
starts a new source).
"""

from __future__ import annotations

import numpy as np

from ._primitives import emit_chunk_vec

__all__ = ["normalize_vc2", "vc2_per_chunk_sign"]


def vc2_per_chunk_sign(
    chunk_exponent_sidecar: np.ndarray,
    is_negative_per_source: np.ndarray,
) -> np.ndarray:
    """Expand a per-source sign array to per-chunk for VC2.

    Source boundaries are recoverable from ``chunk_exponent_sidecar``: a
    chunk with index 0 starts a new source; chunks are emitted low-chunk
    first within each source. The cumulative count of zeros up to (and
    including) position k gives the source index for chunk k.

    Raises if the recovered source indices fall outside
    ``is_negative_per_source`` -- this catches stage-2/stage-3 drift.
    """
    n_chunks = chunk_exponent_sidecar.shape[0]
    if n_chunks == 0:
        return np.zeros(0, dtype=bool)
    is_source_start = chunk_exponent_sidecar == np.uint32(0)
    source_idx_per_chunk = np.cumsum(is_source_start.astype(np.int64)) - 1
    if (source_idx_per_chunk < 0).any() or (
        source_idx_per_chunk >= is_negative_per_source.shape[0]
    ).any():
        raise AssertionError(
            "VC2 chunk_exponent_sidecar inconsistent with "
            "is_negative_per_source length"
        )
    return is_negative_per_source[source_idx_per_chunk].astype(bool)


def normalize_vc2(
    chunk_u64: np.ndarray,
    chunk_exponent_sidecar: np.ndarray,
    is_negative_per_chunk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized VC2 encoder.

    Inputs:
      * ``chunk_u64``: ``u64[n_chunks]`` -- per-chunk integer magnitude
        limbs in stream-position order.
      * ``chunk_exponent_sidecar``: ``u32[n_chunks]`` --
        ``chunk_index_within_source`` for each chunk (0 = LSB chunk of its
        source).
      * ``is_negative_per_chunk``: ``bool[n_chunks]`` -- sign of the SOURCE
        that owns each chunk (all chunks of the same source share sign).
        Use :func:`vc2_per_chunk_sign` to expand a per-source array.

    Returns ``(significand: u64[n_chunks], sign_exp: u32[n_chunks])``.
    """
    chunk_exponent_base = (
        chunk_exponent_sidecar.astype(np.int64) * np.int64(64)
    )
    return emit_chunk_vec(chunk_u64, is_negative_per_chunk, chunk_exponent_base)
