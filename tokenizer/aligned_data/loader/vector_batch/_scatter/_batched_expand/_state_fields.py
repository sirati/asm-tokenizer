"""Boundary-aware InlineDecodeState field math over the flat raw stream.

Single concern: reproduce the per-node
:class:`...decoded._inline_decode_state.InlineDecodeState` fields (run
lengths, the digit cumsum, the is-negative flag) as boundary-aware
vectorized passes so each per-node SLICE equals the scalar
:func:`build_inline_decode_state` on that node's raw stream. The
raw-stream rewrite that consumes these lives in :mod:`._rewrite`; the
orchestration in :mod:`._expansion`.

The MATH is performed by the GIL-released
:func:`build_inline_state_fields_kernel`: a single per-node CSR scalar
walk over the flat raw stream fusing the three numpy passes
(``runlen_number`` + ``runlen_value`` via per-node ``run_lengths``, the
packed per-node ``digit_cumsum``, and ``is_negative_per_position``). The
kernel derives the inline-band / real / value / carrier masks INSIDE its
GIL release from ``raw`` + the three unified-vocab constants, so this
module passes only ``raw`` + the per-node CSR; the digit-cumsum packed
CSR layout (size ``total_raw + n_nodes``, node ``i``'s ``(count_i + 1)``
block at ``rec_starts[i] + i``) is unchanged.
"""

from __future__ import annotations

import numpy as np
from dedup_hashmap import build_inline_state_fields_kernel

from ._constants import (
    _V2_EAGER_BLOCK_END,
    _V2_RESERVED_DIGIT_COUNT,
    _V2_VALUE_NEGATIVE_TOKEN_ID,
)


__all__ = ["build_inline_state_fields"]


def build_inline_state_fields(
    raw: np.ndarray,
    rec_starts: np.ndarray,
    counts: np.ndarray,
):
    """Fused boundary-aware InlineDecodeState field arrays for the batch.

    Returns ``(runlen_number, runlen_value, digit_cumsum,
    is_negative_per_position)`` over the flat raw stream, each byte-identical
    to the per-node scalar :func:`build_inline_decode_state` field on that
    node's window:

    * ``runlen_number`` (``uint16[total]``) -- per-node ``run_lengths`` of
      the inline-digit band (run length carried at each run's FIRST position).
    * ``runlen_value`` (``uint16[total]``) -- per-node ``run_lengths`` of the
      digit/sign (value) band.
    * ``digit_cumsum`` (``uint32[total + n_nodes]``) -- per-node exclusive
      prefix of the inline-digit band, packed CSR: node ``i``'s
      ``(count_i + 1)``-slot block starts at ``rec_starts[i] + i``, with the
      trailing slot = the node's digit total.
    * ``is_negative_per_position`` (``bool[total]``) -- flags a carrier whose
      ``p+1`` slot is a sign marker; never a node's last position.
    """
    return build_inline_state_fields_kernel(
        np.ascontiguousarray(raw, dtype=np.uint16),
        np.ascontiguousarray(rec_starts, dtype=np.int64),
        np.ascontiguousarray(counts, dtype=np.int64),
        int(_V2_RESERVED_DIGIT_COUNT),
        int(_V2_VALUE_NEGATIVE_TOKEN_ID),
        int(_V2_EAGER_BLOCK_END),
    )
