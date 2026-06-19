"""Stage 3a -- per-call-target inline-byte concatenation (ALG-1).

Single concern: lift the surviving inline-digit byte payloads from every
level-4 call_target in a :class:`Stage2Batch` into ONE flat ``u8`` array
with a leading-zero pad at index 0, and emit a parallel per-call-target
CSR of start offsets into that array.

The narrowing-assignment trick (plan ALG-1): assigning a ``u16`` numpy
array into a ``u8`` destination truncates the high byte. For inline-band
ids (``id < 256``) the high byte is zero, so the truncation is lossless;
this lets us copy ``raw_tokens[number_mask]`` straight into a ``u8``
slice without an explicit ``.astype(np.uint8)`` step. Numpy 2.x tightens
casting rules; the explicit ``.astype(np.uint8)`` here documents the
intent and keeps the code forward-compatible.

Cut-aware semantics (plan D2 + Stage 2 step 4)
----------------------------------------------
The stage-2 cutoff is *on the post-promotion expanded stream*, not the
raw stream. A multi-chunk source (3-chunk VC2, 2-chunk F128) at expanded
position ``e_start`` whose row's cut falls at ``e_start + j`` with
``j < chunk_count`` contributes ``j`` visible chunks per ALG-8 -- but
the BYTE layout retains the FULL ``L`` bytes of every consumed carrier
in the buffer. 3c (``_number_decode.py``) uses the canonical ALG-8
``[p_carrier_byte + L - 8*(c+1), p_carrier_byte + L - 8*c)`` offset
formula and simply skips emitting ``idx_2d`` rows for dropped chunks
(via ``K_visible``). Keeping the full payload here means 3c's offset
arithmetic is independent of which chunks survived; the only cost is
a few extra bytes in the buffer per cut multi-chunk source.

For non-multi-chunk carriers (F16 / BF16 / F32 / F64 / F80 + single-
chunk VC2 + IDENTITY-band carriers + instruction reps) ``K = 1``, so
``j`` is either ``0`` (carrier dropped) or ``1`` (carrier fully visible).
F128 NaN/Inf is ``K = 1`` even though the byte payload is still 16
bytes -- so when ``j = 1`` all 16 bytes survive.

Algorithm per call_target
-------------------------
``is_cut == False``: all inline-digit positions of the raw stream
survive, i.e. ``raw_tokens[number_mask]`` (in raw stream order, MSB-
first big-endian within each payload).

``is_cut == True``: walk the expanded-stream extra masks to identify
the last consumed raw carrier. Every consumed raw carrier (including
the last one, even when some of its chunks were dropped past the cut)
contributes its FULL ``L``-byte payload. Carriers AFTER the last
consumed one contribute nothing.

Output layout
-------------
``inline_bytes`` is ``u8[1 + total_buffered_bytes]``. Index 0 is the
leading-zero pad consumed by short-payload ``idx_2d`` gathers (plan D9
+ Stage 3 step 1). Each per-call-target start offset abuts the
previous call_target's end (or ``1`` for the first call_target) so that
the per-call-target ranges tile exactly ``[1, len(inline_bytes))``.

Plan reference: ``batch_decode_plan.md`` -- ALG-1 + ALG-2 + ALG-8 +
``## Stages -- algorithm sketch`` Stage 3 step 2.
"""

from __future__ import annotations

import numpy as np

from dedup_hashmap import build_inline_bytes_kernel

from ._dense_columns import DenseColumns


__all__ = ["build_inline_bytes"]


# ---------------------------------------------------------------------------
# Public entry point: feed the dense columns to the GIL-released gather
# kernel + build the per-call-target start-offset CSR from the counts.
#
# The per-call-target surviving-byte extraction (the old per-node
# ``_surviving_bytes`` Python loop with masks) is now ONE ``py.detach``
# Rust kernel (``dedup_hashmap.build_inline_bytes_kernel``) reading the
# flat :class:`DenseColumns` columns directly. The kernel retains every
# consumed carrier's FULL payload so 3c's ALG-8 offset formula stays
# independent of which chunks survived the cut; the chunk-count branch
# (VC2 ceil(L/8), F128 NaN/Inf detection) is 3c's concern.
# ---------------------------------------------------------------------------


def build_inline_bytes(
    dense: DenseColumns,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate surviving inline bytes across the batch (ALG-1).

    Reads the shared :class:`DenseColumns` front-matter in DFS
    (== emitted-node) order -- the same per-node columns the stage-3
    sites used to re-walk the ``Stage2Batch`` tree for -- and produces:

    * ``inline_bytes`` -- ``u8`` array of length
      ``1 + total_surviving_bytes``. Index 0 is the leading-zero pad
      (consumed by short-payload ``idx_2d`` gathers per plan D9).
    * ``inline_byte_starts`` -- ``int64[n_call_targets]``, one start
      offset per level-4 call_target, parallel to the DFS node axis.
      Entry 0 is ``1`` (past the pad); each subsequent entry abuts the
      previous call_target's end (``start[i] = start[i-1] + count[i-1]``).
      A fully-dropped call_target contributes ``count == 0`` so its
      start equals the next call_target's start (zero-length range).
      The per-call-target ``slice`` objects the staged hierarchy needs
      are materialised leaf-locally from these starts (the only
      consumer of real ``slice`` objects) -- the vector hot path threads
      the start array directly into the idx_2d builders.

    GIL-released (B-S3): the per-node ``_surviving_bytes`` Python gather
    loop (per-CT masked digit pick + the cut-path carrier walk) is a
    single ``py.detach`` Rust kernel reading the flat
    :class:`DenseColumns` columns directly. The kernel returns the flat
    ``inline_bytes`` buffer (leading-zero pad included) + the per-node
    contributed byte count; this entry point derives the abutting start
    offsets from the counts with a vectorised leading-1 + exclusive
    prefix sum (no per-call-target Python loop). The narrowing-assignment
    trick (ALG-1) lives in the kernel (``raw_token as u8`` keeps the low
    byte; inline-band ids ``< 256`` so it is lossless).

    Parameters
    ----------
    dense
        The shared dense front-matter (:class:`DenseColumns`) -- the
        per-node RAW / EXPANDED columns + surviving / cut scalars over
        the full DFS node axis.

    Returns
    -------
    tuple of ``(np.ndarray[np.uint8], np.ndarray[np.int64])``
        The flat byte buffer and the per-call-target start-offset CSR.
    """
    inline_bytes, per_call_target_counts = build_inline_bytes_kernel(
        np.ascontiguousarray(dense.raw_tokens, dtype=np.uint16),
        np.ascontiguousarray(dense.number_mask, dtype=np.bool_),
        np.ascontiguousarray(dense.real_mask, dtype=np.bool_),
        np.ascontiguousarray(dense.runlen_number, dtype=np.uint16),
        np.ascontiguousarray(dense.extra_value_v2_mask, dtype=np.bool_),
        np.ascontiguousarray(dense.extra_f128_mask, dtype=np.bool_),
        np.ascontiguousarray(dense.is_cut, dtype=np.bool_),
        np.ascontiguousarray(dense.surviving_token_count, dtype=np.int64),
        np.ascontiguousarray(dense.raw_offsets, dtype=np.int64),
        np.ascontiguousarray(dense.node_offsets, dtype=np.int64),
    )

    # Per-call-target start offsets into ``inline_bytes``. Entry 0 starts
    # past the leading-zero pad (offset 1); subsequent entries abut the
    # previous call_target's end. Vectorised leading-1 + exclusive prefix
    # sum of the per-call-target byte counts -- byte-identical to the old
    # ``cursor = 1; for count: append(cursor); cursor += count`` loop.
    counts = np.ascontiguousarray(per_call_target_counts, dtype=np.int64)
    inline_byte_starts = 1 + (np.cumsum(counts) - counts)

    return inline_bytes, inline_byte_starts
