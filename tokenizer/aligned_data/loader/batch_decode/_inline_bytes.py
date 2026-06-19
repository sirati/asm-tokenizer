"""Stage 3a -- per-call-target inline-byte concatenation (ALG-1).

Single concern: lift the surviving inline-digit byte payloads from every
level-4 call_target in a :class:`Stage2Batch` into ONE flat ``u8`` array
with a leading-zero pad at index 0, and emit a parallel list of per-call
target slices into that array.

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
+ Stage 3 step 1). Each per-call-target slice starts at
``previous.stop`` (or ``1`` for the first call_target) so that the
slice union is exactly ``[1, len(inline_bytes))``.

Plan reference: ``batch_decode_plan.md`` -- ALG-1 + ALG-2 + ALG-8 +
``## Stages -- algorithm sketch`` Stage 3 step 2.
"""

from __future__ import annotations

import numpy as np

from ._dense_columns import DenseColumns


__all__ = ["build_inline_bytes"]


# ---------------------------------------------------------------------------
# Per-call-target surviving-byte extraction. Buffer layout retains every
# consumed carrier's FULL payload so 3c's ALG-8 offset formula stays
# independent of which chunks survived the cut; the chunk-count branch
# (VC2 ceil(L/8), F128 NaN/Inf detection) is 3c's concern.
# ---------------------------------------------------------------------------


def _surviving_bytes(dense: DenseColumns, e: int) -> np.ndarray:
    """Buffered inline bytes for one call_target as a ``u8`` array.

    Returns the raw inline-digit values (already truncated to ``u8``)
    in raw-stream order. Every consumed raw carrier contributes its
    FULL ``L``-byte payload -- including the last consumed carrier
    when only some of its chunks survived the cut. 3c chooses which
    chunks to emit ``idx_2d`` rows for; the buffer always carries the
    full per-source payload so ALG-8's per-chunk offset formula stays
    valid.

    Parameters
    ----------
    dense
        The shared dense front-matter columns (:class:`DenseColumns`).
    e
        The DFS node index. Reads node ``e``'s ``raw_tokens`` /
        ``number_mask`` / ``real_mask`` / ``runlen_number`` plus
        ``extra_value_v2_mask`` / ``extra_f128_mask`` and the per-node
        ``surviving_token_count`` (== ``partial_cut_length``) / ``is_cut``
        scalars.

    Returns
    -------
    ``np.ndarray[np.uint8]``
        Length equals the contributed byte count for this call_target
        (full payload per consumed carrier). Zero for fully-dropped
        call_targets.

    Notes
    -----
    The narrowing-assignment trick (ALG-1) is realized via an explicit
    ``.astype(np.uint8)`` on the boolean-indexed slice -- boolean
    indexing already produces a fresh array, so the cast is in-place
    on the temporary. For inline-band ids (``< 256``) the cast is
    lossless.
    """
    raw_slice = dense.node_raw_slice(e)
    raw_tokens = dense.raw_tokens[raw_slice]
    number_mask = dense.number_mask[raw_slice]

    # ---- fully-included path ------------------------------------------------
    if not bool(dense.is_cut[e]):
        # All inline-digit positions of the raw stream survive. The
        # ``raw_tokens > number_mask`` boolean index returns a fresh
        # copy already; the ``.astype(np.uint8)`` narrows the dtype
        # cheaply (the cast applies to the temporary, not the
        # underlying memmap).
        return raw_tokens[number_mask].astype(np.uint8)

    # ---- cut path -----------------------------------------------------------
    # ``partial_cut_length`` is the prefix length in expanded_token_ids
    # that survives. Slot 0 of expanded_token_ids is the synthetic
    # prepend (no inline bytes); slots [1, partial_cut_length) are the
    # surviving body. A cut at or before slot 1 contributes no inline
    # bytes (only the prepend or nothing survives).
    partial_cut_length = int(dense.surviving_token_count[e])
    if partial_cut_length <= 1:
        return np.empty(0, dtype=np.uint8)

    expanded_slice = dense.node_expanded_slice(e)
    extra_vc2_mask = dense.extra_value_v2_mask[expanded_slice]
    extra_f128_mask = dense.extra_f128_mask[expanded_slice]

    # Within the visible body [1, partial_cut_length), a slot is either
    # a "painted continuation" (extra_*_mask True -- it was an inline-
    # digit byte the promotion painted into a carrier slot) or a "real
    # carrier" (real_mask True in the raw stream). Painted slots do
    # NOT consume a fresh raw-stream carrier.
    visible_extra_vc2 = extra_vc2_mask[1:partial_cut_length]
    visible_extra_f128 = extra_f128_mask[1:partial_cut_length]
    visible_is_painted = visible_extra_vc2 | visible_extra_f128
    visible_is_real_carrier = ~visible_is_painted

    # Number of raw-stream carriers consumed by the visible body.
    n_carriers_consumed = int(visible_is_real_carrier.sum())
    if n_carriers_consumed == 0:
        # No raw carriers consumed -- e.g. the visible body is entirely
        # painted slots. This shouldn't be reachable (the first slot
        # past the prepend is always a real carrier per the strip
        # walk), but the empty-output guard keeps the function total.
        return np.empty(0, dtype=np.uint8)

    # Raw positions of all carriers in raw-stream order.
    real_mask = dense.real_mask[raw_slice]
    carrier_positions = np.nonzero(real_mask)[0]
    # The last raw carrier consumed by the visible body.
    p_last = int(carrier_positions[n_carriers_consumed - 1])

    # Payload length L = inline-digit run-length immediately following
    # the last carrier. ``runlen_number`` stores run-start lengths;
    # ``runlen_number[p+1]`` is exactly the per-source payload length.
    # The last consumed carrier contributes its FULL ``L`` bytes -- 3c
    # will skip emitting ``idx_2d`` rows for dropped chunks, but the
    # ALG-8 per-chunk offset formula assumes the full payload is
    # addressable in the buffer regardless of which chunks survived.
    runlen_number = dense.runlen_number[raw_slice]
    if p_last + 1 < runlen_number.shape[0]:
        L_last = int(runlen_number[p_last + 1])
    else:
        L_last = 0

    # ---- bytes from all consumed carriers (full per-source payload) ----------
    # Inline-digit positions are owned by their preceding
    # ``carries_inline_mask`` carrier in raw-stream order (the v2 codec
    # emits the payload IMMEDIATELY after its owning carrier, never
    # orphan bytes). To pick up every consumed carrier's FULL payload
    # we slice ``number_mask`` up to ``p_last + 1 + L_last`` -- this
    # captures all inline bytes belonging to carriers strictly before
    # ``p_last`` (their payloads precede ``p_last``) plus the entire
    # ``L_last``-byte payload of the last consumed carrier.
    number_mask_keep = number_mask.copy()
    number_mask_keep[p_last + 1 + L_last :] = False
    bytes_kept = raw_tokens[number_mask_keep]

    return bytes_kept.astype(np.uint8)


# ---------------------------------------------------------------------------
# Public entry point: walk the dense node axis + build the flat buffer.
# ---------------------------------------------------------------------------


def build_inline_bytes(
    dense: DenseColumns,
) -> tuple[np.ndarray, list[slice]]:
    """Concatenate surviving inline bytes across the batch (ALG-1).

    Reads the shared :class:`DenseColumns` front-matter in DFS
    (== emitted-node) order -- the same per-node columns the stage-3
    sites used to re-walk the ``Stage2Batch`` tree for -- and produces:

    * ``inline_bytes`` -- ``u8`` array of length
      ``1 + total_surviving_bytes``. Index 0 is the leading-zero pad
      (consumed by short-payload ``idx_2d`` gathers per plan D9).
    * ``inline_byte_slices`` -- one :class:`slice` per level-4
      call_target, parallel to the DFS node axis. The first slice
      starts at ``1`` (past the pad); each subsequent slice starts at
      the previous slice's ``.stop``. Fully-dropped call_targets get
      a zero-length slice (``stop == start``).

    The narrowing-assignment trick (ALG-1) materializes inside
    :func:`_surviving_bytes` -- each per-call-target byte array is
    already ``u8``, so this entry point is a pure concatenation.

    Parameters
    ----------
    dense
        The shared dense front-matter (:class:`DenseColumns`) -- the
        per-node RAW / EXPANDED columns + surviving / cut scalars over
        the full DFS node axis.

    Returns
    -------
    tuple of ``(np.ndarray[np.uint8], list[slice])``
        The flat byte buffer and the per-call-target slice list.
    """
    # Per-call-target byte arrays, parallel to the DFS node axis. One
    # entry per node in the shared columnar front-matter.
    per_call_target_bytes: list[np.ndarray] = [
        _surviving_bytes(dense, e) for e in range(dense.n_nodes)
    ]

    # Length budget: 1 (leading-zero pad) + sum of per-call-target byte
    # counts. Pre-allocating avoids a second pass over the per-call-
    # target arrays.
    per_call_target_counts = np.array(
        [arr.shape[0] for arr in per_call_target_bytes], dtype=np.int64
    )
    total_bytes = int(per_call_target_counts.sum())
    inline_bytes = np.zeros(1 + total_bytes, dtype=np.uint8)
    # ``inline_bytes[0]`` stays 0 from the allocation -- this is the
    # leading-zero pad referenced by short-payload idx_2d gathers
    # (plan D9 + Stage 3 step 1).

    # Per-call-target slices into ``inline_bytes``. First slice starts
    # past the pad (offset 1); subsequent slices abut the previous
    # stop.
    inline_byte_slices: list[slice] = []
    cursor = 1
    for arr, count in zip(per_call_target_bytes, per_call_target_counts):
        n = int(count)
        sl = slice(cursor, cursor + n)
        if n > 0:
            inline_bytes[sl] = arr
        inline_byte_slices.append(sl)
        cursor += n

    return inline_bytes, inline_byte_slices
