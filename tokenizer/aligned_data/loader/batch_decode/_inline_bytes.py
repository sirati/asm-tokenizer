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
``j < chunk_count`` contributes only the LSB-end ``j`` chunks per ALG-8.
The chunks are emitted low-to-high (chunk 0 = LSB = trailing bytes of
the big-endian payload), so the surviving bytes are the LAST ``8 * j``
bytes of the source's L-byte payload.

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
the last consumed raw carrier and how many of its chunks are visible.
Every raw carrier BEFORE the last consumed one is fully visible (its
full payload survives). The last consumed carrier contributes its
LSB-end ``8 * j`` bytes when ``0 < j < K``, or its full ``L`` bytes
when ``j == K``.

Output layout
-------------
``inline_bytes`` is ``u8[1 + total_surviving_bytes]``. Index 0 is the
leading-zero pad consumed by short-payload ``idx_2d`` gathers (plan D9
+ Stage 3 step 1). Each per-call-target slice starts at
``previous.stop`` (or ``1`` for the first call_target) so that the
slice union is exactly ``[1, len(inline_bytes))``.

Plan reference: ``batch_decode_plan.md`` -- ALG-1 + ALG-2 + ALG-8 +
``## Stages -- algorithm sketch`` Stage 3 step 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tokenizer.token_manager import VocabularyManager


if TYPE_CHECKING:
    from ._types import Stage2Batch, Stage2CallTarget


__all__ = ["build_inline_bytes"]


# ---------------------------------------------------------------------------
# Module-level aliases for the unified-vocab carrier ids.
#
# ALG-1 + ALG-2 + ALG-8 only need the VC2 + F128 vocab ids (the two
# multi-chunk sources) plus the digit-boundary constant for sanity
# guards. Reading them at import time keeps the per-call-target hot
# path free of attribute lookups.
# ---------------------------------------------------------------------------
_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT
_V2_NUMBER_BLOCK_START = VocabularyManager._V2_NUMBER_BLOCK_START
_V2_NUMBER_BLOCK_COUNT = VocabularyManager._V2_NUMBER_BLOCK_COUNT

# VC2 is the first NUMBER carrier; F128 is the LAST NUMBER carrier.
# Per plan vocab table the order is VC2, F16, BF16, F32, F64, F80, F128.
_VC2_VOCAB_ID = _V2_NUMBER_BLOCK_START
_FLOAT128_VOCAB_ID = _V2_NUMBER_BLOCK_START + _V2_NUMBER_BLOCK_COUNT - 1

# F128 fixed payload size + IEEE-754 binary128 exponent mask. ALG-2's
# detection masks the high u16 of the payload with ``0x7FFF`` -- this
# strips the sign bit at the top of byte 0 and leaves exactly the 15-
# bit exponent for the all-ones comparison.
_F128_PAYLOAD_BYTES = 16
_F128_EXPONENT_MASK = np.uint16(0x7FFF)


# ---------------------------------------------------------------------------
# Per-carrier chunk-count helpers (single concern: emitted-chunk count
# for a raw carrier at position ``p``).
# ---------------------------------------------------------------------------


def _f128_is_nan_or_inf(raw_tokens: np.ndarray, p: int) -> bool:
    """ALG-2: detect NaN/Inf via the high u16 of the F128 payload.

    The 15-bit binary128 exponent occupies byte 0's low 7 bits (after
    the sign) PLUS all 8 bits of byte 1. The codec stores the payload
    as 16 consecutive inline-digit slots at ``p+1..p+16`` (each slot
    holds one big-endian byte in its low 8 bits, high 8 bits zero). The
    detection reads positions ``p+1`` (byte 0) and ``p+2`` (byte 1).
    """
    if p + 2 >= raw_tokens.shape[0]:
        raise AssertionError(
            f"F128 carrier at position {p} within 2 slots of stream "
            "tail -- malformed v2 stream (ALG-2 needs the high u16 at "
            "p+1, p+2)."
        )
    high_byte = np.uint16(raw_tokens[p + 1]) << np.uint16(8)
    low_byte = np.uint16(raw_tokens[p + 2])
    high_u16 = high_byte | low_byte
    return bool((high_u16 & _F128_EXPONENT_MASK) == _F128_EXPONENT_MASK)


def _chunk_count_for_carrier(
    raw_tokens: np.ndarray, p: int, payload_length: int
) -> int:
    """Emitted chunk count for the carrier at raw position ``p``.

    * VC2: ``max(1, ceil(L / 8))`` -- matches ``_promote_vc2`` exactly.
    * F128: ``1`` for NaN/Inf (per ALG-2), ``2`` for finite.
    * Everything else: ``1`` (single-chunk sources + identity carriers
      + instruction reps).

    ``payload_length`` is the inline-digit run-length immediately
    following the carrier (``runlen_number[p + 1]``); supplied by the
    caller so this function stays branch-on-vocab-id only.
    """
    carrier_id = int(raw_tokens[p])
    if carrier_id == _VC2_VOCAB_ID:
        # ceil(L / 8) == (L + 7) // 8; max(1, ...) guards the empty-
        # payload edge case (an isolated VC2 with no following digits
        # is still a 1-chunk source -- the carrier itself).
        return max(1, (int(payload_length) + 7) // 8)
    if carrier_id == _FLOAT128_VOCAB_ID:
        return 1 if _f128_is_nan_or_inf(raw_tokens, p) else 2
    return 1


# ---------------------------------------------------------------------------
# Per-call-target surviving-byte extraction.
# ---------------------------------------------------------------------------


def _surviving_bytes(stage2_call_target: "Stage2CallTarget") -> np.ndarray:
    """Surviving inline bytes for one call_target as a ``u8`` array.

    Returns the raw inline-digit values (already truncated to ``u8``)
    in raw-stream order, modulo the LSB-end tail-keep rule for the
    cut call_target's last consumed multi-chunk source.

    Parameters
    ----------
    stage2_call_target
        Reads ``stage1.state.raw_tokens`` / ``.number_mask`` /
        ``.real_mask`` / ``.runlen_number`` plus this stage's
        ``expanded_token_ids`` / ``extra_value_v2_mask`` /
        ``extra_f128_mask`` / ``partial_cut_length`` / ``is_cut``.

    Returns
    -------
    ``np.ndarray[np.uint8]``
        Length equals the contributed byte count for this call_target.
        Zero for fully-dropped call_targets.

    Notes
    -----
    The narrowing-assignment trick (ALG-1) is realized via an explicit
    ``.astype(np.uint8)`` on the boolean-indexed slice -- boolean
    indexing already produces a fresh array, so the cast is in-place
    on the temporary. For inline-band ids (``< 256``) the cast is
    lossless.
    """
    state = stage2_call_target.stage1.state
    raw_tokens = state.raw_tokens
    number_mask = state.number_mask

    # ---- fully-included path ------------------------------------------------
    if not stage2_call_target.is_cut:
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
    partial_cut_length = stage2_call_target.partial_cut_length
    if partial_cut_length <= 1:
        return np.empty(0, dtype=np.uint8)

    extra_vc2_mask = stage2_call_target.extra_value_v2_mask
    extra_f128_mask = stage2_call_target.extra_f128_mask

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
    carrier_positions = np.nonzero(state.real_mask)[0]
    # The last raw carrier consumed by the visible body.
    p_last = int(carrier_positions[n_carriers_consumed - 1])

    # Visible-body offset (0-based within [1, partial_cut_length)) of
    # the last real-carrier slot tells us how many expanded slots that
    # last carrier spans within the cut window: every slot from that
    # offset up to ``partial_cut_length - 1`` (exclusive end in
    # expanded coords) belongs to this carrier (the trailing ones are
    # painted continuations).
    real_carrier_offsets = np.nonzero(visible_is_real_carrier)[0]
    last_carrier_visible_offset = int(real_carrier_offsets[-1])
    j_last = partial_cut_length - 1 - last_carrier_visible_offset

    # Payload length L = inline-digit run-length immediately following
    # the last carrier. ``runlen_number`` stores run-start lengths;
    # ``runlen_number[p+1]`` is exactly the per-source payload length.
    runlen_number = state.runlen_number
    if p_last + 1 < runlen_number.shape[0]:
        L_last = int(runlen_number[p_last + 1])
    else:
        L_last = 0

    # Full chunk count for the last carrier -- needed to decide between
    # "full inclusion" (j == K -> all L bytes) and "mid-cut" (j < K ->
    # LSB-end 8 * j bytes).
    K_last = _chunk_count_for_carrier(raw_tokens, p_last, L_last)

    # ---- bytes from carriers BEFORE the last consumed one --------------------
    # Carriers before ``p_last`` are fully consumed (j == K each), so
    # all their inline-digit payload bytes survive. Inline-digit
    # positions are owned by their preceding ``carries_inline_mask``
    # carrier in raw-stream order; positions strictly before ``p_last``
    # belong to those earlier carriers (the v2 codec emits the payload
    # IMMEDIATELY after its owning carrier, never orphan bytes).
    number_mask_before = number_mask.copy()
    number_mask_before[p_last:] = False
    bytes_before = raw_tokens[number_mask_before]

    # ---- bytes from the last consumed carrier --------------------------------
    if j_last >= K_last:
        # Full inclusion: keep all L bytes of this carrier's payload.
        bytes_from_last = raw_tokens[p_last + 1 : p_last + 1 + L_last]
    else:
        # Mid-cut: per ALG-8 the surviving LSB ``j`` chunks correspond
        # to the LAST ``8 * j`` bytes of the big-endian payload. Clamp
        # to ``L_last`` defensively; for VC2/F128 finite the chunk-byte
        # arithmetic guarantees ``8 * j <= L`` whenever ``j < K``, but
        # the clamp makes the function total even for hypothetical
        # malformed inputs.
        n_bytes_from_last = min(8 * j_last, L_last)
        start = p_last + 1 + (L_last - n_bytes_from_last)
        stop = p_last + 1 + L_last
        bytes_from_last = raw_tokens[start:stop]

    return np.concatenate([bytes_before, bytes_from_last]).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public entry point: walk the stage-2 hierarchy + build the flat buffer.
# ---------------------------------------------------------------------------


def build_inline_bytes(
    stage2_batch: "Stage2Batch",
) -> tuple[np.ndarray, list[slice]]:
    """Concatenate surviving inline bytes across the batch (ALG-1).

    Walks the 4-level hierarchy of ``stage2_batch`` in DFS encounter
    order (sections -> variants -> call_targets, root-first) and
    produces:

    * ``inline_bytes`` -- ``u8`` array of length
      ``1 + total_surviving_bytes``. Index 0 is the leading-zero pad
      (consumed by short-payload ``idx_2d`` gathers per plan D9).
    * ``inline_byte_slices`` -- one :class:`slice` per level-4
      call_target, parallel to the DFS walk order. The first slice
      starts at ``1`` (past the pad); each subsequent slice starts at
      the previous slice's ``.stop``. Fully-dropped call_targets get
      a zero-length slice (``stop == start``).

    The narrowing-assignment trick (ALG-1) materializes inside
    :func:`_surviving_bytes` -- each per-call-target byte array is
    already ``u8``, so this entry point is a pure concatenation.

    Parameters
    ----------
    stage2_batch
        Output of stage 2 (length-predict + cutoff walk).

    Returns
    -------
    tuple of ``(np.ndarray[np.uint8], list[slice])``
        The flat byte buffer and the per-call-target slice list.
    """
    # Per-call-target byte arrays, parallel to the DFS walk order.
    per_call_target_bytes: list[np.ndarray] = []

    for stage2_section in stage2_batch.sections:
        for stage2_variant in stage2_section.variants:
            for stage2_call_target in stage2_variant.call_targets:
                per_call_target_bytes.append(_surviving_bytes(stage2_call_target))

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
