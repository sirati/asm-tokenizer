"""Per-call-target token-stream expansion (stage 2 pre-cutoff step).

Single concern: take a :class:`Stage1CallTarget` and produce the
post-promotion, post-strip, post-shift ``expanded_token_ids`` plus the
companion ``extra_value_v2_mask`` + ``extra_f128_mask`` boolean arrays
that mark the slots filled by multi-chunk promotion. The output of this
module is the per-call-target shape consumed by the cutoff walk
(:mod:`._cutoff_walk`) and the surviving-count predictor
(:mod:`._surviving_counts`); field names mirror
:class:`Stage2CallTarget` so the downstream wiring step is a pure
field-by-field copy.

What this module does (and only does):

1. Promotes VC2 multi-chunk sources -- positions where
   ``raw_tokens[p] == VC2_VOCAB_ID & real_mask[p]`` with
   ``chunk_count = max(1, ceil(runlen_number[p+1] / 8))`` -- painting
   ``working_tokens[p+1 : p+chunk_count] = VC2_VOCAB_ID`` so the strip
   step keeps them in the final stream.
2. Promotes F128 multi-chunk sources -- positions where
   ``raw_tokens[p] == FLOAT128_VOCAB_ID & real_mask[p]`` with the
   per-source chunk count driven by ALG-2's inline NaN/Inf detection on
   the high u16 of the binary128 payload (15-bit exponent against the
   ``0x7FFF`` all-ones pattern, sign bit stripped via ``& 0x7FFF``).
   Finite sources paint ``working_tokens[p+1] = FLOAT128_VOCAB_ID``;
   NaN/Inf sources paint nothing (1 chunk total).
3. Strips inline-digit + sign-marker positions from the working buffer
   then shifts the surviving ids down by ``_V2_RESERVED_DIGIT_COUNT``
   (= 256) so the smallest produced id is 1 and slot 0 stays reserved
   for the model-facing null-content marker.
4. Prepends the calling-category self-token derived from
   ``stage1_call_target.encounter_category``
   (LOCAL_FUNC -> shifted id 9; PLT_FUNC -> shifted id 10). The
   prepended slot's identity counter is written later by stage 4 per
   ALG-9.

What this module does NOT do:

* The cutoff walk -- ``predicted_full_length`` is exposed via
  :class:`ExpandedTokens` and 2b (``_cutoff_walk``) decides which
  call_target is the cut one; 2c (``_surviving_counts``) does the
  per-prefix identity / number-chunk count.
* Side-array population -- the masks here flag which positions in
  ``expanded_token_ids`` are PROMOTED multi-chunk slots; stage 3 uses
  these to size + populate the chunk-exponent sidecars and to drive
  ALG-7's normalization step.

Plan reference: ``batch_decode_plan.md`` ``## Stages -- algorithm
sketch`` Stage 2 step 1 + ALG-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category

if TYPE_CHECKING:
    from ._types import Stage1CallTarget


__all__ = ["ExpandedTokens", "expand_tokens"]


# ---------------------------------------------------------------------------
# Module-level aliases for the unified-vocab layout constants.
#
# Resolving these once at import time keeps the hot path free of
# attribute lookups on :class:`VocabularyManager`. The source of truth is
# the vocab class itself; any drift in the canonical layout surfaces here
# as a constant-import update + a test-failure cascade.
#
# Plan vocab table (``batch_decode_plan.md`` ``### Vocab + wire format
# reference``):
#
# * ``_V2_RESERVED_DIGIT_COUNT`` = 256  -- the strip + shift boundary.
# * ``_V2_NUMBER_BLOCK_START``   = 257  -- first NUMBER carrier (VC2).
# * ``_V2_NUMBER_BLOCK_COUNT``   = 7    -- VC2 + F16 + BF16 + F32 + F64
#                                          + F80 + F128 (source-declaration order).
# * ``_V2_IDENTITY_BLOCK_START`` = 264  -- first IDENTITY carrier
#                                          (BLOCK_V2; then LOCAL_FUNC, PLT_FUNC,
#                                          EXT_FUNC, ...).
# ---------------------------------------------------------------------------
_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT
_V2_NUMBER_BLOCK_START = VocabularyManager._V2_NUMBER_BLOCK_START
_V2_NUMBER_BLOCK_COUNT = VocabularyManager._V2_NUMBER_BLOCK_COUNT
_V2_IDENTITY_BLOCK_START = VocabularyManager._V2_IDENTITY_BLOCK_START

# VC2 is the first NUMBER carrier; F128 is the LAST NUMBER carrier
# (per the plan vocab table, NUMBER order is
# ``VC2, F16, BF16, F32, F64, F80, F128``).
_VC2_VOCAB_ID = _V2_NUMBER_BLOCK_START
_FLOAT128_VOCAB_ID = _V2_NUMBER_BLOCK_START + _V2_NUMBER_BLOCK_COUNT - 1

# Identity-block layout, in the plan's canonical order:
#   index 0 -> BLOCK_V2
#   index 1 -> LOCAL_FUNC
#   index 2 -> PLT_FUNC
#   index 3 -> EXT_FUNC
#   ...
# Per plan D3, EXT_FUNC is NOT inlined (no body); the dispatch below
# rejects it explicitly so a corrupt stage-1 walk surfaces as an
# AssertionError rather than producing wrong shifted ids.
_LOCAL_FUNC_VOCAB_ID = _V2_IDENTITY_BLOCK_START + 1
_PLT_FUNC_VOCAB_ID = _V2_IDENTITY_BLOCK_START + 2

# Shifted (model-facing) versions: id - 256. These are the values that
# end up at ``expanded_token_ids[0]`` for each calling category.
_LOCAL_FUNC_SHIFTED = _LOCAL_FUNC_VOCAB_ID - _V2_RESERVED_DIGIT_COUNT
_PLT_FUNC_SHIFTED = _PLT_FUNC_VOCAB_ID - _V2_RESERVED_DIGIT_COUNT


# ---------------------------------------------------------------------------
# Output dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpandedTokens:
    """Per-call-target output of stage 2's pre-cutoff expansion step.

    Field names match :class:`Stage2CallTarget` so the 2d wiring step is
    a mechanical field-by-field handoff (no rename / reshape needed).

    Consumed by:

    * 2b (``_cutoff_walk``) -- reads only :attr:`predicted_full_length`.
    * 2c (``_surviving_counts``) -- reads
      :attr:`expanded_token_ids` + the partial-cut length to count
      identity-band and number-band positions in the surviving prefix.
    """

    expanded_token_ids: np.ndarray
    """``u16[predicted_full_length]`` -- the function's post-promotion,
    post-strip (drops inline-digit band + sign marker), post-shift
    (``id - 256``) token stream BEFORE stage-4 cutoff. Slot 0 is the
    prepended self-token; slots 1+ are the function body."""

    extra_value_v2_mask: np.ndarray
    """``bool[predicted_full_length]`` -- True at positions that are
    PROMOTED VC2 chunks (slots painted by step 1; the original VC2
    carrier slot stays False because it was already a real token in
    the raw stream). Slot 0 (the prepended self-token) is always
    False."""

    extra_f128_mask: np.ndarray
    """``bool[predicted_full_length]`` -- True at positions that are
    PROMOTED F128 chunks (the second-of-two slot for a finite F128
    source; NaN/Inf sources contribute no promoted slots). Slot 0 is
    always False."""

    predicted_full_length: int
    """``= expanded_token_ids.shape[0]``; pre-cached so the cutoff walk
    reads it without going through numpy."""


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _promote_vc2(
    working_tokens: np.ndarray,
    real_mask: np.ndarray,
    runlen_number: np.ndarray,
) -> np.ndarray:
    """Paint VC2 multi-chunk continuation slots into ``working_tokens``.

    Returns the boolean ``extra_vc2_promoted_mask`` over the raw-stream
    index space (``shape = working_tokens.shape``); True at positions
    that were painted to ``_VC2_VOCAB_ID`` by this call. The leading
    carrier slot stays False -- it was already a real token in the raw
    stream.

    Per-source chunk count:
        ``chunk_count = max(1, ceil(runlen_number[p + 1] / 8))``

    where ``runlen_number[p + 1]`` is the inline-digit run length
    immediately following the carrier; ``max(1, ...)`` handles the
    zero-payload edge case (an isolated VC2 with no following digits is
    still a 1-chunk source -- the carrier itself).
    """
    n = working_tokens.shape[0]
    raw_promoted_mask = np.zeros(n, dtype=bool)

    vc2_carrier_mask = real_mask & (working_tokens == _VC2_VOCAB_ID)
    vc2_positions = np.nonzero(vc2_carrier_mask)[0]

    if vc2_positions.size == 0:
        return raw_promoted_mask

    # Tail-bound check: every VC2 carrier needs a p+1 slot to read its
    # payload-runlength. A carrier at the last raw-stream position is a
    # malformed v2 stream (the codec MUST emit either the leading
    # inline-digit byte of the payload or an explicit empty run -- never
    # nothing at all).
    if int(vc2_positions[-1]) >= n - 1:
        raise AssertionError(
            "VC2 carrier at the last raw-stream position -- malformed "
            "v2 stream (carrier needs a p+1 slot for the payload "
            "inline-digit run)."
        )

    # Per-source chunk count, vectorized.
    payload_lengths = runlen_number[vc2_positions + 1].astype(np.int64)
    # ceil(L / 8) == (L + 7) // 8; max(1, ...) guards the empty-run case.
    chunk_counts = np.maximum(np.int64(1), (payload_lengths + 7) // 8)

    # Paint per-source. The number of VC2 sources per function is
    # typically small (single-digit to low-hundreds even on pathological
    # inputs) so the per-source Python loop here is not a hot-path
    # concern -- the vectorized strip+shift below dominates the cost.
    # Each iteration is a fixed-stride numpy slice fill.
    for source_idx in range(vc2_positions.size):
        p = int(vc2_positions[source_idx])
        chunk_count = int(chunk_counts[source_idx])
        if chunk_count <= 1:
            continue
        # Slots p+1 .. p+chunk_count-1 are continuations. The codec
        # precondition is that the same number of inline-digit slots
        # follow the carrier; this is asserted by the existing
        # InlineDecodeState build (which would have flagged a malformed
        # stream upstream). We still bounds-check here to avoid silently
        # writing past the array end on a corrupt input.
        end = p + chunk_count
        if end > n:
            raise AssertionError(
                f"VC2 carrier at position {p} declares {chunk_count} "
                f"chunks but only {n - p} raw-stream slots remain -- "
                "malformed v2 stream."
            )
        working_tokens[p + 1 : end] = _VC2_VOCAB_ID
        raw_promoted_mask[p + 1 : end] = True

    return raw_promoted_mask


def _promote_f128(
    working_tokens: np.ndarray,
    real_mask: np.ndarray,
) -> np.ndarray:
    """Paint F128 finite-source continuation slots; inline ALG-2 detect.

    Returns the boolean ``extra_f128_promoted_mask`` over the raw-stream
    index space; True at positions painted to ``_FLOAT128_VOCAB_ID``
    here (finite-source slot ``p + 1``). NaN/Inf sources paint nothing
    (their single chunk IS the original carrier).

    ALG-2 detail: IEEE-754 binary128's 15-bit exponent occupies bits
    112..126 (big-endian byte 0's low 7 bits + all 8 bits of byte 1).
    The codec stores the payload as 16 consecutive inline-digit slots
    at positions ``p+1..p+16`` (each slot holds one big-endian byte in
    its low 8 bits, high 8 bits zero). The detection masks the high
    u16 with ``0x7FFF`` -- this strips the sign bit at the top of byte 0
    and leaves exactly the 15-bit exponent for the all-ones comparison
    against ``0x7FFF`` (NaN: mantissa != 0; +-Inf: mantissa == 0; both
    are 1-chunk sources for the chunk-count purpose).
    """
    n = working_tokens.shape[0]
    raw_promoted_mask = np.zeros(n, dtype=bool)

    f128_carrier_mask = real_mask & (working_tokens == _FLOAT128_VOCAB_ID)
    f128_positions = np.nonzero(f128_carrier_mask)[0]

    if f128_positions.size == 0:
        return raw_promoted_mask

    # Tail-bound check: the NaN/Inf detection reads bytes at p+1 AND
    # p+2 (the high u16 of the payload). A carrier within 2 positions
    # of the tail can't satisfy that.
    if int(f128_positions[-1]) >= n - 2:
        raise AssertionError(
            "F128 carrier within 2 positions of the raw-stream tail -- "
            "malformed v2 stream (ALG-2 needs the high u16 of the "
            "binary128 payload at p+1, p+2)."
        )

    # ALG-2: build high u16 from the two big-endian payload bytes.
    # raw_tokens entries are u16 with the low 8 bits holding the byte
    # value (because inline-digit slots live in [0, 256)); so a left
    # shift by 8 of the byte-0 slot OR'd with byte-1 reconstructs the
    # big-endian high u16 of the binary128.
    high_bytes = working_tokens[f128_positions + 1].astype(np.uint16) << np.uint16(8)
    low_bytes = working_tokens[f128_positions + 2].astype(np.uint16)
    high_u16 = high_bytes | low_bytes
    is_nan_or_inf = (high_u16 & np.uint16(0x7FFF)) == np.uint16(0x7FFF)

    # Finite sources -> chunk_count == 2 (paint slot p+1).
    finite_positions = f128_positions[~is_nan_or_inf]
    if finite_positions.size > 0:
        paint_targets = finite_positions + 1
        working_tokens[paint_targets] = _FLOAT128_VOCAB_ID
        raw_promoted_mask[paint_targets] = True

    return raw_promoted_mask


def _calling_category_shifted_id(category: Category) -> int:
    """Map ``encounter_category`` to its shifted vocab id.

    Only LOCAL_FUNC and PLT_FUNC are valid here -- per plan D3 the
    splice tree carries inlined bodies whose calling category is one of
    those two; EXT_FUNC is never inlined (no body to inline) and any
    other Category at this entry point indicates a stage-1 walker bug.
    """
    if category is Category.LOCAL_FUNC:
        return _LOCAL_FUNC_SHIFTED
    if category is Category.PLT_FUNC:
        return _PLT_FUNC_SHIFTED
    raise AssertionError(
        f"expand_tokens received encounter_category={category!r}; only "
        "LOCAL_FUNC and PLT_FUNC are inlined per plan D3 (EXT_FUNC has "
        "no body; all other categories are not call_target categories)."
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def expand_tokens(stage1_call_target: "Stage1CallTarget") -> ExpandedTokens:
    """Promote multi-chunk sources + strip + shift + prepend self-token.

    Stage 2's pre-cutoff per-call-target expansion. See
    ``batch_decode_plan.md`` ``## Stages -- algorithm sketch`` Stage 2
    step 1 + ALG-2 for the algorithm.

    Parameters
    ----------
    stage1_call_target
        The loaded :class:`Stage1CallTarget`. Only its
        :attr:`InlineDecodeState` (``state.raw_tokens``,
        ``state.real_mask``, ``state.runlen_number``) and
        :attr:`encounter_category` are read; the rest of the dataclass
        is irrelevant to this stage.

    Returns
    -------
    :class:`ExpandedTokens`
        With ``expanded_token_ids[0]`` set to the calling-category
        shifted vocab id and ``expanded_token_ids[1:]`` the
        post-promotion, post-strip, post-shift function body.
    """
    state = stage1_call_target.state
    raw_tokens = state.raw_tokens
    real_mask = state.real_mask
    runlen_number = state.runlen_number

    # Working copy: promotion paints into this buffer; the original
    # state's raw_tokens stays untouched (the existing pre-compute
    # contract says raw_tokens is NOT a copy -- consumers must not
    # mutate it).
    working_tokens = raw_tokens.copy()

    # Per-promoter raw-stream-indexed masks; the strip step below
    # filters them through the same `> 256` keep mask so the resulting
    # expanded-stream-indexed masks align with the produced token ids.
    vc2_promoted_raw_mask = _promote_vc2(
        working_tokens, real_mask, runlen_number
    )
    f128_promoted_raw_mask = _promote_f128(working_tokens, real_mask)

    # Strip + shift. Same keep predicate the legacy ``_decode_to_staging``
    # uses (plan section "Null-content contract (already shipped --
    # official API)"): ``working_tokens > 256`` drops both the inline-
    # digit band (ids 0..255) AND the value_negative sign marker (id
    # 256). Subtracting 256 shifts so the smallest produced id is 1 and
    # slot 0 stays reserved for the null-content marker.
    keep_mask = working_tokens > _V2_RESERVED_DIGIT_COUNT
    expanded_real = (
        (working_tokens[keep_mask] - _V2_RESERVED_DIGIT_COUNT)
        .astype(np.uint16)
    )
    extra_value_v2_real = vc2_promoted_raw_mask[keep_mask]
    extra_f128_real = f128_promoted_raw_mask[keep_mask]

    # Prepend the calling-category self-token. The prepended slot's
    # token id is the shifted vocab id (LOCAL_FUNC -> 9, PLT_FUNC -> 10
    # under the canonical unified-vocab layout); the slot's identity
    # counter is written later by stage 4 (ALG-9) and is NOT this
    # module's concern. The prepended slot is False on both extra masks
    # because it is a synthetic single-id token, not a promoted chunk.
    prepend_token_id = np.uint16(
        _calling_category_shifted_id(stage1_call_target.encounter_category)
    )
    expanded_token_ids = np.empty(expanded_real.shape[0] + 1, dtype=np.uint16)
    expanded_token_ids[0] = prepend_token_id
    expanded_token_ids[1:] = expanded_real

    extra_value_v2_mask = np.empty(expanded_token_ids.shape[0], dtype=bool)
    extra_value_v2_mask[0] = False
    extra_value_v2_mask[1:] = extra_value_v2_real

    extra_f128_mask = np.empty(expanded_token_ids.shape[0], dtype=bool)
    extra_f128_mask[0] = False
    extra_f128_mask[1:] = extra_f128_real

    return ExpandedTokens(
        expanded_token_ids=expanded_token_ids,
        extra_value_v2_mask=extra_value_v2_mask,
        extra_f128_mask=extra_f128_mask,
        predicted_full_length=int(expanded_token_ids.shape[0]),
    )
