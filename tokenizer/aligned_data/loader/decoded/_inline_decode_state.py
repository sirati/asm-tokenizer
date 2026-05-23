"""Vectorized pre-compute shared across the v2 decode pass.

Single concern: build the per-stream numpy arrays the identity arm,
the number arm, the sign-handling code, and the postfix-invariant
check ALL read from -- a single source of truth for the masks +
run-length state.

The unified vocab (`format_version=1`) pins three contiguous ranges
into the wire-stream id space:

* ``[0, _V2_RESERVED_DIGIT_COUNT)`` = inline-digit slots (id < 256).
* ``id == _V2_VALUE_NEGATIVE_TOKEN_ID`` (= 256) = postfix sign marker;
  NEITHER inline nor a real-token carrier in the strict
  ``raw_tokens > 256`` sense.
* ``[_V2_RESERVED_TOKEN_COUNT, _V2_EAGER_BLOCK_END)`` (= [257, 272)) =
  the NUMBER + IDENTITY blocks; every id in this band is a "real
  token that carries an inline-byte run" (the carrier band the number
  arm filters on).

These layout invariants are what makes the runlength-diff sign
detection valid -- a sign marker at id 256 sits strictly between the
inline band and the carrier band, so two runlength passes (one over
the inline-digit mask, one over the inline-digit+sign mask) differ by
exactly 1 at every carrier whose postfix is a sign marker.

Plan reference: ``polished-greeting-moler.md`` -- ``## Algorithm`` and
``## Architecture -- InlineDecodeState dataclass``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokenizer.token_manager import VocabularyManager

from .run_lengths import run_lengths


__all__ = [
    "InlineDecodeState",
    "build_inline_decode_state",
    "expanded_to_raw_position_map",
]


# Local aliases of the wire-stream layout constants pinned on
# ``VocabularyManager``.  Reading these once at module load means the
# vectorized factory below does not pay an attribute-lookup cost per
# call.  Source of truth is :class:`VocabularyManager`; any change to
# the v2 layout invariants surfaces here as a constant-import update.
_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT
_V2_VALUE_NEGATIVE_TOKEN_ID = VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID
_V2_EAGER_BLOCK_END = VocabularyManager._V2_EAGER_BLOCK_END


@dataclass(frozen=True)
class InlineDecodeState:
    """Vectorized pre-compute shared by the identity arm, the number arm,
    the sign-handling code, and the postfix-invariant check.

    Single pre-compute per stream at the entry of
    :func:`build_inline_decode_state`; every consumer (the batch
    decode pipeline's expand + bulk-bytes + identity-decode stages)
    reads from these fields rather than rebuilding masks.  All arrays are length ``N`` where
    ``N = raw_tokens.shape[0]`` and are aligned position-by-position
    with the input stream.

    Field semantics:

    * ``raw_tokens``: the original uint16 stream (NOT a copy -- consumers
      MUST NOT mutate it; the working buffer in the batch decode
      pipeline's :mod:`batch_decode._expand_tokens` is the only place
      mutation is allowed).
    * ``real_mask``: ``raw_tokens > 256`` (strict).  At id 256 the slot
      is the postfix sign marker, which is neither inline nor a real
      token in the carrier sense -- the strict inequality treats it as
      "non-real" so the strip + shift step drops it cleanly.
    * ``number_mask``: ``raw_tokens < 256`` (the inline-digit band).
    * ``runlen_number``: per-position run length of ``number_mask``
      (run-start carries length; other positions zero).  Drives the
      identity-arm + number-arm inline-byte extraction.
    * ``runlen_value``: per-position run length of ``~real_mask`` (i.e.
      ``raw_tokens <= 256``).  Differs from ``runlen_number`` ONLY at
      positions where a sign marker sits adjacent to an inline-digit
      run; that local +1 delta is exactly what the sign-detection
      diff picks up.
    * ``carries_inline_mask``: ``real_mask & (raw_tokens <
      _V2_EAGER_BLOCK_END)``.  The set of stream positions that own an
      inline-byte run (number tokens + identity tokens combined).
    * ``is_negative_per_position``: True at carriers whose immediate
      postfix slot is a sign marker; False elsewhere (including
      non-carrier positions).  Pre-computed via the runlength-diff so
      the number arm reads off the sign in O(1) per source.
    * ``digit_cumsum``: ``u32[N + 1]`` -- exclusive-prefix cumsum over
      ``number_mask``; ``digit_cumsum[k] = sum(number_mask[0:k])``.
      ``digit_cumsum[0] = 0``. Consumers read ``digit_cumsum[p + 1]``
      to obtain the count of inline-digit bytes preceding raw position
      ``p + 1`` -- exactly the per-source first-payload-byte offset
      within the call_target's inline-byte slice.  Lifted onto the
      state so the per-call-target cumsum is computed ONCE per source
      stream rather than once per stage-3 consumer arm.
    """

    raw_tokens: np.ndarray
    real_mask: np.ndarray
    number_mask: np.ndarray
    runlen_number: np.ndarray
    runlen_value: np.ndarray
    carries_inline_mask: np.ndarray
    is_negative_per_position: np.ndarray
    digit_cumsum: np.ndarray


def build_inline_decode_state(
    raw_tokens: np.ndarray, *, format_version: int
) -> InlineDecodeState:
    """Build the per-stream :class:`InlineDecodeState`.

    Asserts ``format_version == 1``: the carrier-band layout the
    vectorized lookup depends on (sign marker at id 256, carrier block
    at ids [257, 272)) is only guaranteed under the unified vocab.
    Per-binary vocabs (``format_version != 1``) live in a different
    layout regime; callers that need decoded views on those streams
    must use a vocab-aware code path that does not yet exist.
    """
    if format_version != 1:
        raise AssertionError(
            f"build_inline_decode_state requires format_version=1 "
            f"(unified vocab); got format_version={format_version}. "
            "The vectorized carrier-band lookup is only valid under "
            "the unified id layout."
        )

    n = int(raw_tokens.shape[0])

    real_mask = raw_tokens > _V2_VALUE_NEGATIVE_TOKEN_ID
    number_mask = raw_tokens < _V2_RESERVED_DIGIT_COUNT
    # ``value_mask`` covers numbers AND the optional postfix sign
    # marker; equal to ``~real_mask`` under strict ``> 256``.
    value_mask = ~real_mask
    runlen_number = run_lengths(number_mask)
    runlen_value = run_lengths(value_mask)
    carries_inline_mask = real_mask & (raw_tokens < _V2_EAGER_BLOCK_END)

    is_negative_per_position = np.zeros(n, dtype=bool)
    if n > 1:
        # Carriers in [0, N-1) only -- a carrier at the LAST position
        # has no ``p+1`` slot to read so its sign defaults to False.
        # Filtering to carriers FIRST then comparing keeps the work to
        # K' comparisons (one per non-tail carrier) instead of N-1.
        carriers_excl_last = carries_inline_mask[:-1]
        runlen_num_at_p1 = runlen_number[1:][carriers_excl_last]
        runlen_val_at_p1 = runlen_value[1:][carriers_excl_last]
        is_negative_per_position[:-1][carriers_excl_last] = (
            runlen_val_at_p1 != runlen_num_at_p1
        )

    # Exclusive-prefix cumsum of ``number_mask``.  ``digit_cumsum[0] = 0``
    # by construction; ``digit_cumsum[k] = sum(number_mask[0:k])`` for
    # ``k >= 1``.  Viewed as ``uint8`` for the cumsum operand keeps the
    # working dtype small and matches the ``np.uint32`` accumulator.
    digit_cumsum = np.zeros(n + 1, dtype=np.uint32)
    if n > 0:
        np.cumsum(number_mask.view(np.uint8), out=digit_cumsum[1:])

    return InlineDecodeState(
        raw_tokens=raw_tokens,
        real_mask=real_mask,
        number_mask=number_mask,
        runlen_number=runlen_number,
        runlen_value=runlen_value,
        carries_inline_mask=carries_inline_mask,
        is_negative_per_position=is_negative_per_position,
        digit_cumsum=digit_cumsum,
    )


def expanded_to_raw_position_map(
    state: InlineDecodeState,
    n_expanded_real: int,
    extra_value_v2_mask: np.ndarray,
    extra_f128_mask: np.ndarray,
) -> np.ndarray:
    """Recover the raw-stream position for each ``expanded[1:]`` slot.

    Stage-2a painting (VC2 ceil(L/8) chunks past the carrier; F128
    finite-source continuation) inserts "painted" continuation slots in
    the post-promotion expanded stream that have no ``state.real_mask``
    counterpart.  Painted slots are contiguous in raw-space immediately
    after their carrier, so the painted slot's raw position equals the
    prior expanded[1:] slot's raw position + 1.

    Parameters
    ----------
    state
        The per-source-stream :class:`InlineDecodeState` (only
        ``state.real_mask`` is read here).
    n_expanded_real
        Number of expanded[1:] slots to populate.  The caller computes
        this from ``extra_*_mask.shape[0] - 1`` (the leading slot is
        the synthetic prepend, which has no raw counterpart).
    extra_value_v2_mask, extra_f128_mask
        ``bool[predicted_full_length]`` masks; the ``[1:]`` body
        positions identify painted VC2 / F128 continuation slots.

    Returns
    -------
    np.ndarray
        ``u32[n_expanded_real]``.  Index ``i`` holds the raw-stream
        position for ``expanded[i + 1]``.  At painted slots the value
        is the prior slot's value + 1; at real (carrier or non-promoted
        real) slots the value comes from ``real_mask.nonzero()`` in
        encounter order.

    Notes
    -----
    Vectorised "prefix-paint mask + cumsum" implementation:

    * ``is_extra[i] = extra_value_v2_mask[i + 1] | extra_f128_mask[i + 1]``
      flags painted continuation slots.
    * ``carrier_idx_per_slot = cumsum(~is_extra) - 1`` -- at a real slot
      this advances; at a painted slot it stays put (since ``~is_extra``
      contributes 0), so ``carrier_idx_per_slot[i]`` always points to
      the most recent real slot at or before ``i``.
    * ``base = real_positions[carrier_idx_per_slot]`` -- the latest
      carrier's raw position at every slot.
    * Per-extra-run offset = "how far into a consecutive True run am
      I" (0 at non-extra, 1, 2, ... at consecutive painted slots),
      computed via ``arange - cummax(where(is_extra, -1, arange))``.
      At real slots the cummax catches up to ``arange`` (offset = 0);
      at painted slots the cummax stays anchored at the last real
      slot's index (offset = run position).  Adding ``base + offset``
      reproduces the per-source python loop's ``out[i - 1] + 1``
      recurrence without a Python-level walk.

    Caller invariant: position 0 of the expanded body (i.e.
    ``is_extra[0]``) is always False -- the first slot past the
    synthetic prepend is the body's first real carrier.  The cummax
    trick still handles ``is_extra[0] == True`` gracefully (initial
    cummax sentinel = -1, so offset at slot 0 = 1), but the caller's
    contract makes that path unreachable in practice.
    """
    if n_expanded_real == 0:
        return np.empty(0, dtype=np.uint32)

    real_positions = np.nonzero(state.real_mask)[0].astype(np.uint32)

    is_extra = (
        extra_value_v2_mask[1 : n_expanded_real + 1]
        | extra_f128_mask[1 : n_expanded_real + 1]
    )

    # ``carrier_idx_per_slot[i] = number of real slots in is_extra[:i + 1] - 1``.
    # Painted slots inherit the prior real slot's carrier index because
    # ``~is_extra`` contributes 0 to the cumsum at painted positions.
    not_extra = (~is_extra).view(np.uint8)
    carrier_idx_per_slot = np.cumsum(not_extra, dtype=np.int64) - 1

    base = real_positions[carrier_idx_per_slot]

    # Per-slot offset within the current painted run:
    #   * 0 at a real (non-painted) slot
    #   * 1, 2, ... at the 1st, 2nd, ... consecutive painted slot
    # ``arange - cummax(arange_or_minus_one_at_painted)`` produces this:
    # at real slots the cummax catches up to ``arange``, so offset = 0;
    # at painted slots the cummax stays at the last real position, so
    # offset = (slot index) - (last real slot index) = run position.
    arange_n = np.arange(n_expanded_real, dtype=np.int64)
    real_anchor = np.where(is_extra, np.int64(-1), arange_n)
    last_real_pos = np.maximum.accumulate(real_anchor)
    extra_run_offset = arange_n - last_real_pos

    return (base.astype(np.int64) + extra_run_offset).astype(np.uint32)
