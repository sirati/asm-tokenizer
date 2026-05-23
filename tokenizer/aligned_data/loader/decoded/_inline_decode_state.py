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


__all__ = ["InlineDecodeState", "build_inline_decode_state"]


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
    :func:`_decode_to_staging`; every consumer reads from these fields
    rather than rebuilding masks.  All arrays are length ``N`` where
    ``N = raw_tokens.shape[0]`` and are aligned position-by-position
    with the input stream.

    Field semantics:

    * ``raw_tokens``: the original uint16 stream (NOT a copy -- consumers
      MUST NOT mutate it; the working buffer in :func:`_decode_to_staging`
      is the only place mutation is allowed).
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
    """

    raw_tokens: np.ndarray
    real_mask: np.ndarray
    number_mask: np.ndarray
    runlen_number: np.ndarray
    runlen_value: np.ndarray
    carries_inline_mask: np.ndarray
    is_negative_per_position: np.ndarray


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

    return InlineDecodeState(
        raw_tokens=raw_tokens,
        real_mask=real_mask,
        number_mask=number_mask,
        runlen_number=runlen_number,
        runlen_value=runlen_value,
        carries_inline_mask=carries_inline_mask,
        is_negative_per_position=is_negative_per_position,
    )
