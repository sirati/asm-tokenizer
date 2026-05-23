"""Unit-tests for ``decoded._inline_decode_state.build_inline_decode_state``.

Pins the four fields the downstream consumers (identity arm, number
arm, postfix-invariant check) read from:

* ``runlen_number`` / ``runlen_value`` agree with ``run_lengths`` on
  the input masks.
* ``carries_inline_mask`` is True exactly at ``raw_tokens`` in the
  carrier band ``[257, 272)``.
* ``is_negative_per_position`` matches a brute-force Python reference
  on a multi-source fixture.
* ``format_version != 1`` raises (unified-only contract).

Single source of truth for the carrier-band + sign-marker layout is
:class:`VocabularyManager`; the tests below pull constants from the
state module so any layout drift surfaces here loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    _V2_EAGER_BLOCK_END,
    _V2_RESERVED_DIGIT_COUNT,
    _V2_VALUE_NEGATIVE_TOKEN_ID,
    build_inline_decode_state,
)
from tokenizer.aligned_data.loader.decoded.run_lengths import run_lengths


def _u16(*tokens: int) -> np.ndarray:
    return np.array(tokens, dtype=np.uint16)


def _brute_force_is_negative(raw_tokens: np.ndarray) -> np.ndarray:
    """Reference: per-carrier, walk the immediate inline run and check
    whether the slot right after the run holds the sign marker.

    Mirrors the per-source Python peek the vectorized refactor replaces
    so a divergence between the two surfaces as a test failure.
    """
    n = int(raw_tokens.shape[0])
    out = np.zeros(n, dtype=bool)
    for p in range(n):
        token = int(raw_tokens[p])
        is_carrier = (
            token > _V2_VALUE_NEGATIVE_TOKEN_ID and token < _V2_EAGER_BLOCK_END
        )
        if not is_carrier:
            continue
        cursor = p + 1
        while cursor < n and int(raw_tokens[cursor]) < _V2_RESERVED_DIGIT_COUNT:
            cursor += 1
        if cursor < n and int(raw_tokens[cursor]) == _V2_VALUE_NEGATIVE_TOKEN_ID:
            out[p] = True
    return out


def test_runlength_fields_match_run_lengths_on_masks() -> None:
    raw = _u16(257, 100, 5, 9, 256, 261, 7, 264, 4)
    state = build_inline_decode_state(raw, format_version=1)
    number_mask = raw < _V2_RESERVED_DIGIT_COUNT
    value_mask = raw <= _V2_VALUE_NEGATIVE_TOKEN_ID
    np.testing.assert_array_equal(state.runlen_number, run_lengths(number_mask))
    np.testing.assert_array_equal(state.runlen_value, run_lengths(value_mask))


def test_carries_inline_mask_is_carrier_band() -> None:
    """``carries_inline_mask`` is True iff ``raw_tokens`` in [257, 272)."""
    raw = _u16(
        257,  # carrier (lowest band id)
        100,
        272,  # NOT carrier (just above band)
        264,  # carrier
        5,
        256,  # sign marker -- not carrier
        300,  # NOT carrier (above band)
    )
    state = build_inline_decode_state(raw, format_version=1)
    expected = (raw >= _V2_RESERVED_DIGIT_COUNT + 1) & (raw < _V2_EAGER_BLOCK_END)
    # Equivalent: raw > 256 & raw < 272
    np.testing.assert_array_equal(state.carries_inline_mask, expected)


def test_is_negative_per_position_matches_brute_force_reference() -> None:
    """Multi-source fixture: VC2 negative, FLOAT16 positive, BLOCK_V2 ident."""
    # [VC2, payload, payload, payload, value_negative, FLOAT16, payload,
    #  BLOCK_V2, payload]
    raw = _u16(257, 100, 5, 9, 256, 261, 7, 264, 4)
    state = build_inline_decode_state(raw, format_version=1)
    expected = _brute_force_is_negative(raw)
    np.testing.assert_array_equal(state.is_negative_per_position, expected)


def test_is_negative_worked_example_from_plan() -> None:
    """Plan worked example: only the VC2 source at position 0 is negative."""
    raw = _u16(257, 100, 5, 9, 256, 261, 7, 264, 4)
    state = build_inline_decode_state(raw, format_version=1)
    assert bool(state.is_negative_per_position[0]) is True
    # Every other position is False (carrier positions are 5 and 7;
    # neither has a sign postfix).
    for idx in range(1, raw.shape[0]):
        assert bool(state.is_negative_per_position[idx]) is False, (
            f"position {idx} unexpectedly negative"
        )


def test_carrier_at_tail_position_defaults_to_false() -> None:
    """A carrier at the LAST position has no p+1 slot -> sign stays False."""
    raw = _u16(264)  # identity-block carrier as the only token
    state = build_inline_decode_state(raw, format_version=1)
    assert bool(state.is_negative_per_position[0]) is False


@pytest.mark.parametrize("bad_version", [0, 2, 3, -1, 100])
def test_format_version_other_than_one_raises(bad_version: int) -> None:
    raw = _u16(264, 4)
    with pytest.raises(AssertionError, match="format_version=1"):
        build_inline_decode_state(raw, format_version=bad_version)
