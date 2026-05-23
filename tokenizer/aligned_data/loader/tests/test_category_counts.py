"""Tests for ``category_counts.compute_category_counts``.

Construct synthetic raw_tokens streams with known per-Category
identity-carrier layouts and assert the returned dense ``{Category:
count}`` mapping matches the encoder-side invariant (caller-local ids
are dense 0..K-1; the distinct-count equals K).

The streams follow the v2 wire form: identity carriers are pinned at
vocab ids 264 (BLOCK_V2), 268 (STRING_PTR), 269 (JUMP_TABLE), 270
(RO_DATA_PTR), 271 (RW_DATA_PTR). Inline-digit bytes (caller-local id
payload) ride at ids 0..255 immediately after each carrier.

Plan reference: ``batch_decode_plan.md`` ALG-5 (identity payload
widths) + the loader-side ``category_counts`` contract on
``FunctionData.metadata``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.category_counts import (
    COUNTER_CATEGORIES,
    compute_category_counts,
)
from tokenizer.tokens import Category


# Canonical IDENTITY-block vocab ids under the unified vocab. Mirrors the
# module-under-test's internal table so the tests stay decoupled from
# its private constants.
_BLOCK_ID = 264
_STRING_PTR_ID = 268
_JUMP_TABLE_ID = 269
_RO_DATA_PTR_ID = 270
_RW_DATA_PTR_ID = 271


def _carrier_with_1byte_payload(
    carrier_id: int, caller_local_id: int
) -> list[int]:
    """v2 wire form: ``[carrier_id, <one inline-digit byte>]``."""
    assert 0 <= caller_local_id <= 0xFF
    return [carrier_id, caller_local_id]


def _carrier_with_2byte_payload(
    carrier_id: int, caller_local_id: int
) -> list[int]:
    """v2 wire form: ``[carrier_id, hi, lo]`` (big-endian payload)."""
    assert 0 <= caller_local_id <= 0xFFFF
    hi = (caller_local_id >> 8) & 0xFF
    lo = caller_local_id & 0xFF
    return [carrier_id, hi, lo]


def _carrier_zero_payload(carrier_id: int) -> list[int]:
    """v2 wire form: bare carrier with no inline-digit payload.

    The next slot in the stream is either a real-token carrier (id > 256
    via the ``value_negative`` strict boundary) or end-of-stream. The
    zero-byte payload decodes to caller-local id 0 (the encoder reserves
    that id slot for this case so no real callee collides -- plan ALG-5
    0-byte row).
    """
    return [carrier_id]


def _seal(stream: list[int]) -> np.ndarray:
    """Wrap a stream as ``uint16`` so the helper sees the production dtype."""
    return np.asarray(stream, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Core counting cases.
# ---------------------------------------------------------------------------


def test_three_unique_block_ids_yields_count_three():
    # 3 distinct caller-local ids (0, 1, 2) across 3 BLOCK carriers.
    stream = (
        _carrier_with_1byte_payload(_BLOCK_ID, 0)
        + _carrier_with_1byte_payload(_BLOCK_ID, 1)
        + _carrier_with_1byte_payload(_BLOCK_ID, 2)
    )
    counts = compute_category_counts(_seal(stream))
    assert counts[Category.BLOCK] == 3


def test_repeated_block_id_counts_distinct_only():
    # The encoder emits the SAME caller-local id multiple times when the
    # same callee is referenced repeatedly -- the distinct-count must
    # collapse those occurrences to K, not K*occurrences.
    stream = (
        _carrier_with_1byte_payload(_BLOCK_ID, 0)
        + _carrier_with_1byte_payload(_BLOCK_ID, 0)
        + _carrier_with_1byte_payload(_BLOCK_ID, 1)
        + _carrier_with_1byte_payload(_BLOCK_ID, 1)
    )
    counts = compute_category_counts(_seal(stream))
    assert counts[Category.BLOCK] == 2


def test_zero_carriers_yields_zero_for_string_ptr():
    # Stream with only BLOCK carriers; STRING_PTR (id 268) is absent.
    stream = _carrier_with_1byte_payload(_BLOCK_ID, 0)
    counts = compute_category_counts(_seal(stream))
    assert counts[Category.STRING_PTR] == 0


def test_mixed_categories_independent_counts():
    # 2 BLOCK ids (0, 1), 1 STRING_PTR (0), 3 JUMP_TABLE (0, 1, 2).
    # RO_DATA_PTR + RW_DATA_PTR absent.
    stream = (
        _carrier_with_1byte_payload(_BLOCK_ID, 0)
        + _carrier_with_1byte_payload(_STRING_PTR_ID, 0)
        + _carrier_with_1byte_payload(_BLOCK_ID, 1)
        + _carrier_with_1byte_payload(_JUMP_TABLE_ID, 0)
        + _carrier_with_1byte_payload(_JUMP_TABLE_ID, 1)
        + _carrier_with_1byte_payload(_JUMP_TABLE_ID, 2)
    )
    counts = compute_category_counts(_seal(stream))
    assert counts[Category.BLOCK] == 2
    assert counts[Category.STRING_PTR] == 1
    assert counts[Category.JUMP_TABLE] == 3
    assert counts[Category.RO_DATA_PTR] == 0
    assert counts[Category.RW_DATA_PTR] == 0


def test_all_five_counter_categories_present_in_mapping():
    # Contract: the returned dict ALWAYS has all 5 COUNTER Categories as
    # keys, regardless of which carriers actually appear in the stream.
    # Caller code expects to ``[]``-index without a key-presence dance.
    stream = _carrier_with_1byte_payload(_BLOCK_ID, 0)
    counts = compute_category_counts(_seal(stream))
    assert set(counts.keys()) == set(COUNTER_CATEGORIES)
    # Each value is a non-negative int.
    for value in counts.values():
        assert isinstance(value, int)
        assert value >= 0


def test_empty_stream_yields_all_zeros():
    counts = compute_category_counts(np.zeros(0, dtype=np.uint16))
    assert set(counts.keys()) == set(COUNTER_CATEGORIES)
    for value in counts.values():
        assert value == 0


# ---------------------------------------------------------------------------
# ALG-5 payload-width decoding cases.
# ---------------------------------------------------------------------------


def test_two_byte_payload_decodes_big_endian_u16():
    # caller-local id 0x1234 -> big-endian bytes [0x12, 0x34] after the
    # carrier. The helper must read them as a u16, not as two distinct
    # 1-byte ids.
    stream = _carrier_with_2byte_payload(_BLOCK_ID, 0x1234)
    counts = compute_category_counts(_seal(stream))
    assert counts[Category.BLOCK] == 1


def test_two_byte_payload_distinct_from_one_byte_with_same_low_byte():
    # 1-byte id 0x12 and 2-byte id 0x1234 share the same low byte but
    # are distinct caller-local ids; the helper must NOT collapse them.
    stream = (
        _carrier_with_1byte_payload(_BLOCK_ID, 0x12)
        + _carrier_with_2byte_payload(_BLOCK_ID, 0x1234)
    )
    counts = compute_category_counts(_seal(stream))
    assert counts[Category.BLOCK] == 2


def test_zero_byte_payload_at_stream_tail_decodes_to_id_zero():
    # Carrier at the LAST stream position has no p+1 slot -- treated as
    # 0-byte payload (caller-local id 0). Compose with a leading carrier
    # carrying id 1 so the distinct-count is 2.
    stream = (
        _carrier_with_1byte_payload(_BLOCK_ID, 1)
        + _carrier_zero_payload(_BLOCK_ID)
    )
    counts = compute_category_counts(_seal(stream))
    assert counts[Category.BLOCK] == 2


# ---------------------------------------------------------------------------
# Cross-Category isolation.
# ---------------------------------------------------------------------------


def test_carriers_of_different_categories_dont_interfere():
    # Same caller-local id 0 across all 5 COUNTER Categories must
    # produce count 1 in each (not 5 in one).
    stream = (
        _carrier_with_1byte_payload(_BLOCK_ID, 0)
        + _carrier_with_1byte_payload(_STRING_PTR_ID, 0)
        + _carrier_with_1byte_payload(_JUMP_TABLE_ID, 0)
        + _carrier_with_1byte_payload(_RO_DATA_PTR_ID, 0)
        + _carrier_with_1byte_payload(_RW_DATA_PTR_ID, 0)
    )
    counts = compute_category_counts(_seal(stream))
    for category in COUNTER_CATEGORIES:
        assert counts[category] == 1, (
            f"{category.name} count={counts[category]} (expected 1)"
        )


# ---------------------------------------------------------------------------
# Defensive invariant.
# ---------------------------------------------------------------------------


def test_three_byte_identity_payload_raises():
    # ALG-5 restricts identity payloads to 0/1/2 bytes. A 3-byte run
    # following an identity carrier is a v2-codec violation; the helper
    # surfaces it as an AssertionError so the diagnostic stays local.
    stream = [_BLOCK_ID, 0, 1, 2]  # carrier + 3 inline-digit bytes
    with pytest.raises(AssertionError):
        compute_category_counts(_seal(stream))
