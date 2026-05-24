"""Unit tests for sorted_index._wire (encode_sorted_index + parse_header).

Covers:

* Round-trip equivalence: ``encode -> parse -> reconstruct lengths ->
  re-encode`` produces byte-identical output.
* Edge cases: empty, single section, all-same length, contiguous
  lengths, sparse lengths with internal gaps, max-u32 length.
* Explicit little-endian assertion via a tiny hex-known input.
* Body size invariant: ``num_sections * 4`` bytes.
* Stable-sort invariant: ties preserve input order in the body.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tokenizer.aligned_data.sorted_index import (
    encode_sorted_index,
    parse_header,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reconstruct_lengths_from_blob(blob: bytes) -> np.ndarray:
    """Decode a blob back into the original ``lengths`` array.

    Reverses ``encode_sorted_index``: walks the body (sorted section
    indices) along with the per-bucket count slots to recover the
    original-order length per section. Useful for the round-trip test
    where re-encoding the reconstructed array must produce identical
    bytes.
    """
    min_length, counts, body_offset = parse_header(blob)
    num_lengths = counts.size
    if num_lengths == 0:
        return np.empty(0, dtype=np.uint32)
    body = np.frombuffer(
        blob, dtype=np.uint32, count=int(counts.sum()), offset=body_offset,
    )
    lengths = np.empty(body.size, dtype=np.uint32)
    cursor = 0
    for bucket_idx, bucket_count in enumerate(counts):
        bc = int(bucket_count)
        if bc == 0:
            continue
        bucket_indices = body[cursor : cursor + bc]
        lengths[bucket_indices] = min_length + bucket_idx
        cursor += bc
    return lengths


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lengths",
    [
        np.array([5, 2, 8, 1, 5, 3], dtype=np.uint32),
        np.array([100, 100, 100, 100], dtype=np.uint32),
        np.array([1], dtype=np.uint32),
        np.array([7, 7, 9, 7, 12, 9], dtype=np.uint32),
        np.arange(10, dtype=np.uint32),
        # sparse with gaps inside the range
        np.array([0, 10, 0, 20, 10, 30], dtype=np.uint32),
    ],
)
def test_round_trip_byte_equivalence(lengths: np.ndarray) -> None:
    blob = encode_sorted_index(lengths)
    reconstructed = _reconstruct_lengths_from_blob(blob)
    np.testing.assert_array_equal(reconstructed, lengths)
    re_encoded = encode_sorted_index(reconstructed)
    assert re_encoded == blob


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_eight_byte_header() -> None:
    blob = encode_sorted_index(np.empty(0, dtype=np.uint32))
    assert blob == struct.pack("<II", 0, 0)
    min_length, counts, body_offset = parse_header(blob)
    assert min_length == 0
    assert counts.size == 0
    assert counts.dtype == np.uint32
    assert body_offset == 8


def test_single_section() -> None:
    lengths = np.array([42], dtype=np.uint32)
    blob = encode_sorted_index(lengths)
    min_length, counts, body_offset = parse_header(blob)
    assert min_length == 42
    assert counts.tolist() == [1]
    assert body_offset == 12
    body = np.frombuffer(blob, dtype=np.uint32, offset=body_offset)
    assert body.tolist() == [0]


def test_all_same_length() -> None:
    lengths = np.array([5, 5, 5, 5, 5], dtype=np.uint32)
    blob = encode_sorted_index(lengths)
    min_length, counts, body_offset = parse_header(blob)
    assert min_length == 5
    assert counts.tolist() == [5]
    body = np.frombuffer(blob, dtype=np.uint32, offset=body_offset)
    # stable sort: ties keep original order
    assert body.tolist() == [0, 1, 2, 3, 4]


def test_contiguous_lengths_no_gaps() -> None:
    lengths = np.array([3, 4, 5, 6, 7], dtype=np.uint32)
    blob = encode_sorted_index(lengths)
    min_length, counts, body_offset = parse_header(blob)
    assert min_length == 3
    assert counts.tolist() == [1, 1, 1, 1, 1]


def test_sparse_lengths_with_gaps() -> None:
    # min=2, max=10 -> 9 buckets total, several with count=0
    lengths = np.array([2, 10, 2, 5], dtype=np.uint32)
    blob = encode_sorted_index(lengths)
    min_length, counts, body_offset = parse_header(blob)
    assert min_length == 2
    assert counts.size == 10 - 2 + 1
    # length=2 -> idx 0 -> 2; length=5 -> idx 3 -> 1; length=10 -> idx 8 -> 1
    expected = [0] * counts.size
    expected[0] = 2
    expected[3] = 1
    expected[8] = 1
    assert counts.tolist() == expected
    assert int(counts.sum()) == lengths.size


def test_max_u32_length_value() -> None:
    max_u32 = np.uint32(0xFFFFFFFF)
    lengths = np.array([max_u32], dtype=np.uint32)
    blob = encode_sorted_index(lengths)
    min_length, counts, body_offset = parse_header(blob)
    assert min_length == int(max_u32)
    assert counts.tolist() == [1]
    body = np.frombuffer(blob, dtype=np.uint32, offset=body_offset)
    assert body.tolist() == [0]


# ---------------------------------------------------------------------------
# Endianness
# ---------------------------------------------------------------------------


def test_explicit_little_endian_hex_round_trip() -> None:
    # Two sections, lengths [4, 3] -> sorted as [3, 4], min=3, num_lengths=2,
    # counts=[1, 1], body=[1, 0] (the section originally at index 1 has
    # length 3 and comes first).
    lengths = np.array([4, 3], dtype=np.uint32)
    blob = encode_sorted_index(lengths)
    expected = (
        # min_length=3 (LE u32)
        b"\x03\x00\x00\x00"
        # num_lengths=2 (LE u32)
        b"\x02\x00\x00\x00"
        # counts: 1, 1 (LE u32 each)
        b"\x01\x00\x00\x00\x01\x00\x00\x00"
        # body: 1, 0 (LE u32 each)
        b"\x01\x00\x00\x00\x00\x00\x00\x00"
    )
    assert blob == expected


# ---------------------------------------------------------------------------
# Body shape + stable-sort invariants
# ---------------------------------------------------------------------------


def test_body_size_is_num_sections_times_four() -> None:
    lengths = np.array([5, 2, 8, 1, 5, 3, 12], dtype=np.uint32)
    blob = encode_sorted_index(lengths)
    _min_length, _counts, body_offset = parse_header(blob)
    body_bytes = blob[body_offset:]
    assert len(body_bytes) == lengths.size * 4


def test_stable_sort_preserves_input_order_on_duplicates() -> None:
    # Three sections all of length 7, plus distractors with length 1 and 9.
    # Stable sort must keep the length-7 indices in their original order:
    # original indices 1, 3, 5 -> body should list them in that order.
    lengths = np.array([1, 7, 9, 7, 1, 7], dtype=np.uint32)
    blob = encode_sorted_index(lengths)
    _min_length, _counts, body_offset = parse_header(blob)
    body = np.frombuffer(
        blob, dtype=np.uint32, offset=body_offset, count=lengths.size,
    )
    # length=1 first (indices 0, 4 in original order),
    # then length=7 (indices 1, 3, 5),
    # then length=9 (index 2).
    assert body.tolist() == [0, 4, 1, 3, 5, 2]


# ---------------------------------------------------------------------------
# Header dtype / view properties
# ---------------------------------------------------------------------------


def test_parse_header_counts_view_is_u32_and_correct_length() -> None:
    lengths = np.array([2, 4, 2, 6], dtype=np.uint32)
    blob = encode_sorted_index(lengths)
    _min_length, counts, body_offset = parse_header(blob)
    assert counts.dtype == np.uint32
    # min=2, max=6 -> num_lengths=5
    assert counts.size == 5
    assert body_offset == 8 + 4 * 5
