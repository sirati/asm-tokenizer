"""Tests for ``tokenizer.aligned_data.binary_format``.

Covers the new packed-control-byte header layout, the ``compute_pad``
pure function, and the ``IndexEntrySkip`` overflow guards.
"""

from __future__ import annotations

import itertools

import pytest

from tokenizer.aligned_data.binary_format import (
    HEADER_BYTES,
    OVERLONG_FIELD_BYTES,
    BinaryHeader,
    IndexEntrySkip,
    compute_pad,
    encode_binary_header,
    parse_binary_header,
)


# ---------------------------------------------------------------------------
# Header round-trip (every (block_enc, pad_size) combination).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_enc", [0, 1, 2])
@pytest.mark.parametrize("pad_size", [0, 1, 2, 3])
def test_header_round_trip_all_packed_combinations(block_enc, pad_size):
    insn_len = 12345
    block_len = 678
    encoded = encode_binary_header(insn_len, block_enc, block_len, pad_size)
    assert len(encoded) == HEADER_BYTES

    parsed = parse_binary_header(encoded)
    assert parsed == BinaryHeader(
        insn_len=insn_len,
        block_enc=block_enc,
        block_len=block_len,
        pad_size=pad_size,
    )


def test_header_layout_packed_byte_first():
    """Wire layout: packed byte at 0, insn_len u24 at 1-3, block_len u16 at 4-5."""
    encoded = encode_binary_header(
        insn_len=0x010203, block_enc=2, block_len=0x0405, pad_size=3
    )
    # packed = (2 & 0b11) | ((3 & 0b11) << 2) = 2 | 12 = 14 = 0x0E
    assert encoded[0] == 0x0E
    # little-endian u24 of 0x010203 = bytes 03 02 01
    assert encoded[1:4] == b"\x03\x02\x01"
    # little-endian u16 of 0x0405 = bytes 05 04
    assert encoded[4:6] == b"\x05\x04"


def test_header_min_max_field_values_round_trip():
    """Edges within cap: insn_len just under u24, block_len just under u16."""
    encoded = encode_binary_header(
        insn_len=(1 << 24) - 1,
        block_enc=2,
        block_len=(1 << 16) - 1,
        pad_size=3,
    )
    parsed = parse_binary_header(encoded)
    assert parsed.insn_len == (1 << 24) - 1
    assert parsed.block_len == (1 << 16) - 1
    assert parsed.block_enc == 2
    assert parsed.pad_size == 3


# ---------------------------------------------------------------------------
# compute_pad: broad sweep + invariants.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_overlong", [False, True])
@pytest.mark.parametrize("token_count", [0, 1, 3, 17, 2048])
@pytest.mark.parametrize("block_len", [0, 1, 5, 31, 4096])
@pytest.mark.parametrize("insn_len", [0, 1, 7, 23, 1024])
def test_compute_pad_invariants(insn_len, block_len, token_count, is_overlong):
    pad = compute_pad(insn_len, block_len, token_count, is_overlong)
    assert pad in (0, 1, 2, 3)
    body_prefix = HEADER_BYTES + (OVERLONG_FIELD_BYTES if is_overlong else 0)
    unpadded_total = body_prefix + insn_len + block_len + 2 * token_count
    assert (unpadded_total + pad) % 4 == 0


def test_compute_pad_minimum_padding():
    """The pad is the SMALLEST value in [0,3] that aligns the record.

    No other value in [0,3] should both align AND be smaller. We check
    that anything smaller fails to align (i.e. the function picks the
    minimum, not just *some* aligning value).
    """
    cases = itertools.product(
        [0, 1, 2, 3, 5, 9, 17],
        [0, 1, 2, 3, 7, 11, 100],
        [0, 1, 2, 3, 50],
        [False, True],
    )
    for insn_len, block_len, token_count, is_overlong in cases:
        pad = compute_pad(insn_len, block_len, token_count, is_overlong)
        for smaller in range(pad):
            body_prefix = HEADER_BYTES + (
                OVERLONG_FIELD_BYTES if is_overlong else 0
            )
            unpadded_total = (
                body_prefix + insn_len + block_len + 2 * token_count
            )
            assert (unpadded_total + smaller) % 4 != 0


# ---------------------------------------------------------------------------
# Overflow guards: IndexEntrySkip.
# ---------------------------------------------------------------------------


def test_insn_len_overflow_raises_index_entry_skip():
    with pytest.raises(IndexEntrySkip) as excinfo:
        encode_binary_header(
            insn_len=1 << 24, block_enc=0, block_len=0, pad_size=0
        )
    assert excinfo.value.reason == "insn_len_overflow"
    assert excinfo.value.value == 1 << 24


def test_block_len_overflow_raises_index_entry_skip():
    with pytest.raises(IndexEntrySkip) as excinfo:
        encode_binary_header(
            insn_len=0, block_enc=0, block_len=1 << 16, pad_size=0
        )
    assert excinfo.value.reason == "block_len_overflow"
    assert excinfo.value.value == 1 << 16


def test_insn_len_overflow_priority_over_block_len():
    """When both overflow, insn_len is checked first — this stays
    stable so error logs are deterministic on multi-overflow inputs.
    """
    with pytest.raises(IndexEntrySkip) as excinfo:
        encode_binary_header(
            insn_len=1 << 24,
            block_enc=0,
            block_len=1 << 16,
            pad_size=0,
        )
    assert excinfo.value.reason == "insn_len_overflow"


# ---------------------------------------------------------------------------
# Domain-error guards: ValueError on bad block_enc / pad_size / reserved bits.
# ---------------------------------------------------------------------------


def test_block_enc_out_of_domain_raises_value_error():
    with pytest.raises(ValueError):
        encode_binary_header(insn_len=0, block_enc=3, block_len=0, pad_size=0)


def test_pad_size_out_of_domain_raises_value_error():
    with pytest.raises(ValueError):
        encode_binary_header(insn_len=0, block_enc=0, block_len=0, pad_size=4)


def test_parse_rejects_reserved_bits_set():
    """Reserved bits 4-7 in the packed byte are write-zero; the parser
    refuses anything else so a future packed-byte extension can't be
    silently mis-decoded by an old reader.
    """
    encoded = bytearray(
        encode_binary_header(insn_len=4, block_enc=1, block_len=2, pad_size=2)
    )
    encoded[0] |= 0b00010000  # set a reserved bit
    with pytest.raises(ValueError):
        parse_binary_header(bytes(encoded))
