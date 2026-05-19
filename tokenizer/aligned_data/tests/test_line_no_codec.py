"""Round-trip + invariants for the compact base64 line-no codec."""

from __future__ import annotations

import random

import pytest

from tokenizer.aligned_data.line_no_codec import (
    decode_line_no,
    decode_line_nos_csv,
    encode_line_no,
    encode_line_nos_csv,
)


def test_encode_one_is_aq():
    # b"\x01" -> urlsafe_b64 "AQ==" -> stripped "AQ"
    assert encode_line_no(1) == "AQ"
    assert decode_line_no("AQ") == 1


def test_encode_rejects_zero_and_negative():
    with pytest.raises(ValueError):
        encode_line_no(0)
    with pytest.raises(ValueError):
        encode_line_no(-1)


def test_round_trip_1000_random_line_numbers():
    rng = random.Random(0xDECAFBAD)
    samples = [rng.randint(1, 2**32 - 1) for _ in range(1000)]
    # also throw in the extreme boundaries to exercise edge cases
    samples.extend([1, 2, 255, 256, 65535, 65536, 2**24 - 1, 2**24, 2**32 - 1])
    for n in samples:
        s = encode_line_no(n)
        assert decode_line_no(s) == n, n


def test_encode_csv_round_trip():
    nos = [1, 2, 3]
    s = encode_line_nos_csv(nos)
    assert s == "AQ,Ag,Aw"
    assert decode_line_nos_csv(s) == nos


def test_decode_empty_csv_returns_empty_list():
    assert decode_line_nos_csv("") == []


def test_encode_csv_random_large_round_trip():
    rng = random.Random(0xFEEDC0DE)
    nos = [rng.randint(1, 2**40 - 1) for _ in range(50)]
    s = encode_line_nos_csv(nos)
    assert decode_line_nos_csv(s) == nos


def test_no_padding_emitted():
    # Spot-check across a wide span: no encoded value ever carries '='.
    rng = random.Random(0xBADF00D)
    for _ in range(500):
        n = rng.randint(1, 2**48 - 1)
        s = encode_line_no(n)
        assert "=" not in s, (n, s)


def test_encoded_form_is_shortest_possible():
    # For each n, encoding uses exactly ceil(bit_len/8) bytes (1 minimum)
    # and the base64 length is ceil(byte_len * 4 / 3) with '=' stripped.
    # That's the invariant: encoding length matches the minimum byte form.
    for n in [1, 2, 127, 128, 255, 256, 16383, 16384, 2**31, 2**40 - 1]:
        byte_len = (n.bit_length() + 7) // 8 or 1
        expected_b64_len = (byte_len * 4 + 2) // 3  # ceil(byte_len*4/3)
        assert len(encode_line_no(n)) == expected_b64_len, (
            n,
            byte_len,
            len(encode_line_no(n)),
            expected_b64_len,
        )


def test_csv_preserves_order_and_duplicates():
    nos = [5, 1, 5, 2, 1, 65535]
    s = encode_line_nos_csv(nos)
    assert decode_line_nos_csv(s) == nos
