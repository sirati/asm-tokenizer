"""Equivalence test: bulk token-span decode vs the scalar header parser.

Pins :func:`tokenizer.aligned_data.binary_format._bulk_geometry.
bulk_token_spans` to the scalar source of truth
(:func:`parse_binary_header` + :func:`record_total_size` +
:func:`extract_arrays_from_data`): for any record the bulk path must
locate exactly the token region the scalar path slices.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.binary_format import (
    RECORD_ALIGNMENT,
    BinaryHeader,
    BinaryHeaderFormat,
    encode_binary_header,
    extract_arrays_from_data,
    parse_binary_header,
    record_total_size,
)
from tokenizer.aligned_data.binary_format._bulk_geometry import (
    bulk_token_spans,
)


def _record_bytes(header: BinaryHeader, rng: np.random.Generator) -> bytes:
    """Encode one full on-disk record (header + body + alignment pad)."""
    encoded = encode_binary_header(header)
    parsed, prefix = parse_binary_header(encoded)
    total = record_total_size(parsed)
    body = bytearray(total)
    body[:prefix] = encoded
    # Fill the token region with a recognisable pattern so the test can
    # verify the bulk span points at the same bytes the scalar slicer
    # returns.
    tokens = rng.integers(0, 2**16, size=header.token_count, dtype=np.uint16)
    if header.token_count:
        body[total - 2 * header.token_count :] = tokens.tobytes()
    return bytes(body)


def _random_header(rng: np.random.Generator, entry_idx: int) -> BinaryHeader:
    """Draw a header hitting both wire forms and every token width tag."""
    form = rng.integers(0, 5)
    if form == 0:
        # Ultrashort-eligible.
        return BinaryHeader(
            format=BinaryHeaderFormat.UltraShort,
            block_enc=0,
            insn_len=int(rng.integers(0, 64)),
            block_word_count=int(rng.integers(0, 256)),
            token_count=int(rng.integers(0, 256)),
            entry_idx=entry_idx,
        )
    width_tag = int(form - 1)
    caps = (1 << 12, 1 << 20, 1 << 28, 1 << 36)
    lo = 0 if width_tag == 0 else caps[width_tag - 1]
    # Keep the buffer small: only tag 0/1 get real token payloads; the
    # wider tags are exercised at the cap boundary with token_count
    # clamped to a few thousand above the lower cap... except that
    # would balloon the buffer, so wide tags use lo + small delta.
    token_count = int(rng.integers(lo, min(caps[width_tag], lo + 2048)))
    return BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=int(rng.integers(0, 3)),
        insn_len=int(rng.integers(0, 300)),
        block_word_count=int(rng.integers(0, 1000)),
        token_count=token_count,
        entry_idx=entry_idx,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bulk_token_spans_match_scalar_parser(seed: int) -> None:
    rng = np.random.default_rng(seed)
    headers = [_random_header(rng, i) for i in range(40)]

    buf = bytearray(16)  # fake file prelude; records never start at 0
    offsets = []
    for header in headers:
        offsets.append(len(buf))
        buf += _record_bytes(header, rng)
        assert len(buf) % RECORD_ALIGNMENT == 0

    data = np.frombuffer(bytes(buf), dtype=np.uint8)
    starts, counts = bulk_token_spans(data, np.asarray(offsets))

    for i, (header, offset) in enumerate(zip(headers, offsets)):
        parsed, prefix = parse_binary_header(data[offset : offset + 14])
        total = record_total_size(parsed)
        _insn, _block, tokens = extract_arrays_from_data(
            bytes(data[offset : offset + total]), parsed, prefix
        )
        expected_start = offset + total - 2 * parsed.token_count
        assert counts[i] == parsed.token_count == header.token_count
        assert starts[i] == expected_start
        got = (
            data[starts[i] : starts[i] + 2 * counts[i]]
            .copy()
            .view(np.uint16)
        )
        assert np.array_equal(got, tokens)


def test_bulk_token_spans_empty_input() -> None:
    data = np.zeros(64, dtype=np.uint8)
    starts, counts = bulk_token_spans(data, np.zeros(0, dtype=np.int64))
    assert starts.size == 0 and counts.size == 0
    assert starts.dtype == np.int64 and counts.dtype == np.int64
