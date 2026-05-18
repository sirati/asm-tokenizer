"""Tests for ``tokenizer.variant_tokens.record``.

Covers the handle-level write→memmap-read round trip and the offset
contract (write_record returns the starting offset; subsequent calls
return monotonically increasing offsets equal to file size before
the call).
"""

from __future__ import annotations

import numpy as np

from tokenizer.variant_tokens.encoder import decode_record, encode_record
from tokenizer.variant_tokens.record import read_record, write_record

from ._fakes import FakeVersionInfo, make_vocab_with


def test_write_read_roundtrip_single_record(tmp_path):
    vi = FakeVersionInfo(extra_metadata={"hardening": "full"})
    vocab = make_vocab_with(vi)
    path = tmp_path / "test_variants.bin"
    with open(path, "wb") as f:
        offset = write_record(f, vi, vocab)
    assert offset == 0  # first record always starts at 0
    # Read back via uint8 memmap and compare.
    mmap = np.memmap(path, dtype=np.uint8, mode="r")
    record = read_record(mmap, offset)
    expected = encode_record(vi, vocab)
    assert np.array_equal(record, expected)
    decoded = decode_record(record, vocab)
    assert decoded["compiler"] == "gcc"
    assert decoded["hardening"] == ["full"]


def test_write_returns_pre_write_offset(tmp_path):
    """``write_record`` returns ``handle.tell()`` BEFORE the write —
    that is the offset section-CSV cells will cite. Two consecutive
    writes must place the second record's offset equal to the byte
    length of the first."""
    vi_a = FakeVersionInfo(arch="x86_64", opt="-O2", extra_metadata={})
    vi_b = FakeVersionInfo(arch="x86_64", opt="-Os",
                           extra_metadata={"hardening": "full"})
    # Inventory both → register both → vocab knows all tokens.
    from tokenizer.variant_tokens.prefixes import build_axis_strings
    from ._fakes import FakeVocab

    vocab = FakeVocab()
    for v in (vi_a, vi_b):
        for s in build_axis_strings(v):
            vocab.register(s)

    path = tmp_path / "two_records.bin"
    with open(path, "wb") as f:
        off_a = write_record(f, vi_a, vocab)
        off_b = write_record(f, vi_b, vocab)
    assert off_a == 0
    # vi_a: n=4, 5 u16 = 10 bytes.
    assert off_b == 10
    # On-disk file is the sum of both records.
    assert path.stat().st_size == 10 + (2 + 2 * 5)

    # Both records readable from the same memmap at their offsets.
    mmap = np.memmap(path, dtype=np.uint8, mode="r")
    rec_a = read_record(mmap, off_a)
    rec_b = read_record(mmap, off_b)
    decoded_a = decode_record(rec_a, vocab)
    decoded_b = decode_record(rec_b, vocab)
    assert decoded_a["opt"] == "O2"
    assert decoded_b["opt"] == "Os"
    assert decoded_b["hardening"] == ["full"]


def test_empty_metadata_record_is_10_bytes(tmp_path):
    """Plan §"Verification" step 7: empty-metadata record is 10 bytes
    total (2 header + 4*2 axes)."""
    vi = FakeVersionInfo(extra_metadata={})
    vocab = make_vocab_with(vi)
    path = tmp_path / "empty_meta.bin"
    with open(path, "wb") as f:
        write_record(f, vi, vocab)
    assert path.stat().st_size == 10


def test_read_record_returns_uint16_view(tmp_path):
    vi = FakeVersionInfo(extra_metadata={})
    vocab = make_vocab_with(vi)
    path = tmp_path / "view_check.bin"
    with open(path, "wb") as f:
        write_record(f, vi, vocab)
    mmap = np.memmap(path, dtype=np.uint8, mode="r")
    record = read_record(mmap, 0)
    assert record.dtype == np.uint16
    # Length equals 1 + n_tokens.
    assert record.shape == (1 + int(record[0]),)
