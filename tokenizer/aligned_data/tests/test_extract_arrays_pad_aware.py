"""Reader-side pad awareness for the variable-width record header.

Hand-packs record bytes (variable-width header + insn + pre_pad + block
+ post_pad + tokens) and asserts that ``extract_arrays_from_data``
slices the correct ndarrays back out across every pad placement
exercised by the new rule. Pure parsing -- no calls into the writer
side beyond ``encode_binary_header`` (the *header* writer is part of
this module's surface and is the only correct way to produce the
control byte; the body is laid out by hand to keep the test
self-contained).

Splits independently of the (now-deleted) overlong escape; the variable
width header subsumes that case.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.binary_format import (
    BLOCK_WORD_SIZE,
    RECORD_ALIGNMENT,
    ULTRASHORT_BLOCK_CAP,
    ULTRASHORT_INSN_CAP,
    ULTRASHORT_TOKENS_CAP,
    BinaryHeader,
    BinaryHeaderFormat,
    derive_pad_placement,
    encode_binary_header,
    extract_arrays_from_data,
    parse_binary_header,
    prefix_bytes_for_header,
    record_total_size,
)


_BLOCK_DTYPES = (np.uint8, np.uint16, np.uint32)


def _pack_record(
    insn: np.ndarray,
    block: np.ndarray,
    tokens: np.ndarray,
    *,
    force_normal: bool,
) -> bytes:
    """Hand-pack one record matching the new variable-width layout.

    ``force_normal`` lets the test exercise the normal-form layout even
    when the field tuple would otherwise be eligible for ultrashort
    (e.g. when block_enc != 0). When the tuple is *only* ultrashort-
    eligible (block_enc=0 + small fields), ``force_normal=False`` is
    required since the encoder canonicalises to ultrashort.
    """
    block_enc = _BLOCK_DTYPES.index(block.dtype.type)
    insn_len = len(insn)
    block_word_count = len(block)
    token_count = len(tokens)

    fmt = (
        BinaryHeaderFormat.UltraShort
        if (block_enc == 0 and not force_normal)
        else BinaryHeaderFormat.Normal
    )
    if force_normal and block_enc == 0:
        # Force normal by bumping insn_len above ultrashort cap. (Caller
        # who wants force_normal with block_enc=0 should hand us an
        # already-large insn array.)
        assert insn_len >= ULTRASHORT_INSN_CAP or block_word_count >= ULTRASHORT_BLOCK_CAP \
            or token_count >= ULTRASHORT_TOKENS_CAP, (
            "force_normal with block_enc=0 needs at least one field >= ultrashort cap"
        )

    header = BinaryHeader(
        format=fmt,
        block_enc=block_enc,
        insn_len=insn_len,
        block_word_count=block_word_count,
        token_count=token_count,
        entry_idx=0,
    )
    header_bytes = encode_binary_header(header)
    pre, post = derive_pad_placement(header)
    return (
        header_bytes
        + insn.astype(np.uint8).tobytes()
        + b"\x00" * pre
        + block.tobytes()
        + b"\x00" * post
        + tokens.astype(np.uint16).tobytes()
    )


def _expect_arrays_equal(got, insn, block, tokens):
    got_insn, got_block, got_tokens = got
    np.testing.assert_array_equal(got_insn, insn.astype(np.uint8))
    assert got_block.dtype == block.dtype
    np.testing.assert_array_equal(got_block, block)
    assert got_tokens.dtype == np.uint16
    np.testing.assert_array_equal(got_tokens, tokens.astype(np.uint16))


def _sample_arrays(block_enc: int):
    insn = np.arange(7, dtype=np.uint8)
    block_dtype = _BLOCK_DTYPES[block_enc]
    block = np.arange(1, 6, dtype=block_dtype) * np.array(1, dtype=block_dtype)
    tokens = np.array([10, 20, 30, 40], dtype=np.uint16)
    return insn, block, tokens


# ---------------------------------------------------------------------------
# Normal record, every block_enc, across pad placements.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_enc", [0, 1, 2])
def test_normal_record_round_trip_per_block_enc(block_enc):
    """Normal records of each block dtype round-trip through extract."""
    insn, block, tokens = _sample_arrays(block_enc)
    # Bulk insn up past the ultrashort cap so the encoder picks normal.
    insn = np.arange(ULTRASHORT_INSN_CAP + 3, dtype=np.uint8)
    record = _pack_record(insn, block, tokens, force_normal=True)

    header, prefix = parse_binary_header(record)
    assert header.format is BinaryHeaderFormat.Normal
    assert header.block_enc == block_enc

    arrays = extract_arrays_from_data(record, header, prefix)
    _expect_arrays_equal(arrays, insn, block, tokens)


@pytest.mark.parametrize("insn_len_mod", [0, 1, 2, 3])
def test_normal_record_block_enc_u32_alignment(insn_len_mod):
    """Block words land on a 4-byte boundary when the B <= P branch applies."""
    insn = np.arange(ULTRASHORT_INSN_CAP + insn_len_mod, dtype=np.uint8)
    block = np.array([0xDEADBEEF, 0xCAFEBABE], dtype=np.uint32)
    tokens = np.array([1, 2, 3], dtype=np.uint16)
    record = _pack_record(insn, block, tokens, force_normal=True)

    header, prefix = parse_binary_header(record)
    arrays = extract_arrays_from_data(record, header, prefix)
    _expect_arrays_equal(arrays, insn, block, tokens)

    # When B <= P (the common case), the parsed block array's start
    # offset within the record is a multiple of block_word_size.
    pre, _ = derive_pad_placement(header)
    block_start = prefix + header.insn_len + pre
    assert (
        block_start % BLOCK_WORD_SIZE[header.block_enc] == 0
        or pre == (-(prefix + header.insn_len + header.block_word_count * 4
                     + 2 * header.token_count)) % RECORD_ALIGNMENT
    ), "block start either aligned, or B > P fallback engaged"


# ---------------------------------------------------------------------------
# Ultrashort record round-trip.
# ---------------------------------------------------------------------------


def test_ultrashort_record_round_trip():
    insn = np.arange(5, dtype=np.uint8)
    block = np.arange(1, 4, dtype=np.uint8)
    tokens = np.array([100, 200, 300], dtype=np.uint16)
    record = _pack_record(insn, block, tokens, force_normal=False)

    header, prefix = parse_binary_header(record)
    assert header.format is BinaryHeaderFormat.UltraShort
    assert header.block_enc == 0
    assert prefix == 7

    arrays = extract_arrays_from_data(record, header, prefix)
    _expect_arrays_equal(arrays, insn, block, tokens)


# ---------------------------------------------------------------------------
# Memmap path: synthetic bin with a mix of records.
# ---------------------------------------------------------------------------


def test_extract_arrays_via_memmap_mixed_records(tmp_path):
    """Pack one ultrashort + three normal records (one per non-zero block_enc),
    mmap the file, and decode each at its offset."""
    payloads = []  # (insn, block, tokens, force_normal)

    # Ultrashort.
    payloads.append((
        np.arange(4, dtype=np.uint8),
        np.array([1, 2], dtype=np.uint8),
        np.array([7], dtype=np.uint16),
        False,
    ))
    # Normal, block_enc=0 (forced by large insn).
    payloads.append((
        np.arange(ULTRASHORT_INSN_CAP + 5, dtype=np.uint8),
        np.array([1, 2, 3], dtype=np.uint8),
        np.array([4, 5, 6], dtype=np.uint16),
        True,
    ))
    # Normal, block_enc=1.
    payloads.append((
        np.arange(10, dtype=np.uint8),
        np.array([0x1111, 0x2222, 0x3333], dtype=np.uint16),
        np.array([0xAAAA, 0xBBBB], dtype=np.uint16),
        False,  # block_enc != 0 -> always normal
    ))
    # Normal, block_enc=2.
    payloads.append((
        np.arange(13, dtype=np.uint8),
        np.array([0xDEADBEEF, 0xCAFEBABE], dtype=np.uint32),
        np.array([1, 2, 3, 4, 5], dtype=np.uint16),
        False,
    ))

    bin_path = tmp_path / "synthetic_data.bin"
    offsets = []
    with open(bin_path, "wb") as fh:
        for insn, block, tokens, fn in payloads:
            offsets.append(fh.tell())
            fh.write(_pack_record(insn, block, tokens, force_normal=fn))

    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")
    for (insn, block, tokens, _), off in zip(payloads, offsets):
        header, prefix = parse_binary_header(mmap[off:off + 14])
        # Slice to record_total_size so the extractor stays in bounds.
        end = off + record_total_size(header)
        arrays = extract_arrays_from_data(mmap[off:end], header, prefix)
        _expect_arrays_equal(arrays, insn, block, tokens)


# ---------------------------------------------------------------------------
# Zero-copy guarantee: arrays returned by extract must share memory with
# the underlying buffer (memmap or bytes via np.frombuffer).
# ---------------------------------------------------------------------------


def test_extract_arrays_views_share_memory_with_memmap(tmp_path):
    insn = np.arange(10, dtype=np.uint8)
    block = np.array([1, 2, 3], dtype=np.uint16)
    tokens = np.array([42, 43], dtype=np.uint16)
    record = _pack_record(insn, block, tokens, force_normal=False)

    bin_path = tmp_path / "synthetic.bin"
    bin_path.write_bytes(record)
    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")

    header, prefix = parse_binary_header(mmap[:14])
    end = record_total_size(header)
    insn_view, block_view, tokens_view = extract_arrays_from_data(
        mmap[:end], header, prefix
    )
    for arr in (insn_view, block_view, tokens_view):
        assert np.shares_memory(arr, mmap), (
            f"array {arr.dtype}/{arr.shape} does not share memory with memmap"
        )


def test_extract_arrays_bytes_ndarray_memmap_agree(tmp_path):
    """bytes, np.ndarray, and np.memmap inputs must produce identical
    arrays for the same record."""
    insn = np.arange(8, dtype=np.uint8)
    block = np.array([0xAB, 0xCD], dtype=np.uint8)
    tokens = np.array([0xFEED, 0xFACE], dtype=np.uint16)
    record = _pack_record(insn, block, tokens, force_normal=False)

    bin_path = tmp_path / "synthetic.bin"
    bin_path.write_bytes(record)

    header, prefix = parse_binary_header(record)
    arrays_bytes = extract_arrays_from_data(record, header, prefix)
    arrays_ndarray = extract_arrays_from_data(
        np.frombuffer(record, dtype=np.uint8), header, prefix
    )
    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")
    arrays_memmap = extract_arrays_from_data(
        mmap[: record_total_size(header)], header, prefix
    )

    for got in (arrays_bytes, arrays_ndarray, arrays_memmap):
        _expect_arrays_equal(got, insn, block, tokens)

    # Bytes-for-bytes cross-check on dtype.
    for i in range(3):
        assert arrays_bytes[i].dtype == arrays_ndarray[i].dtype == arrays_memmap[i].dtype


# ---------------------------------------------------------------------------
# Pad-region invariants: pre_pad and post_pad regions are zeroed on a
# round-tripped record (the body-layout code in _pack_record writes
# zeros; this pins down that the reader's prefix arithmetic agrees with
# `derive_pad_placement`).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_enc", [1, 2])
@pytest.mark.parametrize("insn_len_mod", [0, 1, 2, 3])
def test_pad_regions_are_zero_on_packed_record(block_enc, insn_len_mod):
    insn = np.arange(ULTRASHORT_INSN_CAP + insn_len_mod, dtype=np.uint8)
    block = np.arange(1, 5, dtype=_BLOCK_DTYPES[block_enc])
    tokens = np.array([1, 2, 3], dtype=np.uint16)
    record = _pack_record(insn, block, tokens, force_normal=True)

    header, prefix = parse_binary_header(record)
    pre, post = derive_pad_placement(header)
    insn_end = prefix + header.insn_len
    block_start = insn_end + pre
    block_bytes = header.block_word_count * BLOCK_WORD_SIZE[block_enc]
    block_end = block_start + block_bytes
    tokens_start = block_end + post

    assert record[insn_end:block_start] == b"\x00" * pre
    assert record[block_end:tokens_start] == b"\x00" * post
    assert len(record) == record_total_size(header)
    assert len(record) % RECORD_ALIGNMENT == 0


# ---------------------------------------------------------------------------
# UltraShort + every (block_word_count, insn_len, token_count) within caps:
# ensure the encoder/decoder/extract pipeline never drops a byte.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("insn_len", [0, 1, ULTRASHORT_INSN_CAP - 1])
@pytest.mark.parametrize("block_word_count", [0, 1, ULTRASHORT_BLOCK_CAP - 1])
@pytest.mark.parametrize("token_count", [0, 1, ULTRASHORT_TOKENS_CAP - 1])
def test_ultrashort_full_sweep(insn_len, block_word_count, token_count):
    insn = np.arange(insn_len, dtype=np.uint8)
    block = np.arange(block_word_count, dtype=np.uint8) & 0xFF
    tokens = (np.arange(token_count, dtype=np.uint16) + 1) & 0xFFFF
    record = _pack_record(insn, block, tokens, force_normal=False)

    header, prefix = parse_binary_header(record)
    assert header.format is BinaryHeaderFormat.UltraShort
    arrays = extract_arrays_from_data(record, header, prefix)
    _expect_arrays_equal(arrays, insn, block, tokens)
