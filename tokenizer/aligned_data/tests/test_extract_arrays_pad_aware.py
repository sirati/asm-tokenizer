"""Reader-side pad and overlong-prefix awareness.

Hand-packs record bytes (header + optional overlong field + insn + pad +
block + tokens) and asserts that ``extract_arrays_from_data`` /
``parse_function_data_header`` / ``parse_function_data_memmap`` slice
the correct ndarrays back out across every pad value (0..3), both
record variants (normal / overlong), and every ``block_enc`` (0/1/2).

Pure parsing: no calls into the writer side. The 3-byte overlong
field's content is opaque to the reader (the caller resolves the real
length from the index sentinel); only its presence shifts the body
offset.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tokenizer.aligned_data.binary_format import (
    HEADER_BYTES,
    OVERLONG_FIELD_BYTES,
    extract_arrays_from_data,
    parse_binary_header,
)
from tokenizer.aligned_data.io import (
    parse_function_data_header,
    parse_function_data_memmap,
)


# ---------------------------------------------------------------------------
# Hand-packing helpers (test-local — do NOT call into the writer side).
# ---------------------------------------------------------------------------


_BLOCK_DTYPES = (np.uint8, np.uint16, np.uint32)


def _pack_header(insn_len: int, block_enc: int, block_len: int, pad_size: int) -> bytes:
    """Pack the 6-byte record header by hand (no writer reuse)."""
    packed = (block_enc & 0b11) | ((pad_size & 0b11) << 2)
    out = bytearray()
    out.append(packed)
    out.extend(struct.pack("<I", insn_len)[0:3])
    out.extend(struct.pack("<H", block_len))
    return bytes(out)


def _pack_record(
    insn: np.ndarray,
    block: np.ndarray,
    tokens: np.ndarray,
    pad_size: int,
    is_overlong: bool,
    overlong_length_value: int = 0,
) -> bytes:
    """Hand-pack one record matching the v1 layout."""
    block_enc = _BLOCK_DTYPES.index(block.dtype.type)
    insn_bytes = insn.astype(np.uint8).tobytes()
    block_bytes = block.tobytes()
    tokens_bytes = tokens.astype(np.uint16).tobytes()

    parts = bytearray()
    parts.extend(
        _pack_header(len(insn_bytes), block_enc, len(block_bytes), pad_size)
    )
    if is_overlong:
        # u24 shifted; content is opaque to the reader. Only the 3-byte
        # presence matters for body-offset arithmetic.
        parts.extend(struct.pack("<I", overlong_length_value >> 2)[0:3])
    parts.extend(insn_bytes)
    parts.extend(b"\x00" * pad_size)
    parts.extend(block_bytes)
    parts.extend(tokens_bytes)
    return bytes(parts)


def _expect_arrays_equal(
    got: tuple,
    insn: np.ndarray,
    block: np.ndarray,
    tokens: np.ndarray,
) -> None:
    got_insn, got_block, got_tokens = got
    np.testing.assert_array_equal(got_insn, insn.astype(np.uint8))
    assert got_block.dtype == block.dtype
    np.testing.assert_array_equal(got_block, block)
    assert got_tokens.dtype == np.uint16
    np.testing.assert_array_equal(got_tokens, tokens.astype(np.uint16))


def _sample_arrays(block_enc: int) -> tuple:
    """Synthetic insn / block / tokens of distinct, recognizable bytes."""
    insn = np.arange(7, dtype=np.uint8)
    block_dtype = _BLOCK_DTYPES[block_enc]
    block = np.arange(1, 6, dtype=block_dtype) * np.array(1, dtype=block_dtype)
    tokens = np.array([10, 20, 30, 40], dtype=np.uint16)
    return insn, block, tokens


# ---------------------------------------------------------------------------
# Normal record, pad ∈ {0..3}, every block_enc.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_enc", [0, 1, 2])
@pytest.mark.parametrize("pad_size", [0, 1, 2, 3])
def test_normal_record_round_trip(block_enc, pad_size):
    insn, block, tokens = _sample_arrays(block_enc)
    record = _pack_record(
        insn, block, tokens, pad_size=pad_size, is_overlong=False
    )

    header = parse_binary_header(record)
    assert header.pad_size == pad_size
    assert header.block_enc == block_enc
    assert header.insn_len == len(insn)
    assert header.block_len == block.nbytes

    arrays = extract_arrays_from_data(record, header, is_overlong=False)
    _expect_arrays_equal(arrays, insn, block, tokens)


def test_normal_record_is_overlong_required_kwarg():
    """``is_overlong`` is REQUIRED keyword-only -- no silent default.

    The previous ``is_overlong=False`` default silently corrupted
    overlong reads when a caller forgot to thread the sentinel
    through; the audit (blocker #4) removed the default to make
    forgetting impossible. This test pins that down: omitting the
    kwarg raises ``TypeError``, and passing it explicitly behaves
    exactly as the old positive case did.
    """
    insn, block, tokens = _sample_arrays(block_enc=1)
    record = _pack_record(insn, block, tokens, pad_size=0, is_overlong=False)
    with pytest.raises(TypeError):
        parse_function_data_header(record)
    arrays = parse_function_data_header(record, is_overlong=False)
    _expect_arrays_equal(arrays, insn, block, tokens)


# ---------------------------------------------------------------------------
# Overlong record, pad ∈ {0..3}, every block_enc.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_enc", [0, 1, 2])
@pytest.mark.parametrize("pad_size", [0, 1, 2, 3])
def test_overlong_record_round_trip(block_enc, pad_size):
    insn, block, tokens = _sample_arrays(block_enc)
    # The 3-byte overlong field's value is opaque to the reader; we set
    # it to a plausible u24<<2 of, say, 300 KiB to be realistic, but the
    # reader must not consume it.
    overlong_real_length = 300 * 1024
    record = _pack_record(
        insn,
        block,
        tokens,
        pad_size=pad_size,
        is_overlong=True,
        overlong_length_value=overlong_real_length,
    )

    arrays = parse_function_data_header(record, is_overlong=True)
    _expect_arrays_equal(arrays, insn, block, tokens)


def test_overlong_body_starts_at_byte_9():
    """Sanity check on the body offset constant for the overlong variant."""
    insn = np.array([0xAA, 0xBB, 0xCC, 0xDD], dtype=np.uint8)
    block = np.array([0x1111, 0x2222], dtype=np.uint16)
    tokens = np.array([0xDEAD, 0xBEEF], dtype=np.uint16)
    record = _pack_record(
        insn, block, tokens, pad_size=2, is_overlong=True,
        overlong_length_value=1 << 20,
    )
    # First 6 = header; next 3 = overlong field; insn begins at byte 9.
    assert HEADER_BYTES + OVERLONG_FIELD_BYTES == 9
    assert record[9:13] == bytes(insn)

    arrays = parse_function_data_header(record, is_overlong=True)
    _expect_arrays_equal(arrays, insn, block, tokens)


# ---------------------------------------------------------------------------
# Memmap path: synthetic bin with one normal + one overlong record.
# ---------------------------------------------------------------------------


def test_parse_function_data_memmap_normal_and_overlong(tmp_path):
    """Write a tiny synthetic ``_data.bin``, mmap it, decode both records."""
    insn_a, block_a, tokens_a = _sample_arrays(block_enc=0)
    record_a = _pack_record(
        insn_a, block_a, tokens_a, pad_size=1, is_overlong=False
    )

    insn_b, block_b, tokens_b = _sample_arrays(block_enc=2)
    record_b = _pack_record(
        insn_b, block_b, tokens_b, pad_size=3, is_overlong=True,
        overlong_length_value=512 * 1024,
    )

    bin_path = tmp_path / "synthetic_data.bin"
    with open(bin_path, "wb") as fh:
        fh.write(record_a)
        offset_b = fh.tell()
        fh.write(record_b)

    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")
    arrays_a = parse_function_data_memmap(
        mmap, 0, len(record_a), is_overlong=False
    )
    arrays_b = parse_function_data_memmap(
        mmap, offset_b, len(record_b), is_overlong=True
    )

    _expect_arrays_equal(arrays_a, insn_a, block_a, tokens_a)
    _expect_arrays_equal(arrays_b, insn_b, block_b, tokens_b)


# ---------------------------------------------------------------------------
# Zero-copy guarantee: parsing a memmap must not allocate the whole record.
# Returned arrays must be views into the memmap (np.shares_memory), and the
# three reader entry points (bytes / ndarray / memmap) must agree byte-for-byte.
# ---------------------------------------------------------------------------


def test_parse_memmap_returns_views_into_the_memmap(tmp_path):
    """Slicing memmap input must yield memmap-backed views, not copies.

    The audit flagged the previous ``.tobytes()`` path as defeating the
    alignment-driven perf win. The reader must now slice the memmap
    directly so the returned arrays share memory with the underlying
    mapping — independent of pad value and overlong layout.
    """
    insn_a, block_a, tokens_a = _sample_arrays(block_enc=2)
    record_a = _pack_record(
        insn_a, block_a, tokens_a, pad_size=2, is_overlong=False
    )
    insn_b, block_b, tokens_b = _sample_arrays(block_enc=1)
    record_b = _pack_record(
        insn_b, block_b, tokens_b, pad_size=3, is_overlong=True,
        overlong_length_value=400 * 1024,
    )

    bin_path = tmp_path / "synthetic_data.bin"
    with open(bin_path, "wb") as fh:
        fh.write(record_a)
        offset_b = fh.tell()
        fh.write(record_b)

    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")

    insn_p, block_p, tok_p = parse_function_data_memmap(
        mmap, 0, len(record_a), is_overlong=False
    )
    insn_q, block_q, tok_q = parse_function_data_memmap(
        mmap, offset_b, len(record_b), is_overlong=True
    )

    for arr in (insn_p, block_p, tok_p, insn_q, block_q, tok_q):
        assert np.shares_memory(arr, mmap), (
            f"array {arr.dtype}/{arr.shape} does not share memory with memmap "
            f"— a record-sized copy was allocated"
        )


def test_parse_three_input_paths_agree_byte_for_byte(tmp_path):
    """bytes / np.ndarray / np.memmap inputs must produce identical output.

    Same record packed once; fed through the parser as three distinct
    buffer kinds. Byte-for-byte agreement pins down that the
    type-dispatch in ``_as_uint8_view`` is behaviour-preserving.
    """
    insn, block, tokens = _sample_arrays(block_enc=2)
    record = _pack_record(
        insn, block, tokens, pad_size=2, is_overlong=False
    )

    bin_path = tmp_path / "synthetic_data.bin"
    bin_path.write_bytes(record)

    bytes_form = parse_function_data_header(record, is_overlong=False)
    ndarray_form = parse_function_data_header(
        np.frombuffer(record, dtype=np.uint8), is_overlong=False
    )
    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")
    memmap_form = parse_function_data_header(mmap, is_overlong=False)

    for got in (bytes_form, ndarray_form, memmap_form):
        _expect_arrays_equal(got, insn, block, tokens)

    # Cross-check: dtype + bytes agree across all three input paths.
    for i in range(3):
        assert bytes_form[i].dtype == ndarray_form[i].dtype == memmap_form[i].dtype
        assert bytes_form[i].tobytes() == ndarray_form[i].tobytes() == memmap_form[i].tobytes()


def test_parse_binary_header_does_not_copy_full_record(tmp_path):
    """``parse_binary_header`` reads only the first 6 bytes.

    Build a comparatively large record, hand the parser a memmap slice
    pointing at it, and confirm the parsed fields are correct without
    relying on any allocation behaviour — the function's only contract
    is that it never materialises the body. We assert the documented
    fields round-trip correctly; the no-copy guarantee for the body is
    pinned by the dedicated body-slice tests above.
    """
    insn = np.arange(2048, dtype=np.uint8)
    block = np.arange(1024, dtype=np.uint16)
    tokens = np.arange(4096, dtype=np.uint16)
    record = _pack_record(insn, block, tokens, pad_size=1, is_overlong=False)

    bin_path = tmp_path / "big_data.bin"
    bin_path.write_bytes(record)
    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")

    header = parse_binary_header(mmap[: HEADER_BYTES])
    assert header.insn_len == len(insn)
    assert header.block_enc == 1
    assert header.block_len == block.nbytes
    assert header.pad_size == 1


def test_overlong_field_bytes_are_not_consumed_as_insn(tmp_path):
    """The 3-byte overlong field must NOT bleed into the insn slice.

    Set the overlong-field bytes to a distinctive non-zero pattern and
    the insn bytes to a different distinctive pattern; if the reader
    mis-aligned by 3 bytes, the insn array would carry the overlong
    bytes instead.
    """
    insn = np.array([0x10, 0x11, 0x12, 0x13, 0x14], dtype=np.uint8)
    block = np.array([0x21, 0x22], dtype=np.uint8)
    tokens = np.array([0x0101], dtype=np.uint16)
    # Pack manually so we control the overlong-field content explicitly.
    header = _pack_header(
        insn_len=len(insn), block_enc=0, block_len=len(block), pad_size=2
    )
    overlong_field = b"\xFF\xFE\xFD"  # distinctive — must not appear in insn
    body = bytes(insn) + b"\x00" * 2 + bytes(block) + tokens.tobytes()
    record = header + overlong_field + body

    arrays = parse_function_data_header(record, is_overlong=True)
    np.testing.assert_array_equal(arrays[0], insn)
    assert bytes(arrays[0]) != overlong_field[: len(insn)]
