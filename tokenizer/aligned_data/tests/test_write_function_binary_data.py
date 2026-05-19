"""Round-trip + cap-overflow tests for ``write_function_binary_data``.

The writer is the only producer of ``_data.bin`` records; correctness here
is the foundation that the reader-side tests (2A) consume. These tests use
byte-level inspection (small local layout helper) instead of routing
through ``extract_arrays_from_data`` so pad-awareness on the read side is
not a prerequisite for this suite to pass.
"""

from __future__ import annotations

import io as stdio
import random
import struct

import numpy as np
import pytest

from tokenizer.aligned_data.binary_format import (
    HEADER_BYTES,
    OVERLONG_FIELD_BYTES,
    compute_pad,
    determine_block_encoding,
    parse_binary_header,
)
from tokenizer.aligned_data.io import write_function_binary_data

# Mirror the writer's local constants so the test can drive the boundaries
# without importing private names.
_MAX_NORMAL_REAL_LENGTH = 0xFFFF << 2
_MAX_OVERLONG_REAL_LENGTH = 0xFFFFFF << 2


def _make_inputs(insn_len: int, block_enc: int, block_count: int, token_count: int, rng):
    """Build the three ndarrays the writer expects."""
    insn = np.frombuffer(rng.randbytes(insn_len), dtype=np.uint8).copy()
    block_dtype = [np.uint8, np.uint16, np.uint32][block_enc]
    nbytes = block_count * np.dtype(block_dtype).itemsize
    block = np.frombuffer(rng.randbytes(nbytes), dtype=block_dtype).copy()
    tokens = np.frombuffer(rng.randbytes(token_count * 2), dtype=np.uint16).copy()
    return insn, block, tokens


def _parse_written_record(buf: bytes):
    """Minimal byte-level parser mirroring the writer's layout.

    Returns a dict of the slice-views the writer is claimed to have laid
    down. Pad bytes are returned verbatim so the test can assert they are
    ``\\x00``.
    """
    header = parse_binary_header(buf)
    offset = HEADER_BYTES
    overlong_length = None
    if len(buf) > _MAX_NORMAL_REAL_LENGTH:
        # Total exceeds normal cap; writer must have stamped the u24 overlong
        # field. The test seeds enough headroom for this branch.
        overlong_length = int.from_bytes(buf[offset : offset + OVERLONG_FIELD_BYTES], "little") << 2
        offset += OVERLONG_FIELD_BYTES
    insn_bytes = buf[offset : offset + header.insn_len]
    offset += header.insn_len
    pad_bytes = buf[offset : offset + header.pad_size]
    offset += header.pad_size
    block_bytes = buf[offset : offset + header.block_len]
    offset += header.block_len
    tokens_bytes = buf[offset:]
    return {
        "header": header,
        "overlong_length": overlong_length,
        "insn_bytes": insn_bytes,
        "pad_bytes": pad_bytes,
        "block_bytes": block_bytes,
        "tokens_bytes": tokens_bytes,
    }


def _shape_iter(rng, n):
    """Yield ``n`` random (insn_len, block_enc, block_count, token_count) shapes.

    Distribution covers pad=0/1/2/3 incidents (small shapes hit all four
    residues) and occasionally pushes total length over the normal cap so
    the overlong branch gets exercised.
    """
    for _ in range(n):
        roll = rng.random()
        if roll < 0.85:
            insn_len = rng.randint(0, 200)
            block_enc = rng.randint(0, 2)
            block_count = rng.randint(0, 80)
            token_count = rng.randint(0, 80)
        else:
            # Force overlong territory. ~300 KiB of insn bytes is plenty
            # above the 256 KiB normal cap and well under the 64 MiB
            # overlong cap.
            insn_len = rng.randint(300_000, 320_000)
            block_enc = rng.randint(0, 2)
            block_count = rng.randint(0, 40)
            token_count = rng.randint(0, 40)
        yield insn_len, block_enc, block_count, token_count


def test_round_trip_200_random_shapes():
    rng = random.Random(0xA5A5)
    overlong_seen = 0
    for insn_len, block_enc, block_count, token_count in _shape_iter(rng, 200):
        insn, block, tokens = _make_inputs(insn_len, block_enc, block_count, token_count, rng)
        buf = stdio.BytesIO()
        result = write_function_binary_data(buf, tokens, block, insn)
        assert result is not None
        offset, length = result
        assert offset == 0
        assert length == buf.tell()
        assert length % 4 == 0
        # Byte-level inspection.
        rec = _parse_written_record(buf.getvalue())
        assert rec["header"].insn_len == insn_len
        assert rec["header"].block_enc == determine_block_encoding(block)
        # Block byte-count, not element count.
        assert rec["header"].block_len == block.nbytes
        # Pad value must be the one the pure function picks.
        is_overlong = length > _MAX_NORMAL_REAL_LENGTH
        expected_pad = compute_pad(insn_len, block.nbytes, token_count, is_overlong)
        assert rec["header"].pad_size == expected_pad
        assert rec["pad_bytes"] == b"\x00" * expected_pad
        assert rec["insn_bytes"] == insn.tobytes()
        assert rec["block_bytes"] == block.tobytes()
        assert rec["tokens_bytes"] == tokens.tobytes()
        if is_overlong:
            assert rec["overlong_length"] == length
            overlong_seen += 1
    assert overlong_seen >= 1, "shape distribution should hit the overlong branch"


def test_overlong_cap_logs_and_truncates():
    # ~33.6M uint16 tokens = 67.2 MiB tokens stream, above the 64 MiB cap.
    huge_tokens = np.zeros(33_600_000, dtype=np.uint16)
    block = np.zeros(0, dtype=np.uint8)
    insn = np.zeros(0, dtype=np.uint8)
    buf = stdio.BytesIO()
    # Pre-seed some content so we can assert truncate restored the prior
    # file position.
    buf.write(b"\x11" * 4)
    pre_offset = buf.tell()
    log = stdio.StringIO()
    result = write_function_binary_data(
        buf, huge_tokens, block, insn, func_name="huge_fn", error_log=log
    )
    assert result is None
    assert buf.tell() == pre_offset
    # File contents preserved up to pre_offset; nothing beyond.
    assert len(buf.getvalue()) == pre_offset
    log_text = log.getvalue()
    assert "overlong_length_overflow" in log_text
    assert "huge_fn" in log_text


def test_insn_len_cap_logs_and_truncates():
    insn = np.zeros(1 << 24, dtype=np.uint8)  # exactly at cap → guard fires
    block = np.zeros(0, dtype=np.uint8)
    tokens = np.zeros(0, dtype=np.uint16)
    buf = stdio.BytesIO()
    pre_offset = buf.tell()
    log = stdio.StringIO()
    result = write_function_binary_data(
        buf, tokens, block, insn, func_name="big_insn", error_log=log
    )
    assert result is None
    assert buf.tell() == pre_offset
    assert "insn_len_overflow" in log.getvalue()


def test_block_len_cap_logs_and_truncates():
    # 65536 uint8 entries → block_len == 1<<16 triggers the guard.
    insn = np.zeros(0, dtype=np.uint8)
    block = np.zeros(1 << 16, dtype=np.uint8)
    tokens = np.zeros(0, dtype=np.uint16)
    buf = stdio.BytesIO()
    pre_offset = buf.tell()
    log = stdio.StringIO()
    result = write_function_binary_data(
        buf, tokens, block, insn, func_name="big_block", error_log=log
    )
    assert result is None
    assert buf.tell() == pre_offset
    assert "block_len_overflow" in log.getvalue()


def test_no_error_log_silent_skip():
    insn = np.zeros(1 << 24, dtype=np.uint8)
    block = np.zeros(0, dtype=np.uint8)
    tokens = np.zeros(0, dtype=np.uint16)
    buf = stdio.BytesIO()
    pre_offset = buf.tell()
    # error_log defaults to None — no exception propagates, file truncated.
    result = write_function_binary_data(buf, tokens, block, insn)
    assert result is None
    assert buf.tell() == pre_offset


def test_dedup_cache_hit_avoids_rewrite():
    insn = np.arange(10, dtype=np.uint8)
    block = np.arange(5, dtype=np.uint8)
    tokens = np.arange(7, dtype=np.uint16)
    cache = {}
    buf = stdio.BytesIO()
    first = write_function_binary_data(buf, tokens, block, insn, dedup_cache=cache)
    assert first is not None
    bytes_after_first = buf.tell()
    second = write_function_binary_data(buf, tokens, block, insn, dedup_cache=cache)
    assert second == first
    # File position unchanged: cache hit returned without writing.
    assert buf.tell() == bytes_after_first


def test_dedup_cache_not_polluted_on_skip():
    insn = np.zeros(1 << 24, dtype=np.uint8)
    block = np.zeros(0, dtype=np.uint8)
    tokens = np.zeros(0, dtype=np.uint16)
    cache = {}
    buf = stdio.BytesIO()
    log = stdio.StringIO()
    result = write_function_binary_data(
        buf, tokens, block, insn, dedup_cache=cache, error_log=log
    )
    assert result is None
    assert cache == {}


def test_post_write_alignment_invariant_sweep():
    """Every successful write leaves the file at a 4-byte boundary."""
    rng = random.Random(0xBEEF)
    buf = stdio.BytesIO()
    successes = 0
    for insn_len, block_enc, block_count, token_count in _shape_iter(rng, 50):
        insn, block, tokens = _make_inputs(insn_len, block_enc, block_count, token_count, rng)
        result = write_function_binary_data(buf, tokens, block, insn)
        assert result is not None
        assert buf.tell() % 4 == 0
        successes += 1
    assert successes == 50
