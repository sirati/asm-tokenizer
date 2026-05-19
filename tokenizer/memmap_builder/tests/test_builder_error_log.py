"""Builder-level error.log + sentinel plumbing tests.

Drives the writer-chain directly (bypassing the full corpus walk so the
tests run in seconds rather than minutes): each test opens an
in-memory data file + an in-memory error_log, exercises the cap or
sentinel boundary, and asserts:

  * cap overflow (``insn_len >= 2**24``) logs one TSV row into
    ``error_log``, the partial data write is truncated, and the
    corresponding ``write_index_entry`` is skipped — the index file
    stays empty.
  * a record in the [256 KiB, 64 MiB] overlong range writes the
    sentinel ``length_shifted == 0`` into the index entry and stamps
    the real (shifted) length into the u24 overlong field at bytes
    6..8 of the data record.

These exercise the producer plumbing in isolation; the consumer-side
reader updates that interpret the sentinel + content offset live in
``aligned_data/loader``.
"""

from __future__ import annotations

import io as stdio
import struct

import numpy as np
import pytest

from tokenizer.aligned_data.binary_format import (
    HEADER_BYTES,
    OVERLONG_FIELD_BYTES,
)
from tokenizer.aligned_data.io import (
    write_function_binary_data,
    write_index_entry,
)
from tokenizer.memmap_builder.error_log import ALLOWED_REASONS


# The cap the writer enforces on the per-record u24 ``insn_len`` field.
_INSN_LEN_CAP = 1 << 24


def _zero_block_uint8(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.uint8)


def test_insn_len_overflow_logs_and_skips_index_entry() -> None:
    """An insn_runlength of >= 2**24 bytes trips ``insn_len_overflow``.

    The writer truncates the partial data write to zero bytes and
    returns ``None``. The caller (test stand-in for pass-1) honours
    the ``None`` by not calling ``write_index_entry`` — the index file
    stays empty.
    """
    data_buf = stdio.BytesIO()
    error_log = stdio.StringIO()
    index_buf = stdio.BytesIO()

    # 2**24 entries of uint8 = 16,777,216 bytes -> insn_len == 2**24 == cap.
    insn = np.zeros(_INSN_LEN_CAP, dtype=np.uint8)
    block = _zero_block_uint8(4)
    tokens = np.zeros(2, dtype=np.uint16)

    write_result = write_function_binary_data(
        data_buf,
        tokens,
        block,
        insn,
        dedup_cache=None,
        func_name="overflow_fn",
        error_log=error_log,
    )

    assert write_result is None
    assert data_buf.getvalue() == b""

    # Caller side: None signals "skip". index file stays untouched.
    if write_result is not None:  # pragma: no cover - documenting the gate
        write_index_entry(
            index_buf, 0, 8, 0, func_name="overflow_fn", error_log=error_log
        )

    assert index_buf.getvalue() == b""

    log_lines = error_log.getvalue().splitlines()
    assert len(log_lines) == 1
    cols = log_lines[0].split("\t")
    assert cols[0] == "insn_len_overflow"
    assert cols[0] in ALLOWED_REASONS
    assert cols[1] == "overflow_fn"
    assert int(cols[2]) == _INSN_LEN_CAP


def test_overlong_record_triggers_index_sentinel_and_data_field() -> None:
    """A record whose real length is in [256 KiB, 64 MiB] takes the
    overlong path: the in-data u24 ``overlong_length_shifted`` field
    encodes the actual record length, and the index entry's u16
    ``length_shifted`` field is the sentinel ``0x0000``.

    The total here lands at ~300 KiB — well into the overlong band but
    cheap to allocate in memory.
    """
    data_buf = stdio.BytesIO()
    index_buf = stdio.BytesIO()
    error_log = stdio.StringIO()

    insn_count = 300 * 1024  # 300 KiB of uint8 -> insn_len = 307200
    insn = np.zeros(insn_count, dtype=np.uint8)
    block = _zero_block_uint8(4)
    tokens = np.zeros(2, dtype=np.uint16)

    write_result = write_function_binary_data(
        data_buf,
        tokens,
        block,
        insn,
        dedup_cache=None,
        func_name="overlong_fn",
        error_log=error_log,
    )

    assert write_result is not None, "300 KiB record should write successfully"
    data_offset, data_len = write_result
    assert data_offset == 0
    assert data_len > 0xFFFF << 2, "test setup must produce an overlong record"
    assert data_len % 4 == 0

    # The u24 overlong field sits immediately after the 6-byte header.
    raw = data_buf.getvalue()
    overlong_shifted = int.from_bytes(
        raw[HEADER_BYTES : HEADER_BYTES + OVERLONG_FIELD_BYTES], "little"
    )
    assert overlong_shifted << 2 == data_len

    # Now write the index entry and inspect its u16 length field.
    write_index_entry(
        index_buf,
        data_offset,
        data_len,
        16,  # avg_len bucket payload — arbitrary
        func_name="overlong_fn",
        error_log=error_log,
    )

    # No log entries — the overlong path is the success branch, not a cap skip.
    assert error_log.getvalue() == ""

    entry = index_buf.getvalue()
    assert len(entry) == 8
    # Layout: u40 offset, u16 length, u8 avg_len.
    length_field = struct.unpack("<H", entry[5:7])[0]
    assert length_field == 0, (
        f"index entry length field must be sentinel 0 for overlong record; got {length_field}"
    )


def test_normal_record_writes_real_length_no_sentinel() -> None:
    """A small record (< 256 KiB) stays in the normal layout: the index
    entry's u16 length field is the shifted real length, NOT the
    sentinel. Pairs with the overlong test as the negative case."""
    data_buf = stdio.BytesIO()
    index_buf = stdio.BytesIO()

    insn = np.zeros(128, dtype=np.uint8)
    block = _zero_block_uint8(4)
    tokens = np.zeros(2, dtype=np.uint16)

    write_result = write_function_binary_data(
        data_buf, tokens, block, insn, dedup_cache=None,
    )
    assert write_result is not None
    data_offset, data_len = write_result
    assert data_len <= 0xFFFF << 2

    write_index_entry(index_buf, data_offset, data_len, 4)
    entry = index_buf.getvalue()
    length_field = struct.unpack("<H", entry[5:7])[0]
    assert length_field != 0
    assert length_field << 2 == data_len
