"""Unit tests for the ``_index.bin`` prelude wire format.

Covers round-trip, hard-cutover rejection (missing magic + wrong
format_version), truncation, the overlong-length sentinel constant, and the
``INDEX_HEADER_SIZE`` advertised width matching the writer's actual byte
count.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from tokenizer.aligned_data._writers import pack_v1_entry
from tokenizer.aligned_data.index_format import (
    ALIGNMENT_SHIFT,
    INDEX_HEADER_SIZE,
    INDEX_MAGIC,
    IndexPrelude,
    SENTINEL_LENGTH,
    decode_index_entry,
    iter_index_entries,
    read_index_arrays,
    read_index_prelude,
    write_index_prelude,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION


def test_prelude_round_trip():
    buf = io.BytesIO()
    write_index_prelude(buf)
    buf.seek(0)
    prelude = read_index_prelude(buf)
    assert prelude == IndexPrelude(
        format_version=MEMMAP_FORMAT_VERSION,
        alignment_shift=ALIGNMENT_SHIFT,
    )


def test_write_advances_exactly_header_size():
    buf = io.BytesIO()
    write_index_prelude(buf)
    assert buf.tell() == INDEX_HEADER_SIZE


def test_wrong_magic_raises_with_migration_hint():
    buf = io.BytesIO(b"XXXX" + struct.pack("<III", MEMMAP_FORMAT_VERSION, ALIGNMENT_SHIFT, 0))
    with pytest.raises(ValueError) as excinfo:
        read_index_prelude(buf)
    assert "re-run memmap_builder" in str(excinfo.value)


def test_wrong_format_version_raises_mentioning_version():
    bogus_version = MEMMAP_FORMAT_VERSION + 1
    buf = io.BytesIO(INDEX_MAGIC + struct.pack("<III", bogus_version, ALIGNMENT_SHIFT, 0))
    with pytest.raises(ValueError) as excinfo:
        read_index_prelude(buf)
    msg = str(excinfo.value)
    assert str(bogus_version) in msg
    assert str(MEMMAP_FORMAT_VERSION) in msg


def test_truncated_prelude_raises():
    buf = io.BytesIO(b"\x00" * (INDEX_HEADER_SIZE // 2))
    with pytest.raises(struct.error):
        read_index_prelude(buf)


def test_sentinel_constant_round_trips_through_u16():
    packed = struct.pack("<H", SENTINEL_LENGTH)
    (unpacked,) = struct.unpack("<H", packed)
    assert unpacked == SENTINEL_LENGTH
    assert SENTINEL_LENGTH == 0x0000


def test_reserved_field_is_not_checked():
    # Writer stamps reserved=0, but the reader must tolerate arbitrary values
    # so future bumps that repurpose the field stay additive.
    raw = INDEX_MAGIC + struct.pack(
        "<III", MEMMAP_FORMAT_VERSION, ALIGNMENT_SHIFT, 0xDEADBEEF
    )
    buf = io.BytesIO(raw)
    prelude = read_index_prelude(buf)
    assert prelude.format_version == MEMMAP_FORMAT_VERSION
    assert prelude.alignment_shift == ALIGNMENT_SHIFT


# ---------------------------------------------------------------------------
# Vectorised ``read_index_arrays`` matches per-entry decode.
# ---------------------------------------------------------------------------


def _write_synthetic_index(path, triples):
    """Write a v1 ``_index.bin`` with the given ``(offset, length, avg_len)``
    triples. Sentinel + cap-edge values are exercised by the caller.
    """
    with open(path, "wb") as fh:
        write_index_prelude(fh)
        for offset, length, avg_len in triples:
            fh.write(pack_v1_entry(offset, length, avg_len))


def test_read_index_arrays_vectorised_matches_streaming(tmp_path):
    """1000-entry equivalence pin: ``read_index_arrays`` (vectorised)
    must produce arrays identical to iterating ``iter_index_entries``
    and stamping decoded fields into per-row columns.

    Covers the full value range the per-entry path produces:
    * small offsets / lengths in the normal-record band,
    * lengths at the cap (max u16 length_shifted, still normal),
    * a deliberate sentinel marker (length=0 stored, real length lives
      in the data record's overlong field; index reader returns 0),
    * an overlong record whose total triggers the sentinel writer path.
    """
    # Deterministic seed so the corpus is reproducible across runs.
    rng = np.random.default_rng(0xC0FFEE)
    triples = []
    n = 1000
    # 4-byte alignment is enforced by ``pack_v1_entry``; generate
    # offsets / lengths in 4-byte multiples spanning the full normal
    # band plus a handful of overlong triggers + cap-edge lengths.
    for i in range(n - 4):
        # Offsets sized so a 5-byte u40 covers them: spread across
        # 0..(1<<35) to exercise upper bytes.
        offset = int(rng.integers(0, 1 << 35)) & ~0b11
        length = int(rng.integers(4, 0xFFFC)) & ~0b11  # normal-band, 4-aligned
        avg_len = int(rng.integers(0, 4096))  # > 255 to verify >>4 clamp
        triples.append((offset, length, avg_len))
    # Cap-edge lengths (still normal band).
    triples.append((0x40, 0xFFFC, 0))  # max length_shifted = 0xFFFF
    triples.append((0x80, 0x4, 255 * 16))  # avg_len at clamp ceiling
    # Overlong triggers (sentinel path): real length > MAX_NORMAL_REAL_LENGTH.
    from tokenizer.aligned_data.index_format import MAX_NORMAL_REAL_LENGTH

    triples.append((0x100, MAX_NORMAL_REAL_LENGTH + 4, 0))
    triples.append((0x200, MAX_NORMAL_REAL_LENGTH + 0x4000, 0))
    assert len(triples) == n

    path = tmp_path / "test_index.bin"
    _write_synthetic_index(path, triples)

    # Vectorised path.
    starts_vec, lengths_vec, avg_lengths_vec = read_index_arrays(path)

    # Per-entry streaming path -- the reference implementation.
    starts_ref = np.zeros(n, dtype=np.int64)
    lengths_ref = np.zeros(n, dtype=np.uint32)
    avg_lengths_ref = np.zeros(n, dtype=np.uint8)
    for i, (start, length, avg_len) in enumerate(iter_index_entries(path)):
        starts_ref[i] = start
        lengths_ref[i] = length
        avg_lengths_ref[i] = avg_len

    assert np.array_equal(starts_vec, starts_ref), (
        "vectorised starts disagree with streaming decode"
    )
    assert np.array_equal(lengths_vec, lengths_ref), (
        "vectorised lengths disagree with streaming decode"
    )
    assert np.array_equal(avg_lengths_vec, avg_lengths_ref), (
        "vectorised avg_lengths disagree with streaming decode"
    )
    # Dtypes must also match the documented contract.
    assert starts_vec.dtype == np.int64
    assert lengths_vec.dtype == np.uint32
    assert avg_lengths_vec.dtype == np.uint8


def test_read_index_arrays_sentinel_decodes_to_zero_length(tmp_path):
    """An overlong (sentinel) entry yields ``length == 0`` from the
    vectorised reader -- the caller is expected to resolve the real
    length from the data record's u24 overlong field via
    ``_index_decoding.resolve_record_length``. Pinned so the
    vectorised sentinel mask stays bit-equivalent to ``decode_index_entry``.
    """
    from tokenizer.aligned_data.index_format import MAX_NORMAL_REAL_LENGTH

    triples = [
        (0x40, 0x80, 16),  # normal
        (0x100, MAX_NORMAL_REAL_LENGTH + 0x100, 0),  # sentinel
        (0x200, 0x40, 32),  # normal
    ]
    path = tmp_path / "sentinel_index.bin"
    _write_synthetic_index(path, triples)

    starts, lengths, avg_lengths = read_index_arrays(path)
    assert list(lengths) == [0x80, 0, 0x40]  # middle entry is the sentinel
    assert list(starts) == [0x40, 0x100, 0x200]
    # avg_len 0 (sentinel entry was packed with avg_len=0)
    assert avg_lengths[1] == 0
