"""Tests for the v1 prelude / sentinel handling on the reader side.

Covers:

* ``load_index_once`` round-trips a hand-packed v1 ``_index.bin`` that mixes
  normal and sentinel entries.
* A versionless ``_index.bin`` (legacy on-wire shape) raises with the
  migration-pointing message.
* A wrong-magic header raises.
* ``open_sections_csv`` requires the ``# format=N`` first line; legacy
  ``version=N`` and missing-prelude shapes both raise.
* ``load_unmatched_lengths`` decodes mixed normal + overlong records,
  resolving the real length from the data record's overlong field when the
  index entry surfaces the sentinel marker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data._writers import write_function_binary_data
from tokenizer.aligned_data.index_format import (
    SENTINEL_LENGTH,
    write_index_prelude,
)
from tokenizer.aligned_data.io import write_index_entry
from tokenizer.aligned_data.loader.metadata_loader import (
    BinaryArmPaths,
    load_index_once,
    load_unmatched_lengths,
    open_sections_csv,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

_PRELUDE_LINE = f"# format={MEMMAP_FORMAT_VERSION}\n"


# ---------------------------------------------------------------------------
# load_index_once
# ---------------------------------------------------------------------------


def _write_v1_index(path: Path, entries) -> None:
    """Build a v1 ``_index.bin`` from ``(start, length, avg_len)`` triples
    using the production writers so we cover the same wire format the
    reader expects (prelude + 8-byte entries with shift + sentinel).
    """
    with open(path, "wb") as f:
        write_index_prelude(f)
        for start, length, avg_len in entries:
            write_index_entry(f, start, length, avg_len)


def test_load_index_once_round_trips_mixed_normal_and_sentinel(tmp_path):
    """A v1 file mixing normal + overlong entries decodes back to the
    expected starts/lengths/avg_lengths with the sentinel surfaced as
    ``length == 0`` so callers can detect it."""
    # All starts/lengths must be 4-aligned (writer invariant).
    normal_len = 256          # well under the 256 KiB cap
    overlong_len = 1 << 20    # 1 MiB, forces sentinel in the index
    entries = [
        (0, normal_len, 16),
        (4096, normal_len, 32),
        (8192, overlong_len, 48),     # sentinel
        (1 << 24, normal_len, 64),
        (1 << 32, overlong_len, 96),  # sentinel beyond the u32 offset range
    ]
    index_path = tmp_path / "ix_index.bin"
    _write_v1_index(index_path, entries)

    starts, lengths, avg_lengths = load_index_once(index_path)
    assert starts is not None and lengths is not None and avg_lengths is not None
    assert starts.dtype == np.int64
    assert lengths.dtype == np.uint32
    assert avg_lengths.dtype == np.uint8

    expected_starts = [s for s, _l, _a in entries]
    expected_lengths = [
        0 if l > (0xFFFF << 2) else l for _s, l, _a in entries
    ]
    expected_avg = [min(a >> 4, 255) for _s, _l, a in entries]
    assert list(starts) == expected_starts
    assert list(lengths) == expected_lengths
    assert list(avg_lengths) == expected_avg
    # And specifically: every overlong row surfaces SENTINEL_LENGTH==0.
    assert lengths[2] == 0 and lengths[4] == 0
    assert SENTINEL_LENGTH == 0


def test_load_index_once_missing_file_returns_none_tuple(tmp_path):
    starts, lengths, avg_lengths = load_index_once(tmp_path / "absent.bin")
    assert starts is None and lengths is None and avg_lengths is None


def test_load_index_once_legacy_versionless_raises(tmp_path):
    """Pre-v1 ``_index.bin`` is just 8-byte entries with no prelude;
    the v1 reader must reject it with a migration-pointing message."""
    raw = bytearray()
    for _ in range(3):
        # 4-byte start + 3-byte length + 1-byte avg, the legacy entry shape.
        raw.extend((0).to_bytes(4, "little"))
        raw.extend((0).to_bytes(3, "little"))
        raw.append(0)
    path = tmp_path / "legacy_index.bin"
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError) as info:
        load_index_once(path)
    assert "re-run memmap_builder" in str(info.value)


def test_load_index_once_wrong_magic_raises(tmp_path):
    """A 16-byte prelude with the wrong magic still triggers the
    prelude check before any entry decode."""
    fake_prelude = b"XXXX" + (MEMMAP_FORMAT_VERSION).to_bytes(4, "little") + \
        (2).to_bytes(4, "little") + (0).to_bytes(4, "little")
    body = (0).to_bytes(8, "little")  # one well-formed 8-byte entry
    path = tmp_path / "bogus_magic.bin"
    path.write_bytes(fake_prelude + body)
    with pytest.raises(ValueError):
        load_index_once(path)


def test_load_index_once_misaligned_entry_region_raises(tmp_path):
    """The entry region size must be a multiple of 8; otherwise the
    reader cannot slice cleanly and raises."""
    path = tmp_path / "bad_size.bin"
    with open(path, "wb") as f:
        write_index_prelude(f)
        f.write(b"\x00" * 7)  # stray non-multiple-of-8 tail
    with pytest.raises(ValueError):
        load_index_once(path)


# ---------------------------------------------------------------------------
# open_sections_csv
# ---------------------------------------------------------------------------


def test_open_sections_csv_requires_v1_prelude(tmp_path):
    """A v1 sections CSV opens cleanly and reports the prelude width as
    ``content_offset`` so seek-based callers can re-base their offsets."""
    path = tmp_path / "ok_sections.csv"
    path.write_text(_PRELUDE_LINE + "row1,col2\n", encoding="ascii")
    f, content_offset = open_sections_csv(path)
    try:
        assert content_offset == len(_PRELUDE_LINE.encode("ascii"))
        # The handle is parked AFTER the prelude.
        assert f.readline() == "row1,col2\n"
    finally:
        f.close()


def test_open_sections_csv_missing_prelude_raises(tmp_path):
    """A sections CSV without the ``# format=N`` first line is rejected
    with the migration-pointing message."""
    path = tmp_path / "no_prelude.csv"
    path.write_text("row1,col2\n", encoding="ascii")
    with pytest.raises(ValueError) as info:
        open_sections_csv(path)
    assert "re-run memmap_builder" in str(info.value)


def test_open_sections_csv_legacy_version_prelude_raises(tmp_path):
    """The pre-v1 ``version=N`` key-value form is the dead legacy shape
    and must not be silently accepted."""
    path = tmp_path / "legacy_version.csv"
    path.write_text("version=42\nrow1,col2\n", encoding="ascii")
    with pytest.raises(ValueError):
        open_sections_csv(path)


def test_open_sections_csv_wrong_format_version_raises(tmp_path):
    """Even a comment-shaped prelude is rejected unless it carries the
    current ``MEMMAP_FORMAT_VERSION``."""
    path = tmp_path / "wrong_version.csv"
    bogus = MEMMAP_FORMAT_VERSION + 99
    path.write_text(f"# format={bogus}\nrow1,col2\n", encoding="ascii")
    with pytest.raises(ValueError):
        open_sections_csv(path)


def test_open_sections_csv_accepts_crlf_prelude(tmp_path):
    """A Windows-rewritten file with CRLF on the prelude line still
    decodes; the reader tolerates the trailing ``\\r``."""
    path = tmp_path / "crlf.csv"
    path.write_bytes(f"# format={MEMMAP_FORMAT_VERSION}\r\nrow1\n".encode("ascii"))
    f, _content_offset = open_sections_csv(path)
    try:
        # The exact handle position isn't asserted because the readline
        # consumed the CRLF; the important property is that no
        # ValueError fired.
        pass
    finally:
        f.close()


# ---------------------------------------------------------------------------
# load_unmatched_lengths with mixed normal + overlong records
# ---------------------------------------------------------------------------


def _write_overlong_record(file_handle, token_count: int):
    """Synthesise one overlong record big enough to trip the index
    sentinel; uses the production writer so layout details aren't
    duplicated here. ``insn`` carries the bulk because the header's
    ``block_len`` field is only u16 (64 KiB cap), while ``insn_len`` is
    u24 (16 MiB cap) — so > 256 KiB total only fits when insn carries it.

    Returns ``(start, length)`` for the record in the data file.
    """
    insn_len = (1 << 18) + 7  # 256 KiB + tail, forces overlong
    insn = np.zeros(insn_len, dtype=np.uint8)
    block = np.array([1, 2, 3], dtype=np.uint8)
    tokens = np.arange(token_count, dtype=np.uint16)
    result = write_function_binary_data(file_handle, tokens, block, insn)
    assert result is not None, "writer must accept the test record"
    return result  # (offset, length)


def _write_normal_record(file_handle, token_count: int):
    insn = np.array([1, 2, 3], dtype=np.uint8)
    block = np.array([4, 5], dtype=np.uint8)
    tokens = np.arange(token_count, dtype=np.uint16)
    result = write_function_binary_data(file_handle, tokens, block, insn)
    assert result is not None
    return result


def test_load_unmatched_lengths_mixed_normal_and_overlong(tmp_path):
    """Build a data file with one normal + one overlong record, write
    the matching v1 index via the production writer (which auto-emits
    the sentinel), and verify both token counts come back correctly."""
    data_path = tmp_path / "u_unmatched_data.bin"
    index_path = tmp_path / "u_unmatched_index.bin"

    with open(data_path, "wb") as f:
        normal_offset, normal_length = _write_normal_record(f, token_count=10)
        overlong_offset, overlong_length = _write_overlong_record(f, token_count=42)

    with open(index_path, "wb") as f:
        write_index_prelude(f)
        write_index_entry(f, normal_offset, normal_length, 0)
        write_index_entry(f, overlong_offset, overlong_length, 0)

    starts, lengths, _ = load_index_once(index_path)
    # The overlong row must surface the sentinel marker so the caller
    # routes it through the overlong-field path.
    assert lengths[0] != 0
    assert lengths[1] == 0

    paths = BinaryArmPaths(
        sections_csv=tmp_path / "unused.csv",  # not read by load_unmatched_lengths
        index_bin=index_path,
        data_bin=data_path,
    )
    token_counts = load_unmatched_lengths(paths, starts, lengths)
    assert list(token_counts) == [10, 42]
