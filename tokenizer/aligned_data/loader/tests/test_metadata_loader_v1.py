"""Tests for the v1 prelude handling on the sections-CSV reader side
and the post-self-describing-record ``load_unmatched_lengths`` signature.

Covers:

* ``open_sections_csv`` requires the ``# format=N`` first line; legacy
  ``version=N`` and missing-prelude shapes both raise, and a CRLF
  prelude is tolerated.
* ``load_unmatched_lengths`` reads token counts straight from each
  record's self-describing header (no companion ``lengths`` argument).

``load_index_once`` and the v1 sentinel-bearing index were retired
along with the per-record overlong escape (records now carry their
own geometry in the header); the corresponding round-trip / sentinel
tests went with them.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data._writers import write_function_binary_data
from tokenizer.aligned_data.loader.metadata_loader import (
    BinaryArmPaths,
    load_unmatched_lengths,
    open_sections_csv,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

_PRELUDE_LINE = f"# format={MEMMAP_FORMAT_VERSION}\n"


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
# load_unmatched_lengths (post-self-describing-record signature)
# ---------------------------------------------------------------------------


def _write_normal_record(file_handle, token_count: int, *, entry_idx: int = 0):
    """Synthesise one well-formed record via the production writer so
    the on-disk geometry stays the single source of truth. Returns the
    writer's ``(offset, total_bytes)`` tuple.
    """
    insn = np.array([1, 2, 3], dtype=np.uint8)
    block = np.array([4, 5], dtype=np.uint8)
    tokens = np.arange(token_count, dtype=np.uint16)
    result = write_function_binary_data(
        file_handle, tokens, block, insn, entry_idx=entry_idx
    )
    assert result is not None
    return result


def test_load_unmatched_lengths_reads_token_count_from_header(tmp_path):
    """``load_unmatched_lengths(paths, starts)`` returns one int32
    per record, sourced from each record's self-describing header.
    No companion ``lengths`` array is needed -- the wire layout
    encodes ``token_count`` directly.
    """
    data_path = tmp_path / "u_unmatched_data.bin"
    with open(data_path, "wb") as f:
        off_a, _ = _write_normal_record(f, token_count=10, entry_idx=0)
        off_b, _ = _write_normal_record(f, token_count=42, entry_idx=1)
        off_c, _ = _write_normal_record(f, token_count=137, entry_idx=2)
        from tokenizer.aligned_data.memmap_format import encode_data_bin_trailer
        f.write(encode_data_bin_trailer(3, cursor=f.tell()))

    starts = np.array([off_a, off_b, off_c], dtype=np.int64)
    paths = BinaryArmPaths(
        sections_csv=tmp_path / "unused.csv",  # not read here
        sections_bin=tmp_path / "unused.bin",
        index_bin=tmp_path / "unused_index.bin",
        data_bin=data_path,
    )
    token_counts = load_unmatched_lengths(paths, starts)
    assert token_counts.dtype == np.int32
    assert list(token_counts) == [10, 42, 137]


def test_load_unmatched_lengths_empty_starts_returns_empty(tmp_path):
    """No records -> empty int32 array; never touches the data file."""
    paths = BinaryArmPaths(
        sections_csv=tmp_path / "unused.csv",
        sections_bin=tmp_path / "unused.bin",
        index_bin=tmp_path / "unused_index.bin",
        data_bin=tmp_path / "absent_data.bin",  # missing on purpose
    )
    token_counts = load_unmatched_lengths(paths, np.array([], dtype=np.int64))
    assert token_counts.dtype == np.int32
    assert len(token_counts) == 0


def test_load_unmatched_lengths_missing_data_bin_returns_empty(tmp_path):
    """Missing ``_data.bin`` -> empty array (the empty-corpus path)."""
    starts = np.array([0], dtype=np.int64)
    paths = BinaryArmPaths(
        sections_csv=tmp_path / "unused.csv",
        sections_bin=tmp_path / "unused.bin",
        index_bin=tmp_path / "unused_index.bin",
        data_bin=tmp_path / "absent_data.bin",
    )
    token_counts = load_unmatched_lengths(paths, starts)
    assert len(token_counts) == 0
