"""Coverage for the post-audit ``tokenizer.aligned_data.io`` API:

* matched- and unmatched-section writers take pre-encoded ``indexer_hex``
  cells (no ``data_offset, data_len`` pair) -- callers own the inline
  indexer encode.
* ``read_sections_file`` routes through ``open_sections_csv`` so the
  ``# format=N`` prelude is consumed before ``csv.reader`` sees the
  stream.
* All four reader entry points raise ``TypeError`` when ``is_overlong``
  is omitted -- the silent default was audit blocker #4.
"""

from __future__ import annotations

import csv
from io import StringIO

import numpy as np
import pytest

from tokenizer.aligned_data.csv_format import write_csv_prelude
from tokenizer.aligned_data.inline_indexer import encode_inline_indexer
from tokenizer.aligned_data.io import (
    parse_function_data_header,
    parse_function_data_memmap,
    read_data_file,
    read_function_data_memmap,
    read_sections_file,
    write_function_section_csv,
    write_unmatched_section_csv,
)


# ---------------------------------------------------------------------------
# Section writers emit the new 3- and 6-cell layouts.
# ---------------------------------------------------------------------------


def test_write_function_section_csv_emits_three_cells():
    """Matched-section variant row has exactly 3 cells: ref, inlining, indexer."""
    buf = StringIO()
    writer = csv.writer(buf)
    indexer_hex = encode_inline_indexer(0x40, 0x80)
    write_function_section_csv(
        writer,
        variant_ref="0x2a",
        inlining_list=[],
        indexer_hex=indexer_hex,
    )
    rows = list(csv.reader(StringIO(buf.getvalue())))
    assert len(rows) == 1
    assert rows[0] == ["0x2a", "", indexer_hex]
    # Indexer hex is exactly 16 chars (8-byte v1 entry hex-encoded).
    assert len(rows[0][2]) == 16


def test_write_unmatched_section_csv_emits_six_cells():
    """Unmatched-section row has 6 cells: line_no_b64, refs, called,
    inlining, indexer.

    First cell is the caller-computed base64 of the function name's
    sidecar line no (NOT the raw function name); writer treats it as
    an opaque string.
    """
    buf = StringIO()
    writer = csv.writer(buf)
    indexer_hex = encode_inline_indexer(0x100, 0x40)
    write_unmatched_section_csv(
        writer,
        line_no_b64="Aw",  # caller-computed; opaque to writer
        variant_refs=["0x1", "0x2"],
        called_functions_str="Aw,Bw",  # caller-computed (line nos, not names)
        inlining_data_str="",
        indexer_hex=indexer_hex,
    )
    rows = list(csv.reader(StringIO(buf.getvalue())))
    assert len(rows) == 1
    assert rows[0] == ["Aw", "0x1;0x2", "Aw,Bw", "", indexer_hex]


def test_writers_do_not_accept_legacy_offset_length_pair():
    """Old call form passing (data_offset, data_len) must fail loudly.

    Pinned to prevent silent regressions if a caller copy-pastes
    pre-restructuring code.
    """
    buf = StringIO()
    writer = csv.writer(buf)
    with pytest.raises(TypeError):
        write_function_section_csv(writer, "0x0", [], 0, 4)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        write_unmatched_section_csv(writer, "Aw", [], "", "", 0, 4)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sections-CSV reader consumes the prelude.
# ---------------------------------------------------------------------------


def _write_sections_csv_with_prelude(path, sections):
    """Helper: write a v1 sections CSV (prelude + sections separated by
    blank rows). ``sections`` is a list of ``(header_cells, [row_cells, ...])``.
    """
    with open(path, "w", newline="", encoding="ascii") as fh:
        write_csv_prelude(fh)
        writer = csv.writer(fh)
        for header_cells, body_rows in sections:
            writer.writerow(header_cells)
            for row in body_rows:
                writer.writerow(row)
            writer.writerow([])


def test_read_sections_file_strips_prelude(tmp_path):
    """``read_sections_file`` must not surface the ``# format=N`` line as
    a phantom section. The first yielded section is the genuine first
    function block; iteration is otherwise unchanged.
    """
    path = tmp_path / "matched_sections.csv"
    _write_sections_csv_with_prelude(
        path,
        [
            (
                ["foo", "x,y"],
                [
                    ["0x0", "", encode_inline_indexer(0, 4)],
                    ["0x1", "", encode_inline_indexer(8, 4)],
                ],
            ),
            (
                ["bar", ""],
                [["0x2", "", encode_inline_indexer(16, 4)]],
            ),
        ],
    )
    sections = list(read_sections_file(path))
    assert [name for name, _ in sections] == ["foo", "bar"]
    # No phantom "# format=1" section leaked.
    assert not any(name.startswith("#") for name, _ in sections)


def test_read_sections_file_missing_prelude_raises(tmp_path):
    """Files written without the prelude must be rejected (hard cutover)."""
    path = tmp_path / "matched_sections.csv"
    with open(path, "w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh)
        writer.writerow(["foo", "x"])
        writer.writerow(["0x0", "", encode_inline_indexer(0, 4)])
    with pytest.raises(ValueError):
        list(read_sections_file(path))


# ---------------------------------------------------------------------------
# Reader entry points require ``is_overlong`` (audit blocker #4).
# ---------------------------------------------------------------------------


def test_parse_function_data_header_requires_is_overlong():
    """No silent ``is_overlong=False`` default on ``parse_function_data_header``."""
    with pytest.raises(TypeError):
        parse_function_data_header(b"\x00" * 8)  # type: ignore[call-arg]


def test_parse_function_data_memmap_requires_is_overlong(tmp_path):
    """No silent ``is_overlong=False`` default on ``parse_function_data_memmap``."""
    data_path = tmp_path / "data.bin"
    data_path.write_bytes(b"\x00" * 16)
    mmap = np.memmap(data_path, dtype=np.uint8, mode="r")
    with pytest.raises(TypeError):
        parse_function_data_memmap(mmap, 0, 8)  # type: ignore[call-arg]


def test_read_function_data_memmap_requires_is_overlong(tmp_path):
    """No silent ``is_overlong=False`` default on ``read_function_data_memmap``."""
    data_path = tmp_path / "data.bin"
    data_path.write_bytes(b"\x00" * 16)
    with pytest.raises(TypeError):
        read_function_data_memmap(str(data_path), 0, 8)  # type: ignore[call-arg]


def test_read_data_file_requires_is_overlong(tmp_path):
    """No silent ``is_overlong=False`` default on ``read_data_file``."""
    data_path = tmp_path / "data.bin"
    data_path.write_bytes(b"\x00" * 16)
    with pytest.raises(TypeError):
        read_data_file(str(data_path), 0, 8)  # type: ignore[call-arg]
