"""Focused test for ``_matched_arm_loader._parse_section_blob``.

Round-trips a synthetic 2-variant matched section blob through the
parser so the 8-hex-character inline-indexer decode + 3-cell variant
row contract is pinned independently of the orchestrator-level
``SectionArm`` / ``open_sections_csv`` chain (which 2C / 2D rewrite in
parallel batches).
"""

from __future__ import annotations

import base64
import csv
import io
from pathlib import Path

import pytest

from tokenizer.aligned_data.inline_indexer import encode_inline_indexer
from tokenizer.aligned_data.loader._matched_arm_loader import (
    _parse_section_blob,
)


def _blob_for(line_no: int, variant_offsets: list[int]) -> str:
    """Render a synthetic matched-section blob using the production
    CSV shape (``# format`` prelude lives one level up in
    ``open_sections_csv`` so does not appear in the blob)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    line_no_b64 = base64.b64encode(str(line_no).encode("ascii")).decode("ascii")
    writer.writerow([line_no_b64, ""])  # header: line_no + (empty) called csv
    for offset in variant_offsets:
        writer.writerow(["0x10", "", encode_inline_indexer(offset)])
    writer.writerow([])  # trailing blank separator
    return buf.getvalue()


def test_parse_section_blob_two_variants_round_trip(tmp_path: Path) -> None:
    """Two variants -> two ints; matches the 8-char inline indexer
    encode/decode round-trip exactly."""
    line_no = 42
    offsets = [0x10, 0x20]
    blob = _blob_for(line_no, offsets)
    line_to_name = {line_no: "matched_fn"}

    func_name, variant_offsets = _parse_section_blob(
        blob, line_to_name, tmp_path / "fake_sections.csv"
    )

    assert func_name == "matched_fn"
    assert variant_offsets == offsets


def test_parse_section_blob_legacy_4_cell_raises(tmp_path: Path) -> None:
    """4-cell legacy row -> ValueError with migration message."""
    line_no = 7
    line_no_b64 = base64.b64encode(str(line_no).encode("ascii")).decode("ascii")
    blob = (
        f"{line_no_b64},\n"
        "0x10,,deadbeef,00000004\n"  # legacy: 4 cells
        "\n"
    )
    line_to_name = {line_no: "fn"}
    with pytest.raises(ValueError, match="re-run memmap_builder"):
        _parse_section_blob(blob, line_to_name, tmp_path / "x.csv")


def test_parse_section_blob_legacy_16_char_indexer_raises(tmp_path: Path) -> None:
    """3 cells but legacy 16-hex-char indexer -> ``decode_inline_indexer``
    raises with the migration message (delegated hard cutover)."""
    line_no = 9
    line_no_b64 = base64.b64encode(str(line_no).encode("ascii")).decode("ascii")
    blob = (
        f"{line_no_b64},\n"
        "0x10,,deadbeefcafebabe\n"  # 16-char legacy indexer
        "\n"
    )
    line_to_name = {line_no: "fn"}
    with pytest.raises(ValueError, match="re-run memmap_builder"):
        _parse_section_blob(blob, line_to_name, tmp_path / "x.csv")
