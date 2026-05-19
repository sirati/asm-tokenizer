"""Tests for ``tokenizer.aligned_data.loader.metadata_loader``.

Coverage targets:
- ``SectionArm`` equality across matched + unmatched arms.
- Byte-equality between the legacy ``BinaryDataset.{matched,unmatched}_*``
  attribute values and the new ``SectionArm`` fields, so the
  extraction is a pure refactor on the public surface.
- ``section_starts`` for the unmatched arm: each recorded offset
  rewinds a fresh handle to a row whose first column matches the
  corresponding ``func_names`` entry.
- ``SectionKind`` dispatches correctly (matched returns empty
  ``section_starts``; unmatched returns one offset per row).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np

from tokenizer.aligned_data.binary_format import encode_binary_header
from tokenizer.aligned_data.io import write_index_entry
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import (
    BinaryArmPaths,
    SectionArm,
    SectionKind,
    load_section_arm,
    open_sections_csv,
)


# ---------------------------------------------------------------------------
# Fixture builders. Write a tiny on-disk binary using the same helpers
# the memmap_builder pipeline uses, so the format is exercised end-to-end
# without coupling tests to the builder's high-level orchestration.
# ---------------------------------------------------------------------------


def _write_matched_data(path: Path, n_records: int) -> List[tuple]:
    """Write ``n_records`` matched function records, return per-record
    ``(start, length, avg_len)`` triples for the index file."""
    triples: List[tuple] = []
    with open(path, "wb") as f:
        for i in range(n_records):
            start = f.tell()
            # tokens length = i+1 uint16s → 2*(i+1) bytes
            tokens = np.arange(i + 1, dtype=np.uint16)
            insn_rl = np.array([1, 2], dtype=np.uint8)
            block_rl = np.array([3], dtype=np.uint8)
            header = encode_binary_header(
                insn_len=len(insn_rl),
                block_enc=0,  # uint8
                block_len=len(block_rl),
            )
            f.write(header)
            f.write(insn_rl.tobytes())
            f.write(block_rl.tobytes())
            f.write(tokens.tobytes())
            length = f.tell() - start
            avg_len = (i + 1) * 16  # writer clamps via avg_len>>4
            triples.append((start, length, avg_len))
    return triples


def _write_index(path: Path, triples: List[tuple]) -> None:
    with open(path, "wb") as f:
        for start, length, avg_len in triples:
            write_index_entry(f, start, length, avg_len)


def _write_matched_sections(
    path: Path, names_with_offsets: List[tuple]
) -> None:
    """Write a sections CSV whose row offsets equal those passed in.

    For test parity with the production layout, each function gets a
    single-cell header row followed by one variant row, and the index
    file's ``start`` value is the byte offset of the header row.
    """
    with open(path, "wb") as f:
        for name, start, length in names_with_offsets:
            assert f.tell() == start, (
                f"Section CSV builder out of sync at {name}: "
                f"want {start} got {f.tell()}"
            )
            block = name.encode("ascii") + b"\n0,,0,0\n"
            assert len(block) == length, (
                f"Section CSV builder length mismatch at {name}: "
                f"want {length} got {len(block)}"
            )
            f.write(block)


def _build_matched_arm(tmp_path: Path) -> None:
    n = 3
    triples = _write_matched_data(tmp_path / "bin_data.bin", n)

    # Plan the sections-CSV byte layout up front so the index's
    # "start" column truly is the per-section CSV offset (mirrors the
    # builder pipeline's contract).
    names: List[tuple] = []
    cursor = 0
    for i in range(n):
        name = f"matched_fn_{i}"
        block_len = len(name.encode("ascii")) + 1 + len(b"0,,0,0\n")
        names.append((name, cursor, block_len))
        cursor += block_len

    # Rewrite the index to use the sections-CSV offsets as ``start``
    # (matches production layout where ``starts`` are CSV offsets).
    csv_triples = [
        (start, length, triples[i][2])
        for i, (_name, start, length) in enumerate(names)
    ]
    _write_index(tmp_path / "bin_index.bin", csv_triples)
    _write_matched_sections(tmp_path / "bin_sections.csv", names)


def _build_unmatched_arm(tmp_path: Path) -> List[str]:
    """Write the three unmatched files; return the ordered func_names
    so tests can cross-check."""
    names = ["unfn_a", "unfn_bb", "unfn_ccc"]

    # Data bin: one record per function with synthetic token streams
    # of distinct lengths so length-lookup tables differentiate them.
    triples: List[tuple] = []
    with open(tmp_path / "bin_unmatched_data.bin", "wb") as f:
        for i, _name in enumerate(names):
            start = f.tell()
            tokens = np.arange((i + 1) * 2, dtype=np.uint16)
            insn_rl = np.array([1], dtype=np.uint8)
            block_rl = np.array([2, 3], dtype=np.uint8)
            f.write(
                encode_binary_header(
                    insn_len=len(insn_rl),
                    block_enc=0,
                    block_len=len(block_rl),
                )
            )
            f.write(insn_rl.tobytes())
            f.write(block_rl.tobytes())
            f.write(tokens.tobytes())
            length = f.tell() - start
            # avg_len here irrelevant — unmatched re-reads header for
            # actual token count via load_unmatched_lengths.
            triples.append((start, length, 0))

    _write_index(tmp_path / "bin_unmatched_index.bin", triples)

    # Unmatched sections CSV: 6-col rows (func_name, variant_refs,
    # called_funcs, inlining, data_offset, data_len). Plain ASCII,
    # no embedded newlines per row.
    with open(tmp_path / "bin_unmatched_sections.csv", "w",
              newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        for i, name in enumerate(names):
            start, length, _ = triples[i]
            writer.writerow(
                [name, "0x0", "", "", f"{start:x}", f"{length:x}"]
            )

    return names


def _matched_paths(tmp_path: Path) -> BinaryArmPaths:
    return BinaryArmPaths(
        sections_csv=tmp_path / "bin_sections.csv",
        index_bin=tmp_path / "bin_index.bin",
        data_bin=tmp_path / "bin_data.bin",
    )


def _unmatched_paths(tmp_path: Path) -> BinaryArmPaths:
    return BinaryArmPaths(
        sections_csv=tmp_path / "bin_unmatched_sections.csv",
        index_bin=tmp_path / "bin_unmatched_index.bin",
        data_bin=tmp_path / "bin_unmatched_data.bin",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_section_arm_equality_same_inputs(tmp_path):
    """``SectionArm`` is a frozen dataclass: identical inputs produce
    arms whose fields are element-equal."""
    _build_matched_arm(tmp_path)
    arm_a = load_section_arm(SectionKind.MATCHED, _matched_paths(tmp_path))
    arm_b = load_section_arm(SectionKind.MATCHED, _matched_paths(tmp_path))

    assert np.array_equal(arm_a.starts, arm_b.starts)
    assert np.array_equal(arm_a.lengths, arm_b.lengths)
    assert np.array_equal(arm_a.edge_indices, arm_b.edge_indices)
    assert np.array_equal(arm_a.count_per_length, arm_b.count_per_length)
    assert arm_a.func_names == arm_b.func_names
    assert np.array_equal(arm_a.section_starts, arm_b.section_starts)
    assert arm_a.count == arm_b.count


def test_matched_arm_matches_legacy_attributes(tmp_path):
    """``BinaryDataset.matched_*`` attributes equal the matched
    ``SectionArm`` fields — pre/post-refactor surface is byte-equal."""
    _build_matched_arm(tmp_path)
    _build_unmatched_arm(tmp_path)
    dataset = BinaryDataset(tmp_path, "bin")
    arm = dataset._matched_arm

    assert np.array_equal(dataset.matched_starts, arm.starts)
    assert np.array_equal(dataset.matched_lengths, arm.lengths)
    assert np.array_equal(dataset.matched_edge_indices, arm.edge_indices)
    assert np.array_equal(
        dataset.matched_count_per_length, arm.count_per_length
    )
    assert dataset.matched_func_names == arm.func_names
    assert dataset.matched_count == arm.count


def test_unmatched_arm_matches_legacy_attributes(tmp_path):
    """``BinaryDataset.unmatched_*`` attributes equal the unmatched
    ``SectionArm`` fields."""
    _build_matched_arm(tmp_path)
    _build_unmatched_arm(tmp_path)
    dataset = BinaryDataset(tmp_path, "bin")
    arm = dataset._unmatched_arm

    assert np.array_equal(dataset.unmatched_starts, arm.starts)
    assert np.array_equal(dataset.unmatched_lengths, arm.lengths)
    assert np.array_equal(dataset.unmatched_edge_indices, arm.edge_indices)
    assert np.array_equal(
        dataset.unmatched_count_per_length, arm.count_per_length
    )
    assert dataset.unmatched_func_names == arm.func_names
    assert dataset.unmatched_count == arm.count


def test_matched_arm_has_empty_section_starts(tmp_path):
    """Matched callers drive seeks from ``starts`` (which ARE CSV
    offsets); ``section_starts`` is intentionally empty for the
    matched arm."""
    _build_matched_arm(tmp_path)
    arm = load_section_arm(SectionKind.MATCHED, _matched_paths(tmp_path))
    assert len(arm.section_starts) == 0
    # Sanity: the legacy starts ARE the CSV offsets, so seeking to
    # arm.starts[i] in the sections CSV reads the i-th function's
    # header row.
    with open(_matched_paths(tmp_path).sections_csv, "rb") as f:
        for i, start in enumerate(arm.starts):
            f.seek(int(start))
            line = f.readline().rstrip(b"\n").decode("ascii")
            assert line == arm.func_names[i]


def test_unmatched_section_starts_point_to_rows(tmp_path):
    """For unmatched, ``section_starts[i]`` is the byte offset (in the
    sections CSV, content-offset-relative) of the row whose first
    column equals ``func_names[i]``."""
    expected_names = _build_unmatched_arm(tmp_path)
    arm = load_section_arm(SectionKind.UNMATCHED, _unmatched_paths(tmp_path))

    assert arm.func_names == expected_names
    assert len(arm.section_starts) == len(expected_names)

    # Seek-test each offset on a fresh handle.
    f, content_offset = open_sections_csv(
        _unmatched_paths(tmp_path).sections_csv
    )
    try:
        for i, offset in enumerate(arm.section_starts):
            f.seek(int(offset) + content_offset)
            line = f.readline()
            row = list(csv.reader([line]))[0]
            assert row[0] == expected_names[i]
    finally:
        f.close()


def test_section_kind_enum_is_closed_typed():
    """Sanity: ``SectionKind`` is an enum (not a bool), and both
    arms are registered. Future arms add an enum value + spec; no
    boolean toggle in the caller."""
    assert SectionKind.MATCHED is not SectionKind.UNMATCHED
    assert SectionKind.MATCHED.value == "matched"
    assert SectionKind.UNMATCHED.value == "unmatched"


def test_empty_arm_when_index_missing(tmp_path):
    """Missing index file yields the canonical empty arm with
    dtype-preserving placeholders (so downstream length / indexing
    arithmetic doesn't degrade)."""
    paths = BinaryArmPaths(
        sections_csv=tmp_path / "absent_sections.csv",
        index_bin=tmp_path / "absent_index.bin",
        data_bin=tmp_path / "absent_data.bin",
    )
    arm = load_section_arm(SectionKind.MATCHED, paths)
    assert arm.count == 0
    assert arm.starts.dtype == np.uint32
    assert arm.lengths.dtype == np.uint32
    assert arm.edge_indices.dtype == np.int32
    assert arm.count_per_length.dtype == np.int32
    assert arm.func_names == []
    assert arm.section_starts.dtype == np.int64
    assert len(arm.section_starts) == 0


def test_zero_entry_index_yields_empty_arm(tmp_path):
    """An index file that exists but is empty (zero entries) yields
    the empty arm, NOT a None/exception — matches the legacy
    zero-entry handling in ``load_index_once``."""
    (tmp_path / "empty_index.bin").write_bytes(b"")
    paths = BinaryArmPaths(
        sections_csv=tmp_path / "empty_sections.csv",
        index_bin=tmp_path / "empty_index.bin",
        data_bin=tmp_path / "empty_data.bin",
    )
    arm = load_section_arm(SectionKind.MATCHED, paths)
    assert arm.count == 0
    assert len(arm.starts) == 0
    assert len(arm.lengths) == 0
