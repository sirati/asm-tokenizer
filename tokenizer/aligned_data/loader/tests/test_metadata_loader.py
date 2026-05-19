"""Tests for ``tokenizer.aligned_data.loader.metadata_loader`` after the
matched-arm restructuring + function-names sidecar wire-in.

Coverage targets:
- Matched arm: pre-v1 ``<binary>_index.bin`` is the function-to-CSV-
  section locator; sections-CSV variant rows carry a single
  ``indexer_hex`` cell that decodes to a per-VARIANT data-bin record
  position (start, stored_length, is_overlong). ``SectionArm.starts`` /
  ``lengths`` / ``is_overlong`` are per-VARIANT; ``func_names`` /
  ``csv_starts`` / ``csv_lengths`` / ``avg_lengths`` per-FUNCTION.
- Unmatched arm: 5-cell CSV row with base64 line_no in cell 0; v1
  index_bin unchanged.
- Sidecar: ``BinaryDataset`` rejects missing / bad-prelude sidecars
  with a migration-pointing ``ValueError`` (hard cutover).
- Legacy 4-cell matched variant row raises with a migration message.
- Variable-length function names that produce CSV section starts at
  every mod-4 residue (so a future regression that re-adds the v1
  4-byte alignment assertion is caught at test time).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from tokenizer.aligned_data._writers import write_function_binary_data
from tokenizer.aligned_data.csv_format import (
    format_function_line_no,
    format_function_line_nos_csv,
)
from tokenizer.aligned_data.csv_section_index import (
    pack_csv_section_index_entry,
)
from tokenizer.aligned_data.index_format import write_index_prelude
from tokenizer.aligned_data.inline_indexer import encode_inline_indexer
from tokenizer.aligned_data.io import write_index_entry
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import (
    BinaryArmPaths,
    SectionArm,
    SectionKind,
    load_section_arm,
    open_sections_csv,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

_SECTIONS_PRELUDE = f"# format={MEMMAP_FORMAT_VERSION}\n"
_SIDECAR_PRELUDE = f"# format={MEMMAP_FORMAT_VERSION}\n"


# ---------------------------------------------------------------------------
# Fixture builders.
#
# Builds a synthetic corpus exercising:
#   * Variable-length function names (lengths 7-25) so per-function CSV
#     section starts span every mod-4 residue.
#   * One matched function with TWO variants (one overlong).
#   * Other matched functions with one variant each.
#   * One unmatched function with an overlong sentinel record.
#
# Production wire format end-to-end: real ``write_function_binary_data``
# for data records, real ``write_index_entry`` for unmatched v1 index,
# real ``pack_csv_section_index_entry`` for matched pre-v1 index,
# real ``encode_inline_indexer`` for variant-row indexer_hex.
# ---------------------------------------------------------------------------


def _write_normal_record(handle, token_count: int) -> Tuple[int, int]:
    """Write one normal record; return ``(offset, length)``. Each record
    has small distinct insn/block lengths so the per-record geometry is
    valid for the validator's pad/bounds checks.
    """
    insn = np.array([1, 2, 3], dtype=np.uint8)
    block = np.array([4, 5], dtype=np.uint8)
    tokens = np.arange(token_count, dtype=np.uint16)
    result = write_function_binary_data(handle, tokens, block, insn)
    assert result is not None
    return result


def _write_overlong_record(handle, token_count: int) -> Tuple[int, int]:
    """Write one overlong record (forces the index's sentinel marker)."""
    insn_len = (1 << 18) + 7  # > 256 KiB to force overlong
    insn = np.zeros(insn_len, dtype=np.uint8)
    block = np.array([1, 2, 3], dtype=np.uint8)
    tokens = np.arange(token_count, dtype=np.uint16)
    result = write_function_binary_data(handle, tokens, block, insn)
    assert result is not None
    return result


def _write_function_names_sidecar(path: Path, names: List[str]) -> Dict[str, int]:
    """Write the prelude + sorted-deduped names; return name->line_no."""
    sorted_names = sorted(set(names))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(_SIDECAR_PRELUDE)
        for name in sorted_names:
            f.write(name + "\n")
    return {name: i + 1 for i, name in enumerate(sorted_names)}


def _build_matched_corpus(
    tmp_path: Path,
    binary_name: str = "bin",
) -> Tuple[Dict[str, int], Dict[str, List[Tuple[int, int, bool]]], List[str]]:
    """Build a synthetic matched corpus.

    Functions (variable-length names to span CSV-offset mod-4 residues):
      * "fn_short"               -- 1 variant, normal record
      * "fn_with_two_variants"   -- 2 variants, one normal + one overlong
      * "fn_medium_name_here"    -- 1 variant, normal record
      * "fn_longer_name_for_corpus_test"  -- 1 variant, normal record

    Returns ``(name_to_line, name_to_variants, ordered_func_names)``.
    ``name_to_variants[fn]`` is the list of ``(data_offset, data_len,
    is_overlong)`` for that function's variants (in write order).
    """
    matched_data = tmp_path / f"{binary_name}_data.bin"
    funcs: List[Tuple[str, List[Tuple[bool, int]]]] = [
        ("fn_short", [(False, 5)]),  # 8 chars
        ("fn_with_two_variants", [(False, 7), (True, 9)]),  # 20 chars
        ("fn_medium_name_here", [(False, 11)]),  # 19 chars
        ("fn_longer_name_for_corpus_test", [(False, 13)]),  # 30 chars
    ]
    name_to_variants: Dict[str, List[Tuple[int, int, bool]]] = {}
    with open(matched_data, "wb") as f:
        for func_name, variant_specs in funcs:
            entries: List[Tuple[int, int, bool]] = []
            for is_overlong, token_count in variant_specs:
                if is_overlong:
                    offset, length = _write_overlong_record(f, token_count)
                else:
                    offset, length = _write_normal_record(f, token_count)
                entries.append((offset, length, is_overlong))
            name_to_variants[func_name] = entries

    # Function-names sidecar.
    name_to_line = _write_function_names_sidecar(
        tmp_path / f"{binary_name}_function_names.txt",
        [name for name, _ in funcs],
    )

    # Sections CSV: header row [base64_line_no, called_funcs_csv_b64];
    # variant rows [variant_ref, inlining_str, indexer_hex]; trailing
    # blank row separator. We record csv_starts/csv_lengths so we can
    # write the pre-v1 matched_index.bin with the correct locators.
    sections_csv = tmp_path / f"{binary_name}_sections.csv"
    csv_index: List[Tuple[int, int, int]] = []  # (csv_offset, csv_len, avg_len)
    with open(sections_csv, "w", newline="", encoding="ascii") as f:
        f.write(_SECTIONS_PRELUDE)
        prelude_size = f.tell()
        writer = csv.writer(f)
        for func_name, variant_specs in funcs:
            csv_offset = f.tell() - prelude_size
            line_no_b64 = format_function_line_no(name_to_line[func_name])
            # No called funcs in this synthetic corpus -> empty CSV cell.
            called_b64 = format_function_line_nos_csv([])
            writer.writerow([line_no_b64, called_b64])
            total_token_len = 0
            for (is_overlong, token_count), (offset, length, _ovl) in zip(
                variant_specs, name_to_variants[func_name]
            ):
                indexer_hex = encode_inline_indexer(offset, length)
                writer.writerow(["0x0", "", indexer_hex])
                total_token_len += token_count
            writer.writerow([])
            csv_len = (f.tell() - prelude_size) - csv_offset
            avg_len = total_token_len // len(variant_specs)
            csv_index.append((csv_offset, csv_len, avg_len))

    # matched_index.bin in pre-v1 layout (8 bytes per entry, no prelude).
    matched_index = tmp_path / f"{binary_name}_index.bin"
    with open(matched_index, "wb") as f:
        for csv_offset, csv_len, avg_len in csv_index:
            f.write(pack_csv_section_index_entry(csv_offset, csv_len, avg_len))

    ordered_names = [name for name, _ in funcs]
    return name_to_line, name_to_variants, ordered_names


def _build_unmatched_corpus(
    tmp_path: Path,
    binary_name: str = "bin",
    existing_name_to_line: Dict[str, int] | None = None,
) -> Tuple[Dict[str, int], List[Tuple[int, int, bool]], List[str]]:
    """Build a synthetic unmatched corpus.

    One unmatched function with an overlong record + one normal.
    Returns ``(name_to_line, records, ordered_unmatched_names)``.
    """
    unmatched_data = tmp_path / f"{binary_name}_unmatched_data.bin"
    unmatched_funcs = ["lone_fn_one", "another_unmatched_fn_longer"]
    records: List[Tuple[int, int, bool]] = []
    with open(unmatched_data, "wb") as f:
        n_off, n_len = _write_normal_record(f, token_count=10)
        records.append((n_off, n_len, False))
        o_off, o_len = _write_overlong_record(f, token_count=42)
        records.append((o_off, o_len, True))

    # If a matched sidecar was already written we extend it; otherwise
    # create a fresh sidecar over the unmatched names.
    all_names = list(unmatched_funcs)
    if existing_name_to_line:
        all_names = list(existing_name_to_line.keys()) + list(unmatched_funcs)
    name_to_line = _write_function_names_sidecar(
        tmp_path / f"{binary_name}_function_names.txt", all_names
    )

    # v1 unmatched_index.bin.
    unmatched_index = tmp_path / f"{binary_name}_unmatched_index.bin"
    with open(unmatched_index, "wb") as f:
        write_index_prelude(f)
        for offset, length, _ovl in records:
            write_index_entry(f, offset, length, 0)

    # 5-cell unmatched_sections.csv (line_no_b64, variant_refs, called,
    # inlining, indexer_hex). One row per function.
    sections_csv = tmp_path / f"{binary_name}_unmatched_sections.csv"
    with open(sections_csv, "w", newline="", encoding="ascii") as f:
        f.write(_SECTIONS_PRELUDE)
        writer = csv.writer(f)
        for func_name, (offset, length, _ovl) in zip(unmatched_funcs, records):
            indexer_hex = encode_inline_indexer(offset, length)
            line_no_b64 = format_function_line_no(name_to_line[func_name])
            writer.writerow([line_no_b64, "0x0", "", "", indexer_hex])

    return name_to_line, records, unmatched_funcs


def _matched_paths(tmp_path: Path, binary_name: str = "bin") -> BinaryArmPaths:
    return BinaryArmPaths(
        sections_csv=tmp_path / f"{binary_name}_sections.csv",
        index_bin=tmp_path / f"{binary_name}_index.bin",
        data_bin=tmp_path / f"{binary_name}_data.bin",
    )


def _unmatched_paths(tmp_path: Path, binary_name: str = "bin") -> BinaryArmPaths:
    return BinaryArmPaths(
        sections_csv=tmp_path / f"{binary_name}_unmatched_sections.csv",
        index_bin=tmp_path / f"{binary_name}_unmatched_index.bin",
        data_bin=tmp_path / f"{binary_name}_unmatched_data.bin",
    )


# ---------------------------------------------------------------------------
# Matched-arm tests (post-restructuring)
# ---------------------------------------------------------------------------


def test_matched_arm_flattens_variants_to_per_record_arrays(tmp_path):
    """``starts`` / ``lengths`` / ``is_overlong`` are per-VARIANT, flattened
    across functions in CSV (== build) order."""
    name_to_line, name_to_variants, ordered_names = _build_matched_corpus(tmp_path)
    arm = load_section_arm(
        SectionKind.MATCHED, _matched_paths(tmp_path), {v: k for k, v in name_to_line.items()}
    )

    expected_records: List[Tuple[int, int, bool]] = []
    for name in ordered_names:
        expected_records.extend(name_to_variants[name])

    assert arm.record_count == len(expected_records)
    expected_starts = [r[0] for r in expected_records]
    expected_overlong = [r[2] for r in expected_records]
    assert arm.starts.tolist() == expected_starts
    assert arm.is_overlong.tolist() == expected_overlong
    # Stored length is 0 for overlong (sentinel) and real length otherwise.
    for arm_len, (_o, length, is_overlong) in zip(arm.lengths.tolist(), expected_records):
        if is_overlong:
            assert arm_len == 0
        else:
            assert arm_len == length


def test_matched_arm_func_names_resolved_via_sidecar(tmp_path):
    """``func_names`` is per-FUNCTION (one entry per matched section)
    and resolves base64 line numbers through the sidecar's line_to_name.
    """
    name_to_line, _, ordered_names = _build_matched_corpus(tmp_path)
    line_to_name = {v: k for k, v in name_to_line.items()}
    arm = load_section_arm(
        SectionKind.MATCHED, _matched_paths(tmp_path), line_to_name
    )
    assert arm.func_names == ordered_names
    assert arm.count == len(ordered_names)
    # csv_starts/csv_lengths cardinality matches per-function.
    assert len(arm.csv_starts) == len(ordered_names)
    assert len(arm.csv_lengths) == len(ordered_names)
    assert len(arm.avg_lengths) == len(ordered_names)


def test_matched_arm_csv_starts_not_4_aligned(tmp_path):
    """Variable-length function names (8/20/19/30 chars) produce CSV
    section starts at multiple mod-4 residues. This is the bug the
    pre-restructuring v1 reader would silently corrupt (alignment
    assertion); after restructuring the pre-v1 reader accepts them.
    """
    name_to_line, _, _ = _build_matched_corpus(tmp_path)
    line_to_name = {v: k for k, v in name_to_line.items()}
    arm = load_section_arm(
        SectionKind.MATCHED, _matched_paths(tmp_path), line_to_name
    )
    residues = {int(s) % 4 for s in arm.csv_starts}
    assert len(residues) > 1, (
        f"fixture failed to span mod-4 residues: {residues}"
    )


def test_matched_arm_overlong_flag_correct_for_variant(tmp_path):
    """The overlong-second-variant of ``fn_with_two_variants`` is the
    only overlong record in the fixture; ``is_overlong`` lights up
    exactly for it.
    """
    name_to_line, name_to_variants, ordered_names = _build_matched_corpus(tmp_path)
    line_to_name = {v: k for k, v in name_to_line.items()}
    arm = load_section_arm(
        SectionKind.MATCHED, _matched_paths(tmp_path), line_to_name
    )
    # Index of fn_with_two_variants' overlong variant in the flattened
    # records: fn_short (1 variant) + fn_with_two_variants (variant 0:
    # normal, variant 1: overlong) -> overlong is at flat index 2.
    assert arm.is_overlong.tolist().count(True) == 1
    overlong_flat_idx = arm.is_overlong.tolist().index(True)
    assert overlong_flat_idx == 2
    # And the stored length sentinel for it is 0.
    assert int(arm.lengths[overlong_flat_idx]) == 0


def test_matched_arm_avg_lengths_per_function(tmp_path):
    """``avg_lengths`` matches what the writer stamped into the pre-v1
    index: ``min(avg_token_count >> 4, 255)`` per function.
    """
    name_to_line, name_to_variants, ordered_names = _build_matched_corpus(tmp_path)
    line_to_name = {v: k for k, v in name_to_line.items()}
    arm = load_section_arm(
        SectionKind.MATCHED, _matched_paths(tmp_path), line_to_name
    )
    # Compute expected: per-function avg token_count then clamp >> 4.
    expected_token_counts = {
        "fn_short": [5],
        "fn_with_two_variants": [7, 9],
        "fn_medium_name_here": [11],
        "fn_longer_name_for_corpus_test": [13],
    }
    expected = [
        min((sum(tc) // len(tc)) >> 4, 255)
        for tc in (expected_token_counts[n] for n in ordered_names)
    ]
    assert arm.avg_lengths.tolist() == expected


# ---------------------------------------------------------------------------
# Unmatched-arm tests
# ---------------------------------------------------------------------------


def test_unmatched_arm_5_cell_rows_decoded(tmp_path):
    """Unmatched arm exposes one record per function (1:1) with
    func_names resolved via sidecar; ``is_overlong`` flags the
    overlong sentinel record."""
    name_to_line, records, ordered_names = _build_unmatched_corpus(tmp_path)
    line_to_name = {v: k for k, v in name_to_line.items()}
    arm = load_section_arm(
        SectionKind.UNMATCHED, _unmatched_paths(tmp_path), line_to_name
    )
    assert arm.func_names == ordered_names
    assert arm.record_count == len(records)
    # The overlong-record entry has lengths==0 (sentinel) at index 1.
    assert int(arm.lengths[0]) != 0  # normal
    assert int(arm.lengths[1]) == 0  # overlong sentinel
    assert arm.is_overlong.tolist() == [False, True]


# ---------------------------------------------------------------------------
# Hard-cutover smokes
# ---------------------------------------------------------------------------


def test_legacy_4_cell_matched_variant_raises(tmp_path):
    """A matched_sections.csv whose variant row has the legacy 4-cell
    shape (``[variant_ref, inlining_str, data_offset_hex, data_len_hex]``)
    raises with a migration-pointing message."""
    name_to_line, _, ordered_names = _build_matched_corpus(tmp_path)
    sections = tmp_path / "bin_sections.csv"
    # Overwrite the section CSV with the same prelude + one section
    # whose variant row carries 4 cells (legacy shape).
    with open(sections, "w", newline="", encoding="ascii") as f:
        f.write(_SECTIONS_PRELUDE)
        prelude_size = f.tell()
        writer = csv.writer(f)
        line_no_b64 = format_function_line_no(name_to_line[ordered_names[0]])
        writer.writerow([line_no_b64, ""])
        writer.writerow(["0x0", "", "deadbeef", "1234"])  # 4 cells
        writer.writerow([])
        csv_len = (f.tell() - prelude_size)
    # Rewrite the pre-v1 matched_index.bin with one entry pointing at
    # the single section.
    matched_index = tmp_path / "bin_index.bin"
    with open(matched_index, "wb") as f:
        f.write(pack_csv_section_index_entry(0, csv_len, 16))

    line_to_name = {v: k for k, v in name_to_line.items()}
    with pytest.raises(ValueError) as info:
        load_section_arm(
            SectionKind.MATCHED, _matched_paths(tmp_path), line_to_name
        )
    msg = str(info.value)
    assert "legacy 4-cell" in msg
    assert "re-run memmap_builder" in msg


def test_missing_function_names_sidecar_raises_via_binary_dataset(tmp_path):
    """A binary directory missing ``<binary>_function_names.txt`` while
    one of the index files exists trips the hard cutover at
    ``BinaryDataset`` construction."""
    _build_matched_corpus(tmp_path)
    # Delete the sidecar the fixture wrote.
    sidecar = tmp_path / "bin_function_names.txt"
    sidecar.unlink()
    with pytest.raises(ValueError) as info:
        BinaryDataset(tmp_path, "bin")
    msg = str(info.value)
    # The function_names_loader error mentions the prelude / re-run.
    assert "re-run memmap_builder" in msg


def test_bad_sidecar_prelude_raises_via_binary_dataset(tmp_path):
    """A sidecar present but with a wrong / missing prelude line trips
    the cutover with the migration-pointing message."""
    _build_matched_corpus(tmp_path)
    sidecar = tmp_path / "bin_function_names.txt"
    sidecar.write_text("not_a_format_line\nfn_short\n", encoding="utf-8")
    with pytest.raises(ValueError) as info:
        BinaryDataset(tmp_path, "bin")
    assert "function-names sidecar prelude" in str(info.value)


def test_binary_dataset_publishes_new_arm_fields(tmp_path):
    """The new per-record / per-function attributes mirror the SectionArm
    fields on both arms."""
    name_to_line, _, _ = _build_matched_corpus(tmp_path)
    _build_unmatched_corpus(
        tmp_path, existing_name_to_line=name_to_line
    )
    dataset = BinaryDataset(tmp_path, "bin")
    # New per-record arm field on matched (per-variant flat).
    assert np.array_equal(dataset.matched_is_overlong, dataset._matched_arm.is_overlong)
    assert np.array_equal(dataset.matched_csv_starts, dataset._matched_arm.csv_starts)
    assert np.array_equal(dataset.matched_csv_lengths, dataset._matched_arm.csv_lengths)
    assert np.array_equal(dataset.matched_avg_lengths, dataset._matched_arm.avg_lengths)
    # Unmatched mirror.
    assert np.array_equal(dataset.unmatched_is_overlong, dataset._unmatched_arm.is_overlong)


# ---------------------------------------------------------------------------
# Empty-corpus paths kept from the legacy test set.
# ---------------------------------------------------------------------------


def test_empty_arm_when_index_missing(tmp_path):
    """Missing index file yields the canonical empty arm with
    dtype-preserving placeholders."""
    paths = BinaryArmPaths(
        sections_csv=tmp_path / "absent_sections.csv",
        index_bin=tmp_path / "absent_index.bin",
        data_bin=tmp_path / "absent_data.bin",
    )
    arm = load_section_arm(SectionKind.MATCHED, paths, {})
    assert arm.count == 0
    assert arm.record_count == 0
    assert arm.starts.dtype == np.int64
    assert arm.lengths.dtype == np.uint32
    assert arm.is_overlong.dtype == np.bool_
    assert arm.csv_starts.dtype == np.int64
    assert arm.csv_lengths.dtype == np.uint32
    assert arm.avg_lengths.dtype == np.uint8


def test_section_kind_enum_is_closed_typed():
    """Sanity: ``SectionKind`` is an enum (not a bool), and both
    arms are registered."""
    assert SectionKind.MATCHED is not SectionKind.UNMATCHED
    assert SectionKind.MATCHED.value == "matched"
    assert SectionKind.UNMATCHED.value == "unmatched"


def test_open_sections_csv_still_validates_prelude(tmp_path):
    """The shared prelude validator continues to gate every section
    CSV; ``BinaryDataset`` constructions route through it."""
    path = tmp_path / "x_sections.csv"
    path.write_text(_SECTIONS_PRELUDE + "row\n", encoding="ascii")
    f, content_offset = open_sections_csv(path)
    try:
        assert content_offset == len(_SECTIONS_PRELUDE.encode("ascii"))
    finally:
        f.close()
