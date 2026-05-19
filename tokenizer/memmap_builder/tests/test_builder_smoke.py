"""End-to-end smoke for ``build_memmap_files`` against a tiny synthetic corpus.

Asserts the per-binary sidecar contract after the 4B rewire:

  * ``<bin>_variants.bin`` exists (variant-axis token records).
  * ``<bin>_variants.csv`` has exactly the slim ``filename,offset``
    header — no ``length``, ``arch``, ``compiler``, ``version``,
    ``opt``, ``pkg``, or ``flags`` columns.
  * No ``<bin>_versions.json`` is emitted.
  * Each row's ``offset`` cell, used as a byte offset into the bin,
    decodes (via ``variant_tokens.record.read_record`` + the unified
    vocab) to the axes the discovery would produce for that variant.

The corpus is the minimum that satisfies ``lockstep_function_match``:
per-binary CSVs that carry a real v2 vocab def line written through
``save_vocabulary`` (same path the tokenize worker uses) and one
padding line so ``read_last_line_of_file`` finds a newline outside its
64-byte tail. Production CSV bodies have function rows; those are
inessential for the variant-sidecar invariants this test guards, so
the body is empty and the matched / unmatched section passes iterate
zero functions but still emit the empty-section CSVs the dataloader
expects.

A real ``unified_vocab.csv`` (v3) is produced via ``unify_vocab``
against the same synthetic CSVs — exercises the live vocab loader the
builder consumes rather than hand-rolling a fake vocab file.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np

from tokenizer.memmap_builder.builder import (
    BinaryVersionInfo,
    build_memmap_files,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.variant_tokens.prefixes import build_axis_strings
from tokenizer.variant_tokens.record import read_record
from tokenizer.vocab_unifier.loader import load_unified_vocab_manager
from tokenizer.vocab_unifier.saver import save_vocabulary
from tokenizer.vocab_unifier.unifier import unify_vocab


# Padding line so the file body exceeds the 64-byte tail that
# ``read_last_line_of_file`` excludes (it searches the body *before*
# the last 64 bytes for a newline). Same shape as the vocab_unifier
# integration test's padding line — keeps the synthesised CSV
# structurally close to a real per-binary file.
_PADDING_LINE = "function_name,binary_addr," + ("x" * 64) + "\n"


def _write_per_binary_csv(csv_path: Path, platform: str) -> None:
    """Write a synthetic v2 per-binary CSV at ``csv_path``.

    Registers a handful of ``Block_V2`` instruction tokens onto a
    platform-scoped v2 ``VocabularyManager``; saves the vocab def line
    as the file's final row, preceded by one padding line so
    ``read_last_line_of_file`` can find a newline outside the 64-byte
    tail and ``lockstep_function_match`` consumes the padding as a
    no-op header before iterating zero function rows.
    """
    vm = VocabularyManager(platform=platform, format_version=2)
    for bid in (0, 1, 2):
        vm.Block_V2(bid)

    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        fh.write(_PADDING_LINE)
        writer = csv.writer(fh)
        save_vocabulary(vm, writer)


def _build_synthetic_corpus(tmp_path: Path) -> tuple[List[Path], Path]:
    """Lay down two per-binary v2 CSVs + a v3 unified vocab; return
    the CSV paths and the unified-vocab path.

    Filenames follow the production default (
    ``<arch>-<compiler>-<ver>-<opt>_<pkg>_output.csv``) so
    ``find_matching_binaries`` would also parse them in a CLI context;
    this test doesn't invoke the CLI but uses the same shape so the
    inputs are realistic.
    """
    csv_files: List[Path] = []
    for basename, arch in [
        ("x64-gcc-13.2.0-O2_pkga", "x64"),
        ("arm64-clang-15.0.0-O3_pkgb", "arm64"),
    ]:
        path = tmp_path / f"{basename}_output.csv"
        _write_per_binary_csv(path, platform=arch)
        csv_files.append(path)

    unified_vocab_path = tmp_path / "unified_vocab.csv"
    unify_vocab(csv_files, unified_vocab_path)
    return csv_files, unified_vocab_path


def _versions_for(csv_files: List[Path]) -> List[BinaryVersionInfo]:
    """Build the ``BinaryVersionInfo`` list the way the dynrunner
    worker would assemble it post-discovery — one entry per per-binary
    CSV, canonical-4 axes pulled from the filename, ``filename`` set
    so the slim CSV gets a non-empty value.
    """
    return [
        BinaryVersionInfo(
            path=csv_files[0],
            mapping_path=csv_files[0].with_suffix(".mapping.b64c"),
            arch="x64",
            compiler="gcc",
            compilerversion="13.2.0",
            opt="O2",
            pkg="pkga",
            filename="x64-gcc-13.2.0-O2_pkga",
        ),
        BinaryVersionInfo(
            path=csv_files[1],
            mapping_path=csv_files[1].with_suffix(".mapping.b64c"),
            arch="arm64",
            compiler="clang",
            compilerversion="15.0.0",
            opt="O3",
            pkg="pkgb",
            filename="arm64-clang-15.0.0-O3_pkgb",
        ),
    ]


def test_build_memmap_files_emits_slim_csv_bin_no_versions_json(tmp_path: Path) -> None:
    """End-to-end: build_memmap_files against a tiny synthetic corpus
    drops the legacy ``_versions.json`` sidecar and emits the slim
    ``_variants.csv`` (header ``filename,offset``) + the new
    ``_variants.bin``."""
    csv_files, unified_vocab_path = _build_synthetic_corpus(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    versions = _versions_for(csv_files)

    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)

    # Variants bin + slim CSV exist; legacy versions.json does NOT.
    assert (output_dir / "demo_variants.bin").exists()
    assert (output_dir / "demo_variants.csv").exists()
    assert not (output_dir / "demo_versions.json").exists()

    # Slim CSV: first physical line is the format prelude; remainder is
    # the standard CSV header + data rows.
    with open(output_dir / "demo_variants.csv", encoding="ascii") as fh:
        first_line = fh.readline()
        rows = list(csv.reader(fh))
    assert first_line == "# format=1\n"
    assert rows[0] == ["filename", "offset"]
    assert len(rows) == 3  # header + 2 data rows
    for row in rows[1:]:
        assert len(row) == 2, (
            f"slim CSV row must have exactly 2 columns (filename,offset), "
            f"got {len(row)}: {row!r}"
        )

    # Data rows carry the version filenames in registration order.
    data_filenames = [r[0] for r in rows[1:]]
    assert data_filenames == [
        "x64-gcc-13.2.0-O2_pkga",
        "arm64-clang-15.0.0-O3_pkgb",
    ]


def test_variant_ref_offsets_decode_to_input_axes(tmp_path: Path) -> None:
    """The CSV's ``offset`` column is the byte offset into
    ``_variants.bin``. Reading the bin at that offset must yield a
    record whose token IDs (resolved through the unified vocab) match
    the axis strings the encoder built from the input version.

    Covers the headline 4B + 4A invariant: section-CSV ``variant_ref``
    cells are byte offsets into the bin, not row indices, and the new
    semantics produce a faithful round-trip against the live unified
    vocab.
    """
    csv_files, unified_vocab_path = _build_synthetic_corpus(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    versions = _versions_for(csv_files)

    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)

    # Load the unified vocab the builder used so we can match token
    # IDs back to their axis strings.
    unified_vocab = load_unified_vocab_manager(unified_vocab_path)
    assert unified_vocab is not None
    assert unified_vocab.format_version == 1

    # Memmap the bin + parse the slim CSV's filename->offset table.
    bin_mmap = np.memmap(
        output_dir / "demo_variants.bin", dtype=np.uint8, mode="r"
    )
    with open(output_dir / "demo_variants.csv", encoding="ascii") as fh:
        fh.readline()  # skip "# format=N" prelude
        csv_rows = list(csv.reader(fh))
    filename_to_offset = {row[0]: int(row[1], 16) for row in csv_rows[1:]}

    # For each version, decode the bin slice at its offset and confirm
    # the recovered axis-token strings equal those the encoder would
    # produce from the same version (round-trip). The record's
    # ``tokens[0]`` is the n_tokens size header; ``tokens[1:]`` are the
    # axis IDs.
    for version in versions:
        offset = filename_to_offset[version.filename]
        record = read_record(bin_mmap, offset)
        n_tokens = int(record[0])
        decoded_tokens = [
            unified_vocab.get_token_str(int(tid)) for tid in record[1:1 + n_tokens]
        ]
        expected_tokens = build_axis_strings(version)
        assert decoded_tokens == expected_tokens, (
            f"variant {version.filename!r} at offset {offset:#x} decoded "
            f"to {decoded_tokens!r}, expected {expected_tokens!r}"
        )


def test_build_memmap_files_rejects_v2_unified_vocab(tmp_path: Path) -> None:
    """A v2 unified vocab is structurally incompatible (no variant-axis
    tokens registered). build_memmap_files must reject it loudly rather
    than silently emit bin records whose token IDs are stub lookups."""
    csv_files, unified_vocab_path = _build_synthetic_corpus(tmp_path)
    # Hand-roll a v2 unified vocab in place of the v3 one.
    vm_v2 = VocabularyManager(platform=None, format_version=2)
    with open(unified_vocab_path, "w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh)
        save_vocabulary(vm_v2, writer)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    versions = _versions_for(csv_files)

    import pytest

    with pytest.raises(ValueError, match="format_version"):
        build_memmap_files(versions, output_dir, "demo", unified_vocab_path)


def test_build_memmap_files_rejects_missing_unified_vocab(tmp_path: Path) -> None:
    """A nonexistent / unloadable unified-vocab path must surface as a
    ValueError up front, before any sidecar I/O — the registry can't
    encode without it."""
    csv_files, _ = _build_synthetic_corpus(tmp_path)
    bogus_vocab = tmp_path / "does_not_exist.csv"

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    versions = _versions_for(csv_files)

    import pytest

    with pytest.raises(ValueError, match="failed to load unified vocab"):
        build_memmap_files(versions, output_dir, "demo", bogus_vocab)
