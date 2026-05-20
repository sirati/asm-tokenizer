"""Shared synthetic-corpus pipeline for memmap_validation tests.

Single concern: build one canonical v1 corpus from two per-binary v2
CSVs so the validator-side tests stop carrying near-identical copies
of the same pipeline. Returns enough handles that callers compose
whichever subset they need:

* ``build_synthetic_corpus`` — write the two per-binary CSVs + unify
  their vocabulary.
* ``builder_versions_for`` / ``validator_versions_for`` — the matching
  ``BinaryVersionInfo`` / ``VersionInfo`` lists.
* ``build_pipeline`` — one-shot wrapper that also runs the
  ``build_memmap_files`` pass and returns the output directory.

Tests that just need ``output_dir`` slice ``build_pipeline``'s tuple;
tests that re-run only specific steps (variants bin cross-check
fixtures) compose the per-step helpers.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

from tokenizer.memmap_builder.builder import (
    BinaryVersionInfo,
    build_memmap_files,
)
from tokenizer.memmap_validation.validator import VersionInfo
from tokenizer.token_manager import VocabularyManager
from tokenizer.vocab_unifier.saver import save_vocabulary
from tokenizer.vocab_unifier.unifier import unify_vocab


# Padding line so the file body exceeds the 64-byte tail
# ``read_last_line_of_file`` excludes (matches the memmap_builder
# smoke-test padding convention).
PADDING_LINE = "function_name,binary_addr," + ("x" * 64) + "\n"


def write_per_binary_csv(csv_path: Path, platform: str) -> None:
    """Write a synthetic v2 per-binary CSV at ``csv_path``."""
    vm = VocabularyManager(platform=platform, format_version=2)
    for bid in (0, 1, 2):
        vm.Block_V2(bid)
    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        fh.write(PADDING_LINE)
        writer = csv.writer(fh, lineterminator='\n')
        save_vocabulary(vm, writer)


def build_synthetic_corpus(tmp_path: Path) -> Tuple[List[Path], Path]:
    """Lay down two per-binary v2 CSVs + a unified vocab.

    Filenames follow the production default so
    ``VariantInfo.from_csv`` parses them the same way the worker would.
    Returns ``(csv_files, unified_vocab_path)``.
    """
    csv_files: List[Path] = []
    for basename, arch in [
        ("x64-gcc-13.2.0-O2_pkga", "x64"),
        ("arm64-clang-15.0.0-O3_pkgb", "arm64"),
    ]:
        path = tmp_path / f"{basename}_output.csv"
        write_per_binary_csv(path, platform=arch)
        csv_files.append(path)

    unified_vocab_path = tmp_path / "unified_vocab.csv"
    unify_vocab(csv_files, unified_vocab_path)
    return csv_files, unified_vocab_path


def builder_versions_for(csv_files: List[Path]) -> List[BinaryVersionInfo]:
    """``BinaryVersionInfo`` list aligned with ``build_synthetic_corpus``."""
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


def validator_versions_for(csv_files: List[Path]) -> List[VersionInfo]:
    """``VersionInfo`` (validator) list aligned with ``build_synthetic_corpus``.

    The validator's ``VersionInfo`` omits ``extra_metadata`` /
    ``filename`` — the cross-check recovers those by re-parsing
    ``VariantInfo.from_csv(csv_path)`` so validator + builder share one
    canonical identity.
    """
    return [
        VersionInfo(
            csv_path=csv_files[0],
            mapping_path=csv_files[0].with_suffix(".mapping.b64c"),
            arch="x64",
            compiler="gcc",
            compilerversion="13.2.0",
            opt="O2",
        ),
        VersionInfo(
            csv_path=csv_files[1],
            mapping_path=csv_files[1].with_suffix(".mapping.b64c"),
            arch="arm64",
            compiler="clang",
            compilerversion="15.0.0",
            opt="O3",
        ),
    ]


def build_pipeline(
    tmp_path: Path, *, copy_vocab_into_output: bool = True
) -> Tuple[List[Path], Path, Path]:
    """One-shot: build corpus, run memmap_builder, return all handles.

    Returns ``(csv_files, unified_vocab_path, output_dir)``. When
    ``copy_vocab_into_output`` is true (the default), also copies the
    unified vocab into ``output_dir`` so the validator's default path
    resolution (mirroring ``AlignedDataLoader``) picks it up.
    """
    csv_files, unified_vocab_path = build_synthetic_corpus(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    versions = builder_versions_for(csv_files)
    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)
    if copy_vocab_into_output:
        (output_dir / "unified_vocab.csv").write_bytes(unified_vocab_path.read_bytes())
    return csv_files, unified_vocab_path, output_dir


def arrays_or_empty(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(starts, lengths)`` for an index, or empty arrays.

    Pre-v1 layout matched_index.bin or a missing file collapses to
    empty arrays so per-record checks short-circuit harmlessly.
    """
    from tokenizer.aligned_data.index_format import read_index_arrays

    if not path.exists() or path.stat().st_size == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.uint32))
    triple = read_index_arrays(path)
    if triple is None:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.uint32))
    return triple[0], triple[1]
