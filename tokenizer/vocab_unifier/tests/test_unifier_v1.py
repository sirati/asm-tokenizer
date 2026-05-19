"""Unifier output is stamped at ``MEMMAP_FORMAT_VERSION``.

Three concerns, one file:

1. End-to-end ``unify_vocab`` produces a unified CSV whose
   ``format_version`` equals ``MEMMAP_FORMAT_VERSION`` (and equals 1
   — the value of the constant at time of writing — so a stray bump
   without a corresponding cascade rebuild is loud).
2. Per-binary v2 CSVs remain accepted as input — the boundary check
   at ``unifier.py:96`` is unchanged, and out-of-scope per the plan.
3. ``discover_and_register_variants`` asserts the unified VM is at
   ``MEMMAP_FORMAT_VERSION``; any other version raises.

Tests use only the public unifier surface + the public loader
``load_unified_vocab_manager`` (sibling subtask owns the loader), so
no internal coupling beyond what the spec freezes.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION
from tokenizer.token_manager import VocabularyManager
from tokenizer.vocab_unifier.loader import load_unified_vocab_manager
from tokenizer.vocab_unifier.saver import save_vocabulary
from tokenizer.vocab_unifier.unifier import unify_vocab
from tokenizer.vocab_unifier.variant_registration import (
    discover_and_register_variants,
)


# Padding line so the synthetic per-binary CSV body exceeds the 64
# byte tail ``read_last_line_of_file`` excludes. Mirrors the helper
# used by ``test_unify_two_pass.py`` — keeps the input shape close to
# what the tokenize worker produces in production.
_PADDING_LINE = "function_name,binary_addr," + ("x" * 64) + "\n"


def _write_per_binary_v2_csv(
    csv_path: Path,
    platform: str,
    block_ids: list[int],
) -> None:
    """Write a synthetic per-binary v2 vocab CSV.

    Registers ``Block_V2`` instruction tokens onto a platform-scoped
    ``VocabularyManager(format_version=2)``; saves the vocab def line
    via ``save_vocabulary`` (the production writer), preceded by one
    padding line so the last-line search has a newline target outside
    the 64-byte tail.
    """
    vm = VocabularyManager(platform=platform, format_version=2)
    for bid in block_ids:
        vm.Block_V2(bid)

    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        fh.write(_PADDING_LINE)
        writer = csv.writer(fh)
        save_vocabulary(vm, writer)


def _build_v2_per_binary_corpus(tmp_path: Path) -> list[Path]:
    """Two synthetic per-binary v2 CSVs across two arches.

    Filename schema matches the legacy 4-axis format
    ``<platform>-<compiler>-<version>-<opt>_<pkg>_output.csv`` so the
    sidecar-free variant-discovery path resolves.
    """
    csv_files: list[Path] = []
    for basename, arch in [
        ("x64-gcc-13.2.0-O2_pkga", "x64"),
        ("arm64-clang-15.0.0-O3_pkgb", "arm64"),
    ]:
        path = tmp_path / f"{basename}_output.csv"
        _write_per_binary_v2_csv(path, platform=arch, block_ids=[0, 1, 2])
        csv_files.append(path)
    return csv_files


def test_unifier_emits_memmap_format_version(tmp_path: Path) -> None:
    """End-to-end: ``unify_vocab`` over per-binary v2 inputs writes a
    unified CSV whose ``format_version`` equals
    ``MEMMAP_FORMAT_VERSION``.

    Cross-check against the literal ``1`` so a stray bump of the
    constant without the corresponding cascade rebuild trips this
    test loudly rather than silently aligning.
    """
    csv_files = _build_v2_per_binary_corpus(tmp_path)
    out_csv = tmp_path / "unified_vocab.csv"

    unify_vocab(csv_files, out_csv)

    loaded = load_unified_vocab_manager(out_csv)
    assert loaded is not None, "unified vocab failed to load"
    assert loaded.format_version == MEMMAP_FORMAT_VERSION
    assert loaded.format_version == 1, (
        "MEMMAP_FORMAT_VERSION drifted from 1 without a cascade rebuild"
    )


def test_unifier_accepts_per_binary_v2_input(tmp_path: Path) -> None:
    """The per-binary input check at ``unifier.py:96`` stays at v2.

    Out-of-scope for the memmap-format cleanup per the plan boundary
    note. The synthetic v2 corpus must process without
    ``ValueError`` — the absence of an exception IS the assertion.
    """
    csv_files = _build_v2_per_binary_corpus(tmp_path)
    out_csv = tmp_path / "unified_vocab.csv"

    # No raises = boundary check still tolerates v2 inputs.
    unify_vocab(csv_files, out_csv)


def test_variant_registration_accepts_unified_vm(tmp_path: Path) -> None:
    """``discover_and_register_variants`` accepts a VM at
    ``MEMMAP_FORMAT_VERSION`` without raising."""
    csv_path = tmp_path / "x64-gcc-13.2.0-O2_hello_output.csv"
    csv_path.write_text("")  # filename-only; variant discovery reads no body

    vm = VocabularyManager(platform=None, format_version=MEMMAP_FORMAT_VERSION)
    # Plain call; assertion would raise if the version gate were wrong.
    n = discover_and_register_variants([csv_path], vm)
    assert n > 0, "expected at least one variant registered for legacy 4-axis filename"


@pytest.mark.parametrize("bad_version", [2, 3])
def test_variant_registration_rejects_non_memmap_version(
    tmp_path: Path, bad_version: int,
) -> None:
    """Any VM whose format_version != MEMMAP_FORMAT_VERSION trips the
    assertion in ``discover_and_register_variants``."""
    csv_path = tmp_path / "x64-gcc-13.2.0-O2_hello_output.csv"
    csv_path.write_text("")

    vm = VocabularyManager(platform=None, format_version=bad_version)
    with pytest.raises(AssertionError, match="format_version"):
        discover_and_register_variants([csv_path], vm)
