"""End-to-end prelude assertions for ``build_memmap_files``.

Drives the existing tiny-synthetic-corpus fixture (no function bodies,
just the v2 vocab + padding line) end-to-end and asserts every file
the builder writes carries the prelude its layout requires:

  * matched + unmatched ``_sections.csv`` files start with ``# format=N\\n``
  * ``unmatched_index.bin`` starts with ``IDX1`` and the 16-byte v1
    prelude decodes to ``format_version=N`` + ``alignment_shift=2``
  * ``matched_index.bin`` has NO prelude (pre-v1 layout, function-to-
    CSV-section locator only) -- on a zero-function corpus the file
    is therefore empty
  * slim ``_variants.csv`` starts with ``# format=N\\n``
  * ``<binary>.error.log`` is opened on every build (even empty)

The corpus is intentionally body-empty -- pass 1 iterates zero functions,
so no data records are emitted; the prelude bytes (when present) are
the entire ``_index.bin`` payload. That isolates the prelude concern
from the record encoder.
"""

from __future__ import annotations

import csv
import struct
from pathlib import Path
from typing import List

from tokenizer.aligned_data.index_format import (
    ALIGNMENT_SHIFT,
    INDEX_HEADER_SIZE,
    INDEX_MAGIC,
    read_index_prelude,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION
from tokenizer.memmap_builder.builder import (
    BinaryVersionInfo,
    build_memmap_files,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.vocab_unifier.saver import save_vocabulary
from tokenizer.vocab_unifier.unifier import unify_vocab


_PADDING_LINE = "function_name,binary_addr," + ("x" * 64) + "\n"
_EXPECTED_PRELUDE_LINE = f"# format={MEMMAP_FORMAT_VERSION}\n"


def _write_per_binary_csv(csv_path: Path, platform: str) -> None:
    vm = VocabularyManager(platform=platform, format_version=2)
    for bid in (0, 1, 2):
        vm.Block_V2(bid)
    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        fh.write(_PADDING_LINE)
        writer = csv.writer(fh)
        save_vocabulary(vm, writer)


def _build_tiny_corpus(tmp_path: Path) -> tuple[List[BinaryVersionInfo], Path, Path]:
    """Materialise a 2-binary, zero-function corpus + unified vocab.

    Returns (versions, unified_vocab_path, output_dir).
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

    versions = [
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

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    return versions, unified_vocab_path, output_dir


def _first_line(path: Path) -> str:
    with open(path, encoding="ascii") as fh:
        return fh.readline()


def test_matched_sections_csv_has_prelude(tmp_path: Path) -> None:
    versions, unified_vocab_path, output_dir = _build_tiny_corpus(tmp_path)
    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)

    assert _first_line(output_dir / "demo_sections.csv") == _EXPECTED_PRELUDE_LINE


def test_unmatched_sections_csv_has_prelude(tmp_path: Path) -> None:
    versions, unified_vocab_path, output_dir = _build_tiny_corpus(tmp_path)
    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)

    assert (
        _first_line(output_dir / "demo_unmatched_sections.csv")
        == _EXPECTED_PRELUDE_LINE
    )


def test_slim_variants_csv_has_prelude(tmp_path: Path) -> None:
    versions, unified_vocab_path, output_dir = _build_tiny_corpus(tmp_path)
    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)

    assert _first_line(output_dir / "demo_variants.csv") == _EXPECTED_PRELUDE_LINE


def test_matched_index_bin_has_no_prelude(tmp_path: Path) -> None:
    """``matched_index.bin`` is pre-v1 layout -- no ``IDX1`` magic, no
    16-byte file header. On a zero-function corpus the file is empty:
    the opener writes nothing and no entries are appended."""
    versions, unified_vocab_path, output_dir = _build_tiny_corpus(tmp_path)
    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)

    raw = (output_dir / "demo_index.bin").read_bytes()
    assert raw == b"", (
        f"matched_index.bin must be empty on a zero-function corpus "
        f"(no prelude, no entries); got {len(raw)} bytes starting with "
        f"{raw[:8]!r}"
    )


def test_unmatched_index_bin_starts_with_magic(tmp_path: Path) -> None:
    versions, unified_vocab_path, output_dir = _build_tiny_corpus(tmp_path)
    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)

    raw = (output_dir / "demo_unmatched_index.bin").read_bytes()
    assert raw[:4] == INDEX_MAGIC
    assert len(raw) >= INDEX_HEADER_SIZE


def test_unmatched_index_prelude_decodes_to_v1(tmp_path: Path) -> None:
    versions, unified_vocab_path, output_dir = _build_tiny_corpus(tmp_path)
    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)

    with open(output_dir / "demo_unmatched_index.bin", "rb") as fh:
        prelude = read_index_prelude(fh)
    assert prelude.format_version == MEMMAP_FORMAT_VERSION
    assert prelude.alignment_shift == ALIGNMENT_SHIFT


def test_error_log_file_exists_and_is_empty_for_clean_corpus(tmp_path: Path) -> None:
    """A clean (no-overflow) build still opens + closes ``<binary>.error.log``.

    The file is the single chokepoint for cap-overflow reporting; its
    presence (even when empty) means downstream consumers can assume the
    contract is honoured every run.
    """
    versions, unified_vocab_path, output_dir = _build_tiny_corpus(tmp_path)
    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)

    error_log_path = output_dir / "demo.error.log"
    assert error_log_path.exists()
    assert error_log_path.read_text(encoding="ascii") == ""


def test_unmatched_index_header_size_is_exactly_16_bytes(tmp_path: Path) -> None:
    """The v1 unmatched-index prelude width is fixed at 16 bytes;
    corpus-empty builds produce an unmatched-index file that is
    exactly that size. The matched-index file stays empty under the
    pre-v1 layout (see ``test_matched_index_bin_has_no_prelude``)."""
    versions, unified_vocab_path, output_dir = _build_tiny_corpus(tmp_path)
    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)

    raw_unmatched = (output_dir / "demo_unmatched_index.bin").read_bytes()
    # No function bodies in the fixture -> no entries; file IS the prelude.
    assert len(raw_unmatched) == INDEX_HEADER_SIZE
    # And the reserved + version + magic words all unpack cleanly.
    magic, fmt_v, align, reserved = struct.unpack("<4sIII", raw_unmatched)
    assert magic == INDEX_MAGIC
    assert fmt_v == MEMMAP_FORMAT_VERSION
    assert align == ALIGNMENT_SHIFT
    assert reserved == 0
