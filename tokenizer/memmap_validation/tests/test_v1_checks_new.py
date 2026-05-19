"""Validator pad-consistency + record-bounds checks.

Mirrors ``test_validator_v1_checks`` for the two new invariants added to
``_v1_checks.run_v1_post_checks``: ``check_pad_consistency`` (header's
``pad_size`` agrees with the shared ``compute_pad`` rule) and
``check_record_bounds`` (no record extends past EOF).

The "passes on freshly built corpus" tests reuse the same minimal
``unify_vocab`` + ``build_memmap_files`` pipeline as the sibling test
file (the synthetic corpus has zero functions, so the checks short-
circuit on empty ``starts``; this still proves the wiring + dispatcher
contract holds). The "flags" tests hand-craft a one-record corpus via
``write_function_binary_data`` so the per-record path is exercised even
when the synthetic builder produces no functions.
"""

from __future__ import annotations

import csv
import struct
from pathlib import Path
from typing import List

import numpy as np

from tokenizer.aligned_data.binary_format import HEADER_BYTES
from tokenizer.aligned_data.index_format import (
    INDEX_MAGIC,
    read_index_arrays,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION
from tokenizer.memmap_builder.builder import BinaryVersionInfo, build_memmap_files
from tokenizer.memmap_validation._v1_checks import (
    check_pad_consistency,
    check_record_bounds,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.vocab_unifier.saver import save_vocabulary
from tokenizer.vocab_unifier.unifier import unify_vocab


_PADDING_LINE = "function_name,binary_addr," + ("x" * 64) + "\n"


def _write_per_binary_csv(csv_path: Path, platform: str) -> None:
    vm = VocabularyManager(platform=platform, format_version=2)
    for bid in (0, 1, 2):
        vm.Block_V2(bid)
    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        fh.write(_PADDING_LINE)
        writer = csv.writer(fh)
        save_vocabulary(vm, writer)


def _pipeline(tmp_path: Path) -> Path:
    """Lay down a clean v1 corpus and return ``output_dir``."""
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

    output_dir = tmp_path / "out"
    output_dir.mkdir()
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
    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)
    (output_dir / "unified_vocab.csv").write_bytes(unified_vocab_path.read_bytes())
    return output_dir


def _arrays_or_empty(path: Path):
    """Return ``(starts, lengths)`` for an index, or empty arrays.

    Pre-v1 layout matched_index.bin or a missing file collapses to empty
    arrays so the per-record checks short-circuit harmlessly.
    """
    if not path.exists() or path.stat().st_size == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.uint32))
    arr = read_index_arrays(path)
    if arr is None:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.uint32))
    return arr[0], arr[1]


def _write_one_record_corpus(data_path: Path, index_path: Path) -> tuple[int, int]:
    """Lay down one synthetic v1 record + matching v1 index entry.

    Shapes pick ``HEADER_BYTES (6) + insn_len (1) + block_len (0) +
    2 * token_count (0) = 7``, so the writer adds ``pad_size = 1``. The
    real record total is 8 bytes; index entry encodes ``length>>2 = 2``.
    Returns ``(record_start, record_length)``.
    """
    from tokenizer.aligned_data._writers import write_function_binary_data

    insn = np.arange(1, dtype=np.uint8)
    block = np.zeros(0, dtype=np.uint8)
    tokens = np.zeros(0, dtype=np.uint16)
    with open(data_path, "wb") as fh:
        result = write_function_binary_data(fh, tokens, block, insn)
    assert result is not None
    start, length = result
    assert length % 4 == 0

    prelude = struct.pack("<4sIII", INDEX_MAGIC, MEMMAP_FORMAT_VERSION, 2, 0)
    entry = (
        struct.pack("<Q", start >> 2)[:5]
        + struct.pack("<H", length >> 2)
        + struct.pack("B", 0)
    )
    index_path.write_bytes(prelude + entry)
    return start, length


# ---------------------------------------------------------------------------
# check_pad_consistency
# ---------------------------------------------------------------------------


def test_check_pad_consistency_passes_on_freshly_built_corpus(tmp_path: Path) -> None:
    """Clean v1 corpus: pad headers agree with ``compute_pad`` on both arms.

    The synthetic corpus produces zero functions, so the per-record
    iteration short-circuits on ``len(starts) == 0``; the assertion is
    that the check returns ``[]`` (no false positives + the wiring
    handles empty arms cleanly).
    """
    output_dir = _pipeline(tmp_path)
    for arm in ("", "_unmatched"):
        data_path = output_dir / f"demo{arm}_data.bin"
        index_path = output_dir / f"demo{arm}_index.bin"
        starts, lengths = _arrays_or_empty(index_path)
        errors = check_pad_consistency(data_path, starts, lengths, str(data_path))
        assert errors == [], f"{arm or 'matched'} arm dirty on fresh build: {errors!r}"


def test_check_pad_consistency_flags_corrupted_pad(tmp_path: Path) -> None:
    """Flip the pad-size bits in one record's header; check fires.

    The packed control byte at record offset 0 carries ``pad_size`` in
    bits 2-3. Toggling those bits to a value other than what the body
    geometry implies makes ``header.pad_size != compute_pad(...)``.
    """
    data_path = tmp_path / "tiny_data.bin"
    index_path = tmp_path / "tiny_index.bin"
    record_start, _ = _write_one_record_corpus(data_path, index_path)

    raw = bytearray(data_path.read_bytes())
    # Force pad_size bits (bits 2-3) to a different value: XOR with 0b11
    # so 1 becomes 2, 0 becomes 3, etc. -- guaranteed mismatch.
    raw[record_start] ^= 0b00001100
    data_path.write_bytes(bytes(raw))

    starts, lengths = _arrays_or_empty(index_path)
    errors = check_pad_consistency(data_path, starts, lengths, str(data_path))
    assert any(
        f"start={record_start}" in e and "disagrees with compute_pad" in e for e in errors
    ), f"tampered pad_size should surface, got: {errors!r}"


# ---------------------------------------------------------------------------
# check_record_bounds
# ---------------------------------------------------------------------------


def test_check_record_bounds_passes_on_freshly_built_corpus(tmp_path: Path) -> None:
    """Clean v1 corpus: every record fits within ``_data.bin``."""
    output_dir = _pipeline(tmp_path)
    for arm in ("", "_unmatched"):
        data_path = output_dir / f"demo{arm}_data.bin"
        index_path = output_dir / f"demo{arm}_index.bin"
        starts, lengths = _arrays_or_empty(index_path)
        errors = check_record_bounds(data_path, starts, lengths, str(data_path))
        assert errors == [], f"{arm or 'matched'} arm dirty on fresh build: {errors!r}"


def test_check_record_bounds_flags_truncated_data(tmp_path: Path) -> None:
    """Truncate ``_data.bin`` mid-last-record; ``check_record_bounds`` fires."""
    data_path = tmp_path / "tiny_data.bin"
    index_path = tmp_path / "tiny_index.bin"
    record_start, record_length = _write_one_record_corpus(data_path, index_path)

    original_size = data_path.stat().st_size
    # Lop off ``HEADER_BYTES`` bytes so the last record's real end
    # exceeds file_size. For the one-record corpus that truncates the
    # only record's tail.
    raw = data_path.read_bytes()[: original_size - HEADER_BYTES]
    data_path.write_bytes(raw)
    new_size = data_path.stat().st_size

    starts, lengths = _arrays_or_empty(index_path)
    errors = check_record_bounds(data_path, starts, lengths, str(data_path))
    expected_end = record_start + record_length
    assert any(
        f"start={record_start}" in e
        and f"file_size={new_size}" in e
        and f"extends to {expected_end}" in e
        for e in errors
    ), f"truncated data should surface, got: {errors!r}"
