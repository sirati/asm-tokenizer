"""Validator v1-format invariant checks.

Mirrors ``test_variants_bin_check`` for corpus bring-up: a tiny synthetic
v1 corpus via ``unify_vocab`` + ``build_memmap_files``, then assert each
new invariant fires on the matching failure injection and the clean
build stays error-free.
"""

from __future__ import annotations

import csv
import struct
from pathlib import Path
from typing import List

import numpy as np

from tokenizer.aligned_data.binary_format import HEADER_BYTES
from tokenizer.aligned_data.index_format import (
    INDEX_HEADER_SIZE,
    INDEX_MAGIC,
    SENTINEL_LENGTH,
    read_index_arrays,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION
from tokenizer.memmap_builder.builder import BinaryVersionInfo, build_memmap_files
from tokenizer.memmap_validation._v1_checks import (
    check_csv_prelude,
    check_index_prelude,
    check_pad_bytes_zero,
    check_sentinel_overlong_coupling,
    check_starts_alignment,
    run_v1_post_checks,
)
from tokenizer.memmap_validation.validator import (
    ValidatorConfig,
    VersionInfo,
    validate_memmap_output,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.vocab_unifier.saver import save_vocabulary
from tokenizer.vocab_unifier.unifier import unify_vocab


# Same padding-line convention as test_variants_bin_check: ensures the
# per-binary CSV body exceeds the 64-byte tail ``read_last_line_of_file``
# excludes, so the builder accepts the file.
_PADDING_LINE = "function_name,binary_addr," + ("x" * 64) + "\n"


def _write_per_binary_csv(csv_path: Path, platform: str) -> None:
    vm = VocabularyManager(platform=platform, format_version=2)
    for bid in (0, 1, 2):
        vm.Block_V2(bid)
    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        fh.write(_PADDING_LINE)
        writer = csv.writer(fh)
        save_vocabulary(vm, writer)


def _pipeline(tmp_path: Path) -> tuple[Path, Path]:
    """Lay down a clean v1 corpus and return ``(output_dir, vocab_path)``.

    The vocab is copied into ``output_dir`` so the validator's default
    path resolution (mirroring ``AlignedDataLoader``) picks it up.
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
    return output_dir, unified_vocab_path


def _validator_versions(tmp_path: Path) -> List[VersionInfo]:
    return [
        VersionInfo(
            csv_path=tmp_path / "x64-gcc-13.2.0-O2_pkga_output.csv",
            mapping_path=tmp_path / "x64-gcc-13.2.0-O2_pkga_output.mapping.b64c",
            arch="x64",
            compiler="gcc",
            compilerversion="13.2.0",
            opt="O2",
        ),
        VersionInfo(
            csv_path=tmp_path / "arm64-clang-15.0.0-O3_pkgb_output.csv",
            mapping_path=tmp_path / "arm64-clang-15.0.0-O3_pkgb_output.mapping.b64c",
            arch="arm64",
            compiler="clang",
            compilerversion="15.0.0",
            opt="O3",
        ),
    ]


def _run(tmp_path: Path):
    output_dir, _ = _pipeline(tmp_path)
    config = ValidatorConfig(
        versions=_validator_versions(tmp_path),
        output_dir=output_dir,
        binary_name="demo",
    )
    stats = validate_memmap_output(config)
    return output_dir, stats


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_clean_v1_corpus_passes(tmp_path: Path) -> None:
    """End-to-end: the new checks add no errors on a fresh build."""
    _, stats = _run(tmp_path)
    assert stats.errors == [], f"clean corpus should pass v1 checks, got: {stats.errors!r}"


# ---------------------------------------------------------------------------
# Index prelude failures
# ---------------------------------------------------------------------------


def test_missing_index_prelude_raises(tmp_path: Path) -> None:
    """Zeroing the 16-byte prelude removes the magic; validator reports it.

    Targets ``_unmatched_index.bin`` -- the only index file still carrying
    the v1 ``IDX1`` prelude. ``_index.bin`` (matched arm) is pre-v1
    layout (no prelude, function->CSV-section locator); its structural
    integrity is enforced by ``read_csv_section_index_arrays`` at
    ``BinaryDataset`` construction time, not by the validator's prelude
    probe.
    """
    output_dir, _ = _pipeline(tmp_path)
    idx_path = output_dir / "demo_unmatched_index.bin"
    raw = bytearray(idx_path.read_bytes())
    raw[0:INDEX_HEADER_SIZE] = b"\x00" * INDEX_HEADER_SIZE
    idx_path.write_bytes(bytes(raw))

    config = ValidatorConfig(
        versions=_validator_versions(tmp_path),
        output_dir=output_dir,
        binary_name="demo",
    )
    stats = validate_memmap_output(config)
    assert any("missing magic header" in e for e in stats.errors), (
        f"missing magic should surface in errors, got: {stats.errors!r}"
    )


def test_wrong_index_format_version_raises(tmp_path: Path) -> None:
    """Rewrite the prelude with format_version=99; validator reports it.

    Targets ``_unmatched_index.bin`` for the same reason as
    :func:`test_missing_index_prelude_raises`.
    """
    output_dir, _ = _pipeline(tmp_path)
    idx_path = output_dir / "demo_unmatched_index.bin"
    raw = bytearray(idx_path.read_bytes())
    # Preserve magic; rewrite version+shift+reserved with version=99.
    raw[0:INDEX_HEADER_SIZE] = struct.pack("<4sIII", INDEX_MAGIC, 99, 2, 0)
    idx_path.write_bytes(bytes(raw))

    config = ValidatorConfig(
        versions=_validator_versions(tmp_path),
        output_dir=output_dir,
        binary_name="demo",
    )
    stats = validate_memmap_output(config)
    assert any("format_version" in e and "99" in e for e in stats.errors), (
        f"wrong version should surface, got: {stats.errors!r}"
    )


# ---------------------------------------------------------------------------
# Sections / variants CSV prelude failures
# ---------------------------------------------------------------------------


def _rewrite_csv_dropping_prelude(path: Path) -> None:
    """Strip the ``# format=N\\n`` first line; keep the rest as-is."""
    lines = path.read_text(encoding="ascii").splitlines(keepends=True)
    assert lines[0].startswith("# format="), lines[0]
    path.write_text("".join(lines[1:]), encoding="ascii")


def _rewrite_csv_prelude_version(path: Path, version: int) -> None:
    """Replace the ``# format=N\\n`` first line with the given version."""
    lines = path.read_text(encoding="ascii").splitlines(keepends=True)
    assert lines[0].startswith("# format=")
    lines[0] = f"# format={version}\n"
    path.write_text("".join(lines), encoding="ascii")


def test_missing_sections_csv_prelude_raises(tmp_path: Path) -> None:
    output_dir, _ = _pipeline(tmp_path)
    _rewrite_csv_dropping_prelude(output_dir / "demo_sections.csv")

    config = ValidatorConfig(
        versions=_validator_versions(tmp_path),
        output_dir=output_dir,
        binary_name="demo",
    )
    stats = validate_memmap_output(config)
    assert any(
        "demo_sections.csv" in e and "prelude" in e for e in stats.errors
    ), f"missing sections prelude should surface, got: {stats.errors!r}"


def test_wrong_sections_csv_prelude_raises(tmp_path: Path) -> None:
    output_dir, _ = _pipeline(tmp_path)
    _rewrite_csv_prelude_version(output_dir / "demo_sections.csv", 99)

    config = ValidatorConfig(
        versions=_validator_versions(tmp_path),
        output_dir=output_dir,
        binary_name="demo",
    )
    stats = validate_memmap_output(config)
    assert any(
        "demo_sections.csv" in e and "format=99" in e for e in stats.errors
    ), f"wrong sections version should surface, got: {stats.errors!r}"


def test_missing_variants_csv_prelude_raises(tmp_path: Path) -> None:
    output_dir, _ = _pipeline(tmp_path)
    _rewrite_csv_dropping_prelude(output_dir / "demo_variants.csv")

    config = ValidatorConfig(
        versions=_validator_versions(tmp_path),
        output_dir=output_dir,
        binary_name="demo",
    )
    stats = validate_memmap_output(config)
    assert any(
        "demo_variants.csv" in e and "prelude" in e for e in stats.errors
    ), f"missing variants prelude should surface, got: {stats.errors!r}"


# ---------------------------------------------------------------------------
# Starts-alignment check (synthetic — no writer can produce an unaligned start)
# ---------------------------------------------------------------------------


def test_starts_alignment_detects_unaligned_value(tmp_path: Path) -> None:
    """The check is defensive; drive it directly with a hand-crafted array.

    The writer is incapable of producing an unaligned start (offsets are
    written as ``start >> 2``), so the only way to exercise this branch
    is to call the helper with a synthetic ``starts`` array.
    """
    starts = np.array([0, 8, 13, 16], dtype=np.int64)
    errors = check_starts_alignment(starts, "synthetic")
    assert len(errors) == 1
    assert "13" in errors[0]
    assert "synthetic" in errors[0]


# ---------------------------------------------------------------------------
# Pad-byte zero invariant
# ---------------------------------------------------------------------------


def _write_one_record_with_pad(data_path: Path, index_path: Path) -> tuple[int, int]:
    """Lay down one synthetic v1 record that requires a non-zero pad.

    The smallest record-total residue mod 4 the writer ever pads to 0 is
    when ``HEADER_BYTES + insn_len + block_len + 2*token_count`` is
    already a multiple of 4; picking shapes whose sum is ``4k+1`` forces
    ``pad_size == 3``. Used by both the pad-tamper test and the
    sentinel-coupling test to avoid depending on an organic corpus that
    happens to have pad-bearing records.

    Returns ``(record_start, pad_byte_offset)``.
    """
    from tokenizer.aligned_data._writers import write_function_binary_data

    # 6 (header) + 1 (insn) + 0 (block) + 0 (tokens) = 7 → pad_size = 1.
    insn = np.arange(1, dtype=np.uint8)
    block = np.zeros(0, dtype=np.uint8)
    tokens = np.zeros(0, dtype=np.uint16)
    with open(data_path, "wb") as fh:
        result = write_function_binary_data(fh, tokens, block, insn)
    assert result is not None
    start, length = result
    assert length % 4 == 0

    # Stamp a matching index entry so ``check_pad_bytes_zero`` has
    # ``starts`` + ``lengths`` to iterate.
    prelude = struct.pack("<4sIII", INDEX_MAGIC, MEMMAP_FORMAT_VERSION, 2, 0)
    entry = (
        struct.pack("<Q", start >> 2)[:5]
        + struct.pack("<H", length >> 2)
        + struct.pack("B", 0)
    )
    index_path.write_bytes(prelude + entry)

    # The single pad byte sits immediately after insn_bytes; for our
    # 1-byte insn that is offset 6 + 1 = 7 inside the record.
    return start, start + HEADER_BYTES + insn.nbytes


def test_pad_bytes_zero_detects_tamper(tmp_path: Path) -> None:
    """Mutate ONE pad byte; ``check_pad_bytes_zero`` must flag it.

    Driven on a self-contained one-record corpus so the test does not
    depend on the synthetic-binary builder happening to produce a
    pad-bearing record.
    """
    data_path = tmp_path / "tiny_data.bin"
    index_path = tmp_path / "tiny_index.bin"
    record_start, pad_byte_offset = _write_one_record_with_pad(data_path, index_path)

    raw = bytearray(data_path.read_bytes())
    raw[pad_byte_offset] = 0xFF
    data_path.write_bytes(bytes(raw))

    starts, lengths, _ = read_index_arrays(index_path)
    errors = check_pad_bytes_zero(data_path, starts, lengths, str(data_path))
    assert any(
        f"start={record_start}" in e and "non-zero pad" in e for e in errors
    ), f"tampered pad should surface, got: {errors!r}"


# ---------------------------------------------------------------------------
# Sentinel / overlong-field coupling
# ---------------------------------------------------------------------------


def test_sentinel_overlong_mismatch_detected(tmp_path: Path) -> None:
    """Hand-craft a sentinel index entry whose data record is normal.

    The insn prefix is chosen so the bytes immediately after the header
    (record-offset 6..8) decode as a u24 LE value of 0 -- the resolver
    then returns ``(0, True)``, which is in the overlong band but well
    under the normal u16 cap, firing the coupling check.
    """
    from tokenizer.aligned_data._writers import write_function_binary_data

    data_path = tmp_path / "demo_data.bin"
    insn = np.array([0x00, 0x00, 0x00, 0xAB, 0xCD, 0xEF, 0x01, 0x02], dtype=np.uint8)
    block = np.zeros(0, dtype=np.uint8)
    tokens = np.zeros(0, dtype=np.uint16)
    with open(data_path, "wb") as fh:
        result = write_function_binary_data(fh, tokens, block, insn)
    assert result is not None
    start, _length = result

    # Force the sentinel marker on the only index entry.
    index_path = tmp_path / "demo_index.bin"
    prelude = struct.pack("<4sIII", INDEX_MAGIC, MEMMAP_FORMAT_VERSION, 2, 0)
    entry = (
        struct.pack("<Q", start >> 2)[:5]
        + struct.pack("<H", SENTINEL_LENGTH)
        + struct.pack("B", 0)
    )
    index_path.write_bytes(prelude + entry)

    starts, lengths, _ = read_index_arrays(index_path)
    assert lengths[0] == 0
    errors = check_sentinel_overlong_coupling(
        data_path, starts, lengths, str(index_path)
    )
    assert any(
        "not resolve as overlong" in e or "fits the normal u16 cap" in e for e in errors
    ), f"unexpected error wording: {errors!r}"


# ---------------------------------------------------------------------------
# Smoke: prelude + post-checks helpers compose on a clean corpus
# ---------------------------------------------------------------------------


def test_helpers_clean_on_fresh_corpus(tmp_path: Path) -> None:
    """Per-helper prelude probes + ``run_v1_post_checks`` clean on a v1 build.

    Composes the prelude checks the same way the validator does --
    individual ``check_csv_prelude`` + ``check_index_prelude`` calls,
    skipping the matched ``_index.bin`` because that file is pre-v1
    layout (no IDX1 magic; structurally validated at ``BinaryDataset``
    construction via ``read_csv_section_index_arrays``).
    """
    output_dir, _ = _pipeline(tmp_path)

    prelude_errors: list[str] = []
    for path in [
        output_dir / "demo_sections.csv",
        output_dir / "demo_unmatched_sections.csv",
        output_dir / "demo_variants.csv",
    ]:
        prelude_errors.extend(check_csv_prelude(path, str(path)))
    unmatched_idx = output_dir / "demo_unmatched_index.bin"
    prelude_errors.extend(check_index_prelude(unmatched_idx, str(unmatched_idx)))
    assert prelude_errors == [], f"prelude checks dirty on fresh build: {prelude_errors!r}"

    def _arrays_or_empty(path: Path):
        arr = read_index_arrays(path)
        if arr is None:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.uint32),
            )
        return arr[0], arr[1]

    # Matched arm: source per-record starts/lengths from the loaded
    # ``BinaryDataset`` (decoded inline_indexer hex per variant), NOT
    # from the pre-v1 ``demo_index.bin`` (which now holds CSV byte
    # positions, not data-bin positions). Unmatched arm still reads
    # the v1 index file.
    from tokenizer.aligned_data.loader import BinaryDataset
    from tokenizer.aligned_data.loader.unified_vocab_gate import (
        load_and_validate_unified_vocab,
    )

    vocab = load_and_validate_unified_vocab(output_dir / "unified_vocab.csv")
    dataset = BinaryDataset(output_dir, "demo", vocab_manager=vocab)

    unmatched_starts, unmatched_lengths = _arrays_or_empty(
        output_dir / "demo_unmatched_index.bin"
    )
    post_errors = run_v1_post_checks(
        matched_index=output_dir / "demo_index.bin",
        unmatched_index=output_dir / "demo_unmatched_index.bin",
        matched_data=output_dir / "demo_data.bin",
        unmatched_data=output_dir / "demo_unmatched_data.bin",
        matched_starts=dataset.matched_starts,
        matched_lengths=dataset.matched_lengths,
        unmatched_starts=unmatched_starts,
        unmatched_lengths=unmatched_lengths,
    )
    assert post_errors == [], f"post checks dirty on fresh build: {post_errors!r}"
