"""End-to-end tests for the ``_variants.bin`` cross-check.

Strategy: stand up a tiny synthetic corpus the way
``tokenizer.memmap_builder.tests.test_builder_smoke`` does — two
per-binary v2 CSVs through ``save_vocabulary`` + a real v3 unified
vocab from ``unify_vocab`` — then run ``validate_memmap_output``
against the freshly-built memmap output. The bodies of the per-binary
CSVs are empty (one padding line so ``read_last_line_of_file`` finds a
newline outside its 64-byte tail); the data-token validation loop
therefore iterates zero functions and the test isolates the variants-
bin cross-check from the unrelated data-token paths.

Failure-path tests deliberately corrupt the ``_variants.bin`` (random
byte rewrite) or the slim ``_variants.csv`` (wrong filename) and
confirm the validator surfaces the discrepancy as a ``stats.errors``
entry. A v2 (or missing) unified vocab must short-circuit the
validator at load time with a ``ValueError`` — same hard-cutover
contract the dataloader enforces.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import pytest

from tokenizer.memmap_validation.validator import (
    ValidatorConfig,
    VersionInfo,
    validate_memmap_output,
)
from tokenizer.memmap_validation.variants_bin_check import (
    cross_check_variants_bin,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.vocab_unifier.loader import load_unified_vocab_manager
from tokenizer.vocab_unifier.saver import save_vocabulary

from ._pipeline import (
    build_pipeline,
    build_synthetic_corpus as _build_synthetic_corpus,
    validator_versions_for as _validator_versions_for,
)


def _pipeline(tmp_path: Path) -> tuple[List[Path], Path, Path]:
    """Run unify -> build_memmap on the synthetic corpus.

    Returns ``(csv_files, unified_vocab_path, output_dir)``.
    """
    return build_pipeline(tmp_path, copy_vocab_into_output=False)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_cross_check_passes_on_freshly_built_corpus(tmp_path: Path) -> None:
    """End-to-end: build memmap, then run validate_memmap_output. The
    cross-check decodes the bin via the same vocab the builder used and
    asserts every variant's axis strings round-trip."""
    csv_files, unified_vocab_path, output_dir = _pipeline(tmp_path)

    # Place the unified vocab under output_dir so the validator's
    # default vocab-path resolution finds it (mirrors production layout).
    (output_dir / "unified_vocab.csv").write_bytes(unified_vocab_path.read_bytes())

    config = ValidatorConfig(
        versions=_validator_versions_for(csv_files),
        output_dir=output_dir,
        binary_name="demo",
    )
    stats = validate_memmap_output(config)

    assert stats.errors == [], (
        f"variants-bin cross-check should be clean on a freshly-built "
        f"corpus, got: {stats.errors!r}"
    )


def test_cross_check_helper_pure_function_no_errors(tmp_path: Path) -> None:
    """The cross-check helper itself, called with the same artefacts
    the validator hands it, returns no errors."""
    csv_files, unified_vocab_path, output_dir = _pipeline(tmp_path)
    vocab_manager = load_unified_vocab_manager(unified_vocab_path)
    assert vocab_manager is not None

    errors = cross_check_variants_bin(
        _validator_versions_for(csv_files),
        output_dir,
        "demo",
        vocab_manager,
    )
    assert errors == []


# ---------------------------------------------------------------------------
# Failure-path coverage
# ---------------------------------------------------------------------------


def test_cross_check_flags_corrupted_bin(tmp_path: Path) -> None:
    """Flip a byte inside the bin's first record (offset 2..end-of-
    record). The cross-check must surface the decoded-string mismatch as
    a non-empty error list."""
    csv_files, unified_vocab_path, output_dir = _pipeline(tmp_path)
    vocab_manager = load_unified_vocab_manager(unified_vocab_path)
    assert vocab_manager is not None

    bin_path = output_dir / "demo_variants.bin"
    raw = bytearray(bin_path.read_bytes())
    # Flip a token byte inside the first record's payload (offset 2 is
    # the start of token #0; XOR a low bit so the resulting ID still
    # falls inside the vocab range — guarantees the test exercises the
    # axis-string mismatch path rather than a separate vocab-miss
    # branch).
    assert len(raw) > 4
    raw[2] ^= 0x01
    bin_path.write_bytes(bytes(raw))

    errors = cross_check_variants_bin(
        _validator_versions_for(csv_files),
        output_dir,
        "demo",
        vocab_manager,
    )
    assert errors, "expected at least one cross-check error from corrupted bin"
    assert any("decoded to" in e for e in errors), (
        f"corruption should produce an 'decoded to ... expected' "
        f"mismatch error, got: {errors!r}"
    )


def test_cross_check_flags_missing_filename_in_slim_csv(tmp_path: Path) -> None:
    """Rewrite the slim CSV so its filenames no longer match the
    per-variant CSV identities. Cross-check must report
    'not present in the slim _variants.csv offset table'."""
    csv_files, unified_vocab_path, output_dir = _pipeline(tmp_path)
    vocab_manager = load_unified_vocab_manager(unified_vocab_path)
    assert vocab_manager is not None

    slim_csv_path = output_dir / "demo_variants.csv"
    with open(slim_csv_path, encoding="ascii") as f:
        rows = list(csv.reader(f))
    # rows[0] is the ``["# format=1"]`` prelude marker; rows[1] is the
    # ``filename,variant_id,offset`` header; data rows start at rows[2:].
    for row in rows[2:]:
        row[0] = "wrong-filename-" + row[0]
    with open(slim_csv_path, "w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(rows)

    errors = cross_check_variants_bin(
        _validator_versions_for(csv_files),
        output_dir,
        "demo",
        vocab_manager,
    )
    assert errors, "expected cross-check errors from missing filename"
    assert all("not present in the slim" in e for e in errors), (
        f"filename mismatch must surface the slim-CSV miss diagnostic, "
        f"got: {errors!r}"
    )


def test_cross_check_flags_missing_bin(tmp_path: Path) -> None:
    """A missing ``_variants.bin`` is a builder-level failure; the
    cross-check must surface it explicitly rather than silently skip."""
    csv_files, unified_vocab_path, output_dir = _pipeline(tmp_path)
    vocab_manager = load_unified_vocab_manager(unified_vocab_path)
    assert vocab_manager is not None

    (output_dir / "demo_variants.bin").unlink()

    errors = cross_check_variants_bin(
        _validator_versions_for(csv_files),
        output_dir,
        "demo",
        vocab_manager,
    )
    assert any("_variants.bin missing" in e for e in errors), (
        f"missing bin must produce the explicit missing-bin diagnostic, "
        f"got: {errors!r}"
    )


def test_cross_check_flags_missing_slim_csv(tmp_path: Path) -> None:
    """A missing slim CSV is equally surface-able — the cross-check has
    no offset table to look up against."""
    csv_files, unified_vocab_path, output_dir = _pipeline(tmp_path)
    vocab_manager = load_unified_vocab_manager(unified_vocab_path)
    assert vocab_manager is not None

    (output_dir / "demo_variants.csv").unlink()

    errors = cross_check_variants_bin(
        _validator_versions_for(csv_files),
        output_dir,
        "demo",
        vocab_manager,
    )
    assert any("_variants.csv missing" in e for e in errors), (
        f"missing slim CSV must produce the explicit missing-CSV "
        f"diagnostic, got: {errors!r}"
    )


# ---------------------------------------------------------------------------
# Hard-cutover: validator entry rejects non-v3 / missing unified vocab
# ---------------------------------------------------------------------------


def test_validate_memmap_output_rejects_v2_unified_vocab(tmp_path: Path) -> None:
    """A v2 unified vocab predates variant-axis tokens; the validator
    must short-circuit at vocab-load time with a ValueError, mirroring
    the AlignedDataLoader's hard-cutover gate."""
    csv_files, unified_vocab_path, output_dir = _pipeline(tmp_path)

    # Overwrite the in-output unified vocab with a v2 one. The builder
    # already ran (it consumed the v3 vocab via unified_vocab_path), so
    # the bin / slim CSV / sections / data files exist; the validator's
    # vocab-load is what we want to fail.
    vm_v2 = VocabularyManager(platform=None, format_version=2)
    with open(output_dir / "unified_vocab.csv", "w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh, lineterminator='\n')
        save_vocabulary(vm_v2, writer)

    config = ValidatorConfig(
        versions=_validator_versions_for(csv_files),
        output_dir=output_dir,
        binary_name="demo",
    )

    with pytest.raises(ValueError, match="format_version"):
        validate_memmap_output(config)


def test_validate_memmap_output_rejects_missing_unified_vocab(tmp_path: Path) -> None:
    """A missing unified vocab in the output directory is a hard error;
    the validator must not silently fall back to running without a vocab
    (the legacy behaviour had a downgrade path; v3 does not)."""
    csv_files, unified_vocab_path, output_dir = _pipeline(tmp_path)

    # Do NOT copy the unified vocab into output_dir; the validator's
    # default resolution should fail to find it.
    config = ValidatorConfig(
        versions=_validator_versions_for(csv_files),
        output_dir=output_dir,
        binary_name="demo",
    )
    # Explicitly point at a path that does not exist (so the default
    # resolution does not accidentally hit a v3 vocab a previous test
    # left behind via a shared tmp_path — pytest fixtures isolate
    # tmp_path per test, but pin the path anyway for diagnostic clarity).
    config.unified_vocab_path = tmp_path / "no_such_unified_vocab.csv"

    with pytest.raises(ValueError, match="unified_vocab.csv not found"):
        validate_memmap_output(config)
