"""Tests for ``vocab_unifier.variant_registration``.

Synthetic discovery: build a directory of fake per-binary CSV files
(filename only — ``VariantInfo.from_csv`` parses the legacy
filename-with-optional-sidecar shape) and confirm
``discover_and_register_variants`` registers each distinct
prefixed token string into the unified VM at IDs starting at 256.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenizer.token_manager import VocabularyManager
from tokenizer.vocab_unifier.variant_registration import (
    discover_and_register_variants,
)


def _write_legacy_csv(dir_path: Path, basename: str) -> Path:
    """Create an empty ``<basename>_output.csv`` file.

    The discovery pass only reads the filename, not the file body —
    so a touch suffices. ``basename`` must match the legacy 4-axis
    format ``<platform>-<compiler>-<version>-<opt>_<binary_name>``
    (the default ``platform-compiler-version-optimisationlevel_binaryname``
    format string used by ``build_binary_filename_format``).
    """
    path = dir_path / f"{basename}_output.csv"
    path.write_text("")  # body is irrelevant for from_csv
    return path


def _write_sidecar(
    csv_path: Path,
    *,
    arch: str,
    compiler_family: str,
    compiler_version: str,
    optimization: str,
    pkg: str,
    extra_metadata: dict | None = None,
) -> None:
    """Write the canonical ``<base>_meta.json`` sibling sidecar."""
    base = csv_path.name.removesuffix("_output.csv")
    sidecar = csv_path.with_name(base + "_meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "arch": arch,
                "compiler_family": compiler_family,
                "compiler_version": compiler_version,
                "optimization": optimization,
                "pkg": pkg,
                "extra_metadata": extra_metadata or {},
            }
        )
    )


def test_register_simple_corpus(tmp_path: Path) -> None:
    """Three legacy-format CSVs with disjoint arch/compiler combos.

    Default field regexes reject ``_`` inside platform/compiler/version
    so the production pipeline already uses the alias-collapsed form
    (``x64``, ``arm64``) in filenames. The sidecar path is the only
    way to deliver the underscored ``x86_64`` / ``aarch64`` and is
    exercised separately in ``test_register_with_sidecar_extra_metadata``.
    """
    csvs = [
        _write_legacy_csv(tmp_path, "x64-gcc-13.2.0-O2_hello"),
        _write_legacy_csv(tmp_path, "armv7l-clang-15.0.0-O3_hello"),
        _write_legacy_csv(tmp_path, "arm64-gcc-13.2.0-O2_hello"),
    ]
    vm = VocabularyManager(platform=None, format_version=3)
    n = discover_and_register_variants(csvs, vm)

    # 3 archs (already alias-collapsed in filename; identity through
    # ``arch_to_variant_arch`` for x64/arm64/armv7l)
    # 2 compilers (gcc, clang)
    # 2 cver strings (gcc:13.2.0 dedups across two binaries;
    #                 clang:15.0.0 once)
    # 2 opt levels (O2, O3)
    expected_arch = {"arch:x64", "arch:arm64", "arch:armv7l"}
    expected_comp = {"comp:gcc", "comp:clang"}
    expected_cver = {"cver:gcc:13.2.0", "cver:clang:15.0.0"}
    expected_opt = {"opt:O2", "opt:O3"}
    expected = expected_arch | expected_comp | expected_cver | expected_opt
    assert n == len(expected), f"expected {len(expected)} distinct, got {n}"

    # Every expected token is registered and resolves; v3 reserved-digit
    # boundary places variant ids at >= 256.
    for token in expected:
        tid = vm.get_token_id(token)
        assert tid >= 256, f"variant token {token!r} got id {tid}"

    # Variant tokens occupy a contiguous low block [256, 256+n).
    variant_id_range = set(range(256, 256 + n))
    actual_ids = {vm.get_token_id(t) for t in expected}
    assert actual_ids == variant_id_range, (
        f"variant ids {sorted(actual_ids)} do not match expected "
        f"contiguous range {sorted(variant_id_range)}"
    )


def test_register_with_sidecar_extra_metadata(tmp_path: Path) -> None:
    """Sidecar wins per axis + metadata pairs become individual tokens."""
    csv = _write_legacy_csv(tmp_path, "x64-gcc-13-O2_zlib")
    _write_sidecar(
        csv,
        arch="amd64",  # sidecar collapses via arch_to_variant_arch -> x64
        compiler_family="gcc",
        compiler_version="13.2.0",  # sidecar's richer version wins
        optimization="-O2",
        pkg="zlib",
        extra_metadata={
            "hardening": ["full", "fortify"],
            "sanitizer": "address",
        },
    )
    vm = VocabularyManager(platform=None, format_version=3)
    n = discover_and_register_variants([csv], vm)

    # 4 positional + 3 metadata (hardening:full, hardening:fortify,
    # sanitizer:address) = 7 tokens
    assert n == 7
    for token in [
        "arch:x64",
        "comp:gcc",
        "cver:gcc:13.2.0",
        "opt:O2",
        "hardening:full",
        "hardening:fortify",
        "sanitizer:address",
    ]:
        assert vm.get_token_id(token) != -1, f"missing token {token!r}"


def test_register_deduplicates_across_corpus(tmp_path: Path) -> None:
    """Same arch/compiler/version/opt across many CSVs registers once."""
    csvs = [
        _write_legacy_csv(tmp_path, f"x64-gcc-13.2.0-O2_pkg{i}")
        for i in range(5)
    ]
    vm = VocabularyManager(platform=None, format_version=3)
    n = discover_and_register_variants(csvs, vm)
    # 4 positional axes, all identical across the 5 CSVs
    assert n == 4


def test_register_order_is_deterministic(tmp_path: Path) -> None:
    """Two runs over the same corpus produce identical id assignments."""
    csvs = [
        _write_legacy_csv(tmp_path, "armv7l-clang-15.0.0-O3_foo"),
        _write_legacy_csv(tmp_path, "x64-gcc-13.2.0-O2_bar"),
    ]
    vm_a = VocabularyManager(platform=None, format_version=3)
    vm_b = VocabularyManager(platform=None, format_version=3)
    n_a = discover_and_register_variants(csvs, vm_a)
    n_b = discover_and_register_variants(list(reversed(csvs)), vm_b)
    assert n_a == n_b
    # Same string -> same id even with reversed input order, because the
    # inventory sorts before registering.
    for tok in [
        "arch:x64", "arch:armv7l", "comp:gcc", "comp:clang",
        "cver:gcc:13.2.0", "cver:clang:15.0.0", "opt:O2", "opt:O3",
    ]:
        assert vm_a.get_token_id(tok) == vm_b.get_token_id(tok), (
            f"non-deterministic id for {tok!r}: a={vm_a.get_token_id(tok)} "
            f"b={vm_b.get_token_id(tok)}"
        )


def test_register_rejects_non_v3_vm(tmp_path: Path) -> None:
    """Calling against a v1 or v2 VM is a programmer error."""
    csv = _write_legacy_csv(tmp_path, "x64-gcc-13.2.0-O2_hello")
    vm_v2 = VocabularyManager(platform=None, format_version=2)
    with pytest.raises(AssertionError, match="v3 unified VM"):
        discover_and_register_variants([csv], vm_v2)


def test_register_warns_and_skips_unparseable_csv(tmp_path: Path) -> None:
    """A CSV whose name doesn't match the legacy format and has no
    sidecar is logged + skipped rather than aborting the whole pass."""
    # No legacy-format match, no sidecar -> from_csv raises
    bogus = tmp_path / "garbage_name.csv"
    bogus.write_text("")
    good = _write_legacy_csv(tmp_path, "x64-gcc-13.2.0-O2_hello")

    vm = VocabularyManager(platform=None, format_version=3)
    n = discover_and_register_variants([bogus, good], vm)
    # Good CSV's 4 positional axes still registered
    assert n == 4
    assert vm.get_token_id("arch:x64") != -1
