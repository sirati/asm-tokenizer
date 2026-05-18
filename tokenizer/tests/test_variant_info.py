"""Unit tests for ``VariantInfo.from_csv``.

The classmethod is the single source of truth for filename + sidecar
identity recovery (used by the vocab-unifier discovery pass and the
memmap-builder worker). Tests cover:

* legacy-only fallback (no sidecar on disk)
* sidecar-only (filename doesn't match the canonical 4-axis shape)
* both agreeing (no warning)
* per-axis disagreement (sidecar wins, warning emitted)
* empty + list-valued ``extra_metadata`` round-trip
* pkg populated from either source
* ``__<8hex>`` variant-id suffix peeling
* declared-but-missing meta_path emits the warning
* both sources missing raises ``ValueError``
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import pytest

from tokenizer.variant_info import VariantInfo


# Canonical-format basename used across the happy-path tests. Maps to
# arch=x64, compiler=gcc, compiler_version=13.2.0, opt=O2, pkg=hello.
_CANONICAL_BASE = "x64-gcc-13.2.0-O2_hello"


def _write_csv(dir_: Path, base: str) -> Path:
    """Touch an empty ``<base>_output.csv`` so the path exists for
    ``from_csv`` (the body only inspects ``csv_path.name`` and its
    sibling sidecar — never the CSV contents)."""
    csv_path = dir_ / f"{base}_output.csv"
    csv_path.write_text("")
    return csv_path


def _write_sidecar(
    dir_: Path,
    base: str,
    *,
    arch: str = "x64",
    compiler: str = "gcc",
    compiler_version: str = "13.2.0",
    opt: str = "O2",
    pkg: str = "hello",
    variant_id: int = 0,
    extra_metadata: dict | None = None,
) -> Path:
    """Serialize a canonical ``_meta.json`` payload (the same shape
    ``tokenizer.run_tokenizer._write_meta_sidecar`` emits — verbatim
    ``dataclasses.asdict(VariantInfo)``)."""
    payload = {
        "arch": arch,
        "compiler": compiler,
        "compiler_version": compiler_version,
        "opt": opt,
        "pkg": pkg,
        "variant_id": variant_id,
        "extra_metadata": extra_metadata if extra_metadata is not None else {},
    }
    meta_path = dir_ / f"{base}_meta.json"
    meta_path.write_text(json.dumps(payload))
    return meta_path


# ---------------------------------------------------------------- legacy-only


def test_legacy_filename_only(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """All four axes parse from the canonical filename; no sidecar
    present on disk → silent fallback, no warnings."""
    csv = _write_csv(tmp_path, _CANONICAL_BASE)
    with caplog.at_level(logging.WARNING, logger="tokenizer.variant_info"):
        info = VariantInfo.from_csv(csv)

    assert info.arch == "x64"
    assert info.compiler == "gcc"
    assert info.compiler_version == "13.2.0"
    assert info.opt == "O2"
    assert info.pkg == "hello"
    assert info.variant_id == 0
    assert info.extra_metadata == {}
    assert info.filename == _CANONICAL_BASE
    assert caplog.records == []


def test_legacy_filename_with_variant_id_suffix(tmp_path: Path):
    """``__<8hex>`` suffix is peeled into ``variant_id`` and the
    cleaned ``pkg`` is preserved. Matches the existing
    ``_split_variant_suffix`` semantics."""
    base = "arm32-clang-10-O3_minigzip__15f3f338"
    csv = _write_csv(tmp_path, base)
    info = VariantInfo.from_csv(csv)

    assert info.arch == "arm32"
    assert info.compiler == "clang"
    assert info.compiler_version == "10"
    assert info.opt == "O3"
    assert info.pkg == "minigzip"
    assert info.variant_id == 0x15F3F338
    # ``filename`` is the canonical base verbatim — keeps the suffix
    # so the slim ``_variants.csv`` can round-trip it.
    assert info.filename == base


# ---------------------------------------------------------------- sidecar-only


def test_sidecar_only(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Filename does not match the legacy format → sidecar fills every
    canonical axis. ``filename`` still comes from the CSV stem."""
    base = "weird-bin"
    csv = _write_csv(tmp_path, base)
    _write_sidecar(
        tmp_path,
        base,
        arch="x64",
        compiler="gcc",
        compiler_version="13.2.0",
        opt="O2",
        pkg="hello",
        variant_id=42,
        extra_metadata={"flavor": "minimal"},
    )

    with caplog.at_level(logging.WARNING, logger="tokenizer.variant_info"):
        info = VariantInfo.from_csv(csv)

    assert info.arch == "x64"
    assert info.compiler == "gcc"
    assert info.compiler_version == "13.2.0"
    assert info.opt == "O2"
    assert info.pkg == "hello"
    assert info.variant_id == 42
    assert info.extra_metadata == {"flavor": "minimal"}
    assert info.filename == "weird-bin"
    assert caplog.records == []


# ---------------------------------------------------------------- agreement


def test_both_sources_agree_no_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Filename and sidecar produce identical canonical axes → no
    disagreement warnings; sidecar's extra_metadata still flows
    through."""
    csv = _write_csv(tmp_path, _CANONICAL_BASE)
    _write_sidecar(
        tmp_path,
        _CANONICAL_BASE,
        arch="x64",
        compiler="gcc",
        compiler_version="13.2.0",
        opt="O2",
        pkg="hello",
        variant_id=0,
        extra_metadata={"hardening": ["full", "fortify"]},
    )

    with caplog.at_level(logging.WARNING, logger="tokenizer.variant_info"):
        info = VariantInfo.from_csv(csv)

    assert info.arch == "x64"
    assert info.compiler == "gcc"
    assert info.compiler_version == "13.2.0"
    assert info.opt == "O2"
    assert info.pkg == "hello"
    assert info.variant_id == 0
    assert info.extra_metadata == {"hardening": ["full", "fortify"]}
    assert caplog.records == []


# ---------------------------------------------------------------- disagreement


def test_disagreement_sidecar_wins_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Per the documented precedence: sidecar overrides filename on
    canonical axes, one warning per disagreeing axis."""
    csv = _write_csv(tmp_path, _CANONICAL_BASE)
    _write_sidecar(
        tmp_path,
        _CANONICAL_BASE,
        # Filename says compiler_version=13.2.0; sidecar overrides
        # with the richer full string (mirrors the legacy-major vs
        # sidecar-full asymmetry called out in the module docstring).
        compiler_version="13.2.1",
        pkg="hello-renamed",
    )

    with caplog.at_level(logging.WARNING, logger="tokenizer.variant_info"):
        info = VariantInfo.from_csv(csv)

    assert info.compiler_version == "13.2.1"
    assert info.pkg == "hello-renamed"
    # Untouched axes still come from the filename / sidecar match.
    assert info.arch == "x64"
    assert info.opt == "O2"

    # Two disagreeing axes (compiler_version + pkg) → two warnings.
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2
    messages = " ".join(r.getMessage() for r in warnings)
    assert "compiler_version" in messages
    assert "pkg" in messages
    assert "sidecar wins" in messages


# ---------------------------------------------------------------- metadata


def test_empty_extra_metadata_roundtrips(tmp_path: Path):
    csv = _write_csv(tmp_path, _CANONICAL_BASE)
    _write_sidecar(tmp_path, _CANONICAL_BASE, extra_metadata={})
    info = VariantInfo.from_csv(csv)
    assert info.extra_metadata == {}


def test_list_valued_metadata_roundtrips(tmp_path: Path):
    csv = _write_csv(tmp_path, _CANONICAL_BASE)
    _write_sidecar(
        tmp_path,
        _CANONICAL_BASE,
        extra_metadata={"hardening": ["full", "fortify"]},
    )
    info = VariantInfo.from_csv(csv)
    assert info.extra_metadata == {"hardening": ["full", "fortify"]}
    # JSON round-trip preserves list identity (not e.g. tuple).
    assert isinstance(info.extra_metadata["hardening"], list)


def test_sidecar_missing_extra_metadata_defaults_to_empty(tmp_path: Path):
    """Forward-compat: a sidecar that omits ``extra_metadata`` (e.g.
    from an older producer) decodes as ``{}``, not ``None``."""
    base = _CANONICAL_BASE
    csv = _write_csv(tmp_path, base)
    # Hand-write a sidecar lacking the ``extra_metadata`` key.
    (tmp_path / f"{base}_meta.json").write_text(
        json.dumps(
            {
                "arch": "x64",
                "compiler": "gcc",
                "compiler_version": "13.2.0",
                "opt": "O2",
                "pkg": "hello",
                "variant_id": 0,
            }
        )
    )
    info = VariantInfo.from_csv(csv)
    assert info.extra_metadata == {}


# ---------------------------------------------------------------- pkg


def test_pkg_from_filename(tmp_path: Path):
    """No sidecar: ``pkg`` comes from the filename's binary-name slot."""
    csv = _write_csv(tmp_path, "x64-gcc-13.2.0-O2_libfoo")
    info = VariantInfo.from_csv(csv)
    assert info.pkg == "libfoo"


def test_pkg_from_sidecar_overrides(tmp_path: Path):
    """Sidecar pkg differs from filename → sidecar wins (warns)."""
    csv = _write_csv(tmp_path, "x64-gcc-13.2.0-O2_libfoo")
    _write_sidecar(tmp_path, "x64-gcc-13.2.0-O2_libfoo", pkg="libfoo-renamed")
    info = VariantInfo.from_csv(csv)
    assert info.pkg == "libfoo-renamed"


# ---------------------------------------------------------------- edge cases


def test_explicit_meta_path_missing_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Mirrors the existing ``dynrunner/build_memmap/worker.py``
    behavior: caller named a sidecar path that does not exist → warn,
    fall back to filename. Implicit default-path absence is silent
    (covered by ``test_legacy_filename_only``)."""
    csv = _write_csv(tmp_path, _CANONICAL_BASE)
    missing_meta = tmp_path / "definitely_not_here.json"

    with caplog.at_level(logging.WARNING, logger="tokenizer.variant_info"):
        info = VariantInfo.from_csv(csv, meta_path=missing_meta)

    assert info.compiler_version == "13.2.0"  # filename fallback worked
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "declared but missing" in warnings[0].getMessage()


def test_neither_source_usable_raises(tmp_path: Path):
    """Non-legacy filename AND no sidecar on disk → ValueError. The
    failure surfaces at the boundary, not as a corrupt VariantInfo."""
    csv = _write_csv(tmp_path, "totally-not-a-canonical-name")
    with pytest.raises(ValueError, match="cannot derive variant identity"):
        VariantInfo.from_csv(csv)


def test_writer_roundtrip(tmp_path: Path):
    """Construct a VariantInfo, write the same JSON the tokenize
    worker produces (``dataclasses.asdict``), read it back via
    ``from_csv`` — every canonical axis + extra_metadata + filename
    survives unchanged."""
    original = VariantInfo(
        arch="armv7l-hf",
        compiler="clang",
        compiler_version="14.0.6",
        opt="Oz",
        pkg="busybox",
        variant_id=0xDEADBEEF,
        extra_metadata={"hardening": ["full"], "sanitizer": "address"},
    )
    base = "armv7l-hf-clang-14.0.6-Oz_busybox__deadbeef"
    csv = _write_csv(tmp_path, base)
    (tmp_path / f"{base}_meta.json").write_text(
        json.dumps(dataclasses.asdict(original))
    )

    decoded = VariantInfo.from_csv(csv)

    assert decoded.arch == original.arch
    assert decoded.compiler == original.compiler
    assert decoded.compiler_version == original.compiler_version
    assert decoded.opt == original.opt
    assert decoded.pkg == original.pkg
    assert decoded.variant_id == original.variant_id
    assert decoded.extra_metadata == original.extra_metadata
    assert decoded.filename == base
