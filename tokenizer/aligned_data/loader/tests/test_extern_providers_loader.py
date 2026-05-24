"""Unit tests for :func:`load_extern_providers`.

Covers the materializer's contract: round-trip through the registry's
sidecar, prelude validation (delegated to ``iter_extern_providers``),
missing-file rejection, and the empty-registry "prelude-only" edge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer.aligned_data.extern_providers import ExternProviderRegistry
from tokenizer.aligned_data.loader.extern_providers_loader import (
    load_extern_providers,
)


def test_load_returns_1indexed_mapping_in_encounter_order(tmp_path: Path) -> None:
    reg = ExternProviderRegistry()
    libs = ["libc.so.6", "libstdc++.so.6", "ld-linux-x86-64.so.2"]
    for lib in libs:
        reg.add(lib)
    path = reg.write_sidecar(tmp_path, "mybin")

    mapping = load_extern_providers(path)

    assert mapping == {1: libs[0], 2: libs[1], 3: libs[2]}
    # Line 0 is the reserved "library unknown" sentinel and must not be
    # a key in the returned mapping.
    assert 0 not in mapping


def test_load_empty_registry_returns_empty_mapping(tmp_path: Path) -> None:
    reg = ExternProviderRegistry()
    path = reg.write_sidecar(tmp_path, "mybin")
    assert load_extern_providers(path) == {}


def test_load_missing_file_raises_value_error(tmp_path: Path) -> None:
    missing = tmp_path / "absent_extern_providers.txt"
    with pytest.raises(ValueError, match="extern-providers sidecar missing"):
        load_extern_providers(missing)


def test_load_bad_prelude_raises_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad_extern_providers.txt"
    bad.write_text("# format=999999\nlibc.so.6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prelude"):
        load_extern_providers(bad)


def test_load_missing_prelude_raises_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "no_prelude_extern_providers.txt"
    bad.write_text("libc.so.6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prelude"):
        load_extern_providers(bad)


def test_load_idempotent_on_duplicate_calls(tmp_path: Path) -> None:
    reg = ExternProviderRegistry()
    reg.add("libc.so.6")
    reg.add("libm.so.6")
    path = reg.write_sidecar(tmp_path, "mybin")

    first = load_extern_providers(path)
    second = load_extern_providers(path)
    assert first == second == {1: "libc.so.6", 2: "libm.so.6"}
