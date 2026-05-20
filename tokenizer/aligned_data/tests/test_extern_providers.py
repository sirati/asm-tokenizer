"""Round-trip + invariants for the extern-provider sidecar codec."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer.aligned_data.extern_providers import (
    ExternProviderRegistry,
    iter_extern_providers,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION


def test_round_trip_preserves_encounter_order(tmp_path: Path) -> None:
    reg = ExternProviderRegistry()
    # Intentionally NOT alphabetical: encounter order is the contract.
    libs = ["libc.so.6", "libstdc++.so.6", "ld-linux-x86-64.so.2", "libm.so.6"]
    for lib in libs:
        reg.add(lib)

    path = reg.write_sidecar(tmp_path, "mybin")

    assert path == tmp_path / "mybin_extern_providers.txt"
    read_back = list(iter_extern_providers(path))
    assert read_back == [(1, libs[0]), (2, libs[1]), (3, libs[2]), (4, libs[3])]


def test_add_is_idempotent_on_duplicate_library() -> None:
    reg = ExternProviderRegistry()
    first = reg.add("libc.so.6")
    again = reg.add("libc.so.6")
    assert first == again


def test_add_does_not_emit_duplicate_rows(tmp_path: Path) -> None:
    reg = ExternProviderRegistry()
    reg.add("libc.so.6")
    reg.add("libm.so.6")
    reg.add("libc.so.6")  # duplicate; must not produce a new row
    path = reg.write_sidecar(tmp_path, "mybin")
    rows = list(iter_extern_providers(path))
    assert rows == [(1, "libc.so.6"), (2, "libm.so.6")]


def test_first_added_library_returns_line_one() -> None:
    # Line 0 is reserved for the "library unknown" sentinel; the first
    # real library must land on line 1.
    reg = ExternProviderRegistry()
    assert reg.add("libc.so.6") == 1


def test_prelude_mismatch_raises_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad_extern_providers.txt"
    bad.write_text("# format=999999\nlibc.so.6\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_extern_providers(bad))


def test_prelude_missing_raises_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "no_prelude.txt"
    # No prelude at all: first line is a library name.
    bad.write_text("libc.so.6\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_extern_providers(bad))


def test_written_prelude_matches_current_format_version(tmp_path: Path) -> None:
    reg = ExternProviderRegistry()
    reg.add("libc.so.6")
    path = reg.write_sidecar(tmp_path, "mybin")
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert first == f"# format={MEMMAP_FORMAT_VERSION}"


def test_write_sidecar_is_re_callable(tmp_path: Path) -> None:
    # Restamping after additional adds rewrites the same file with the
    # newly grown registry. Earlier rows keep their line numbers.
    reg = ExternProviderRegistry()
    reg.add("libc.so.6")
    reg.write_sidecar(tmp_path, "mybin")
    line_two = reg.add("libm.so.6")
    assert line_two == 2
    path = reg.write_sidecar(tmp_path, "mybin")
    rows = list(iter_extern_providers(path))
    assert rows == [(1, "libc.so.6"), (2, "libm.so.6")]
