"""Unit tests for filesystem discovery.

Covers phase-1 binary discovery across the flat and nested-per-binary
layouts, the build_memmap subtree exclusion, sidecar attachment, and
phase-3 program-name + kind derivation (including dotted program names
like ``libz.so.1.2.11`` whose tail must still normalise correctly).
"""

from __future__ import annotations

from pathlib import Path

from scripts.collect_stats.discovery import discover_binaries, discover_phase3


def _touch(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_binary(directory: Path, fullname: str) -> None:
    _touch(directory / f"{fullname}_output.csv", b"vocab\n")
    _touch(directory / f"{fullname}_meta.json", b"{}")
    _touch(directory / f"{fullname}_strings.bin", b"S" * 10)
    _touch(directory / f"{fullname}_function_ranges.txt", b"0 1\n")


def test_flat_layout_discovery(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _make_binary(out / "zlib", "x64-clang-3.5-O0_minigzipsh")
    found = discover_binaries(out)
    assert len(found) == 1
    b = found[0]
    assert b.fullname == "x64-clang-3.5-O0_minigzipsh"
    assert b.package == "zlib"
    kinds = {f.kind for f in b.files}
    assert kinds == {"output_csv", "meta_json", "strings_bin", "function_ranges"}


def test_nested_layout_discovery_package_is_first_component(tmp_path: Path) -> None:
    out = tmp_path / "out"
    nested = out / "dataset" / "hello" / "clang10_armv7l-hf_Oz_15f3f338"
    _make_binary(nested, "armv7l-hf-clang-10.0.1-Oz_hello__15f3f338")
    found = discover_binaries(out)
    assert len(found) == 1
    assert found[0].package == "dataset"
    assert found[0].fullname == "armv7l-hf-clang-10.0.1-Oz_hello__15f3f338"


def test_build_memmap_subtree_excluded_from_binaries(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _make_binary(out / "zlib", "x64-clang-3.5-O0_minigzip")
    # A stray _output.csv inside build_memmap must NOT be counted as a
    # phase-1 binary (the subtree is a phase-3 concern).
    _touch(out / "build_memmap" / "minigzip" / "x_output.csv", b"y")
    found = discover_binaries(out)
    assert len(found) == 1
    assert found[0].fullname == "x64-clang-3.5-O0_minigzip"


def test_noise_dir_without_output_csv_is_skipped(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _make_binary(out / "zlib", "x64-clang-3.5-O0_minigzip")
    _touch(out / "sec-0" / "worker_0.log", b"log\n")
    found = discover_binaries(out)
    assert {b.fullname for b in found} == {"x64-clang-3.5-O0_minigzip"}


def test_phase3_program_and_kind_derivation(tmp_path: Path) -> None:
    out = tmp_path / "out"
    bm = out / "build_memmap"
    prog = bm / "minigzipsh"
    _touch(prog / "minigzipsh_data.bin", b"D" * 5)
    _touch(prog / "minigzipsh_index.bin", b"I" * 3)
    _touch(prog / "minigzipsh_unmatched_sections.csv", b"u\n")
    _touch(prog / "minigzipsh.warn.log", b"w\n")
    _touch(prog / "minigzipsh.error.log", b"")

    programs = discover_phase3(out)
    assert len(programs) == 1
    p = programs[0]
    assert p.program == "minigzipsh"
    by_kind = {f.kind: f.size_bytes for f in p.files}
    assert by_kind["data_bin"] == 5
    assert by_kind["index_bin"] == 3
    assert by_kind["unmatched_sections_csv"] == 2  # len(b"u\n")
    assert by_kind["warn_log"] == 2
    assert by_kind["error_log"] == 0  # genuine empty file ⇒ 0, recorded


def test_phase3_dotted_program_name(tmp_path: Path) -> None:
    """A program name with dots (``libz.so.1.2.11``) must strip cleanly:
    the kind is the suffix after the exact dir-name prefix, not a naive
    split on the first dot/underscore."""
    out = tmp_path / "out"
    prog = out / "build_memmap" / "libz.so.1.2.11"
    _touch(prog / "libz.so.1.2.11_data.bin", b"D" * 9)
    _touch(prog / "libz.so.1.2.11_variants.bin", b"V" * 2)

    programs = discover_phase3(out)
    assert len(programs) == 1
    assert programs[0].program == "libz.so.1.2.11"
    by_kind = {f.kind: f.size_bytes for f in programs[0].files}
    assert by_kind == {"data_bin": 9, "variants_bin": 2}


def test_phase3_absent_build_memmap(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _make_binary(out / "zlib", "x64-clang-3.5-O0_minigzip")
    assert discover_phase3(out) == []
