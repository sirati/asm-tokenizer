"""``tokenizer.binary_discovery.walk_dataset`` sidecar-folder enumeration.

A sidecar variant folder may hold SEVERAL real binaries (a CLI tool and
its siblings, or a library's lone ``.so``); one ``*.json`` sidecar's
metadata applies to all of them. Discovery must therefore yield one
``(handle, variant)`` per ELF binary — shared ``variant``, distinct
``handle.path`` / ``handle.binary_name`` — selecting members by ELF
magic (not extension) and excluding split debug-symbol ``*.debug``
objects.

Pins:
* multi-binary folder (pkg-named exec + sibling exec + dotted ``.so``)
  yields all three; the ``.debug`` and the non-ELF README are excluded
* library-only folder yields its single ``.so``
* a folder with no qualifying binary yields nothing (graceful skip)
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizer.binary_discovery import walk_dataset

_ELF = b"\x7fELF" + b"\x00" * 12


def _sidecar_folder(
    parent: Path,
    stem: str,
    *,
    pkg: str,
    arch: str = "x86_64",
    compiler_family: str = "gcc",
    compiler_version: str = "13.2.0",
    optimization: str = "O2",
    variant_hex: str = "15f3f338",
) -> Path:
    """Create a ``<stem>_<8hex>.json`` sidecar paired with a same-stem
    folder, and return the (empty) folder for the caller to populate."""
    sidecar_name = f"{stem}_{variant_hex}.json"
    (parent / sidecar_name).write_text(
        json.dumps(
            {
                "arch": arch,
                "compiler_family": compiler_family,
                "compiler_version": compiler_version,
                "optimization": optimization,
                "pkg": pkg,
            }
        )
    )
    folder = parent / f"{stem}_{variant_hex}"
    folder.mkdir()
    return folder


def _names(pairs) -> list[str]:
    return sorted(h.binary_name for h, _ in pairs)


def test_multi_binary_folder_yields_all_elf_excludes_debug_and_nonelf(
    tmp_path: Path,
) -> None:
    folder = _sidecar_folder(tmp_path, "flac", pkg="flac")
    # pkg-named exec, a differently-named sibling, a dotted shared object.
    (folder / "flac").write_bytes(_ELF)
    (folder / "metaflac").write_bytes(_ELF)
    (folder / "libFLAC.so.12").write_bytes(_ELF)
    # Split debug-symbol object: ELF magic but name ends in .debug.
    (folder / "flac.debug").write_bytes(_ELF)
    # Non-ELF file: README.
    (folder / "README").write_text("not a binary\n")

    pairs = list(walk_dataset(tmp_path))
    assert _names(pairs) == ["flac", "libFLAC.so.12", "metaflac"]

    # All three share the SAME variant (one sidecar's metadata).
    variants = {id(v) for _, v in pairs}
    assert len(variants) == 1
    one_variant = pairs[0][1]
    assert one_variant.pkg == "flac"
    assert one_variant.arch == "x86_64"
    # Each handle points at its own on-disk file inside the folder.
    for handle, _ in pairs:
        assert handle.path.name == handle.binary_name
        assert handle.path.parent == folder
        assert handle.variant_dir == folder


def test_library_only_folder_yields_single_so(tmp_path: Path) -> None:
    folder = _sidecar_folder(tmp_path, "zlib", pkg="zlib")
    (folder / "libz.so.1.3.2").write_bytes(_ELF)

    pairs = list(walk_dataset(tmp_path))
    assert _names(pairs) == ["libz.so.1.3.2"]
    handle, variant = pairs[0]
    assert handle.binary_name == "libz.so.1.3.2"
    assert variant.pkg == "zlib"


def test_renamed_tool_folder_yields_actual_name(tmp_path: Path) -> None:
    # sqlite package ships the `sqlite3` tool (name != pkg).
    folder = _sidecar_folder(tmp_path, "sqlite", pkg="sqlite")
    (folder / "sqlite3").write_bytes(_ELF)
    # Split debug object alongside it must be dropped.
    (folder / "sqlite3.debug").write_bytes(_ELF)

    pairs = list(walk_dataset(tmp_path))
    assert _names(pairs) == ["sqlite3"]
    assert pairs[0][0].binary_name == "sqlite3"


def test_empty_or_debug_only_folder_yields_nothing(tmp_path: Path) -> None:
    # Folder with only a .debug object and a non-ELF file → no binaries.
    folder = _sidecar_folder(tmp_path, "ghost", pkg="ghost")
    (folder / "ghost.debug").write_bytes(_ELF)
    (folder / "manifest.txt").write_text("x")

    assert list(walk_dataset(tmp_path)) == []
