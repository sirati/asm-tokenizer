"""Walk the ``out/`` tree and yield discovered binaries and programs.

Single concern: filesystem discovery.  No axis parsing, no vocabulary
counting, no DB.  Callers receive plain typed records describing what
files exist and how big they are.

Layout handled (uniformly, at any nesting depth):

* Phase-1 per-binary outputs live under ``out/<package>/.../`` and are
  **anchored by the canonical ``_output.csv`` artifact**.  A binary's
  ``fullname`` is the filename with the ``_output.csv`` suffix removed;
  its sidecars are the same-stem files in the same directory.  Because
  discovery anchors on ``_output.csv`` (not on directory names) it works
  for the flat layout (``out/clamav/<fullname>_output.csv``), the
  per-binary-subdir layout (``out/dataset/<prog>/<dir>/<fullname>_*``),
  and ignores log/aux directories (``sec-*``, timestamped runner dirs,
  ``unify_vocab`` — none of which carry ``_output.csv`` files).
  ``package`` is the first path component under ``out_root``.

* Phase-3 outputs live under ``out/build_memmap/<program>/`` (one
  directory per program).  Every file in such a directory is prefixed
  by the exact program (directory) name; the ``kind`` is the suffix that
  follows, normalised to a stable identifier.

The phase-1 kind suffixes are a fixed registry (the artifacts the
tokenize worker emits); the phase-3 kinds are derived from the per-file
suffix so a never-before-seen artifact still lands in the table under a
normalised kind rather than being dropped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# The phase-1 build_memmap subtree is NOT a package — it is discovered
# separately by :func:`discover_phase3`.
_BUILD_MEMMAP_DIRNAME = "build_memmap"

# Canonical anchor: a per-binary tokenize output always emits this file.
_OUTPUT_CSV_SUFFIX = "_output.csv"

# Phase-1 sidecar kind suffix → stable kind identifier.  The longest
# suffix that matches a sidecar filename's tail wins (``_output.csv`` is
# also the anchor and gets its own ``output_csv`` kind).
_PHASE1_KIND_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_output.csv", "output_csv"),
    ("_meta.json", "meta_json"),
    ("_strings.bin", "strings_bin"),
    ("_function_ranges.txt", "function_ranges"),
    ("_consts.txt", "consts_txt"),
    ("_output.mapping.b64c", "output_mapping"),
)


@dataclass(frozen=True, slots=True)
class Phase1File:
    """One phase-1 sidecar/artifact file present on disk."""

    kind: str
    path: Path
    size_bytes: int


@dataclass(frozen=True, slots=True)
class BinaryDir:
    """A discovered phase-1 binary: its identity plus the files present.

    ``output_csv`` is the anchor path (always present — it is what the
    binary was discovered by).  ``files`` lists every recognised sidecar
    found alongside it (one :class:`Phase1File` per file present)."""

    fullname: str
    package: str
    output_csv: Path
    files: list[Phase1File] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Phase3File:
    """One phase-3 build_memmap artifact file."""

    kind: str
    path: Path
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Phase3Program:
    """A discovered phase-3 program directory and its artifacts."""

    program: str
    files: list[Phase3File] = field(default_factory=list)


def _kind_for_phase1(filename: str, fullname: str) -> str | None:
    """Return the stable kind for a phase-1 sidecar belonging to
    ``fullname``, or ``None`` when the file is not a recognised
    artifact of that binary."""
    for suffix, kind in _PHASE1_KIND_SUFFIXES:
        if filename == fullname + suffix:
            return kind
    return None


def _scan_binary_dir(directory: Path, fullname: str) -> list[Phase1File]:
    """Collect every recognised phase-1 sidecar for ``fullname`` that
    sits in ``directory`` (a single os.scandir; no recursion)."""
    files: list[Phase1File] = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            kind = _kind_for_phase1(entry.name, fullname)
            if kind is None:
                continue
            files.append(
                Phase1File(kind=kind, path=Path(entry.path), size_bytes=entry.stat().st_size)
            )
    return files


def discover_binaries(out_root: Path) -> list[BinaryDir]:
    """Discover every phase-1 binary under ``out_root`` (excluding the
    ``build_memmap`` subtree).

    Anchors on ``*_output.csv`` files found at any depth.  ``package`` is
    the first path component under ``out_root``.  Sidecars are read from
    the same directory as the anchor only (so an ``output.mapping.b64c``
    copied elsewhere, e.g. under ``unify_vocab``, is never mis-attached).
    """
    build_memmap = out_root / _BUILD_MEMMAP_DIRNAME
    binaries: list[BinaryDir] = []
    for dirpath, dirnames, filenames in os.walk(out_root):
        current = Path(dirpath)
        # Prune the build_memmap subtree — it is a phase-3 concern.
        if current == build_memmap:
            dirnames[:] = []
            continue
        for filename in filenames:
            if not filename.endswith(_OUTPUT_CSV_SUFFIX):
                continue
            fullname = filename[: -len(_OUTPUT_CSV_SUFFIX)]
            rel = current.relative_to(out_root)
            package = rel.parts[0] if rel.parts else current.name
            binaries.append(
                BinaryDir(
                    fullname=fullname,
                    package=package,
                    output_csv=current / filename,
                    files=_scan_binary_dir(current, fullname),
                )
            )
    return binaries


def _kind_for_phase3(filename: str, program: str) -> str:
    """Normalise a phase-3 filename's program-stripped tail to a stable
    kind identifier.  Every build_memmap file is prefixed by its program
    directory name, so the tail is what distinguishes the artifact."""
    tail = filename[len(program) :]
    # Strip the leading separator (``_`` for the *_bin/csv family,
    # ``.`` for ``.error.log`` / ``.warn.log``) then normalise the rest
    # into an identifier: dots → underscores, lower-cased.
    tail = tail.lstrip("_.")
    return tail.replace(".", "_").lower()


def discover_phase3(out_root: Path) -> list[Phase3Program]:
    """Discover every phase-3 program directory under
    ``out/build_memmap`` and the artifacts inside it."""
    build_memmap = out_root / _BUILD_MEMMAP_DIRNAME
    if not build_memmap.is_dir():
        return []
    programs: list[Phase3Program] = []
    with os.scandir(build_memmap) as prog_entries:
        for prog_entry in prog_entries:
            if not prog_entry.is_dir():
                continue
            program = prog_entry.name
            files: list[Phase3File] = []
            with os.scandir(prog_entry.path) as file_entries:
                for file_entry in file_entries:
                    if not file_entry.is_file():
                        continue
                    files.append(
                        Phase3File(
                            kind=_kind_for_phase3(file_entry.name, program),
                            path=Path(file_entry.path),
                            size_bytes=file_entry.stat().st_size,
                        )
                    )
            programs.append(Phase3Program(program=program, files=files))
    return programs
