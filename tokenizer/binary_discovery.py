"""Per-directory dispatch between legacy 4-axis filenames and the
sidecar-JSON dataset layout.

Single concern: turn a dataset directory tree into a flat stream of
``(BinaryHandle, VariantInfo)`` pairs. Format detection is per-subdir
and lazy: each directory is classified once at descent time as
"sidecar" (``*.json`` paired with same-stem directory containing the
binary) or "legacy" (4-axis filenames). Both flavors can coexist in
one directory — pairs are emitted per stem.

This module owns *only* the walk + dispatch concern. Filename / sidecar
parsing is delegated to ``VariantInfo.from_legacy_filename`` and
``VariantInfo.from_sidecar`` — the single source of truth for either
format. No filtering (platform / compiler / opt allowlists) lives here:
that is the caller's concern, applied to the emitted pairs.

The opaque ``BinaryHandle`` exists because the two formats reach the
binary content via different paths:

* Legacy: ``path`` is the binary file itself; ``variant_dir`` is
  ``None``.
* Sidecar: ``path`` is one binary file inside the variant's folder;
  ``variant_dir`` is that folder. A folder may hold SEVERAL binaries
  (a CLI tool plus its siblings, or a library's ``.so``); one sidecar's
  metadata applies to all of them, so the folder yields one handle per
  binary — same ``VariantInfo``, distinct ``handle.path`` /
  ``handle.binary_name``. The worker reads ``path`` directly — discovery
  already located the binary. Worker-side platform-derive routing keys
  on ``VariantInfo.variant_id != 0`` (sidecar) vs ``== 0`` (legacy).

Downstream consumers (tokenize / vocab / memmap discover_items) only
need ``handle.path`` (used as ``TaskInfo.path``) and
``handle.binary_name`` (the per-binary identity slot for output
filenames / task ids); the worker reads ``path`` directly in both
flavors.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from tokenizer.variant_info import VariantInfo

__all__ = ["BinaryHandle", "walk_dataset"]

_logger = logging.getLogger(__name__)

# Sidecar pairing: a ``*.json`` carrying the variant's metadata next
# to a same-stem subdirectory carrying the binary content. Detection
# looks for at least one matched pair in the directory; orphans on
# either side are reported (warning) and skipped.
_SIDECAR_JSON_SUFFIX = ".json"

# A sidecar variant folder holds the package's real binaries: the CLI
# tool(s) and/or shared object(s). Membership is decided by ELF magic
# (read, never trusted to extension) so a differently-named tool
# (``sqlite`` → ``sqlite3``) and version-suffixed shared objects
# (``libz.so.1.3.2``) are captured uniformly. Split debug-symbol ELFs
# (``*.debug``) ARE ELF but not real binaries — excluded by name.
_ELF_MAGIC = b"\x7fELF"
_DEBUG_SUFFIX = ".debug"


@dataclass(frozen=True)
class BinaryHandle:
    """Opaque locator for the binary content of one variant.

    Two flavors, distinguished by whether ``variant_dir`` is set:

    * Legacy (``variant_dir is None``): ``path`` is the binary file on
      disk (its filename encodes the 4-axis variant info).
    * Sidecar (``variant_dir`` is a Path): ``path`` is one binary file
      inside the variant's folder; the JSON sidecar metadata is already
      decoded into ``VariantInfo``. One sidecar's metadata applies to
      EVERY binary in its folder, so a single ``variant_dir`` yields one
      handle per binary — each with its own ``path`` (and therefore its
      own ``binary_name``) but the same shared ``VariantInfo``.

    ``binary_name`` is the per-binary identity slot that distinguishes
    this handle within its variant group. It is the single seam the
    downstream output-filename / task-id machinery keys on (instead of
    ``VariantInfo.pkg``, which is package-level and shared across all
    binaries of a multi-binary sidecar folder). Discovery is the single
    owner that knows the flavor and fills this:

    * Sidecar: the on-disk file's basename (``flac``, ``libz.so.1.3.2``,
      ...) — the real binary name, distinct per binary in the folder.
    * Legacy: the binary-name slot the 4-axis filename already encodes
      (i.e. ``VariantInfo.pkg``), so legacy outputs stay byte-identical.

    Callers never branch on flavor — they read ``binary_name`` directly.
    """

    path: Path
    binary_name: str
    variant_dir: Path | None = None

    def binary_size(self) -> int:
        """Uncompressed size of the binary content this handle locates.

        Single source of truth for "how big is the thing tokenization
        will operate on". Both flavors return ``path.stat().st_size``
        — ``path`` is always a regular file on disk.
        """
        return self.path.stat().st_size


def _classify_dir_files(
    filenames: list[str], dirnames: list[str]
) -> set[str]:
    """Return stems where a ``.json`` sidecar pairs with a same-stem
    subdirectory in this dir. An empty set means the directory is in
    legacy fallback."""
    json_stems = {
        name[: -len(_SIDECAR_JSON_SUFFIX)]
        for name in filenames
        if name.endswith(_SIDECAR_JSON_SUFFIX)
    }
    return json_stems & set(dirnames)


def _is_elf_binary(path: Path) -> bool:
    """True iff ``path`` is a regular file whose first bytes are the ELF
    magic and whose name is not a split debug-symbol object.

    Membership is decided by content (magic bytes), never by extension,
    so shared objects (``libz.so.1.3.2``) and differently-named tools
    (``sqlite3``) qualify uniformly while a README / packing manifest
    does not. ``*.debug`` files are ELF but carry only split debug
    symbols, not a real binary — excluded by name. Hidden files (leading
    dot) are likewise excluded by name: a real binary never starts with a
    dot, and an atomic-publish leftover such as
    ``.<name>.publish-tmp.<host>.<pid>.<nanos>`` is a full ELF *copy* of
    the binary that must NOT be enumerated as a second binary."""
    if path.name.startswith(".") or path.name.endswith(_DEBUG_SUFFIX):
        return False
    try:
        with path.open("rb") as fh:
            return fh.read(len(_ELF_MAGIC)) == _ELF_MAGIC
    except OSError:
        return False


def _emit_sidecar_dir(
    dir_path: Path,
    folder_stems: set[str],
    skip_tally: "_SkipTally",
) -> Iterator[tuple[BinaryHandle, VariantInfo]]:
    """Yield ``(handle, variant)`` for EVERY ELF binary in each
    sidecar-paired variant folder, sorted by stem then by binary name.

    One sidecar's metadata applies to all binaries in its folder, so a
    folder with several binaries (CLI tool + siblings, or a library's
    ``.so``) yields one handle per binary — same ``variant``, distinct
    ``path`` / ``binary_name``. Each ``path`` is read directly by the
    worker with no extraction. Folders with zero qualifying binaries
    (empty / malformed / dangling link / only ``*.debug``) are recorded
    on ``skip_tally``, which the walk summarises in ONE aggregate warning
    instead of one line per folder (a 55k-folder corpus with a systematic
    mount/link fault otherwise floods the log)."""
    for stem in sorted(folder_stems):
        json_path = dir_path / f"{stem}{_SIDECAR_JSON_SUFFIX}"
        variant = VariantInfo.from_sidecar(json_path)
        variant_dir = dir_path / stem
        try:
            entries = sorted(variant_dir.iterdir())
        except OSError:
            # Dangling link / unreadable folder — treated like an
            # empty folder (no qualifying binaries), tallied below.
            entries = []
        emitted = 0
        for entry in entries:
            if not _is_elf_binary(entry):
                continue
            emitted += 1
            yield (
                BinaryHandle(
                    path=entry,
                    binary_name=entry.name,
                    variant_dir=variant_dir,
                ),
                variant,
            )
        if emitted == 0:
            skip_tally.record(variant_dir)


class _SkipTally:
    """Aggregates zero-binary sidecar-folder skips across one walk.

    Single concern: count skips, keep the first few example paths, and
    emit one summary warning at walk end (nothing when count is 0).
    """

    _MAX_EXAMPLES = 3

    def __init__(self) -> None:
        self.count = 0
        self.examples: list[Path] = []

    def record(self, variant_dir: Path) -> None:
        self.count += 1
        if len(self.examples) < self._MAX_EXAMPLES:
            self.examples.append(variant_dir)
        _logger.debug("sidecar folder %s has no ELF binary — skipping", variant_dir)

    def emit_summary(self) -> None:
        if not self.count:
            return
        _logger.warning(
            "skipped %d sidecar folder(s) with no qualifying ELF binary "
            "(empty/dangling symlink, malformed sidecar, or only debug "
            "objects); examples: %s",
            self.count,
            ", ".join(str(p) for p in self.examples),
        )


def _emit_orphan_json_warnings(
    dir_path: Path,
    filenames: list[str],
    paired_stems: set[str],
) -> None:
    """Warn about ``*.json`` sidecars that aren't paired with a
    same-stem directory."""
    for name in sorted(filenames):
        if not name.endswith(_SIDECAR_JSON_SUFFIX):
            continue
        stem = name[: -len(_SIDECAR_JSON_SUFFIX)]
        if stem in paired_stems:
            continue
        _logger.warning(
            "sidecar JSON %s has no matching %s/ — skipping",
            dir_path / name,
            stem,
        )


def _emit_legacy_dir(
    dir_path: Path,
    filenames: list[str],
) -> Iterator[tuple[BinaryHandle, VariantInfo]]:
    """Yield ``(handle, variant)`` for every legacy-format filename in
    this directory, sorted for deterministic order. Files whose name
    does not match the 4-axis convention are silently skipped — the
    legacy parser returns ``None`` and ``VariantInfo.from_legacy_filename``
    surfaces that as ``ValueError``."""
    for name in sorted(filenames):
        if name.startswith("."):
            continue
        candidate = dir_path / name
        if not candidate.is_file():
            continue
        try:
            variant = VariantInfo.from_legacy_filename(candidate)
        except ValueError:
            continue
        # The 4-axis filename already encodes the binary-name slot as
        # ``variant.pkg``; carry it verbatim so legacy outputs stay
        # byte-identical (no flavor branch downstream).
        yield (
            BinaryHandle(
                path=candidate,
                binary_name=variant.pkg,
                variant_dir=None,
            ),
            variant,
        )


def walk_dataset(
    source_dir: Path,
) -> Iterator[tuple[BinaryHandle, VariantInfo]]:
    """Walk ``source_dir`` recursively, classify each directory as
    sidecar or legacy, and yield ``(handle, variant)`` pairs.

    Detection is lazy and per-directory: a directory is in sidecar
    mode iff it contains at least one ``*.json`` paired with a
    same-stem subdirectory. Sidecar-paired folders are NOT descended
    into for further walking (they only contain the variant's binary
    — descending would re-emit it through the legacy fallback). Other
    subdirectories are walked normally, so a sidecar directory under
    a legacy parent (or vice versa) works.

    Order: directories visited in sorted top-down order; within a
    directory, sidecar pairs emit first (alphabetical) then legacy
    files. Iteration order is a deterministic function of the tree.
    """
    skip_tally = _SkipTally()
    try:
        for root, dirs, files in os.walk(source_dir):
            dirs.sort()
            dir_path = Path(root)
            paired_stems = _classify_dir_files(files, dirs)
            if paired_stems:
                yield from _emit_sidecar_dir(dir_path, paired_stems, skip_tally)
                _emit_orphan_json_warnings(dir_path, files, paired_stems)
                dirs[:] = [d for d in dirs if d not in paired_stems]
            else:
                yield from _emit_legacy_dir(dir_path, files)
    finally:
        skip_tally.emit_summary()
