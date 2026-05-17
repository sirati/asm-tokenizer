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
* Sidecar: ``path`` is the binary file inside the variant's folder
  (``<variant_dir>/<pkg>``); ``variant_dir`` is that folder. The
  worker reads ``path`` directly — discovery already located the
  binary. Worker-side platform-derive routing keys on
  ``VariantInfo.variant_id != 0`` (sidecar) vs ``== 0`` (legacy).

Downstream consumers (tokenize / vocab / memmap discover_items) only
need ``handle.path`` (used as ``TaskInfo.path``); the worker reads it
directly in both flavors.
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


@dataclass(frozen=True)
class BinaryHandle:
    """Opaque locator for the binary content of one variant.

    Two flavors, distinguished by whether ``variant_dir`` is set:

    * Legacy (``variant_dir is None``): ``path`` is the binary file on
      disk (its filename encodes the 4-axis variant info).
    * Sidecar (``variant_dir`` is a Path): ``path`` is the binary file
      inside the variant's folder (``<variant_dir>/<pkg>``); the JSON
      sidecar metadata is already decoded into ``VariantInfo``.
    """

    path: Path
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


def _emit_sidecar_dir(
    dir_path: Path,
    folder_stems: set[str],
) -> Iterator[tuple[BinaryHandle, VariantInfo]]:
    """Yield ``(handle, variant)`` for every sidecar-paired variant in
    this directory, sorted by stem. ``path`` is the binary file inside
    the variant's folder (``<stem>/<variant.pkg>``); the worker reads
    it directly with no extraction. Folders that don't contain the
    expected ``<pkg>`` file are warned-and-skipped (malformed
    sidecar)."""
    for stem in sorted(folder_stems):
        json_path = dir_path / f"{stem}{_SIDECAR_JSON_SUFFIX}"
        variant = VariantInfo.from_sidecar(json_path)
        variant_dir = dir_path / stem
        binary_path = variant_dir / variant.pkg
        if not binary_path.is_file():
            _logger.warning(
                "sidecar folder %s missing expected binary %r — skipping",
                variant_dir,
                variant.pkg,
            )
            continue
        yield BinaryHandle(path=binary_path, variant_dir=variant_dir), variant


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
        yield BinaryHandle(path=candidate, variant_dir=None), variant


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
    for root, dirs, files in os.walk(source_dir):
        dirs.sort()
        dir_path = Path(root)
        paired_stems = _classify_dir_files(files, dirs)
        if paired_stems:
            yield from _emit_sidecar_dir(dir_path, paired_stems)
            _emit_orphan_json_warnings(dir_path, files, paired_stems)
            dirs[:] = [d for d in dirs if d not in paired_stems]
        else:
            yield from _emit_legacy_dir(dir_path, files)
