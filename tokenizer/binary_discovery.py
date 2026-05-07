"""Per-directory dispatch between legacy 4-axis filenames and the new
sidecar-JSON dataset layout.

Single concern: turn a dataset directory tree into a flat stream of
``(BinaryHandle, VariantInfo)`` pairs. Format detection is per-subdir
and lazy: each directory is classified once at descent time as
"sidecar" (contains at least one ``*.json`` paired with a same-stem
``*.tar.zst``) or "legacy" (everything else). Mixed-format trees work
naturally because dispatch happens per directory, not for the tree as
a whole.

This module owns *only* the walk + dispatch concern. Filename / sidecar
parsing is delegated to ``VariantInfo.from_legacy_filename`` and
``VariantInfo.from_sidecar`` — the single source of truth for either
format. No filtering (platform / compiler / opt allowlists) lives here:
that is the caller's concern, applied to the emitted pairs.

The opaque ``BinaryHandle`` exists because the two formats reach the
binary content via different paths:

* Legacy: ``path`` is the binary file itself; ``tarball`` is ``None``.
* Sidecar: ``path`` is the JSON sidecar; ``tarball`` is the sibling
  ``*.tar.zst`` archive (worker extracts at task time per the
  per-task-extraction decision in the plan).

Downstream consumers (tokenize / vocab / memmap discover_items) only
need ``handle.path`` (used as ``TaskInfo.path`` — legacy convention
preserved) and, in sidecar mode, ``handle.tarball`` (forwarded to the
worker via payload for extraction).
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

# Sidecar filename pair: the JSON carries metadata; the same-stem
# .tar.zst carries the binary. Detection looks for at least one
# matched pair in the directory; orphans on either side are reported
# (warning) and skipped.
_SIDECAR_JSON_SUFFIX = ".json"
_SIDECAR_TARBALL_SUFFIX = ".tar.zst"


@dataclass(frozen=True)
class BinaryHandle:
    """Opaque locator for the binary content of one variant.

    Two flavors, distinguished by whether ``tarball`` is set:

    * Legacy (``tarball is None``): ``path`` is the binary file on
      disk. The caller can pass it straight to a tokenizer worker.
    * Sidecar (``tarball`` is a Path): ``path`` is the JSON sidecar;
      ``tarball`` is the matching ``*.tar.zst`` archive. The worker
      extracts the archive at task time and finds the binary inside.
    """

    path: Path
    tarball: Path | None = None


def _classify_dir_files(filenames: list[str]) -> set[str]:
    """Return the set of stems for which BOTH a ``.json`` sidecar and a
    matching ``.tar.zst`` archive exist in this directory. An empty set
    means the directory is not in sidecar mode (legacy fallback)."""
    json_stems = {
        name[: -len(_SIDECAR_JSON_SUFFIX)]
        for name in filenames
        if name.endswith(_SIDECAR_JSON_SUFFIX)
    }
    tarball_stems = {
        name[: -len(_SIDECAR_TARBALL_SUFFIX)]
        for name in filenames
        if name.endswith(_SIDECAR_TARBALL_SUFFIX)
    }
    return json_stems & tarball_stems


def _emit_sidecar_dir(
    dir_path: Path,
    filenames: list[str],
    paired_stems: set[str],
) -> Iterator[tuple[BinaryHandle, VariantInfo]]:
    """Yield ``(handle, variant)`` for every paired sidecar in this
    directory, sorted by stem for deterministic order. Orphan JSONs
    (no matching tarball) are warned-and-skipped per the plan."""
    json_names = sorted(
        name for name in filenames if name.endswith(_SIDECAR_JSON_SUFFIX)
    )
    for json_name in json_names:
        stem = json_name[: -len(_SIDECAR_JSON_SUFFIX)]
        if stem not in paired_stems:
            _logger.warning(
                "sidecar JSON %s has no matching %s%s — skipping",
                dir_path / json_name,
                stem,
                _SIDECAR_TARBALL_SUFFIX,
            )
            continue
        json_path = dir_path / json_name
        tarball_path = dir_path / f"{stem}{_SIDECAR_TARBALL_SUFFIX}"
        variant = VariantInfo.from_sidecar(json_path)
        yield BinaryHandle(path=json_path, tarball=tarball_path), variant


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
        yield BinaryHandle(path=candidate, tarball=None), variant


def walk_dataset(
    source_dir: Path,
) -> Iterator[tuple[BinaryHandle, VariantInfo]]:
    """Walk ``source_dir`` recursively, classify each directory as
    sidecar or legacy, and yield ``(handle, variant)`` pairs.

    Detection is lazy and per-directory: a directory is in sidecar mode
    iff it contains at least one ``*.json`` paired with a same-stem
    ``*.tar.zst``. Otherwise the directory is treated as legacy. The
    walk descends into all subdirectories of either flavor, so a
    sidecar directory under a legacy parent (or vice versa) works.

    Order: directories are visited in sorted top-down order, files
    within a directory are emitted in sorted order. Caller can rely on
    the iteration order being a deterministic function of the tree.
    """
    for root, dirs, files in os.walk(source_dir):
        dirs.sort()
        dir_path = Path(root)
        paired_stems = _classify_dir_files(files)
        if paired_stems:
            yield from _emit_sidecar_dir(dir_path, files, paired_stems)
        else:
            yield from _emit_legacy_dir(dir_path, files)
