"""Per-binary memmap-directory scan + post-discovery selection.

Single concern: turn a memmap directory path into a sorted list of
discovered binaries (each recognised by its matched-arm
``<name>_index.bin`` sidecar, paired with the directory that sidecar
lives in) plus the ``--only`` / ``--max-binaries`` post-selection.

Two on-disk layouts are auto-detected by entry kind in ONE scan:

* FLAT -- the ``<name>_index.bin`` sidecars sit directly in
  ``memmap_dir`` (the standalone CLIs / per-package validator usage, and
  the standalone-publish mode where ``staged_publish`` ignores its
  scope). Each binary's ``memmap_dir`` IS the scanned directory.
* NESTED one level -- ``memmap_dir`` holds per-binary SUBDIRECTORIES
  (the container/SLURM publish mode, where ``build_memmap`` republishes
  under its ``build_memmap/<name>/`` scope). Each subdirectory IS the
  ``memmap_dir`` for whatever ``<name>_index.bin`` it contains.

The per-entry rule -- "files are binaries here, directories hold
binaries one level down" -- handles both with a single shared
``_index_names_in(dir)`` exclusion applied at every level, so the
GEOMETRY/unmatched/realized-lengths sidecar exclusions can never drift
between the flat and nested cases.

Owned by the ``tokenizer.aligned_data`` package because every consumer
operates on the aligned-data memmap layout and this scan already keys
off the realized_lengths arm grammars: the realized-length and
sorted-index generators (their ``__main__`` discovery) and the dynrunner
phase-4 ``build_index`` task all select their binary set through here,
and ``tools.run_batch_smoke`` re-exports it for the same scan. Living in
``aligned_data`` keeps it inside the production image (``tools`` is not
packaged), so the mesh workers can import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

# Import the arm grammar from the realized_lengths ``_format`` submodule
# (not the package ``__init__``, whose generator pulls in sorted_index
# and risks a circular import): ``_format`` is a leaf module.
from tokenizer.aligned_data.realized_lengths._format import ARMS
from tokenizer.aligned_data.realized_lengths._geometry_format import GEOMETRY_ARMS

_INDEX_SUFFIX = "_index.bin"
_UNMATCHED_INDEX_SUFFIX = "_unmatched_index.bin"

# Realized-length sidecars also end in ``_index.bin`` but are NOT
# binary-existence signals -- they co-exist in the memmap dir once the
# realized-lengths pass has run (the sorted-index build's Phase-4a
# precondition). BOTH sidecar families must be excluded: the length-CSR
# arms (``_lengths_index.bin`` / ``_unmatched_lengths_index.bin``) AND
# the realized-geometry RLG3 arms (``_realized_index.bin`` /
# ``_unmatched_realized_index.bin``). Sourced from BOTH arm grammars so
# this exclusion never drifts from the generator's filenames.
_REALIZED_LENGTHS_INDEX_SUFFIXES = tuple(
    arm.index_suffix for arm in ARMS
) + tuple(arm.index_suffix for arm in GEOMETRY_ARMS)


@dataclass(frozen=True)
class DiscoveredBinary:
    """A binary found by the memmap-directory scan.

    ``name`` is the matched-arm stem (``<name>_index.bin`` minus the
    suffix). ``memmap_dir`` is the directory that sidecar lives in -- the
    scanned directory itself in the flat layout, or its per-binary
    subdirectory in the nested layout. Consumers pass ``memmap_dir`` as
    the ``base_path`` to the realized-length / sorted-index generators so
    every per-binary sidecar path resolves regardless of layout.
    """

    name: str
    memmap_dir: Path


def _index_names_in(directory: Path) -> Iterator[str]:
    """Yield the matched-arm binary names whose ``<name>_index.bin``
    sidecar sits DIRECTLY in ``directory``.

    The single point that turns a directory's files into binary names:
    the ``<name>_unmatched_index.bin`` companion and the realized-length
    CSR / realized-geometry sidecars (which also end in ``_index.bin``)
    are skipped so each binary is yielded once via its matched arm. Used
    for BOTH the top-level flat scan and each nested subdirectory, so the
    exclusion set is identical at every level.
    """
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(_UNMATCHED_INDEX_SUFFIX):
            continue
        if name.endswith(_REALIZED_LENGTHS_INDEX_SUFFIXES):
            continue
        if not name.endswith(_INDEX_SUFFIX):
            continue
        yield name[: -len(_INDEX_SUFFIX)]


def discover_binaries(memmap_dir: Path) -> List[DiscoveredBinary]:
    """Return the binaries present in ``memmap_dir``, sorted by name.

    Auto-detects layout by entry kind in a single ``iterdir``: matched-arm
    ``<name>_index.bin`` FILES are binaries living directly in
    ``memmap_dir`` (flat), while each SUBDIRECTORY is scanned one level
    down for its own ``<name>_index.bin`` (nested -- the container publish
    layout). The same ``_index_names_in`` exclusion runs at both levels.
    Recursion is exactly one level: subdirectories of subdirectories are
    not descended.
    """
    found: List[DiscoveredBinary] = []
    for name in _index_names_in(memmap_dir):
        found.append(DiscoveredBinary(name=name, memmap_dir=memmap_dir))
    for entry in memmap_dir.iterdir():
        if not entry.is_dir():
            continue
        for name in _index_names_in(entry):
            found.append(DiscoveredBinary(name=name, memmap_dir=entry))
    return sorted(found, key=lambda b: b.name)


def filter_binaries(
    binaries: Sequence[DiscoveredBinary],
    *,
    only: Optional[str],
    max_binaries: Optional[int],
) -> List[DiscoveredBinary]:
    """Apply the ``--only`` allow-list then the ``--max-binaries`` cap.

    Order is preserved (so the discovery sort still drives
    reproducibility) and the cap is applied AFTER the allow-list so
    ``--only`` semantics are exact. ``--only`` matches on the binary name
    (``DiscoveredBinary.name``).
    """
    selected = list(binaries)
    if only is not None:
        keep = {x.strip() for x in only.split(",") if x.strip()}
        selected = [b for b in selected if b.name in keep]
    if max_binaries is not None:
        selected = selected[:max_binaries]
    return selected
