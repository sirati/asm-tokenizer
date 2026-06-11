"""Discovery + cross-directory naming for the collection layer.

Single concern: *given a set of memmap directories + a (reduction,
depth) request, resolve the kept :class:`CollectionMember` list (with
unbiased-exclusion accounting) plus the per-member readers + datasets.*

Two orthogonal discovery questions are answered here by reuse, NOT by
restating any grammar:

* "which ``.idx`` files exist per directory" delegates to
  :func:`discover_indices` (the canonical filename grammar lives in
  ``_reader``).
* "which binaries physically exist" is the ``<binary>_index.bin``
  presence rule (:func:`_existing_binaries`) -- the unmatched-arm
  sidecar shares the suffix tail and is excluded.

The cross-directory naming rule (unique names stay bare; colliding
names are qualified by directory; same-``dir.name`` collisions are
refused) is owned by :func:`_qualify_names`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset

from .._reader import SortedIndexReader, discover_indices
from .._types import LengthReduction
from ._member import CollectionMember, MissingIndexPolicy


__all__ = ["discover_members"]


_LOGGER = logging.getLogger(__name__)

# Per-binary catalog suffixes. A binary EXISTS iff its matched-arm
# ``<binary>_index.bin`` is present; the unmatched-arm sidecar shares
# the suffix tail and must be excluded when deriving names.
_INDEX_SUFFIX = "_index.bin"
_UNMATCHED_INDEX_SUFFIX = "_unmatched_index.bin"


def _existing_binaries(memmap_dir: Path) -> List[str]:
    """Binary names whose matched-arm ``<binary>_index.bin`` is present.

    The unmatched-arm sidecar ``<binary>_unmatched_index.bin`` shares
    the ``_index.bin`` tail; it is NOT a binary-existence signal and is
    excluded. Returns names in directory-iteration order (the caller
    sorts).
    """
    names: List[str] = []
    for entry in Path(memmap_dir).iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(_UNMATCHED_INDEX_SUFFIX):
            continue
        if not name.endswith(_INDEX_SUFFIX):
            continue
        names.append(name[: -len(_INDEX_SUFFIX)])
    return names


def _has_index(
    reductions: Optional[Sequence[LengthReduction]],
    reduction: LengthReduction,
) -> bool:
    """Whether ``reduction`` appears among a binary's discovered indices.

    :func:`discover_indices` keys by the filename tag, so equality is on
    the canonical :meth:`LengthReduction.filename_tag` spelling.
    """
    if reductions is None:
        return False
    tag = reduction.filename_tag()
    return any(red.filename_tag() == tag for red in reductions)


def _claim(
    used: Dict[str, Tuple[Path, str]],
    qualified: Dict[Tuple[Path, str], str],
    label: str,
    key: Tuple[Path, str],
) -> None:
    """Assign ``label`` to ``key``, refusing an already-claimed label."""
    if label in used:
        prior = used[label]
        raise ValueError(
            "IndexedMemmapCollection: ambiguous qualified name "
            f"{label!r} maps to both {prior!r} and {key!r}; two memmap "
            "directories share a name and the same binary -- refusing.",
        )
    used[label] = key
    qualified[key] = label


def _qualify_names(
    raw: Sequence[Tuple[Path, str]],
) -> Dict[Tuple[Path, str], str]:
    """Map each ``(memmap_dir, binary_name)`` pair to its qualified name.

    A ``binary_name`` unique across the whole collection keeps its bare
    name. A colliding name is qualified ``<dir.name>/<binary>`` for ALL
    colliding members. If qualification still collides (two directories
    with the same ``dir.name``), this raises :class:`ValueError` --
    ambiguity is refused rather than silently resolved.
    """
    by_binary: Dict[str, List[Tuple[Path, str]]] = {}
    for memmap_dir, binary_name in raw:
        by_binary.setdefault(binary_name, []).append((memmap_dir, binary_name))

    qualified: Dict[Tuple[Path, str], str] = {}
    used: Dict[str, Tuple[Path, str]] = {}
    for binary_name, members in by_binary.items():
        if len(members) == 1:
            memmap_dir, _ = members[0]
            _claim(used, qualified, binary_name, (memmap_dir, binary_name))
            continue
        for memmap_dir, _ in members:
            label = f"{memmap_dir.name}/{binary_name}"
            _claim(used, qualified, label, (memmap_dir, binary_name))
    return qualified


def _idx_path(memmap_dir: Path, binary_name: str, reduction, depth: int) -> Path:
    """Canonical sorted-index path for ``(binary, reduction, depth)``.

    Mirrors the grammar :func:`write_sorted_index_files` stamps; the
    presence of this exact file was already confirmed in discovery, so
    this only re-derives the path the :class:`SortedIndexReader` opens.
    """
    return (
        memmap_dir
        / f"{binary_name}_sorted_{reduction.filename_tag()}_d{depth:03d}.idx"
    )


def discover_members(
    memmap_dirs: Sequence[Path],
    *,
    reduction: LengthReduction,
    depth: int,
    vocab_manager: Optional[Any],
    on_missing: MissingIndexPolicy,
) -> Tuple[
    List[CollectionMember],
    Dict[str, SortedIndexReader],
    Dict[str, BinaryDataset],
]:
    """Resolve kept members + their readers + (unopened) datasets.

    For each directory, a binary EXISTS iff its matched-arm
    ``<binary>_index.bin`` is present. For each existing binary the
    ``.idx`` matching ``(reduction.filename_tag(), depth)`` must exist
    (via :func:`discover_indices`). Missing pairs are handled per
    ``on_missing`` (RAISE: one :class:`ValueError` listing all; SKIP: an
    ERROR log per excluded binary, then exclude). Returns the
    alphabetical-by-``qualified_name`` member list plus the
    ``{qualified_name -> SortedIndexReader}`` and
    ``{qualified_name -> BinaryDataset}`` maps the collection wires into
    its sampler + session machinery.
    """
    kept: List[Tuple[Path, str]] = []
    missing: List[Tuple[Path, str]] = []
    for memmap_dir in memmap_dirs:
        memmap_dir = Path(memmap_dir)
        by_binary = discover_indices(memmap_dir, depth=depth)
        for binary_name in _existing_binaries(memmap_dir):
            if _has_index(by_binary.get(binary_name), reduction):
                kept.append((memmap_dir, binary_name))
            else:
                missing.append((memmap_dir, binary_name))

    _handle_missing(missing, reduction=reduction, depth=depth, on_missing=on_missing)

    qualified = _qualify_names(kept)
    members: List[CollectionMember] = [
        CollectionMember(
            qualified_name=qualified[(memmap_dir, binary_name)],
            memmap_dir=memmap_dir,
            binary_name=binary_name,
        )
        for memmap_dir, binary_name in kept
    ]
    members.sort(key=lambda m: m.qualified_name)

    readers: Dict[str, SortedIndexReader] = {}
    datasets: Dict[str, BinaryDataset] = {}
    for member in members:
        readers[member.qualified_name] = SortedIndexReader(
            _idx_path(member.memmap_dir, member.binary_name, reduction, depth),
            reduction=reduction,
            depth=depth,
        )
        datasets[member.qualified_name] = BinaryDataset(
            member.memmap_dir,
            member.binary_name,
            vocab_manager=vocab_manager,
        )
    return members, readers, datasets


def _handle_missing(
    missing: List[Tuple[Path, str]],
    *,
    reduction: LengthReduction,
    depth: int,
    on_missing: MissingIndexPolicy,
) -> None:
    """Apply ``on_missing`` to the excluded ``(dir, binary)`` pairs."""
    if not missing:
        return
    if on_missing is MissingIndexPolicy.RAISE:
        listing = ", ".join(
            f"({d}, {b})"
            for d, b in sorted(missing, key=lambda p: (str(p[0]), p[1]))
        )
        raise ValueError(
            "IndexedMemmapCollection.discover: missing sorted-index file "
            f"for reduction={reduction.filename_tag()!r} depth={depth} in: "
            f"{listing}",
        )
    for d, b in missing:  # SKIP_WITH_ERROR_LOG
        _LOGGER.error(
            "IndexedMemmapCollection: excluding binary %r in %s -- no "
            "sorted-index for reduction=%s depth=%d (this is a sampling "
            "bias)",
            b,
            d,
            reduction.filename_tag(),
            depth,
        )
