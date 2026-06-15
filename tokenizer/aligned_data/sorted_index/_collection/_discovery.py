"""Discovery + cross-directory naming for the collection layer.

Single concern: *given a set of memmap directories + a list of
:class:`IndexSpec` requests, resolve the kept :class:`CollectionMember`
list (with unbiased-exclusion accounting) plus the per-spec per-member
readers + the shared per-member datasets.*

Two orthogonal discovery questions are answered here by reuse, NOT by
restating any grammar:

* "which ``.idx`` files exist per directory" delegates to
  :func:`discover_indices` (the canonical filename grammar lives in
  ``_reader``).
* "which binaries physically exist" is the ``<binary>_index.bin``
  presence rule (:func:`_existing_binaries`) -- the unmatched-arm
  sidecar shares the suffix tail and is excluded.

Membership is spec-INDEPENDENT and uniform across every requested spec:
a binary is kept iff it carries the ``.idx`` for EVERY spec. A binary
missing any spec's index is excluded from the WHOLE collection (all
specs), per :class:`MissingIndexPolicy`. This keeps ``members`` /
qualified naming / sessions collection-level so the per-spec sampling
pools differ only in their lengths, never in their population --
mixed membership would silently shrink one spec's universe relative to
another's (a sampling bias).

The cross-directory naming rule (unique names stay bare; colliding
names are qualified by directory; same-``dir.name`` collisions are
refused) is owned by :func:`_qualify_names`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
# ``_format`` is a leaf module; the realized_lengths package ``__init__``
# pulls in the generator (which imports sorted_index), so import the arm
# grammar from the submodule to stay clear of the import cycle.
from tokenizer.aligned_data.realized_lengths._format import ARMS as _ARMS

from .._reader import SortedIndexReader, discover_indices
from .._types import IndexSpec, LengthReduction
from ._member import CollectionMember, MissingIndexPolicy
from ._spec import spec_tag


__all__ = ["discover_members"]


_LOGGER = logging.getLogger(__name__)

# Per-binary catalog suffixes. A binary EXISTS iff its matched-arm
# ``<binary>_index.bin`` is present; the unmatched-arm catalog sidecar
# and the realized-length CSR sidecars share the ``_index.bin`` tail and
# must be excluded when deriving names.
_INDEX_SUFFIX = "_index.bin"
_UNMATCHED_INDEX_SUFFIX = "_unmatched_index.bin"

# Realized-length CSR sidecars (``<binary>_lengths_index.bin`` /
# ``<binary>_unmatched_lengths_index.bin``) also end in ``_index.bin``
# but are NOT binary-existence signals. Sourced from the realized_lengths
# arm grammar so this exclusion never drifts from the generator's
# filenames.
_REALIZED_LENGTHS_INDEX_SUFFIXES = tuple(arm.index_suffix for arm in _ARMS)


def _existing_binaries(memmap_dir: Path) -> List[str]:
    """Binary names whose matched-arm ``<binary>_index.bin`` is present.

    The unmatched-arm catalog sidecar ``<binary>_unmatched_index.bin``
    and the realized-length CSR sidecars
    (``<binary>_lengths_index.bin`` / ``<binary>_unmatched_lengths_index.bin``)
    share the ``_index.bin`` tail; none is a binary-existence signal and
    all are excluded. Returns names in directory-iteration order (the
    caller sorts).
    """
    names: List[str] = []
    for entry in Path(memmap_dir).iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(_UNMATCHED_INDEX_SUFFIX):
            continue
        if name.endswith(_REALIZED_LENGTHS_INDEX_SUFFIXES):
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


def _idx_path(memmap_dir: Path, binary_name: str, spec: IndexSpec) -> Path:
    """Canonical sorted-index path for ``(binary, spec)``.

    Mirrors the grammar :func:`write_sorted_index_files` stamps; the
    presence of this exact file was already confirmed in discovery, so
    this only re-derives the path the :class:`SortedIndexReader` opens.
    """
    return (
        memmap_dir
        / f"{binary_name}_sorted_{spec.reduction.filename_tag()}"
        f"_d{spec.depth:03d}.idx"
    )


def discover_members(
    memmap_dirs: Sequence[Path],
    *,
    specs: Sequence[IndexSpec],
    vocab_manager: Optional[Any],
    on_missing: MissingIndexPolicy,
) -> Tuple[
    List[CollectionMember],
    Dict[IndexSpec, Dict[str, SortedIndexReader]],
    Dict[str, BinaryDataset],
]:
    """Resolve kept members + their per-spec readers + (unopened) datasets.

    For each directory, a binary EXISTS iff its matched-arm
    ``<binary>_index.bin`` is present. A binary is KEPT iff, for EVERY
    requested :class:`IndexSpec`, its ``.idx`` matching
    ``(reduction.filename_tag(), depth)`` exists (via
    :func:`discover_indices`). A binary missing ANY spec's index is
    excluded from the WHOLE collection -- membership is uniform across
    specs. Missing ``(dir, binary, spec)`` triples are handled per
    ``on_missing`` (RAISE: one :class:`ValueError` listing every triple;
    SKIP: one ERROR log per excluded binary naming exactly which specs
    were missing, then exclude). Returns the
    alphabetical-by-``qualified_name`` member list plus the
    ``{IndexSpec -> {qualified_name -> SortedIndexReader}}`` per-spec
    reader maps and the shared ``{qualified_name -> BinaryDataset}`` map
    the collection wires into its per-spec samplers + shared session
    machinery.
    """
    # Cache the per-(dir, depth) discovery so a multi-spec request over
    # the same depth scans each directory once per distinct depth.
    discovery_cache: Dict[Tuple[Path, int], Dict[str, List[LengthReduction]]]
    discovery_cache = {}

    def _indices_for(memmap_dir: Path, depth: int) -> Dict[str, List[LengthReduction]]:
        key = (memmap_dir, depth)
        cached = discovery_cache.get(key)
        if cached is None:
            cached = discover_indices(memmap_dir, depth=depth)
            discovery_cache[key] = cached
        return cached

    kept: List[Tuple[Path, str]] = []
    # ``{(dir, binary) -> [missing spec, ...]}`` -- a binary present in
    # this map is excluded; its value lists exactly which specs it lacks.
    missing: Dict[Tuple[Path, str], List[IndexSpec]] = {}
    for memmap_dir in memmap_dirs:
        memmap_dir = Path(memmap_dir)
        for binary_name in _existing_binaries(memmap_dir):
            lacked = [
                spec
                for spec in specs
                if not _has_index(
                    _indices_for(memmap_dir, spec.depth).get(binary_name),
                    spec.reduction,
                )
            ]
            if lacked:
                missing[(memmap_dir, binary_name)] = lacked
            else:
                kept.append((memmap_dir, binary_name))

    _handle_missing(missing, on_missing=on_missing)

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

    readers_by_spec: Dict[IndexSpec, Dict[str, SortedIndexReader]] = {
        spec: {} for spec in specs
    }
    datasets: Dict[str, BinaryDataset] = {}
    for member in members:
        for spec in specs:
            readers_by_spec[spec][member.qualified_name] = SortedIndexReader(
                _idx_path(member.memmap_dir, member.binary_name, spec),
                reduction=spec.reduction,
                depth=spec.depth,
            )
        datasets[member.qualified_name] = BinaryDataset(
            member.memmap_dir,
            member.binary_name,
            vocab_manager=vocab_manager,
        )
    return members, readers_by_spec, datasets


def _handle_missing(
    missing: Dict[Tuple[Path, str], List[IndexSpec]],
    *,
    on_missing: MissingIndexPolicy,
) -> None:
    """Apply ``on_missing`` to the excluded ``(dir, binary) -> [spec]`` map.

    RAISE lists every missing ``(dir, binary, spec-tag)`` triple in one
    :class:`ValueError`; SKIP emits one ERROR record per excluded binary
    naming exactly which spec(s) it lacked (the exclusion drops the
    binary from ALL specs, which is a sampling bias and must be LOUD).
    """
    if not missing:
        return
    ordered = sorted(missing.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))
    if on_missing is MissingIndexPolicy.RAISE:
        listing = ", ".join(
            f"({d}, {b}, {spec_tag(spec)})"
            for (d, b), specs in ordered
            for spec in specs
        )
        raise ValueError(
            "IndexedMemmapCollection.discover: missing sorted-index file(s) "
            f"in: {listing}",
        )
    for (d, b), specs in ordered:  # SKIP_WITH_ERROR_LOG
        tags = ", ".join(spec_tag(spec) for spec in specs)
        _LOGGER.error(
            "IndexedMemmapCollection: excluding binary %r in %s from the "
            "whole collection -- no sorted-index for spec(s) %s (this is a "
            "sampling bias)",
            b,
            d,
            tags,
        )
