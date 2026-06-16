"""Output-set-complete ``--skip-existing`` filtering for per-binary tasks.

Single concern: own the policy "skip a task only when its COMPLETE
per-binary output set is already present at the output root." A task
that publishes its outputs set-atomically (one ``publish_all``
transaction) either lands the whole set or none of it, so a complete
set on disk is a sound skip signal; conversely a PARTIAL set (any
required member missing) must NOT be skipped — it has to re-publish.
Gating on a single coarse marker would wrongly skip a partial set left
by a kill mid-publish (the corruption + ``--skip-existing`` poisoning
the set-atomic publish path was added to prevent).

The boundary each caller crosses:

  *Given the dispatch's ``args`` (carrying ``skip_existing`` +
  ``resolved_output_root``) and a per-item callback naming that item's
  required output basenames, drop the items whose required set is fully
  present.* The caller owns WHAT files complete an item (its on-disk
  artifact grammar); this module owns the walk + the all-present
  predicate, so no task duplicates the enumerate-existing-outputs walk.

The walk enters every subfolder, so the nested container layout
(per-binary subdirs) and the flat layout both flatten to one basename
set; per-binary basenames are ``<binary>_*`` prefixed, so no two
binaries' required files ever collide on basename.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

from dynamic_runner import _native

from dynrunner.binary_selection import TaskInfo


_logger = logging.getLogger(__name__)


class _OutputFilenameCollector:
    """Visitor that records every file's basename. Used by the
    task-side `--skip-existing` filter to enumerate already-completed
    outputs at `args.resolved_output_root` (gateway-side in SLURM
    pre-staged mode, local otherwise)."""

    def __init__(self) -> None:
        self.filenames: set[str] = set()

    def visit(
        self,
        parent_payload: object,
        subfolders: list,
        files: list,
    ) -> None:
        for folder in subfolders:
            folder.enter(True)
        for f in files:
            self.filenames.add(f.name)


def existing_output_filenames(output_root: str) -> set[str]:
    """Walk `output_root` via `_native.find_items` and return the set of
    file basenames present. A non-existent or unreadable `output_root`
    yields an empty set rather than failing the dispatch — first-run
    deployments don't have the directory yet.
    """
    collector = _OutputFilenameCollector()
    try:
        _native.find_items(collector, output_root)
    except OSError:
        return set()
    return collector.filenames


def filter_items_with_complete_outputs(
    items: list[TaskInfo],
    *,
    skip_existing: bool,
    output_root: str | None,
    required_outputs: Callable[[TaskInfo], Iterable[str]],
    label: str = "skip-existing",
) -> list[TaskInfo]:
    """Drop items whose COMPLETE required output set is already present.

    ``required_outputs(item)`` yields the basenames that must ALL exist
    for ``item`` to count as done; an item is skipped only when every
    one is present in the ``output_root`` walk. A partial set (any
    member absent) is retained so it re-runs and re-publishes its whole
    set atomically. When ``skip_existing`` is false or ``output_root`` is
    unset (first run, no directory) the items pass through unchanged.
    """
    if not skip_existing or not output_root:
        return items
    completed = existing_output_filenames(output_root)
    before = len(items)
    kept = [
        item
        for item in items
        if not all(name in completed for name in required_outputs(item))
    ]
    _logger.info(
        "%s: %d candidates → %d remaining "
        "(%d skipped via %d existing outputs at %s)",
        label,
        before,
        len(kept),
        before - len(kept),
        len(completed),
        output_root,
    )
    return kept
