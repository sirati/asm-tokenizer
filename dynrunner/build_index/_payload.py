"""Shared wire-payload decode for the phase-4 build-index workers.

Single concern: turn one build-index task's wire payload into the
binary's ``memmap_dir`` Path -- the directory its memmap sidecars live
in, resolved by discovery (flat scanned dir OR a per-binary nested
subdir). Both phase-4 workers (realized_lengths + sorted_index) need the
identical decode, so it lives here once rather than duplicated on each
wire end. The ``memmap_dir`` is the generator's ``base_path``; ``--output``
is no longer the per-binary read root (it stays the framework dir, which
equals the scanned dir only in the flat layout).

The producing end (``BuildIndexTask.items_for_binary``) owns the same
``PAYLOAD_MEMMAP_DIR`` literal; importing it here keeps the key defined
once across both wire ends.
"""

from __future__ import annotations

import json
from pathlib import Path

from dynamic_runner.worker import NonRecoverableError, Task

from dynrunner.build_index.build_index_task import PAYLOAD_MEMMAP_DIR


def memmap_dir_from_task(task: Task) -> Path:
    """Return the binary's ``memmap_dir`` from a phase-4 task payload.

    A missing payload / missing ``memmap_dir`` key is a deterministic
    dispatch error (``discover_items`` must always emit it), so it is
    surfaced as ``NonRecoverableError`` rather than burning the retry
    pass -- mirroring the build_memmap worker's empty-payload contract.
    """
    payload_str = task.payload_str
    if not payload_str:
        raise NonRecoverableError(
            f"build-index worker received task without payload for "
            f"binary_name={task.relative_path!r}; discover_items must "
            f"emit non-empty TaskInfo.payload carrying {PAYLOAD_MEMMAP_DIR!r}."
        )
    data = json.loads(payload_str)
    memmap_dir = data.get(PAYLOAD_MEMMAP_DIR)
    if not memmap_dir:
        raise NonRecoverableError(
            f"build-index task payload for binary_name="
            f"{task.relative_path!r} is missing {PAYLOAD_MEMMAP_DIR!r}; "
            f"discover_items must populate it."
        )
    return Path(memmap_dir)
