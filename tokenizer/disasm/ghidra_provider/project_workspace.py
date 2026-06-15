"""Ephemeral per-task Ghidra project-directory lifecycle.

Ghidra writes a project (``.gpr`` file + ``.rep`` lock directory) into its
``project_location`` during import and auto-analysis. This module owns that
directory's whole life so the provider doesn't have to:

1. **Unique per provider.** Every provider instance gets a fresh
   ``mkdtemp`` directory, so two binaries that happen to share a parent
   directory name never collide on the ``<binary-name>_ghidra`` project name
   pyghidra derives (the historic ``LockException`` source), and a worker
   reused across tasks never reopens a stale project.
2. **Removed when released.** The directory is deleted once the task
   releases the provider, so long-lived workers don't balloon ``/tmp``
   across thousands of binaries.

The writable root is the container's ephemeral ``/app/out-tmp`` when present
(SLURM packaging), else ``/tmp``. The source tree itself is a read-only
bind-mount under ``--source-already-staged``, so the project must never land
beside the binary — which is pyghidra's default.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_WORKSPACE_PARENT = "ghidra-projects"
_WORKSPACE_PREFIX = "proj-"


def _ephemeral_root() -> Path:
    """Writable root for project directories: ``/app/out-tmp`` when the
    container provides it, else ``/tmp``."""
    out_tmp = Path("/app/out-tmp")
    return out_tmp if out_tmp.is_dir() else Path("/tmp")


def create_project_workspace() -> Path:
    """Create and return a fresh, unique ephemeral Ghidra project directory."""
    parent = _ephemeral_root() / _WORKSPACE_PARENT
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=_WORKSPACE_PREFIX, dir=parent))


def remove_project_workspace(workspace: Path) -> None:
    """Remove a workspace previously returned by ``create_project_workspace``.

    Hard-guarded: only a directory that lives directly under a
    ``ghidra-projects`` parent and carries the workspace prefix is removed,
    so a misconfigured caller can never ``rmtree`` an unrelated path.
    Best-effort — a removal failure is logged, never raised, so it cannot
    mask the task's own outcome.
    """
    workspace = Path(workspace)
    if (
        workspace.parent.name != _WORKSPACE_PARENT
        or not workspace.name.startswith(_WORKSPACE_PREFIX)
    ):
        logger.error("refusing to remove non-workspace path %s", workspace)
        return
    shutil.rmtree(workspace, ignore_errors=True)
