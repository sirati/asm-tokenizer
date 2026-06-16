"""Staged-publish helper for worker outputs.

Single concern: own the policy "write task outputs to a scratch
directory; atomic-publish them to the durable destination only on
clean exit." Workers wrap their library calls in
``staged_publish(task, output_dir, scope)`` and write to the yielded
staging dir; on clean exit the helper enumerates the staged files
and hands them to ``task.publish_all(...)`` (Rust-side, atomic) so
they appear at ``output_dir/<rel>``. On exception, nothing reaches
the destination — the partial files stay in the staging dir for
inspection (the SLURM wrapper rm-rf's ``/app/out-tmp`` on container
exit).

Mount-aware behaviour:

* SLURM container deployment (``/app/out-tmp`` exists): stage under
  ``/app/out-tmp/<scope>/``. Publish via ``task.publish_all(*paths)``
  which routes through ``dynrunner-publish`` (Rust) — same-FS
  rename(2) where possible, cross-FS copy+fsync+rename+fsync(parent)
  +unlink(src) when staging and destination live on different
  filesystems (typical: tmpfs → NFS).
* Standalone (no ``/app/out-tmp``): stage_dir IS ``output_dir``;
  writes land at their final location directly. No atomic-publish
  guarantee — standalone users can re-run on partial-write failure.

The walk-the-stage-dir-and-enumerate step is per-path explicit
registration in the operational sense — only files the task wrote
under its scoped stage dir are surfaced to ``publish_all``. The
stage dir is not shared with debug pickles, Ghidra Project state,
or other backend scratch (those use other locations under
``/app/out-tmp`` directly).

The library functions (``run_tokenizer``, ``unify_vocab``,
``build_memmap_files``) take an ``output_dir`` and write inside it.
The staging concern is layered on top by the worker — the library
stays unaware of where ``output_dir`` actually lives.
"""

from __future__ import annotations

import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Hashable, Iterator, Mapping, TypeVar

from dynamic_runner.worker import Task

from tokenizer.progress import log_stage

logger = logging.getLogger(__name__)

_SLURM_OUT_TMP = Path("/app/out-tmp")

# Node-local scratch root for staged *inputs*, kept distinct from the
# publish staging subtree (``/app/out-tmp/<scope>/``) so input copies
# and pending output writes never share a directory. The SLURM wrapper
# rm-rf's all of ``/app/out-tmp`` on container exit; the per-scope
# cleanup here additionally frees it eagerly when each task finishes,
# bounding tmpfs (RAM) residency to the inputs in flight.
_STAGED_INPUTS_SUBDIR = "staged-inputs"

# Caller-chosen key type for ``staged_inputs``: each source path is
# requested under an arbitrary hashable key (often the source ``Path``
# itself) and the yielded mapping returns the local copy under that
# same key, so callers re-point reads without re-deriving any layout.
_K = TypeVar("_K", bound=Hashable)

# Scope sentinel for the unify_vocab phase's single-task worker. Owned
# here — next to ``published_path``, which encodes the scope → on-disk
# layout rule — so the publishing side (``unify_vocab`` worker's
# ``staged_publish(..., scope=...)``) and the reading side
# (``build_memmap`` worker resolving ``--unified-vocab`` via
# ``published_path``) share one literal. Neither worker imports the
# other; both depend only on this staging module's contract.
UNIFY_VOCAB_SCOPE = "unify_vocab"


def is_container_deployment() -> bool:
    """True when the SLURM-wrapper-provided ephemeral mount is
    present, indicating container deployment. Stages writes through
    ``/app/out-tmp`` and publishes via the framework's Rust-side
    publish API in this mode.
    """
    return _SLURM_OUT_TMP.exists()


def published_path(output_root: Path, scope: str, filename: str) -> Path:
    """Where a ``staged_publish(scope=...)`` write of ``filename`` lands
    under ``output_root``, in the current deployment mode.

    The inverse of ``staged_publish``'s layout decision: container mode
    stages under ``<scope>/`` and republishes mirroring that subdir, so
    the file appears at ``<output_root>/<scope>/<filename>``; standalone
    mode writes to ``output_root`` directly, so it appears at
    ``<output_root>/<filename>`` (scope ignored). A reader resolving a
    scoped-publish artifact joins through here to stay mode-symmetric
    with the writer without knowing which mode produced the file.

    ``filename`` may be absolute (a standalone caller handing an
    explicit location); pathlib's ``/`` keeps an absolute right-hand
    side, so it passes through unchanged in both branches.
    """
    if is_container_deployment():
        return output_root / scope / filename
    return output_root / filename


@contextmanager
def staged_publish(
    task: Task, final_dir: Path, scope: str
) -> Iterator[Path]:
    """Yield a staging directory; atomic-publish its contents to
    ``final_dir`` on clean exit, leave them in place on exception.

    ``scope`` must be unique per concurrent task in the same worker —
    use the binary identifier, group key, or a literal sentinel for
    single-task workers. In container mode this names a subdir under
    ``/app/out-tmp``; in standalone mode it's unused (writes go to
    ``final_dir`` directly).

    Files are republished mirroring the relative subdir layout under
    the staging dir, so callers can write subdirs (vocab_unifier's
    per-CSV mapping files preserve their source layout under
    ``<final_dir>/<rel>/<binary>.mapping.b64c``). Mirroring is
    handled by the framework's publish API in container mode and is
    implicit in standalone mode (since stage_dir == final_dir).
    """
    final_dir.mkdir(parents=True, exist_ok=True)

    if is_container_deployment():
        staging_dir = _SLURM_OUT_TMP / scope
        staging_dir.mkdir(parents=True, exist_ok=True)
        yield staging_dir
        # Walk only this task's scoped stage dir (no shared debug
        # pickles, no Ghidra Project state) and surface each written
        # file to the per-path publish API. The Rust-side publish_all
        # handles intra-/cross-FS atomicity.
        outputs = [p for p in sorted(staging_dir.rglob("*")) if p.is_file()]
        if outputs:
            with log_stage(
                logger, f"atomic publish of {len(outputs)} staged files ({scope})"
            ):
                task.publish_all(*outputs)
    else:
        # Standalone: writes land at their final destination directly.
        # Atomic-publish only applies to container mode where the
        # wrapper trap protects /app/out-tmp from leaking partials.
        yield final_dir


def _local_input_path(scratch_root: Path, source: Path) -> Path:
    """Where ``source`` is copied to under ``scratch_root``.

    Mirrors the source's absolute path tail (leading ``/`` stripped)
    under the scratch root, so two distinct absolute sources never
    collide on basename — symmetric to how ``staged_publish`` mirrors a
    relative subdir layout into the destination. ``Path.parts[0]`` is
    the anchor (``"/"`` for an absolute path); dropping it yields the
    root-relative components to rejoin under the scratch dir.
    """
    return scratch_root.joinpath(*source.parts[1:])


@contextmanager
def staged_inputs(
    sources: Mapping[_K, Path], scope: str
) -> Iterator[dict[_K, Path]]:
    """Yield node-local copies of ``sources``; delete them on exit.

    Single concern: own the policy "copy the NFS inputs a task needs to
    node-local scratch, read against the copies, drop them when done" —
    the input-side mirror of ``staged_publish``. Reading directly off
    the corpus/output NFS bind-mounts (random-access ``np.memmap`` of
    ``_data.bin`` especially) page-fault-storms the shared filesystem;
    staging confines each task's reads to local scratch.

    ``sources`` maps caller-chosen hashable keys to NFS source paths.
    The yielded dict returns each local copy under the **same key**, so
    a caller re-points its reads by key lookup without knowing where the
    scratch lives or how names are de-collided.

    Mount-aware behaviour:

    * SLURM container deployment (``/app/out-tmp`` exists): each source
      is copied (``shutil.copy2``, preserving mtime) under
      ``/app/out-tmp/staged-inputs/<scope>/`` at a path mirroring the
      source's absolute layout, so distinct sources never collide. On
      exit (clean OR exception) the whole ``<scope>`` subtree is
      removed, eagerly freeing tmpfs ahead of the wrapper's exit-time
      rm-rf.
    * Standalone (no ``/app/out-tmp``): a no-op that yields the original
      ``sources`` mapping unchanged — no copying, no scratch, so
      standalone ``--task build-memmap`` etc. read inputs in place.

    Node-local target rationale: ``/app/out-tmp`` is tmpfs (RAM). The
    per-task inputs are modest (CSVs up to ~25 MB, mappings tiny,
    ``_data.bin`` ~40 MB avg) and the per-scope cleanup bounds residency
    to the inputs of tasks in flight — well within the host's 118 GB
    across 56 workers. If the wrapper later exposes a node-local *disk*
    scratch mount better suited than tmpfs, redirect the scratch root
    here; callers are unaffected (they only see the yielded keys).

    ``scope`` must be unique per concurrent task in the same worker
    (use the binary identifier or a single-task sentinel), mirroring
    ``staged_publish``'s scope contract.
    """
    if not is_container_deployment():
        # Standalone: read inputs in place; nothing to stage or clean.
        yield dict(sources)
        return

    scratch_root = _SLURM_OUT_TMP / _STAGED_INPUTS_SUBDIR / scope
    local_by_key: dict[_K, Path] = {}
    try:
        with log_stage(
            logger, f"stage {len(sources)} inputs to node-local scratch ({scope})"
        ):
            for key, source in sources.items():
                local = _local_input_path(scratch_root, source)
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, local)
                local_by_key[key] = local
        yield local_by_key
    finally:
        # Free the scratch on clean exit AND on exception — bounding
        # tmpfs residency to tasks actually in flight. Tolerate a
        # partially-populated tree (a copy failed mid-loop).
        shutil.rmtree(scratch_root, ignore_errors=True)
