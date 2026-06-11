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
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from dynamic_runner.worker import Task

from tokenizer.progress import log_stage

logger = logging.getLogger(__name__)

_SLURM_OUT_TMP = Path("/app/out-tmp")

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
