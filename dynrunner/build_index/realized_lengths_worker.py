"""Worker subprocess for the ``realized_lengths`` phase-4 task type.

Receives one task per binary. The wire's ``relative_path`` is the
binary_name (an opaque identifier; not a filesystem path) and the
payload carries that binary's ``memmap_dir`` -- the directory its memmap
sidecars actually live in, resolved by discovery (the scanned dir in the
flat layout, a per-binary subdir in the nested container layout). The
worker reads that dir off the wire, stages the binary's memmap sidecar
INPUTS node-local, calls
``tokenizer.aligned_data.realized_lengths.generate_realized_lengths``
once against those local copies — emitting the four realized-length
sidecars (``_lengths.bin`` + ``_lengths_index.bin`` per arm) — and
atomically publishes those output sidecars back to the NFS
``memmap_dir``.

The worker reimplements no generation logic and owns no layout
knowledge: it reads the binary_name + memmap_dir off the wire and
forwards a single library call. Discovery (and the flat-vs-nested layout
decision) lives in ``binary_discovery``; the per-(section, variant)
length compute lives in the realized_lengths package.

NFS-staging convention: the generator memmaps the arm ``_data.bin``
random-access and re-reads ``_sections.bin`` / ``_index.bin`` — reading
those directly off the shared corpus/output NFS page-fault-storms the
filesystem (the core DDOS). So the worker stages the binary's existing
inputs to node-local scratch via ``staged_inputs``, runs the generator
against the local copies (which reads AND writes there), then publishes
ONLY the generator's returned output sidecars back to the NFS
``memmap_dir`` via ``task.publish(src, dst=...)`` (native copy→fsync→
atomic-rename, so a mid-publish kill leaves a ``.publish-tmp`` sibling,
never a partial sidecar). The published location is byte-for-byte the
old in-place location, so the co-located sorted-index build still reads
``_lengths.bin`` from the same NFS dir.

Standalone (no ``/app/out-tmp``): ``staged_inputs`` is a no-op yielding
the originals, so the local dir IS the NFS ``memmap_dir`` — the generator
reads+writes in place and the publish step is skipped (the outputs are
already at their final location).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dynamic_runner.worker import Task, WorkerOutput, run, task_function

from shared import remove_stream_handlers
from tokenizer.aligned_data.realized_lengths import generate_realized_lengths
from tokenizer.output_staging import staged_inputs

from dynrunner.build_index._payload import memmap_dir_from_task

logger = logging.getLogger(__name__)

#: The per-binary memmap sidecar files the generator reads from the
#: memmap dir, named by their ``<binary_name>`` suffix. Both arms' catalog
#: reads touch ``_index.bin`` (matched locator + unmatched region start)
#: and ``_sections.bin`` (columnar + structural walk); the per-arm dedup
#: compute memmaps ``_data.bin`` (matched) / ``_unmatched_data.bin``
#: (unmatched). Each is conditionally present — the generator gates the
#: catalog files on ``.exists()`` and only opens a ``_data.bin`` for a
#: non-empty arm — so the worker stages only those that exist on NFS (a
#: missing optional input must not become a spurious stage failure).
_INPUT_SUFFIXES = (
    "_index.bin",
    "_sections.bin",
    "_data.bin",
    "_unmatched_data.bin",
)


def _existing_inputs(memmap_dir: Path, binary_name: str) -> dict[Path, Path]:
    """NFS source paths of the binary's present memmap inputs, keyed by
    themselves (the ``staged_inputs`` key convention).

    The ``.exists()`` gate matches the generator's own input gating, so
    staging never widens a permanent miss into a copy error: a truly
    absent input is simply not staged and the generator handles its
    absence exactly as it would reading NFS directly.
    """
    inputs: dict[Path, Path] = {}
    for suffix in _INPUT_SUFFIXES:
        src = memmap_dir / f"{binary_name}{suffix}"
        if src.exists():
            inputs[src] = src
    return inputs


@task_function
def handle(task: Task) -> WorkerOutput | None:
    """Generate the realized-length sidecars for one binary.

    ``task.relative_path`` is the binary_name (opaque identifier); the
    payload carries the binary's ``memmap_dir``. The worker stages that
    binary's existing memmap inputs node-local, runs the generator
    against the local copies, and publishes the four returned
    realized-length sidecars back to the NFS ``memmap_dir``. A missing
    input is a deterministic, permanent miss for this binary (re-running
    won't make it appear), so it is surfaced as NonRecoverable.
    """
    binary_name = task.relative_path
    memmap_dir = memmap_dir_from_task(task)
    logger.info("[*] realized-lengths: %s (%s)", binary_name, memmap_dir)
    task.set_phase("realized_lengths")

    try:
        written = _generate_and_publish(task, memmap_dir, binary_name)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as e:
        from dynamic_runner.worker import NonRecoverableError

        raise NonRecoverableError(f"{type(e).__name__}: {e}") from e

    for arm_name, paths in written.items():
        for path in paths:
            logger.info("    %s -> %s", arm_name, path)
    return None


def _generate_and_publish(
    task: Task, memmap_dir: Path, binary_name: str
) -> dict[str, list[Path]]:
    """Stage inputs, run the generator locally, publish outputs to NFS.

    ``staged_inputs`` confines the generator's random-access reads to
    node-local scratch; the local memmap dir is the parent of any staged
    copy (the helper mirrors each source's absolute tail, so all of the
    binary's sidecars land in one dir). Standalone (no scratch) yields
    the originals, so the local dir IS ``memmap_dir`` and the generator
    runs fully in place.

    Only the generator's RETURNED output paths are published — atomically,
    with an explicit destination at ``memmap_dir/<filename>`` so the
    sidecars land at the unchanged NFS location regardless of the staged
    layout. When the local dir already is ``memmap_dir`` (standalone, or
    nothing staged) the outputs are already final, so publishing is
    skipped: ``task.publish`` would otherwise self-copy through a publish
    root that need not exist.
    """
    inputs = _existing_inputs(memmap_dir, binary_name)
    with staged_inputs(inputs, scope=f"realized_lengths/{binary_name}") as local:
        local_dir = next(iter(local.values())).parent if local else memmap_dir
        written = generate_realized_lengths(local_dir, binary_name)
        if local_dir != memmap_dir:
            for paths in written.values():
                for path in paths:
                    task.publish(path, dst=memmap_dir / path.name)
    return written


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realized-length sidecar worker (per-binary).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dynamic_queue",
        type=int,
        metavar="SOCKET_FD",
        help="Receive tasks via socket file descriptor (anonymous socket)",
    )
    group.add_argument(
        "--socket-path",
        type=str,
        metavar="SOCKET_PATH",
        help="Receive tasks via named Unix socket at this path",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help=(
            "Source directory. Informational at this layer — the "
            "generator reads/writes the --output memmap directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Memmap directory: the binary's memmap sidecars are read "
            "from here and the realized-length sidecars are written here."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file instead of stdout/err",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "Accepted for framework compatibility; per-binary skip is "
            "governed by the starting instance's discover_items filter."
        ),
    )
    return parser


def _on_args(args: argparse.Namespace) -> None:
    """Hook invoked by ``run()`` before the loop starts.

    The per-binary memmap dir arrives on each task's payload, so there is
    no run-level memmap directory to bind here; ``--output`` is accepted
    for framework compatibility only.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if args.log_file:
        if args.dynamic_queue:
            remove_stream_handlers(root)
        log_file = Path(args.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch()
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                "%(levelname)s | %(asctime)s,%(msecs)03d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s | %(asctime)s,%(msecs)03d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


if __name__ == "__main__":
    run(argparser=_build_argparser(), on_args=_on_args)
