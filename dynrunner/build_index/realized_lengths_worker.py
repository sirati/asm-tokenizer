"""Worker subprocess for the ``realized_lengths`` phase-4 task type.

Receives one task per binary. The wire's ``relative_path`` is the
binary_name (an opaque identifier; not a filesystem path) and the
payload carries that binary's ``memmap_dir`` -- the directory its memmap
sidecars actually live in, resolved by discovery (the scanned dir in the
flat layout, a per-binary subdir in the nested container layout). The
worker reads that dir off the wire and calls
``tokenizer.aligned_data.realized_lengths.generate_realized_lengths``
once — emitting the four realized-length sidecars
(``_lengths.bin`` + ``_lengths_index.bin`` per arm) next to the binary's
memmap inputs.

The worker reimplements no generation logic and owns no layout
knowledge: it reads the binary_name + memmap_dir off the wire and
forwards a single library call. Discovery (and the flat-vs-nested layout
decision) lives in ``binary_discovery``; the per-(section, variant)
length compute lives in the realized_lengths package.

Output-directory convention: phase 4 reads from and writes to the SAME
memmap directory (the per-binary sidecars and the realized-length
sidecars are co-located by design — the sorted-index build reads both).
The generator's ``base_path`` is the payload ``memmap_dir`` (not
``--output``), so a nested per-binary subdir resolves correctly; in the
flat layout that dir equals the ``--output`` root. A worker killed
mid-write leaves a partial sidecar that a re-run regenerates wholesale
(the build is cheap and deterministic).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dynamic_runner.worker import Task, WorkerOutput, run, task_function

from shared import remove_stream_handlers
from tokenizer.aligned_data.realized_lengths import generate_realized_lengths

from dynrunner.build_index._payload import memmap_dir_from_task

logger = logging.getLogger(__name__)


@task_function
def handle(task: Task) -> WorkerOutput | None:
    """Generate the realized-length sidecars for one binary.

    ``task.relative_path`` is the binary_name (opaque identifier); the
    payload carries the binary's ``memmap_dir``. The generator reads the
    binary's ``_sections.bin`` / ``_index.bin`` / ``_data.bin`` from that
    dir and writes the four realized-length sidecars there. A missing
    input is a deterministic, permanent miss for this binary (re-running
    won't make it appear), so it is surfaced as NonRecoverable.
    """
    binary_name = task.relative_path
    memmap_dir = memmap_dir_from_task(task)
    logger.info("[*] realized-lengths: %s (%s)", binary_name, memmap_dir)
    task.set_phase("realized_lengths")

    try:
        written = generate_realized_lengths(memmap_dir, binary_name)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as e:
        from dynamic_runner.worker import NonRecoverableError

        raise NonRecoverableError(f"{type(e).__name__}: {e}") from e

    for arm_name, paths in written.items():
        for path in paths:
            logger.info("    %s -> %s", arm_name, path)
    return None


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
