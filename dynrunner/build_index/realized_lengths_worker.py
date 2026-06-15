"""Worker subprocess for the ``realized_lengths`` phase-4 task type.

Receives one task per binary. The wire's ``relative_path`` is the
binary_name (an opaque identifier; not a filesystem path); the worker
re-derives every path from its ``--output`` (the memmap directory) plus
that binary_name and calls
``tokenizer.aligned_data.realized_lengths.generate_realized_lengths``
once — emitting the four realized-length sidecars
(``_lengths.bin`` + ``_lengths_index.bin`` per arm) next to the binary's
memmap inputs.

The worker reimplements no generation logic: it parses the binary_name
off the wire and forwards a single library call. Discovery, pairing, and
the per-(section, variant) length compute all live in the
realized_lengths package.

Output-directory convention: phase 4 reads from and writes to the SAME
memmap directory (the per-binary sidecars and the realized-length
sidecars are co-located by design — the sorted-index build reads both).
Under SLURM that directory is the durable ``/app/out-network`` mount the
framework points ``--output`` at, so the worker uses ``--output`` as the
generator's ``base_path``: inputs are already staged there by the memmap
phase, and the small idempotent length sidecars are written alongside.
A worker killed mid-write leaves a partial sidecar that a re-run
regenerates wholesale (the build is cheap and deterministic).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dynamic_runner.worker import Task, WorkerOutput, run, task_function

from shared import remove_stream_handlers
from tokenizer.aligned_data.realized_lengths import generate_realized_lengths

logger = logging.getLogger(__name__)

# Module-level config populated by ``_on_args`` before the run loop;
# the ``@task_function`` handler closes over it per task. Mirrors the
# build_memmap worker's on-args contract.
_OUTPUT_DIR: Path


@task_function
def handle(task: Task) -> WorkerOutput | None:
    """Generate the realized-length sidecars for one binary.

    ``task.relative_path`` is the binary_name (opaque identifier). The
    generator reads the binary's ``_sections.bin`` / ``_index.bin`` /
    ``_data.bin`` from ``_OUTPUT_DIR`` and writes the four
    realized-length sidecars there. A missing input is a deterministic,
    permanent miss for this binary (re-running won't make it appear), so
    it is surfaced as NonRecoverable.
    """
    binary_name = task.relative_path
    logger.info("[*] realized-lengths: %s", binary_name)
    task.set_phase("realized_lengths")

    try:
        written = generate_realized_lengths(_OUTPUT_DIR, binary_name)
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
    """Hook invoked by ``run()`` before the loop starts."""
    global _OUTPUT_DIR

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

    _OUTPUT_DIR = Path(args.output).resolve()
    logger.info("[*] Memmap directory: %s", _OUTPUT_DIR)


if __name__ == "__main__":
    run(argparser=_build_argparser(), on_args=_on_args)
