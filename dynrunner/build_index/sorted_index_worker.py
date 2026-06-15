"""Worker subprocess for the ``sorted_index`` phase-4 task type.

Receives one task per binary. The wire's ``relative_path`` is the
binary_name (an opaque identifier; not a filesystem path); the worker
re-derives every path from its ``--output`` (the memmap directory) plus
that binary_name and calls
``tokenizer.aligned_data.sorted_index.write_sorted_index_files`` once,
emitting one ``<binary>_sorted_<mode>_d<depth>.idx`` per requested
(reduction, depth) pair.

The per-run sorted-index configuration (``--mode`` / ``--depth`` + the
gating / duplicate knobs) arrives on argv — it is identical for every
binary in the dispatch, so it rides the worker command line, not the
per-task payload. Text→typed conversion happens here exactly as in the
standalone ``sorted_index`` CLI (``parse_reduction`` at the argparse
boundary, ``VariantGate`` validation) so the build receives fully-typed
inputs. The worker reimplements no build logic — discovery, the catalog
pre-pass, the walk-free length compute, and wire encoding all live in
the sorted_index package.

Dependency: each sorted-index task DEPENDS on its same-binary
realized-length task (wired by ``BuildIndexTask.items_for_binary`` via
``TaskInfo.task_depends_on``), so the framework holds it until the
realized-length sidecars this build consumes have been produced.

Output-directory convention: the build reads the binary's memmap +
realized-length sidecars from the ``--output`` memmap directory and
writes the ``.idx`` files there (``output_dir`` defaults to the input
dir — the conventional sidecar-adjacent placement). Under SLURM that is
the durable ``/app/out-network`` mount.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dynamic_runner.worker import Task, WorkerOutput, run, task_function

from shared import remove_stream_handlers
from tokenizer.aligned_data.sorted_index import (
    DEDUP_BY_DATA_POINTER,
    PLAIN,
    VariantGate,
    parse_reduction,
    write_sorted_index_files,
)

logger = logging.getLogger(__name__)

# Module-level config populated by ``_on_args`` before the run loop;
# the ``@task_function`` handler closes over it per task.
_OUTPUT_DIR: Path
_REDUCTIONS: list
_DEPTHS: list
_GATE: VariantGate
_DUPLICATE_HANDLING: object


@task_function
def handle(task: Task) -> WorkerOutput | None:
    """Build the sorted-index ``.idx`` files for one binary.

    ``task.relative_path`` is the binary_name (opaque identifier). The
    builder reads the binary's memmap sidecars (and the realized-length
    sidecars produced by the dependency task) from ``_OUTPUT_DIR`` and
    writes one ``.idx`` per (reduction, depth). A missing input is a
    deterministic permanent miss → NonRecoverable.
    """
    binary_name = task.relative_path
    logger.info("[*] sorted-index: %s", binary_name)
    task.set_phase("sorted_index")

    try:
        written = write_sorted_index_files(
            _OUTPUT_DIR,
            binary_name,
            reductions=_REDUCTIONS,
            depths=_DEPTHS,
            gate=_GATE,
            duplicate_handling=_DUPLICATE_HANDLING,
            output_dir=_OUTPUT_DIR,
        )
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as e:
        from dynamic_runner.worker import NonRecoverableError

        raise NonRecoverableError(f"{type(e).__name__}: {e}") from e

    for spec, path in written.items():
        logger.info("    %s -> %s", spec, path)
    return None


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sorted-index builder worker (per-binary).",
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
            "builder reads/writes the --output memmap directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Memmap directory: the binary's memmap + realized-length "
            "sidecars are read from here and the .idx files written here."
        ),
    )
    # Per-run sorted-index config. Same text→typed conversion the
    # standalone CLI applies, so the builder receives typed inputs.
    parser.add_argument(
        "--mode",
        action="append",
        type=parse_reduction,
        required=True,
        metavar="MODE",
        help="Reduction mode (repeatable): 'max' or 'p<NN>' (1<=NN<=99).",
    )
    parser.add_argument(
        "--depth",
        action="append",
        type=int,
        required=True,
        dest="depth",
        metavar="DEPTH",
        help="Splice depth (repeatable).",
    )
    parser.add_argument(
        "--min-variants",
        type=int,
        default=0,
        metavar="N",
        help="Top-level minimum-variant emission gate.",
    )
    parser.add_argument(
        "--min-variants-unique",
        type=int,
        default=0,
        metavar="M",
        help="Unique-variant minimum (composes with --min-variants).",
    )
    parser.add_argument(
        "--adjust-for-duplicates",
        action="store_true",
        help="Collapse data-pointer-duplicate variants before reduction.",
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

    Validates the gating combination here (``VariantGate.__post_init__``)
    so a bad ``--min-variants*`` combo fails loud at worker startup
    rather than per-task. Mirrors the standalone CLI's validation point.
    """
    global _OUTPUT_DIR, _REDUCTIONS, _DEPTHS, _GATE, _DUPLICATE_HANDLING

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
    _REDUCTIONS = list(args.mode)
    _DEPTHS = list(args.depth)
    _GATE = VariantGate(
        min_variants=args.min_variants,
        min_variants_unique=args.min_variants_unique,
    )
    _DUPLICATE_HANDLING = (
        DEDUP_BY_DATA_POINTER if args.adjust_for_duplicates else PLAIN
    )

    logger.info("[*] Memmap directory: %s", _OUTPUT_DIR)
    logger.info("[*] Reductions: %s  Depths: %s", _REDUCTIONS, _DEPTHS)


if __name__ == "__main__":
    run(argparser=_build_argparser(), on_args=_on_args)
