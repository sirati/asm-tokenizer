"""Worker subprocess for the ``sorted_index`` phase-4 task type.

Receives one task per binary. The wire's ``relative_path`` is the
binary_name (an opaque identifier; not a filesystem path) and the
payload carries that binary's ``memmap_dir`` -- the directory its memmap
+ realized-length sidecars actually live in, resolved by discovery (the
scanned dir in the flat layout, a per-binary subdir in the nested
container layout). The worker reads that dir off the wire and calls
``tokenizer.aligned_data.sorted_index.write_sorted_index_files`` once,
emitting one ``<binary>_sorted_<mode>_d<depth>.idx`` per requested
(reduction, depth) pair next to those sidecars.

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

Output-directory convention: the ``.idx`` files land at
``<memmap_dir>/<binary>_sorted_<mode>_d<depth>.idx`` -- the conventional
sidecar-adjacent placement consumers read. That ``memmap_dir`` is a
per-binary subdir under ``/app/out-network`` in the nested container
layout, or equals ``--output`` in the flat layout. ``--output`` is
accepted for framework compatibility only.

NFS-staging: the build's input footprint
(:func:`tokenizer.aligned_data.sorted_index.sorted_index_input_paths`:
the catalog + locator the pre-pass memmaps and the matched-arm realized-
length sidecar pair) is page-fault-heavy ``np.memmap`` random access. The
worker copies that footprint to node-local scratch via
:func:`tokenizer.output_staging.staged_inputs`, runs the builder against
the local copies (read AND ``.idx`` write), and set-atomically publishes
the binary's FULL ``.idx`` set back to its canonical NFS location in ONE
``publish_all`` transaction -- confining the storm to local tmpfs and
guaranteeing a kill mid-publish never leaves a PARTIAL ``.idx`` set (which
would poison ``--skip-existing``). ``staged_inputs`` is a no-op standalone
(no ``/app/out-tmp``), so the build reads/writes ``memmap_dir`` in place
and the publish self-skips (src==dst pairs dropped → empty no-op); the
on-disk result is identical in both modes.
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
    sorted_index_input_paths,
    write_sorted_index_files,
)
from tokenizer.output_staging import staged_inputs

from dynrunner.build_index._payload import memmap_dir_from_task

logger = logging.getLogger(__name__)

# Module-level config populated by ``_on_args`` before the run loop;
# the ``@task_function`` handler closes over it per task. The per-binary
# memmap dir is NOT here -- it arrives on each task's payload.
_REDUCTIONS: list
_DEPTHS: list
_GATE: VariantGate
_DUPLICATE_HANDLING: object


def _sorted_index_scope(binary_name: str) -> str:
    """The ``staged_inputs`` scope for one binary's sorted-index build.

    Unique per concurrent task in this worker (the binary identifier),
    mirroring ``staged_inputs``'s scope contract; namespaced under
    ``sorted_index/`` so this phase-4 worker's scratch subtree never
    aliases another worker's.
    """
    return f"sorted_index/{binary_name}"


@task_function
def handle(task: Task) -> WorkerOutput | None:
    """Build the sorted-index ``.idx`` files for one binary.

    ``task.relative_path`` is the binary_name (opaque identifier); the
    payload carries the binary's ``memmap_dir`` -- the NFS directory its
    memmap + realized-length sidecars live in. To keep the build off the
    shared filesystem (the NFS DDOS: ``np.memmap`` random-access over the
    sidecars page-fault-storms the corpus mount), the worker stages the
    build's input footprint to node-local scratch via ``staged_inputs``,
    runs the builder against the local copies (reading AND writing the
    ``.idx`` there), and atomic-publishes each produced ``.idx`` back to
    its canonical NFS location ``<memmap_dir>/<binary>_sorted_<mode>_
    d<depth>.idx`` -- the path consumers read, unchanged.

    A missing input is a deterministic permanent miss → NonRecoverable.
    """
    binary_name = task.relative_path
    memmap_dir = memmap_dir_from_task(task)
    logger.info("[*] sorted-index: %s (%s)", binary_name, memmap_dir)
    task.set_phase("sorted_index")

    # Stage only the inputs that exist: ``staged_inputs`` (``shutil.copy2``)
    # would raise on an absent source, while the builder legitimately
    # tolerates absence (a binary with no matched arm lacks the locator /
    # catalog and yields the canonical empty index). The realized-length
    # sidecar is a hard precondition the dependency edge guarantees, but a
    # genuinely missing input is reproduced locally exactly as on NFS --
    # the builder's generator-pointing FileNotFoundError still fires below.
    present_inputs = {
        p: p
        for p in sorted_index_input_paths(memmap_dir, binary_name)
        if p.exists()
    }

    try:
        with staged_inputs(
            present_inputs, scope=_sorted_index_scope(binary_name)
        ) as local:
            # The staged copies all mirror the one NFS ``memmap_dir`` under
            # the scratch root, so they share a single parent -- the local
            # memmap dir the builder reads from and writes the ``.idx`` into.
            # With nothing staged (no matched arm) the build runs against
            # the original NFS dir, reproducing the same empty result with
            # no reads to storm. Standalone mode: ``staged_inputs`` is a
            # no-op, so ``local`` is the originals and the local dir IS
            # ``memmap_dir`` -- the publish below is then a self-skip.
            local_memmap_dir = (
                next(iter(local.values())).parent if local else memmap_dir
            )
            written = write_sorted_index_files(
                local_memmap_dir,
                binary_name,
                reductions=_REDUCTIONS,
                depths=_DEPTHS,
                gate=_GATE,
                duplicate_handling=_DUPLICATE_HANDLING,
                output_dir=local_memmap_dir,
            )
            # Publish the binary's FULL ``.idx`` set to its canonical NFS
            # location as ONE set-atomic ``publish_all`` transaction, each
            # pair carrying an EXPLICIT dst (the deleted auto-mirror would
            # route a staged-subtree source to a polluted ``dst_root/
            # staged-inputs/...`` path). Publishing the whole per-binary
            # set in one call means a kill mid-publish can never leave a
            # PARTIAL ``.idx`` set on NFS (which would corrupt the per-
            # binary index and poison ``--skip-existing``); the native
            # batch stages all then commits the renames back-to-back under
            # a signal mask. The src==dst pairs are dropped (standalone:
            # local dir == ``memmap_dir``), so an empty list republishes
            # nothing onto itself.
            pairs = [
                (idx_path, memmap_dir / idx_path.name)
                for idx_path in written.values()
                if idx_path.resolve() != (memmap_dir / idx_path.name).resolve()
            ]
            task.publish_all(pairs)
            for spec, idx_path in written.items():
                logger.info("    %s -> %s", spec, memmap_dir / idx_path.name)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as e:
        from dynamic_runner.worker import NonRecoverableError

        raise NonRecoverableError(f"{type(e).__name__}: {e}") from e

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
    global _REDUCTIONS, _DEPTHS, _GATE, _DUPLICATE_HANDLING

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

    _REDUCTIONS = list(args.mode)
    _DEPTHS = list(args.depth)
    _GATE = VariantGate(
        min_variants=args.min_variants,
        min_variants_unique=args.min_variants_unique,
    )
    _DUPLICATE_HANDLING = (
        DEDUP_BY_DATA_POINTER if args.adjust_for_duplicates else PLAIN
    )

    logger.info("[*] Reductions: %s  Depths: %s", _REDUCTIONS, _DEPTHS)


if __name__ == "__main__":
    run(argparser=_build_argparser(), on_args=_on_args)
