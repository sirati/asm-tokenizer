"""Worker subprocess for `dynrunner.build_memmap`.

Receives one ``ProcessBinaryCommand`` per group. The wire's
``relative_path`` is the binary_name (an opaque identifier; not a
filesystem path) and ``payload`` is a JSON string containing the
per-version pairing data the starting instance prepared. The worker:

1. Parses ``command.payload`` into a ``versions`` list of
   ``{csv_path, mapping_path, arch, compiler, compilerversion, opt}``.
2. Reconstructs ``BinaryVersionInfo`` instances (per-variant pkg +
   ``extra_metadata`` recovered via ``VariantInfo.from_csv`` when a
   ``_meta.json`` sidecar was declared by the planner).
3. Calls ``tokenizer.memmap_builder.builder.build_memmap_files`` once
   for the group and replies ``done``.

The worker scans nothing, reads no manifest from disk, and does no
pairing or grouping — those concerns live entirely in
``MemmapBuilderTask.discover_items`` on the starting instance.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dynamic_runner.worker import Task, WorkerOutput, run, task_function

from shared import increase_csv_field_size_limit, remove_stream_handlers
from tokenizer.memmap_builder.builder import BinaryVersionInfo, build_memmap_files
from tokenizer.output_staging import UNIFY_VOCAB_SCOPE, published_path, staged_publish
from tokenizer.variant_info import VariantInfo

logger = logging.getLogger(__name__)

# Module-level config populated by `_on_args` before the run-loop
# starts; the `@task_function` handler reads it for each task. This
# mirrors the runtime's contract that the handler closes over the
# parsed CLI args via the on_args hook rather than passing them
# through the Task object (which carries only per-task data).
#
# `_UNIFIED_VOCAB_PATH` is a per-run constant (the same corpus-wide
# vocab is read by every task in this dispatch) so it lands as a CLI
# arg here, mirroring `--vocab-source` and the unify_vocab worker's
# `--out-unified-vocab`. Per-variant data still travels through
# `TaskInfo.payload`; per-run config travels through CLI argv.
_SOURCE_DIR: Path
_VOCAB_DIR: Path
_OUTPUT_DIR: Path
_UNIFIED_VOCAB_PATH: Path


def _process_payload(
    task: Task,
    binary_name: str,
    payload_json: str,
    source_dir: Path,
    vocab_dir: Path,
    output_dir: Path,
    unified_vocab_path: Path,
) -> None:
    """Parse the inline payload, reconstruct BinaryVersionInfo, build memmap.

    `csv_path` and `mapping_path` in each payload entry are relative
    paths produced by the starting instance's `find_items` walks; we
    resolve them against `source_dir` / `vocab_dir` here so the same
    payload shape works under SLURM (where source_dir is the
    bind-mounted in-container path) and under local dispatch (where
    source_dir is the primary's filesystem root).

    Per-version resilience: the planner emits one entry per
    (compiler, version, opt, variant_id) the binary was built with —
    but phase 1 or phase 2 may have failed individually for some of
    those. Skip entries whose csv or mapping is missing on disk; only
    fail the whole group if NO entries survive (no usable input at
    all).

    `variant_id` and the optional `_meta.json` sidecar plumb the
    per-variant metadata through to `build_memmap_files`. The
    canonical-4 axes plus `variant_id` come from the wire payload
    (authoritative — the planner already parsed them out of the
    matched filename). The `_meta.json` sidecar (when present)
    supplements with `pkg` and the opaque `extra_metadata` dict via
    `VariantInfo.from_csv`, which is the single source of truth for
    filename + sidecar identity recovery. Legacy entries without a
    sidecar default `pkg` to the group's `binary_name` and
    `extra_metadata` to an empty dict.

    `BinaryVersionInfo.filename` keeps the planner-supplied value
    verbatim (the variant folder name for sidecar variants, the CSV
    basename for legacy 4-axis) — it is the canonical on-disk
    identity for the slim `_variants.csv`. `VariantInfo.from_csv`
    derives a different `filename` shape (the CSV basename always),
    which would collide same-binary sidecar variants onto a single
    row; we deliberately use only its `pkg` + `extra_metadata`
    outputs at this boundary.
    """
    data = json.loads(payload_json)
    versions_raw = data["versions"]
    versions: list[BinaryVersionInfo] = []
    skipped: list[str] = []
    for entry in versions_raw:
        csv_path = source_dir / entry["csv_path"]
        mapping_path = vocab_dir / entry["mapping_path"]
        if not csv_path.exists():
            skipped.append(f"{entry['arch']}-{entry['compiler']}-{entry['compilerversion']}-{entry['opt']} (csv missing: {csv_path})")
            continue
        if not mapping_path.exists():
            skipped.append(f"{entry['arch']}-{entry['compiler']}-{entry['compilerversion']}-{entry['opt']} (mapping missing: {mapping_path})")
            continue

        # Resolve per-variant metadata via the canonical
        # ``VariantInfo.from_csv`` entry-point. Only invoked when the
        # planner declared a sidecar path (legacy entries skip the
        # call entirely to preserve the old default shape — pkg =
        # group's binary_name, extra_metadata = {}). The declared-
        # but-missing warning is emitted by ``from_csv`` itself
        # (mirrors the prior in-worker behavior; just relocated to
        # the canonical owner).
        pkg: str = binary_name
        extra_metadata: dict = {}
        meta_path_rel = entry.get("meta_path")
        if meta_path_rel is not None:
            info = VariantInfo.from_csv(
                csv_path, meta_path=source_dir / meta_path_rel
            )
            pkg = info.pkg
            extra_metadata = info.extra_metadata

        versions.append(
            BinaryVersionInfo(
                path=csv_path,
                mapping_path=mapping_path,
                arch=entry["arch"],
                compiler=entry["compiler"],
                compilerversion=entry["compilerversion"],
                opt=entry["opt"],
                pkg=pkg,
                variant_id=int(entry.get("variant_id", 0)),
                extra_metadata=extra_metadata,
                # ``filename`` flows from the planner verbatim (the
                # CSV's parent folder name for sidecar variants, CSV
                # basename for legacy). See function docstring for
                # why ``VariantInfo.from_csv``'s filename is *not*
                # used here.
                filename=entry.get("filename", "") or "",
            )
        )

    if skipped:
        logger.warning(
            f"[!] {binary_name}: skipped {len(skipped)} of "
            f"{len(versions_raw)} versions due to missing inputs:\n  "
            + "\n  ".join(skipped)
        )

    if not versions:
        # Every version's inputs are missing — phase 1/2 failed
        # entirely for this binary group. Re-running won't fix it.
        raise FileNotFoundError(
            f"build_memmap: 0 of {len(versions_raw)} versions for "
            f"binary {binary_name!r} have both csv and mapping on disk."
        )

    # Stage the seven per-binary memmap artifacts (`*_data.bin`,
    # `*_unmatched_data.bin`, `*_sections.csv`, `*_index.bin`,
    # `*_unmatched_sections.csv`, `*_unmatched_index.bin`,
    # `<binary>.warn.log`) under `/app/out-tmp/build_memmap/<binary>/`
    # and atomic-publish to `output_dir` only on clean exit. A worker
    # killed mid-write leaves nothing partial on `/app/out-network`.
    with staged_publish(task, output_dir, scope=f"build_memmap/{binary_name}") as stage_dir:
        build_memmap_files(versions, stage_dir, binary_name, unified_vocab_path)


@task_function
def handle(task: Task) -> WorkerOutput | None:
    """Per-task body — runtime owns the read/run/respond cycle.

    `task.relative_path` is the binary_name (opaque identifier);
    `task.payload_str` is the per-version pairing JSON the starting
    instance emits via TaskInfo.payload. The runtime's exception →
    wire mapping turns:
      * MemoryError                 → OUT_OF_MEMORY (exit)
      * RecoverableError            → RECOVERABLE
      * NonRecoverableError         → NON_RECOVERABLE
      * FileNotFoundError /
        IsADirectoryError /
        NotADirectoryError          → routed to NON_RECOVERABLE below
        because filesystem-shape errors are deterministic for this
        input set — re-running won't make a missing CSV appear.
      * everything else             → WorkerExceptionResponse
                                      (default: RECOVERABLE).
    """
    binary_name = task.relative_path
    payload_str = task.payload_str
    if not payload_str:
        from dynamic_runner.worker import NonRecoverableError

        raise NonRecoverableError(
            f"build_memmap worker received task without payload for "
            f"binary_name={binary_name!r}; discover_items must emit "
            f"non-empty TaskInfo.payload for this task type."
        )
    logger.info(f"[*] Processing group: {binary_name}")
    task.set_phase("build_memmap")

    try:
        _process_payload(
            task,
            binary_name,
            payload_str,
            _SOURCE_DIR,
            _VOCAB_DIR,
            _OUTPUT_DIR,
            _UNIFIED_VOCAB_PATH,
        )
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as e:
        # Filesystem-shape errors are permanent for this input set —
        # re-running won't make a missing path appear. Surface as
        # NonRecoverable so the framework doesn't burn the retry-pass
        # on a deterministic miss.
        from dynamic_runner.worker import NonRecoverableError

        raise NonRecoverableError(f"{type(e).__name__}: {e}") from e
    return None


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Memmap-builder worker (per-binary-group).",
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
            "Source directory. Informational only at this layer — "
            "the per-version csv/mapping paths come absolute inside "
            "the wire's task payload."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for memmap files",
    )
    parser.add_argument(
        "--vocab-source",
        type=str,
        default=None,
        help=(
            "Vocab source directory (informational; the wire payload "
            "carries absolute mapping paths)."
        ),
    )
    parser.add_argument(
        "--unified-vocab",
        type=str,
        required=True,
        help=(
            "Path to the corpus-wide unified_vocab.csv (format_version=3) "
            "produced by the unify_vocab phase. Loaded once per worker "
            "and threaded into build_memmap_files so variant-axis tokens "
            "resolve to their assigned uint16 IDs. Required: v3 cutover "
            "removed the vocab-less mode."
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
            "Accepted for framework compatibility; per-group skip "
            "is governed by the starting instance's discover_items "
            "filter, not the worker."
        ),
    )
    return parser


def _on_args(args: argparse.Namespace) -> None:
    """Hook invoked by `run()` before the loop starts. Sets up
    logging and the module-level config the handler closes over.
    """
    global _SOURCE_DIR, _VOCAB_DIR, _OUTPUT_DIR, _UNIFIED_VOCAB_PATH

    increase_csv_field_size_limit()

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

    _SOURCE_DIR = Path(args.source).resolve()
    _VOCAB_DIR = (
        Path(args.vocab_source).resolve() if args.vocab_source else _SOURCE_DIR
    )
    _OUTPUT_DIR = Path(args.output).resolve()
    # `--unified-vocab` is resolved against the source root the same
    # mode-aware way the unify_vocab phase PUBLISHED it: that phase writes
    # the corpus vocab via `staged_publish(scope=UNIFY_VOCAB_SCOPE)`, which
    # lands the file at `<root>/<scope>/<name>` under container deployment
    # but flat at `<root>/<name>` standalone. `published_path` is the
    # inverse of that layout decision, so a bare-basename chain arg binds
    # to wherever the co-deployed unify worker actually placed the vocab —
    # `_SOURCE_DIR` being build_memmap's source root per the pipeline
    # routing. An absolute value (a standalone `--task build-memmap`
    # caller handing an explicit location) passes through unchanged
    # (pathlib `/` keeps an absolute right-hand side), so the standalone
    # path keeps working in both branches.
    _UNIFIED_VOCAB_PATH = published_path(
        _SOURCE_DIR, UNIFY_VOCAB_SCOPE, args.unified_vocab
    ).resolve()

    # Fail fast at worker startup if the unified vocab isn't reachable —
    # every task in this dispatch will hit the same miss inside
    # `build_memmap_files`, so surfacing here turns a per-task storm of
    # identical errors into a single readable startup failure.
    if not _UNIFIED_VOCAB_PATH.is_file():
        raise FileNotFoundError(
            f"build_memmap worker: --unified-vocab path does not exist "
            f"or is not a regular file: {_UNIFIED_VOCAB_PATH}"
        )

    logger.info(f"[*] Source directory: {_SOURCE_DIR}")
    logger.info(f"[*] Output directory: {_OUTPUT_DIR}")
    logger.info(f"[*] Unified vocab: {_UNIFIED_VOCAB_PATH}")


if __name__ == "__main__":
    run(argparser=_build_argparser(), on_args=_on_args)
