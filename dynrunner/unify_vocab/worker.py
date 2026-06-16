"""Worker subprocess for `dynrunner.unify_vocab`.

Receives one ``ProcessBinaryCommand`` per run. The wire's
``relative_path`` is the literal "unify_vocab" sentinel and
``payload`` is a JSON string carrying the list of discovered
per-binary CSV relative paths. The worker:

1. Parses ``command.payload`` into a ``csv_paths`` list.
2. Joins each entry against ``--source`` (the bind-mount root inside
   the container, or the local source dir for non-SLURM dispatch).
3. Calls ``tokenizer.vocab_unifier.unifier.unify_vocab(csv_files,
   unified_vocab_path)`` and replies ``done``.

The standalone CLI at ``tokenizer.vocab_unifier.__main__`` does its
own discovery; this worker doesn't. Discovery is the starting
instance's concern under the dynrunner Protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dynamic_runner.worker import NonRecoverableError, Task, WorkerOutput, run, task_function

from shared import remove_stream_handlers
from tokenizer.output_staging import UNIFY_VOCAB_SCOPE, staged_publish
from tokenizer.vocab_unifier.unifier import unify_vocab

logger = logging.getLogger(__name__)

# Module-level config populated by `_on_args` before the run-loop
# starts; the handler closes over it. The runtime's contract is
# that on_args owns CLI-arg → config translation, the handler reads
# from there.
_SOURCE_DIR: Path
_OUTPUT_DIR: Path
_UNIFIED_VOCAB_FILENAME: str
_INSERT_VALUE_NEGATIVE: bool = False


def _process_payload(
    task: Task,
    payload_json: str,
    source_dir: Path,
    output_dir: Path,
    unified_vocab_filename: str,
    insert_value_negative: bool,
) -> None:
    data = json.loads(payload_json)
    csv_paths_raw = data["csv_paths"]
    csv_files = [source_dir / rel for rel in csv_paths_raw]
    # Stage all writes (unified vocab CSV + per-CSV mapping files) under
    # `/app/out-tmp/unify_vocab/` and atomic-publish to `output_dir` only
    # on clean exit. Cross-mount publish keeps `/app/out-network` free
    # of partials when the container is killed mid-run.
    #
    # `source_dir` is the relative root the planner's CSV paths resolve
    # against. Passing it as `mapping_source_root` makes the unifier
    # preserve each CSV's subdir under the staging dir for its mapping
    # file; the staged_publish layer then mirrors that subdir layout
    # into `output_dir` so build_memmap can pair `<rel>/<binary>_output.csv`
    # with `<rel>/<binary>_output.mapping.b64c` by relative path.
    with staged_publish(task, output_dir, scope=UNIFY_VOCAB_SCOPE) as stage_dir:
        unify_vocab(
            csv_files,
            stage_dir / unified_vocab_filename,
            mapping_output_dir=stage_dir,
            mapping_source_root=source_dir,
            insert_value_negative=insert_value_negative,
        )


@task_function
def handle(task: Task) -> WorkerOutput | None:
    """Per-task body — runtime owns the read/run/respond cycle.

    The runtime's exception → wire mapping turns:
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
    if not task.payload_str:
        raise NonRecoverableError(
            "vocab_unifier worker received task without payload; "
            "discover_items must emit non-empty TaskInfo.payload for "
            "this task type."
        )
    logger.info("[*] Processing unify_vocab task")
    task.set_phase("unify_vocab")

    try:
        _process_payload(
            task,
            task.payload_str,
            _SOURCE_DIR,
            _OUTPUT_DIR,
            _UNIFIED_VOCAB_FILENAME,
            _INSERT_VALUE_NEGATIVE,
        )
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as e:
        raise NonRecoverableError(f"{type(e).__name__}: {e}") from e
    return None


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vocab-unifier worker (single-process aggregation).",
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
            "Source directory. Per-CSV relative paths in the worker's "
            "task payload resolve against this; in SLURM bind-mount "
            "deployments this is `/app/src-network`."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the unified vocab CSV.",
    )
    parser.add_argument(
        "--out-unified-vocab",
        type=str,
        default="unified_vocab.csv",
        help="Filename for the unified vocab CSV (default: unified_vocab.csv).",
    )
    parser.add_argument(
        "--insert-neg-value",
        action="store_true",
        help=(
            "Legacy-compat: per-binary CSVs were generated BEFORE "
            "value_negative was reserved at slot 256 (first real token at "
            "per-binary id 256). Load them with only the 256 digit slots "
            "reserved and remap real tokens (256+) into the canonical "
            "unified layout. Use for corpora tokenized pre-value_negative-"
            "cutover; the emitted unified vocab keeps the canonical "
            "257-reserved layout."
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
            "Accepted for framework compatibility; the discover_items "
            "filter on the starting instance owns the skip-existing "
            "check, not the worker."
        ),
    )
    return parser


def _on_args(args: argparse.Namespace) -> None:
    global _SOURCE_DIR, _OUTPUT_DIR, _UNIFIED_VOCAB_FILENAME, _INSERT_VALUE_NEGATIVE

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
    _OUTPUT_DIR = Path(args.output).resolve()
    _UNIFIED_VOCAB_FILENAME = args.out_unified_vocab
    _INSERT_VALUE_NEGATIVE = args.insert_neg_value

    logger.info(f"[*] Source directory: {_SOURCE_DIR}")
    logger.info(f"[*] Output directory: {_OUTPUT_DIR}")


if __name__ == "__main__":
    run(argparser=_build_argparser(), on_args=_on_args)
