"""TaskDefinition for vocabulary unification.

Vocabulary unification is inherently a single-process aggregation:
each binary's per-binary vocabulary is registered onto a shared
unified `VocabularyManager`, and per-binary mapping files are emitted
beside the input CSV. The framework can still drive it as a runner
task — one TaskInfo carrying the discovered CSV list inline in
`TaskInfo.payload`, dispatched to one worker. Same shape as
build_memmap's `uses_file_based_items=False` pattern.

Why bother with a runner task at all when there's no parallelism to
exploit? Autonomous SLURM dispatch. With phase 2 as a runner task,
the user can submit `dynrunner --task all --multi-computer slurm`
and the cluster runs phase 1 (multi-worker tokenize) → phase 2
(single-worker unify-vocab) → phase 3 (multi-worker build-memmap)
without local interleaving — each phase's outputs land on cluster
NFS, the next phase's secondaries see them through the bind-mount.
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from pathlib import Path

from dynamic_runner import _native
from dynamic_runner.task_protocol import PhaseSpec, TaskTypeSpec, TypeId

from dynrunner.binary_selection import (
    BinaryIdentifier,
    SelectionFilters,
    TaskInfo,
    add_asm_selection_arguments,
    compile_selection_filters,
    is_excluded_subfolder,
    match_filename,
    process_selection_arguments,
)


_logger = logging.getLogger(__name__)

_PHASE_ID = "unify_vocab"
_TYPE_ID = "unify_vocab"

# Tokenize phase emits *_output.csv next to each binary's auxiliary
# files. The unifier reads these to build the shared vocab + per-CSV
# mapping files.
_CSV_SUFFIX = "_out\\put.\\csv"


def _csv_format(base_format: str) -> str:
    return base_format + _CSV_SUFFIX


class _CsvVisitor:
    """Visitor for `_native.find_items`: marks every CSV output file
    matching the compiled selection filters with its parsed
    `BinaryIdentifier` as the per-file payload.
    """

    def __init__(self, filters: SelectionFilters) -> None:
        self._filters = filters

    def visit(
        self,
        parent_payload: str | None,
        subfolders: list,
        files: list,
    ) -> None:
        current_rel = parent_payload or ""
        for folder in subfolders:
            child_rel = (
                f"{current_rel}/{folder.name}" if current_rel else folder.name
            )
            if is_excluded_subfolder(child_rel, self._filters):
                folder.enter(False)
            else:
                folder.enter(True, payload=child_rel)
        for f in files:
            identifier = match_filename(f.name, self._filters)
            if identifier is not None:
                f.mark(True, payload=identifier)


class _OutputFilenameCollector:
    """Visitor used by `--skip-existing`: walks the resolved output
    root and records every basename. Vocab unification's canonical
    done-marker is `unified_vocab.csv` (single file produced once
    per run); we treat the run as already-done if that file exists
    in the output root.
    """

    def __init__(self) -> None:
        self.filenames: set[str] = set()

    def visit(
        self,
        parent_payload: object,
        subfolders: list,
        files: list,
    ) -> None:
        for folder in subfolders:
            folder.enter(True)
        for f in files:
            self.filenames.add(f.name)


def _walk_with_filters(
    root: str, gateway_url: str | None, filters: SelectionFilters
) -> list:
    visitor = _CsvVisitor(filters)
    return _native.find_items(visitor, root, gateway_url=gateway_url)


def _collect_existing_output_filenames(
    output_root: str, gateway_url: str | None
) -> set[str]:
    collector = _OutputFilenameCollector()
    try:
        _native.find_items(collector, output_root, gateway_url=gateway_url)
    except OSError:
        return set()
    return collector.filenames


class VocabUnifierTask:
    """Vocab-unifier task: one phase, one type, one item per run.

    The single TaskInfo carries the full list of discovered CSV
    relative paths in `TaskInfo.payload`. The worker reads the
    payload, joins each entry against its `--source` (the bind-mount
    root inside the container, or the local source dir for non-SLURM
    dispatch), and runs `unify_vocab(csv_files, unified_vocab_path)`.
    """

    # `TaskInfo.path` is the literal "unify_vocab" sentinel — not a
    # real file. Skip the framework's file-based staging.
    uses_file_based_items: bool = False

    # ── Topology ───────────────────────────────────────────────────────

    def get_phases(self) -> tuple[PhaseSpec, ...]:
        return (
            PhaseSpec(
                phase_id=_PHASE_ID,
                types=(
                    TaskTypeSpec(
                        type_id=_TYPE_ID,
                        worker_module="dynrunner.unify_vocab.worker",
                        # Vocab unification reads each CSV's tail
                        # (vocab definition row) and registers tokens
                        # on a shared VocabularyManager. For 235
                        # binaries the inner loop is dominated by I/O;
                        # 5 min is generous.
                        timeout_seconds=300.0,
                    ),
                ),
            ),
        )

    # ── Item discovery ─────────────────────────────────────────────────

    def discover_items(
        self, source_dir: Path, args: Namespace
    ) -> Iterable[TaskInfo]:
        """Walk source dir for `*_output.csv` files via
        `_native.find_items`, emit ONE TaskInfo whose payload lists
        every matched CSV's relative path. Worker resolves against
        its own source_dir at run time.
        """
        config = process_selection_arguments(args)
        if getattr(args, "source_already_staged", None):
            root = args.source_already_staged
            gateway_url = getattr(args, "gateway", None)
        else:
            root = str(config.source_dir)
            gateway_url = None

        csv_filters = compile_selection_filters(
            _replace_format(config, _csv_format(config.file_format))
        )
        csv_items = _walk_with_filters(root, gateway_url, csv_filters)

        if not csv_items:
            _logger.warning(
                "vocab_unifier discovery: no CSV files matched %s under %s; "
                "no work to dispatch.",
                _csv_format(config.file_format),
                root,
            )
            return []

        relative_paths = sorted(str(item.path) for item in csv_items)
        total_size = sum(item.size for item in csv_items)
        # Pick a representative identifier for the single TaskInfo; the
        # actual unifier doesn't use it (it's a per-run aggregate, not
        # per-binary), but the framework expects a non-None identifier.
        rep = csv_items[0]

        # Skip-existing: vocab unification's canonical done-marker is
        # `unified_vocab.csv`. If it's already at the output root, this
        # run has nothing to do.
        if getattr(args, "skip_existing", False):
            output_root = getattr(args, "resolved_output_root", None)
            if output_root:
                completed = _collect_existing_output_filenames(
                    output_root, gateway_url
                )
                unified_name = self.get_output_filename_pattern(_TYPE_ID, None)
                if unified_name in completed:
                    _logger.info(
                        "skip-existing: %s already at %s; vocab unification "
                        "needs no re-dispatch.",
                        unified_name,
                        output_root,
                    )
                    return []

        return [
            TaskInfo(
                # `path` is an opaque sentinel — TaskInfo is one-per-run,
                # not one-per-file. uses_file_based_items=False bypasses
                # framework's hash-staging.
                path=Path("unify_vocab"),
                size=total_size,
                identifier=BinaryIdentifier(
                    binary_name="unify_vocab",
                    platform=rep.identifier.platform,
                    compiler=rep.identifier.compiler,
                    version=rep.identifier.version,
                    opt_level=rep.identifier.opt_level,
                ),
                phase_id=_PHASE_ID,
                type_id=_TYPE_ID,
                affinity_id=None,
                payload={
                    "csv_paths": relative_paths,
                },
            )
        ]

    # ── Per-type plumbing ──────────────────────────────────────────────

    def estimate_memory(self, item: TaskInfo) -> int:
        """Working set is the shared unified `VocabularyManager` plus
        the CSV reader's per-file tail buffer. For 235 binaries the
        unified vocab is ~600 tokens; total RAM stays well under
        512 MiB. Use a flat 768 MiB budget — gives headroom while
        leaving room for other phases on the same secondary.
        """
        return 768 * 1024 * 1024

    def add_task_arguments(self, parser: ArgumentParser) -> None:
        # Vendored asm-binary corpus selection flags (--platform etc.) —
        # see TokenizerTask.add_task_arguments for rationale.
        add_asm_selection_arguments(parser)
        parser.add_argument(
            "--out-unified-vocab",
            type=str,
            default="unified_vocab.csv",
            help=(
                "Filename for the unified vocabulary CSV (placed in the "
                "output dir; default: unified_vocab.csv)."
            ),
        )

    def build_worker_command_args(
        self,
        type_id: TypeId,
        args: Namespace,
        source_dir: Path,
        output_dir: Path,
        skip_existing: bool,
    ) -> list[str]:
        cmd: list[str] = []
        if getattr(args, "out_unified_vocab", None):
            cmd.extend(["--out-unified-vocab", str(args.out_unified_vocab)])
        return cmd

    def get_output_filename_pattern(
        self, type_id: TypeId, item: TaskInfo | None
    ) -> str:
        # Matches the standalone CLI's default and what
        # `tokenizer.vocab_unifier.unifier.unify_vocab` writes.
        return "unified_vocab.csv"

    # ── Lifecycle hooks ────────────────────────────────────────────────

    def on_run_start(
        self, source_dir: Path, output_dir: Path, args: Namespace
    ) -> None:
        pass

    def on_run_end(self, success: bool) -> None:
        pass

    def on_phase_start(self, phase_id: str) -> None:
        pass

    def on_phase_end(self, phase_id: str, completed: int, failed: int) -> None:
        pass


def _replace_format(config, file_format: str):
    import dataclasses
    return dataclasses.replace(config, file_format=file_format)
