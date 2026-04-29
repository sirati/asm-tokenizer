"""TaskDefinition for the per-binary-group memmap builder.

The library function `tokenizer.memmap_builder.builder.build_memmap_files`
consumes a list of `BinaryVersionInfo` (one per (compiler, version, opt)
triple) for a single `binary_name` group. The dynrunner wire protocol,
however, only carries `relative_path` per item, so we cannot pass the
version list directly to the worker.

Resolution: **manifest-on-disk**. The starting instance does ALL discovery
+ pairing + grouping inside `discover_items` (and the compat shim
`organize_and_sort_items`). For each binary_name group it writes a JSON
manifest to ``<output_dir>/.dynrunner-memmap/<binary_name>.json`` and
yields a single TaskInfo whose `path` is the manifest file. The worker
reads its assigned manifest, reconstructs `BinaryVersionInfo` instances,
and calls `build_memmap_files`. The worker scans nothing.

Compat shim: `dynamic_runner.run.run()` (`run.py:82-128`) still uses the
legacy path — `_shared.find_matching_binaries(...)` followed by
`task.organize_and_sort_items(binaries)`. New-API tasks must implement
*both* methods until the runtime migrates. To keep the file-format
defaults useful in this hybrid era, `add_task_arguments` rewrites
``--file-format``'s default so the framework's pre-shim scan discovers
CSV outputs (otherwise it returns ``[]`` and the run aborts before our
shim runs).
"""

from __future__ import annotations

import json
import logging
import shutil
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from shared import (
    BinaryInfo,
    find_matching_binaries,
    format_binary_info,
    process_selection_arguments,
)

from dynamic_runner._shared.binary_info import BinaryIdentifier, TaskInfo
from dynamic_runner.task_protocol import PhaseSpec, TaskTypeSpec, TypeId
from tokenizer.memmap_builder._pairing import match_csv_to_mapping


logger = logging.getLogger(__name__)


_PHASE_ID = "memmap"
_TYPE_ID = "memmap"
_STAGING_SUBDIR = ".dynrunner-memmap"

_CSV_SUFFIX = "_out\\put.\\csv"
_MAPPING_SUFFIX = "_out\\put.ma\\p\\ping.b64\\c"


def _csv_format(base_format: str) -> str:
    return base_format + _CSV_SUFFIX


def _mapping_format(base_format: str) -> str:
    return base_format + _MAPPING_SUFFIX


def _write_manifest(
    manifest_path: Path,
    csv_binaries: list[BinaryInfo],
    matched_pairs: dict,
) -> int:
    """Write a per-group manifest. Returns total CSV size (rough work estimate)."""
    entries = []
    total_size = 0
    for csv_bin in csv_binaries:
        map_bin = matched_pairs[csv_bin.identifier]
        size = csv_bin.size
        total_size += size
        entries.append(
            {
                "csv_path": str(csv_bin.path),
                "mapping_path": str(map_bin.path),
                "arch": csv_bin.platform,
                "compiler": csv_bin.compiler,
                "compilerversion": csv_bin.version,
                "opt": csv_bin.opt_level,
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"versions": entries}, indent=2))
    return total_size


class MemmapBuilderTask:
    """Memmap-builder task: one phase, one type, one item per binary_name group."""

    def __init__(self) -> None:
        # Captured by the parse_args-wrapping shim in `add_task_arguments`
        # so `organize_and_sort_items` can recover source/output dirs and
        # the optional `--vocab-source`. Populated by the time any item
        # is dispatched, which is after `parser.parse_args()` returns.
        self._captured_args: Namespace | None = None
        # User's BASE file_format (before we append the CSV suffix to
        # satisfy the legacy ``_collect_binaries`` scan). Used by the
        # shim to derive both CSV and mapping format strings.
        self._base_file_format: str | None = None

    # ── Topology ───────────────────────────────────────────────────────

    def get_phases(self) -> tuple[PhaseSpec, ...]:
        return (
            PhaseSpec(
                phase_id=_PHASE_ID,
                types=(
                    TaskTypeSpec(
                        type_id=_TYPE_ID,
                        worker_module="dynrunner.build_memmap.worker",
                        # Memmap building scans large CSVs and writes
                        # multiple bin/csv outputs; per-group runtime is
                        # dominated by I/O on the largest version. 5 min
                        # is a generous keepalive ceiling — the worker
                        # has no inner-loop keepalive yet.
                        timeout_seconds=300.0,
                    ),
                ),
            ),
        )

    # ── Item discovery ─────────────────────────────────────────────────

    def discover_items(
        self, source_dir: Path, args: Namespace
    ) -> Iterable[TaskInfo]:
        """Discover CSV outputs, pair each with its mapping file, group by
        binary_name, write a manifest per group, and yield one TaskInfo
        per manifest. All decisional work happens here, on the starting
        instance.
        """
        config = process_selection_arguments(args)
        vocab_source_dir = (
            Path(args.vocab_source).resolve() if args.vocab_source else config.source_dir
        )
        output_dir = config.output_dir
        staging_dir = output_dir / _STAGING_SUBDIR

        # Use the BASE format (before the CSV suffix the wrapping shim
        # appended). When invoked outside the wrapping shim — e.g.
        # directly by the new-API runtime — fall back to ``config.file_format``.
        base_format = self._base_file_format or config.file_format

        csv_binaries = find_matching_binaries(
            source_dir=config.source_dir,
            platforms=config.platforms,
            compiler=config.compiler,
            compiler_versions=config.compiler_versions,
            opt_levels=config.opt_levels,
            format_string=_csv_format(base_format),
            version_regex=config.version_regex,
            opt_regex=config.opt_regex,
            name_regex=config.name_regex,
            exclude_subfolders=config.exclude_subfolders,
        )
        mapping_binaries = find_matching_binaries(
            source_dir=vocab_source_dir,
            platforms=config.platforms,
            compiler=config.compiler,
            compiler_versions=config.compiler_versions,
            opt_levels=config.opt_levels,
            format_string=_mapping_format(base_format),
            version_regex=config.version_regex,
            opt_regex=config.opt_regex,
            name_regex=config.name_regex,
            exclude_subfolders=config.exclude_subfolders,
        )

        matched_pairs, unmatched_csv = match_csv_to_mapping(csv_binaries, mapping_binaries)
        if unmatched_csv:
            logger.warning(
                f"{len(unmatched_csv)} CSV file(s) have no matching mapping file:"
            )
            for csv_bin in unmatched_csv:
                logger.warning(f"  {format_binary_info(csv_bin, config.source_dir)}")

        csv_by_identifier = {csv_bin.identifier: csv_bin for csv_bin in csv_binaries}
        matched_csv_binaries = [
            csv_by_identifier[identifier] for identifier in matched_pairs.keys()
        ]

        groups: dict[str, list[BinaryInfo]] = defaultdict(list)
        for csv_bin in matched_csv_binaries:
            groups[csv_bin.binary_name].append(csv_bin)

        for binary_name, group in groups.items():
            manifest_path = staging_dir / f"{binary_name}.json"
            total_size = _write_manifest(manifest_path, group, matched_pairs)

            representative = group[0]
            yield TaskInfo(
                path=manifest_path,
                size=total_size,
                identifier=BinaryIdentifier(
                    binary_name=binary_name,
                    platform=representative.platform,
                    compiler=representative.compiler,
                    version=representative.version,
                    opt_level=representative.opt_level,
                ),
                phase_id=_PHASE_ID,
                type_id=_TYPE_ID,
                affinity_id=None,
                payload={},
            )

    # ── Compat shim for the legacy run.py path ─────────────────────────

    def organize_and_sort_items(self, items: list) -> list[TaskInfo]:
        """Compat shim: ignore the framework's pre-discovered items
        (scanned with the wrong format string for CSV/mapping pairing)
        and run our own discovery via `discover_items`.

        `dynamic_runner.run.run()` calls this BEFORE `on_run_start`, so
        we cannot rely on a lifecycle hook to hand us source/output dirs
        — we recover them from `self._captured_args` instead.
        """
        if self._captured_args is None:
            # If parse_args never ran (unusual; e.g. unit-test path)
            # fall back to keeping whatever the framework gave us.
            return list(items)
        config = process_selection_arguments(self._captured_args)
        return list(self.discover_items(config.source_dir, self._captured_args))

    # ── Per-type plumbing ──────────────────────────────────────────────

    def estimate_memory(self, item: TaskInfo) -> int:
        """Estimate per-group RAM. ``item.size`` is the sum of CSV sizes
        in this group; the builder's working set is dominated by
        ``lockstep_function_match`` buffering + numpy mapping arrays.
        We have no fitted model yet, so use 4x size + 512 MiB floor.
        """
        return max(4 * item.size, 512 * 1024 * 1024) + 256 * 1024 * 1024

    def add_task_arguments(self, parser: ArgumentParser) -> None:
        """Add ``--vocab-source``.

        We also wrap ``parser.parse_args`` to:
        1. Append ``_CSV_SUFFIX`` to ``args.file_format`` so the runtime's
           legacy ``_collect_binaries`` (which scans with that format
           string) finds CSV outputs — otherwise the run aborts before
           our compat shim can re-discover with proper formats.
           ``self._base_file_format`` retains the user's BASE format for
           use in pairing.
        2. Capture the parsed Namespace into ``self._captured_args``;
           the runtime calls ``organize_and_sort_items`` without args,
           so this wrapping is the only way to recover them.
        """
        parser.add_argument(
            "--vocab-source",
            type=str,
            default=None,
            help=(
                "Source directory for vocabulary and mapping files. "
                "If not specified, uses the same as --source."
            ),
        )

        original_parse_args = parser.parse_args

        def _capturing_parse_args(*pa_args, **pa_kwargs):
            namespace = original_parse_args(*pa_args, **pa_kwargs)
            # Stash the user's BASE format BEFORE rewriting it for the
            # runtime's CSV-discovery scan.
            self._base_file_format = namespace.file_format
            namespace.file_format = _csv_format(namespace.file_format)
            self._captured_args = namespace
            return namespace

        parser.parse_args = _capturing_parse_args  # type: ignore[method-assign]

    def build_worker_command_args(
        self,
        type_id: TypeId,
        args: Namespace,
        source_dir: Path,
        output_dir: Path,
        skip_existing: bool,
    ) -> list[str]:
        # Worker reads its manifest and writes outputs under output_dir;
        # vocab-source is informational here (the manifest already
        # contains absolute mapping_paths), but pass it through for
        # logging parity.
        cmd: list[str] = []
        if getattr(args, "vocab_source", None):
            cmd.extend(["--vocab-source", str(args.vocab_source)])
        return cmd

    def get_output_filename_pattern(
        self, type_id: TypeId, item: TaskInfo
    ) -> str:
        # `build_memmap_files` writes ``<binary_name>_data.bin`` (and a
        # few siblings) — pick the index file as the canonical
        # done-marker for skip-existing checks.
        return f"{item.binary_name}_index.bin"

    # ── Lifecycle hooks ────────────────────────────────────────────────

    def on_run_start(
        self, source_dir: Path, output_dir: Path, args: Namespace
    ) -> None:
        pass

    def on_run_end(self, success: bool) -> None:
        # Best-effort cleanup of the staging dir. We only remove it on
        # success — failed runs may want the manifest list for triage.
        if not success or self._captured_args is None:
            return
        try:
            config = process_selection_arguments(self._captured_args)
            staging_dir = config.output_dir / _STAGING_SUBDIR
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
        except Exception as exc:  # pragma: no cover - cleanup is best-effort
            logger.warning(f"Failed to clean staging dir: {exc}")

    def on_phase_start(self, phase_id: str) -> None:
        pass

    def on_phase_end(self, phase_id: str, completed: int, failed: int) -> None:
        pass
