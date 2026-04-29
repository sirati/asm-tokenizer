"""TokenizerTask: migrate binary files into per-binary CSV token streams.

The dynamic_runner framework dispatches one task type — the tokenizer
worker module — through one phase. The worker internally cycles through
three angr passes (the old TokenizerPhase enum) and a final
tokenization writeout; that internal phasing is invisible to the
framework now (the new design has no `get_stages`).
"""

from __future__ import annotations

import math
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

from dynamic_runner._shared import (
    TaskInfo,
    find_matching_binaries,
    process_selection_arguments,
)
from dynamic_runner.task_protocol import PhaseSpec, TaskTypeSpec, TypeId


_PHASE_ID = "tokenize"
_TYPE_ID = "tokenizer"


class TokenizerPhase(str, Enum):
    """Worker-internal progress labels.

    The framework no longer cares about these — the new task protocol
    has only one phase per type. The worker still emits these strings
    via `PhaseUpdateResponse` so the runner's logs/progress bar can show
    which sub-stage of tokenization is running.
    """

    ANGR_1 = "angr-1"
    ANGR_2 = "angr-2"
    TOKENIZATION = "tokenization"


class TokenizerTask:
    """Task definition for binary tokenization."""

    # ── Topology ───────────────────────────────────────────────────────

    def get_phases(self) -> tuple[PhaseSpec, ...]:
        return (
            PhaseSpec(
                phase_id=_PHASE_ID,
                types=(
                    TaskTypeSpec(
                        type_id=_TYPE_ID,
                        worker_module="tokenizer",
                        # Old design: per-stage timeouts (None, None,
                        # 10.0s). The new design has one timeout per
                        # type; pick the tightest meaningful value (the
                        # final csv-writing keepalive).
                        timeout_seconds=10.0,
                    ),
                ),
            ),
        )

    # ── Item discovery ─────────────────────────────────────────────────

    def discover_items(
        self, source_dir: Path, args: Namespace
    ) -> Iterable[TaskInfo]:
        """Yield binaries under `source_dir`, grouped by binary name with
        intra-group size-DESC ordering, and tagged with this task's
        phase + type.

        The old `organize_and_sort_items` did the re-ordering after the
        framework's file scan; the new design folds the scan and the
        ordering into one method.
        """
        return self._sort_and_tag(_scan(source_dir, args))

    def organize_and_sort_items(
        self, items: Iterable[TaskInfo]
    ) -> list[TaskInfo]:
        """Compat shim for the legacy `dynamic_runner.run()` path.

        The runner's `run.py` still calls `find_matching_binaries` +
        `task.organize_and_sort_items()` instead of `discover_items()`.
        This method preserves the historical sort+tag behaviour so the
        legacy entry point keeps working until the runner is updated to
        call `discover_items`. Both methods share `_sort_and_tag`.
        """
        return list(self._sort_and_tag(items))

    @staticmethod
    def _sort_and_tag(items: Iterable[TaskInfo]) -> Iterable[TaskInfo]:
        """Group by binary_name; sort within group by size DESC; order
        groups by group-average size DESC; tag each item with this
        task's phase + type."""
        groups: dict[str, list[TaskInfo]] = defaultdict(list)
        for binary in items:
            groups[binary.binary_name].append(binary)

        group_averages: list[tuple[str, float, list[TaskInfo]]] = []
        for binary_name, group in groups.items():
            avg_size = sum(b.size for b in group) / len(group)
            group.sort(key=lambda b: b.size, reverse=True)
            group_averages.append((binary_name, avg_size, group))
        group_averages.sort(key=lambda x: x[1], reverse=True)

        for _, _, group in group_averages:
            for b in group:
                # Tag each item with phase + type so the framework
                # routes it correctly. affinity_id stays None — the
                # tokenizer has no cache-locality classes worth
                # exploiting.
                b.phase_id = _PHASE_ID
                b.type_id = _TYPE_ID
                yield b

    # ── Per-type plumbing ──────────────────────────────────────────────

    def estimate_memory(self, item: TaskInfo) -> int:
        """Power-law estimator: RAM_MiB = 430.870 * size_MiB^1.051 + 260.15.

        R² = 0.9866, RMSE = 203.66 MiB. Adds the RMSE so we
        rather over-estimate than under-estimate.
        """
        mb = item.size / 1024 / 1024
        a = 430.870
        b = 1.051
        c = 260.15
        rmse = 203.66
        ram_mb = a * (mb**b) + c + rmse
        return math.ceil(ram_mb * 1024 * 1024)

    def add_task_arguments(self, parser: ArgumentParser) -> None:
        # No tokenizer-specific arguments beyond the framework's
        # standard file-discovery parameters.
        pass

    def build_worker_command_args(
        self,
        type_id: TypeId,
        args: Namespace,
        source_dir: Path,
        output_dir: Path,
        skip_existing: bool,
    ) -> list[str]:
        cmd_args = ["--platform", "auto"]
        if hasattr(args, "simulate_errors") and args.simulate_errors is not None:
            cmd_args.extend(["--simulate-errors", str(args.simulate_errors)])
        return cmd_args

    def get_output_filename_pattern(
        self, type_id: TypeId, item: TaskInfo
    ) -> str:
        return f"{item.path.name}_output.csv"

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


def _scan(source_dir: Path, args: Namespace) -> list[TaskInfo]:
    """Scan `source_dir` for binaries matching the framework's standard
    selection arguments (`--platform`, `--compiler`, `--compiler-versions`,
    `--opt`, `--file-format`, `--version-regex`, `--opt-regex`,
    `--name-regex`, `--exclude-subfolder`).

    Mirrors the pattern used by `tokenizer/vocab_unifier/__main__.py`:
    build a `SelectionConfig` from the parsed `args` Namespace and pass
    its fields into `find_matching_binaries`. `source_dir` is passed by
    the framework but `config.source_dir` is the canonical, resolved
    value (Path-resolved, validated to exist).
    """
    config = process_selection_arguments(args)
    return find_matching_binaries(
        source_dir=config.source_dir,
        platforms=config.platforms,
        compiler=config.compiler,
        compiler_versions=config.compiler_versions,
        opt_levels=config.opt_levels,
        format_string=config.file_format,
        version_regex=config.version_regex,
        opt_regex=config.opt_regex,
        name_regex=config.name_regex,
        exclude_subfolders=config.exclude_subfolders,
    )
