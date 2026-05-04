"""TokenizerTask: migrate binary files into per-binary CSV token streams.

The dynamic_runner framework dispatches one task type — the tokenizer
worker module — through one phase. The worker internally cycles through
three angr passes (the old TokenizerPhase enum) and a final
tokenization writeout; that internal phasing is invisible to the
framework now (the new design has no `get_stages`).
"""

from __future__ import annotations

import math
import re
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from re import Pattern

from dynamic_runner import _native
from dynamic_runner._shared import (
    BinaryIdentifier,
    TaskInfo,
    process_selection_arguments,
)
from dynamic_runner._shared.binary_info import (
    build_binary_filename_format,
    build_field_regexes,
    parse_binary_filename,
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
        """Drive the Rust walker via `_native.find_items`, then sort+tag.

        When `args.source_already_staged` is set (SLURM mode), discovery
        walks the gateway-side filesystem via SSH at that path; otherwise
        it walks the local `source_dir`. Filter state is stashed on
        `self` for the duration of the walk so `visit()` can read it
        without re-deriving from args per directory.
        """
        config = process_selection_arguments(args)
        self._filters = _build_filters(config)
        try:
            if getattr(args, "source_already_staged", None):
                root = args.source_already_staged
                gateway_url = getattr(args, "gateway", None)
            else:
                root = str(config.source_dir)
                gateway_url = None
            items = _native.find_items(self, root, gateway_url=gateway_url)
            return self._sort_and_tag(root, items)
        finally:
            self._filters = None

    def visit(
        self,
        parent_payload: str | None,
        subfolders: list,
        files: list,
    ) -> None:
        """Per-directory policy callback driven by `_native.find_items`.

        `parent_payload` carries the relative path of the current
        directory (set by the parent's `enter()`; `None` at root).
        Mirrors today's `find_matching_binaries`: prune subfolders
        matching `--exclude-subfolder`, mark files matching the format
        regex + field allowlists with the parsed `BinaryIdentifier` as
        the per-file payload.
        """
        filters = self._filters
        if filters is None:
            return

        current_rel = parent_payload or ""

        for folder in subfolders:
            child_rel = (
                f"{current_rel}/{folder.name}" if current_rel else folder.name
            )
            if (
                filters.exclude_pattern is not None
                and filters.exclude_pattern.search(child_rel)
            ):
                folder.enter(False)
            else:
                folder.enter(True, payload=child_rel)

        for f in files:
            if f.name.startswith("."):
                continue
            parsed = parse_binary_filename(f.name, filters.binary_format)
            if not parsed:
                continue
            platform, comp, version, opt, binary_name = parsed
            if platform not in filters.platforms:
                continue
            if filters.compiler and comp != filters.compiler:
                continue
            if (
                filters.compiler_versions
                and version not in filters.compiler_versions
            ):
                continue
            if (
                filters.normalized_opt_levels
                and opt not in filters.normalized_opt_levels
            ):
                continue
            f.mark(
                True,
                payload=BinaryIdentifier(
                    binary_name=binary_name,
                    platform=platform,
                    compiler=comp,
                    version=version,
                    opt_level=opt,
                ),
            )

    @staticmethod
    def _sort_and_tag(root: str, items: Iterable) -> Iterable[TaskInfo]:
        """Group by binary_name; sort within group by size DESC; order
        groups by group-average size DESC; emit fresh Python `TaskInfo`
        instances tagged with this task's phase + type.

        `items` are `_native.PyTaskInfo` objects from `find_items`
        (read-only) carrying *relative* paths; we re-construct mutable
        Python `TaskInfo`s with paths joined back to `root` so the
        framework's downstream `compute_file_hash`/`strip_prefix` pass
        (in `queue_initial_staging`) sees the same absolute-path
        contract today's `find_matching_binaries` provided.
        affinity_id stays `None` — the tokenizer has no cache-locality
        classes worth exploiting.
        """
        root_path = Path(root)
        groups: dict[str, list] = defaultdict(list)
        for item in items:
            groups[item.identifier.binary_name].append(item)

        group_averages: list[tuple[str, float, list]] = []
        for binary_name, group in groups.items():
            avg_size = sum(b.size for b in group) / len(group)
            group.sort(key=lambda b: b.size, reverse=True)
            group_averages.append((binary_name, avg_size, group))
        group_averages.sort(key=lambda x: x[1], reverse=True)

        for _, _, group in group_averages:
            for b in group:
                yield TaskInfo(
                    path=root_path / str(b.path),
                    size=b.size,
                    identifier=BinaryIdentifier(
                        binary_name=b.identifier.binary_name,
                        platform=b.identifier.platform,
                        compiler=b.identifier.compiler,
                        version=b.identifier.version,
                        opt_level=b.identifier.opt_level,
                    ),
                    phase_id=_PHASE_ID,
                    type_id=_TYPE_ID,
                )

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


@dataclass
class _ScanFilters:
    """Pre-compiled per-walk filter state read by `TokenizerTask.visit`.

    Replaces the field-by-field args munging that today's
    `find_matching_binaries` does inline. Compiled once per
    `discover_items` call; the visitor reads it for every directory
    without re-deriving regexes.
    """

    binary_format: object
    platforms: list[str]
    compiler: str | None
    compiler_versions: list[str] | None
    normalized_opt_levels: list[str] | None
    exclude_pattern: Pattern[str] | None


def _build_filters(config) -> _ScanFilters:
    """Compile per-walk filter state from a `SelectionConfig`.

    Mirrors the field-regex / opt-level normalisation / exclude-pattern
    logic in `dynamic_runner._shared.binary_selector.find_matching_binaries`.
    Refactor candidate: hoist this into a framework helper so any task
    using the standard `platform-compiler-version-opt-binary` filename
    format can share the compilation step instead of re-implementing.
    """
    field_regexes = build_field_regexes(
        platforms=config.platforms,
        compilers=[config.compiler] if config.compiler else None,
        versions=config.compiler_versions,
        opt_levels=config.opt_levels,
        version_regex=config.version_regex,
        opt_regex=config.opt_regex,
        name_regex=config.name_regex,
    )
    binary_format = build_binary_filename_format(
        config.file_format, field_regexes
    )

    normalized_opt_levels: list[str] | None = None
    if config.opt_levels:
        opt_pattern = config.opt_regex if config.opt_regex else r"[oO]([0123s])"
        opt_re = re.compile(opt_pattern)
        has_subgroup = "(" in opt_pattern and ")" in opt_pattern
        normalized_opt_levels = []
        for opt in config.opt_levels:
            match = opt_re.fullmatch(opt)
            if match:
                if has_subgroup and len(match.groups()) > 0:
                    normalized_opt_levels.append("O" + match.group(1))
                else:
                    normalized_opt_levels.append(match.group(0))
            else:
                normalized_opt_levels.append(opt)

    exclude_pattern: Pattern[str] | None = None
    if config.exclude_subfolders:
        exclude_pattern = re.compile(
            "(" + "|".join(config.exclude_subfolders) + ")"
        )

    return _ScanFilters(
        binary_format=binary_format,
        platforms=config.platforms,
        compiler=config.compiler,
        compiler_versions=config.compiler_versions,
        normalized_opt_levels=normalized_opt_levels,
        exclude_pattern=exclude_pattern,
    )
