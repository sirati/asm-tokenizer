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
from pathlib import Path

from shared import BinaryInfo

from dynamic_runner.task_protocol import PhaseSpec, TaskTypeSpec, TypeId


_PHASE_ID = "tokenize"
_TYPE_ID = "tokenizer"


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
    ) -> Iterable[BinaryInfo]:
        """Yield binaries discovered under `source_dir`, grouped by
        binary name with intra-group size-DESC ordering preserved.

        The old `organize_and_sort_items` did this re-ordering after
        the framework's file scan; the new design folds the scan and
        the ordering into one method.
        """
        # The scan helper lives in the framework; importing lazily so
        # the file still parses without dynamic_runner installed.
        from dynamic_runner._shared.binary_info import format_size  # noqa: F401  (proves import works)
        # Real scan: delegate to the framework's matching helper. We
        # don't have the full filter args here in the stub; in
        # production this method receives `args` populated with
        # whatever `add_task_arguments` declared and the framework's
        # standard `--platforms / --compilers / ...` filters.
        items: list[BinaryInfo] = list(_collect_unsorted(source_dir, args))

        # Group by binary_name; sort within group by size DESC; order
        # groups by group-average size DESC. Same shape as the old
        # `organize_and_sort_items`; now lives inside discovery.
        groups: dict[str, list[BinaryInfo]] = defaultdict(list)
        for binary in items:
            groups[binary.binary_name].append(binary)

        group_averages: list[tuple[str, float, list[BinaryInfo]]] = []
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

    def estimate_memory(self, item: BinaryInfo) -> int:
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
        self, type_id: TypeId, item: BinaryInfo
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


def _collect_unsorted(
    source_dir: Path, args: Namespace
) -> Iterable[BinaryInfo]:
    """Stub file scanner.

    Production code would call into `dynamic_runner._shared`'s
    `find_matching_binaries` helper to apply standard filters; the
    stub returns nothing because this task is a fixture, not an
    actual production tokenizer.
    """
    return ()
