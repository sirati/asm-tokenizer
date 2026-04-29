"""TaskDefinition stub for a hypothetical disassembler task.

The point isn't to actually disassemble — it's to demonstrate that the
runner accepts a task with a different identifier shape, different
memory model, and a different worker module without any change to the
`dynamic_runner` package.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from pathlib import Path

from shared import BinaryInfo

from dynamic_runner.task_protocol import PhaseSpec, TaskTypeSpec, TypeId


_PHASE_ID = "disassemble"
_TYPE_ID = "disasm"


class DisasmTask:
    """Implements `dynamic_runner.task_protocol.TaskDefinition` structurally
    — no inheritance required."""

    # ── Topology ───────────────────────────────────────────────────────

    def get_phases(self) -> tuple[PhaseSpec, ...]:
        return (
            PhaseSpec(
                phase_id=_PHASE_ID,
                types=(
                    TaskTypeSpec(
                        type_id=_TYPE_ID,
                        worker_module="disasm_worker",
                        # The old code had a 30s timeout on the
                        # SYMBOL_EXTRACT stage. With one consolidated
                        # type the worker reports its internal
                        # progress via keepalive; choose the tighter
                        # of the two old timeouts.
                        timeout_seconds=30.0,
                        reserved_memory_per_worker=128 * 1024 * 1024,
                    ),
                ),
            ),
        )

    # ── Item discovery ─────────────────────────────────────────────────

    def discover_items(
        self, source_dir: Path, args: Namespace
    ) -> Iterable[BinaryInfo]:
        """Scan `source_dir` for binaries; tag each with this task's
        single phase + type. The actual file scan is whatever the
        existing disasm-task pipeline produced before — for the
        stub task this is a noop placeholder."""
        # Stub task: this is a hypothetical disassembler that doesn't
        # actually scan anything. Real consumers replace this with their
        # own scan (e.g. via `dynamic_runner._shared.find_matching_binaries`).
        return ()

    # ── Per-type plumbing ──────────────────────────────────────────────

    def estimate_memory(self, item: BinaryInfo) -> int:
        # Constant 256 MiB per binary. Receives the full item now
        # (was: just `binary_size: int`).
        return 256 * 1024 * 1024

    def add_task_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--symbol-format",
            choices=["dwarf", "stabs"],
            default="dwarf",
            help="symbol table format the worker should emit",
        )

    def build_worker_command_args(
        self,
        type_id: TypeId,
        args: Namespace,
        source_dir: Path,
        output_dir: Path,
        skip_existing: bool,
    ) -> list[str]:
        return ["--symbol-format", args.symbol_format]

    def get_output_filename_pattern(
        self, type_id: TypeId, item: BinaryInfo
    ) -> str:
        # `item.path.name` is the binary filename.
        return f"{item.path.name}.disasm"

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
