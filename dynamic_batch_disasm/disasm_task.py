"""Stub TaskDefinition for a hypothetical disassembler task.

The point isn't to actually disassemble — it's to demonstrate that the
runner accepts a task with a different identifier shape, different
phase set, different memory model, and a different worker module
without any change to `rust/` or `dynamic_batch/`.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from shared import BinaryInfo

from dynamic_batch.task_protocol import Phase, StageDefinition


class DisasmPhase(Phase):
    DISASSEMBLE = "disassemble"
    SYMBOL_EXTRACT = "symbol-extract"


class DisasmTask:
    """Implements the dynamic_batch.task_protocol.TaskDefinition shape
    structurally — no inheritance required.
    """

    def get_stages(self) -> list[StageDefinition]:
        return [
            StageDefinition(phase=DisasmPhase.DISASSEMBLE, timeout_seconds=None),
            StageDefinition(phase=DisasmPhase.SYMBOL_EXTRACT, timeout_seconds=30.0),
        ]

    def organize_and_sort_items(self, items: list[BinaryInfo]) -> list[BinaryInfo]:
        # Smallest first — opposite of the tokenizer, just to demonstrate
        # tasks pick their own ordering.
        return sorted(items, key=lambda b: b.size)

    def estimate_memory(self, binary_size: int) -> int:
        # Constant 256 MiB per binary, regardless of size. Different
        # model than the tokenizer's power-law estimator.
        return 256 * 1024 * 1024

    def get_worker_module(self) -> str:
        return "disasm_worker"

    def add_task_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--symbol-format",
            choices=["dwarf", "stabs"],
            default="dwarf",
            help="symbol table format the worker should emit",
        )

    def build_worker_command_args(
        self,
        args: Namespace,
        source_dir: Path,
        output_dir: Path,
        skip_existing: bool,
    ) -> list[str]:
        return ["--symbol-format", args.symbol_format]

    def get_output_filename_pattern(self, input_filename: str) -> str:
        return f"{input_filename}.disasm"

    def get_reserved_memory_per_worker(self) -> int:
        return 128 * 1024 * 1024
