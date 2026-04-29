"""Identifier for the (hypothetical) disassembler task.

Demonstrates the "task-agnostic" property of the runner: a sibling
package can compose its own identifier shape (here: file_hash + arch)
without changing anything in the `dynamic_runner` package. The runner
ingests the canonical key string and treats it opaquely.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DisasmIdentifier:
    file_hash: str
    arch: str

    def identifier_key(self) -> str:
        # Different shape than the tokenizer's 5-field key — proves the
        # runner doesn't depend on a particular field count or order.
        return f"{self.arch}:{self.file_hash}"

    def __str__(self) -> str:
        return self.identifier_key()
