"""Stable identifier for a tokenizer task.

Five fields uniquely name an angr/ghidra tokenization run: the binary
itself, the build platform, the compiler, the version, and the
optimisation level. The runner only needs an opaque hashable string per
work item; `identifier_key()` produces the canonical "<binary_name>/...
/<opt_level>" form that Rust ingests as `Arc<str>`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerIdentifier:
    binary_name: str
    platform: str
    compiler: str
    version: str
    opt_level: str

    def identifier_key(self) -> str:
        """Canonical string form, suitable for use as a `RunnerIdentifier`.

        Path-style join to match the on-disk layout the worker produces
        (`<platform>/<compiler>/<version>/<opt_level>/<binary_name>`).
        """
        return f"{self.binary_name}/{self.platform}/{self.compiler}/{self.version}/{self.opt_level}"

    def __str__(self) -> str:
        return self.identifier_key()
