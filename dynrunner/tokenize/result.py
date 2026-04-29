"""Per-task tokenizer result.

Wire format on the `result_data` channel: ASCII `b"<warnings>:<filtered>"`,
matching the legacy `done:N:M` shape. Both numbers are u32-range
non-negative ints. Decoding is forgiving — missing or unparseable
fields fall back to zero so a partially-broken worker doesn't take down
the run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerResult:
    warnings: int
    filtered: int

    def encode(self) -> bytes:
        return f"{self.warnings}:{self.filtered}".encode("ascii")

    @classmethod
    def decode(cls, data: bytes) -> "TokenizerResult":
        try:
            text = data.decode("ascii")
            parts = text.split(":", 1)
            warnings = int(parts[0]) if parts and parts[0].isdigit() else 0
            filtered = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        except UnicodeDecodeError:
            warnings, filtered = 0, 0
        return cls(warnings=warnings, filtered=filtered)
