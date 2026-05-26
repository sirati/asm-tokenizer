"""Per-binary ``<binary>_function_ranges.txt`` sidecar — wire format owner.

Debug-feature sidecar that mirrors the per-binary output CSV row-by-row
for FUNCTION rows only. Each line records the function's address range
``<func_min_addr_hex>,<func_max_addr_hex>`` so a downstream verifier
can co-step the two files and cross-reference any
``valued_const_v2:<addr>`` in the CSV's token stream against the
hosting function's body extent.

Co-stepping semantics
---------------------
One line per FUNCTION row written to ``<binary>_output.csv``. The CSV
also carries non-function rows (the ``version=2`` prelude, the header,
periodic ``save_vocabulary`` snapshots interleaved every 16384
functions, and the trailing vocab snapshot). The sidecar mirrors only
FUNCTION rows; the verifier walks the CSV and advances the sidecar by
one line per function-row encountered (skipping the non-function rows
in the CSV).

Wire format
-----------
First line: ``# format=1`` (comment-prefixed header so naive readers
can skip it; bumped when a future schema change lands).
Subsequent lines: ``<func_min:x>,<func_max:x>`` — lowercase hex, no
``0x`` prefix, comma-separated, one entry per FUNCTION row, in the
same order. Newline-terminated (``\\n``).

The address values are inherently binary-specific (different VMA on
x86 vs arm); this sidecar is a debugging aid WITHIN one binary's
tokenize output, not a cross-ISA aligned key.

This module is the *single* place that knows the on-disk format. The
tokenizer pipeline (``main_loop.py``) opens the sidecar alongside the
CSV, calls :py:meth:`FunctionRangeSidecar.add` once per FUNCTION row
written, and closes it at the bottom of the per-binary task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, TextIO


# Header line written exactly once on construction. Format version is
# bumped only on a wire-format-breaking change; readers should consume
# this line and ignore any other ``#``-prefixed lines for forward
# compatibility.
_HEADER_LINE = "# format=1\n"


class FunctionRangeSidecar:
    """Writer for the per-binary ``<binary>_function_ranges.txt`` sidecar.

    Opens the file in text write mode (truncating any existing contents)
    on construction and writes the ``# format=1`` header; flushes and
    closes on :py:meth:`close` / context-manager exit.

    Each :py:meth:`add` call appends one line recording the function's
    address range as ``<func_min_addr:x>,<func_max_addr:x>``. No dedup —
    a function appearing twice in the CSV (legacy occurrence-bumped
    second body) MUST also appear twice in the sidecar so the
    row-by-row co-stepping invariant holds.

    The caller is responsible for skipping :py:meth:`add` on
    dedup-folded functions (the CSV write was also skipped in that
    case; the sidecar mirrors the CSV).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        # Text-mode write: hex digits + ``,`` + ``\n`` are all 7-bit
        # ASCII; no escape table needed (the line separator can never
        # appear inside a line by construction).
        self._fh: TextIO | None = path.open("w", encoding="ascii", newline="\n")
        self._fh.write(_HEADER_LINE)
        self._line_count: int = 0

    def add(self, func_min_addr: int, func_max_addr: int) -> None:
        """Append one row recording the function's address range.

        ``func_min_addr`` is the function entry address; ``func_max_addr``
        is the function's body upper bound (max of ``block.addr +
        block.size`` across all blocks). Both values are written in
        lowercase hex without a ``0x`` prefix.

        Negative addresses are rejected — VMAs are unsigned.
        """
        if self._fh is None:
            raise ValueError("FunctionRangeSidecar is closed")
        if func_min_addr < 0 or func_max_addr < 0:
            raise ValueError(
                f"negative address in sidecar entry: "
                f"min={func_min_addr}, max={func_max_addr}"
            )
        self._fh.write(f"{func_min_addr:x},{func_max_addr:x}\n")
        self._line_count += 1

    @property
    def line_count(self) -> int:
        """Number of data lines written (excluding the header)."""
        return self._line_count

    def close(self) -> None:
        """Flush and close the underlying file. Idempotent."""
        if self._fh is None:
            return
        self._fh.flush()
        self._fh.close()
        self._fh = None

    def __enter__(self) -> "FunctionRangeSidecar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def iter_sidecar_lines(path: Path) -> Iterator[tuple[int, int]]:
    """Yield each ``(func_min_addr, func_max_addr)`` pair in order.

    Skips the header line and any other ``#``-prefixed line (forward
    compatibility with future schema-extension comments). Reads in text
    mode; parses each data line as ``<min_hex>,<max_hex>``.
    """
    with path.open("r", encoding="ascii") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            min_str, sep, max_str = line.partition(",")
            if not sep:
                raise ValueError(
                    f"malformed sidecar line (missing comma): {line!r}"
                )
            yield int(min_str, 16), int(max_str, 16)
