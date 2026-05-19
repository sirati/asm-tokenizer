"""Per-version (tokenizer-output) CSV decoders + skip predicates.

Single concern: turn one ``lockstep_function_match`` row into the
``(tokens, block_runlength, insn_runlength)`` triple the per-arm
comparators expect, plus the small predicate set the orchestrator
uses to gate which rows are worth validating.

Lives in its own module so the orchestrator (``validator.py``) stays
focused on flow control and the comparators do not have to know
where the decode helpers come from.

The per-version CSV format detection (v1 vs. v2 prelude) lives here
too -- it consumes the same wire format. Prelude *consumption*
during data iteration stays with
``aligned_data.match.open_csv_skip_vocab``; this module only
*peeks* the first row to surface a format mismatch before iteration.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from tokenizer.compact_base64_utils import base64_to_ndarray_vec


def detect_csv_format_version(csv_path: Path) -> int:
    """Return the wire-format version of a per-version tokenizer CSV.

    Peeks only the first CSV row. v2 files start with a single-cell
    ``["version=2"]`` prelude; v1 files start directly with the header.
    Raises ``ValueError`` for an empty file or an unrecognised prelude
    payload so a corrupt input is surfaced rather than silently treated
    as v1.

    Prelude *consumption* during data iteration is handled by
    ``aligned_data.match.open_csv_skip_vocab``; this helper deliberately
    only peeks (it does not advance any shared reader state).
    """
    with open(csv_path, "r", newline="", encoding="ascii") as f:
        reader = csv.reader(f)
        try:
            first_row = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty CSV: {csv_path}") from exc

    if len(first_row) == 1 and first_row[0].startswith("version="):
        payload = first_row[0][len("version=") :]
        if payload == "2":
            return 2
        raise ValueError(f"unrecognised CSV version prelude {first_row[0]!r} in {csv_path}")

    # No prelude row: v1 by construction. The first row is the header.
    return 1


def load_mapping(mapping_path: Path) -> Optional[np.ndarray]:
    """Load mapping file if it exists."""
    if mapping_path and mapping_path.exists():
        with open(mapping_path, "r", encoding="ascii") as f:
            return base64_to_ndarray_vec(f.read())
    return None


def decode_csv_row_data(
    row: dict, mapping: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode tokens and runlengths from CSV row."""
    tokens = base64_to_ndarray_vec(row["tokens_base64"])
    if mapping is not None:
        tokens = mapping[tokens]
    tokens = tokens.astype(np.uint16)

    block_runlength = base64_to_ndarray_vec(row["block_runlength_base64"])
    insn_runlength = base64_to_ndarray_vec(row["instruction_runlength_base64"])

    return tokens, block_runlength, insn_runlength


def should_skip_matched_function(rows: List[Optional[dict]]) -> bool:
    """Check if matched function should be skipped based on builder logic."""
    for row in rows:
        if row is not None and "block_runlength_base64" in row:
            try:
                block_runlength = base64_to_ndarray_vec(row["block_runlength_base64"])
                if block_runlength.sum() >= 4096:
                    return True
            except Exception:
                return True
    return False


def should_skip_unmatched_function(row: dict) -> bool:
    """Check if unmatched function should be skipped based on builder logic.

    Note: The builder checks for 'block_runlength' (not 'block_runlength_base64'),
    which doesn't exist in the CSV, so it never actually skips unmatched functions.
    This matches that behavior.
    """
    return False


def has_unique_offsets(version_data_list: List[dict]) -> bool:
    """Check if versions have unique binary offsets (would not be deduplicated)."""
    unique_offsets = set()
    for vdata in version_data_list:
        cache_key = (
            vdata["tokens"].tobytes(),
            vdata["block_rl"].tobytes(),
            vdata["insn_rl"].tobytes(),
        )
        unique_offsets.add(cache_key)
    return len(unique_offsets) > 1
