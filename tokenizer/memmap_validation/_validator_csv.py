"""Validator helpers that survived the parsed-record refactor.

Row decoding now lives on the per-CSV iterator
(:mod:`tokenizer.aligned_data.parsed_record_iter`); the validator
consumes :class:`ParsedRecord` objects directly. What's left here:

* Format detection (v1 vs. v2 prelude) — peeks the first row to
  surface a format mismatch before iteration. Prelude *consumption*
  during data iteration stays with
  ``aligned_data.match.open_csv_skip_vocab``.
* Per-variant mapping loader (used by the validator's setup step).
* Skip predicates mirroring the pass-1 walkers' gating.
* Dedup-key heuristic for unique-offset detection across variants of
  the same function (matched arm's drop gate).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tokenizer.aligned_data.parsed_record_iter import ParsedRecord
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


def should_skip_matched_function(records: Dict[int, ParsedRecord]) -> bool:
    """``True`` when any variant's block runlength reaches the cap.

    Mirrors the pass-1 matched walker's gate
    (:func:`tokenizer.memmap_builder.helpers.should_skip_for_matched`)
    applied across every variant; one over-cap variant drops the
    whole function from the matched arm.
    """
    for rec in records.values():
        if int(rec.block_runlength.sum()) >= 4096:
            return True
    return False


def should_skip_unmatched_function(_block_runlength: np.ndarray) -> bool:
    """No-op preserved from the pre-refactor walker (see helpers.py)."""
    return False


def has_unique_offsets(version_data_list: List[dict]) -> bool:
    """Check if variants have unique encoded bodies (would not be deduplicated).

    The matched arm drops a function when every variant encodes to the
    same byte sequence (``len(unique_offsets) == 1`` at
    ``passes.process_matched_function``). This predicate predicts that
    outcome from the per-variant ndarrays alone so the validator can
    skip the matched comparison when the builder would have rerouted
    the data to the unmatched arm.
    """
    unique_keys = set()
    for vdata in version_data_list:
        cache_key = (
            vdata["tokens"].tobytes(),
            vdata["block_rl"].tobytes(),
            vdata["insn_rl"].tobytes(),
        )
        unique_keys.add(cache_key)
    return len(unique_keys) > 1
