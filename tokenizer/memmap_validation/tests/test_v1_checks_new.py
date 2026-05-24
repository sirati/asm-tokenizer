"""Validator pad-consistency + record-bounds checks.

Mirrors ``test_validator_v1_checks`` for the two new invariants added to
``_v1_checks.run_v1_post_checks``: ``check_pad_consistency`` (the
writer's ``(pre_pad, post_pad)`` layout matches the rule the reader
derives from the parsed header) and ``check_record_bounds`` (no record
extends past EOF).

The "passes on freshly built corpus" tests reuse the same minimal
``unify_vocab`` + ``build_memmap_files`` pipeline as the sibling test
file (the synthetic corpus has zero functions, so the checks short-
circuit on empty ``starts``; this still proves the wiring + dispatcher
contract holds). The "flags" tests hand-craft a one-record corpus via
``write_function_binary_data`` so the per-record path is exercised even
when the synthetic builder produces no functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from tokenizer.aligned_data._writers import (
    write_function_binary_data,
    write_index_entry,
)
from tokenizer.aligned_data.index_format import (
    read_index_arrays,
    write_index_prelude,
)
from tokenizer.memmap_validation._v1_checks import (
    check_pad_consistency,
    check_record_bounds,
)

from ._pipeline import build_pipeline as _build_pipeline


def _pipeline(tmp_path: Path) -> Path:
    """Lay down a clean v1 corpus and return ``output_dir``."""
    _, _, output_dir = _build_pipeline(tmp_path)
    return output_dir


def _starts_or_empty(index_path: Path) -> np.ndarray:
    """Read per-record starts from a v1 ``_index.bin``, or an empty array."""
    if not index_path.exists() or index_path.stat().st_size == 0:
        return np.zeros(0, dtype=np.int64)
    arr = read_index_arrays(index_path)
    if arr is None:
        return np.zeros(0, dtype=np.int64)
    return arr


def _write_one_record_corpus(data_path: Path, index_path: Path) -> Tuple[int, int]:
    """Lay down one synthetic record + matching v1 index entry.

    Uses the production writer so the on-disk bytes are byte-identical
    to a real build (self-describing record header + 16-byte
    alignment). Returns ``(record_start, record_total_size)`` so the
    truncation test below can derive ``expected_end`` without a second
    decode.
    """
    # 17-byte insn + 0 block + 4 tokens (8 bytes) drives the ultrashort
    # form (insn<64, block<256, tokens<256) with a comfortably large
    # body so the truncation test below can still lop bytes off the
    # tail and leave the header readable for the per-record probe.
    insn = np.arange(17, dtype=np.uint8)
    block = np.zeros(0, dtype=np.uint8)
    tokens = np.arange(4, dtype=np.uint16)
    with open(data_path, "wb") as fh:
        result = write_function_binary_data(fh, tokens, block, insn, entry_idx=0)
    assert result is not None
    start, total = result

    with open(index_path, "wb") as fh:
        write_index_prelude(fh)
        write_index_entry(fh, start)
    return start, total


# ---------------------------------------------------------------------------
# check_pad_consistency
# ---------------------------------------------------------------------------


def test_check_pad_consistency_passes_on_freshly_built_corpus(tmp_path: Path) -> None:
    """Clean v1 corpus: pad layout matches the rule on both arms.

    The synthetic corpus produces zero functions, so the per-record
    iteration short-circuits on ``len(starts) == 0``; the assertion is
    that the check returns ``[]`` (no false positives + the wiring
    handles empty arms cleanly).
    """
    output_dir = _pipeline(tmp_path)
    for arm in ("", "_unmatched"):
        data_path = output_dir / f"demo{arm}_data.bin"
        index_path = output_dir / f"demo{arm}_index.bin"
        starts = _starts_or_empty(index_path)
        errors = check_pad_consistency(data_path, starts, str(data_path))
        assert errors == [], f"{arm or 'matched'} arm dirty on fresh build: {errors!r}"


def test_check_pad_consistency_flags_helper_regression(
    tmp_path: Path, monkeypatch
) -> None:
    """If ``derive_pad_placement`` drifts from the rule, the check fires.

    The check is a regression guard on
    :func:`derive_pad_placement` itself -- the inline rule
    re-derivation encodes the plan §2 placement verbatim, so any
    helper edit that breaks the rule (without simultaneously editing
    the inline derivation here) trips the check on every record.
    Monkeypatching the helper on the validator's import path simulates
    that drift; no on-disk tampering required.
    """
    data_path = tmp_path / "tiny_data.bin"
    index_path = tmp_path / "tiny_index.bin"
    record_start, _ = _write_one_record_corpus(data_path, index_path)

    import tokenizer.memmap_validation._v1_checks as checks_mod

    def _buggy_placement(header):
        # Swap the canonical (pre_pad, post_pad) split: pretend a
        # 4-byte pre-pad lives where the real rule has zero. The
        # inline re-derivation will compute (0, 0) for our ultrashort
        # zero-pad record and flag the mismatch.
        return (4, 0)

    monkeypatch.setattr(checks_mod, "derive_pad_placement", _buggy_placement)

    starts = _starts_or_empty(index_path)
    errors = checks_mod.check_pad_consistency(data_path, starts, str(data_path))
    assert any(
        f"start={record_start}" in e and "pad split" in e and "disagrees with rule" in e
        for e in errors
    ), f"helper-regression should surface, got: {errors!r}"


# ---------------------------------------------------------------------------
# check_record_bounds
# ---------------------------------------------------------------------------


def test_check_record_bounds_passes_on_freshly_built_corpus(tmp_path: Path) -> None:
    """Clean v1 corpus: every record fits within ``_data.bin``."""
    output_dir = _pipeline(tmp_path)
    for arm in ("", "_unmatched"):
        data_path = output_dir / f"demo{arm}_data.bin"
        index_path = output_dir / f"demo{arm}_index.bin"
        starts = _starts_or_empty(index_path)
        errors = check_record_bounds(data_path, starts, str(data_path))
        assert errors == [], f"{arm or 'matched'} arm dirty on fresh build: {errors!r}"


def test_check_record_bounds_flags_truncated_data(tmp_path: Path) -> None:
    """Truncate ``_data.bin`` past the last record; ``check_record_bounds`` fires."""
    data_path = tmp_path / "tiny_data.bin"
    index_path = tmp_path / "tiny_index.bin"
    record_start, record_total = _write_one_record_corpus(data_path, index_path)

    original_size = data_path.stat().st_size
    # Lop off one full 16-byte alignment unit so the last record's
    # claimed end exceeds file_size by exactly that amount. For the
    # one-record corpus this truncates the only record's tail.
    raw = data_path.read_bytes()[: original_size - 16]
    data_path.write_bytes(raw)
    new_size = data_path.stat().st_size

    starts = _starts_or_empty(index_path)
    errors = check_record_bounds(data_path, starts, str(data_path))
    expected_end = record_start + record_total
    assert any(
        f"start={record_start}" in e
        and f"file_size={new_size}" in e
        and f"extends to {expected_end}" in e
        for e in errors
    ), f"truncated data should surface, got: {errors!r}"
