"""Tests for ``AlignedDataLoader``'s format-version gate and per-binary
batching.

Two concerns covered:

1. **Hard cutover gate.** A staged v2 ``unified_vocab.csv`` must raise
   ``ValueError`` before any ``BinaryDataset`` is constructed. The
   exception message must name both the actual version and the required
   version so the operator can act on a single log line.
2. **One session per binary.** Batch helpers
   (``load_matched_functions``, ``load_unmatched_functions``) must open
   ``binary.open_session()`` at most once per *touched binary*, not once
   per function. Verified via a stub ``BinaryDataset`` patched into the
   loader module that records every session entry/exit on a shared
   ledger.

Support helpers live in ``_loader_test_support`` to keep this file under
the 300 LOC project cap.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from tokenizer.aligned_data.loader.aligned_data_loader import AlignedDataLoader

from ._loader_test_support import (
    patch_binary_dataset_factory,
    stage_v2_unified_vocab,
    stage_v1_unified_vocab,
)


# ---------------------------------------------------------------------------
# Hard-cutover gate — negative tests
# ---------------------------------------------------------------------------
def test_v2_unified_vocab_raises_value_error(tmp_path: Path) -> None:
    """A v2 ``unified_vocab.csv`` must trip the format-version gate BEFORE
    any ``BinaryDataset`` is constructed.
    """
    stage_v2_unified_vocab(tmp_path)

    with pytest.raises(ValueError) as exc_info:
        AlignedDataLoader(
            base_path=tmp_path,
            binary_names=["nonexistent"],  # never reached
        )

    msg = str(exc_info.value)
    # Names both the offending version and the required version so the
    # operator can fix without grepping logs.
    assert "format_version=2" in msg
    assert "v1 required" in msg
    assert "tokenizer.vocab_unifier" in msg


def test_missing_unified_vocab_raises_value_error(tmp_path: Path) -> None:
    """The gate must fire even when the vocab file is absent — silent
    fall-through would let a stale dataset slip past.
    """
    with pytest.raises(ValueError) as exc_info:
        AlignedDataLoader(base_path=tmp_path, binary_names=[])
    assert "unified_vocab.csv" in str(exc_info.value)


def test_unified_vocab_resolved_from_parent_when_bins_in_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end resolver use: dataloader pointed at ``<root>/memmap/``
    must pick up the vocab at ``<root>/unified_vocab.csv``.

    This is the layout the user runs with — memmap bins live in a
    ``memmap/`` subdir of the corpus root, and the unified vocab sits
    one level up at the corpus root. Without the parent-dir fallback
    in the search policy this construction would raise.
    """
    corpus_root = tmp_path / "corpus"
    memmap_dir = corpus_root / "memmap"
    memmap_dir.mkdir(parents=True)
    # Stage the vocab at the corpus root (one level up from memmap_dir).
    stage_v1_unified_vocab(corpus_root)

    ledger: List[str] = []
    patch_binary_dataset_factory(
        monkeypatch,
        layouts={
            "binA": {"matched": [], "unmatched": [], "matched_length": 16},
        },
        session_ledger=ledger,
    )

    loader = AlignedDataLoader(
        base_path=memmap_dir,
        binary_names=["binA"],
    )
    assert loader.unified_vocab_path == corpus_root / "unified_vocab.csv"
    assert loader.vocab_manager.format_version == 1


def test_explicit_unified_vocab_path_overrides_default(
    tmp_path: Path,
) -> None:
    """The ``unified_vocab_path`` ctor kwarg must take precedence over the
    default ``base_path / "unified_vocab.csv"`` location. Negative test:
    a v2 vocab at the explicit path still trips the gate even when the
    default location is empty.
    """
    alt_dir = tmp_path / "alt"
    alt_dir.mkdir()
    alt_vocab = stage_v2_unified_vocab(alt_dir)

    with pytest.raises(ValueError) as exc_info:
        AlignedDataLoader(
            base_path=tmp_path,
            binary_names=[],
            unified_vocab_path=alt_vocab,
        )
    assert "format_version=2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Per-binary batching — positive tests
# ---------------------------------------------------------------------------
def test_load_matched_groups_by_binary_one_session_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Across a batch of 10 matched functions split over 2 binaries, the
    loader must open exactly 2 sessions (one per touched binary), not 10.
    """
    stage_v1_unified_vocab(tmp_path)

    ledger: List[str] = []
    patch_binary_dataset_factory(
        monkeypatch,
        layouts={
            "binA": {"matched": list(range(5)), "unmatched": [], "matched_length": 16},
            "binB": {"matched": list(range(5)), "unmatched": [], "matched_length": 16},
        },
        session_ledger=ledger,
    )

    loader = AlignedDataLoader(
        base_path=tmp_path,
        binary_names=["binA", "binB"],
        seed=42,
    )
    out = loader.load_matched_functions(n=10, target_length=16)

    assert len(out) == 10
    enters = [e for e in ledger if e.startswith("enter:")]
    exits = [e for e in ledger if e.startswith("exit:")]
    assert len(enters) == 2, (
        f"expected one session per touched binary, got enters={enters}"
    )
    assert len(exits) == 2
    assert sorted(enters) == sorted(f"enter:{b}" for b in ("binA", "binB"))
    assert sorted(exits) == sorted(f"exit:{b}" for b in ("binA", "binB"))


def test_load_unmatched_groups_by_binary_one_session_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same one-session-per-binary discipline on the unmatched path."""
    stage_v1_unified_vocab(tmp_path)

    ledger: List[str] = []
    patch_binary_dataset_factory(
        monkeypatch,
        layouts={
            "binA": {"matched": [], "unmatched": list(range(6)), "matched_length": 16},
            "binB": {"matched": [], "unmatched": list(range(6)), "matched_length": 16},
        },
        session_ledger=ledger,
    )

    loader = AlignedDataLoader(
        base_path=tmp_path,
        binary_names=["binA", "binB"],
        seed=7,
    )
    out = loader.load_unmatched_functions(n=8)

    assert len(out) == 8
    enters = [e for e in ledger if e.startswith("enter:")]
    exits = [e for e in ledger if e.startswith("exit:")]
    touched = {e.split(":", 1)[1] for e in enters}
    # 1 session per touched binary (random sample may not always touch
    # every binary, so compare to the touched set rather than the
    # binary_names list).
    assert len(enters) == len(touched), (
        f"opened {len(enters)} sessions for {len(touched)} touched binaries "
        f"(should be 1:1)"
    )
    assert sorted(enters) == sorted(f"enter:{b}" for b in touched)
    assert sorted(exits) == sorted(f"exit:{b}" for b in touched)


def test_session_count_bounded_by_touched_binaries_not_function_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sharpened invariant: load 20 functions across 3 binaries; session
    count must equal the number of distinct touched binaries (<= 3) and
    must be strictly less than the per-function count.
    """
    stage_v1_unified_vocab(tmp_path)

    ledger: List[str] = []
    patch_binary_dataset_factory(
        monkeypatch,
        layouts={
            "binA": {"matched": list(range(20)), "unmatched": [], "matched_length": 16},
            "binB": {"matched": list(range(20)), "unmatched": [], "matched_length": 16},
            "binC": {"matched": list(range(20)), "unmatched": [], "matched_length": 16},
        },
        session_ledger=ledger,
    )

    loader = AlignedDataLoader(
        base_path=tmp_path,
        binary_names=["binA", "binB", "binC"],
        seed=99,
    )
    out = loader.load_matched_functions(n=20, target_length=16)
    assert len(out) == 20

    enters = [e for e in ledger if e.startswith("enter:")]
    touched = {e.split(":", 1)[1] for e in enters}
    assert len(enters) == len(touched)
    assert len(enters) <= 3
    # Strict win over the naive per-function path.
    assert len(enters) < len(out)


def test_vocab_manager_threaded_into_binary_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loaded ``VocabularyManager`` must reach each constructed
    ``BinaryDataset`` via the ``vocab_manager`` kwarg (receiver contract
    for the shell-integrator subtask).
    """
    stage_v1_unified_vocab(tmp_path)

    ledger: List[str] = []
    patch_binary_dataset_factory(
        monkeypatch,
        layouts={
            "binA": {"matched": [], "unmatched": [], "matched_length": 16},
        },
        session_ledger=ledger,
    )

    loader = AlignedDataLoader(
        base_path=tmp_path,
        binary_names=["binA"],
    )
    stub = loader.datasets["binA"]
    assert stub.vocab_manager is loader.vocab_manager
    assert loader.vocab_manager.format_version == 1
