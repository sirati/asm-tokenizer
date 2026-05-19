"""Test-support helpers for ``AlignedDataLoader`` tests.

Two concerns:

1. **Unified-vocab staging.** Build a v2 or v3 ``unified_vocab.csv`` in
   a tmpdir using the real ``save_vocabulary`` writer — same path the
   production unifier uses, so the gate test exercises the real on-disk
   shape.
2. **Stub ``BinaryDataset`` + session spy.** Records every session
   ``enter``/``exit`` on a shared ledger list so tests can assert on
   per-binary batching without touching real on-disk data.

The stub-BinaryDataset approach decouples these tests from the sibling
subtasks that own the real ``BinaryDataset`` shell and ``BinarySession``.
Until those land, the real ``BinaryDataset`` does not yet expose
``open_session()`` or accept a ``vocab_manager`` kwarg; the stub
documents the receiver contract the loader expects.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from tokenizer.aligned_data.loader import aligned_data_loader as ldr_mod
from tokenizer.token_manager import VocabularyManager
from tokenizer.vocab_unifier.saver import save_vocabulary


# ---------------------------------------------------------------------------
# Vocab staging
# ---------------------------------------------------------------------------
def _write_unified_csv(vm: VocabularyManager, csv_path: Path) -> None:
    """Mirror the writer call inside ``unify_vocab`` — the unified file is
    exactly one row (the vocab def), written via a single
    ``csv.writer.writerow`` call. No header line.
    """
    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh)
        save_vocabulary(vm, writer)


def stage_v2_unified_vocab(tmp_path: Path) -> Path:
    """Write a v2-shaped ``unified_vocab.csv`` into ``tmp_path``."""
    vm = VocabularyManager(platform=None, format_version=2)
    vm.Block_V2(0)
    vm.Block_V2(1)
    csv_path = tmp_path / "unified_vocab.csv"
    _write_unified_csv(vm, csv_path)
    return csv_path


def stage_v3_unified_vocab(tmp_path: Path) -> Path:
    """Write a v3-shaped ``unified_vocab.csv`` into ``tmp_path``."""
    vm = VocabularyManager(platform=None, format_version=3)
    vm.Variant_Axis("arch:x64")
    vm.Variant_Axis("opt:O2")
    vm.Block_V2(0)
    csv_path = tmp_path / "unified_vocab.csv"
    _write_unified_csv(vm, csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Stub BinaryDataset + session spy
# ---------------------------------------------------------------------------
class SessionSpy:
    """Records loads; returns canned function objects.

    Each ``load_matched(idx)`` / ``load_unmatched(idx)`` call returns a
    cheap tuple identifying the request so the test can assert on order
    and grouping without pulling in real ``FunctionData`` semantics.
    """

    def __init__(self, ledger: List[str], binary_name: str) -> None:
        self._ledger = ledger
        self._name = binary_name
        self.loads: List[int] = []

    def load_matched(self, idx: int) -> object:
        self.loads.append(idx)
        return ("matched", self._name, int(idx))

    def load_unmatched(self, idx: int) -> object:
        self.loads.append(idx)
        return ("unmatched", self._name, int(idx))


class BinaryDatasetStub:
    """Stub matching the surface ``AlignedDataLoader`` consumes.

    Exposes everything the loader touches:

    * ``matched_count`` / ``unmatched_count`` for ``get_statistics``
    * ``matched_count_per_length`` for the loader's length-sampling branch
    * ``get_matched_indices_by_length`` / ``get_matched_indices_in_range``
    * ``get_unmatched_indices_in_range``
    * ``open_session()`` context manager whose entries/exits are recorded
      on the shared session-ledger list.
    """

    def __init__(
        self,
        binary_name: str,
        matched_indices: List[int],
        unmatched_indices: List[int],
        matched_length: int,
        session_ledger: List[str],
        vocab_manager: Optional[VocabularyManager] = None,
    ) -> None:
        self.binary_name = binary_name
        self._matched = list(matched_indices)
        self._unmatched = list(unmatched_indices)
        self._matched_length = matched_length
        self._session_ledger = session_ledger
        self.vocab_manager = vocab_manager
        self.matched_count = len(self._matched)
        self.unmatched_count = len(self._unmatched)
        # Match real ``BinaryDataset``: ``matched_count_per_length[L]`` is
        # the count of matched functions whose token length is ``L``. All
        # matched stub functions share ``matched_length``.
        self.matched_count_per_length = np.zeros(matched_length + 1, dtype=np.int32)
        self.matched_count_per_length[matched_length] = len(self._matched)

    # ---- index lookups -------------------------------------------------
    def get_matched_indices_in_range(self, min_len: int, max_len: int) -> np.ndarray:
        if min_len <= self._matched_length <= max_len:
            return np.array(self._matched, dtype=np.int32)
        return np.array([], dtype=np.int32)

    def get_unmatched_indices_in_range(self, min_len: int, max_len: int) -> np.ndarray:
        return np.array(self._unmatched, dtype=np.int32)

    def get_matched_indices_by_length(self, target_length: int, min_count: int = 1) -> np.ndarray:
        if target_length == self._matched_length:
            return np.array(self._matched, dtype=np.int32)
        return np.array([], dtype=np.int32)

    # ---- session lifecycle --------------------------------------------
    @contextmanager
    def open_session(self):
        self._session_ledger.append(f"enter:{self.binary_name}")
        spy = SessionSpy(self._session_ledger, self.binary_name)
        try:
            yield spy
        finally:
            self._session_ledger.append(f"exit:{self.binary_name}")


def patch_binary_dataset_factory(
    monkeypatch: pytest.MonkeyPatch,
    layouts: Dict[str, Dict],
    session_ledger: List[str],
) -> None:
    """Patch ``BinaryDataset`` in the loader module so construction returns
    the stub configured for the given binary name.

    ``layouts`` maps ``binary_name -> {"matched": [...], "unmatched": [...],
    "matched_length": int}``.
    """
    def _ctor(base_path, binary_name, vocab_manager=None):
        if binary_name not in layouts:
            raise KeyError(f"no stub layout configured for {binary_name!r}")
        cfg = layouts[binary_name]
        return BinaryDatasetStub(
            binary_name=binary_name,
            matched_indices=cfg["matched"],
            unmatched_indices=cfg["unmatched"],
            matched_length=cfg.get("matched_length", 32),
            session_ledger=session_ledger,
            vocab_manager=vocab_manager,
        )

    monkeypatch.setattr(ldr_mod, "BinaryDataset", _ctor)
