"""Tests for the unified-vocab gate's memmap-format-version enforcement.

Decoupled from the actual unifier/loader plumbing by patching
``load_unified_vocab_manager`` at the gate's import site: the unit under
test is the gate's version-check policy, NOT the wire-format loader.
A future bump of ``MEMMAP_FORMAT_VERSION`` should not change these tests
beyond the constant they import — that's the whole point of routing the
gate through a single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tokenizer.aligned_data.loader import unified_vocab_gate
from tokenizer.aligned_data.loader.unified_vocab_gate import (
    REQUIRED_UNIFIED_VOCAB_FORMAT_VERSION,
    load_and_validate_unified_vocab,
    resolve_unified_vocab_path,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION


def _stub_vocab_manager(format_version: int) -> SimpleNamespace:
    """A duck-typed stand-in for ``VocabularyManager``.

    The gate only consults ``.format_version`` for its accept/reject
    decision; the rest of the VM surface is irrelevant here.
    """
    return SimpleNamespace(format_version=format_version)


def _touch(path: Path) -> None:
    """Create a zero-byte file so the gate's existence check passes."""
    path.write_bytes(b"")


def test_required_version_equals_memmap_format_version() -> None:
    """The gate's required version is the single-source-of-truth constant.

    A future bump of ``MEMMAP_FORMAT_VERSION`` must cascade to the gate
    without a separate touch; this guards against accidental drift.
    """
    assert REQUIRED_UNIFIED_VOCAB_FORMAT_VERSION == MEMMAP_FORMAT_VERSION


def test_v1_vocab_passes(tmp_path: Path) -> None:
    """A vocab whose ``format_version`` matches the constant is accepted."""
    vocab_path = tmp_path / "unified_vocab.csv"
    _touch(vocab_path)
    stub = _stub_vocab_manager(format_version=MEMMAP_FORMAT_VERSION)

    with patch(
        "tokenizer.vocab_unifier.loader.load_unified_vocab_manager", return_value=stub
    ) as mocked:
        result = load_and_validate_unified_vocab(vocab_path)

    assert result is stub
    mocked.assert_called_once_with(vocab_path)


@pytest.mark.parametrize("bad_version", [0, 2, 3, 4, 99])
def test_non_v1_vocab_raises(tmp_path: Path, bad_version: int) -> None:
    """Any non-current-version vocab is rejected with a typed ValueError.

    Covers legacy v2 (per-binary CSV format mistakenly fed as unified),
    legacy v3 (the variant-axis unified vocab that v1 supersedes), and
    arbitrary other values. The reader has NO knowledge of specific old
    version numbers — they all fail through the same branch.
    """
    if bad_version == MEMMAP_FORMAT_VERSION:
        pytest.skip("parameter collides with the current required version")

    vocab_path = tmp_path / "unified_vocab.csv"
    _touch(vocab_path)
    stub = _stub_vocab_manager(format_version=bad_version)

    with patch(
        "tokenizer.vocab_unifier.loader.load_unified_vocab_manager", return_value=stub
    ):
        with pytest.raises(ValueError) as excinfo:
            load_and_validate_unified_vocab(vocab_path)

    message = str(excinfo.value)
    assert f"format_version={bad_version}" in message, message
    assert f"v{MEMMAP_FORMAT_VERSION}" in message, message


def test_missing_file_raises(tmp_path: Path) -> None:
    """A non-existent vocab path raises before any loader call."""
    vocab_path = tmp_path / "does_not_exist.csv"

    with patch(
        "tokenizer.vocab_unifier.loader.load_unified_vocab_manager"
    ) as mocked:
        with pytest.raises(ValueError) as excinfo:
            load_and_validate_unified_vocab(vocab_path)

    assert "not found" in str(excinfo.value)
    assert str(vocab_path) in str(excinfo.value)
    mocked.assert_not_called()


def test_unparseable_vocab_raises(tmp_path: Path) -> None:
    """A loader that returns ``None`` (parse failure) is surfaced cleanly."""
    vocab_path = tmp_path / "unified_vocab.csv"
    _touch(vocab_path)

    with patch(
        "tokenizer.vocab_unifier.loader.load_unified_vocab_manager", return_value=None
    ):
        with pytest.raises(ValueError) as excinfo:
            load_and_validate_unified_vocab(vocab_path)

    assert "failed to parse" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Search-path policy — ``resolve_unified_vocab_path``
# ---------------------------------------------------------------------------
def test_resolve_finds_vocab_alongside_memmap_bins(tmp_path: Path) -> None:
    """Primary lookup: vocab sits in the same directory as the memmap bins."""
    memmap_dir = tmp_path / "memmap"
    memmap_dir.mkdir()
    in_dir = memmap_dir / "unified_vocab.csv"
    _touch(in_dir)

    assert resolve_unified_vocab_path(memmap_dir) == in_dir


def test_resolve_falls_back_to_parent_when_bins_in_subdir(tmp_path: Path) -> None:
    """Corpus-root layout: bins in ``<root>/memmap/``, vocab at ``<root>/``.

    This is the layout the user runs with — the memmap subdir holds the
    per-binary bins and the vocab lives one level up at the corpus root.
    Resolver must walk up one level when nothing is found alongside.
    """
    corpus_root = tmp_path / "corpus"
    memmap_dir = corpus_root / "memmap"
    memmap_dir.mkdir(parents=True)
    at_root = corpus_root / "unified_vocab.csv"
    _touch(at_root)

    assert resolve_unified_vocab_path(memmap_dir) == at_root


def test_resolve_prefers_in_directory_over_parent(tmp_path: Path) -> None:
    """When both candidates exist, the in-directory copy wins.

    An explicit per-memmap-dir vocab is the operator's signal that it
    overrides any inherited corpus-root copy. Without this priority a
    legitimate per-dir override could be silently shadowed by a stale
    parent vocab.
    """
    corpus_root = tmp_path / "corpus"
    memmap_dir = corpus_root / "memmap"
    memmap_dir.mkdir(parents=True)
    in_dir = memmap_dir / "unified_vocab.csv"
    at_root = corpus_root / "unified_vocab.csv"
    _touch(in_dir)
    _touch(at_root)

    assert resolve_unified_vocab_path(memmap_dir) == in_dir


def test_resolve_missing_lists_all_candidates(tmp_path: Path) -> None:
    """No candidate exists -> ValueError enumerating every probed path.

    Operator-facing diagnostic: the message must name both candidates so
    a copy-to-the-right-place fix is mechanical.
    """
    corpus_root = tmp_path / "corpus"
    memmap_dir = corpus_root / "memmap"
    memmap_dir.mkdir(parents=True)

    with pytest.raises(ValueError) as excinfo:
        resolve_unified_vocab_path(memmap_dir)

    message = str(excinfo.value)
    assert "unified_vocab.csv not found" in message
    assert str(memmap_dir / "unified_vocab.csv") in message
    assert str(corpus_root / "unified_vocab.csv") in message
