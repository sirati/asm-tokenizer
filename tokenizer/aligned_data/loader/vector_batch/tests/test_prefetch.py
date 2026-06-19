"""Unit tests for the page-prefetch helper (advisory ``MADV_WILLNEED``).

Covers the four pure-numpy transforms the helper owns -- page-alignment,
EOF clamping, coalescing, and the empty / non-positive skips -- using a
recording fake mmap so the exact ``(start, length)`` syscalls are
asserted (no real page-fault side effects are observable).
"""

from __future__ import annotations

import mmap
import os
import tempfile

import numpy as np

from tokenizer.aligned_data.loader.vector_batch._prefetch import (
    prefetch_willneed,
)


_PAGE = mmap.PAGESIZE


class _RecordingMM:
    """A fake mmap recording ``madvise(MADV_WILLNEED, start, length)``."""

    def __init__(self, size: int) -> None:
        self._size = int(size)
        self.calls: list[tuple[int, int]] = []

    def size(self) -> int:
        return self._size

    def madvise(self, option: int, start: int, length: int) -> None:
        assert option == mmap.MADV_WILLNEED
        self.calls.append((int(start), int(length)))


def test_page_alignment_covers_original_range():
    """An unaligned start is aligned DOWN; the range stays covered."""
    mm = _RecordingMM(size=10 * _PAGE)
    start = _PAGE + 100  # unaligned
    length = 50
    prefetch_willneed(
        mm, np.array([start], dtype=np.int64), np.array([length], dtype=np.int64)
    )
    assert len(mm.calls) == 1
    adv_start, adv_len = mm.calls[0]
    assert adv_start == _PAGE  # aligned down
    assert adv_start % _PAGE == 0
    # Original [start, start+length) still inside the advised span.
    assert adv_start <= start
    assert adv_start + adv_len >= start + length


def test_eof_clamp_does_not_overrun():
    """A length past mmap size is clamped to the size (no raise)."""
    size = 3 * _PAGE
    mm = _RecordingMM(size=size)
    start = 2 * _PAGE
    length = 10 * _PAGE  # far past EOF
    prefetch_willneed(
        mm, np.array([start], dtype=np.int64), np.array([length], dtype=np.int64)
    )
    assert len(mm.calls) == 1
    adv_start, adv_len = mm.calls[0]
    assert adv_start + adv_len == size  # clamped to EOF


def test_coalesce_adjacent_into_one_call():
    """Touching / overlapping ranges merge into a single madvise call."""
    mm = _RecordingMM(size=100 * _PAGE)
    # Three page-aligned, contiguous ranges -> one merged run.
    starts = np.array([0, _PAGE, 2 * _PAGE], dtype=np.int64)
    lengths = np.array([_PAGE, _PAGE, _PAGE], dtype=np.int64)
    prefetch_willneed(mm, starts, lengths)
    assert len(mm.calls) == 1
    adv_start, adv_len = mm.calls[0]
    assert adv_start == 0
    assert adv_len == 3 * _PAGE


def test_non_adjacent_kept_separate():
    """A gap between ranges yields two distinct madvise calls."""
    mm = _RecordingMM(size=100 * _PAGE)
    starts = np.array([0, 50 * _PAGE], dtype=np.int64)
    lengths = np.array([_PAGE, _PAGE], dtype=np.int64)
    prefetch_willneed(mm, starts, lengths)
    assert len(mm.calls) == 2


def test_empty_input_is_noop():
    """Empty arrays issue no syscalls."""
    mm = _RecordingMM(size=10 * _PAGE)
    prefetch_willneed(
        mm, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    )
    assert mm.calls == []


def test_non_positive_and_past_eof_skipped():
    """``length <= 0`` and ``start >= size`` ranges drop out."""
    size = 5 * _PAGE
    mm = _RecordingMM(size=size)
    starts = np.array([0, _PAGE, 10 * _PAGE], dtype=np.int64)
    lengths = np.array([0, -5, _PAGE], dtype=np.int64)  # all invalid
    prefetch_willneed(mm, starts, lengths)
    assert mm.calls == []


def test_real_mmap_does_not_raise():
    """A real file-backed mmap (as np.memmap uses) accepts the hint.

    Uses a file-backed mmap because ``mmap.madvise(MADV_WILLNEED)`` and
    ``mmap.size()`` exercise the file-backed path that ``np.memmap._mmap``
    takes in production (an anonymous ``mmap(-1, N)`` has no fd).
    """
    fd, path = tempfile.mkstemp()
    try:
        os.ftruncate(fd, 8 * _PAGE)
        mm = mmap.mmap(fd, 8 * _PAGE, access=mmap.ACCESS_READ)
        try:
            prefetch_willneed(
                mm,
                np.array([100, 3 * _PAGE + 7], dtype=np.int64),
                np.array([_PAGE, 2 * _PAGE], dtype=np.int64),
            )
            assert mm.size() == 8 * _PAGE
        finally:
            mm.close()
    finally:
        os.close(fd)
        os.unlink(path)
