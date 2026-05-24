"""BinarySession exception-safety + lifecycle-violation tests.

Companion to :mod:`test_session` (the happy-path lifecycle tests).
Split out so both files stay under the project 300-LOC cap.

The plan explicitly calls out: "BinarySession exception safety is
mandatory." These tests verify that:

* an exception raised mid-slice still closes EVERY other handle
  that was opened earlier in the session,
* calling ``.close()`` from inside the with-block leaves a clean
  no-op for ``__exit__``,
* the session refuses slice calls outside its ``with`` block,
* out-of-range indices raise ``IndexError``,
* an unresolvable variant_ref returns ``None`` rather than raising.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pytest

from tokenizer.aligned_data.loader.session import BinarySession

from ._session_fixture import synthetic_binary  # noqa: F401


class _RaisingMemmap(np.ndarray):
    """Subclass of ndarray that pretends to be a memmap and raises on
    any indexing -- used to inject a mid-slice failure after the
    sections handle has been opened.
    """

    def __getitem__(self, key):  # type: ignore[override]
        raise RuntimeError("simulated data-bin read failure")


def test_exception_in_slice_still_closes_other_handles(synthetic_binary, monkeypatch):
    """Force a failure inside ``load_matched`` AFTER the sections BIN is
    loaded into the session's memoryview. The ExitStack must close
    the data-bin memmap (and any later-opened variants memmap) on
    ``__exit__`` even though the slice raised, and the session must
    release its memoryview of ``sections.bin``.
    """
    fb = synthetic_binary

    opened_memmaps: List[Any] = []
    real_memmap = np.memmap

    def memmap_factory(*args, **kwargs):
        m = real_memmap(*args, **kwargs)
        opened_memmaps.append(m)
        view = m.view(_RaisingMemmap)
        return view

    monkeypatch.setattr(np, "memmap", memmap_factory)

    sess = BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    )
    with pytest.raises(RuntimeError, match="simulated"):
        with sess:
            sess.load_matched(0)
    # The session opens the data-bin memmap before the simulated
    # failure fires (the memmap_factory wraps it in a _RaisingMemmap
    # whose first __getitem__ raises). The ExitStack must release it.
    assert opened_memmaps, "test did not exercise the data-bin memmap open path"
    # Sections BIN was read fully into a bytes object before the
    # failure; the session's __exit__ releases the memoryview and
    # drops the bytes reference. Verify the session reports closed.
    assert sess._closed is True
    assert sess._stack is None
    assert sess._sections_bin_view is None


def test_close_called_explicitly_inside_with(synthetic_binary):
    """Calling ``.close()`` inside the with-block leaves nothing for
    ``__exit__`` to unwind -- the second close must be a clean no-op.
    """
    fb = synthetic_binary
    sess = BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    )
    with sess:
        sess.load_matched(0)
        sess.close()
        with pytest.raises(RuntimeError):
            sess.load_matched(0)
    assert sess._closed is True


def test_slice_outside_with_block_raises(synthetic_binary):
    fb = synthetic_binary
    sess = BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    )
    with pytest.raises(RuntimeError):
        sess.load_matched(0)
    with pytest.raises(RuntimeError):
        sess.load_unmatched(0)


# --------------------------------------------------------------------
# Out-of-range + bad-input handling
# --------------------------------------------------------------------


def test_load_matched_oob_raises(synthetic_binary):
    fb = synthetic_binary
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        with pytest.raises(IndexError):
            sess.load_matched(99)


def test_load_unmatched_oob_raises(synthetic_binary):
    fb = synthetic_binary
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        with pytest.raises(IndexError):
            sess.load_unmatched(99)


def test_variant_by_ref_missing_returns_none(synthetic_binary):
    fb = synthetic_binary
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        assert sess.get_variant_by_ref("") is None
        # malformed hex
        assert sess.get_variant_by_ref("not-hex") is None


def test_corrupt_trailer_raises_did_not_pass_validation(synthetic_binary):
    """Flip the trailing ``total_entries`` u32 of ``_data.bin`` and
    verify the per-lookup integrity check fires.

    The session caches ``total_entries`` at ``_open_data`` time; bumping
    the on-disk u32 by an offset that makes every parsed record's
    ``entry_idx < total_entries`` would silently pass, so we flip it to
    zero (smaller than every real record's idx) instead. Any
    ``load_matched`` after that must raise the canonical corrupt-file
    error.
    """
    fb = synthetic_binary
    data_path = fb["base_path"] / f"{fb['binary_name']}_data.bin"
    raw = bytearray(data_path.read_bytes())
    # Overwrite the trailing u32 with zero (no record has idx < 0).
    raw[-4:] = b"\x00\x00\x00\x00"
    data_path.write_bytes(bytes(raw))

    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        with pytest.raises(ValueError, match="did not pass validation"):
            sess.load_matched(0)
