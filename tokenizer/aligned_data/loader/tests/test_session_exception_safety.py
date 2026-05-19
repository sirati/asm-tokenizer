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
    """Force a failure inside ``load_matched`` AFTER the sections handle
    is open. The ExitStack must close the sections handle (and any
    later-opened variants handle) on ``__exit__`` even though the
    slice raised.
    """
    fb = synthetic_binary
    opened: List[Any] = []

    import builtins

    orig_open = builtins.open

    def tracking_open(*args, **kwargs):
        f = orig_open(*args, **kwargs)
        opened.append(f)
        return f

    monkeypatch.setattr(builtins, "open", tracking_open)

    real_memmap = np.memmap

    def memmap_factory(*args, **kwargs):
        m = real_memmap(*args, **kwargs)
        view = m.view(_RaisingMemmap)
        return view

    monkeypatch.setattr(np, "memmap", memmap_factory)

    sess = BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    )
    with pytest.raises(RuntimeError, match="simulated"):
        with sess:
            sess.load_matched(0)
    sections_path = fb["base_path"] / f"{fb['binary_name']}_sections.csv"
    sections_opens = [
        f for f in opened
        if hasattr(f, "name") and str(f.name) == str(sections_path)
    ]
    assert sections_opens, "test did not exercise sections-open path"
    for f in sections_opens:
        assert f.closed, "ExitStack did not close the sections handle on exception"
    assert sess._closed is True
    assert sess._stack is None


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
