"""BinarySession happy-path lifecycle + fd-stability tests.

Covers the basic open + slice + close + idempotent-exit flow against
the post matched-arm restructuring wire format
(``[variant_ref, call_targets, indexer_hex]`` per matched variant row;
``[line_no_b64, variant_refs, called_line_nos, call_targets,
indexer_hex]`` per unmatched row). The fixture and ``_FakeArm`` /
``_FakeVocab`` shims live in :mod:`._session_fixture`.

Exception-safety / out-of-range / lifecycle-violation tests live in
:mod:`test_session_exception_safety` to keep both files under the
300 LOC project cap.
"""

from __future__ import annotations

import gc

import pytest

from tokenizer.aligned_data.loader.session import BinarySession

from ._session_fixture import count_open_fds, synthetic_binary  # noqa: F401


# --------------------------------------------------------------------
# Lifecycle / load
# --------------------------------------------------------------------


def test_session_opens_and_closes_cleanly(synthetic_binary):
    fb = synthetic_binary
    sess = BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    )
    with sess:
        m = sess.load_matched(0)
        assert m.func_name == "my_func"
        assert len(m.versions) == 2
        v = m.versions[0]
        # variant resolver attached the filename + axes
        assert v.metadata.get("filename") == "tinybin-x64-gcc-13.2.0-O2"
        assert v.metadata.get("arch") == "x64"
        assert v.metadata.get("compiler") == "gcc"
    # After exit, the stack is closed and slice methods should raise.
    with pytest.raises(RuntimeError):
        sess._open_data("matched")


def test_session_load_unmatched(synthetic_binary):
    fb = synthetic_binary
    # The reader returns memmap-backed views; ndarray consumers must read
    # them inside the session's ``with`` so the underlying mapping is
    # still live. Metadata (plain Python objects) is safe to use after.
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        f = sess.load_unmatched(0)
        assert f.func_name == "lonely_func"
        assert len(f.tokens) > 0
    variants = f.metadata.get("variants", [])
    assert len(variants) == 1
    assert variants[0].get("arch") == "x64"


def test_session_fd_no_leak(synthetic_binary):
    """Open + close N sessions; fd count must be stable."""
    fb = synthetic_binary
    gc.collect()
    before = count_open_fds()
    for _ in range(10):
        with BinarySession(
            fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
        ) as sess:
            sess.load_matched(0)
            sess.get_variant_by_ref(f"{fb['variant_offset']:x}")
        gc.collect()
    after = count_open_fds()
    if before >= 0:  # /proc is Linux only
        # ``np.memmap`` can leave a tracking fd open in some kernel
        # configs; allow a small slack but flag a 10x leak.
        assert after - before <= 2, f"fd leak: {before} -> {after}"


def test_exit_is_idempotent(synthetic_binary):
    fb = synthetic_binary
    sess = BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    )
    with sess:
        sess.load_matched(0)
    # Double-exit: should be a no-op, no exception.
    assert sess.__exit__(None, None, None) is False
    assert sess.__exit__(None, None, None) is False
    sess.close()
    sess.close()
    assert sess._closed is True
    assert sess._stack is None
