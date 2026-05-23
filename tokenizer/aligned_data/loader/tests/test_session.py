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
        assert len(m.variants) == 2
        v = m.variants[0]
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


def test_loader_populates_category_counts_on_both_arms(synthetic_binary):
    """``FunctionData.metadata["category_counts"]`` is loader-populated.

    Stage 4a's ALG-4 dedup walk reads per-function COUNTER-Category
    unique-id counts via the metadata key on every matched and
    unmatched function. The loader is the single source of truth -- a
    missing key would surface as a ``KeyError`` deep inside the dedup
    walk under production loads. Asserting the key is populated at
    load time keeps that contract close to the wiring site.
    """
    from tokenizer.aligned_data.loader.category_counts import (
        COUNTER_CATEGORIES,
    )

    fb = synthetic_binary
    # BinarySession opens one data arm per session, so the two arms get
    # one session each. Metadata dicts survive close (plain Python objects)
    # so the asserts can read them outside the ``with``.
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        matched = sess.load_matched(0)
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        unmatched = sess.load_unmatched(0)

    # Matched arm: each variant carries the dense {Category: int} dict.
    for variant in matched.variants:
        counts = variant.metadata.get("category_counts")
        assert counts is not None, "matched variant missing category_counts"
        assert set(counts.keys()) == set(COUNTER_CATEGORIES)
        assert all(isinstance(v, int) and v >= 0 for v in counts.values())

    # Unmatched arm: same metadata contract.
    counts = unmatched.metadata.get("category_counts")
    assert counts is not None, "unmatched function missing category_counts"
    assert set(counts.keys()) == set(COUNTER_CATEGORIES)
    assert all(isinstance(v, int) and v >= 0 for v in counts.values())


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
