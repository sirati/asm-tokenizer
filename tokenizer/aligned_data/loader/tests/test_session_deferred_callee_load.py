"""BinarySession: deferred per-variant callee-body load is byte-identical.

The callee walk splits resolution into a metadata-only decision
(``_matched_section_meta`` / ``_unmatched_section_meta``, no ``_data.bin``
touch) and a per-survivor body load (``_load_matched_variant_body`` /
``_load_unmatched_for_splice``). This pins the load-scheduling invariant:
the single-variant body the deferred load produces is byte-for-byte the
same ``FunctionData`` as the all-variants load's
``MatchedFunction.variants[i]``, and the metadata-only parse returns the
same ``Section`` + BIN offset as the full load -- so deferring the body
read past the once-only prune can never change which bytes splice.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset

from ._session_fixture import synthetic_binary  # noqa: F401


def _assert_function_data_equal(a, b) -> None:
    np.testing.assert_array_equal(a.tokens, b.tokens)
    np.testing.assert_array_equal(a.insn_runlength, b.insn_runlength)
    np.testing.assert_array_equal(a.block_runlength, b.block_runlength)
    np.testing.assert_array_equal(a.variant_tokens, b.variant_tokens)
    assert a.func_name == b.func_name
    assert a.metadata == b.metadata


def _count_section_parses(sess):
    """Wrap ``sess._parse_section_at`` with a call counter (context mgr).

    Returns a 0-length list whose length tracks the number of
    ``_parse_section_at`` invocations while the wrapper is installed --
    the pin for "the body load does NOT re-parse the threaded section".
    """
    from contextlib import contextmanager

    @contextmanager
    def _wrap():
        calls: list = []
        original = sess._parse_section_at

        def _spy(offset):
            calls.append(offset)
            return original(offset)

        sess._parse_section_at = _spy
        try:
            yield calls
        finally:
            sess._parse_section_at = original

    return _wrap()


def test_matched_variant_body_matches_all_variants_load(synthetic_binary) -> None:
    """``_load_matched_variant_body(idx, v, section)`` equals the
    all-variants load's ``variants[v]`` for every variant of every matched
    section, and issues NO second section parse (the threaded
    ``section`` is reused, not re-derived via ``_parse_section_at``)."""
    fb = synthetic_binary
    ds = BinaryDataset(
        fb["base_path"], fb["binary_name"], vocab_manager=fb["vocab"]
    )
    with ds.open_session() as sess:
        n_matched = len(sess.get_metadata("matched_arm").bin_starts)
        for idx in range(n_matched):
            section, offset, matched = (
                sess._load_matched_section_and_variants(idx)
            )
            # Metadata-only parse: same Section + offset, no body needed.
            meta_section, meta_offset = sess._matched_section_meta(idx)
            assert meta_offset == offset
            assert meta_section.section_offset == section.section_offset
            assert len(meta_section.variants) == len(section.variants)
            # Deferred single-variant body == the all-variants body, with
            # NO second ``_parse_section_at`` (threaded section reused).
            for v_idx in range(len(matched.variants)):
                with _count_section_parses(sess) as parses:
                    one = sess._load_matched_variant_body(
                        idx, v_idx, meta_section
                    )
                assert parses == [], (
                    f"matched body load re-parsed _sections.bin "
                    f"(idx={idx}, v={v_idx}, offsets={parses})"
                )
                _assert_function_data_equal(one, matched.variants[v_idx])


def test_unmatched_variant_body_matches_full_load(synthetic_binary) -> None:
    """``_load_unmatched_variant_body(idx, section)`` equals the
    section-deriving full record load's ``FunctionData`` and issues NO
    second section parse (the threaded ``section`` is reused)."""
    fb = synthetic_binary
    ds = BinaryDataset(
        fb["base_path"], fb["binary_name"], vocab_manager=fb["vocab"]
    )
    with ds.open_session() as sess:
        starts = sess.get_metadata("unmatched_arm").starts
        for idx in range(len(starts)):
            _section, _offset, full_fd = (
                sess._load_unmatched_record_and_section(idx)
            )
            meta_section, _meta_offset = sess._unmatched_section_meta(idx)
            with _count_section_parses(sess) as parses:
                one = sess._load_unmatched_variant_body(idx, meta_section)
            assert parses == [], (
                f"unmatched body load re-parsed _sections.bin "
                f"(idx={idx}, offsets={parses})"
            )
            _assert_function_data_equal(one, full_fd)


def test_unmatched_section_meta_matches_full_load(synthetic_binary) -> None:
    """``_unmatched_section_meta(idx)`` returns the same ``Section`` +
    offset the full unmatched record load does (no body sliced)."""
    fb = synthetic_binary
    ds = BinaryDataset(
        fb["base_path"], fb["binary_name"], vocab_manager=fb["vocab"]
    )
    with ds.open_session() as sess:
        starts = sess.get_metadata("unmatched_arm").starts
        for idx in range(len(starts)):
            section, offset, _fd = (
                sess._load_unmatched_record_and_section(idx)
            )
            meta_section, meta_offset = sess._unmatched_section_meta(idx)
            assert meta_offset == offset
            assert meta_section.section_offset == section.section_offset
            assert len(meta_section.variants) == len(section.variants)
