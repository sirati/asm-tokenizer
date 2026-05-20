"""Fixture-builder invariants for :mod:`_corpus`.

These tests guard the on-disk shape the fixture builder produces -- so
the loader integration tests sit on top of a verified ground truth
rather than an opaque blob. Each test exercises one wire-format
property of the synthetic corpus:

* 4-byte alignment of matched-arm CSV starts (the writer-enforced
  invariant after the trailing-newline padding step).
* function-names sidecar shape: ``# format=N`` prelude + alphabetical
  deduplicated names.
* per-function CSV intervals partition the matched-sections body
  without gaps or overlap (writer regression guard).

The overlong-escape / sentinel concept is gone -- records are now
self-describing in ``_data.bin`` -- so the corresponding sentinel
round-trip tests went with it.
"""

from __future__ import annotations

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

from ._corpus import (
    assert_starts_4_byte_aligned,
    build_corpus,
    make_variable_length_names,
    matched_spec,
    unmatched_spec,
)


def test_matched_csv_starts_are_all_4_byte_aligned(tmp_path):
    """Writer-enforced invariant: every matched-section CSV start is
    4-byte aligned (the writer pads each section with 1-4 trailing
    ``\\n`` bytes so the next section header lands on a 4-aligned
    offset). Use variable-length names so the padding code runs
    against every input residue.
    """
    names = sorted(make_variable_length_names("fn", count=8))
    corpus = build_corpus(
        tmp_path, "bin", matched=[matched_spec(n) for n in names]
    )
    starts = corpus.read_matched_csv_starts()
    assert len(starts) >= 4
    assert_starts_4_byte_aligned(starts)


def test_function_names_sidecar_has_prelude_and_is_alphabetical(tmp_path):
    """Sidecar contract: ``# format=N`` prelude on line 1, then names
    alphabetical + deduplicated."""
    matched = [matched_spec("zeta_fn", called=("alpha_callee", "beta_callee"))]
    unmatched = [unmatched_spec("zulu_fn", called=("alpha_callee",))]
    corpus = build_corpus(
        tmp_path, "bin", matched=matched, unmatched=unmatched
    )
    text = corpus.function_names_sidecar.read_text("utf-8")
    lines = text.splitlines()
    assert lines[0] == f"# format={MEMMAP_FORMAT_VERSION}"
    names = lines[1:]
    assert names == sorted(set(names)), "sidecar must be alphabetical + deduped"
    # ``alpha_callee`` referenced from both arms; appears exactly once.
    assert names.count("alpha_callee") == 1
    # Every expected name landed on disk.
    expected = {"alpha_callee", "beta_callee", "zeta_fn", "zulu_fn"}
    assert set(names) == expected


def test_matched_index_lengths_partition_csv_body(tmp_path):
    """Per-function ``(csv_offset, csv_length)`` intervals from
    matched_index.bin must partition the post-prelude CSV body without
    gaps or overlap.

    Catches off-by-one writer regressions in
    ``write_matched_sections_pass2`` and ``write_csv_section_index_entry``.
    """
    names = sorted(make_variable_length_names("fn", count=6))
    corpus = build_corpus(
        tmp_path, "bin", matched=[matched_spec(n) for n in names]
    )
    pair = read_csv_section_index_arrays(corpus.matched_index_bin)
    assert pair is not None
    starts, lengths = pair
    # Sort by offset so the partition assertions are independent of
    # the index file's emit order.
    pairs = sorted(zip(starts.tolist(), lengths.tolist()))
    raw = corpus.matched_sections_csv.read_bytes()
    prelude_end = raw.index(b"\n") + 1
    body_len = len(raw) - prelude_end
    assert pairs[0][0] == 0
    for (start_a, len_a), (start_b, _len_b) in zip(pairs, pairs[1:]):
        assert start_a + len_a == start_b, (
            f"sections do not partition cleanly: "
            f"{start_a} + {len_a} != {start_b}"
        )
    assert pairs[-1][0] + pairs[-1][1] == body_len
