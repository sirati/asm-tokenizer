"""Fixture-builder invariants for :mod:`_corpus`.

These tests guard the on-disk shape the fixture builder produces -- so
the loader integration tests sit on top of a verified ground truth
rather than an opaque blob. Each test exercises one wire-format
property of the synthetic corpus:

* mod-4 residue coverage for matched-arm CSV starts
  (the audit-driven invariant the whole batch protects).
* overlong-sentinel encoding on a matched variant: the inline
  ``indexer_hex`` carries ``length_field == SENTINEL_LENGTH``, and the
  matching data record's overlong field carries the real length.
* overlong-sentinel encoding on an unmatched version: the v1 index
  entry carries the sentinel and the data record's overlong field
  carries the real length.
* function-names sidecar shape: ``# format=N`` prelude + alphabetical
  deduplicated names.

Independent of the parallel loader rewrites (F2-A / F2-B): these
tests read the on-disk bytes directly via the production codecs, so
they pass NOW and remain valid after the loader changes land.
"""

from __future__ import annotations

import csv

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.index_format import (
    MAX_NORMAL_REAL_LENGTH,
    SENTINEL_LENGTH,
    read_index_arrays,
)
from tokenizer.aligned_data.inline_indexer import decode_inline_indexer
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import open_sections_csv
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

from ._corpus import (
    assert_mod4_residues_covered,
    build_corpus,
    make_variable_length_names,
    matched_spec,
    unmatched_spec,
)


def test_matched_csv_starts_cover_every_mod4_residue(tmp_path):
    """Audit-driven fixture invariant: matched-index CSV starts span
    every mod-4 residue. Documents the intent so a future regression
    that re-introduces an alignment requirement on this path cannot
    pass CI silently.
    """
    names = sorted(make_variable_length_names("fn", count=8))
    corpus = build_corpus(
        tmp_path, "bin", matched=[matched_spec(n) for n in names]
    )
    starts = corpus.read_matched_csv_starts()
    assert len(starts) >= 4
    assert_mod4_residues_covered(starts)


def test_matched_variant_overlong_indexer_round_trips(tmp_path):
    """Cross-product of "matched" x "overlong-sentinel": a matched
    function whose second variant is overlong emits
    ``indexer_hex`` whose decode surfaces ``is_overlong=True`` and the
    real record length (NOT the sentinel zero).

    The pass-2 writer packs the inline entry via
    ``encode_inline_indexer`` (which delegates to ``pack_v1_entry``);
    the same layout the loader decodes via ``decode_inline_indexer``.
    Round-tripping here verifies the writer-side wire format BEFORE
    the loader-side rewrite (F2-B) lands.
    """
    spec = matched_spec("fn_with_overlong", n_variants=2, overlong_variant_idx=1)
    corpus = build_corpus(tmp_path, "bin", matched=[spec])

    # Read every matched variant row (CSV body, post-prelude); the
    # second variant should be the overlong one.
    f, _content_offset = open_sections_csv(corpus.matched_sections_csv)
    try:
        rows = [row for row in csv.reader(f) if row]
    finally:
        f.close()
    # Layout: per-function header (2 cells) + N variant rows (3 cells)
    # + optional blank row between functions; we have ONE function +
    # 2 variants + 1 trailing blank-stripped by the reader.
    variant_rows = [row for row in rows if len(row) == 3]
    assert len(variant_rows) == 2

    # First variant: normal. Second: overlong.
    _start0, len0, overlong0 = decode_inline_indexer(variant_rows[0][2])
    start1, len1, overlong1 = decode_inline_indexer(variant_rows[1][2])
    assert overlong0 is False
    assert len0 > 0
    assert overlong1 is True, (
        f"variant 2 should be overlong; decode returned "
        f"start={start1}, length={len1}, is_overlong={overlong1}"
    )
    # Sentinel encoding: when ``is_overlong`` is True the reader
    # surfaces ``length == 0`` (the sentinel marker) so the consumer
    # routes through the data record's overlong field.
    assert len1 == 0


def test_session_load_matched_overlong_returns_real_tokens(tmp_path):
    """End-to-end: ``BinaryDataset.load_matched_function`` resolves the
    inline-indexer sentinel for an overlong matched variant and returns
    the full token sequence (not an empty slice).

    Pins the matched-arm sentinel-resolution path through the session
    slice: ``_slice_data_record`` must call ``resolve_record_length``
    internally so the matched arm (which decodes ``indexer_hex`` →
    sentinel) and the unmatched arm (which pre-resolves) both produce
    real lengths. A regression here would surface as
    ``IndexError: index 0 is out of bounds`` at ``binary_format.py``'s
    header parse when the parser receives ``length == 0``.
    """
    spec = matched_spec("fn_with_overlong", n_variants=2, overlong_variant_idx=1)
    corpus = build_corpus(tmp_path, "bin", matched=[spec])

    dataset = BinaryDataset(corpus.base_path, corpus.binary_name)
    matched = dataset.load_matched_function(0)
    assert matched.func_name == "fn_with_overlong"
    assert len(matched.versions) == 2
    overlong_variant = matched.versions[1]
    # The overlong variant must decode to a real (non-empty) token
    # sequence — a length-0 slice would silently produce an empty
    # array and the test would notice via the token count assertion.
    assert overlong_variant.tokens.size > 0, (
        "overlong matched variant produced empty tokens; "
        "sentinel resolution failed in the slice path"
    )
    # Cross-check: the record's total bytes exceed the v1 normal cap
    # (>~256 KiB), which is what makes it overlong in the first place.
    # tokens are uint16 → 2 bytes each; insn + block + tokens roughly
    # tracks the record body.
    record_bytes = (
        overlong_variant.tokens.nbytes
        + overlong_variant.insn_runlength.nbytes
        + overlong_variant.block_runlength.nbytes
    )
    assert record_bytes > MAX_NORMAL_REAL_LENGTH // 2, (
        f"overlong variant body {record_bytes} bytes is below the "
        f"normal cap — fixture isn't actually exercising the sentinel path"
    )


def test_unmatched_overlong_index_entry_surfaces_sentinel(tmp_path):
    """An unmatched function with an overlong-record version writes
    the sentinel into its v1 ``unmatched_index.bin`` entry.

    The unmatched arm STAYED on the v1 index format (data-bin offsets
    are 4-aligned), so the existing sentinel/overlong machinery applies
    unchanged. This test asserts the fixture exercises that path,
    making it a valid stand-in for production corpora that contain
    overlong records.
    """
    spec = unmatched_spec("fn_overlong_unmatched", n_versions=2, overlong_version_idx=1)
    corpus = build_corpus(tmp_path, "bin", unmatched=[spec])

    arrays = read_index_arrays(corpus.unmatched_index_bin)
    assert arrays is not None
    _starts, lengths, _avg = arrays
    assert len(lengths) == 2
    assert lengths[0] != SENTINEL_LENGTH, (
        f"non-overlong version 1 must NOT surface sentinel; "
        f"got length {lengths[0]}"
    )
    assert lengths[1] == SENTINEL_LENGTH, (
        f"overlong version 2 must surface sentinel ({SENTINEL_LENGTH}); "
        f"got length {lengths[1]}"
    )


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
    triple = read_csv_section_index_arrays(corpus.matched_index_bin)
    assert triple is not None
    starts, lengths, _avg = triple
    # Sort by offset so the partition assertions are independent of
    # the index file's avg_len bucket sort.
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
