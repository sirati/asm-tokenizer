"""Matched-arm validator coverage post-restructuring.

These tests target three integrity guarantees that the matched-arm
restructuring + Audit Blocker #2 fix put in scope for the validator:

  1. ``run_v1_post_checks`` is clean on a matched corpus with an
     overlong-sentinel variant -- the pad-zero and sentinel-overlong
     coupling probes both run over DATA-bin positions (decoded from
     ``indexer_hex`` per variant), so an overlong record passes the
     checks instead of tripping spurious false positives.
  2. ``run_v1_post_checks`` is clean on a matched corpus whose section
     CSV starts are NOT 4-byte aligned -- documents that the validator
     accepts the pre-v1 matched_index.bin layout (the v1 alignment
     rule applies only to data-bin positions, not CSV text-file
     positions).
  3. A deliberately-tampered overlong field surfaces as a clear error
     from ``check_sentinel_overlong_coupling`` -- proving the matched
     arm's data-bin checks actually exercise the sentinel decode and
     would catch a writer regression.

The corpus is built via :mod:`tokenizer.aligned_data.loader.tests._corpus`
which drives the same production pass-2 writers used by
``build_memmap_files`` -- so the on-disk bytes the validator inspects
are byte-identical to a real build.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.tests._corpus import (
    assert_mod4_residues_covered,
    build_corpus,
    make_variable_length_names,
    matched_spec,
)
from tokenizer.memmap_validation._v1_checks import (
    check_sentinel_overlong_coupling,
    run_v1_post_checks,
)


def _matched_dataset(corpus) -> BinaryDataset:
    """Open a ``BinaryDataset`` against the corpus output directory.

    Variants sidecar is absent from the synthetic corpus (the loader
    tolerates it); we only need the matched arm here.
    """
    return BinaryDataset(corpus.base_path, corpus.binary_name)


# ---------------------------------------------------------------------------
# (1) Overlong matched variant: post-checks clean
# ---------------------------------------------------------------------------


def test_run_v1_post_checks_clean_with_overlong_matched_variant(tmp_path: Path) -> None:
    """An overlong variant must pass pad+sentinel checks unchanged.

    Audit Blocker #2 root cause: the previous wiring fed CSV byte
    offsets (NOT 4-aligned, NOT data-bin positions) into the v1 pad +
    sentinel checks, producing both false positives (CSV offsets that
    happened to look unaligned) and silent misses (real data-bin
    sentinels never reached the check). Sourcing per-variant
    starts/lengths from inline_indexer decode fixes both directions.
    """
    spec = matched_spec("fn_with_overlong", n_variants=2, overlong_variant_idx=1)
    corpus = build_corpus(tmp_path, "bin", matched=[spec])
    dataset = _matched_dataset(corpus)

    # Sanity: at least one overlong variant was emitted by the corpus.
    assert dataset.matched_is_overlong.any(), (
        "fixture should have produced at least one overlong-sentinel "
        "matched variant (per the spec)"
    )

    errors = run_v1_post_checks(
        matched_index=corpus.matched_index_bin,
        unmatched_index=corpus.unmatched_index_bin,
        matched_data=corpus.matched_data_bin,
        unmatched_data=corpus.unmatched_data_bin,
        matched_starts=dataset.matched_starts,
        matched_lengths=dataset.matched_lengths,
        unmatched_starts=dataset.unmatched_starts,
        unmatched_lengths=dataset.unmatched_lengths,
    )
    assert errors == [], (
        f"post-checks should be clean on a matched+overlong corpus; "
        f"got: {errors!r}"
    )


# ---------------------------------------------------------------------------
# (2) Non-4-aligned matched CSV section starts: post-checks clean
# ---------------------------------------------------------------------------


def test_run_v1_post_checks_clean_with_non_4_aligned_csv_starts(tmp_path: Path) -> None:
    """Matched-arm CSV starts NOT being 4-aligned must not trip checks.

    The matched_index.bin holds TEXT-file byte offsets (variable-
    length function names mean section starts land at every mod-4
    residue). v1's 4-byte alignment rule applies to data-bin
    positions only; the validator's per-record alignment check
    consumes ``matched_starts`` (per-variant DATA-bin positions),
    not the CSV starts. Documents that the pre-v1 matched_index.bin
    is accepted by construction.
    """
    names = sorted(make_variable_length_names("fn", count=8))
    corpus = build_corpus(
        tmp_path, "bin", matched=[matched_spec(n) for n in names]
    )
    dataset = _matched_dataset(corpus)

    # The fixture is supposed to span every mod-4 residue; assert here
    # so a future regression in the names generator surfaces in this
    # validator-side test as well, not only in the loader-side fixture
    # invariants.
    csv_starts = corpus.read_matched_csv_starts()
    assert_mod4_residues_covered(csv_starts)

    errors = run_v1_post_checks(
        matched_index=corpus.matched_index_bin,
        unmatched_index=corpus.unmatched_index_bin,
        matched_data=corpus.matched_data_bin,
        unmatched_data=corpus.unmatched_data_bin,
        matched_starts=dataset.matched_starts,
        matched_lengths=dataset.matched_lengths,
        unmatched_starts=dataset.unmatched_starts,
        unmatched_lengths=dataset.unmatched_lengths,
    )
    assert errors == [], (
        f"post-checks should be clean on a non-4-aligned CSV-starts "
        f"matched corpus; got: {errors!r}"
    )


# ---------------------------------------------------------------------------
# (3) Tampered matched overlong field: sentinel-coupling check fires
# ---------------------------------------------------------------------------


def test_check_sentinel_overlong_coupling_flags_matched_data_tamper(
    tmp_path: Path,
) -> None:
    """Mutate the overlong u24 field of a matched overlong variant.

    The resolver decodes the tampered field as a normal-band real
    length; ``check_sentinel_overlong_coupling`` reports it as a
    sentinel/overlong-field mismatch (the sentinel was promised in
    the index entry, but the record now claims a length that fits the
    normal u16 cap).
    """
    spec = matched_spec("fn_with_overlong", n_variants=2, overlong_variant_idx=1)
    corpus = build_corpus(tmp_path, "bin", matched=[spec])
    dataset = _matched_dataset(corpus)

    # Identify the overlong matched variant + its data-bin start.
    overlong_idx = int(np.where(dataset.matched_is_overlong)[0][0])
    overlong_start = int(dataset.matched_starts[overlong_idx])

    # The overlong field is the u24 immediately after the 6-byte header.
    from tokenizer.aligned_data.binary_format import HEADER_BYTES

    raw = bytearray(corpus.matched_data_bin.read_bytes())
    # Overwrite the u24 with a tiny value (decodes well below the u16 cap),
    # which the resolver returns as ``is_overlong=True`` but with a
    # real_length the sentinel rule should have ruled out.
    tampered = struct.pack("<I", 7)[:3]  # 3 bytes, little-endian "7"
    raw[overlong_start + HEADER_BYTES : overlong_start + HEADER_BYTES + 3] = tampered
    corpus.matched_data_bin.write_bytes(bytes(raw))

    errors = check_sentinel_overlong_coupling(
        corpus.matched_data_bin,
        dataset.matched_starts,
        dataset.matched_lengths,
        str(corpus.matched_index_bin),
    )
    assert errors, (
        "tampered overlong field on a sentinel-flagged matched variant "
        "should fire the coupling check"
    )
    assert any(
        "fits the normal u16 cap" in e or "did not resolve as overlong" in e
        for e in errors
    ), f"unexpected error wording: {errors!r}"
