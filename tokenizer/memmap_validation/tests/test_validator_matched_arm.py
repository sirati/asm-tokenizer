"""Matched-arm validator coverage post-restructuring.

Both tests run the validator's per-record v1 invariant checks against
a synthetic matched corpus laid down by
:mod:`tokenizer.aligned_data.loader.tests._corpus` -- the same fixture
the loader integration tests use -- so the bytes the validator
inspects are byte-identical to a real build:

  1. ``run_v1_post_checks`` is clean on a multi-variant matched corpus
     -- post-restructuring the per-record checks consume per-VARIANT
     data-bin positions (decoded from the section CSV's
     ``indexer_hex`` cell), not the per-FUNCTION CSV-section byte
     offsets the matched_index.bin carries. The clean-pass test
     proves both arms' starts arrays survive the pad + bounds probes.
  2. ``run_v1_post_checks`` is clean on a matched corpus whose
     function names span a wide length spread (the variable-length
     names fixture). Section CSV starts are 4-byte aligned by writer
     construction; the validator's per-record alignment check
     consumes 16-byte-aligned DATA-bin positions, not CSV offsets, so
     the check stays clean even when the CSV section starts happen
     to span every mod-N residue.
"""

from __future__ import annotations

from pathlib import Path

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.tests._corpus import (
    assert_starts_4_byte_aligned,
    build_corpus,
    make_variable_length_names,
    matched_spec,
)
from tokenizer.memmap_validation._v1_checks import run_v1_post_checks


def _matched_dataset(corpus) -> BinaryDataset:
    """Open a ``BinaryDataset`` against the corpus output directory.

    Variants sidecar is absent from the synthetic corpus (the loader
    tolerates it); we only need the matched arm here.
    """
    return BinaryDataset(corpus.base_path, corpus.binary_name)


# ---------------------------------------------------------------------------
# (1) Multi-variant matched corpus: post-checks clean
# ---------------------------------------------------------------------------


def test_run_v1_post_checks_clean_with_multi_variant_matched_corpus(
    tmp_path: Path,
) -> None:
    """A multi-variant matched function must pass post-checks unchanged.

    Pins that the per-record path consumes per-VARIANT data-bin
    positions (decoded from inline_indexer per section), and that
    those positions land at 16-byte-aligned offsets with zero pad
    bytes in both pre- and post-pad regions.
    """
    spec = matched_spec("fn_multi_variant", n_variants=3)
    corpus = build_corpus(tmp_path, "bin", matched=[spec])
    dataset = _matched_dataset(corpus)

    errors = run_v1_post_checks(
        matched_index=corpus.matched_index_bin,
        unmatched_index=corpus.unmatched_index_bin,
        matched_data=corpus.matched_data_bin,
        unmatched_data=corpus.unmatched_data_bin,
        matched_starts=dataset.matched_starts,
        unmatched_starts=dataset.unmatched_starts,
    )
    assert errors == [], (
        f"post-checks should be clean on a multi-variant matched corpus; "
        f"got: {errors!r}"
    )


# ---------------------------------------------------------------------------
# (2) Variable-length function names: CSV starts 4-aligned, data starts
#     16-aligned, post-checks clean.
# ---------------------------------------------------------------------------


def test_run_v1_post_checks_clean_with_variable_length_names(
    tmp_path: Path,
) -> None:
    """Wide name-length spread must not trip the data-bin checks.

    The matched-arm CSV writer pads each section so the next section
    starts on a 4-byte boundary (writer invariant), and the
    per-record validator alignment check consumes ``matched_starts``
    -- DATA-bin positions, 16-byte aligned by construction -- not the
    CSV offsets. Documents that the post-checks read from the correct
    arrays under organic name-length variation.
    """
    names = sorted(make_variable_length_names("fn", count=8))
    corpus = build_corpus(
        tmp_path, "bin", matched=[matched_spec(n) for n in names]
    )
    dataset = _matched_dataset(corpus)

    # Pin the writer's BIN-section-start alignment invariant. A
    # regression here would also surface in this validator-side test,
    # not only in the loader-side fixture invariants.
    bin_starts = corpus.read_matched_bin_starts()
    assert_starts_4_byte_aligned(bin_starts)

    errors = run_v1_post_checks(
        matched_index=corpus.matched_index_bin,
        unmatched_index=corpus.unmatched_index_bin,
        matched_data=corpus.matched_data_bin,
        unmatched_data=corpus.unmatched_data_bin,
        matched_starts=dataset.matched_starts,
        unmatched_starts=dataset.unmatched_starts,
    )
    assert errors == [], (
        f"post-checks should be clean on a variable-length-names matched "
        f"corpus; got: {errors!r}"
    )
