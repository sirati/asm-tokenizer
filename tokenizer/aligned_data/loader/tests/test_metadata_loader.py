"""Tests for ``tokenizer.aligned_data.loader.metadata_loader``.

Post matched-arm restructuring: corpora come from :mod:`_corpus`,
which drives the production pass-2 writers + the function-names
registry. Function-name lengths cycle across 8 distinct values
(``7..14``) so matched-section CSV starts span every ``mod 4`` residue
non-coincidentally (see
:func:`test_matched_index_csv_starts_cover_every_mod4_residue` -- the
audit-driven assertion that documents the fixture's intent and
prevents a future "let's reintroduce alignment on this path"
regression from passing CI).

The matched arm tests rely on the post-batch-2A
``metadata_loader.load_section_arm`` consuming
``matched_index.bin`` via :func:`read_csv_section_index_arrays`
(pre-v1 layout, no alignment shift); they will fail with the
current v1 ``read_index_arrays``-based loader until F2-A lands.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.index_format import write_index_prelude
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.function_names_loader import (
    load_function_names,
)
from tokenizer.aligned_data.loader.metadata_loader import (
    BinaryArmPaths,
    SectionKind,
    load_section_arm,
    open_sections_csv,
)

from ._corpus import (
    assert_mod4_residues_covered,
    build_corpus,
    make_variable_length_names,
    matched_spec,
    unmatched_spec,
)


def _matched_paths(corpus) -> BinaryArmPaths:
    return BinaryArmPaths(
        sections_csv=corpus.matched_sections_csv,
        index_bin=corpus.matched_index_bin,
        data_bin=corpus.matched_data_bin,
    )


def _unmatched_paths(corpus) -> BinaryArmPaths:
    return BinaryArmPaths(
        sections_csv=corpus.unmatched_sections_csv,
        index_bin=corpus.unmatched_index_bin,
        data_bin=corpus.unmatched_data_bin,
    )


def _line_to_name(corpus):
    """Read the function-names sidecar produced by the corpus builder
    and return the ``line_no -> name`` dict; ``load_section_arm`` needs
    it to resolve the base64 line-no cells back to function names.
    """
    _, line_to_name = load_function_names(corpus.function_names_sidecar)
    return line_to_name


def _matched_corpus(tmp_path: Path):
    """A matched-only corpus with variable-length names spanning every
    mod-4 residue. No callees -> uniform section widths -> the residue
    coverage rides exclusively on the name-length cycling.
    """
    names = sorted(make_variable_length_names("matched_fn", count=8))
    specs = [matched_spec(n) for n in names]
    return build_corpus(tmp_path, "bin", matched=specs)


def _full_corpus(tmp_path: Path):
    """Matched + unmatched + cross-arm callee references. Use this when
    a test wants ``BinaryDataset`` (the full shell) rather than a single
    arm: both arms are non-empty, so both ``_arm_paths`` resolve to real
    files."""
    matched_names = sorted(make_variable_length_names("matched_fn", count=6))
    unmatched_names = ["unfn_a", "unfn_bb", "unfn_ccc"]
    matched = [matched_spec(n) for n in matched_names]
    unmatched = [unmatched_spec(n) for n in unmatched_names]
    return build_corpus(
        tmp_path, "bin", matched=matched, unmatched=unmatched
    )


# ---------------------------------------------------------------------------
# Residue-coverage intent assertion (the load-bearing fixture invariant)
# ---------------------------------------------------------------------------


def test_matched_index_csv_starts_cover_every_mod4_residue(tmp_path):
    """Fixture invariant: matched-index CSV starts span every mod-4 residue.

    Documents the audit-driven intent: the original
    ``matched_fn_{i}`` fixtures' 12-char names + 1-variant rows landed
    on 4-byte boundaries coincidentally, hiding the matched-arm
    alignment-assertion blocker. Asserting residue coverage here
    catches any regression that re-introduces an alignment requirement
    on this path.
    """
    corpus = _matched_corpus(tmp_path)
    starts = corpus.read_matched_csv_starts()
    assert len(starts) >= 4, (
        "fixture must produce at least 4 sections to demonstrate mod-4 "
        f"residue coverage; got {len(starts)}"
    )
    assert_mod4_residues_covered(starts)


# ---------------------------------------------------------------------------
# SectionArm contract (unchanged across batches)
# ---------------------------------------------------------------------------


def test_section_arm_equality_same_inputs(tmp_path):
    """``SectionArm`` is a frozen dataclass: identical inputs produce
    arms whose fields are element-equal."""
    corpus = _matched_corpus(tmp_path)
    line_to_name = _line_to_name(corpus)
    arm_a = load_section_arm(SectionKind.MATCHED, _matched_paths(corpus), line_to_name)
    arm_b = load_section_arm(SectionKind.MATCHED, _matched_paths(corpus), line_to_name)

    assert np.array_equal(arm_a.starts, arm_b.starts)
    assert np.array_equal(arm_a.edge_indices, arm_b.edge_indices)
    assert np.array_equal(arm_a.count_per_length, arm_b.count_per_length)
    assert arm_a.func_names == arm_b.func_names
    assert np.array_equal(arm_a.section_starts, arm_b.section_starts)
    assert arm_a.count == arm_b.count


def test_matched_arm_matches_legacy_attributes(tmp_path):
    """``BinaryDataset.matched_*`` attributes equal the matched
    ``SectionArm`` fields -- pre/post-refactor surface is byte-equal."""
    corpus = _full_corpus(tmp_path)
    dataset = BinaryDataset(corpus.base_path, corpus.binary_name)
    arm = dataset._matched_arm

    assert np.array_equal(dataset.matched_starts, arm.starts)
    assert np.array_equal(dataset.matched_edge_indices, arm.edge_indices)
    assert np.array_equal(
        dataset.matched_count_per_length, arm.count_per_length
    )
    assert dataset.matched_func_names == arm.func_names
    assert dataset.matched_count == arm.count


def test_unmatched_arm_matches_legacy_attributes(tmp_path):
    """``BinaryDataset.unmatched_*`` attributes equal the unmatched
    ``SectionArm`` fields."""
    corpus = _full_corpus(tmp_path)
    dataset = BinaryDataset(corpus.base_path, corpus.binary_name)
    arm = dataset._unmatched_arm

    assert np.array_equal(dataset.unmatched_starts, arm.starts)
    assert np.array_equal(dataset.unmatched_edge_indices, arm.edge_indices)
    assert np.array_equal(
        dataset.unmatched_count_per_length, arm.count_per_length
    )
    assert dataset.unmatched_func_names == arm.func_names
    assert dataset.unmatched_count == arm.count


# ---------------------------------------------------------------------------
# section_starts semantics
# ---------------------------------------------------------------------------


def test_matched_arm_csv_starts_index_section_csv_bytes(tmp_path):
    """Matched ``load(idx)`` seeks the section CSV via
    ``csv_starts``/``csv_lengths`` (per-function); ``starts`` carries
    per-variant data-bin offsets for the validator. Function names
    recovered from the sidecar match ``arm.func_names``.
    """
    corpus = _matched_corpus(tmp_path)
    arm = load_section_arm(
        SectionKind.MATCHED, _matched_paths(corpus), _line_to_name(corpus)
    )
    assert arm.csv_starts is not None and len(arm.csv_starts) == len(arm.func_names)
    assert arm.csv_lengths is not None and len(arm.csv_lengths) == len(arm.func_names)
    sidecar_names = (
        corpus.function_names_sidecar.read_text("utf-8").splitlines()[1:]
    )
    for name in arm.func_names:
        assert name in sidecar_names, (
            f"matched walker recovered name {name!r} not in sidecar"
        )


def test_unmatched_section_starts_point_to_rows(tmp_path):
    """``section_starts[i]`` is the byte offset of the row whose first
    cell (base64 line_no) maps to ``func_names[i]``.

    Post-restructure: 5-cell row ``[line_no_b64, variant_refs,
    called_line_nos_b64, inlining_data, indexer_hex]``; the post-F2-A
    walker translates the base64 line_no via the function-names
    sidecar so ``arm.func_names`` returns raw names.
    """
    corpus = build_corpus(
        tmp_path,
        "bin",
        unmatched=[unmatched_spec(n) for n in ("unfn_a", "unfn_bb", "unfn_ccc")],
    )
    arm = load_section_arm(
        SectionKind.UNMATCHED, _unmatched_paths(corpus), _line_to_name(corpus)
    )

    assert arm.func_names == ["unfn_a", "unfn_bb", "unfn_ccc"]
    assert len(arm.section_starts) == 3

    f, content_offset = open_sections_csv(corpus.unmatched_sections_csv)
    try:
        for i, offset in enumerate(arm.section_starts):
            f.seek(int(offset) + content_offset)
            line = f.readline()
            row = next(csv.reader([line]), None)
            assert row is not None and len(row) == 5, (
                f"unmatched row at section_starts[{i}]={offset} did not "
                f"parse as 5-cell: got {row!r}"
            )
            # First cell is a non-empty base64 line number.
            assert row[0], f"row[0] (base64 line_no) is empty at index {i}"
    finally:
        f.close()


def test_section_kind_enum_is_closed_typed():
    """Sanity: ``SectionKind`` is an enum (not a bool), and both arms
    are registered. Future arms add an enum value + spec; no boolean
    toggle in the caller.
    """
    assert SectionKind.MATCHED is not SectionKind.UNMATCHED
    assert SectionKind.MATCHED.value == "matched"
    assert SectionKind.UNMATCHED.value == "unmatched"


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_arm_when_index_missing(tmp_path):
    """Missing index file yields the canonical empty arm with
    dtype-preserving placeholders (so downstream length / indexing
    arithmetic doesn't degrade)."""
    paths = BinaryArmPaths(
        sections_csv=tmp_path / "absent_sections.csv",
        index_bin=tmp_path / "absent_index.bin",
        data_bin=tmp_path / "absent_data.bin",
    )
    arm = load_section_arm(SectionKind.MATCHED, paths)
    assert arm.count == 0
    assert arm.starts.dtype == np.int64
    assert arm.edge_indices.dtype == np.int32
    assert arm.count_per_length.dtype == np.int32
    assert arm.func_names == []
    assert arm.section_starts.dtype == np.int64
    assert len(arm.section_starts) == 0


def test_zero_entry_index_yields_empty_arm(tmp_path):
    """v1 index with only the 16-byte prelude (zero entries) -> empty arm.

    Exercises the unmatched arm: matched ``_index.bin`` is pre-v1
    layout (no prelude); the v1 reader path lives only on the
    unmatched side post-restructure.
    """
    index_path = tmp_path / "empty_index.bin"
    with open(index_path, "wb") as f:
        write_index_prelude(f)
    paths = BinaryArmPaths(
        sections_csv=tmp_path / "empty_sections.csv",
        index_bin=index_path,
        data_bin=tmp_path / "empty_data.bin",
    )
    arm = load_section_arm(SectionKind.UNMATCHED, paths)
    assert arm.count == 0
    assert len(arm.starts) == 0
