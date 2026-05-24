"""Tests for :mod:`sorted_index._builder`.

Covers the per-binary build entry + file-writing wrapper:

* :func:`build_sorted_index_bytes` returns a dict keyed by the input
  reductions; each value is wire-format bytes parseable by
  :func:`parse_header` AND byte-equal to a Python-loop oracle that
  re-runs :func:`compute_reduced_lengths` + :func:`encode_sorted_index`.
* Round-trip through :class:`SortedIndexReader`: per-length bucket
  counts match an :func:`np.bincount` oracle over the same per-mode
  length array.
* Multi-mode shared walk (plan §D8): a single
  :func:`build_sorted_index_bytes` call with K reductions invokes
  :func:`compute_reduced_lengths` EXACTLY ONCE (asserted via
  :func:`unittest.mock.patch`).
* :func:`write_sorted_index_files` writes files matching the canonical
  filename grammar; the written bytes are byte-equal to
  :func:`build_sorted_index_bytes` output.
* :func:`write_sorted_index_files` honours ``output_dir`` -- when
  passed a directory distinct from the memmap dir the files land there
  and NOT next to the sidecars.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.sorted_index import (
    LengthReduction,
    ReductionKind,
    SortedIndexReader,
    build_sorted_index_bytes,
    compute_reduced_lengths,
    encode_sorted_index,
    parse_header,
    write_sorted_index_files,
)
from tokenizer.aligned_data.sorted_index._length_compute import (
    _count_variants_per_section,
)

from .fixtures import build_combined_fixture


_BINARY_NAME = "sortbin"

_MAX = LengthReduction(kind=ReductionKind.MAX)
_P50 = LengthReduction(kind=ReductionKind.PERCENTILE, percentile=50)
_P95 = LengthReduction(kind=ReductionKind.PERCENTILE, percentile=95)


# Canonical filename grammar (plan §D5).  Mirrored here intentionally so
# a regression in the writer's filename construction fails the test
# instead of silently agreeing with a drifted regex.
_FILENAME_RE = re.compile(
    r"^(?P<binary>.+)_sorted_(?P<mode>max|p\d{2})_d(?P<depth>\d{3})\.idx$",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_dataset(base: Path) -> BinaryDataset:
    return BinaryDataset(base, _BINARY_NAME, vocab_manager=None)


def _oracle_bytes(
    base: Path, reductions: List[LengthReduction], depth: int
) -> dict:
    """Python-loop oracle: pre-pass + compute + encode, executed inline.

    Mirrors what :func:`build_sorted_index_bytes` does, but spelled out
    in the test so an implementation regression that e.g. drops a
    reduction or shuffles arrays surfaces as a byte-mismatch instead of
    a self-consistent-but-wrong dict.
    """
    section_variant_counts = _count_variants_per_section(base, _BINARY_NAME)
    num_sections = int(section_variant_counts.size)
    dataset = _open_dataset(base)
    with dataset.open_session() as session:
        per_mode_lengths = compute_reduced_lengths(
            session,
            num_sections=num_sections,
            section_variant_counts=section_variant_counts,
            depth=depth,
            reductions=reductions,
        )
    return {
        red: encode_sorted_index(per_mode_lengths[red]) for red in reductions
    }


# ---------------------------------------------------------------------------
# build_sorted_index_bytes
# ---------------------------------------------------------------------------


def test_build_returns_one_blob_per_reduction(tmp_path: Path) -> None:
    """Returned dict has exactly the requested reductions as keys."""
    base = build_combined_fixture(tmp_path)
    dataset = _open_dataset(base)
    with dataset.open_session() as session:
        out = build_sorted_index_bytes(
            session, base, _BINARY_NAME,
            reductions=[_MAX, _P50, _P95],
            depth=3,
        )
    assert set(out.keys()) == {_MAX, _P50, _P95}
    for blob in out.values():
        assert isinstance(blob, (bytes, bytearray))


def test_build_blobs_parse_clean(tmp_path: Path) -> None:
    """Every blob is parseable by :func:`parse_header` without raising."""
    base = build_combined_fixture(tmp_path)
    dataset = _open_dataset(base)
    with dataset.open_session() as session:
        out = build_sorted_index_bytes(
            session, base, _BINARY_NAME,
            reductions=[_MAX, _P50, _P95],
            depth=3,
        )
    for blob in out.values():
        min_length, counts, body_offset = parse_header(blob)
        assert body_offset == 8 + 4 * counts.size
        assert counts.dtype == np.uint32
        assert min_length >= 0


def test_build_bytes_match_oracle(tmp_path: Path) -> None:
    """Per-mode blobs are byte-equal to an inline pre-pass + compute + encode oracle.

    The oracle re-runs the same pipeline against the same fixture; any
    drift in the production builder's call sequence (e.g. missing
    pre-pass, wrong reduction list, wrong encoder) surfaces here.
    """
    base = build_combined_fixture(tmp_path)
    dataset = _open_dataset(base)
    with dataset.open_session() as session:
        out = build_sorted_index_bytes(
            session, base, _BINARY_NAME,
            reductions=[_MAX, _P50, _P95],
            depth=3,
        )
    oracle = _oracle_bytes(base, [_MAX, _P50, _P95], depth=3)
    for red in [_MAX, _P50, _P95]:
        assert bytes(out[red]) == bytes(oracle[red]), f"{red}: byte-mismatch"


def test_build_reader_round_trip(tmp_path: Path) -> None:
    """Reader buckets match a :func:`np.bincount` oracle over the same lengths.

    Round-trip path: build -> write to temp file -> open reader ->
    compare ``count_at`` per length against a bincount on the same
    per-mode length array.  Validates both the wire encode AND the
    reader's bucket lookup arithmetic against the same source of
    truth.
    """
    base = build_combined_fixture(tmp_path)

    # Compute per-mode length arrays once via the oracle path and
    # reuse them for both the builder call (via the inline oracle) AND
    # the reader-side bincount comparison.
    section_variant_counts = _count_variants_per_section(base, _BINARY_NAME)
    num_sections = int(section_variant_counts.size)
    dataset = _open_dataset(base)
    with dataset.open_session() as session:
        per_mode_lengths = compute_reduced_lengths(
            session,
            num_sections=num_sections,
            section_variant_counts=section_variant_counts,
            depth=3,
            reductions=[_MAX, _P50, _P95],
        )
        out = build_sorted_index_bytes(
            session, base, _BINARY_NAME,
            reductions=[_MAX, _P50, _P95],
            depth=3,
        )

    for red in [_MAX, _P50, _P95]:
        path = tmp_path / f"rt_{red.filename_tag()}.idx"
        path.write_bytes(out[red])
        rdr = SortedIndexReader(path, reduction=red, depth=3)
        lengths = per_mode_lengths[red]
        if lengths.size == 0:
            assert rdr.total_sections() == 0
            continue
        min_len = int(lengths.min())
        max_len = int(lengths.max())
        # Oracle bucket counts via bincount on the same array.
        bin_counts = np.bincount(
            lengths - min_len, minlength=(max_len - min_len + 1)
        )
        assert rdr.min_length == min_len
        assert rdr.max_length == max_len
        assert rdr.total_sections() == int(bin_counts.sum())
        for offset, expected in enumerate(bin_counts):
            assert rdr.count_at(min_len + offset) == int(expected), (
                f"{red}: bucket at length {min_len + offset} disagrees"
            )


def test_build_multi_mode_single_compute_call(tmp_path: Path) -> None:
    """``build_sorted_index_bytes`` invokes ``compute_reduced_lengths`` once.

    Plan §D8 amortisation: K reductions share ONE Stage 1+2 walk.  This
    test pins the *builder*'s contract -- the underlying compute
    function is called exactly once across all requested reductions,
    NOT once per reduction.
    """
    base = build_combined_fixture(tmp_path)
    dataset = _open_dataset(base)

    real_compute = compute_reduced_lengths
    call_counter = {"n": 0}

    def _counting_compute(*args, **kwargs):
        call_counter["n"] += 1
        return real_compute(*args, **kwargs)

    with patch(
        "tokenizer.aligned_data.sorted_index._builder.compute_reduced_lengths",
        side_effect=_counting_compute,
    ):
        with dataset.open_session() as session:
            build_sorted_index_bytes(
                session, base, _BINARY_NAME,
                reductions=[_MAX, _P50, _P95],
                depth=3,
            )
    assert call_counter["n"] == 1, (
        f"expected 1 compute_reduced_lengths call for 3 reductions; "
        f"got {call_counter['n']}"
    )


def test_build_empty_reductions_returns_empty_dict(tmp_path: Path) -> None:
    """Empty reductions list short-circuits to an empty dict (no walk)."""
    base = build_combined_fixture(tmp_path)
    dataset = _open_dataset(base)

    with patch(
        "tokenizer.aligned_data.sorted_index._builder.compute_reduced_lengths",
    ) as mock_compute:
        with dataset.open_session() as session:
            out = build_sorted_index_bytes(
                session, base, _BINARY_NAME,
                reductions=[],
                depth=3,
            )
    assert out == {}
    mock_compute.assert_not_called()


# ---------------------------------------------------------------------------
# write_sorted_index_files
# ---------------------------------------------------------------------------


def test_write_default_output_dir_is_input_dir(tmp_path: Path) -> None:
    """Without ``output_dir`` files land next to the memmap sidecars."""
    base = build_combined_fixture(tmp_path)
    written = write_sorted_index_files(
        base, _BINARY_NAME,
        reductions=[_MAX, _P95],
        depth=3,
    )
    assert set(written.keys()) == {_MAX, _P95}
    for red, path in written.items():
        assert path.parent == base, (
            f"{red}: wrote to {path.parent}, expected {base}"
        )
        assert path.is_file()
        m = _FILENAME_RE.match(path.name)
        assert m is not None, f"{red}: filename {path.name} fails grammar"
        assert m.group("binary") == _BINARY_NAME
        assert m.group("depth") == "003"
        assert m.group("mode") == red.filename_tag()


def test_write_explicit_output_dir(tmp_path: Path) -> None:
    """``output_dir`` overrides the memmap dir; no .idx files in memmap dir."""
    base = build_combined_fixture(tmp_path)
    out_dir = tmp_path / "indexes_elsewhere"
    written = write_sorted_index_files(
        base, _BINARY_NAME,
        reductions=[_MAX, _P50],
        depth=3,
        output_dir=out_dir,
    )
    for red, path in written.items():
        assert path.parent == out_dir, (
            f"{red}: wrote to {path.parent}, expected {out_dir}"
        )
        assert path.is_file()
    # Affirmative check: no .idx files in the memmap dir itself.
    sidecar_idx = list(base.glob("*_sorted_*.idx"))
    assert sidecar_idx == [], (
        f"output_dir override leaked .idx files into memmap dir: {sidecar_idx}"
    )


def test_write_bytes_match_build_output(tmp_path: Path) -> None:
    """Written file bytes are byte-equal to :func:`build_sorted_index_bytes`."""
    base = build_combined_fixture(tmp_path)
    dataset = _open_dataset(base)
    with dataset.open_session() as session:
        in_memory = build_sorted_index_bytes(
            session, base, _BINARY_NAME,
            reductions=[_MAX, _P50, _P95],
            depth=3,
        )
    out_dir = tmp_path / "out"
    written = write_sorted_index_files(
        base, _BINARY_NAME,
        reductions=[_MAX, _P50, _P95],
        depth=3,
        output_dir=out_dir,
    )
    for red in [_MAX, _P50, _P95]:
        assert written[red].read_bytes() == bytes(in_memory[red]), (
            f"{red}: file bytes diverge from build_sorted_index_bytes"
        )


def test_write_creates_output_dir(tmp_path: Path) -> None:
    """A non-existent ``output_dir`` is created on demand."""
    base = build_combined_fixture(tmp_path)
    out_dir = tmp_path / "deep" / "tree" / "out"
    assert not out_dir.exists()
    written = write_sorted_index_files(
        base, _BINARY_NAME,
        reductions=[_MAX],
        depth=3,
        output_dir=out_dir,
    )
    assert out_dir.is_dir()
    assert written[_MAX].is_file()


def test_write_empty_reductions(tmp_path: Path) -> None:
    """Empty reductions list: no files, no session opened."""
    base = build_combined_fixture(tmp_path)
    out_dir = tmp_path / "out"
    written = write_sorted_index_files(
        base, _BINARY_NAME,
        reductions=[],
        depth=3,
        output_dir=out_dir,
    )
    assert written == {}
    # Empty reductions short-circuits BEFORE the output_dir is created
    # (no work to do).  Asserting this catches accidental mkdir-on-empty.
    assert not out_dir.exists()


def test_write_depth_zero_padding(tmp_path: Path) -> None:
    """``depth`` is zero-padded to three digits in the filename (plan §D5).

    Filenames lexsort by depth only when the depth tag has a fixed
    width.  Pin the width to 3 digits via this test so a sloppy
    ``f"_d{depth}"`` regression is loud.
    """
    base = build_combined_fixture(tmp_path)
    written = write_sorted_index_files(
        base, _BINARY_NAME,
        reductions=[_MAX],
        depth=7,
    )
    path = written[_MAX]
    assert path.name.endswith("_d007.idx"), (
        f"depth zero-pad regression: {path.name}"
    )
