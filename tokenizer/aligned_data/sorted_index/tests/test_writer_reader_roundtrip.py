"""Writer <-> reader byte-format cross-check via actual file I/O.

Closes the round-1 audit finding that previously the writer
(:func:`encode_sorted_index`) and the reader (:func:`parse_header` /
:class:`SortedIndexReader`) were tested independently: this fixture
drives :func:`write_sorted_index_files` to disk, reads the raw bytes
back, and independently re-computes the expected bytes via
:func:`compute_reduced_lengths` + :func:`encode_sorted_index` -- so a
regression in the writer's session-lifecycle, filename construction,
file-write code path, or any new layer between the pure encoder and
the on-disk file surfaces as a byte-mismatch here.

Single concern: validate that the *actually-written file's contents*
equal the *expected wire-format bytes* across every supported
reduction shape (single MAX, single PERCENTILE, multi-mode), plus
verify the reader-side derived metadata (``min_length``,
``total_sections``, per-length ``count_at`` distribution) against the
same source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.sorted_index import (
    LengthReduction,
    ReductionKind,
    SortedIndexReader,
    compute_reduced_lengths,
    encode_sorted_index,
    write_sorted_index_files,
)
from tokenizer.aligned_data.sorted_index._length_compute import (
    _count_variants_per_section,
)

from .fixtures import build_combined_fixture


_BINARY_NAME = "sortbin"
_DEPTH = 3

_MAX = LengthReduction(kind=ReductionKind.MAX)
_P05 = LengthReduction(kind=ReductionKind.PERCENTILE, percentile=5)
_P50 = LengthReduction(kind=ReductionKind.PERCENTILE, percentile=50)
_P95 = LengthReduction(kind=ReductionKind.PERCENTILE, percentile=95)


# ---------------------------------------------------------------------------
# Independent recompute (the "expected" side of the cross-check)
# ---------------------------------------------------------------------------


def _independent_recompute(
    base: Path,
    binary_name: str,
    reductions: List[LengthReduction],
    depth: int,
) -> dict:
    """Re-run compute + encode in this file with no shared helpers.

    Mirrors what the writer does under the hood -- pre-pass over
    sections.bin for variant counts, then ONE Stage 1+2 walk across
    every reduction, then per-mode wire encode.  Spelled out inline
    (no call into ``_builder``) so a writer-side drift between the
    in-memory bytes path and the file-write path surfaces here as a
    byte-mismatch instead of a self-consistent-but-wrong agreement.

    Returns ``{reduction -> (bytes, per_mode_lengths_array)}`` so the
    reader-side bucket-distribution check below can reuse the same
    length arrays without re-running the (expensive) compute walk.
    """
    section_variant_counts = _count_variants_per_section(base, binary_name)
    num_sections = int(section_variant_counts.size)
    dataset = BinaryDataset(base, binary_name, vocab_manager=None)
    with dataset.open_session() as session:
        per_mode_lengths = compute_reduced_lengths(
            session,
            num_sections=num_sections,
            section_variant_counts=section_variant_counts,
            depth=depth,
            reductions=reductions,
        )
    return {
        red: (encode_sorted_index(per_mode_lengths[red]), per_mode_lengths[red])
        for red in reductions
    }


def _read_written_bytes(
    memmap_dir: Path,
    binary_name: str,
    reduction: LengthReduction,
    depth: int,
) -> bytes:
    """Read the raw on-disk bytes of the canonical-grammar sidecar."""
    path = memmap_dir / (
        f"{binary_name}_sorted_{reduction.filename_tag()}"
        f"_d{depth:03d}.idx"
    )
    assert path.is_file(), f"writer did not produce {path}"
    return path.read_bytes()


# ---------------------------------------------------------------------------
# Tests -- parametrised over single + multi-mode reduction shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reductions",
    [
        pytest.param([_MAX], id="max"),
        pytest.param([_P05], id="p05"),
        pytest.param([_P50], id="p50"),
        pytest.param([_MAX, _P95], id="max+p95"),
    ],
)
def test_written_bytes_match_independent_recompute(
    tmp_path: Path,
    reductions: List[LengthReduction],
) -> None:
    """File-on-disk bytes equal an inline ``compute + encode`` oracle.

    Every reduction shape (single MAX, single PERCENTILE, multi-mode)
    is exercised so the per-shape file-write path is covered.
    """
    base = build_combined_fixture(tmp_path)
    write_sorted_index_files(
        base, _BINARY_NAME,
        reductions=reductions,
        depth=_DEPTH,
    )
    expected = _independent_recompute(base, _BINARY_NAME, reductions, _DEPTH)
    for red in reductions:
        on_disk = _read_written_bytes(base, _BINARY_NAME, red, _DEPTH)
        expected_bytes, _ = expected[red]
        assert on_disk == expected_bytes, (
            f"{red}: file-on-disk bytes diverge from "
            f"compute_reduced_lengths + encode_sorted_index oracle"
        )


@pytest.mark.parametrize(
    "reductions",
    [
        pytest.param([_MAX], id="max"),
        pytest.param([_P05], id="p05"),
        pytest.param([_P50], id="p50"),
        pytest.param([_MAX, _P95], id="max+p95"),
    ],
)
def test_reader_metadata_matches_recomputed_lengths(
    tmp_path: Path,
    reductions: List[LengthReduction],
) -> None:
    """Opening the written file via :class:`SortedIndexReader` recovers
    ``min_length`` / ``total_sections`` / per-length ``count_at``
    matching an independent oracle over the same length arrays.

    The oracle's source of truth is the same ``per_mode_lengths``
    array the writer encoded -- so any drift between what the writer
    serialised and what the reader can recover surfaces here.
    """
    base = build_combined_fixture(tmp_path)
    write_sorted_index_files(
        base, _BINARY_NAME,
        reductions=reductions,
        depth=_DEPTH,
    )
    expected = _independent_recompute(base, _BINARY_NAME, reductions, _DEPTH)

    for red in reductions:
        path = base / (
            f"{_BINARY_NAME}_sorted_{red.filename_tag()}_d{_DEPTH:03d}.idx"
        )
        reader = SortedIndexReader(path, reduction=red, depth=_DEPTH)
        _, lengths = expected[red]
        num_sections = int(lengths.size)

        # min_length matches the independently-computed minimum.
        expected_min = int(lengths.min()) if num_sections > 0 else 0
        assert reader.min_length == expected_min, (
            f"{red}: reader.min_length={reader.min_length}, "
            f"expected {expected_min}"
        )

        # total_sections sums to num_sections (every section lands in
        # exactly one length bucket).
        assert reader.total_sections() == num_sections, (
            f"{red}: reader.total_sections()={reader.total_sections()}, "
            f"expected {num_sections}"
        )

        # count_at over every valid length sums back to num_sections.
        if num_sections == 0:
            continue
        expected_max = int(lengths.max())
        summed = 0
        for L in range(expected_min, expected_max + 1):
            summed += reader.count_at(L)
        assert summed == num_sections, (
            f"{red}: count_at sum {summed} != {num_sections} over "
            f"[{expected_min}, {expected_max}]"
        )

        # Per-bucket count matches a bincount oracle over the same
        # length array -- pins the writer's stable-sort + bincount path
        # at the reader's exposed bucket lookup.
        bin_counts = np.bincount(
            lengths - expected_min,
            minlength=(expected_max - expected_min + 1),
        )
        for offset, expected_count in enumerate(bin_counts):
            L = expected_min + offset
            assert reader.count_at(L) == int(expected_count), (
                f"{red}: count_at({L})={reader.count_at(L)}, "
                f"expected {int(expected_count)}"
            )
