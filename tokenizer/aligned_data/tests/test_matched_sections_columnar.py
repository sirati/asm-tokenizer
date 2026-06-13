"""Equivalence: columnar sections decode vs the scalar section parser.

Pins :func:`tokenizer.aligned_data.matched_sections_columnar.
parse_sections_columnar` to :func:`~tokenizer.aligned_data.
matched_sections_bin.parse_section_bin` on production-writer fixtures
(every sorted_index edge-case corpus: 0-variant sections, odd/even
variant counts for the jump-table pad, MISSING sentinel slots,
call-target tables).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.matched_sections_bin import (
    iter_sections_bin,
)
from tokenizer.aligned_data.matched_sections_columnar import (
    parse_sections_columnar,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_0_variant_section_fixture,
    build_combined_fixture,
    build_many_variant_section_fixture,
    build_missing_variant_index_fixture,
)


def _binary_name(base: Path) -> str:
    (idx,) = (
        p
        for p in base.glob("*_index.bin")
        if not p.name.endswith("_unmatched_index.bin")
    )
    return idx.name[: -len("_index.bin")]


def _assert_columnar_matches_scalar(base: Path) -> None:
    name = _binary_name(base)
    starts, lengths = read_csv_section_index_arrays(
        base / f"{name}_index.bin"
    )
    blob = np.fromfile(base / f"{name}_sections.bin", dtype=np.uint8)

    cols = parse_sections_columnar(blob, starts, lengths)

    scalar = []
    for i, section in enumerate(iter_sections_bin(base / f"{name}_sections.bin")):
        if i >= len(starts):
            break
        scalar.append(section)
    assert len(scalar) == len(starts)

    for s, section in enumerate(scalar):
        assert cols.function_name_ptr[s] == section.function_name_ptr
        assert bool(cols.is_duplicated[s]) == section.is_duplicated
        assert cols.n_call_targets[s] == len(section.call_targets)
        assert cols.n_variants[s] == len(section.variants)

        c0, c1 = cols.ct_offsets[s], cols.ct_offsets[s + 1]
        for j, ct in enumerate(section.call_targets):
            k = c0 + j
            assert cols.ct_function_name_ptr[k] == ct.function_name_ptr
            assert cols.ct_function_section_ptr[k] == ct.function_section_ptr
            assert cols.ct_type[k] == int(ct.type)
            assert cols.ct_is_matched[k] == ct.is_matched
        assert c1 - c0 == len(section.call_targets)

        v0, v1 = cols.var_offsets[s], cols.var_offsets[s + 1]
        for j, variant in enumerate(section.variants):
            v = v0 + j
            assert cols.var_ref_offset[v] == variant.variant_ref_offset
            assert (
                cols.var_data_offset_shifted[v]
                == variant.data_offset_shifted
            )
            assert cols.var_n_calls[v] == len(variant.per_call_entries)
            p0, p1 = cols.pce_offsets[v], cols.pce_offsets[v + 1]
            got = list(
                zip(
                    cols.pce_called_idx[p0:p1].tolist(),
                    cols.pce_section_variant_index[p0:p1].tolist(),
                )
            )
            assert got == list(variant.per_call_entries)
        assert v1 - v0 == len(section.variants)


@pytest.mark.parametrize(
    "builder",
    [
        build_0_variant_section_fixture,
        build_many_variant_section_fixture,
        build_missing_variant_index_fixture,
        build_combined_fixture,
    ],
)
def test_columnar_matches_scalar(builder, tmp_path: Path) -> None:
    _assert_columnar_matches_scalar(builder(tmp_path))


def test_length_validation_catches_drift(tmp_path: Path) -> None:
    base = build_combined_fixture(tmp_path)
    name = _binary_name(base)
    starts, lengths = read_csv_section_index_arrays(
        base / f"{name}_index.bin"
    )
    blob = np.fromfile(base / f"{name}_sections.bin", dtype=np.uint8)
    bad_lengths = lengths.copy()
    bad_lengths[0] += 4
    with pytest.raises(ValueError, match="columnar decode drift"):
        parse_sections_columnar(blob, starts, bad_lengths)


def test_empty_offsets() -> None:
    cols = parse_sections_columnar(
        np.zeros(16, dtype=np.uint8), np.zeros(0, dtype=np.int64)
    )
    assert cols.n_variants.size == 0
    assert cols.var_offsets.tolist() == [0]
    assert cols.pce_offsets.tolist() == [0]
