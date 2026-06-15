"""Geometry format + reader unit tests: 3-block body, shared CSR, errors.

Covers the on-disk geometry pair codec (:mod:`.._geometry_format`) and
the lazy reader (:mod:`.._geometry_reader`) directly, without the
generator: synthetic triple / CSR arrays exercise the shared jump-table
addressing across the three parallel axes on 0-variant sections,
many-variant sections, and the last section; prelude validation rejects
a wrong magic / version; the body-not-divisible-by-3 and CSR-terminator
guards fire; the missing-file path names the generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.memmap_format import (
    MEMMAP_FORMAT_VERSION,
    encode_bin_prelude,
)
from tokenizer.aligned_data.realized_lengths import (
    GEOMETRY_MATCHED_ARM,
    GEOMETRY_UNMATCHED_ARM,
    RealizedGeometryReader,
    realized_geometry_present,
    require_realized_geometry,
)
from tokenizer.aligned_data.realized_lengths._geometry_format import (
    GEOMETRY_DTYPE,
    read_geometry_pair,
    write_geometry_pair,
)


def _write(tmp_path, body, ids, values, csr, *, arm=GEOMETRY_MATCHED_ARM, name="bin"):
    write_geometry_pair(
        arm.geometry_path(tmp_path, name),
        arm.index_path(tmp_path, name),
        body_lengths=np.asarray(body, dtype=np.uint32),
        id_counts=np.asarray(ids, dtype=np.uint32),
        value_counts=np.asarray(values, dtype=np.uint32),
        csr_offsets=np.asarray(csr, dtype=np.uint32),
    )
    return tmp_path, name


# ---------------------------------------------------------------------------
# CSR addressing across the three parallel axes
# ---------------------------------------------------------------------------


def test_three_blocks_share_one_csr(tmp_path: Path) -> None:
    # Sections: [0 variants], [3 variants], [1 variant], [2 variants (last)].
    body = [10, 20, 30, 40, 50, 60]
    ids = [1, 2, 3, 4, 5, 6]
    values = [100, 200, 300, 400, 500, 600]
    csr = [0, 0, 3, 4, 6]
    base, name = _write(tmp_path, body, ids, values, csr)
    r = RealizedGeometryReader.open(base, name, GEOMETRY_MATCHED_ARM)
    try:
        assert r.n_sections == 4
        np.testing.assert_array_equal(r.body_lengths, body)
        np.testing.assert_array_equal(r.id_counts, ids)
        np.testing.assert_array_equal(r.value_counts, values)
        np.testing.assert_array_equal(r.csr_offsets, csr)
        # Section 1: 3 variants on every axis via the SHARED CSR
        # (csr[1]:csr[2] == 0:3).
        b, i, v = r.per_section(1)
        np.testing.assert_array_equal(b, [10, 20, 30])
        np.testing.assert_array_equal(i, [1, 2, 3])
        np.testing.assert_array_equal(v, [100, 200, 300])
        # Empty + last section (csr[3]:csr[4] == 4:6).
        eb, ei, ev = r.per_section(0)
        assert eb.size == 0 and ei.size == 0 and ev.size == 0
        lb, li, lv = r.per_section(3)
        np.testing.assert_array_equal(lb, [50, 60])
        np.testing.assert_array_equal(li, [5, 6])
        np.testing.assert_array_equal(lv, [500, 600])
        # Scalar triple lookup (section 1 variant 2 == row 2).
        assert r.geometry(1, 2) == (30, 3, 300)
        # Section 3 variant 1 == row 5.
        assert r.geometry(3, 1) == (60, 6, 600)
    finally:
        r.close()


def test_per_section_views_are_zero_copy(tmp_path: Path) -> None:
    base, name = _write(tmp_path, [7, 8, 9], [1, 1, 1], [2, 2, 2], [0, 3])
    r = RealizedGeometryReader.open(base, name, GEOMETRY_MATCHED_ARM)
    try:
        b, i, v = r.per_section(0)
        assert b.base is r.body_lengths or b.base is r.body_lengths.base
        assert i.base is r.id_counts or i.base is r.id_counts.base
        assert v.base is r.value_counts or v.base is r.value_counts.base
    finally:
        r.close()


def test_out_of_range_section_and_variant_raise(tmp_path: Path) -> None:
    base, name = _write(tmp_path, [1, 2], [3, 4], [5, 6], [0, 1, 2])
    r = RealizedGeometryReader.open(base, name, GEOMETRY_MATCHED_ARM)
    try:
        with pytest.raises(IndexError):
            r.per_section(2)
        with pytest.raises(IndexError):
            r.per_section(-1)
        with pytest.raises(IndexError):
            r.geometry(0, 1)  # section 0 has a single variant
    finally:
        r.close()


def test_empty_arm_roundtrip(tmp_path: Path) -> None:
    base, name = _write(tmp_path, [], [], [], [0], arm=GEOMETRY_UNMATCHED_ARM)
    r = RealizedGeometryReader.open(base, name, GEOMETRY_UNMATCHED_ARM)
    try:
        assert r.n_sections == 0
        assert r.body_lengths.size == 0
        assert r.id_counts.size == 0
        assert r.value_counts.size == 0
    finally:
        r.close()


# ---------------------------------------------------------------------------
# Prelude + structural validation
# ---------------------------------------------------------------------------


def test_wrong_geometry_magic_rejected(tmp_path: Path) -> None:
    geometry_path = GEOMETRY_MATCHED_ARM.geometry_path(tmp_path, "bin")
    index_path = GEOMETRY_MATCHED_ARM.index_path(tmp_path, "bin")
    write_geometry_pair(
        geometry_path, index_path,
        body_lengths=np.array([1], dtype=np.uint32),
        id_counts=np.array([2], dtype=np.uint32),
        value_counts=np.array([3], dtype=np.uint32),
        csr_offsets=np.array([0, 1], dtype=np.uint32),
    )
    geometry_path.write_bytes(
        encode_bin_prelude(b"XXXX") + geometry_path.read_bytes()[16:]
    )
    with pytest.raises(ValueError, match="unexpected.*magic"):
        read_geometry_pair(geometry_path, index_path)


def test_wrong_version_rejected(tmp_path: Path) -> None:
    geometry_path = GEOMETRY_MATCHED_ARM.geometry_path(tmp_path, "bin")
    index_path = GEOMETRY_MATCHED_ARM.index_path(tmp_path, "bin")
    write_geometry_pair(
        geometry_path, index_path,
        body_lengths=np.array([1], dtype=np.uint32),
        id_counts=np.array([2], dtype=np.uint32),
        value_counts=np.array([3], dtype=np.uint32),
        csr_offsets=np.array([0, 1], dtype=np.uint32),
    )
    raw = bytearray(index_path.read_bytes())
    raw[4:8] = (MEMMAP_FORMAT_VERSION + 1).to_bytes(4, "little")
    index_path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="format_version"):
        read_geometry_pair(geometry_path, index_path)


def test_csr_terminator_mismatch_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="terminator"):
        write_geometry_pair(
            GEOMETRY_MATCHED_ARM.geometry_path(tmp_path, "bin"),
            GEOMETRY_MATCHED_ARM.index_path(tmp_path, "bin"),
            body_lengths=np.array([1, 2, 3], dtype=np.uint32),
            id_counts=np.array([1, 2, 3], dtype=np.uint32),
            value_counts=np.array([1, 2, 3], dtype=np.uint32),
            csr_offsets=np.array([0, 2], dtype=np.uint32),  # says 2, body is 3
        )


def test_unequal_axis_lengths_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parallel"):
        write_geometry_pair(
            GEOMETRY_MATCHED_ARM.geometry_path(tmp_path, "bin"),
            GEOMETRY_MATCHED_ARM.index_path(tmp_path, "bin"),
            body_lengths=np.array([1, 2, 3], dtype=np.uint32),
            id_counts=np.array([1, 2], dtype=np.uint32),  # short axis
            value_counts=np.array([1, 2, 3], dtype=np.uint32),
            csr_offsets=np.array([0, 3], dtype=np.uint32),
        )


def test_body_not_divisible_by_three_rejected(tmp_path: Path) -> None:
    geometry_path = GEOMETRY_MATCHED_ARM.geometry_path(tmp_path, "bin")
    index_path = GEOMETRY_MATCHED_ARM.index_path(tmp_path, "bin")
    write_geometry_pair(
        geometry_path, index_path,
        body_lengths=np.array([1], dtype=np.uint32),
        id_counts=np.array([2], dtype=np.uint32),
        value_counts=np.array([3], dtype=np.uint32),
        csr_offsets=np.array([0, 1], dtype=np.uint32),
    )
    # Append one stray u32 so the body is 4 elements, not a multiple of 3.
    with open(geometry_path, "ab") as fh:
        fh.write(np.array([99], dtype=GEOMETRY_DTYPE).tobytes())
    with pytest.raises(ValueError, match="not divisible by 3"):
        read_geometry_pair(geometry_path, index_path)


# ---------------------------------------------------------------------------
# Existence / discovery helpers
# ---------------------------------------------------------------------------


def test_missing_files_message_names_generator(tmp_path: Path) -> None:
    assert not realized_geometry_present(tmp_path, "absent", GEOMETRY_MATCHED_ARM)
    with pytest.raises(FileNotFoundError, match="realized-geometry generator"):
        require_realized_geometry(tmp_path, "absent", GEOMETRY_MATCHED_ARM)
    with pytest.raises(FileNotFoundError, match="realized-geometry generator"):
        RealizedGeometryReader.open(tmp_path, "absent", GEOMETRY_MATCHED_ARM)


def test_present_after_write(tmp_path: Path) -> None:
    base, name = _write(tmp_path, [1], [2], [3], [0, 1])
    assert realized_geometry_present(base, name, GEOMETRY_MATCHED_ARM)
    assert not realized_geometry_present(base, name, GEOMETRY_UNMATCHED_ARM)
