"""Format + reader unit tests: CSR addressing, prelude validation, errors.

Covers the on-disk pair codec (:mod:`.._format`) and the lazy reader
(:mod:`.._reader`) directly, without the generator: synthetic length /
CSR arrays exercise the jump-table addressing on 0-variant sections,
many-variant sections, and the last section; prelude validation rejects
a wrong magic / version; the missing-file path names the generator.
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
    MATCHED_ARM,
    UNMATCHED_ARM,
    RealizedLengths,
    realized_lengths_present,
    require_realized_lengths,
)
from tokenizer.aligned_data.realized_lengths._format import (
    LENGTH_DTYPE,
    read_lengths_pair,
    write_lengths_pair,
)


def _write(tmp_path: Path, lengths, csr, *, arm=MATCHED_ARM, name="bin"):
    write_lengths_pair(
        arm.lengths_path(tmp_path, name),
        arm.index_path(tmp_path, name),
        lengths=np.asarray(lengths, dtype=np.uint32),
        csr_offsets=np.asarray(csr, dtype=np.uint32),
    )
    return tmp_path, name


# ---------------------------------------------------------------------------
# CSR addressing
# ---------------------------------------------------------------------------


def test_csr_addressing_mixed_section_shapes(tmp_path: Path) -> None:
    # Sections: [0 variants], [3 variants], [1 variant], [2 variants (last)].
    lengths = [10, 20, 30, 40, 50, 60]
    csr = [0, 0, 3, 4, 6]
    base, name = _write(tmp_path, lengths, csr)
    r = RealizedLengths.open(base, name, MATCHED_ARM)
    try:
        assert r.n_sections == 4
        np.testing.assert_array_equal(r.per_section(0), np.array([], dtype=LENGTH_DTYPE))
        np.testing.assert_array_equal(r.per_section(1), [10, 20, 30])
        np.testing.assert_array_equal(r.per_section(2), [40])
        np.testing.assert_array_equal(r.per_section(3), [50, 60])  # last section
        assert r.length(1, 2) == 30
        assert r.length(3, 1) == 60
        # Raw vectorized accessors.
        np.testing.assert_array_equal(r.lengths, lengths)
        np.testing.assert_array_equal(r.csr_offsets, csr)
    finally:
        r.close()


def test_per_section_view_is_zero_copy(tmp_path: Path) -> None:
    base, name = _write(tmp_path, [7, 8, 9], [0, 3])
    r = RealizedLengths.open(base, name, MATCHED_ARM)
    try:
        view = r.per_section(0)
        # A slice of the body memmap shares its base buffer (zero-copy).
        assert view.base is r.lengths or view.base is r.lengths.base
    finally:
        r.close()


def test_out_of_range_section_and_variant_raise(tmp_path: Path) -> None:
    base, name = _write(tmp_path, [1, 2], [0, 1, 2])
    r = RealizedLengths.open(base, name, MATCHED_ARM)
    try:
        with pytest.raises(IndexError):
            r.per_section(2)
        with pytest.raises(IndexError):
            r.per_section(-1)
        with pytest.raises(IndexError):
            r.length(0, 1)  # section 0 has a single variant
    finally:
        r.close()


def test_empty_arm_roundtrip(tmp_path: Path) -> None:
    # 0-section arm: empty body + single 0 CSR terminator.
    base, name = _write(tmp_path, [], [0], arm=UNMATCHED_ARM)
    r = RealizedLengths.open(base, name, UNMATCHED_ARM)
    try:
        assert r.n_sections == 0
        assert r.lengths.size == 0
    finally:
        r.close()


# ---------------------------------------------------------------------------
# Prelude validation
# ---------------------------------------------------------------------------


def test_wrong_lengths_magic_rejected(tmp_path: Path) -> None:
    lengths_path = MATCHED_ARM.lengths_path(tmp_path, "bin")
    index_path = MATCHED_ARM.index_path(tmp_path, "bin")
    # Valid index, but the lengths file carries a foreign magic.
    write_lengths_pair(
        lengths_path, index_path,
        lengths=np.array([1], dtype=np.uint32),
        csr_offsets=np.array([0, 1], dtype=np.uint32),
    )
    lengths_path.write_bytes(
        encode_bin_prelude(b"XXXX") + lengths_path.read_bytes()[16:]
    )
    with pytest.raises(ValueError, match="unexpected.*magic"):
        read_lengths_pair(lengths_path, index_path)


def test_wrong_version_rejected(tmp_path: Path) -> None:
    lengths_path = MATCHED_ARM.lengths_path(tmp_path, "bin")
    index_path = MATCHED_ARM.index_path(tmp_path, "bin")
    write_lengths_pair(
        lengths_path, index_path,
        lengths=np.array([1], dtype=np.uint32),
        csr_offsets=np.array([0, 1], dtype=np.uint32),
    )
    # Corrupt the version u32 in the index prelude (bytes 4..7).
    raw = bytearray(index_path.read_bytes())
    bad_version = (MEMMAP_FORMAT_VERSION + 1).to_bytes(4, "little")
    raw[4:8] = bad_version
    index_path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="format_version"):
        read_lengths_pair(lengths_path, index_path)


def test_csr_terminator_mismatch_rejected(tmp_path: Path) -> None:
    # write_lengths_pair guards the CSR terminator == body length.
    with pytest.raises(ValueError, match="terminator"):
        write_lengths_pair(
            MATCHED_ARM.lengths_path(tmp_path, "bin"),
            MATCHED_ARM.index_path(tmp_path, "bin"),
            lengths=np.array([1, 2, 3], dtype=np.uint32),
            csr_offsets=np.array([0, 2], dtype=np.uint32),  # says 2, body is 3
        )


# ---------------------------------------------------------------------------
# Existence / discovery helpers
# ---------------------------------------------------------------------------


def test_missing_files_message_names_generator(tmp_path: Path) -> None:
    assert not realized_lengths_present(tmp_path, "absent", MATCHED_ARM)
    with pytest.raises(FileNotFoundError, match="realized-lengths generator"):
        require_realized_lengths(tmp_path, "absent", MATCHED_ARM)
    with pytest.raises(FileNotFoundError, match="realized-lengths generator"):
        RealizedLengths.open(tmp_path, "absent", MATCHED_ARM)


def test_present_after_write(tmp_path: Path) -> None:
    base, name = _write(tmp_path, [1], [0, 1])
    assert realized_lengths_present(base, name, MATCHED_ARM)
    # Unmatched pair was not written -> not present for that arm.
    assert not realized_lengths_present(base, name, UNMATCHED_ARM)
