"""Tests for ``tokenizer.aligned_data._writers.pack_v1_entry``.

The packer is the pure-function single source of truth for the on-wire
8-byte v1 index entry. ``write_index_entry`` is now a thin wrapper that
forwards to it; both must produce bit-for-bit identical bytes.

Approach for equivalence: pack via ``pack_v1_entry`` and verify the
bytes match what ``write_index_entry`` writes to a ``BytesIO``. Any
divergence between the two would be a regression of the refactor.
"""

from __future__ import annotations

import io
import struct

import pytest

from tokenizer.aligned_data._writers import pack_v1_entry
from tokenizer.aligned_data.binary_format import IndexEntrySkip
from tokenizer.aligned_data.index_format import (
    MAX_NORMAL_REAL_LENGTH,
    SENTINEL_LENGTH,
)
from tokenizer.aligned_data.io import write_index_entry

# Mirror the writer's private constants so test failures point at intent.
_MAX_OVERLONG_REAL_LENGTH = 0xFFFFFF << 2  # 67_108_860 bytes ≈ 64 MiB
_MAX_OFFSET = ((1 << 40) - 1) << 2  # ~4 TiB


# ---------------------------------------------------------------------------
# pack_v1_entry / write_index_entry equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0, 4, 8, 1024, _MAX_OFFSET])
@pytest.mark.parametrize(
    "length",
    [4, 8, 64, 100_000, MAX_NORMAL_REAL_LENGTH, MAX_NORMAL_REAL_LENGTH + 4, 1 << 20, _MAX_OVERLONG_REAL_LENGTH],
)
@pytest.mark.parametrize("avg_len", [0, 16, 4096])
def test_pack_matches_write_index_entry(offset: int, length: int, avg_len: int):
    """``pack_v1_entry`` and ``write_index_entry`` must produce identical bytes.

    Pins the refactor invariant: extracting the packer cannot change
    the bytes that land in ``_index.bin``.
    """
    packed = pack_v1_entry(offset, length, avg_len)

    buf = io.BytesIO()
    write_index_entry(buf, offset, length, avg_len)
    written = buf.getvalue()

    assert packed == written
    assert len(packed) == 8


# ---------------------------------------------------------------------------
# Bit-exact byte layout for the simplest valid record
# ---------------------------------------------------------------------------


def test_minimum_valid_entry():
    """``offset=0, length=4, avg=0`` — the smallest valid record.

    offset_shifted=0 → ``\\x00\\x00\\x00\\x00\\x00``
    length_shifted=1 → ``\\x01\\x00``
    avg=0            → ``\\x00``
    """
    assert pack_v1_entry(0, 4, 0) == b"\x00\x00\x00\x00\x00\x01\x00\x00"


def test_bit_exact_layout_with_avg():
    """Pin a known triple including a non-zero avg byte clamping."""
    # offset=4 → offset_shifted=1; length=8 → length_shifted=2; avg=32 → 32>>4=2.
    assert pack_v1_entry(4, 8, 32) == b"\x01\x00\x00\x00\x00\x02\x00\x02"


# ---------------------------------------------------------------------------
# Overlong sentinel transition
# ---------------------------------------------------------------------------


def test_overlong_sentinel_length_field_zero():
    """Real length one step above the normal cap sets ``length_field == SENTINEL_LENGTH``."""
    packed = pack_v1_entry(0, MAX_NORMAL_REAL_LENGTH + 4, 0)
    length_field = struct.unpack("<H", packed[5:7])[0]
    assert length_field == SENTINEL_LENGTH


def test_max_normal_is_not_sentinel():
    """At exactly the normal cap, ``length_field == 0xFFFF`` (distinct from sentinel)."""
    packed = pack_v1_entry(0, MAX_NORMAL_REAL_LENGTH, 0)
    length_field = struct.unpack("<H", packed[5:7])[0]
    assert length_field == 0xFFFF
    assert length_field != SENTINEL_LENGTH


def test_overlong_at_cap_is_sentinel():
    """The exact 64 MiB overlong cap is the LAST acceptable length; sentinel set, no raise."""
    packed = pack_v1_entry(0, _MAX_OVERLONG_REAL_LENGTH, 0)
    length_field = struct.unpack("<H", packed[5:7])[0]
    assert length_field == SENTINEL_LENGTH


# ---------------------------------------------------------------------------
# Cap-overflow IndexEntrySkip propagation
# ---------------------------------------------------------------------------


def test_overlong_cap_overflow_raises():
    """One 4-byte step past the overlong cap raises ``IndexEntrySkip``."""
    over_cap = _MAX_OVERLONG_REAL_LENGTH + 4
    with pytest.raises(IndexEntrySkip) as info:
        pack_v1_entry(0, over_cap, 0)
    assert info.value.reason == "overlong_length_overflow"
    assert info.value.value == over_cap


def test_offset_cap_overflow_raises():
    """An offset above the u40 cap raises ``IndexEntrySkip``."""
    over_offset = 1 << 44  # > MAX_OFFSET (~4 TiB), still 4-aligned
    assert over_offset > _MAX_OFFSET
    with pytest.raises(IndexEntrySkip) as info:
        pack_v1_entry(over_offset, 8, 0)
    assert info.value.reason == "offset_overflow"
    assert info.value.value == over_offset


# ---------------------------------------------------------------------------
# Alignment / zero-length programmer-error assertions
# ---------------------------------------------------------------------------


def test_unaligned_offset_asserts():
    """Misaligned offsets reveal a writer bug; raise unconditionally."""
    with pytest.raises(AssertionError):
        pack_v1_entry(1, 4, 0)


def test_unaligned_length_asserts():
    """Misaligned lengths reveal a writer bug; raise unconditionally."""
    with pytest.raises(AssertionError):
        pack_v1_entry(0, 5, 0)


def test_zero_length_asserts():
    """A real length of 0 is impossible; the smallest padded record is 8 bytes."""
    with pytest.raises(AssertionError):
        pack_v1_entry(0, 0, 0)
