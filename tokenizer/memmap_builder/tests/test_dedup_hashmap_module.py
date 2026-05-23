"""Direct contract tests for the ``dedup_hashmap`` PyO3 classes.

``HashMapU64U32`` backs the primary side of the content-addressed dedup
in :mod:`tokenizer.memmap_builder._dedup`. ``HashMapU32U32`` backs
``SectionWriter._known_sections`` in
:mod:`tokenizer.aligned_data.matched_sections_bin`. ``HashMapU32U16``
backs the per-Category dedup walk in stage 4 of the batch-vectorized
v2 dataloader. Both legacy maps are exercised indirectly via their
consumers' integration tests; this module is the direct contract check
for the scalar API (``new``/``get``/``set``/``__contains__``/``__len__``)
plus the new surface (``capacity`` constructor, ``lookup``/``insert``
aliases, ``lookup_ndarray``/``insert_ndarray`` batch APIs, ``clean()``
reuse) so a future crate-internal refactor doesn't silently break the
FFI surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from dedup_hashmap import (
    HashMapBoolBool,
    HashMapI32F64,
    HashMapU8U8,
    HashMapU32U16,
    HashMapU32U32,
    HashMapU64U32,
)


class TestHashMapU64U32:
    def test_empty(self) -> None:
        m = HashMapU64U32()
        assert len(m) == 0
        assert 0 not in m
        assert m.get(0) is None

    def test_set_then_get(self) -> None:
        m = HashMapU64U32()
        m.set(2**40 + 7, 1234)
        assert m.get(2**40 + 7) == 1234
        assert (2**40 + 7) in m
        assert len(m) == 1

    def test_overwrite(self) -> None:
        m = HashMapU64U32()
        m.set(5, 1)
        m.set(5, 2)
        assert m.get(5) == 2
        assert len(m) == 1

    def test_get_missing_returns_none(self) -> None:
        m = HashMapU64U32()
        m.set(1, 10)
        assert m.get(2) is None

    def test_large_keys_fit_u64(self) -> None:
        m = HashMapU64U32()
        big = (1 << 64) - 1
        m.set(big, 7)
        assert m.get(big) == 7

    def test_rejects_oversize_value(self) -> None:
        m = HashMapU64U32()
        with pytest.raises(OverflowError):
            m.set(1, 1 << 32)

    def test_capacity_constructor_respected(self) -> None:
        # Preallocated capacity must not change observable behavior.
        m = HashMapU64U32(capacity=1024)
        for k in range(100):
            m.set(k, k + 1)
        assert len(m) == 100
        for k in range(100):
            assert m.get(k) == k + 1


class TestHashMapU32U32:
    def test_empty(self) -> None:
        m = HashMapU32U32()
        assert len(m) == 0
        assert 0 not in m
        assert m.get(0) is None

    def test_set_then_get(self) -> None:
        m = HashMapU32U32()
        m.set(42, 99)
        assert m.get(42) == 99
        assert 42 in m
        assert len(m) == 1

    def test_overwrite(self) -> None:
        m = HashMapU32U32()
        m.set(5, 1)
        m.set(5, 2)
        assert m.get(5) == 2
        assert len(m) == 1

    def test_get_missing_returns_none(self) -> None:
        m = HashMapU32U32()
        m.set(1, 10)
        assert m.get(2) is None

    def test_distinct_keys_independent(self) -> None:
        m = HashMapU32U32()
        for k in range(100):
            m.set(k, k * 2)
        assert len(m) == 100
        for k in range(100):
            assert m.get(k) == k * 2
            assert k in m

    def test_large_keys_fit_u32(self) -> None:
        m = HashMapU32U32()
        big = (1 << 32) - 1
        m.set(big, big)
        assert m.get(big) == big

    def test_rejects_oversize_key(self) -> None:
        m = HashMapU32U32()
        with pytest.raises(OverflowError):
            m.set(1 << 32, 0)

    def test_rejects_oversize_value(self) -> None:
        m = HashMapU32U32()
        with pytest.raises(OverflowError):
            m.set(0, 1 << 32)


class TestHashMapU32U16:
    """The Category-side dedup map used by the stage-4 batch decoder."""

    def test_capacity_constructor(self) -> None:
        # capacity=N should be accepted; behaviour identical to default.
        m = HashMapU32U16(capacity=64)
        assert len(m) == 0
        m.insert(7, 3)
        assert m.lookup(7) == 3
        assert 7 in m

    def test_clean_reuses_storage(self) -> None:
        m = HashMapU32U16(capacity=64)
        for k in range(50):
            m.insert(k, k + 1)
        assert len(m) == 50
        m.clean()
        assert len(m) == 0
        assert m.lookup(0) is None
        # Reuse the same map for a second batch.
        for k in range(20):
            m.insert(k * 10, k + 100)
        assert len(m) == 20
        assert m.lookup(0) == 100
        assert m.lookup(10) == 101
        assert m.lookup(11) is None

    def test_lookup_ndarray_with_miss_sentinel(self) -> None:
        m = HashMapU32U16(capacity=16)
        m.insert(1, 10)
        m.insert(2, 20)
        keys = np.array([1, 2, 3, 4], dtype=np.uint32)
        out = m.lookup_ndarray(keys)
        assert out.dtype == np.uint16
        assert out[0] == 10
        assert out[1] == 20
        assert out[2] == 0xFFFF
        assert out[3] == 0xFFFF

    def test_insert_ndarray_round_trip(self) -> None:
        m = HashMapU32U16()
        keys = np.array([5, 6, 7], dtype=np.uint32)
        values = np.array([50, 60, 70], dtype=np.uint16)
        m.insert_ndarray(keys, values)
        assert len(m) == 3
        out = m.lookup_ndarray(keys)
        assert out.tolist() == [50, 60, 70]

    def test_set_then_lookup_ndarray_cross_api(self) -> None:
        # Scalar set must be visible to batch lookup, and vice versa.
        m = HashMapU32U16()
        m.set(11, 111)
        m.insert(12, 112)
        out = m.lookup_ndarray(np.array([11, 12, 13], dtype=np.uint32))
        assert out[0] == 111
        assert out[1] == 112
        assert out[2] == 0xFFFF
        # Now batch-insert and verify scalar get sees it.
        m.insert_ndarray(
            np.array([13, 14], dtype=np.uint32),
            np.array([113, 114], dtype=np.uint16),
        )
        assert m.get(13) == 113
        assert m.get(14) == 114

    def test_insert_ndarray_duplicate_keys_last_wins(self) -> None:
        m = HashMapU32U16()
        keys = np.array([1, 2, 1, 2], dtype=np.uint32)
        values = np.array([10, 20, 100, 200], dtype=np.uint16)
        m.insert_ndarray(keys, values)
        assert len(m) == 2
        assert m.lookup(1) == 100
        assert m.lookup(2) == 200

    def test_insert_ndarray_mismatched_lengths(self) -> None:
        m = HashMapU32U16()
        with pytest.raises(ValueError):
            m.insert_ndarray(
                np.array([1, 2], dtype=np.uint32),
                np.array([10], dtype=np.uint16),
            )


class TestHashMapU8U8:
    """Smallest dtypes — verifies the unsigned-MAX miss sentinel byte width."""

    def test_lookup_ndarray_sentinel_is_0xff(self) -> None:
        m = HashMapU8U8()
        m.insert(0, 1)
        m.insert(7, 2)
        out = m.lookup_ndarray(np.array([0, 7, 99], dtype=np.uint8))
        assert out.dtype == np.uint8
        assert out[0] == 1
        assert out[1] == 2
        assert out[2] == 0xFF

    def test_value_zero_is_distinguishable_from_miss_via_contains(self) -> None:
        # Sentinel = 0xFF, so a real-stored value 0 must not be confused with a miss
        # — callers should use __contains__ / scalar lookup for disambiguation.
        m = HashMapU8U8()
        m.insert(3, 0)
        assert m.lookup(3) == 0
        assert 3 in m
        assert m.lookup(4) is None
        out = m.lookup_ndarray(np.array([3, 4], dtype=np.uint8))
        assert out[0] == 0
        assert out[1] == 0xFF


class TestHashMapI32F64:
    """Mixed signed-int key + float value — exercises the NaN sentinel rule."""

    def test_float_miss_sentinel_is_nan(self) -> None:
        m = HashMapI32F64()
        m.insert(-1, 1.5)
        m.insert(0, 0.0)
        m.insert(1, -3.25)
        keys = np.array([-1, 0, 1, 999], dtype=np.int32)
        out = m.lookup_ndarray(keys)
        assert out.dtype == np.float64
        assert out[0] == 1.5
        assert out[1] == 0.0
        assert out[2] == -3.25
        assert np.isnan(out[3])

    def test_negative_key_round_trip(self) -> None:
        m = HashMapI32F64()
        m.set(-(2**30), 2.71828)
        assert m.get(-(2**30)) == pytest.approx(2.71828)
        assert -(2**30) in m


class TestHashMapBoolBool:
    """Degenerate edge — both key and value are 1-bit."""

    def test_two_keys_two_values(self) -> None:
        m = HashMapBoolBool()
        m.insert(False, True)
        m.insert(True, False)
        assert m.get(False) is True
        assert m.get(True) is False
        assert len(m) == 2

    def test_lookup_ndarray_miss_is_false(self) -> None:
        m = HashMapBoolBool()
        m.insert(True, True)
        # Only True maps; False is a miss → sentinel False.
        out = m.lookup_ndarray(np.array([True, False], dtype=np.bool_))
        assert out.dtype == np.bool_
        assert out[0]
        assert not out[1]
