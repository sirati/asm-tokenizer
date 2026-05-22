"""Direct parity tests for the two ``dedup_hashmap`` PyO3 classes.

``HashMapU64U32`` backs the primary side of the content-addressed dedup
in :mod:`tokenizer.memmap_builder._dedup`. ``HashMapU32U32`` backs
``SectionWriter._known_sections`` in
:mod:`tokenizer.aligned_data.matched_sections_bin`. Both are exercised
indirectly via their consumers' integration tests; this module is the
direct contract check for the five-method API
(``new``, ``get``, ``set``, ``__contains__``, ``__len__``) so a future
crate-internal refactor doesn't silently break the FFI surface.
"""

from __future__ import annotations

import pytest

from dedup_hashmap import HashMapU32U32, HashMapU64U32


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
