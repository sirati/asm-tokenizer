"""Contract tests for :class:`dedup_hashmap.IntEnumHashMap`.

The wrapper auto-selects the narrowest backing dtype, translates the
scalar API between IntEnum members and raw ints, and disambiguates
sentinel-valued hits from genuine misses via ``__contains__``.

These tests pin the public surface so a future macro / wrapper refactor
cannot silently change the type of the returned value or the
miss-detection semantics.
"""

from __future__ import annotations

from enum import IntEnum

import pytest

from dedup_hashmap import IntDtype, IntEnumHashMap, PlainBool, PlainInt


class _SmallKey(IntEnum):
    A = 1
    B = 2
    C = 255


class _SmallValue(IntEnum):
    X = 0
    Y = 200
    Z = 250


class _SignedValue(IntEnum):
    NEG = -10
    ZERO = 0
    POS = 1000


class _WideKey(IntEnum):
    BASE = 0
    HIGH = 70_000  # forces U32


class TestAutoDtypeSelection:
    def test_picks_smallest_unsigned_when_range_fits_u8(self) -> None:
        m = IntEnumHashMap(_SmallKey, _SmallValue)
        assert m.key_dtype is IntDtype.U8
        assert m.value_dtype is IntDtype.U8

    def test_picks_signed_when_min_negative(self) -> None:
        m = IntEnumHashMap(_SmallKey, _SignedValue)
        assert m.value_dtype is IntDtype.I16

    def test_picks_wider_unsigned_when_max_exceeds_u8(self) -> None:
        m = IntEnumHashMap(_WideKey, _SmallValue)
        assert m.key_dtype is IntDtype.U32

    def test_plain_int_spec_uses_supplied_dtype(self) -> None:
        m = IntEnumHashMap(PlainInt[IntDtype.U16], _SmallValue)
        assert m.key_dtype is IntDtype.U16

    def test_plain_bool_value_uses_bool_dtype(self) -> None:
        m = IntEnumHashMap(_SmallKey, PlainBool)
        assert m.value_dtype is IntDtype.Bool


class TestScalarApi:
    def test_insert_get_round_trips_enum_member(self) -> None:
        m = IntEnumHashMap(_SmallKey, _SmallValue)
        m.insert(_SmallKey.A, _SmallValue.Y)
        result = m.get(_SmallKey.A)
        assert result is _SmallValue.Y
        # ``get`` returns the actual enum member, not the raw int.
        assert isinstance(result, _SmallValue)

    def test_miss_returns_none(self) -> None:
        m = IntEnumHashMap(_SmallKey, _SmallValue)
        m.insert(_SmallKey.A, _SmallValue.Y)
        assert m.get(_SmallKey.B) is None
        assert m.get(_SmallKey.C) is None

    def test_contains_matches_insert(self) -> None:
        m = IntEnumHashMap(_SmallKey, _SmallValue)
        m.insert(_SmallKey.A, _SmallValue.X)
        assert _SmallKey.A in m
        assert _SmallKey.B not in m

    def test_clean_drops_entries_but_preserves_dtype(self) -> None:
        m = IntEnumHashMap(_SmallKey, _SmallValue)
        m.insert(_SmallKey.A, _SmallValue.Y)
        m.insert(_SmallKey.B, _SmallValue.Z)
        assert len(m) == 2
        m.clean()
        assert len(m) == 0
        assert m.get(_SmallKey.A) is None
        # Dtype binding stays intact across clean.
        assert m.value_dtype is IntDtype.U8

    def test_signed_value_round_trips_including_negative(self) -> None:
        m = IntEnumHashMap(_SmallKey, _SignedValue)
        m.insert(_SmallKey.A, _SignedValue.NEG)
        m.insert(_SmallKey.B, _SignedValue.POS)
        assert m.get(_SmallKey.A) is _SignedValue.NEG
        assert m.get(_SmallKey.B) is _SignedValue.POS


class TestBoolValueSentinelDisambiguation:
    """The Bool-value class uses ``false`` as the miss sentinel.

    Without ``__contains__`` filtering, an inserted ``False`` would be
    indistinguishable from a miss. The wrapper guards against that.
    """

    def test_false_value_round_trips_separately_from_miss(self) -> None:
        m = IntEnumHashMap(_SmallKey, PlainBool)
        m.insert(_SmallKey.A, False)
        assert m.get(_SmallKey.A) is False
        assert m.get(_SmallKey.B) is None
        assert _SmallKey.A in m
        assert _SmallKey.B not in m


class TestTypeErrorOnMismatch:
    def test_insert_with_wrong_key_enum_raises(self) -> None:
        m = IntEnumHashMap(_SmallKey, _SmallValue)
        with pytest.raises(TypeError):
            m.insert(99, _SmallValue.X)  # type: ignore[arg-type]

    def test_insert_with_wrong_value_enum_raises(self) -> None:
        m = IntEnumHashMap(_SmallKey, _SmallValue)
        with pytest.raises(TypeError):
            m.insert(_SmallKey.A, 5)  # type: ignore[arg-type]


class TestConstructionValidation:
    def test_overlarge_enum_raises_value_error(self) -> None:
        class _TooBig(IntEnum):
            BIG = 1 << 70

        with pytest.raises(ValueError):
            IntEnumHashMap(_SmallKey, _TooBig)

    def test_empty_enum_raises(self) -> None:
        class _Empty(IntEnum):
            pass

        with pytest.raises(ValueError):
            IntEnumHashMap(_Empty, _SmallValue)


class TestRawAccessor:
    def test_raw_exposes_underlying_pyo3_class(self) -> None:
        m = IntEnumHashMap(_SmallKey, _SmallValue)
        # The concrete class name follows the macro's
        # ``HashMap<KeyDtype><ValueDtype>`` pattern.
        assert type(m.raw).__name__ == "HashMapU8U8"
