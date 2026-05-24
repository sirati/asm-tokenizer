"""Typed wrapper over ``dedup_hashmap._native.HashMap<K><V>``.

Single concern: pick the narrowest backing integer dtype that fits the
caller-supplied :class:`enum.IntEnum` value range, instantiate the
matching generated PyO3 class from :mod:`dedup_hashmap._native`, and
translate scalar API calls between Python ``IntEnum`` members and raw
integers at the boundary.

The wrapper is generic over key and value types::

    table: IntEnumHashMap[MyKeyEnum, MyValueEnum] = IntEnumHashMap(
        MyKeyEnum, MyValueEnum
    )
    table.insert(MyKeyEnum.X, MyValueEnum.A)
    table.get(MyKeyEnum.X)            # -> MyValueEnum.A
    table.get(MyKeyEnum.Y)            # -> None
    MyKeyEnum.X in table              # -> True

The scalar API (``get`` / ``insert`` / ``__contains__``) takes and
returns the enum types. Misses surface as ``None`` (NOT the per-dtype
sentinel from the underlying native class): the wrapper uses
``HashMap.__contains__`` to disambiguate sentinel-valued hits from
genuine misses, so legitimate entries whose value happens to equal the
sentinel still round-trip cleanly.

Type aliases :class:`PlainInt` and :class:`PlainBool` are also accepted
in the K/V slot for tables whose keys or values are raw ints / bools
(no enum membership). With those, no boundary conversion happens.

Constructing the wrapper validates that every declared enum member
fits the chosen backing dtype; this catches future schema drift (e.g.
a new enum member that pushes the max above the picked dtype's range)
at construction rather than at insert time.

The vectorized batch path (``lookup_ndarray`` / ``insert_ndarray``) is
intentionally NOT proxied: vector consumers want raw numpy arrays of
the backing dtype and would lose all benefit from per-element enum
boxing. Callers that need vector access read the underlying native
instance via :attr:`raw` and operate on it directly with the dtype
exposed by :attr:`key_dtype` / :attr:`value_dtype`.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Final, Generic, Optional, Type, TypeVar, Union

import numpy as np

from . import _native


__all__ = [
    "IntDtype",
    "IntEnumHashMap",
    "PlainBool",
    "PlainInt",
]


# ---------------------------------------------------------------------------
# Backing-dtype catalog. The named instances below correspond to the
# (signed?, bit_width) cells the ``dedup_hashmap`` macro emits classes
# for. Boolean is treated as a distinct zero-bit-width unsigned cell
# because the macro emits dedicated ``HashMap*Bool`` / ``HashMapBool*``
# classes for it.
# ---------------------------------------------------------------------------


class IntDtype:
    """A scalar integer dtype supported by ``dedup_hashmap._native``.

    Carries:

    - ``name``: PascalCase tag used in the ``HashMap<K><V>`` class name
      (``"U8"``, ``"I16"``, ``"Bool"``, ...).
    - ``signed``: whether the dtype is signed (``False`` for ``Bool``).
    - ``bits``: bit-width (``1`` for ``Bool``, ``8``/``16``/``32``/``64``
      otherwise).
    - ``min`` / ``max``: inclusive range of representable values.
    - ``numpy_dtype``: matching numpy dtype, for ndarray paths.

    Instances are interned via the module-level catalog so
    ``IntDtype.U8 is IntDtype.U8`` holds; equality is identity.
    """

    __slots__ = ("name", "signed", "bits", "min", "max", "numpy_dtype")

    def __init__(
        self,
        name: str,
        *,
        signed: bool,
        bits: int,
        value_min: int,
        value_max: int,
        numpy_dtype: Any,
    ) -> None:
        self.name = name
        self.signed = signed
        self.bits = bits
        self.min = value_min
        self.max = value_max
        self.numpy_dtype = numpy_dtype

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"IntDtype({self.name})"


IntDtype.Bool = IntDtype(
    "Bool", signed=False, bits=1, value_min=0, value_max=1, numpy_dtype=np.bool_
)
IntDtype.U8 = IntDtype(
    "U8", signed=False, bits=8, value_min=0, value_max=0xFF, numpy_dtype=np.uint8
)
IntDtype.U16 = IntDtype(
    "U16",
    signed=False,
    bits=16,
    value_min=0,
    value_max=0xFFFF,
    numpy_dtype=np.uint16,
)
IntDtype.U32 = IntDtype(
    "U32",
    signed=False,
    bits=32,
    value_min=0,
    value_max=0xFFFFFFFF,
    numpy_dtype=np.uint32,
)
IntDtype.U64 = IntDtype(
    "U64",
    signed=False,
    bits=64,
    value_min=0,
    value_max=0xFFFFFFFFFFFFFFFF,
    numpy_dtype=np.uint64,
)
IntDtype.I8 = IntDtype(
    "I8",
    signed=True,
    bits=8,
    value_min=-(1 << 7),
    value_max=(1 << 7) - 1,
    numpy_dtype=np.int8,
)
IntDtype.I16 = IntDtype(
    "I16",
    signed=True,
    bits=16,
    value_min=-(1 << 15),
    value_max=(1 << 15) - 1,
    numpy_dtype=np.int16,
)
IntDtype.I32 = IntDtype(
    "I32",
    signed=True,
    bits=32,
    value_min=-(1 << 31),
    value_max=(1 << 31) - 1,
    numpy_dtype=np.int32,
)
IntDtype.I64 = IntDtype(
    "I64",
    signed=True,
    bits=64,
    value_min=-(1 << 63),
    value_max=(1 << 63) - 1,
    numpy_dtype=np.int64,
)

# Ordered narrowest-to-widest for the auto-select walks. Bool is excluded
# from the integer-range selection because IntEnum members are never bools;
# Bool is reachable only through the explicit :class:`PlainBool` hint.
_UNSIGNED_WIDTHS: Final[tuple[IntDtype, ...]] = (
    IntDtype.U8,
    IntDtype.U16,
    IntDtype.U32,
    IntDtype.U64,
)
_SIGNED_WIDTHS: Final[tuple[IntDtype, ...]] = (
    IntDtype.I8,
    IntDtype.I16,
    IntDtype.I32,
    IntDtype.I64,
)


def _select_dtype_for_range(value_min: int, value_max: int) -> IntDtype:
    """Pick the narrowest dtype whose range covers ``[value_min, value_max]``.

    Negative ``value_min`` forces a signed dtype; non-negative ranges
    prefer the unsigned ladder so a tight ``[0, 30]`` IntEnum (e.g.
    :class:`tokenizer.tokens.TokenType` excluding ``UNRESOLVED``) lands
    in U8 rather than I8.

    Raises :class:`ValueError` when even I64 / U64 cannot cover the
    requested range.
    """
    if value_min < 0:
        for dtype in _SIGNED_WIDTHS:
            if value_min >= dtype.min and value_max <= dtype.max:
                return dtype
        raise ValueError(
            f"range [{value_min}, {value_max}] does not fit in any signed dtype"
        )
    for dtype in _UNSIGNED_WIDTHS:
        if value_max <= dtype.max:
            return dtype
    raise ValueError(
        f"range [{value_min}, {value_max}] does not fit in any unsigned dtype"
    )


# ---------------------------------------------------------------------------
# Type-spec sentinels for non-enum slots. ``PlainInt`` lets the wrapper
# stand in for ``dict[int, ...]`` / ``dict[..., int]`` callsites that
# already carry their own dtype hint; ``PlainBool`` for the int->bool
# dispatch tables.
# ---------------------------------------------------------------------------


class PlainInt:
    """Marker for a raw-int slot with an explicit dtype hint.

    Use as ``IntEnumHashMap(PlainInt[IntDtype.U8], MyEnum)`` for a
    callsite whose keys (or values) are bare ints. The subscript form
    ``PlainInt[IntDtype.U16]`` returns an opaque spec the constructor
    unpacks; bare ``PlainInt`` is invalid (no width to pick).
    """

    __slots__ = ("dtype",)
    dtype: IntDtype

    def __init__(self, dtype: IntDtype) -> None:
        self.dtype = dtype

    def __class_getitem__(cls, dtype: IntDtype) -> "PlainInt":
        return cls(dtype)


class PlainBool:
    """Marker for a raw-bool slot (always :class:`IntDtype.Bool`)."""

    __slots__ = ()


_TypeSpec = Union[Type[IntEnum], PlainInt, Type[PlainBool]]


def _resolve_spec(spec: _TypeSpec) -> tuple[IntDtype, Optional[Type[IntEnum]]]:
    """Map a key/value type spec to ``(backing_dtype, enum_class)``.

    Returns ``enum_class is None`` for raw-int / raw-bool cases so the
    wrapper can skip boundary conversion. Raises :class:`TypeError` on
    an unrecognized spec and :class:`ValueError` on an empty IntEnum.
    """
    if isinstance(spec, PlainInt):
        return spec.dtype, None
    if spec is PlainBool:
        return IntDtype.Bool, None
    if isinstance(spec, type) and issubclass(spec, IntEnum):
        members = tuple(spec)
        if not members:
            raise ValueError(
                f"IntEnum {spec.__name__!r} has no members; cannot pick a dtype"
            )
        min_v = min(int(m) for m in members)
        max_v = max(int(m) for m in members)
        return _select_dtype_for_range(min_v, max_v), spec
    raise TypeError(
        "type spec must be an IntEnum subclass, PlainInt(dtype), or PlainBool; "
        f"got {spec!r}"
    )


# ---------------------------------------------------------------------------
# Generic wrapper.
# ---------------------------------------------------------------------------

K = TypeVar("K")
V = TypeVar("V")


class IntEnumHashMap(Generic[K, V]):
    """Typed wrapper around a generated ``HashMap<K><V>`` native class.

    See module docstring for the full design rationale.
    """

    __slots__ = (
        "_raw",
        "_key_dtype",
        "_value_dtype",
        "_key_enum",
        "_value_enum",
    )

    def __init__(
        self,
        key_type: _TypeSpec,
        value_type: _TypeSpec,
        *,
        capacity: int = 0,
    ) -> None:
        key_dtype, key_enum = _resolve_spec(key_type)
        value_dtype, value_enum = _resolve_spec(value_type)
        cls_name = f"HashMap{key_dtype.name}{value_dtype.name}"
        try:
            cls = getattr(_native, cls_name)
        except AttributeError as e:  # pragma: no cover - all 99 emitted
            raise RuntimeError(
                f"dedup_hashmap._native does not expose {cls_name!r}; "
                "check Cartesian-product table in dedup_hashmap/src/lib.rs"
            ) from e
        self._raw = cls(capacity)
        self._key_dtype = key_dtype
        self._value_dtype = value_dtype
        self._key_enum = key_enum
        self._value_enum = value_enum

    @property
    def raw(self) -> Any:
        """The underlying PyO3 instance (for vectorized ndarray paths)."""
        return self._raw

    @property
    def key_dtype(self) -> IntDtype:
        return self._key_dtype

    @property
    def value_dtype(self) -> IntDtype:
        return self._value_dtype

    def _coerce_key(self, key: K) -> Any:
        if self._key_enum is not None and not isinstance(key, self._key_enum):
            raise TypeError(
                f"expected {self._key_enum.__name__} key, got {type(key).__name__}"
            )
        # The Bool-keyed native classes (``HashMapBool*``) require a real
        # Python ``bool`` -- passing an ``int`` triggers PyO3's strict
        # type check. Everything else converts cleanly through ``int``.
        if self._key_dtype is IntDtype.Bool:
            return bool(key)
        return int(key)  # type: ignore[arg-type]

    def _coerce_value(self, value: V) -> Any:
        if self._value_enum is not None and not isinstance(
            value, self._value_enum
        ):
            raise TypeError(
                f"expected {self._value_enum.__name__} value, "
                f"got {type(value).__name__}"
            )
        if self._value_dtype is IntDtype.Bool:
            return bool(value)
        return int(value)  # type: ignore[arg-type]

    def _wrap_value(self, raw: Any) -> V:
        if self._value_enum is None:
            return raw  # type: ignore[return-value]
        return self._value_enum(raw)  # type: ignore[return-value]

    def get(self, key: K) -> Optional[V]:
        """Lookup ``key``; return ``None`` for a miss.

        Disambiguated by ``__contains__`` so the per-dtype miss sentinel
        in ``dedup_hashmap`` never aliases a legitimate hit.
        """
        raw_key = self._coerce_key(key)
        if not self._raw.__contains__(raw_key):
            return None
        return self._wrap_value(self._raw.get(raw_key))

    def insert(self, key: K, value: V) -> None:
        """Insert / overwrite ``(key, value)``."""
        self._raw.insert(self._coerce_key(key), self._coerce_value(value))

    def __contains__(self, key: K) -> bool:
        return self._raw.__contains__(self._coerce_key(key))

    def __len__(self) -> int:
        return len(self._raw)

    def clean(self) -> None:
        """Clear all entries; retain the bucket allocation."""
        self._raw.clean()
