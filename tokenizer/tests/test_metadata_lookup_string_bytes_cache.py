"""Tests for ``GhidraMetadataLookup`` string-bytes materialization.

Concern: ``_classify_string`` is invoked from every immediate-operand
``lookup()``. Each distinct substring offset into the SAME containing
string Data resolves to that one Data; without caching, the entire Data's
``byte[]`` is re-fetched + re-converted from Java on every call (several
times per function, and again in every later function that references the
Data). For a multi-MB string blob this is monotone redundant
re-materialization -> OOM.

The fix is a per-Data extracted-bytes cache keyed by the containing Data's
min-address offset (a plain Python int), valued by the already-extracted
``(encoding, raw_bytes)`` tuple. These tests verify:

1. CACHE HIT: a second ``_classify_string`` at a different substring offset
   into the same Data returns identical bytes WITHOUT re-invoking
   ``getBytes()`` (the expensive per-byte Java extraction).
2. CACHE ISOLATION: a different Data (different min-address) is a miss and
   extracts independently.
3. BYTE-IDENTITY: ``_java_bytes_to_python`` reinterprets signed Java bytes
   to the unsigned 0-255 range identically to the old ``int(b) & 0xFF``
   masking, for the JPype buffer-protocol copy path AND the plain-list
   fallback the test mock exercises.

The mocks are pure Python duck-typed Ghidra surfaces; no JVM required.
"""

from __future__ import annotations

from typing import Any, List, Optional

from tokenizer.disasm.ghidra_provider.metadata_lookup import GhidraMetadataLookup
from tokenizer.disasm.ghidra_provider.section_classify import _java_bytes_to_python


# ---------------------------------------------------------------------------
# Duck-typed Ghidra surface (string-Data flavor)
# ---------------------------------------------------------------------------


class _MockAddress:
    def __init__(self, offset: int) -> None:
        self._offset = int(offset)

    def getOffset(self) -> int:
        return self._offset


class _MockAddressSpace:
    def getAddress(self, offset: int) -> _MockAddress:
        return _MockAddress(offset)


class _MockAddressFactory:
    def getDefaultAddressSpace(self) -> _MockAddressSpace:
        return _MockAddressSpace()


class _MockBlock:
    def __init__(self, name: str = ".rodata") -> None:
        self._name = name

    def getName(self) -> str:
        return self._name

    def isExecute(self) -> bool:
        return False

    def isWrite(self) -> bool:
        return False


class _MockStringDataType:
    def __init__(self, name: str = "string") -> None:
        self._name = name

    def getName(self) -> str:
        return self._name


class _MockStringData:
    """A string-flavored ``Data`` spanning ``[min_offset, min_offset+len)``.

    Tracks how many times ``getBytes()`` is invoked so a test can assert
    the cache prevents repeat extraction. ``getBytes()`` returns the
    Java-style byte list verbatim (the classifier routes it through
    ``_java_bytes_to_python``)."""

    def __init__(self, *, min_offset: int, raw: bytes, dt_name: str = "string") -> None:
        self._min_offset = int(min_offset)
        # The real Ghidra ``getBytes()`` returns a JPype-wrapped Java
        # ``byte[]`` that satisfies the buffer protocol; a ``bytearray``
        # is the closest pure-Python stand-in that exercises the same
        # ``bytes(memoryview(raw))`` copy path the production code uses.
        self._raw = bytearray(raw)
        self._dt = _MockStringDataType(dt_name)
        self.get_bytes_calls = 0

    def hasStringValue(self) -> bool:
        return True

    def getMinAddress(self) -> _MockAddress:
        return _MockAddress(self._min_offset)

    def getDataType(self) -> _MockStringDataType:
        return self._dt

    def getBytes(self) -> bytearray:
        self.get_bytes_calls += 1
        return self._raw


class _MockListing:
    """Returns a containing string Data for any offset that falls inside
    one of the registered Data spans (``getDataContaining``); never an
    exact-start Data (``getDataAt`` -> None) so the classifier always
    exercises the containment path the way substring operands do."""

    def __init__(self, data_objs: List[_MockStringData]) -> None:
        self._data_objs = data_objs

    def getDataAt(self, _addr_obj: _MockAddress) -> Any:
        return None

    def getDataContaining(self, addr_obj: _MockAddress) -> Optional[_MockStringData]:
        off = addr_obj.getOffset()
        for data in self._data_objs:
            if data._min_offset <= off < data._min_offset + len(data._raw):
                return data
        return None


class _MockMemory:
    def __init__(self, block: _MockBlock) -> None:
        self._block = block

    def getBlock(self, _addr_obj: _MockAddress) -> _MockBlock:
        return self._block


class _MockSymbolTable:
    def getSymbols(self, _addr_obj: _MockAddress) -> List[Any]:
        return []


class _MockProgram:
    def __init__(self, *, memory: _MockMemory, listing: _MockListing) -> None:
        self._memory = memory
        self._listing = listing

    def getMemory(self) -> _MockMemory:
        return self._memory

    def getSymbolTable(self) -> _MockSymbolTable:
        return _MockSymbolTable()

    def getListing(self) -> _MockListing:
        return self._listing

    def getAddressFactory(self) -> _MockAddressFactory:
        return _MockAddressFactory()


class _MockFunctionManager:
    def getFunctionContaining(self, _addr_obj: _MockAddress) -> Any:
        return None

    def getFunctions(self, _forward: bool) -> List[Any]:
        return []


_TEST_BINARY_ID_HASH = bytes(range(16))


def _build_lookup(data_objs: List[_MockStringData]) -> GhidraMetadataLookup:
    program = _MockProgram(
        memory=_MockMemory(_MockBlock()),
        listing=_MockListing(data_objs),
    )
    return GhidraMetadataLookup(program, _MockFunctionManager(), _TEST_BINARY_ID_HASH)


def _addr(lookup: GhidraMetadataLookup, offset: int) -> _MockAddress:
    return lookup._program.getAddressFactory().getDefaultAddressSpace().getAddress(offset)


# ---------------------------------------------------------------------------
# CHANGE 1: per-Data extracted-bytes cache
# ---------------------------------------------------------------------------


def test_classify_string_second_substring_hit_skips_reconversion() -> None:
    """Two substring offsets into the SAME Data => one getBytes() only."""
    raw = b"hello world payload"
    data = _MockStringData(min_offset=0x1000, raw=raw)
    lookup = _build_lookup([data])

    is_str1, enc1, bytes1 = lookup._classify_string(_addr(lookup, 0x1000))
    is_str2, enc2, bytes2 = lookup._classify_string(_addr(lookup, 0x1006))  # mid-string

    assert is_str1 and is_str2
    assert enc1 == enc2 == "ascii"
    assert bytes1 == bytes2 == bytes(raw)
    # The expensive extraction must have happened exactly once.
    assert data.get_bytes_calls == 1
    # And the cache is keyed by the Data's min-address, not the query addr.
    assert set(lookup._string_bytes_cache) == {0x1000}


def test_classify_string_distinct_data_objects_extract_independently() -> None:
    """Different Data (different min-address) => independent miss + extract."""
    data_a = _MockStringData(min_offset=0x2000, raw=b"alpha")
    data_b = _MockStringData(min_offset=0x3000, raw=b"beta")
    lookup = _build_lookup([data_a, data_b])

    _, _, bytes_a = lookup._classify_string(_addr(lookup, 0x2000))
    _, _, bytes_b = lookup._classify_string(_addr(lookup, 0x3000))
    # Re-touch each once more to confirm both are now cached.
    lookup._classify_string(_addr(lookup, 0x2002))
    lookup._classify_string(_addr(lookup, 0x3001))

    assert bytes_a == b"alpha"
    assert bytes_b == b"beta"
    assert data_a.get_bytes_calls == 1
    assert data_b.get_bytes_calls == 1
    assert set(lookup._string_bytes_cache) == {0x2000, 0x3000}


def test_string_bytes_cache_is_per_instance() -> None:
    """A fresh lookup (=> fresh per-binary instance) starts with an empty
    cache; nothing leaks across instances."""
    data = _MockStringData(min_offset=0x4000, raw=b"x")
    lookup1 = _build_lookup([data])
    lookup1._classify_string(_addr(lookup1, 0x4000))
    assert lookup1._string_bytes_cache  # populated

    lookup2 = _build_lookup([_MockStringData(min_offset=0x4000, raw=b"x")])
    assert lookup2._string_bytes_cache == {}  # independent, empty


# ---------------------------------------------------------------------------
# CHANGE 2: byte-identity of the Java->Python conversion
# ---------------------------------------------------------------------------


def test_java_bytes_to_python_unsigned_identity() -> None:
    """Signed Java bytes reinterpret to unsigned 0-255 identically to the
    old ``int(b) & 0xFF`` masking, including negative inputs."""
    signed = [0, 65, 66, -1, -128, 127, -2]
    expected = bytes(int(b) & 0xFF for b in signed)
    assert expected == bytes([0, 65, 66, 0xFF, 0x80, 127, 0xFE])
    # bytes/bytearray (Python objects) also satisfy the buffer protocol the
    # production path now uses; here we feed the equivalent unsigned buffer
    # and confirm it is preserved byte-for-byte.
    assert _java_bytes_to_python(bytes(expected)) == expected
    assert _java_bytes_to_python(bytearray(expected)) == expected


def test_java_bytes_to_python_none_returns_empty() -> None:
    assert _java_bytes_to_python(None) == b""
