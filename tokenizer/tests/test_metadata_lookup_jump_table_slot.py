"""Tests for ``GhidraMetadataLookup._is_jump_table_slot``.

Concern: the structural Pointer-in-rodata-block predicate is necessary
but NOT sufficient. Hardened-relro builds (``-z relro -z now``) place
the PLT GOT analogue in ``.data.rel.ro``; C++ RTTI puts typeinfo and
vtable component pointers in the same rodata-flavored sections. Every
one of those is a Pointer in a rodata-flavored block — the structural
match alone would mis-classify them as ``JUMP_TABLE_SLOT``, triggering
the per-instruction ``Category.JUMP_TABLE`` identity registration and
the downstream ``_emit_jump_table_footer`` slot-classification fallback
on what is in fact a relro GOT slot or a vtable component pointer.

The fix tightens the predicate to ALSO require an inbound
``COMPUTED_JUMP`` back-reference into the Data's address span. That
back-ref is recorded by Ghidra's analyser when it recovers a
computed-jump dispatch site that reads the table; its presence is the
authoritative evidence that the Data IS a switch table (vs a
coincidental pointer array).

These tests build a duck-typed mock Program surface that satisfies the
predicate's Java-side calls (``program.getReferenceManager``,
``listing.getDataAt`` / ``getDataContaining``, ``memory.getBlock``,
``ReferenceManager.getReferenceDestinationIterator`` /
``getReferencesTo``) and exercise both the structural and semantic
gates without a Ghidra JVM. To exercise the primary ``isinstance``-based
structural check (rather than the string-name fallback), a stub
``ghidra.program.model.data`` module is injected into ``sys.modules``
exposing the ``Array`` / ``Pointer`` interface names used by the
predicate's lazy import.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Iterable, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Stub ``ghidra.program.model.data`` so the predicate's primary
# ``isinstance(dt, Pointer)`` / ``isinstance(dt, Array)`` path executes
# in tests; without these stubs the ``import ghidra...`` call raises and
# the predicate falls through to the string-name fallback (which is a
# narrower path).
# ---------------------------------------------------------------------------


class _StubPointer:
    """Marker base class for the predicate's ``isinstance(dt, Pointer)`` check."""


class _StubArray:
    """Marker base class for the predicate's ``isinstance(dt, Array)`` check.

    Mock Array DataTypes inherit from this and expose ``getDataType()``
    returning the element DataType (mirrors Ghidra's
    ``Array.getDataType()`` accessor).
    """


def _install_stub_ghidra_data_module() -> None:
    """Inject a minimal ``ghidra.program.model.data`` module into
    ``sys.modules`` so the predicate's lazy ``from ghidra... import
    Array, Pointer`` resolves to our marker classes."""
    ghidra_pkg = sys.modules.setdefault("ghidra", types.ModuleType("ghidra"))
    program_pkg = sys.modules.setdefault(
        "ghidra.program", types.ModuleType("ghidra.program")
    )
    model_pkg = sys.modules.setdefault(
        "ghidra.program.model", types.ModuleType("ghidra.program.model")
    )
    data_mod = types.ModuleType("ghidra.program.model.data")
    data_mod.Pointer = _StubPointer
    data_mod.Array = _StubArray
    sys.modules["ghidra.program.model.data"] = data_mod
    # Also expose attributes on the parent packages so dotted attribute
    # access (some import paths) would work; this is belt-and-suspenders.
    ghidra_pkg.program = program_pkg
    program_pkg.model = model_pkg
    model_pkg.data = data_mod


def _install_stub_ghidra_symbol_module() -> None:
    """Inject a minimal ``ghidra.program.model.symbol`` module exposing
    a ``RefType`` namespace. The helper's defensive ``COMPUTED_JUMP``
    identity-compare fallback uses this; the primary ``isJump() and
    isComputed()`` path on our mock RefType is exercised first, so the
    fallback is only reached if the methods raise.
    """
    symbol_mod = types.ModuleType("ghidra.program.model.symbol")

    class _RefType:
        # Sentinel object distinct from any mock RefType instance; the
        # fallback ``rt == RefType.COMPUTED_JUMP`` compare therefore
        # never spuriously matches in tests that exercise the primary
        # method path.
        COMPUTED_JUMP = object()

    symbol_mod.RefType = _RefType
    sys.modules["ghidra.program.model.symbol"] = symbol_mod


@pytest.fixture(autouse=True)
def _stub_ghidra_modules() -> None:
    """Auto-applied per-test fixture: install ghidra stubs before each
    test runs. Idempotent (``setdefault`` for the parent packages)."""
    _install_stub_ghidra_data_module()
    _install_stub_ghidra_symbol_module()


# Import AFTER the fixture installs the stubs so the predicate's lazy
# import resolves cleanly the first time it runs.
from tokenizer.disasm.ghidra_provider.metadata_lookup import (  # noqa: E402
    GhidraMetadataLookup,
)


# ---------------------------------------------------------------------------
# Duck-typed Ghidra surface — only the calls ``_is_jump_table_slot`` and
# its helper ``has_inbound_computed_jump`` actually issue are implemented.
# ---------------------------------------------------------------------------


class _MockAddress:
    """Stand-in for Ghidra's ``Address`` — exposes ``getOffset`` and
    ``compareTo`` (the helper compares destination addresses against
    the Data's ``max_addr`` to terminate the scan)."""

    def __init__(self, offset: int) -> None:
        self._offset = int(offset)

    def getOffset(self) -> int:
        return self._offset

    def compareTo(self, other: "_MockAddress") -> int:
        if self._offset < other._offset:
            return -1
        if self._offset > other._offset:
            return 1
        return 0

    def __repr__(self) -> str:  # pragma: no cover — debug aid only
        return f"_MockAddress(0x{self._offset:x})"


class _MockAddressSpace:
    def getAddress(self, offset: int) -> _MockAddress:
        return _MockAddress(offset)


class _MockAddressFactory:
    def getDefaultAddressSpace(self) -> _MockAddressSpace:
        return _MockAddressSpace()


class _MockBlock:
    """Stand-in for ``MemoryBlock`` — the predicate only reads ``getName``."""

    def __init__(self, name: str) -> None:
        self._name = name

    def getName(self) -> str:
        return self._name


class _MockPointerDataType(_StubPointer):
    """Pointer DataType — passes the predicate's
    ``isinstance(dt, Pointer)`` check."""

    def getName(self) -> str:
        return "pointer"


class _MockArrayOfPointerDataType(_StubArray):
    """Array DataType whose ``getDataType()`` returns a Pointer — passes
    the predicate's ``isinstance(dt, Array)`` + inner-pointer check."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def getName(self) -> str:
        return "pointer[]"

    def getDataType(self) -> Any:
        return self._inner


class _MockNonPointerDataType:
    """DataType that is neither Pointer nor Array — fails the structural
    check, must produce ``False`` regardless of back-refs."""

    def getName(self) -> str:
        return "int"


class _MockData:
    """Stand-in for Ghidra's ``Data``.

    Exposes ``getDataType``, ``getMinAddress``, ``getMaxAddress``. The
    address span covers a contiguous range ``[min, max]`` inclusive;
    array Data uses a wider span than a single Pointer.
    """

    def __init__(self, datatype: Any, min_offset: int, max_offset: int) -> None:
        self._datatype = datatype
        self._min = _MockAddress(min_offset)
        self._max = _MockAddress(max_offset)

    def getDataType(self) -> Any:
        return self._datatype

    def getMinAddress(self) -> _MockAddress:
        return self._min

    def getMaxAddress(self) -> _MockAddress:
        return self._max


class _MockListing:
    """``getDataAt`` / ``getDataContaining`` resolve from a single
    address -> Data mapping. The predicate calls ``_data_at`` first and
    falls back to ``_data_containing``; for the tests we wire both to
    the same mapping so either lookup hits."""

    def __init__(self, data_by_addr: dict[int, _MockData]) -> None:
        self._data = data_by_addr

    def getDataAt(self, addr_obj: _MockAddress) -> Optional[_MockData]:
        return self._data.get(addr_obj.getOffset())

    def getDataContaining(self, addr_obj: _MockAddress) -> Optional[_MockData]:
        return self._data.get(addr_obj.getOffset())


class _MockMemory:
    def __init__(self, blocks_by_addr: dict[int, _MockBlock]) -> None:
        self._blocks = blocks_by_addr

    def getBlock(self, addr_obj: _MockAddress) -> Optional[_MockBlock]:
        return self._blocks.get(addr_obj.getOffset())


class _MockSymbolTable:
    def getSymbols(self, _addr_obj: _MockAddress) -> Iterable[Any]:
        return ()


class _MockRefType:
    """Stand-in for ``RefType``. ``is_jump`` + ``is_computed`` are the
    two booleans the helper inspects; ``COMPUTED_JUMP`` ref-types are
    constructed with both True, ``DATA`` ref-types with both False."""

    def __init__(self, *, is_jump: bool, is_computed: bool) -> None:
        self._is_jump = is_jump
        self._is_computed = is_computed

    def isJump(self) -> bool:
        return self._is_jump

    def isComputed(self) -> bool:
        return self._is_computed


_COMPUTED_JUMP_REF_TYPE = _MockRefType(is_jump=True, is_computed=True)
_DATA_REF_TYPE = _MockRefType(is_jump=False, is_computed=False)


class _MockReference:
    def __init__(self, ref_type: _MockRefType) -> None:
        self._ref_type = ref_type

    def getReferenceType(self) -> _MockRefType:
        return self._ref_type


class _MockAddressDestIterator:
    """Stand-in for ``AddressIterator`` returned by
    ``getReferenceDestinationIterator``. Walks a sorted list of
    destination addresses in ascending order; the helper stops via its
    own ``compareTo > 0`` check, but we honour the iterator protocol
    completely for fidelity."""

    def __init__(self, addrs: List[_MockAddress]) -> None:
        self._addrs = sorted(addrs, key=lambda a: a.getOffset())
        self._idx = 0

    def hasNext(self) -> bool:
        return self._idx < len(self._addrs)

    def next(self) -> _MockAddress:
        addr = self._addrs[self._idx]
        self._idx += 1
        return addr


class _MockReferenceManager:
    """Stand-in for ``ReferenceManager``.

    ``destinations`` is a dict ``{offset: [_MockReference, ...]}``; the
    iterator walks the sorted offsets and ``getReferencesTo`` returns
    the recorded refs for each offset. The helper trims at ``max_addr``
    via its own ``compareTo`` check, so we expose ALL destinations
    above the requested start address without pre-filtering.
    """

    def __init__(self, destinations: dict[int, List[_MockReference]]) -> None:
        self._destinations = destinations

    def getReferenceDestinationIterator(
        self, start_addr: _MockAddress, _forward: bool
    ) -> _MockAddressDestIterator:
        start = start_addr.getOffset()
        addrs = [
            _MockAddress(off) for off in self._destinations.keys() if off >= start
        ]
        return _MockAddressDestIterator(addrs)

    def getReferencesTo(self, addr_obj: _MockAddress) -> List[_MockReference]:
        return self._destinations.get(addr_obj.getOffset(), [])


class _MockProgram:
    def __init__(
        self,
        *,
        memory: _MockMemory,
        listing: _MockListing,
        reference_manager: _MockReferenceManager,
    ) -> None:
        self._memory = memory
        self._listing = listing
        self._reference_manager = reference_manager
        self._symbol_table = _MockSymbolTable()

    def getMemory(self) -> _MockMemory:
        return self._memory

    def getSymbolTable(self) -> _MockSymbolTable:
        return self._symbol_table

    def getListing(self) -> _MockListing:
        return self._listing

    def getAddressFactory(self) -> _MockAddressFactory:
        return _MockAddressFactory()

    def getReferenceManager(self) -> _MockReferenceManager:
        return self._reference_manager

    def getDefaultPointerSize(self) -> int:
        return 8


class _MockFunctionManager:
    """The predicate itself does not consult the FunctionManager, but
    ``GhidraMetadataLookup.__init__`` accepts one. A no-op stand-in
    suffices."""

    def getFunctions(self, _forward: bool) -> Iterable[Any]:
        return ()


# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------


_TEST_BINARY_ID_HASH = bytes(range(16))


def _build_lookup(
    *,
    block: _MockBlock,
    block_range: range,
    data_by_addr: dict[int, _MockData],
    ref_destinations: dict[int, List[_MockReference]],
) -> GhidraMetadataLookup:
    """Wire a ``GhidraMetadataLookup`` around the mock Program surface.

    ``block_range`` enumerates the addresses ``memory.getBlock`` reports
    as belonging to ``block``; ``data_by_addr`` wires both
    ``getDataAt`` and ``getDataContaining``; ``ref_destinations`` feeds
    the ReferenceManager mock so the semantic gate can be exercised.
    """
    memory = _MockMemory({addr: block for addr in block_range})
    listing = _MockListing(data_by_addr)
    ref_mgr = _MockReferenceManager(ref_destinations)
    program = _MockProgram(memory=memory, listing=listing, reference_manager=ref_mgr)
    fm = _MockFunctionManager()
    return GhidraMetadataLookup(program, fm, _TEST_BINARY_ID_HASH)


def _call_predicate(
    lookup: GhidraMetadataLookup, addr: int, block: _MockBlock
) -> bool:
    """Invoke the predicate the way ``_classify_address`` does."""
    addr_obj = (
        lookup._program.getAddressFactory()
        .getDefaultAddressSpace()
        .getAddress(addr)
    )
    return lookup._is_jump_table_slot(addr_obj, block)


# ---------------------------------------------------------------------------
# 1. POSITIVE: structural match + inbound COMPUTED_JUMP back-ref -> True.
# ---------------------------------------------------------------------------


def test_pointer_in_rodata_with_computed_jump_backref_is_jump_table_slot() -> None:
    """Canonical positive: a Pointer Data in ``.data.rel.ro`` WITH an
    inbound COMPUTED_JUMP back-ref into its address range is the target
    of a switch-table dispatch — the predicate must return True."""
    addr = 0x4000
    block = _MockBlock(".data.rel.ro")
    data = _MockData(_MockPointerDataType(), min_offset=addr, max_offset=addr + 7)
    lookup = _build_lookup(
        block=block,
        block_range=range(addr, addr + 8),
        data_by_addr={addr: data},
        ref_destinations={addr: [_MockReference(_COMPUTED_JUMP_REF_TYPE)]},
    )
    assert _call_predicate(lookup, addr, block) is True


# ---------------------------------------------------------------------------
# 2. NEGATIVE: structural match BUT no inbound COMPUTED_JUMP -> False.
#
# This is the LATENT-BUG GUARD. Without the semantic gate, the previous
# predicate returned True here, causing relro PLT GOT slots / vtable
# components / C++ typeinfo Pointer Data in ``.data.rel.ro`` to be
# mis-classified as ``JUMP_TABLE_SLOT``. The fix introduces this case
# as a False; without this test the latent-bug fix has no proof.
# ---------------------------------------------------------------------------


def test_pointer_in_rodata_without_any_backref_is_not_jump_table_slot() -> None:
    """Hardened-relro PLT GOT analogue / vtable component pointer:
    structurally a Pointer in ``.data.rel.ro``, but NO inbound refs at
    all. The semantic gate must reject this."""
    addr = 0x5000
    block = _MockBlock(".data.rel.ro")
    data = _MockData(_MockPointerDataType(), min_offset=addr, max_offset=addr + 7)
    lookup = _build_lookup(
        block=block,
        block_range=range(addr, addr + 8),
        data_by_addr={addr: data},
        ref_destinations={},  # no back-refs at all
    )
    assert _call_predicate(lookup, addr, block) is False


def test_pointer_in_rodata_with_only_data_refs_is_not_jump_table_slot() -> None:
    """Pointer in ``.data.rel.ro`` referenced ONLY by DATA reads (not
    COMPUTED_JUMP) — e.g. a C++ ``typeinfo`` pointer read at runtime
    via dynamic_cast. Structural match holds; semantic gate must
    reject."""
    addr = 0x6000
    block = _MockBlock(".data.rel.ro")
    data = _MockData(_MockPointerDataType(), min_offset=addr, max_offset=addr + 7)
    lookup = _build_lookup(
        block=block,
        block_range=range(addr, addr + 8),
        data_by_addr={addr: data},
        ref_destinations={addr: [_MockReference(_DATA_REF_TYPE)]},
    )
    assert _call_predicate(lookup, addr, block) is False


# ---------------------------------------------------------------------------
# 3. NEGATIVE: non-rodata block -> False (unchanged behaviour). Preserves
#    the pre-fix rejection of e.g. ``.got`` Pointers (the existing
#    reproducer ``arm32-gcc-4.8-O0_minigzip`` lands its GOT in ``.got``,
#    not the rodata-name set, so the original Bug never surfaced).
# ---------------------------------------------------------------------------


def test_pointer_in_got_block_is_not_jump_table_slot() -> None:
    """``.got`` is NOT a rodata-flavored block — the predicate must
    reject regardless of back-refs."""
    addr = 0x7000
    block = _MockBlock(".got")
    data = _MockData(_MockPointerDataType(), min_offset=addr, max_offset=addr + 7)
    lookup = _build_lookup(
        block=block,
        block_range=range(addr, addr + 8),
        data_by_addr={addr: data},
        # Even with a (hypothetical) computed-jump backref, the
        # block-name gate suppresses the match.
        ref_destinations={addr: [_MockReference(_COMPUTED_JUMP_REF_TYPE)]},
    )
    assert _call_predicate(lookup, addr, block) is False


# ---------------------------------------------------------------------------
# 4. NEGATIVE: rodata block, non-Pointer Data -> False (unchanged).
# ---------------------------------------------------------------------------


def test_non_pointer_data_in_rodata_is_not_jump_table_slot() -> None:
    """An ``int`` (or any non-Pointer) Data in ``.rodata`` fails the
    structural check; back-refs are irrelevant."""
    addr = 0x8000
    block = _MockBlock(".rodata")
    data = _MockData(_MockNonPointerDataType(), min_offset=addr, max_offset=addr + 3)
    lookup = _build_lookup(
        block=block,
        block_range=range(addr, addr + 4),
        data_by_addr={addr: data},
        ref_destinations={addr: [_MockReference(_COMPUTED_JUMP_REF_TYPE)]},
    )
    assert _call_predicate(lookup, addr, block) is False


# ---------------------------------------------------------------------------
# 5. POSITIVE (Array path): Array-of-Pointer with one slot referenced by
#    COMPUTED_JUMP -> True. Mirrors the canonical switch-table shape
#    where the dispatch site references the table's FIRST slot and
#    subsequent slots are addressed by offset; only the first slot
#    has the back-ref but the whole array is the switch table.
# ---------------------------------------------------------------------------


def test_array_of_pointer_with_one_computed_jump_backref_is_jump_table_slot() -> None:
    """Array-of-Pointer Data spanning several slots; only the first
    slot has an inbound COMPUTED_JUMP back-ref (the dispatch site
    indirected through the table base). The whole array qualifies as
    a switch table — the predicate must return True when queried at
    any slot covered by the array's ``[min, max]`` range."""
    table_base = 0xA000
    slot_size = 8
    slot_count = 4
    table_end = table_base + slot_size * slot_count - 1  # inclusive max
    block = _MockBlock(".rodata")

    inner_pointer = _MockPointerDataType()
    array_dt = _MockArrayOfPointerDataType(inner_pointer)
    array_data = _MockData(array_dt, min_offset=table_base, max_offset=table_end)

    # Every offset inside the array's range resolves to the SAME Data
    # (mirrors Ghidra's ``getDataContaining`` semantics where the
    # containing Data is the array, regardless of the queried slot).
    data_by_addr = {table_base + i: array_data for i in range(slot_size * slot_count)}

    lookup = _build_lookup(
        block=block,
        block_range=range(table_base, table_base + slot_size * slot_count),
        data_by_addr=data_by_addr,
        # The back-ref lands at the table base — first slot only.
        ref_destinations={table_base: [_MockReference(_COMPUTED_JUMP_REF_TYPE)]},
    )

    # Querying any slot of the array (here: slot 2) must qualify the
    # whole table.
    queried_slot_addr = table_base + 2 * slot_size
    assert _call_predicate(lookup, queried_slot_addr, block) is True


# ---------------------------------------------------------------------------
# 6. POSITIVE (Array path, mid-array back-ref): an Array-of-Pointer where
#    the COMPUTED_JUMP back-ref lands on a slot OTHER than the first.
#    Confirms the back-ref walk genuinely covers the full array span,
#    not just the min-address. Defensive against an off-by-one regression
#    in the ``getReferenceDestinationIterator`` walk's termination.
# ---------------------------------------------------------------------------


def test_array_with_computed_jump_backref_at_middle_slot_is_jump_table_slot() -> None:
    table_base = 0xB000
    slot_size = 8
    slot_count = 4
    table_end = table_base + slot_size * slot_count - 1
    middle_slot_addr = table_base + 2 * slot_size  # slot index 2 of 4
    block = _MockBlock(".rodata")

    array_dt = _MockArrayOfPointerDataType(_MockPointerDataType())
    array_data = _MockData(array_dt, min_offset=table_base, max_offset=table_end)
    data_by_addr = {
        table_base + i: array_data for i in range(slot_size * slot_count)
    }

    lookup = _build_lookup(
        block=block,
        block_range=range(table_base, table_base + slot_size * slot_count),
        data_by_addr=data_by_addr,
        ref_destinations={
            middle_slot_addr: [_MockReference(_COMPUTED_JUMP_REF_TYPE)]
        },
    )

    assert _call_predicate(lookup, table_base, block) is True


# ---------------------------------------------------------------------------
# 7. NEGATIVE (defensive): an inbound back-ref at an address OUTSIDE
#    the Data's span must NOT qualify the Data. This pins the
#    ``compareTo(max_addr) > 0`` termination of the iterator walk —
#    a regression that didn't stop scanning at ``max_addr`` would let
#    an unrelated downstream switch table's back-ref leak in and
#    falsely classify the Pointer as a slot.
# ---------------------------------------------------------------------------


def test_computed_jump_backref_outside_data_span_does_not_qualify() -> None:
    addr = 0xC000
    block = _MockBlock(".data.rel.ro")
    data = _MockData(_MockPointerDataType(), min_offset=addr, max_offset=addr + 7)
    # Back-ref lands well past the Data's max-address. The iterator
    # starts at min_addr and walks upward; the helper must stop at
    # ``compareTo(max_addr) > 0`` before reaching this far-away ref.
    far_offset = addr + 0x10000
    lookup = _build_lookup(
        block=block,
        block_range=range(addr, addr + 8),
        data_by_addr={addr: data},
        ref_destinations={far_offset: [_MockReference(_COMPUTED_JUMP_REF_TYPE)]},
    )
    assert _call_predicate(lookup, addr, block) is False
