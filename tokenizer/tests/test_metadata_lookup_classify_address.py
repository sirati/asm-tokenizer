"""Tests for ``GhidraMetadataLookup._classify_address`` symbol-vs-function
fork.

Concern: when Ghidra auto-creates a ``LAB_*`` label at an intra-function
branch target, the address has BOTH a (LABEL-typed) symbol AND a
containing function. The previous symbol-branch logic only consulted
``getFunctionAt`` (which returns the function only when the address IS
its entry point) and only promoted ``type_str`` to ``local_function``
for FUNCTION-typed symbols — so interior LAB_* addresses landed as
``AddressKind.UNKNOWN`` with a zero-width range rooted at the raw
address. That defeated ``_pred_block`` (which fires only for
LOCAL_FUNCTION + ``value != start_addr``) and the intra-function jump
target collapsed to ``valued_const_v2`` rather than ``block_v2``.

The fix hoists ``getFunctionContaining`` out of the symbol/no-symbol
fork; the symbol branch then drives both ``type_str = local_function``
and the entry-offset / body-size range off the containing function
whenever the address is in a code block. The label name itself is
preserved on ``meta.name`` for downstream debugging fidelity.

These tests build a duck-typed mock surface that satisfies the
lookup's Java-side calls (``program.getAddressFactory()``,
``memory.getBlock``, ``symbol_table.getSymbols``,
``listing.getDataAt`` / ``getDataContaining``, ``fm.getFunctionAt`` /
``getFunctionContaining``) so the real ``_classify_address`` method
runs end-to-end without a Ghidra JVM.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, List, Optional

import pytest

from tokenizer.disasm.ghidra_provider.metadata_lookup import GhidraMetadataLookup
from tokenizer.disasm.metadata import AddressKind


# ---------------------------------------------------------------------------
# Duck-typed Ghidra surface
# ---------------------------------------------------------------------------


class _MockAddress:
    """Stand-in for Ghidra's ``Address`` -- exposes ``getOffset()``."""

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
    """Stand-in for Ghidra's ``MemoryBlock``.

    Only the surface the classifier actually reads is implemented:
    ``getName()`` (drives ``_section_type_from_block``),
    ``isExternalBlock()`` (synthetic-extern detection), the
    permission-bit predicates (``isExecute`` / ``isWrite``), and the
    range/size accessors used by branch 3.
    """

    def __init__(
        self,
        name: str,
        *,
        external: bool = False,
        start: int = 0,
        end: int = 0,
        size: int = 0,
        is_execute: bool = True,
        is_write: bool = False,
    ) -> None:
        self._name = name
        self._external = external
        self._start = start
        self._end = end
        self._size = size
        self._is_execute = is_execute
        self._is_write = is_write

    def getName(self) -> str:
        return self._name

    def isExternalBlock(self) -> bool:
        return self._external

    def isExecute(self) -> bool:
        return self._is_execute

    def isWrite(self) -> bool:
        return self._is_write

    def getStart(self) -> _MockAddress:
        return _MockAddress(self._start)

    def getEnd(self) -> _MockAddress:
        return _MockAddress(self._end)

    def getSize(self) -> int:
        return self._size


class _MockSymbol:
    """Stand-in for ``Symbol`` -- the classifier reads ``getName`` and
    ``getSymbolType`` (the latter via ``str().upper()``-compare against
    ``FUNCTION``)."""

    def __init__(self, name: str, symbol_type: str) -> None:
        self._name = name
        self._symbol_type = symbol_type

    def getName(self) -> str:
        return self._name

    def getSymbolType(self) -> str:
        return self._symbol_type


class _MockBody:
    def __init__(self, num_addresses: int) -> None:
        self._num = int(num_addresses)

    def getNumAddresses(self) -> int:
        return self._num


class _MockFunction:
    """Stand-in for ``Function``. Implements the subset the classifier
    + canonical-name helpers touch: ``getEntryPoint``, ``getBody``,
    ``getName``, ``isExternal``, ``isThunk``, ``getComment``,
    ``getThunkedFunction``."""

    def __init__(
        self,
        *,
        entry: int,
        body_size: int,
        name: str = "func",
        external: bool = False,
        thunk: bool = False,
    ) -> None:
        self._entry = int(entry)
        self._body = _MockBody(body_size)
        self._name = name
        self._external = external
        self._thunk = thunk

    def getEntryPoint(self) -> _MockAddress:
        return _MockAddress(self._entry)

    def getBody(self) -> _MockBody:
        return self._body

    def getName(self) -> str:
        return self._name

    def isExternal(self) -> bool:
        return self._external

    def isThunk(self) -> bool:
        return self._thunk

    def getComment(self) -> Optional[str]:
        return None

    def getThunkedFunction(self, _follow: bool) -> Any:
        return None


class _MockMemory:
    def __init__(self, blocks_by_addr: dict[int, _MockBlock]) -> None:
        self._blocks = blocks_by_addr

    def getBlock(self, addr_obj: _MockAddress) -> Optional[_MockBlock]:
        return self._blocks.get(addr_obj.getOffset())


class _MockSymbolTable:
    def __init__(self, symbols_by_addr: dict[int, List[_MockSymbol]]) -> None:
        self._symbols = symbols_by_addr

    def getSymbols(self, addr_obj: _MockAddress) -> Iterable[_MockSymbol]:
        return self._symbols.get(addr_obj.getOffset(), [])


class _MockListing:
    """No string / data classification: every Data lookup returns None
    so the string + slot-detection branches stay silent. The classifier
    falls through to the section-based path under test."""

    def getDataAt(self, _addr_obj: _MockAddress) -> Any:
        return None

    def getDataContaining(self, _addr_obj: _MockAddress) -> Any:
        return None


class _MockFunctionManager:
    def __init__(
        self,
        *,
        functions_at: dict[int, _MockFunction],
        functions_containing: dict[int, _MockFunction],
    ) -> None:
        self._functions_at = functions_at
        self._functions_containing = functions_containing

    def getFunctionAt(self, addr_obj: _MockAddress) -> Optional[_MockFunction]:
        return self._functions_at.get(addr_obj.getOffset())

    def getFunctionContaining(self, addr_obj: _MockAddress) -> Optional[_MockFunction]:
        return self._functions_containing.get(addr_obj.getOffset())

    def getFunctions(self, _forward: bool) -> Iterable[_MockFunction]:
        # Used only by the lazy switch-table cache builder; the tests in
        # this file never resolve a slot_target, so an empty iterable is
        # sufficient.
        return ()


class _MockProgram:
    def __init__(
        self,
        *,
        memory: _MockMemory,
        symbol_table: _MockSymbolTable,
        listing: _MockListing,
        pointer_size: int = 8,
    ) -> None:
        self._memory = memory
        self._symbol_table = symbol_table
        self._listing = listing
        self._pointer_size = pointer_size

    def getMemory(self) -> _MockMemory:
        return self._memory

    def getSymbolTable(self) -> _MockSymbolTable:
        return self._symbol_table

    def getListing(self) -> _MockListing:
        return self._listing

    def getAddressFactory(self) -> _MockAddressFactory:
        return _MockAddressFactory()

    def getDefaultPointerSize(self) -> int:
        return self._pointer_size


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _build_lookup(
    *,
    block: _MockBlock,
    block_range: range,
    symbols_by_addr: dict[int, List[_MockSymbol]],
    functions_at: dict[int, _MockFunction],
    functions_containing: dict[int, _MockFunction],
) -> GhidraMetadataLookup:
    """Wire a ``GhidraMetadataLookup`` over the duck-typed mocks.

    ``block_range`` enumerates the offsets that should report ``block``
    from ``memory.getBlock``; the test cases pass a small range covering
    the addresses they query.
    """
    blocks_by_addr = {addr: block for addr in block_range}
    memory = _MockMemory(blocks_by_addr)
    symbol_table = _MockSymbolTable(symbols_by_addr)
    listing = _MockListing()
    program = _MockProgram(
        memory=memory,
        symbol_table=symbol_table,
        listing=listing,
    )
    fm = _MockFunctionManager(
        functions_at=functions_at,
        functions_containing=functions_containing,
    )
    return GhidraMetadataLookup(program, fm)


# ---------------------------------------------------------------------------
# The regression: LAB_* at an intra-function jump target must classify as
# LOCAL_FUNCTION with the containing function's range -- not UNKNOWN with
# a zero-width range at the raw address.
# ---------------------------------------------------------------------------


def test_classify_address_promotes_label_at_intra_function_code_to_local_function() -> None:
    """Ghidra auto-creates ``LAB_*`` LABEL symbols at branch targets.
    When the targeted address is strictly inside a function body, the
    symbol branch must resolve the LOCAL_FUNCTION range from the
    containing function so ``_pred_block`` fires (and the intra-function
    jump emits ``block_v2``, not ``valued_const_v2``).
    """
    func_entry = 0x1000
    func_body_size = 0x100
    intra_target = 0x1080  # strictly inside [func_entry, func_entry+body_size)

    func = _MockFunction(entry=func_entry, body_size=func_body_size, name="my_func")
    block = _MockBlock(".text", is_execute=True)
    label = _MockSymbol(name="LAB_00001080", symbol_type="LABEL")

    lookup = _build_lookup(
        block=block,
        block_range=range(func_entry, func_entry + func_body_size + 1),
        symbols_by_addr={intra_target: [label]},
        # ``getFunctionAt`` returns None at intra-function addresses
        # (only entry points report); ``getFunctionContaining`` returns
        # the function for any address in the body.
        functions_at={func_entry: func},
        functions_containing={
            addr: func for addr in range(func_entry, func_entry + func_body_size)
        },
    )

    view = lookup.lookup(intra_target)

    assert view.kind == AddressKind.LOCAL_FUNCTION, (
        f"LAB_* at intra-function address must classify as LOCAL_FUNCTION; "
        f"got {view.kind!r}"
    )
    assert view.start_addr == func_entry, (
        f"start_addr must be the containing function's entry offset "
        f"(0x{func_entry:x}); got 0x{view.start_addr:x}"
    )
    assert view.end_addr == func_entry + func_body_size, (
        f"end_addr must be entry + body_size (0x{func_entry + func_body_size:x}); "
        f"got 0x{view.end_addr:x}"
    )
    assert view.size == func_body_size, (
        f"size must match the function body's address count "
        f"({func_body_size}); got {view.size}"
    )
    # Label name preserved (NOT overwritten with the function name) so
    # downstream debugging consumers retain the precise per-address label.
    assert view.name == "LAB_00001080", (
        f"label name preservation broke -- got {view.name!r}"
    )


def test_classify_address_function_entry_with_function_symbol_preserved() -> None:
    """Sanity check on the unchanged happy path: a FUNCTION-typed symbol
    at the function entry still classifies as LOCAL_FUNCTION with the
    function's own range. The hoisted ``getFunctionContaining`` resolves
    to the same function at the entry point, so the previous behaviour
    is preserved byte-for-byte for canonical function entries.
    """
    func_entry = 0x2000
    func_body_size = 0x40

    func = _MockFunction(entry=func_entry, body_size=func_body_size, name="entry_func")
    block = _MockBlock(".text", is_execute=True)
    sym = _MockSymbol(name="entry_func", symbol_type="FUNCTION")

    lookup = _build_lookup(
        block=block,
        block_range=range(func_entry, func_entry + func_body_size + 1),
        symbols_by_addr={func_entry: [sym]},
        functions_at={func_entry: func},
        functions_containing={
            addr: func for addr in range(func_entry, func_entry + func_body_size)
        },
    )

    view = lookup.lookup(func_entry)

    assert view.kind == AddressKind.LOCAL_FUNCTION
    assert view.start_addr == func_entry
    assert view.end_addr == func_entry + func_body_size
    assert view.size == func_body_size
    assert view.name == "entry_func"


def test_classify_address_label_outside_function_body_stays_non_function() -> None:
    """Regression-guard for the opposite direction: a LABEL symbol on a
    ``.text`` byte that is NOT inside any function body must NOT be
    promoted to LOCAL_FUNCTION. ``getFunctionContaining`` returns None
    in this case, so the symbol branch falls back to the section's
    natural type. The pre-fix code took the same path here (no FUNCTION
    symbol -> no promotion), and the fix must not regress this.
    """
    addr = 0x3000
    block = _MockBlock(".text", is_execute=True)
    label = _MockSymbol(name="orphan_label", symbol_type="LABEL")

    lookup = _build_lookup(
        block=block,
        block_range=range(addr - 1, addr + 2),
        symbols_by_addr={addr: [label]},
        functions_at={},
        functions_containing={},  # no containing function
    )

    view = lookup.lookup(addr)

    # No containing function -> base_type "code" -> AddressKind.UNKNOWN
    # (``address_kind_from_string("code")`` returns UNKNOWN per
    # ``tokenizer/disasm/metadata.py``). The behaviour is unchanged from
    # before the fix; this test pins it explicitly.
    assert view.kind != AddressKind.LOCAL_FUNCTION, (
        f"orphan code label must NOT promote to LOCAL_FUNCTION; "
        f"got kind={view.kind!r}"
    )
    # And the range must NOT have been inflated -- start/end stay at the
    # raw address.
    assert view.start_addr == addr
    assert view.end_addr == addr
