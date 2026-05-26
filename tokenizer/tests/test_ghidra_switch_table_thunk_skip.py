"""Tests for the thunk short-circuit in Ghidra switch-table recovery.

Concern: ``GhidraDisassemblyProvider.iter_switch_tables`` and
``GhidraMetadataLookup._ensure_switch_table_cache`` walk every function
to harvest computed-jump dispatches. PLT trampolines (``ldr pc,
[GOT_slot]`` on ARM, ``jmp [GOT_slot]`` on x86) have a reference-graph
shape that is structurally identical to a 1-target switch-table
dispatch:

    flow_type = COMPUTED_JUMP
    refs_from = [DATA-READ -> GOT_slot, COMPUTED_JUMP -> PLT0]

That tripped both call sites into emitting bogus ``(GOT_slot, [PLT0])``
"jump tables" for every PLT thunk in the binary. The fix gates both
walks on ``func.isThunk()`` — the authoritative Ghidra signal already
consumed by ``_ghidra_identity_key`` for cross-binary thunk-canonical
naming.

These tests exercise the gate against hand-rolled mocks (no JVM
required); the existing ``_ghidra_identity_key`` mock harness in
``test_ghidra_function_identity_key.py`` is the reference style.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Iterable, Optional

# The provider's ``iter_switch_tables`` does a deferred
# ``from ghidra.program.model.symbol import RefType`` (the import is
# guarded by all the predicate branches above it, but Python still
# resolves the module). The cache builder does the same import under a
# try/except. Stub the namespace path with a sentinel ``RefType``
# attribute so both call sites resolve without a JVM — the tests don't
# exercise the equality-fallback branch that uses the real object.
_STUB_GHIDRA_MODS = (
    "ghidra",
    "ghidra.program",
    "ghidra.program.model",
    "ghidra.program.model.symbol",
)
for _name in _STUB_GHIDRA_MODS:
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["ghidra.program.model.symbol"].RefType = object()

from tokenizer.disasm.ghidra_provider.metadata_lookup import GhidraMetadataLookup  # noqa: E402
from tokenizer.disasm.ghidra_provider.provider import GhidraDisassemblyProvider  # noqa: E402


# ---------------------------------------------------------------------------
# Mock Ghidra handles
#
# Minimum surface to drive the two switch-table walks: ``isThunk()`` on
# the function; ``getBody()`` returning an opaque marker the mock listing
# keys on; a listing that yields a single computed-jump instruction; and
# that instruction's ``getReferencesFrom()`` returning a READ + a
# COMPUTED_JUMP ref (the PLT-trampoline shape).
# ---------------------------------------------------------------------------


class _MockAddress:
    def __init__(self, offset: int) -> None:
        self._offset = offset

    def getOffset(self) -> int:
        return self._offset


class _MockRefType:
    """The discriminator object returned by ``Reference.getReferenceType()``.

    The walks call ``.isData()`` / ``.isRead()`` to identify the table-
    base READ ref, and ``.isJump()`` / ``.isComputed()`` to identify
    each resolved target ref. Both walks also accept a direct equality
    fallback against ``RefType.COMPUTED_JUMP`` — not exercised here
    because the predicate path matches first.
    """

    def __init__(
        self,
        *,
        is_data: bool = False,
        is_read: bool = False,
        is_jump: bool = False,
        is_computed: bool = False,
    ) -> None:
        self._is_data = is_data
        self._is_read = is_read
        self._is_jump = is_jump
        self._is_computed = is_computed

    def isData(self) -> bool:
        return self._is_data

    def isRead(self) -> bool:
        return self._is_read

    def isJump(self) -> bool:
        return self._is_jump

    def isComputed(self) -> bool:
        return self._is_computed


class _MockReference:
    def __init__(self, ref_type: _MockRefType, to_addr: int) -> None:
        self._type = ref_type
        self._to = _MockAddress(to_addr)

    def getReferenceType(self) -> _MockRefType:
        return self._type

    def getToAddress(self) -> _MockAddress:
        return self._to


class _MockFlowType:
    def __init__(self, *, is_jump: bool, is_computed: bool) -> None:
        self._is_jump = is_jump
        self._is_computed = is_computed

    def isJump(self) -> bool:
        return self._is_jump

    def isComputed(self) -> bool:
        return self._is_computed


class _MockInstruction:
    """A computed-jump instruction with the PLT-trampoline reference shape."""

    def __init__(self, refs: list[_MockReference]) -> None:
        self._refs = refs

    def getFlowType(self) -> _MockFlowType:
        return _MockFlowType(is_jump=True, is_computed=True)

    def getReferencesFrom(self) -> list[_MockReference]:
        return list(self._refs)

    def getNumOperands(self) -> int:  # pragma: no cover - not reached on PLT shape
        return 0

    def getOpObjects(self, _idx: int) -> tuple:  # pragma: no cover
        return ()


class _MockInsnIterator:
    def __init__(self, insns: list[_MockInstruction]) -> None:
        self._insns = list(insns)
        self._i = 0

    def hasNext(self) -> bool:
        return self._i < len(self._insns)

    def next(self) -> _MockInstruction:
        insn = self._insns[self._i]
        self._i += 1
        return insn


class _MockBody:
    """Opaque marker passed to ``listing.getInstructions(body, True)``.

    The mock listing keys on object identity to look up the matching
    instruction stream.
    """


class _MockListing:
    def __init__(self, by_body: dict[int, list[_MockInstruction]]) -> None:
        # Keyed by id(body) so each function's body is its own marker.
        self._by_body = by_body

    def getInstructions(self, body: _MockBody, _forward: bool) -> _MockInsnIterator:
        return _MockInsnIterator(list(self._by_body.get(id(body), [])))


class _MockFunction:
    def __init__(
        self,
        *,
        is_thunk: bool,
        entry_offset: int,
        instructions: list[_MockInstruction],
    ) -> None:
        self._is_thunk = is_thunk
        self._entry = _MockAddress(entry_offset)
        self._body = _MockBody()
        self._instructions = instructions

    def isThunk(self) -> bool:
        return self._is_thunk

    def getEntryPoint(self) -> _MockAddress:
        return self._entry

    def getBody(self) -> _MockBody:
        return self._body


class _MockFunctionManager:
    def __init__(self, funcs: list[_MockFunction]) -> None:
        self._funcs = list(funcs)

    def getFunctions(self, _forward: bool) -> Iterable[_MockFunction]:
        return iter(self._funcs)


class _FunctionViewStub:
    """Stand-in for the real ``FunctionView`` shape that
    ``iter_switch_tables`` reads.

    The method only touches ``.entry`` (an int offset). Anything richer
    would couple the test to the FunctionView protocol's other slots
    irrelevant to switch-table recovery.
    """

    def __init__(self, entry: int) -> None:
        self.entry = entry


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


# The exact reference shape that PLT trampolines exhibit and that fooled
# the recovery into emitting a bogus 1-target jump table. Built fresh
# per test (``_MockReference`` is stateless but reuse risks accidental
# aliasing between independent fixtures).
def _plt_trampoline_refs(*, got_slot: int, plt0: int) -> list[_MockReference]:
    return [
        _MockReference(_MockRefType(is_data=True, is_read=True), got_slot),
        _MockReference(_MockRefType(is_jump=True, is_computed=True), plt0),
    ]


def _build_provider(funcs: list[_MockFunction]) -> GhidraDisassemblyProvider:
    """Hand-build a provider with the minimum state ``iter_switch_tables``
    reads.

    The provider's ``__init__`` boots pyghidra / the JVM; the
    switch-table method only touches ``_funcs_by_entry`` (entry -> Ghidra
    Function handle) and ``_listing`` (yields instructions for a given
    body). Bypassing ``__init__`` via ``object.__new__`` keeps the test
    JVM-free.
    """
    provider = object.__new__(GhidraDisassemblyProvider)
    by_body: dict[int, list[_MockInstruction]] = {
        id(f.getBody()): list(f._instructions) for f in funcs
    }
    provider._funcs_by_entry = {int(f.getEntryPoint().getOffset()): f for f in funcs}
    provider._listing = _MockListing(by_body)
    return provider


def _build_metadata_lookup(
    funcs: list[_MockFunction],
) -> GhidraMetadataLookup:
    """Hand-build a metadata lookup with the minimum state
    ``_ensure_switch_table_cache`` reads.

    The constructor walks ``program.getMemory()`` / ``getSymbolTable()``
    / ``getListing()`` — none of which the cache builder needs. Bypass
    ``__init__`` and stamp ``_fm`` (the function-manager iterator
    source), ``_listing`` (instruction stream for each function body),
    and ``_switch_table_cache`` (the None sentinel that triggers a
    rebuild).
    """
    lookup = object.__new__(GhidraMetadataLookup)
    by_body: dict[int, list[_MockInstruction]] = {
        id(f.getBody()): list(f._instructions) for f in funcs
    }
    lookup._fm = _MockFunctionManager(funcs)
    lookup._listing = _MockListing(by_body)
    lookup._switch_table_cache = None
    return lookup


# ---------------------------------------------------------------------------
# Tests: iter_switch_tables (provider canonical path)
# ---------------------------------------------------------------------------


def test_iter_switch_tables_skips_thunk_with_plt_trampoline_shape() -> None:
    """Positive: a thunk whose body matches the PLT-trampoline reference
    shape (DATA-READ to GOT slot + COMPUTED_JUMP to PLT0) is skipped,
    NOT emitted as a bogus 1-target jump table. This is the bug the
    fix closes."""
    thunk = _MockFunction(
        is_thunk=True,
        entry_offset=0x8720,
        instructions=[
            _MockInstruction(_plt_trampoline_refs(got_slot=0x21044, plt0=0x86C0)),
        ],
    )
    provider = _build_provider([thunk])
    assert list(provider.iter_switch_tables(_FunctionViewStub(entry=0x8720))) == []


def test_iter_switch_tables_preserves_real_switch_table() -> None:
    """Negative: a non-thunk function with the same reference-graph
    shape as a real switch dispatch (DATA-READ to table base in
    rodata + N COMPUTED_JUMP targets) is preserved. The fix must NOT
    over-skip — real switch tables stay."""
    targets = [0x9870, 0x98E8, 0x98C8, 0x98B8, 0x98D8, 0x9860]
    refs = [
        _MockReference(
            _MockRefType(is_data=True, is_read=True), 0x5C
        ),  # table base in rodata
    ] + [
        _MockReference(
            _MockRefType(is_jump=True, is_computed=True), t
        )
        for t in targets
    ]
    real_switch = _MockFunction(
        is_thunk=False,
        entry_offset=0x9800,
        instructions=[_MockInstruction(refs)],
    )
    provider = _build_provider([real_switch])
    yielded = list(provider.iter_switch_tables(_FunctionViewStub(entry=0x9800)))
    assert yielded == [(0x5C, targets)]


def test_iter_switch_tables_thunk_skip_robust_to_isthunk_failure() -> None:
    """Defensive: a Function handle whose ``isThunk()`` raises must not
    crash the iter loop. The walk falls through to the normal path —
    here that means the (non-thunk-shaped) body emits whatever it
    emits, which for an empty body is nothing."""

    class _AngryFunction(_MockFunction):
        def isThunk(self) -> bool:
            raise RuntimeError("isThunk explosion")

    func = _AngryFunction(
        is_thunk=False,  # unused; isThunk raises
        entry_offset=0xDEAD,
        instructions=[],  # empty body — fall-through path yields nothing
    )
    provider = _build_provider([func])
    # The point is "does NOT raise"; the empty-body outcome is incidental.
    assert list(provider.iter_switch_tables(_FunctionViewStub(entry=0xDEAD))) == []


# ---------------------------------------------------------------------------
# Tests: _ensure_switch_table_cache (metadata-lookup parity path)
# ---------------------------------------------------------------------------


def test_ensure_switch_table_cache_skips_thunks() -> None:
    """The cache walk mirrors the gate: a thunk with PLT-trampoline
    reference shape contributes no entries to the cache, while a real
    switch-table function (same reference shape minus the thunk flag)
    does.

    This is the parity test for the second short-circuit; without it,
    ``GhidraMetadataLookup._ensure_switch_table_cache`` would still
    emit bogus (GOT_slot, [PLT0]) entries, leaking them into the
    JUMP_TABLE_SLOT slot_target resolution path even after the
    canonical path is fixed.
    """
    thunk_refs = _plt_trampoline_refs(got_slot=0x21044, plt0=0x86C0)
    real_targets = [0x9870, 0x988C]
    real_refs = [
        _MockReference(_MockRefType(is_data=True, is_read=True), 0x5C),
    ] + [
        _MockReference(_MockRefType(is_jump=True, is_computed=True), t)
        for t in real_targets
    ]
    thunk = _MockFunction(
        is_thunk=True,
        entry_offset=0x8720,
        instructions=[_MockInstruction(thunk_refs)],
    )
    real_switch = _MockFunction(
        is_thunk=False,
        entry_offset=0x9800,
        instructions=[_MockInstruction(real_refs)],
    )
    lookup = _build_metadata_lookup([thunk, real_switch])
    cache = lookup._ensure_switch_table_cache()

    # Real switch's table base is present with its targets...
    assert cache.get(0x5C) == real_targets
    # ...and the PLT thunk's GOT slot is NOT registered as a "table".
    assert 0x21044 not in cache


def test_ensure_switch_table_cache_thunk_skip_robust_to_isthunk_failure() -> None:
    """Defensive: an ``isThunk()`` failure during the cache walk does
    NOT abort the walk; the function falls through to the normal
    body-iteration path (which, for an empty body, yields nothing)."""

    class _AngryFunction(_MockFunction):
        def isThunk(self) -> bool:
            raise RuntimeError("isThunk explosion")

    angry = _AngryFunction(
        is_thunk=False,
        entry_offset=0xDEAD,
        instructions=[],
    )
    lookup = _build_metadata_lookup([angry])
    # Must not raise; cache is empty since the body has no instructions.
    cache = lookup._ensure_switch_table_cache()
    assert cache == {}
