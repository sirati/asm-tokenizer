"""Tests for ``walk_switch_tables_for_function`` in
``tokenizer.disasm.ghidra_provider.switch_table_walker``.

Concern: the helper is the single source of truth for the per-function
computed-jump dispatch walk; both
``GhidraDisassemblyProvider.iter_switch_tables`` and
``GhidraMetadataLookup._ensure_switch_table_cache`` delegate to it.

Coverage targets:
- The thunk gate (``skip_thunks=True`` skips, ``skip_thunks=False``
  walks anyway — the parameter exists for callers who want raw output).
- A real switch-table shape (DATA-READ + N COMPUTED_JUMP refs) yields
  ``(table_addr, [targets])``.
- Multiple dispatches in one function each yield their own tuple.
- The operand-object fallback fires when no DATA-READ ref is present
  but a typed Address operand carries the table base.
- The legacy ``RefType.COMPUTED_JUMP`` direct-equality path resolves
  targets even when the modern ``isJump()+isComputed()`` predicate is
  off (this is the fuller-recovery branch that the previous inline
  copies of the walk had drifted apart on; consolidation adopted it
  everywhere via ``is_computed_jump_reftype``).

The mock harness mirrors ``test_ghidra_switch_table_thunk_skip.py`` so
the two test files share a vocabulary for Ghidra fakery.
"""

from __future__ import annotations

import sys
import types
from typing import Iterable, List, Optional


# Stub the Ghidra symbol module so ``is_computed_jump_reftype`` can do
# its lazy ``from ghidra... import RefType`` without a JVM. Tests that
# exercise the legacy-equality path overwrite ``RefType.COMPUTED_JUMP``
# with the sentinel their mock RefType identity-compares against.
_STUB_GHIDRA_MODS = (
    "ghidra",
    "ghidra.program",
    "ghidra.program.model",
    "ghidra.program.model.symbol",
)
for _name in _STUB_GHIDRA_MODS:
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["ghidra.program.model.symbol"].RefType = types.SimpleNamespace(
    COMPUTED_JUMP=object()
)


from tokenizer.disasm.ghidra_provider.switch_table_walker import (  # noqa: E402
    walk_switch_tables_for_function,
)


# ---------------------------------------------------------------------------
# Mock Ghidra handles. Minimal surface to drive the walker; identical in
# spirit to test_ghidra_switch_table_thunk_skip.py's mocks but rebuilt
# locally so each test file owns its own fakes.
# ---------------------------------------------------------------------------


class _MockAddress:
    def __init__(self, offset: int) -> None:
        self._offset = offset

    def getOffset(self) -> int:
        return self._offset


class _MockRefType:
    """``Reference.getReferenceType()`` return value.

    The walker dispatches on the predicate methods (``isData``,
    ``isRead``, ``isJump``, ``isComputed``); the legacy-fallback path
    additionally identity-compares against ``RefType.COMPUTED_JUMP``,
    which tests opt into by mutating
    ``sys.modules['ghidra.program.model.symbol'].RefType.COMPUTED_JUMP``.
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
    """A dispatch instruction. By default it claims to be a
    computed-jump (flow=JUMP+COMPUTED); the operand-fallback test
    overrides this to leave operand-objects available."""

    def __init__(
        self,
        refs: list[_MockReference],
        *,
        op_objects: Optional[list[list[object]]] = None,
        flow_is_jump: bool = True,
        flow_is_computed: bool = True,
    ) -> None:
        self._refs = refs
        self._op_objects = op_objects or []
        self._flow_is_jump = flow_is_jump
        self._flow_is_computed = flow_is_computed

    def getFlowType(self) -> _MockFlowType:
        return _MockFlowType(
            is_jump=self._flow_is_jump, is_computed=self._flow_is_computed
        )

    def getReferencesFrom(self) -> list[_MockReference]:
        return list(self._refs)

    def getNumOperands(self) -> int:
        return len(self._op_objects)

    def getOpObjects(self, idx: int) -> tuple:
        if 0 <= idx < len(self._op_objects):
            return tuple(self._op_objects[idx])
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
    """Opaque sentinel: the listing keys instruction streams by id()."""


class _MockListing:
    def __init__(self, by_body: dict[int, list[_MockInstruction]]) -> None:
        self._by_body = by_body

    def getInstructions(self, body: _MockBody, _forward: bool) -> _MockInsnIterator:
        return _MockInsnIterator(list(self._by_body.get(id(body), [])))


class _MockFunction:
    def __init__(
        self,
        *,
        is_thunk: bool,
        instructions: list[_MockInstruction],
    ) -> None:
        self._is_thunk = is_thunk
        self._body = _MockBody()
        self._instructions = instructions

    def isThunk(self) -> bool:
        return self._is_thunk

    def getBody(self) -> _MockBody:
        return self._body


def _listing_for(func: _MockFunction) -> _MockListing:
    return _MockListing({id(func.getBody()): list(func._instructions)})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_yields_table_for_real_switch_dispatch() -> None:
    """A non-thunk function with one DATA-READ + N COMPUTED_JUMP refs
    yields a single ``(table_addr, [targets])`` tuple."""
    targets = [0x9870, 0x98E8, 0x98C8]
    refs = [
        _MockReference(_MockRefType(is_data=True, is_read=True), 0x5C),
    ] + [
        _MockReference(_MockRefType(is_jump=True, is_computed=True), t)
        for t in targets
    ]
    func = _MockFunction(is_thunk=False, instructions=[_MockInstruction(refs)])
    yielded = list(walk_switch_tables_for_function(func, _listing_for(func)))
    assert yielded == [(0x5C, targets)]


def test_skip_thunks_true_skips_plt_trampoline_shape() -> None:
    """The thunk gate suppresses output for PLT-trampoline-shaped
    functions even though the reference graph would otherwise produce a
    1-target table. This is the canonical thunk-bug regression case."""
    plt_refs = [
        _MockReference(_MockRefType(is_data=True, is_read=True), 0x21044),
        _MockReference(_MockRefType(is_jump=True, is_computed=True), 0x86C0),
    ]
    thunk = _MockFunction(is_thunk=True, instructions=[_MockInstruction(plt_refs)])
    yielded = list(walk_switch_tables_for_function(thunk, _listing_for(thunk)))
    assert yielded == []


def test_skip_thunks_false_walks_thunk_anyway() -> None:
    """Opt-out: callers wanting raw dispatch data on a thunk can pass
    ``skip_thunks=False``. Exposed for future audits / tests; both
    production callers leave the default (True)."""
    plt_refs = [
        _MockReference(_MockRefType(is_data=True, is_read=True), 0x21044),
        _MockReference(_MockRefType(is_jump=True, is_computed=True), 0x86C0),
    ]
    thunk = _MockFunction(is_thunk=True, instructions=[_MockInstruction(plt_refs)])
    yielded = list(
        walk_switch_tables_for_function(
            thunk, _listing_for(thunk), skip_thunks=False
        )
    )
    assert yielded == [(0x21044, [0x86C0])]


def test_multiple_dispatches_each_yield_a_tuple() -> None:
    """Two computed-jump instructions in one function emit two
    independent tuples in dispatch-instruction order."""
    t1 = [0x1000, 0x1010]
    t2 = [0x2000, 0x2010, 0x2020]
    insn1 = _MockInstruction(
        [_MockReference(_MockRefType(is_data=True, is_read=True), 0xAAA)]
        + [
            _MockReference(_MockRefType(is_jump=True, is_computed=True), t)
            for t in t1
        ]
    )
    insn2 = _MockInstruction(
        [_MockReference(_MockRefType(is_data=True, is_read=True), 0xBBB)]
        + [
            _MockReference(_MockRefType(is_jump=True, is_computed=True), t)
            for t in t2
        ]
    )
    func = _MockFunction(is_thunk=False, instructions=[insn1, insn2])
    yielded = list(walk_switch_tables_for_function(func, _listing_for(func)))
    assert yielded == [(0xAAA, t1), (0xBBB, t2)]


def test_operand_object_fallback_when_no_read_ref() -> None:
    """When no DATA-READ ref is present, the walker scans
    ``getOpObjects()`` for a typed Address operand carrying the table
    base. This is the fuller-recovery path adopted from the
    provider's inline walk during consolidation."""

    class _AddrObj:
        def __init__(self, offset: int) -> None:
            self._offset = offset

        def getOffset(self) -> int:
            return self._offset

    targets = [0x3000, 0x3010]
    insn = _MockInstruction(
        refs=[
            _MockReference(_MockRefType(is_jump=True, is_computed=True), t)
            for t in targets
        ],
        op_objects=[[], [_AddrObj(0xCAFE)]],
    )
    func = _MockFunction(is_thunk=False, instructions=[insn])
    yielded = list(walk_switch_tables_for_function(func, _listing_for(func)))
    assert yielded == [(0xCAFE, targets)]


def test_legacy_refType_equality_fallback_resolves_targets() -> None:
    """When ``isJump()`` / ``isComputed()`` are off, a RefType whose
    identity equals ``RefType.COMPUTED_JUMP`` still resolves as a
    computed-jump target. Mirrors older Ghidra builds where the
    predicate methods are unavailable."""
    sentinel = object()
    sys.modules["ghidra.program.model.symbol"].RefType = types.SimpleNamespace(
        COMPUTED_JUMP=sentinel
    )

    class _LegacyRefType:
        """RefType whose predicates are False — only the legacy
        identity-equality path can recognise it as COMPUTED_JUMP."""

        def isData(self) -> bool:
            return False

        def isRead(self) -> bool:
            return False

        def isJump(self) -> bool:
            return False

        def isComputed(self) -> bool:
            return False

        def __eq__(self, other: object) -> bool:
            return other is sentinel

        def __hash__(self) -> int:
            return 0

    refs = [
        _MockReference(_MockRefType(is_data=True, is_read=True), 0x4000),
        _MockReference(_LegacyRefType(), 0x4040),
        _MockReference(_LegacyRefType(), 0x4080),
    ]
    func = _MockFunction(is_thunk=False, instructions=[_MockInstruction(refs)])
    yielded = list(walk_switch_tables_for_function(func, _listing_for(func)))
    assert yielded == [(0x4000, [0x4040, 0x4080])]


def test_no_targets_emits_nothing() -> None:
    """An empty / non-dispatch function body yields nothing — covers
    the early-exit branches of the walker (no instructions, no
    matching flow type, no COMPUTED_JUMP refs)."""
    func = _MockFunction(is_thunk=False, instructions=[])
    yielded = list(walk_switch_tables_for_function(func, _listing_for(func)))
    assert yielded == []


def test_no_table_addr_skips_emission() -> None:
    """A computed-jump dispatch with COMPUTED_JUMP refs but no DATA-READ
    AND no typed-Address operand object cannot place the table base, so
    the walker drops it rather than emit a synthetic. Documents the
    'no locatable table base' branch."""
    insn = _MockInstruction(
        refs=[
            _MockReference(_MockRefType(is_jump=True, is_computed=True), 0x5050),
        ],
        op_objects=[],  # no operand objects → fallback finds nothing
    )
    func = _MockFunction(is_thunk=False, instructions=[insn])
    yielded = list(walk_switch_tables_for_function(func, _listing_for(func)))
    assert yielded == []


def test_none_function_yields_nothing() -> None:
    """Defensive: ``None`` ghidra_function returns the empty iterator."""
    assert list(walk_switch_tables_for_function(None, _MockListing({}))) == []


def test_isthunk_failure_falls_through() -> None:
    """A function whose ``isThunk()`` raises must not crash the walk;
    the gate falls through to the normal path and the function is
    walked anyway. Defensive parity with the previous inline copies."""

    class _AngryFunction(_MockFunction):
        def isThunk(self) -> bool:
            raise RuntimeError("isThunk explosion")

    refs = [
        _MockReference(_MockRefType(is_data=True, is_read=True), 0x6000),
        _MockReference(_MockRefType(is_jump=True, is_computed=True), 0x6040),
    ]
    func = _AngryFunction(is_thunk=False, instructions=[_MockInstruction(refs)])
    yielded = list(walk_switch_tables_for_function(func, _listing_for(func)))
    assert yielded == [(0x6000, [0x6040])]
