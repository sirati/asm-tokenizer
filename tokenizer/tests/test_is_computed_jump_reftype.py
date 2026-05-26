"""Tests for ``is_computed_jump_reftype`` in
``tokenizer.disasm.ghidra_provider.jump_table_predicates``.

Concern: the helper folds the modern ``isJump() and isComputed()``
predicate AND the legacy ``rt == RefType.COMPUTED_JUMP`` direct-equality
fallback into one defensive call. Four behavioral branches matter:

1. Modern predicate path (``isJump()`` AND ``isComputed()`` both True).
2. Legacy fallback (predicate False / unavailable, but ``rt ==
   RefType.COMPUTED_JUMP`` by identity).
3. Neither matches (regular jump, call, data ref, ...).
4. Defensive: JPype exception while calling the predicate methods or
   doing the equality compare must return False rather than propagating.
"""

from __future__ import annotations

import sys
import types


# Stub the ``ghidra.program.model.symbol`` module path so the helper's
# lazy ``from ghidra... import RefType`` resolves without a JVM. Each
# test then binds the sentinel ``RefType`` to whatever value the
# specific scenario needs.
_STUB_GHIDRA_MODS = (
    "ghidra",
    "ghidra.program",
    "ghidra.program.model",
    "ghidra.program.model.symbol",
)
for _name in _STUB_GHIDRA_MODS:
    sys.modules.setdefault(_name, types.ModuleType(_name))


from tokenizer.disasm.ghidra_provider.jump_table_predicates import (  # noqa: E402
    is_computed_jump_reftype,
)


class _RefTypeWithPredicates:
    """Stand-in RefType exposing ``isJump`` / ``isComputed``.

    Mirrors the Ghidra ``RefType`` interface surface that the modern
    predicate path of the helper touches.
    """

    def __init__(self, *, is_jump: bool, is_computed: bool) -> None:
        self._is_jump = is_jump
        self._is_computed = is_computed

    def isJump(self) -> bool:
        return self._is_jump

    def isComputed(self) -> bool:
        return self._is_computed


class _RefTypeThatRaises:
    """RefType whose predicate methods both raise — the defensive branch.

    A partially-populated JPype proxy can blow up under either call;
    the helper must absorb the exception and continue (no propagation,
    no crash; just falls through to the legacy-equality path and then
    to False).
    """

    def isJump(self) -> bool:
        raise RuntimeError("JPype boom (isJump)")

    def isComputed(self) -> bool:
        raise RuntimeError("JPype boom (isComputed)")


def test_modern_predicate_path_returns_true_for_jump_and_computed() -> None:
    """Modern path: both ``isJump`` and ``isComputed`` return True."""
    rt = _RefTypeWithPredicates(is_jump=True, is_computed=True)
    # Sentinel RefType.COMPUTED_JUMP must NOT equal `rt`, so the modern
    # path is the only branch that can return True.
    sys.modules["ghidra.program.model.symbol"].RefType = types.SimpleNamespace(
        COMPUTED_JUMP=object()
    )
    assert is_computed_jump_reftype(rt) is True


def test_modern_predicate_returns_false_for_jump_only() -> None:
    """``isJump=True`` alone (e.g. a direct conditional jump) must NOT match."""
    rt = _RefTypeWithPredicates(is_jump=True, is_computed=False)
    sys.modules["ghidra.program.model.symbol"].RefType = types.SimpleNamespace(
        COMPUTED_JUMP=object()
    )
    assert is_computed_jump_reftype(rt) is False


def test_modern_predicate_returns_false_for_computed_only() -> None:
    """``isComputed=True`` alone (e.g. a computed-call) must NOT match."""
    rt = _RefTypeWithPredicates(is_jump=False, is_computed=True)
    sys.modules["ghidra.program.model.symbol"].RefType = types.SimpleNamespace(
        COMPUTED_JUMP=object()
    )
    assert is_computed_jump_reftype(rt) is False


def test_modern_predicate_returns_false_for_neither() -> None:
    """A vanilla data/read ref: neither isJump nor isComputed."""
    rt = _RefTypeWithPredicates(is_jump=False, is_computed=False)
    sys.modules["ghidra.program.model.symbol"].RefType = types.SimpleNamespace(
        COMPUTED_JUMP=object()
    )
    assert is_computed_jump_reftype(rt) is False


def test_legacy_equality_fallback_returns_true() -> None:
    """Older Ghidra versions where the predicate methods are missing /
    misclassify still match via ``rt == RefType.COMPUTED_JUMP``."""
    sentinel = object()
    sys.modules["ghidra.program.model.symbol"].RefType = types.SimpleNamespace(
        COMPUTED_JUMP=sentinel
    )

    class _RawRefType:
        # No predicate methods → AttributeError → modern path fails →
        # legacy equality check fires.
        def __eq__(self, other: object) -> bool:  # noqa: D401
            return other is sentinel

        def __hash__(self) -> int:
            return 0

    assert is_computed_jump_reftype(_RawRefType()) is True


def test_jpype_exception_path_returns_false() -> None:
    """Defensive: predicate raises AND equality is False → helper returns
    False rather than propagating the exception."""
    # RefType.COMPUTED_JUMP set to a sentinel that won't match an
    # un-overridden ``__eq__`` on the raising object.
    sys.modules["ghidra.program.model.symbol"].RefType = types.SimpleNamespace(
        COMPUTED_JUMP=object()
    )
    rt = _RefTypeThatRaises()
    assert is_computed_jump_reftype(rt) is False


def test_jpype_exception_then_legacy_match_returns_true() -> None:
    """Defensive: predicate raises, but legacy equality matches the
    sentinel — helper must still return True."""
    sentinel = object()
    sys.modules["ghidra.program.model.symbol"].RefType = types.SimpleNamespace(
        COMPUTED_JUMP=sentinel
    )

    class _RaisingThenEqual:
        def isJump(self) -> bool:
            raise RuntimeError("boom")

        def isComputed(self) -> bool:  # pragma: no cover - short-circuit
            raise RuntimeError("boom")

        def __eq__(self, other: object) -> bool:
            return other is sentinel

        def __hash__(self) -> int:
            return 0

    assert is_computed_jump_reftype(_RaisingThenEqual()) is True


def test_none_returns_false() -> None:
    """Defensive: ``rt is None`` must not crash; returns False."""
    assert is_computed_jump_reftype(None) is False
