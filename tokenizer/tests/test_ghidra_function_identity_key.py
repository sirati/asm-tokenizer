"""Tests for the Ghidra-side ``FunctionView.identity_key`` extraction.

Concern: ``_ghidra_identity_key`` in
:mod:`tokenizer.disasm.ghidra_views.function` reads ``isThunk``,
``getThunkedFunction(True)``, ``isExternal`` and either ``getName`` (for
external-target thunks) or ``getEntryPoint().getOffset()`` (for local-
target thunks) off a Ghidra ``Function`` Java handle. The unit test
exercises the helper against hand-rolled mocks (no JVM required),
covering:

* Non-thunk functions return ``None``.
* External-target thunks return ``ThunkIdentity(EXTERNAL, name)`` —
  the imported symbol name is the cross-binary-stable key (Ghidra's
  per-binary EXTERNAL placeholder offset would NOT be).
* Local-target thunks (rare; hand-written aliases, IFUNCs) return
  ``ThunkIdentity(LOCAL, hex_offset)`` — within-binary stable, the
  legacy disambiguation behaviour preserved.
* Defensive fallbacks: missing ``getThunkedFunction`` attr,
  ``getThunkedFunction`` raising, the thunk returning ``None``, the
  external having a misbehaving address - all collapse to ``None``
  rather than crashing the iter loop.
"""

from __future__ import annotations

from typing import Any, Optional

from tokenizer.disasm.ghidra_views.function import _ghidra_identity_key
from tokenizer.function_deduper import ThunkIdentity, ThunkTargetKind


# ---------------------------------------------------------------------------
# Mock Ghidra handles
# ---------------------------------------------------------------------------


class _MockAddress:
    def __init__(self, offset: int) -> None:
        self._offset = offset

    def getOffset(self) -> int:
        return self._offset


class _MockExternalFunction:
    """The Function returned by ``ghidra_func.getThunkedFunction(True)``.

    Carries the two axes the identity-key helper reads off a resolved
    thunk target: ``isExternal()`` (the EXTERNAL-block discriminator),
    ``getName()`` (the imported symbol name when external — the cross-
    binary stable key), and ``getEntryPoint().getOffset()`` (the local
    entry-point when the target is real code).
    """

    def __init__(
        self,
        *,
        is_external: bool,
        name: Optional[str] = None,
        entry_offset: int = 0,
    ) -> None:
        self._is_external = is_external
        self._name = name
        self._entry = _MockAddress(entry_offset)

    def isExternal(self) -> bool:
        return self._is_external

    def getName(self) -> str:
        return self._name if self._name is not None else ""

    def getEntryPoint(self) -> _MockAddress:
        return self._entry


class _MockFunction:
    """Minimal Ghidra ``Function`` mock for identity-key extraction."""

    def __init__(
        self,
        is_thunk: bool,
        thunked: Any = None,
        thunked_raises: bool = False,
        missing_thunk_getter: bool = False,
    ) -> None:
        self._is_thunk = is_thunk
        self._thunked = thunked
        self._thunked_raises = thunked_raises
        if missing_thunk_getter:
            # Drop the attr entirely so ``getattr(func, "getThunkedFunction",
            # None)`` returns None - mirrors a Ghidra version without that
            # method.
            self._has_thunk_getter = False
        else:
            self._has_thunk_getter = True

    def isThunk(self) -> bool:
        return self._is_thunk

    def __getattr__(self, name: str) -> Any:
        if name == "getThunkedFunction":
            if not self.__dict__.get("_has_thunk_getter", True):
                raise AttributeError(name)
            if self._thunked_raises:

                def _raise(*_: Any) -> Any:
                    raise RuntimeError("thunked-getter explosion")

                return _raise
            return lambda _recurse: self._thunked
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_identity_key_none_for_non_thunk() -> None:
    """A regular (non-thunk) function has no stronger-than-name
    identity; the extractor returns None and the consumer falls back
    to the legacy disambiguation path."""
    func = _MockFunction(is_thunk=False)
    assert _ghidra_identity_key(func) is None


def test_identity_key_external_thunk_uses_symbol_name() -> None:
    """An external-target PLT thunk returns
    ``ThunkIdentity(EXTERNAL, name)`` — the imported symbol name is
    cross-binary stable; Ghidra's per-binary EXTERNAL placeholder
    offset is NOT (it shifts with link order across binaries)."""
    external = _MockExternalFunction(is_external=True, name="gzseek")
    func = _MockFunction(is_thunk=True, thunked=external)
    assert _ghidra_identity_key(func) == ThunkIdentity(
        kind=ThunkTargetKind.EXTERNAL, key="gzseek"
    )


def test_identity_key_local_thunk_uses_hex_offset() -> None:
    """A local-target thunk (rare — hand-written assembly aliases,
    IFUNCs, some toolchain trampolines) returns
    ``ThunkIdentity(LOCAL, hex_offset)`` — within-binary stable,
    matching the legacy disambiguation invariant."""
    local = _MockExternalFunction(is_external=False, entry_offset=0x41E000)
    func = _MockFunction(is_thunk=True, thunked=local)
    assert _ghidra_identity_key(func) == ThunkIdentity(
        kind=ThunkTargetKind.LOCAL, key="41e000"
    )


def test_identity_key_distinguishes_external_from_local_with_same_key() -> None:
    """The kind axis matters: an EXTERNAL ``"abc"`` and a LOCAL
    ``"abc"`` identity must NOT compare equal (the deduper would
    otherwise collide unrelated thunks). The frozen dataclass equality
    folds in both fields."""
    extern = ThunkIdentity(kind=ThunkTargetKind.EXTERNAL, key="abc")
    local = ThunkIdentity(kind=ThunkTargetKind.LOCAL, key="abc")
    assert extern != local
    # Hashable distinct -- safe to use as dict keys side by side.
    assert {extern: 1, local: 2} == {extern: 1, local: 2}


def test_identity_key_cross_binary_external_thunk_is_stable() -> None:
    """The cross-binary-stability invariant: two binaries' EXTERNAL
    blocks place ``gzseek`` at DIFFERENT placeholder offsets, but the
    extractor reads the symbol name (not the offset) — so both
    binaries' thunks produce the SAME identity_key, and downstream
    ``canonical_function_name`` produces the SAME canonical name."""
    binary_a_thunk = _MockFunction(
        is_thunk=True,
        thunked=_MockExternalFunction(
            is_external=True, name="gzseek", entry_offset=0x1056296
        ),
    )
    binary_b_thunk = _MockFunction(
        is_thunk=True,
        thunked=_MockExternalFunction(
            is_external=True, name="gzseek", entry_offset=0x1056438
        ),
    )
    assert _ghidra_identity_key(binary_a_thunk) == _ghidra_identity_key(binary_b_thunk)


def test_identity_key_none_when_thunked_function_returns_none() -> None:
    """If Ghidra reports ``isThunk()`` True but the thunked-Function
    getter returns ``None`` (no resolved external), the extractor
    declines to assert identity (returns None)."""
    func = _MockFunction(is_thunk=True, thunked=None)
    assert _ghidra_identity_key(func) is None


def test_identity_key_none_when_thunked_getter_raises() -> None:
    """Defensive: a partially-populated program where the
    ``getThunkedFunction`` Java call raises must not crash the iter
    loop; the extractor swallows + returns None."""
    func = _MockFunction(is_thunk=True, thunked_raises=True)
    assert _ghidra_identity_key(func) is None


def test_identity_key_none_when_thunked_getter_missing() -> None:
    """Defensive: a Ghidra version without ``getThunkedFunction``
    returns None rather than AttributeError."""
    func = _MockFunction(is_thunk=True, missing_thunk_getter=True)
    assert _ghidra_identity_key(func) is None
