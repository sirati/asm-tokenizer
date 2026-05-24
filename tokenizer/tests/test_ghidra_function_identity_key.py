"""Tests for the Ghidra-side ``FunctionView.identity_key`` extraction.

Concern: ``_ghidra_identity_key`` in
:mod:`tokenizer.disasm.ghidra_views.function` reads ``isThunk`` and
``getThunkedFunction(True).getEntryPoint().getOffset()`` off a Ghidra
``Function`` Java handle. The unit test exercises the helper against
hand-rolled mocks (no JVM required), covering:

* Non-thunk functions return ``None``.
* Thunk functions return the resolved external's entry-point offset.
* Defensive fallbacks: missing ``getThunkedFunction`` attr,
  ``getThunkedFunction`` raising, the thunk returning ``None``, the
  external having a misbehaving address - all collapse to ``None``
  rather than crashing the iter loop.
"""

from __future__ import annotations

from typing import Any

from tokenizer.disasm.ghidra_views.function import _ghidra_identity_key


# ---------------------------------------------------------------------------
# Mock Ghidra handles
# ---------------------------------------------------------------------------


class _MockAddress:
    def __init__(self, offset: int) -> None:
        self._offset = offset

    def getOffset(self) -> int:
        return self._offset


class _MockExternalFunction:
    """The Function returned by ``ghidra_func.getThunkedFunction(True)``."""

    def __init__(self, entry_offset: int) -> None:
        self._entry = _MockAddress(entry_offset)

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


def test_identity_key_offset_for_thunk_resolving_to_external() -> None:
    """A PLT-thunk function returns the resolved external's
    entry-point offset (the stable cross-ISA-variant key)."""
    external = _MockExternalFunction(entry_offset=0xCAFEBABE)
    func = _MockFunction(is_thunk=True, thunked=external)
    assert _ghidra_identity_key(func) == 0xCAFEBABE


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
