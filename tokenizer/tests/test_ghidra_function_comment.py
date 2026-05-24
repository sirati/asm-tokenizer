"""Tests for the Ghidra-side ``FunctionView.comment`` extraction.

Concern: ``_ghidra_function_comment`` in
:mod:`tokenizer.disasm.ghidra_views.function` reads ``getComment()``
off a Ghidra ``Function`` Java handle. The unit test exercises the
helper against hand-rolled mocks (no JVM required), covering:

* Functions without a plate comment return ``None``.
* Functions with a plate comment return the string verbatim
  (the demangled C++ scoped signature, in production).
* Defensive fallbacks: missing ``getComment`` attr, ``getComment``
  raising, non-string return values — all collapse to ``None``
  rather than crashing the iter loop.
"""

from __future__ import annotations

from typing import Any

from tokenizer.disasm.ghidra_views.function import _ghidra_function_comment


# ---------------------------------------------------------------------------
# Mock Ghidra handles
# ---------------------------------------------------------------------------


class _MockFunction:
    """Minimal Ghidra ``Function`` mock for plate-comment extraction.

    The production ``Function.getComment()`` Java method returns the
    plate comment string (or ``None`` if unset); these mocks cover
    that nominal path plus the defensive edge cases.
    """

    def __init__(
        self,
        comment: Any = None,
        raises: bool = False,
        missing_getter: bool = False,
    ) -> None:
        self._comment = comment
        self._raises = raises
        if missing_getter:
            self._has_getter = False
        else:
            self._has_getter = True

    def __getattr__(self, name: str) -> Any:
        if name == "getComment":
            if not self.__dict__.get("_has_getter", True):
                raise AttributeError(name)
            if self._raises:

                def _raise(*_: Any) -> Any:
                    raise RuntimeError("getComment explosion")

                return _raise
            return lambda: self._comment
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_comment_none_when_unset() -> None:
    """A function with no plate comment (the common C / asm symbol
    case) surfaces ``None``."""
    func = _MockFunction(comment=None)
    assert _ghidra_function_comment(func) is None


def test_comment_str_for_cpp_demangled_symbol() -> None:
    """C++ symbols carry the demangled scoped signature in the plate
    comment — verbatim string passthrough."""
    cpp = "ARPHeader::storeRecvData(unsigned char const*, unsigned int)"
    func = _MockFunction(comment=cpp)
    assert _ghidra_function_comment(func) == cpp


def test_comment_none_when_getter_missing() -> None:
    """Defensive: a Ghidra version without ``getComment`` returns None
    rather than AttributeError (matches the identity-key extractor's
    pattern)."""
    func = _MockFunction(missing_getter=True)
    assert _ghidra_function_comment(func) is None


def test_comment_none_when_getter_raises() -> None:
    """Defensive: a partially-populated program where the Java
    ``getComment`` call raises must not crash the iter loop."""
    func = _MockFunction(raises=True)
    assert _ghidra_function_comment(func) is None
