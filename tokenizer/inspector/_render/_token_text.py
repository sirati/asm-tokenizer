"""Per-token display substitution shared by both rendering backends.

Single concern: convert a raw token text atom (either the BatchDecode
vocab-string form ``MEM_OPEN_BRACKET`` / ``REG_LIST_OPEN_BRACE`` /
``asm_writeback_detect`` or the FTL asm-value form ``mem[`` /
``reglist{``) to the polished display char (``[`` / ``]`` / ``{`` /
``}`` / ``!`` / ``+`` / ...).

Both backends emit the same enum members
(:class:`MemoryOperandSymbol` + :class:`RegisterListSymbol`), but they
emit DIFFERENT raw forms:

* :class:`BatchDecodeBackend` reaches the symbol via
  ``VocabularyManager.get_token_str(id)`` which returns the registered
  vocab string (``MEM_OPEN_BRACKET`` ... ``asm_post_index_separator`` /
  ``REG_LIST_OPEN_BRACE`` / ``asm_writeback_detect`` / ...).
* :class:`FtlBackend` reaches the symbol via
  :meth:`MemoryOperandTokenInner.to_asm_like` /
  :meth:`RegisterListTokenInner.to_asm_like` which return
  ``str(self.symbol.value)`` (``mem[``, ``]mem``, ``reglist{``,
  ``}reglist``, ``!``, ``+``, ...).

The substitution table maps BOTH source forms to the display char so a
single helper covers both backends. Asm-value forms that already match
the display char (``+`` / ``-`` / ``*`` / ``,`` / ``!``) collapse to
identity; the bracket / brace symbols get two distinct keys.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Union

from tokenizer.tokens import MemoryOperandSymbol, RegisterListSymbol


__all__ = ["DISPLAY_SUBSTITUTION", "substitute_display_chars"]


_DisplayKey = Union[MemoryOperandSymbol, RegisterListSymbol]


# Per-symbol display char. Bracket / brace symbols are the only ones
# whose asm-value differs from the polished display char; the
# operators (``+`` ``-`` ``*`` ``,`` ``!``) are identity. Mechanically
# derived from the enum members so any future enum extension fails
# loud at the tripwire below until the implementer wires a display
# char for the new member.
_DISPLAY_CHAR: Mapping[_DisplayKey, str] = {
    MemoryOperandSymbol.OPEN_BRACKET: "[",
    MemoryOperandSymbol.CLOSE_BRACKET: "]",
    MemoryOperandSymbol.PLUS: "+",
    MemoryOperandSymbol.MINUS: "-",
    MemoryOperandSymbol.MULTIPLY: "*",
    MemoryOperandSymbol.POST_INDEX_SEPARATOR: ",",
    RegisterListSymbol.OPEN_BRACE: "{",
    RegisterListSymbol.CLOSE_BRACE: "}",
    RegisterListSymbol.WRITEBACK: "!",
}


# Module-load tripwire #1: ``_DISPLAY_CHAR`` must cover every member of
# BOTH :class:`MemoryOperandSymbol` and :class:`RegisterListSymbol`.
# An enum extension that ships a new symbol without wiring its display
# char fails this assert at import time -- preventing the "fabricated
# vocab string" failure mode caught in the Wave-4 audits (H-7) before
# any caller can ship an incomplete substitution table.
_EXPECTED_SYMBOLS = set(MemoryOperandSymbol) | set(RegisterListSymbol)
_DISPLAY_COVERAGE = set(_DISPLAY_CHAR)
assert _DISPLAY_COVERAGE == _EXPECTED_SYMBOLS, (
    "_DISPLAY_CHAR missing one or more display-symbol enum members: "
    f"{_EXPECTED_SYMBOLS - _DISPLAY_COVERAGE}"
)


def _build_substitution_table() -> Mapping[str, str]:
    """Build the raw-text -> display-char lookup, covering BOTH source
    forms each backend can produce for every member of
    :class:`MemoryOperandSymbol` + :class:`RegisterListSymbol`.

    Two entries per symbol:

    * ``symbol.token_str()`` -- the registered vocab string returned by
      :meth:`VocabularyManager.get_token_str` (BatchDecode path).
    * ``str(symbol.value)`` -- the asm-value returned by
      :meth:`MemoryOperandTokenInner.to_asm_like` /
      :meth:`RegisterListTokenInner.to_asm_like` (FTL path).

    Both map to the same display char so callers can apply the same
    substitution irrespective of which backend produced the atom. The
    operator members collapse the two keys to a single identity entry
    (``"+" -> "+"``); the bracket / brace members get two distinct
    keys.
    """
    table: dict[str, str] = {}
    for symbol in (*MemoryOperandSymbol, *RegisterListSymbol):
        display = _DISPLAY_CHAR[symbol]
        table[symbol.token_str()] = display
        table[str(symbol.value)] = display
    return MappingProxyType(table)


DISPLAY_SUBSTITUTION: Mapping[str, str] = _build_substitution_table()
"""Raw text -> display char for every
:class:`MemoryOperandSymbol` + :class:`RegisterListSymbol` member,
in BOTH source forms.

Frozen via :class:`MappingProxyType` so callers cannot mutate the
shared dict.
"""


# Module-load tripwire #2: every covered enum member must contribute
# its vocab-string to the built table. Belt-and-braces vs tripwire #1:
# a refactor that drops the ``symbol.token_str()`` line in
# ``_build_substitution_table`` would silently lose BatchDecode
# coverage; this assert pins it.
_TOKEN_STR_COVERAGE = {s.token_str() for s in _EXPECTED_SYMBOLS}
assert _TOKEN_STR_COVERAGE.issubset(DISPLAY_SUBSTITUTION), (
    "DISPLAY_SUBSTITUTION missing one or more enum token_str keys: "
    f"{_TOKEN_STR_COVERAGE - set(DISPLAY_SUBSTITUTION)}"
)


def substitute_display_chars(token_str: str) -> str:
    """Substitute MEM / reg-list vocab strings + asm-values with display chars.

    Pure function. Returns the display char (``"["`` / ``"]"`` / ``"{"``
    / ``"}"`` / ``"!"`` / ``"+"`` / ...) when ``token_str`` matches any
    :class:`MemoryOperandSymbol` or :class:`RegisterListSymbol` source
    form; otherwise returns ``token_str`` unchanged.

    Backend-agnostic: callers do not need to know whether they hold a
    vocab-string (BatchDecode) or an asm-value (FTL) -- the same lookup
    covers both.
    """
    return DISPLAY_SUBSTITUTION.get(token_str, token_str)
