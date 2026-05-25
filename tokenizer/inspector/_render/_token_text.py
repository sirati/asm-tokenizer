"""Per-token display substitution shared by both rendering backends.

Single concern: convert a raw token text atom (either the BatchDecode
vocab-string form ``MEM_OPEN_BRACKET`` or the FTL asm-value form
``mem[``) to the polished display char (``[`` / ``]`` / ``+`` / ...).

Both backends emit the same six :class:`MemoryOperandSymbol` members,
but they emit DIFFERENT raw forms:

* :class:`BatchDecodeBackend` reaches the symbol via
  ``VocabularyManager.get_token_str(id)`` which returns the registered
  vocab string (``MEM_OPEN_BRACKET`` ... ``asm_post_index_separator``).
* :class:`FtlBackend` reaches the symbol via
  :meth:`MemoryOperandTokenInner.to_asm_like` which returns
  ``str(self.symbol.value)`` (``mem[``, ``]mem``, ``+``, ``-``, ``*``,
  ``,``).

The substitution table maps BOTH source forms to the display char so a
single helper covers both backends. The plus / minus / multiply /
comma asm-value forms collapse to identity (they ARE the display char
already); the override exists only for the two bracket symbols where
the asm-value (``mem[``) differs from the display char (``[``).

Plan reference: ``inspector-followup.md`` §W3-3 (W4-amended; symmetric
across both backends). W3-4 (RegisterListSymbol extension) is REVERTED
pending user answer to Q-W4-A.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from tokenizer.tokens import MemoryOperandSymbol


__all__ = ["MEM_DISPLAY_SUBSTITUTION", "substitute_mem_chars"]


# Per-symbol display char. The two bracket symbols are the only ones
# whose asm-value differs from the polished display char; the four
# operators are identity. Mechanically derived from the enum members
# so an enum extension fails loud at the tripwire below until the
# implementer wires a display char for the new member.
_DISPLAY_CHAR: Mapping[MemoryOperandSymbol, str] = {
    MemoryOperandSymbol.OPEN_BRACKET: "[",
    MemoryOperandSymbol.CLOSE_BRACKET: "]",
    MemoryOperandSymbol.PLUS: "+",
    MemoryOperandSymbol.MINUS: "-",
    MemoryOperandSymbol.MULTIPLY: "*",
    MemoryOperandSymbol.POST_INDEX_SEPARATOR: ",",
}


# Module-load tripwire #1: ``_DISPLAY_CHAR`` must cover every
# :class:`MemoryOperandSymbol` member. An enum extension that ships a
# new symbol without wiring its display char fails this assert at
# import time -- preventing the "fabricated vocab string" failure mode
# caught in the Wave-4 audits (H-7) before any caller can ship an
# incomplete substitution table.
_DISPLAY_COVERAGE = set(_DISPLAY_CHAR)
assert _DISPLAY_COVERAGE == set(MemoryOperandSymbol), (
    "_DISPLAY_CHAR missing one or more MemoryOperandSymbol members: "
    f"{set(MemoryOperandSymbol) - _DISPLAY_COVERAGE}"
)


def _build_substitution_table() -> Mapping[str, str]:
    """Build the raw-text -> display-char lookup, covering BOTH source
    forms each backend can produce for every
    :class:`MemoryOperandSymbol` member.

    Two entries per symbol:

    * ``symbol.token_str()`` -- the registered vocab string returned by
      :meth:`VocabularyManager.get_token_str` (BatchDecode path).
    * ``str(symbol.value)`` -- the asm-value returned by
      :meth:`MemoryOperandTokenInner.to_asm_like` (FTL path).

    Both map to the same display char so callers can apply the same
    substitution irrespective of which backend produced the atom. The
    operator members collapse the two keys to a single identity entry
    (``"+" -> "+"``); the bracket members get two distinct keys.
    """
    table: dict[str, str] = {}
    for symbol in MemoryOperandSymbol:
        display = _DISPLAY_CHAR[symbol]
        table[symbol.token_str()] = display
        table[str(symbol.value)] = display
    return MappingProxyType(table)


MEM_DISPLAY_SUBSTITUTION: Mapping[str, str] = _build_substitution_table()
"""Raw text -> display char for every
:class:`MemoryOperandSymbol` member, in BOTH source forms.

Frozen via :class:`MappingProxyType` so callers cannot mutate the
shared dict.
"""


# Module-load tripwire #2: every MemoryOperandSymbol member must
# contribute its vocab-string to the built table. Belt-and-braces vs
# tripwire #1: a refactor that drops the ``symbol.token_str()`` line in
# ``_build_substitution_table`` would silently lose BatchDecode
# coverage; this assert pins it.
_TOKEN_STR_COVERAGE = {s.token_str() for s in MemoryOperandSymbol}
assert _TOKEN_STR_COVERAGE.issubset(MEM_DISPLAY_SUBSTITUTION), (
    "MEM_DISPLAY_SUBSTITUTION missing one or more MemoryOperandSymbol "
    f"token_str keys: {_TOKEN_STR_COVERAGE - set(MEM_DISPLAY_SUBSTITUTION)}"
)


def substitute_mem_chars(token_str: str) -> str:
    """Substitute MEM-operand vocab strings / asm-values with display chars.

    Pure function. Returns the display char (``"["`` / ``"]"`` / ``"+"``
    / ...) when ``token_str`` matches any :class:`MemoryOperandSymbol`
    source form; otherwise returns ``token_str`` unchanged.

    Backend-agnostic: callers do not need to know whether they hold a
    vocab-string (BatchDecode) or an asm-value (FTL) -- the same lookup
    covers both.
    """
    return MEM_DISPLAY_SUBSTITUTION.get(token_str, token_str)
