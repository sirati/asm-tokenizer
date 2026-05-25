"""Tests for the shared per-token text substitution helper.

Pins three invariants:

* :data:`MEM_DISPLAY_SUBSTITUTION` covers every
  :class:`MemoryOperandSymbol` member in BOTH its source forms
  (vocab-string ``MEM_OPEN_BRACKET`` AND asm-value ``mem[``).
* :func:`substitute_mem_chars` returns the polished display char for
  every source form and passes through unknown atoms unchanged.
* The module-load tripwire fires when the underlying dict drops a
  symbol -- the "fabricated vocab string" failure mode Wave-4 audit
  H-7 baked into R1b.

Plan reference: ``inspector-followup.md`` §W3-3 (W4-amended; symmetric
across both backends).
"""

from __future__ import annotations

import importlib
import sys

import pytest

from tokenizer.inspector._render import _token_text
from tokenizer.inspector._render._token_text import (
    MEM_DISPLAY_SUBSTITUTION,
    substitute_mem_chars,
)
from tokenizer.tokens import MemoryOperandSymbol


# Locked vocab-string -> display-char pairs (mechanically derived from
# the MemoryOperandSymbol enum at module load; pinning the expected
# strings here protects against a silent enum rename or value drift).
_EXPECTED_PAIRS: tuple[tuple[MemoryOperandSymbol, str, str, str], ...] = (
    (MemoryOperandSymbol.OPEN_BRACKET, "MEM_OPEN_BRACKET", "mem[", "["),
    (MemoryOperandSymbol.CLOSE_BRACKET, "MEM_CLOSE_BRACKET", "]mem", "]"),
    (MemoryOperandSymbol.PLUS, "MEM_PLUS", "+", "+"),
    (MemoryOperandSymbol.MINUS, "MEM_MINUS", "-", "-"),
    (MemoryOperandSymbol.MULTIPLY, "MEM_MULTIPLY", "*", "*"),
    (
        MemoryOperandSymbol.POST_INDEX_SEPARATOR,
        "asm_post_index_separator",
        ",",
        ",",
    ),
)


@pytest.mark.parametrize("symbol,vocab_str,asm_value,display", _EXPECTED_PAIRS)
def test_substitute_covers_every_memory_operand_symbol_vocab_form(
    symbol: MemoryOperandSymbol,
    vocab_str: str,
    asm_value: str,
    display: str,
) -> None:
    """Both the vocab-string AND the asm-value substitute to the polished
    display char. Backends MAY emit either form (BatchDecode emits the
    vocab string; FTL emits the asm-value via ``Inner.to_asm_like``)."""
    assert symbol.token_str() == vocab_str
    assert str(symbol.value) == asm_value
    assert substitute_mem_chars(vocab_str) == display
    assert substitute_mem_chars(asm_value) == display


def test_substitute_passes_unknown_atoms_through() -> None:
    """Non-MEM atoms (instruction mnemonics, registers, immediates,
    ``v2:HEX`` etc.) MUST pass through unchanged. The substitution is
    purely additive on the six MEM-symbol source forms."""
    for atom in ("mov", "rax", "0xdeadbeef", "v2:42", "block_v2:7", ""):
        assert substitute_mem_chars(atom) == atom


def test_substitution_table_covers_every_symbol() -> None:
    """The shared dict must include every
    :class:`MemoryOperandSymbol`'s :meth:`token_str` AND
    :attr:`value`. This is the same invariant as the module-load
    tripwire; pinning it here keeps the failure mode visible as a
    test (not just an import-time assert) if a future refactor moves
    the table out of module scope."""
    for symbol in MemoryOperandSymbol:
        assert symbol.token_str() in MEM_DISPLAY_SUBSTITUTION
        assert str(symbol.value) in MEM_DISPLAY_SUBSTITUTION


def test_module_load_tripwire_fires_on_dropped_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch :data:`MemoryOperandSymbol` so a new member appears
    AFTER the module is reloaded but the substitution table's
    ``_DISPLAY_CHAR`` does NOT cover it. The module reload MUST raise
    :class:`AssertionError` from the tripwire.

    This protects against the "fabricated vocab string" failure mode
    (Wave-4 audit H-7): if a future enum extension adds a new symbol
    without wiring its display char + tripwire coverage, the import
    fails loudly instead of silently shipping an incomplete dict.
    """
    import enum

    # Build a stand-in enum with one EXTRA member the table won't cover.
    class _StubMemorySymbol(enum.Enum):
        OPEN_BRACKET = "mem["
        CLOSE_BRACKET = "]mem"
        PLUS = "+"
        MINUS = "-"
        MULTIPLY = "*"
        POST_INDEX_SEPARATOR = ","
        # Extra member: not in _DISPLAY_CHAR -- tripwire MUST fire.
        UNKNOWN_NEW_SYMBOL = "??"

        def token_str(self) -> str:
            return self.name

    # Re-import the module with the patched symbol set.
    monkeypatch.setattr(
        "tokenizer.tokens.MemoryOperandSymbol", _StubMemorySymbol
    )
    sys.modules.pop(_token_text.__name__, None)
    with pytest.raises(AssertionError):
        importlib.import_module(_token_text.__name__)
    # Restore the real module for downstream tests in the same process.
    sys.modules.pop(_token_text.__name__, None)
    monkeypatch.undo()
    importlib.import_module(_token_text.__name__)
