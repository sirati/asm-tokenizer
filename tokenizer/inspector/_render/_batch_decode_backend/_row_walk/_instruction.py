"""Per-instruction text assembly for the BatchDecodeBackend row walker.

Single concern: assemble one instruction's atom stream into a single
display text via bracket-aware joining (W3-11 W4-AMENDED). The R2c
per-instruction collector pre-paving lives here too (the
``_start_new_instruction`` / ``_consume_*_slot`` / ``_finalize_instruction``
quartet will land alongside the :class:`_NumberAccumulator` in R2c);
for R2a this module owns the standalone joiner only.

Plan reference: ``inspector-followup.md`` W3-11 W4-AMENDED + cluster #5
(subpackage split) + A-L2 H2 (typed emit policy on :mod:`._state`).
"""

from __future__ import annotations


__all__ = ["_join_instruction_text"]


# W4-CORRECTED: original W3-11 frozenset had a matching-bracket
# SYNTAX ERROR (``[..."})``) + included phantom ``(`` / ``)`` atoms
# that no MemoryOperandSymbol value produces today (A-L5 H3). Kept
# tight to the actual atom set produced by
# ``_MEM_DISPLAY_SUBSTITUTION`` + the per-token text renderer.
_NO_SPACE_BEFORE: frozenset[str] = frozenset({",", "]"})
_NO_SPACE_AFTER: frozenset[str] = frozenset({"["})


def _join_instruction_text(atoms: list[str]) -> str:
    """Join per-token atoms into one instruction text with bracket-
    aware spacing.

    Spacing rules (W3-11): no space between ``[`` and the next atom;
    no space between an atom and ``]`` / ``,``; otherwise a single
    space separator. Leading position behaves like a no-space-after
    boundary so the first atom never emits a leading space.
    """
    out: list[str] = []
    prev_no_space_after = True  # leading position
    for atom in atoms:
        if atom in _NO_SPACE_BEFORE:
            out.append(atom)
        elif prev_no_space_after:
            out.append(atom)
        else:
            out.append(" ")
            out.append(atom)
        prev_no_space_after = atom in _NO_SPACE_AFTER
    return "".join(out)
