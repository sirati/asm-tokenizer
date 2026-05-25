"""Per-instruction collector for the BatchDecodeBackend row walker.

Single concern: assemble one instruction's atom stream into a single
display text + sidecar-openables tuple via the quartet
``_start_new_instruction`` / ``_consume_text_slot`` /
``_consume_openable_slot`` / ``_finalize_instruction`` plus the
bracket-aware text joiner :func:`_join_instruction_text` (W3-11
W4-AMENDED).

The driver (:mod:`._driver`) calls :func:`_start_new_instruction` at
each runlength-derived instruction-start col and
:func:`_finalize_instruction` at the next start col (or end-of-row);
the band-emit + IDENTITY-dispatch paths route their atoms into the
in-flight instruction's text-parts / openables buffers via the
``_consume_*_slot`` helpers instead of directly mutating
``state.current_items``. :func:`_finalize_instruction` is the single
emit site (cluster A-L1 H2): one :class:`AsmLine` per finalize, or
zero under :attr:`_InsnEmitPolicy.SILENT_HEADER`.

Plan reference: ``inspector-followup.md`` W3-11 W4-AMENDED + W3-16
W4-AMENDED + W3-17 W4-AMENDED + cluster #5 (subpackage split) +
A-L2 H2 (typed emit policy on :mod:`._state`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tokenizer.inspector._render._protocol import AsmLine, Openable

from ._state import _InsnEmitPolicy, _WalkState


if TYPE_CHECKING:
    from ._state import _RowSidecars


__all__ = [
    "_consume_openable_slot",
    "_consume_text_slot",
    "_finalize_instruction",
    "_join_instruction_text",
    "_start_new_instruction",
]


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


def _slot_count_for_next_instruction(
    state: _WalkState, sidecars: "_RowSidecars"
) -> int:
    """Slot count for the next instruction, by section policy.

    VARIANT_HEADER + FUNCTION_ID sections size every instruction at 1
    slot (each variant-prefix INSTR_REP and the LOCAL_FUNC self-prepend
    are atomic 1-slot instructions). BODY sections consume the next
    entry from :attr:`_RowSidecars.insn_runlength_row` and advance
    :attr:`_WalkState.insn_cursor`; when the sidecar is exhausted
    (test fixtures omit it; production cut variants may also exhaust)
    the fallback is a single slot.
    """
    from tokenizer.inspector._render._protocol import BlockKind

    if state.current_kind is not BlockKind.BODY:
        return 1
    if state.insn_cursor >= int(sidecars.insn_runlength_row.size):
        return 1
    slots = int(sidecars.insn_runlength_row[state.insn_cursor])
    state.insn_cursor += 1
    # Defensive: an emit_block_n_insns_runlength contract violation
    # (slot==0) would mean the BODY section never finalizes; fall back
    # to 1 so the driver still makes progress.
    return slots if slots > 0 else 1


def _start_new_instruction(
    state: _WalkState, sidecars: "_RowSidecars"
) -> None:
    """Reset the per-instruction buffers for the next instruction.

    Latches :attr:`_WalkState.current_insn_in_silent_header` from the
    current :attr:`WalkSectionState.pending_header` so the finalize
    knows whether to emit (real / footer) or skip (silent header) the
    AsmLine. Resets :attr:`_WalkState.saw_jump_table_this_insn` so the
    per-instruction JUMP_TABLE flag starts clean (the per-block
    :attr:`WalkSectionState.inside_jump_table_footer_block` flag is
    section-scoped, not per-instruction).

    Sets :attr:`_WalkState.slots_remaining_in_insn` to the section's
    instruction size (see :func:`_slot_count_for_next_instruction`),
    consuming the next ``insn_runlength_row`` entry when in BODY.
    """
    state.current_insn_text_parts = []
    state.current_insn_openables = []
    state.current_insn_in_silent_header = state.pending_header
    state.saw_jump_table_this_insn = False
    state.slots_remaining_in_insn = _slot_count_for_next_instruction(
        state, sidecars
    )
    state.has_active_instruction = True


def _consume_text_slot(state: _WalkState, *, text: str) -> None:
    """Append one rendered text atom to the in-flight instruction.

    The leading ``Block_Def`` INSTR_REP of a silent-header pair is
    routed here while ``pending_header=True``; under
    :attr:`_InsnEmitPolicy.SILENT_HEADER` the finalize asserts the
    text buffer is empty, so callers MUST gate ``Block_Def``
    suppression at the dispatch level (the driver's INSTR_REP branch
    does this today via ``if state.pending_header: continue``). This
    helper does NOT inspect ``pending_header`` -- it is a dumb sink
    that lets the caller choose what gets buffered.
    """
    state.current_insn_text_parts.append(text)


def _consume_openable_slot(
    state: _WalkState,
    *,
    openable: Openable,
    placeholder_text: Optional[str] = None,
) -> None:
    """Append one :data:`Openable` (plus optional placeholder text) to
    the in-flight instruction.

    ``placeholder_text`` is the legacy ``<category counter>`` /
    canonical ``inline_call_label`` / ``inline_jump_label`` text that
    USED to ride as a sibling :class:`AsmLine` and now folds into the
    owning instruction's text buffer. Pass ``None`` when the openable's
    presence on the AsmLine is enough (e.g. a number-precision entry
    whose short_text was already appended via :func:`_consume_text_slot`).
    """
    state.current_insn_openables.append(openable)
    if placeholder_text is not None:
        state.current_insn_text_parts.append(placeholder_text)


def _finalize_instruction(
    state: _WalkState, *, end_of_row: bool = False
) -> None:
    """Emit one :class:`AsmLine` (or zero, under SILENT_HEADER) into
    :attr:`WalkSectionState.current_items`.

    W3-16 W4-AMENDED policy resolution:

    * ``current_insn_in_silent_header AND NOT saw_jump_table_this_insn``
      -> :attr:`_InsnEmitPolicy.SILENT_HEADER`: assert the text /
      openables buffers are empty (the ``Block_Def`` carrier was
      suppressed by the driver and the ``block_v2:N`` IDENTITY was
      consumed as a header by :func:`_handle_block`); emit nothing.
    * ``current_insn_in_silent_header AND saw_jump_table_this_insn``
      -> :attr:`_InsnEmitPolicy.JUMP_TABLE_FOOTER`: the instruction
      ATE the silent-header latch BUT the JUMP_TABLE IDENTITY proves
      this is the footer instruction, not a silent header. Emit a
      real AsmLine with the buffered text + openables (trailing
      BLOCK_V2 targets as :class:`InlineJumpEntry` openables).
    * Otherwise -> :attr:`_InsnEmitPolicy.REAL`: emit a real AsmLine.

    W3-17 W4-AMENDED encoder-invariant assert: at mid-row instruction
    boundaries the :class:`_NumberAccumulator` MUST be empty (a
    multi-chunk source cannot span instructions). At end-of-row the
    accumulator MAY have pending chunks from a cut-variant tail; the
    assert is skipped and the pending source is best-effort flushed.

    Idempotent if no instruction is active (no-op when
    :attr:`_WalkState.has_active_instruction` is False) so
    :func:`close_current_section` can defensively force a finalize
    at section boundaries without tracking active-state itself.
    """
    if not state.has_active_instruction:
        return
    # W3-17 encoder-invariant + cut-variant tolerance: flush + assert.
    _drain_accumulator_into_buffer(state, end_of_row=end_of_row)
    # Policy decision (A-L2 H2: typed enum replaces boolean cross-prod).
    if state.current_insn_in_silent_header and not state.saw_jump_table_this_insn:
        state.insn_emit_policy = _InsnEmitPolicy.SILENT_HEADER
        assert not state.current_insn_text_parts, (
            f"silent header had text emissions: "
            f"{state.current_insn_text_parts!r}"
        )
        assert not state.current_insn_openables, (
            f"silent header carried openables: "
            f"{state.current_insn_openables!r}"
        )
    elif state.current_insn_in_silent_header and state.saw_jump_table_this_insn:
        # Latched as silent but JUMP_TABLE fired -> footer instruction.
        state.insn_emit_policy = _InsnEmitPolicy.JUMP_TABLE_FOOTER
        _emit_one_asmline(state)
    else:
        state.insn_emit_policy = _InsnEmitPolicy.REAL
        _emit_one_asmline(state)
    # Clear per-instruction state regardless of emit policy.
    state.current_insn_text_parts = []
    state.current_insn_openables = []
    state.current_insn_in_silent_header = False
    state.saw_jump_table_this_insn = False
    state.slots_remaining_in_insn = 0
    state.has_active_instruction = False


def _drain_accumulator_into_buffer(
    state: _WalkState, *, end_of_row: bool
) -> None:
    """Flush :attr:`_WalkState.number_accumulator` into the in-flight
    instruction's text + openables buffers.

    W3-17 W4-AMENDED: at mid-row instruction boundaries the accumulator
    MUST be empty (encoder invariant: multi-chunk sources never span
    instructions). At end-of-row a cut-variant tail MAY leave pending
    chunks; best-effort flush their lead-chunk contribution.
    """
    if state.number_accumulator.has_pending() and not end_of_row:
        # Drain into the buffer first so the assert message can quote
        # the offending source's rendered short form for diagnosis.
        emission = state.number_accumulator.flush()
        if emission is not None:
            state.current_insn_text_parts.append(emission.short_text)
            if emission.precision_entry is not None:
                state.current_insn_openables.append(emission.precision_entry)
        raise AssertionError(
            "encoder invariant: multi-chunk NUMBER source spanned "
            f"instruction boundary at row={state.row}, "
            f"insn_cursor={state.insn_cursor}"
        )
    emission = state.number_accumulator.flush()
    if emission is None:
        return
    state.current_insn_text_parts.append(emission.short_text)
    if emission.precision_entry is not None:
        state.current_insn_openables.append(emission.precision_entry)


def _emit_one_asmline(state: _WalkState) -> None:
    """Append one :class:`AsmLine` to :attr:`WalkSectionState.current_items`.

    Joined via :func:`_join_instruction_text`; openables tuple captures
    the per-instruction sidecar list. Empty text + empty openables
    still emit an AsmLine (defensive: a non-silent-header instruction
    that somehow produced no atoms is a wire-format anomaly worth
    surfacing as a visibly-empty row rather than silently dropping).
    """
    text = _join_instruction_text(state.current_insn_text_parts)
    openables = tuple(state.current_insn_openables)
    state.current_items.append(AsmLine(text=text, openables=openables))
