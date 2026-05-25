"""Per-col loop driver for the BatchDecodeBackend row walker.

Single concern: walk one row's tokens column-by-column, dispatching
each shifted-id to its band-specific emit path AND driving the
per-instruction collector boundaries (W3-16 W4-AMENDED + W3-17
W4-AMENDED). Section transitions (VARIANT_HEADER -> FUNCTION_ID,
FUNCTION_ID -> BODY) and the runlength-derived ``pending_header``
latch live here; the per-band emit primitives are in
:mod:`.._band_emitters` and the IDENTITY dispatch is in
:mod:`._dispatch`. The per-instruction collector quartet
(:mod:`._instruction`) is invoked at instruction boundaries so the
section's :attr:`WalkSectionState.current_items` accumulates one
:class:`AsmLine` per instruction, not one per slot. Plan reference:
``inspector-render-backends.md`` §6 + decisions #16/#17/#18/#29/#30;
``inspector-followup.md`` W3-16 + W3-17 + cluster A-L1 H2.
"""

from __future__ import annotations

from typing import Callable, List, Mapping, Optional, Sequence

from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
    SectionPointerSpec,
)
from tokenizer.aligned_data.matched_sections_bin import (
    CallTarget,
)
from tokenizer.inspector._render._protocol import (
    BlockKind,
)
from tokenizer.token_manager import VocabularyManager

from .._band import Band, classify_shifted_id
from .._band_emitters import emit_instr_rep, emit_number
from .._fid_table import FidBaseTable
from .._sections import (
    RowSection,
    close_current_section,
    enter_variant_header_to_function_id,
    maybe_advance_call_target,
)
from ._dispatch import _emit_identity
from ._instruction import (
    _finalize_instruction,
    _flush_accumulator_into_current_insn,
    _start_new_instruction,
)
from ._state import _RowSidecars, _WalkState, _init_walk_state


__all__ = ["render_row_blocks"]


def _maybe_open_function_id_section(state: _WalkState, *, col: int) -> None:
    """VARIANT_HEADER -> FUNCTION_ID at the root CT start col.

    Finalizes the in-flight VARIANT_HEADER instruction (if any) so its
    AsmLine lands in the VARIANT_HEADER section before the section
    closes; subsequent FUNCTION_ID -> BODY transitions are token-content-
    driven; see :func:`.._sections.enter_body_after_function_id`.
    """
    if (
        not state.ct_start_cols
        or col != state.ct_start_cols[0]
        or state.current_kind is not BlockKind.VARIANT_HEADER
    ):
        return
    _finalize_instruction(state)
    enter_variant_header_to_function_id(state)


def _maybe_latch_header_trigger(
    state: _WalkState, sidecars: _RowSidecars, *, col: int,
) -> None:
    """Latch ``pending_header`` at runlength-derived trigger cols.

    The BODY-block open is deferred to :func:`._dispatch._handle_block`
    when the upcoming BLOCK_V2 consumes the latch (legacy "no Block_V2
    = no new block" rule -- jump-table footer blocks stay folded).
    Skipped while in VARIANT_HEADER / FUNCTION_ID; those own their own
    ``pending_header`` handling. A fresh latch also clears
    :attr:`WalkSectionState.inside_jump_table_footer_block`: a new
    header trigger means "new block ahead", which is incompatible with
    "still inside footer".

    The in-flight instruction is also finalized here: a header trigger
    marks the start of a new BODY block, so any prior in-flight
    instruction in the same (or prior) block MUST finalize before the
    next ``_start_new_instruction`` re-latches
    ``current_insn_in_silent_header`` from the freshly-set
    ``pending_header``.
    """
    if col not in state.header_trigger_cols:
        return
    if state.current_kind in (BlockKind.VARIANT_HEADER, BlockKind.FUNCTION_ID):
        return
    # Finalize any in-flight instruction before flipping the latch so
    # the prior instruction's policy is decided on the OLD pending_header.
    _finalize_instruction(state)
    state.pending_header = True
    state.inside_jump_table_footer_block = False


def _maybe_flush_accumulator_at_boundary(
    state: _WalkState, *, upcoming_shifted_id: int, upcoming_band: Band,
) -> None:
    """Pre-finalize drain at legitimate source-completion boundaries.

    Runs at the TOP of each per-col iteration, BEFORE any finalize-
    causing logic (section transitions, header-trigger latch,
    slot-budget boundary). The accumulator's pending source has
    completed when EITHER:

    * the upcoming token is NOT in the NUMBER band, OR
    * the upcoming token IS a NUMBER token whose shifted id differs
      from the pending source's shifted id (the encoder switched chunk
      types, so the prior source ended).

    In both cases the drain lands in the IN-FLIGHT instruction's text
    + openables buffers (the instruction that consumed the pending
    source's chunks), so the subsequent finalize sees an empty
    accumulator and does not falsely flag the boundary as a W3-17
    invariant violation. Genuine spanning (same shifted id NUMBER
    crossing an instruction boundary) leaves the accumulator pending;
    :func:`._instruction._drain_accumulator_into_buffer`'s W3-17 check
    then fires inside finalize.

    No-op when the accumulator is empty.
    """
    if not state.number_accumulator.has_pending():
        return
    if upcoming_band is Band.NUMBER:
        pending_id = state.number_accumulator.pending_shifted_id()
        if upcoming_shifted_id == pending_id:
            # Same source candidate; let emit_number / finalize decide.
            return
    _flush_accumulator_into_current_insn(state)


def _maybe_finalize_and_start_instruction(
    state: _WalkState, sidecars: _RowSidecars,
) -> None:
    """Finalize the in-flight instruction (if any) and start a new one.

    Triggered at every per-col loop iteration BEFORE the slot's emit:
    if the current instruction's slot budget is exhausted (or no
    instruction is active yet), finalize-then-start so the next slot
    lands on a fresh per-instruction buffer.
    """
    if state.has_active_instruction and state.slots_remaining_in_insn > 0:
        return
    _finalize_instruction(state)
    _start_new_instruction(state, sidecars)


def render_row_blocks(
    *,
    result: BatchDecodeResult,
    row: int,
    n_axis: int,
    partial_cut_lengths: list[int],
    call_targets_per_ct: Sequence[Sequence[CallTarget]],
    vocab_manager: VocabularyManager,
    fid_table: FidBaseTable,
    line_to_name: Mapping[int, str],
    line_to_provider: Mapping[int, str],
    callee_arm_resolver: Callable[[int], Optional[SectionPointerSpec]],
    arch_prefixes: tuple[str, ...] = (),
) -> List[RowSection]:
    """Walk one row and split into typed sections.

    Section flow: VARIANT_HEADER (cols ``[0, n_axis)``) -> FUNCTION_ID
    (root CT start col) -> BODY blocks (each runlength-derived header
    trigger col; ``Block_Def`` + ``block_v2`` header pair consumed
    silently). Terminates on ``token == 0`` or end-of-row.

    Per-instruction emit discipline (W3-16 W4-AMENDED + cluster A-L1
    H2): the band emit + IDENTITY dispatch paths route their atoms +
    openables onto an in-flight instruction's text-parts / openables
    buffers (on :class:`_WalkState`); the per-col loop finalizes the
    in-flight instruction at its slot-budget boundary and starts a
    fresh one. One :class:`AsmLine` per instruction, not per slot;
    inline call / jump / number-precision sidecars ride on
    :attr:`AsmLine.openables`.

    ``call_targets_per_ct`` carries each Stage1CallTarget's own
    ``call_targets_section`` so inlined-callee call sites resolve
    against THEIR table; ``callee_arm_resolver`` returns ``None`` for
    cross-arm/missing; EXTERN call_targets always carry ``None``.
    """
    state, sidecars = _init_walk_state(
        result=result, row=row, n_axis=n_axis,
        partial_cut_lengths=partial_cut_lengths,
        call_targets_per_ct=call_targets_per_ct,
    )

    n_cols = int(sidecars.tokens_row.shape[0])
    for col in range(n_cols):
        state.current_col = col
        shifted_id = int(sidecars.tokens_row[col])
        if shifted_id == 0:
            break  # null-content padding tail
        band = classify_shifted_id(shifted_id)
        # Pre-finalize drain at legitimate source-completion boundaries.
        # MUST run before any finalize-causing logic (section transitions,
        # header-trigger latch, slot-budget boundary) so the in-flight
        # instruction's NUMBER-source chunks land in its OWN buffer rather
        # than tripping the W3-17 encoder-invariant assert at finalize.
        _maybe_flush_accumulator_at_boundary(
            state, upcoming_shifted_id=shifted_id, upcoming_band=band,
        )
        _maybe_open_function_id_section(state, col=col)
        _maybe_latch_header_trigger(state, sidecars, col=col)
        maybe_advance_call_target(state, col=col)
        _maybe_finalize_and_start_instruction(state, sidecars)
        if band is Band.INSTR_REP:
            if state.pending_header:
                # ``Block_Def`` carrier -- absorbed silently so the
                # BODY section's first emitted item is the first
                # real instruction. The per-instruction finalize
                # under SILENT_HEADER policy asserts the text buffer
                # stayed empty across this slot.
                pass
            else:
                emit_instr_rep(
                    state,
                    shifted_id=shifted_id,
                    vocab_manager=vocab_manager,
                    arch_prefixes=arch_prefixes,
                )
        elif band is Band.NUMBER:
            emit_number(
                state,
                shifted_id=shifted_id,
                numbers_sig=sidecars.numbers_sig,
                numbers_se=sidecars.numbers_se,
            )
        elif band is Band.IDENTITY:
            _emit_identity(
                state,
                shifted_id=shifted_id,
                identities_row=sidecars.identities_row,
                fid_table=fid_table,
                line_to_name=line_to_name,
                line_to_provider=line_to_provider,
                call_targets_per_ct=call_targets_per_ct,
                kind_to_called_idx_per_ct=sidecars.kind_to_called_idx_per_ct,
                callee_arm_resolver=callee_arm_resolver,
            )
        else:
            raise ValueError(
                f"render_row_blocks: unexpected Band {band!r} at "
                f"row={row}, col={col} (shifted_id={shifted_id})"
            )
        # Consumed one slot in the in-flight instruction.
        state.slots_remaining_in_insn -= 1

    # End-of-row finalize per cluster #21 H-4: any in-flight instruction
    # (including a cut-variant partial) MUST emit best-effort so its
    # surviving slot's content is not silently dropped.
    _finalize_instruction(state, end_of_row=True)
    close_current_section(state)
    return state.completed
