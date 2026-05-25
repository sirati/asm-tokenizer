"""Per-row band-emit walker for the BatchDecodeBackend.

Single concern: translate one row of :attr:`BatchDecodeResult.tokens`
+ its aligned sidecars into ``[RowSection, ...]`` -- variant header,
LOCAL_FUNC self-prepend (Function ID), and per-basic-block BODY
sections with the ``Block_Def`` + ``block_v2`` header pair consumed
silently (the parent tree row's label already encodes the block
index). Cursors ``id_cursor`` + ``num_cursor`` track per-row sidecar
positions. The section state machine (open / close / per-col
transitions) is factored into :mod:`._sections` so this module owns
only the per-band emission concern. Plan reference:
``inspector-render-backends.md`` §6 + decisions #16/#17/#18/#29/#30.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional, Sequence

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    FUNCTION_CATEGORIES,
    _SHIFTED_ID_TO_CATEGORY,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
    SectionPointerSpec,
)
from tokenizer.aligned_data.loader.decoded._number_render_collector import (
    _NumberAccumulator,
)
from tokenizer.aligned_data.matched_sections_bin import (
    MISSING_VARIANT_INDEX,
    CallTarget,
)
from tokenizer.inspector._render._protocol import (
    AsmLine,
    BlockKind,
    InlineCallEntry,
    InlineJumpEntry,
    LineItem,
)
from tokenizer.inspector._render._render_block import (
    partition_call_target_kinds,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category

from ._band import Band, classify_shifted_id
from ._band_emitters import emit_instr_rep, emit_number, flush_accumulator_into
from ._boundaries import call_target_starts, header_trigger_cols
from ._callee_resolver import resolve_callee_pointer
from ._fid_table import FidBaseTable
from ._sections import (
    RowSection,
    WalkSectionState,
    close_current_section,
    enter_body_after_function_id,
    enter_new_body_block,
    enter_variant_header_to_function_id,
    maybe_advance_call_target,
    set_current_body_block_idx,
)


__all__ = ["RowSection", "render_row_blocks"]


# Per-Category -> CallTargetType for FUNCTION-band identity tokens.
# Mirrors ``_dedup_walk._constants._CALL_TARGET_TYPE_TO_CATEGORY`` in
# reverse so we never invert a dict at the call site.
_CATEGORY_TO_CALL_TARGET_TYPE: dict[Category, CallTargetType] = {
    Category.LOCAL_FUNC: CallTargetType.LOCAL,
    Category.PLT_FUNC: CallTargetType.PLT,
    Category.EXT_FUNC: CallTargetType.EXTERN,
}
assert set(_CATEGORY_TO_CALL_TARGET_TYPE) == set(FUNCTION_CATEGORIES), (
    "FUNCTION_CATEGORIES must match the CallTargetType reverse map"
)


@dataclass
class _WalkState(WalkSectionState):
    """Per-row walk state: composes :class:`WalkSectionState` (the
    section concern; see :mod:`._sections`) with the band-emitter
    cursors that the per-col loop reads + updates.

    ``id_cursor`` / ``num_cursor`` track per-row sidecar positions.
    ``current_col`` is the column index the loop is processing;
    crosses :func:`_handle_block`'s ``n_axis`` invariant check.
    ``number_accumulator`` is the per-row :class:`_NumberAccumulator`
    instance that groups multi-chunk NUMBER sources (W3-17). A band
    switch off NUMBER -- an :class:`Band.INSTR_REP` or
    :class:`Band.IDENTITY` token -- flushes pending chunks into the
    current section; an end-of-row break also force-flushes per
    cluster #21 H-4 cut-variant tolerance.
    """

    row: int = 0
    n_axis: int = 0
    id_cursor: int = 0
    num_cursor: int = 0
    current_col: int = 0
    number_accumulator: _NumberAccumulator = field(
        default_factory=_NumberAccumulator
    )


def _handle_block(state: _WalkState, *, counter: int) -> None:
    """BLOCK_V2 IDENTITY dispatch: header vs inline jump.

    With ``pending_header=True`` (latched at runlength-derived
    trigger cols) the BLOCK_V2 is a body-block header: it flushes
    any prior BODY content and opens a fresh BODY block whose
    ``block_idx`` is the encoded ``counter`` (the BLOCK identity
    index, matching the InlineJumpEntry target scheme). BLOCK_V2
    tokens with ``pending_header=False`` are in-function jumps. If
    the latch fires while still in FUNCTION_ID (bare-BLOCK_V2 test
    layouts), the FUNCTION_ID -> BODY transition runs first. A
    BLOCK_V2 inside ``[0, n_axis)`` is raised loud (the
    variant_tokens prefix is pure INSTR_REP by row-writer contract).
    """
    if state.current_col < state.n_axis:
        raise AssertionError(
            f"BLOCK_V2 IDENTITY token at col={state.current_col} lies "
            f"inside the variant_tokens prefix (n_axis={state.n_axis}); "
            f"the prefix is pure instruction-rep by row-writer contract."
        )
    if state.current_kind is BlockKind.FUNCTION_ID:
        # Bare-BLOCK_V2 test layout: no self-prepend slot exists. The
        # FUNCTION_ID section stays empty; transition to BODY block
        # before consuming the header.
        enter_body_after_function_id(state)
    if state.pending_header:
        if state.current_items:
            # Flush prior BODY block (e.g., the previous basic block
            # before this header) and open a fresh one. The first
            # BLOCK_V2 after FUNCTION_ID -> BODY transition has an
            # empty section so this branch is skipped.
            enter_new_body_block(state)
        set_current_body_block_idx(state, block_idx=counter)
        state.pending_header = False
        return
    state.current_items.append(InlineJumpEntry(target_block_idx=counter))


def _handle_function_category(
    state: _WalkState,
    *,
    cat: Category,
    counter: int,
    fid_table: FidBaseTable,
    line_to_name: Mapping[int, str],
    line_to_provider: Mapping[int, str],
    call_targets_per_ct: Sequence[Sequence[CallTarget]],
    kind_to_called_idx_per_ct: Sequence[Mapping[CallTargetType, list[int]]],
    callee_arm_resolver: Callable[[int], Optional[SectionPointerSpec]],
) -> None:
    """FUNCTION-Category dispatch: FID lookup + InlineCallEntry.

    LOCAL / PLT call_targets resolve their ``function_section_ptr``
    via the caller-supplied ``callee_arm_resolver`` closure; EXT
    call_targets keep ``callee_section_pointer=None``. The LOCAL/PLT
    path consumes the CURRENT call-target's ``call_targets_section``
    so inlined-callee call sites resolve against THEIR table. The
    root CT's LOCAL_FUNC self-prepend (counter 0) emits into the
    FUNCTION_ID section; its callee pointer cycles back to the same
    function so the row is expandable into the function's own
    variants (a useful self-reference for the inspector).
    """
    call_kind = _CATEGORY_TO_CALL_TARGET_TYPE[cat]
    fid = fid_table.lookup(row=state.row, cat=cat, counter=counter)
    callee_name = line_to_name.get(fid, "?")
    provider = (
        line_to_provider.get(fid) if call_kind is CallTargetType.EXTERN else None
    )
    callee_section_pointer = resolve_callee_pointer(
        call_kind=call_kind,
        counter=counter,
        call_targets_section=call_targets_per_ct[state.current_call_target_idx],
        kind_to_called_idx=kind_to_called_idx_per_ct[
            state.current_call_target_idx
        ],
        callee_arm_resolver=callee_arm_resolver,
    )
    state.current_items.append(
        InlineCallEntry(
            kind=call_kind, counter_id=counter, callee_name=callee_name,
            callee_section_pointer=callee_section_pointer,
            variant_idx=MISSING_VARIANT_INDEX,
            provider=provider,
        )
    )
    if state.current_kind is BlockKind.FUNCTION_ID:
        # The self-prepend just emitted; transition to BODY block 0.
        enter_body_after_function_id(state)


def _emit_identity(
    state: _WalkState,
    *,
    shifted_id: int,
    identities_row: np.ndarray,
    fid_table: FidBaseTable,
    line_to_name: Mapping[int, str],
    line_to_provider: Mapping[int, str],
    call_targets_per_ct: Sequence[Sequence[CallTarget]],
    kind_to_called_idx_per_ct: Sequence[Mapping[CallTargetType, list[int]]],
    callee_arm_resolver: Callable[[int], Optional[SectionPointerSpec]],
) -> None:
    """IDENTITY band: resolve Category + dispatch per-Category.

    BLOCK -> :func:`_handle_block` (header vs jump driven by the
    section-transition latch). FUNCTION categories ->
    :func:`_handle_function_category` (one InlineCallEntry per token;
    EXT_FUNC provider keyed by FID per decision #28; LOCAL/PLT
    callee_section_pointer resolved via the session-backed
    callee_arm_resolver). COUNTER-but-not-BLOCK -> placeholder
    AsmLine (plan §11 follow-up).
    """
    counter = int(identities_row[state.id_cursor])
    state.id_cursor += 1
    cat = _SHIFTED_ID_TO_CATEGORY[int(shifted_id)]
    if cat is Category.BLOCK:
        _handle_block(state, counter=counter)
        return
    if cat in _CATEGORY_TO_CALL_TARGET_TYPE:
        _handle_function_category(
            state,
            cat=cat,
            counter=counter,
            fid_table=fid_table,
            line_to_name=line_to_name,
            line_to_provider=line_to_provider,
            call_targets_per_ct=call_targets_per_ct,
            kind_to_called_idx_per_ct=kind_to_called_idx_per_ct,
            callee_arm_resolver=callee_arm_resolver,
        )
        return
    state.current_items.append(AsmLine(text=f"<{cat.name.lower()} {counter}>"))


def _maybe_open_function_id_section(state: _WalkState, *, col: int) -> None:
    """Open FUNCTION_ID at the root CT start col (VARIANT_HEADER ->
    FUNCTION_ID). Subsequent FUNCTION_ID -> BODY transitions are
    token-content-driven; see :func:`enter_body_after_function_id`.
    """
    if (
        not state.ct_start_cols
        or col != state.ct_start_cols[0]
        or state.current_kind is not BlockKind.VARIANT_HEADER
    ):
        return
    enter_variant_header_to_function_id(state)


def _maybe_latch_header_trigger(state: _WalkState, *, col: int) -> None:
    """Latch :attr:`pending_header` at runlength-derived trigger cols.

    The actual BODY-block open is deferred to :func:`_handle_block`
    when the upcoming BLOCK_V2 consumes the latch -- matches the
    legacy walker's "no BLOCK_V2 = no new block" rule so jump-table
    footer blocks (no BLOCK_V2 header) stay folded into the prior
    BODY section. Latching is skipped while in VARIANT_HEADER /
    FUNCTION_ID; those transitions own their own ``pending_header``
    handling.
    """
    if col not in state.header_trigger_cols:
        return
    if state.current_kind in (BlockKind.VARIANT_HEADER, BlockKind.FUNCTION_ID):
        return
    state.pending_header = True


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

    Section flow: pre-open VARIANT_HEADER (cols ``[0, n_axis)``); at
    root CT start col transition to FUNCTION_ID; once the self-
    prepend IDENTITY emits (or the first BLOCK_V2 IDENTITY in bare-
    BLOCK_V2 test layouts), open BODY block 0 with ``pending_header``
    set so the ``Block_Def`` + ``block_v2`` header pair is consumed
    silently. Subsequent BODY blocks open at runlength-derived
    :func:`header_trigger_cols`. Terminates on ``token == 0`` or
    end-of-row. ``call_targets_per_ct`` carries each Stage1CallTarget's
    own ``call_targets_section`` so inlined-callee call sites resolve
    against THEIR table; ``callee_arm_resolver`` is the session-
    backed ``_idx_for_section_offset`` closure (returns ``None`` for
    cross-arm/missing; EXTERN call_targets always carry ``None``).
    """
    tokens_row = result.tokens[row]
    id_lo, id_hi = int(result.identity_row_offsets[row]), int(result.identity_row_offsets[row + 1])
    num_lo, num_hi = int(result.number_row_offsets[row]), int(result.number_row_offsets[row + 1])
    identities_row = result.identities[id_lo:id_hi]
    numbers_sig = result.numbers_significant[num_lo:num_hi]
    numbers_se = result.numbers_sign_exponent[num_lo:num_hi]
    if (
        result.block_runlength is None
        or result.block_runlength_row_offsets is None
        or result.insn_runlength is None
        or result.insn_runlength_row_offsets is None
    ):
        raise ValueError(
            "render_row_blocks: BatchDecodeResult missing runlength "
            "sidecars; call batch_decode with "
            "emit_block_n_insns_runlength=True."
        )
    b_lo = int(result.block_runlength_row_offsets[row])
    b_hi = int(result.block_runlength_row_offsets[row + 1])
    i_lo = int(result.insn_runlength_row_offsets[row])
    i_hi = int(result.insn_runlength_row_offsets[row + 1])
    block_runlength_row = result.block_runlength[b_lo:b_hi]
    insn_runlength_row = result.insn_runlength[i_lo:i_hi]
    triggers = header_trigger_cols(
        n_axis=n_axis,
        partial_cut_lengths=partial_cut_lengths,
        block_runlength_row=block_runlength_row,
        insn_runlength_row=insn_runlength_row,
    )
    ct_start_cols = call_target_starts(
        n_axis=n_axis, partial_cut_lengths=partial_cut_lengths,
    )
    # Pre-build per-CT kind partition (LOCAL/PLT/EXTERN -> indices).
    kind_to_called_idx_per_ct: list[Mapping[CallTargetType, list[int]]] = [
        partition_call_target_kinds(ct.type for ct in call_targets_section)
        for call_targets_section in call_targets_per_ct
    ]

    state = _WalkState(
        current_kind=BlockKind.VARIANT_HEADER,
        current_block_idx=-1,
        current_items=[],
        completed=[],
        ct_start_cols=ct_start_cols,
        header_trigger_cols=triggers,
        row=row, n_axis=n_axis,
    )

    n_cols = int(tokens_row.shape[0])
    for col in range(n_cols):
        state.current_col = col
        _maybe_open_function_id_section(state, col=col)
        _maybe_latch_header_trigger(state, col=col)
        maybe_advance_call_target(state, col=col)
        shifted_id = int(tokens_row[col])
        if shifted_id == 0:
            break  # null-content padding tail
        band = classify_shifted_id(shifted_id)
        if band is Band.INSTR_REP:
            # Band switch off NUMBER -> flush any pending multi-chunk
            # source so its rendered text lands BEFORE the upcoming
            # INSTR_REP item (encoder invariant: multi-chunk sources
            # never cross instruction boundaries, and every instruction
            # starts with an INSTR_REP mnemonic).
            flush_accumulator_into(
                state.current_items, accumulator=state.number_accumulator,
            )
            if state.pending_header:
                # ``Block_Def`` carrier -- absorbed silently so the
                # BODY section's first emitted item is the first
                # real instruction.
                continue
            emit_instr_rep(
                state.current_items,
                shifted_id=shifted_id,
                vocab_manager=vocab_manager,
                arch_prefixes=arch_prefixes,
            )
        elif band is Band.NUMBER:
            state.num_cursor = emit_number(
                state.current_items,
                shifted_id=shifted_id,
                numbers_sig=numbers_sig,
                numbers_se=numbers_se,
                num_cursor=state.num_cursor,
                accumulator=state.number_accumulator,
            )
        elif band is Band.IDENTITY:
            # Band switch off NUMBER -> flush before the IDENTITY emit
            # so the IDENTITY-band item appears after the prior
            # NUMBER source's rendered text.
            flush_accumulator_into(
                state.current_items, accumulator=state.number_accumulator,
            )
            _emit_identity(
                state,
                shifted_id=shifted_id,
                identities_row=identities_row,
                fid_table=fid_table,
                line_to_name=line_to_name,
                line_to_provider=line_to_provider,
                call_targets_per_ct=call_targets_per_ct,
                kind_to_called_idx_per_ct=kind_to_called_idx_per_ct,
                callee_arm_resolver=callee_arm_resolver,
            )
        else:
            raise ValueError(
                f"render_row_blocks: unexpected Band {band!r} at "
                f"row={row}, col={col} (shifted_id={shifted_id})"
            )

    # End-of-row flush per cluster #21 H-4: cut variants may have a
    # multi-chunk source whose trailing chunks were truncated. Flush
    # the lead-chunk contribution as best-effort so the visible
    # operand text is not silently dropped.
    flush_accumulator_into(
        state.current_items, accumulator=state.number_accumulator,
    )
    close_current_section(state)
    return state.completed
