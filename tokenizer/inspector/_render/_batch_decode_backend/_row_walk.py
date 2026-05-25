"""Per-row stream walker for the BatchDecodeBackend.

Single concern: translate one row of :attr:`BatchDecodeResult.tokens`
+ its aligned sidecars into ``[RowBlock, ...]``. Plan
``inspector-render-backends.md`` §6 + decisions #16/#17/#18/#29/#30 +
audits B-CRIT-1..4 / B-HIGH-5..7 / B-MED-9..11 / B-LOW-12,14. Cursors
``id_cursor`` + ``num_cursor`` track per-row sidecar positions.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from tokenizer.aligned_data.matched_sections_bin import (
    MISSING_VARIANT_INDEX,
    CallTarget,
)
from tokenizer.inspector._render._protocol import (
    AsmLine,
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
from ._band_emitters import emit_instr_rep, emit_number
from ._boundaries import call_target_starts, header_trigger_cols
from ._callee_resolver import resolve_callee_pointer
from ._fid_table import FidBaseTable


__all__ = ["RowBlock", "render_row_blocks"]


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


@dataclass(frozen=True)
class RowBlock:
    """One rendered block (a block_idx + its ordered LineItems)."""

    block_idx: int
    items: List[LineItem]


@dataclass
class _WalkState:
    """Mutable per-row walk cursors + block accumulator.

    ``current_call_target_idx`` tracks which Stage1CallTarget we're
    walking; it advances when we cross a call-target start column
    (computed from ``partial_cut_lengths`` per
    :func:`_call_target_starts`). The InlineCallEntry resolver reads
    the current CT's ``call_targets_section`` to map (kind, counter) ->
    ``CallTarget.function_section_ptr`` -> SectionPointerSpec.

    ``last_number_shifted_id`` carries the most-recently-emitted
    NUMBER-band token's shifted id (``-1`` = "none"). Resets to ``-1``
    on every non-NUMBER token; drives multi-chunk trailing-slot
    detection inside :func:`_emit_number`.

    ``pending_header`` latches True at every block-header trigger
    column (CT-boundary cols + runlength-computed in-CT block-start
    cols); the NEXT BLOCK_V2 IDENTITY token (in stream order) clears
    the latch and is treated as the block's header (overwrite +
    flush+open). BLOCK_V2 tokens with ``pending_header=False`` are
    in-function jumps. Latching matches both the production layout
    (where col ``n_axis`` holds the LOCAL_FUNC self-prepend, NOT a
    BLOCK_V2, and the actual header lands one slot later) and the
    test layouts that place BLOCK_V2 directly at the CT-boundary col.
    """

    row: int
    n_axis: int
    id_cursor: int
    num_cursor: int
    current_block_idx: int
    current_items: List[LineItem]
    completed: List[RowBlock]
    header_trigger_cols: frozenset[int]
    ct_start_cols: list[int]
    current_call_target_idx: int = 0
    current_col: int = 0
    pending_header: bool = False
    header_seen: bool = False
    last_number_shifted_id: int = -1


def _close_current_block(state: _WalkState) -> None:
    state.completed.append(
        RowBlock(block_idx=state.current_block_idx, items=state.current_items)
    )


def _open_new_block(state: _WalkState, block_idx: int) -> None:
    state.current_block_idx = block_idx
    state.current_items = []


def _handle_block(state: _WalkState, *, counter: int) -> None:
    """BLOCK_V2 IDENTITY dispatch (header vs inline jump).

    With ``pending_header=True`` (latched from a CT-boundary or in-CT
    block-start trigger), this BLOCK_V2 is the block's header: at row
    start it overwrites the pre-allocated block_idx in place (no
    flush); at any later header trigger it flushes the current block
    and opens a new one. BLOCK_V2 tokens with ``pending_header=False``
    are in-function jumps.

    Row layout invariant: BLOCK_V2 IDENTITY tokens never land inside
    the ``[0, n_axis)`` variant_tokens prefix range (the prefix is
    pure instruction-rep). A BLOCK_V2 column at or below the prefix
    end is a data-integrity violation (raised loud for the inspector
    diagnostic surface; the row writer is upstream).
    """
    if state.current_col < state.n_axis:
        raise AssertionError(
            f"BLOCK_V2 IDENTITY token at col={state.current_col} lies "
            f"inside the variant_tokens prefix (n_axis={state.n_axis}); "
            f"the prefix is pure instruction-rep by row-writer contract."
        )
    if state.pending_header:
        if not state.header_seen:
            state.current_block_idx = counter
            state.header_seen = True
        else:
            if state.current_items:
                _close_current_block(state)
            _open_new_block(state, block_idx=counter)
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

    For LOCAL / PLT call_targets, resolves the callee's
    ``function_section_ptr`` to a :class:`SectionPointerSpec` via the
    caller-supplied ``callee_arm_resolver`` closure (the inspector
    factory threads in a session-backed
    ``_idx_for_section_offset`` wrapper). EXT call_targets keep
    ``callee_section_pointer=None`` -- there is no body to inline. The
    LOCAL/PLT path consumes the CURRENT call-target's
    ``call_targets_section`` (each Stage1CallTarget owns its own
    table; inlined-callee call sites resolve against THEIR table, not
    the row's root section's table).
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
    pre-computed block-start columns from the runlength sidecars).
    FUNCTION categories -> :func:`_handle_function_category` (one
    InlineCallEntry per token; EXT_FUNC provider keyed by FID per
    decision #28; LOCAL/PLT callee_section_pointer resolved via the
    session-backed callee_arm_resolver). COUNTER-but-not-BLOCK ->
    placeholder AsmLine (plan §11 follow-up).
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
) -> List[RowBlock]:
    """Walk one row and split into blocks.

    Pre-allocates an empty entry block; the variant-tokens prefix lands
    in it as instruction-rep AsmLines, and the first BLOCK_V2 at a
    runlength-computed block-start column overwrites the pre-allocated
    block's index. Subsequent block-start columns flush + open a new
    block. Terminates on ``token == 0`` (null padding tail) or end-of-
    row.

    Block boundaries are driven by the runlength sidecars
    (:attr:`BatchDecodeResult.block_runlength` +
    :attr:`BatchDecodeResult.insn_runlength` + their per-row offsets),
    not by BLOCK_V2 tokens directly -- a single call_target may contain
    multiple block headers, and BLOCK_V2 tokens NOT at a sidecar-
    computed block-start column are in-function jumps.

    ``call_targets_per_ct`` is the per-Stage1CallTarget
    ``call_targets_section`` (the same list the FTL render path
    consumes). ``callee_arm_resolver`` is the session-backed
    ``_idx_for_section_offset`` closure -- LOCAL / PLT call_targets
    resolve their ``function_section_ptr`` to a
    :class:`SectionPointerSpec` (or ``None`` for cross-arm / missing)
    via this closure; EXTERN call_targets always carry ``None``.
    """
    tokens_row = result.tokens[row]
    id_lo, id_hi = int(result.identity_row_offsets[row]), int(result.identity_row_offsets[row + 1])
    num_lo, num_hi = int(result.number_row_offsets[row]), int(result.number_row_offsets[row + 1])
    identities_row = result.identities[id_lo:id_hi]
    numbers_sig = result.numbers_significant[num_lo:num_hi]
    numbers_se = result.numbers_sign_exponent[num_lo:num_hi]

    # Pull runlength sidecars -- the canonical block-boundary source.
    # Required when ``emit_block_n_insns_runlength=True`` was passed to
    # batch_decode; the BatchDecodeBackend pins this at True.
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
    ct_boundary_set = set(ct_start_cols)
    # Pre-build per-CT kind partition (LOCAL/PLT/EXTERN -> indices).
    kind_to_called_idx_per_ct: list[Mapping[CallTargetType, list[int]]] = [
        partition_call_target_kinds(ct.type for ct in call_targets_section)
        for call_targets_section in call_targets_per_ct
    ]

    state = _WalkState(
        row=row, n_axis=n_axis, id_cursor=0, num_cursor=0,
        current_block_idx=0, current_items=[], completed=[],
        header_trigger_cols=triggers,
        ct_start_cols=ct_start_cols,
    )

    n_cols = int(tokens_row.shape[0])
    for col in range(n_cols):
        state.current_col = col
        if col in ct_boundary_set and col != n_axis:
            # Advance to next call_target (root index 0 stays at col
            # n_axis; subsequent CTs increment).
            state.current_call_target_idx += 1
        if col in triggers:
            state.pending_header = True
        shifted_id = int(tokens_row[col])
        if shifted_id == 0:
            break  # null-content padding tail
        band = classify_shifted_id(shifted_id)
        if band is Band.INSTR_REP:
            emit_instr_rep(
                state.current_items,
                shifted_id=shifted_id,
                vocab_manager=vocab_manager,
            )
            state.last_number_shifted_id = -1
        elif band is Band.NUMBER:
            state.num_cursor = emit_number(
                state.current_items,
                shifted_id=shifted_id,
                numbers_sig=numbers_sig,
                numbers_se=numbers_se,
                num_cursor=state.num_cursor,
                last_number_shifted_id=state.last_number_shifted_id,
            )
            state.last_number_shifted_id = int(shifted_id)
        elif band is Band.IDENTITY:
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
            state.last_number_shifted_id = -1
        else:
            raise ValueError(
                f"render_row_blocks: unexpected Band {band!r} at "
                f"row={row}, col={col} (shifted_id={shifted_id})"
            )

    if state.current_items:
        _close_current_block(state)
    return state.completed
