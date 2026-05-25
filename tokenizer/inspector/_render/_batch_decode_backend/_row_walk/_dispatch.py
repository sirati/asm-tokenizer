"""IDENTITY-band per-Category dispatch for the row walker.

Single concern: route an IDENTITY token to the right emit path
based on its resolved :class:`Category`. BLOCK -> header consume vs
inline jump vs jump-table target. FUNCTION categories -> InlineCallEntry.
JUMP_TABLE -> arms the per-block jump-table-footer flag (W3-16).
Other counter-bearing categories -> ``<category counter>`` placeholder.

The per-col loop driver (:mod:`._driver`) is the only caller; this
module knows nothing about runlength sidecars or wire-level cursors.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    _SHIFTED_ID_TO_CATEGORY,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
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
)
from tokenizer.tokens import Category

from .._callee_resolver import resolve_callee_pointer
from .._fid_table import FidBaseTable
from .._sections import (
    enter_body_after_function_id,
    enter_new_body_block,
    set_current_body_block_idx,
)
from ._state import _CATEGORY_TO_CALL_TARGET_TYPE, _WalkState


__all__ = [
    "_emit_identity",
    "_handle_block",
    "_handle_function_category",
]


def _handle_block(state: _WalkState, *, counter: int) -> None:
    """BLOCK_V2 IDENTITY dispatch: header vs inline jump vs jump-table target.

    ``pending_header=True`` (latched at runlength-derived trigger cols)
    consumes BLOCK_V2 as a body-block header: flushes any prior BODY
    content and opens a fresh BODY block with ``block_idx = counter``
    (matching the InlineJumpEntry target scheme). Otherwise BLOCK_V2 is
    an in-function jump. A BLOCK_V2 inside ``[0, n_axis)`` is raised
    loud -- the variant_tokens prefix is pure INSTR_REP by contract.

    W3-16 W4-AMENDED: once
    :attr:`WalkSectionState.inside_jump_table_footer_block` is set,
    BLOCK_V2 is a jump-table target -- emit :class:`InlineJumpEntry`
    and clear ``pending_header`` once so subsequent INSTR_REPs are not
    absorbed silently. Footer stays folded into the prior BODY section
    per W3-16's "no Block_V2 = no new block" rule.
    """
    if state.current_col < state.n_axis:
        raise AssertionError(
            f"BLOCK_V2 IDENTITY token at col={state.current_col} lies "
            f"inside the variant_tokens prefix (n_axis={state.n_axis}); "
            f"the prefix is pure instruction-rep by row-writer contract."
        )
    if state.current_kind is BlockKind.FUNCTION_ID:
        # Bare-BLOCK_V2 test layout: no self-prepend; transition first.
        enter_body_after_function_id(state)
    if state.inside_jump_table_footer_block:
        # Footer target: emit InlineJumpEntry; clear latch once.
        state.current_items.append(InlineJumpEntry(target_block_idx=counter))
        state.pending_header = False
        return
    if state.pending_header:
        if state.current_items:
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

    LOCAL/PLT resolve ``function_section_ptr`` via
    ``callee_arm_resolver`` indexed into the CURRENT call-target's
    ``call_targets_section`` (so inlined-callee call sites resolve
    against THEIR table). EXT keeps ``callee_section_pointer=None``
    and carries the provider name (decision #28). The root CT's
    LOCAL_FUNC self-prepend (counter 0) emits into FUNCTION_ID and
    then transitions to BODY block 0.
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

    BLOCK -> :func:`_handle_block`. FUNCTION categories ->
    :func:`_handle_function_category`. JUMP_TABLE -> arms the
    per-block jump-table-footer flag (W3-16) + emits the placeholder
    line. Other counter-bearing categories -> ``<category counter>``
    AsmLine placeholder (plan §11 follow-up).
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
    if cat is Category.JUMP_TABLE:
        # Footer begins: arm block-level flag so trailing Block_V2
        # tokens route as InlineJumpEntry. The per-instruction
        # sibling ``saw_jump_table_this_insn`` is reset by R2c's
        # per-instruction finalizer.
        state.inside_jump_table_footer_block = True
        state.saw_jump_table_this_insn = True
    state.current_items.append(AsmLine(text=f"<{cat.name.lower()} {counter}>"))
