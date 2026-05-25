"""Per-row walk state for the BatchDecodeBackend row walker.

Single concern: the mutable state the per-col loop reads / mutates,
plus the typed enums that drive instruction-grouping policy decisions.

:class:`_WalkState` composes :class:`WalkSectionState` (the section-
state-machine concern owned by :mod:`.._sections`) with the band-
emitter cursors. The :class:`_InsnEmitPolicy` enum encodes the
per-instruction emit-decision policy (silent-header pair / jump-table
footer / real instruction); :data:`saw_jump_table_this_insn` +
:data:`inside_jump_table_footer_block` are the typed flags W3-16
threads through the dispatch + finalizer surfaces. Plan reference:
``inspector-followup.md`` W3-16 W4-AMENDED + A-L2 H2 (the enum
replaces the implicit cross-product of legacy boolean flags) +
A-L3 H-3 (jump-table footer detection).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    FUNCTION_CATEGORIES,
)
from tokenizer.tokens import Category

from .._sections import WalkSectionState


__all__ = [
    "_CATEGORY_TO_CALL_TARGET_TYPE",
    "_InsnEmitPolicy",
    "_WalkState",
]


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


class _InsnEmitPolicy(Enum):
    """Per-instruction emit-decision policy.

    Replaces the implicit cross-product of legacy boolean flags
    (``current_insn_in_silent_header`` x ``pending_header`` x
    ``saw_jump_table``) with a single typed discriminator per
    audit A-L2 H2 + W3-16 W4-AMENDED.

    State transitions live on :class:`_WalkState`:

    * :attr:`SILENT_HEADER` -- the ``Block_Def`` + ``block_v2:N``
      header pair currently being consumed. :func:`_finalize_instruction`
      asserts empty text/openables, emits no :class:`AsmLine`. Set on
      :func:`_start_new_instruction` when :attr:`WalkSectionState.pending_header`
      is True at the instruction's first slot.
    * :attr:`JUMP_TABLE_FOOTER` -- the instruction has seen a
      :attr:`Category.JUMP_TABLE` IDENTITY token, so it is the
      jump-table footer instruction, NOT a silent header. Each
      subsequent ``Block_V2`` target IDENTITY emits an
      :class:`InlineJumpEntry` rather than consuming the latch as a
      block header. Transition: SILENT_HEADER -> JUMP_TABLE_FOOTER on
      :attr:`Category.JUMP_TABLE` IDENTITY emission (see :mod:`._dispatch`).
    * :attr:`REAL` -- ordinary instruction; emits one :class:`AsmLine`
      per :func:`_finalize_instruction` call. Default state at the
      start of every non-silent-header instruction.

    The enum is consumed by :mod:`._instruction`'s
    :func:`_finalize_instruction` (R2c) and by :mod:`._dispatch`'s
    :func:`_handle_block` (R2a: routes :class:`Block_V2` to
    :class:`InlineJumpEntry` when the block-level
    :attr:`_WalkState.inside_jump_table_footer_block` flag is set,
    preserving the latch for any pending header still to come).
    """

    SILENT_HEADER = "silent_header"
    JUMP_TABLE_FOOTER = "jump_table_footer"
    REAL = "real"


@dataclass
class _WalkState(WalkSectionState):
    """Per-row walk state: composes :class:`WalkSectionState` (the
    section concern; see :mod:`.._sections`) with the band-emitter
    cursors that the per-col loop reads + updates.

    ``id_cursor`` / ``num_cursor`` track per-row sidecar positions.
    ``current_col`` is the column index the loop is processing;
    crosses :func:`._dispatch._handle_block`'s ``n_axis`` invariant
    check. ``last_number_shifted_id`` carries the prior NUMBER-band
    token's shifted id (``-1`` = none); drives multi-chunk trailing-
    slot detection inside :func:`emit_number` and resets on every
    non-NUMBER emit.

    ``insn_emit_policy`` carries the per-instruction emit-decision
    policy described on :class:`_InsnEmitPolicy`. Pre-paved on
    :class:`_WalkState` so the R2c per-instruction collector
    (:mod:`._instruction`) reads + writes it through a single typed
    surface; the R2a structural split lands the type so the R2c
    diff is a code-add only. ``saw_jump_table_this_insn`` is the
    per-instruction sibling of
    :attr:`WalkSectionState.inside_jump_table_footer_block`: set on
    :class:`Category.JUMP_TABLE` IDENTITY emission, reset on
    instruction-boundary finalize (W3-16 W4-AMENDED).
    """

    row: int = 0
    n_axis: int = 0
    id_cursor: int = 0
    num_cursor: int = 0
    current_col: int = 0
    last_number_shifted_id: int = -1
    insn_emit_policy: _InsnEmitPolicy = _InsnEmitPolicy.REAL
    saw_jump_table_this_insn: bool = False
