"""Per-row walk state for the BatchDecodeBackend row walker.

Single concern: the state the per-col loop reads/mutates plus the
typed enums that drive instruction-grouping policy decisions, plus
the construction-time slicing that derives the initial
:class:`_WalkState` + :class:`_RowSidecars` from one
:class:`BatchDecodeResult` row. Plan reference:
``inspector-followup.md`` W3-16 W4-AMENDED + A-L2 H2 (the enum
replaces the implicit cross-product of legacy boolean flags) +
A-L3 H-3 (jump-table footer detection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    FUNCTION_CATEGORIES,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
)
from tokenizer.aligned_data.loader.decoded._number_render_collector import (
    _NumberAccumulator,
)
from tokenizer.aligned_data.matched_sections_bin import CallTarget
from tokenizer.inspector._render._protocol import BlockKind
from tokenizer.inspector._render._render_block import (
    partition_call_target_kinds,
)
from tokenizer.tokens import Category

from .._boundaries import call_target_starts, header_trigger_cols
from .._sections import WalkSectionState


__all__ = [
    "_CATEGORY_TO_CALL_TARGET_TYPE",
    "_InsnEmitPolicy",
    "_RowSidecars",
    "_WalkState",
    "_init_walk_state",
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
    ``saw_jump_table``) per audit A-L2 H2 + W3-16 W4-AMENDED.

    * :attr:`SILENT_HEADER` -- ``Block_Def`` + ``block_v2:N`` header
      pair being consumed; finalize asserts empty text/openables.
    * :attr:`JUMP_TABLE_FOOTER` -- instruction has seen a
      :attr:`Category.JUMP_TABLE` IDENTITY; subsequent ``Block_V2``
      targets emit :class:`InlineJumpEntry`, NOT block headers.
    * :attr:`REAL` -- ordinary instruction; emits one
      :class:`AsmLine` per :func:`_finalize_instruction` (R2c).

    Consumed by :mod:`._instruction`'s finalize (R2c) and by
    :mod:`._dispatch._handle_block` (R2a: routes ``Block_V2`` via
    :attr:`WalkSectionState.inside_jump_table_footer_block`).
    """

    SILENT_HEADER = "silent_header"
    JUMP_TABLE_FOOTER = "jump_table_footer"
    REAL = "real"


@dataclass
class _WalkState(WalkSectionState):
    """Per-row walk state: composes :class:`WalkSectionState` with
    band-emitter cursors + per-instruction grouping fields.

    ``id_cursor`` / ``num_cursor`` track per-row sidecar positions.
    ``current_col`` is the column index the loop is processing.
    ``number_accumulator`` is the per-row :class:`_NumberAccumulator`
    instance that groups multi-chunk NUMBER sources (W3-17). A band
    switch off NUMBER -- an :class:`Band.INSTR_REP` or
    :class:`Band.IDENTITY` token -- flushes pending chunks into the
    current section; an end-of-row break also force-flushes per
    cluster #21 H-4 cut-variant tolerance.

    ``insn_emit_policy`` + ``saw_jump_table_this_insn`` are
    pre-paved per-instruction surfaces (R2c per-instruction
    collector consumes them; R2a's :mod:`._dispatch` sets the
    sibling block-level flag :attr:`WalkSectionState.inside_jump_table_footer_block`).
    """

    row: int = 0
    n_axis: int = 0
    id_cursor: int = 0
    num_cursor: int = 0
    current_col: int = 0
    number_accumulator: _NumberAccumulator = field(
        default_factory=_NumberAccumulator
    )
    insn_emit_policy: _InsnEmitPolicy = _InsnEmitPolicy.REAL
    saw_jump_table_this_insn: bool = False


@dataclass(frozen=True)
class _RowSidecars:
    """Per-row sidecar slices threaded from :func:`_init_walk_state`
    to the per-col loop. Frozen so loop bodies never accidentally
    mutate the slice views.
    """

    tokens_row: np.ndarray
    identities_row: np.ndarray
    numbers_sig: np.ndarray
    numbers_se: np.ndarray
    kind_to_called_idx_per_ct: list[Mapping[CallTargetType, list[int]]]


def _init_walk_state(
    *,
    result: BatchDecodeResult,
    row: int,
    n_axis: int,
    partial_cut_lengths: list[int],
    call_targets_per_ct: Sequence[Sequence[CallTarget]],
) -> tuple[_WalkState, _RowSidecars]:
    """Slice per-row sidecar arrays + build the initial :class:`_WalkState`.

    Raises :class:`ValueError` when the runlength sidecars are
    missing (call :func:`batch_decode` with
    ``emit_block_n_insns_runlength=True``).
    """
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
    id_lo = int(result.identity_row_offsets[row])
    id_hi = int(result.identity_row_offsets[row + 1])
    num_lo = int(result.number_row_offsets[row])
    num_hi = int(result.number_row_offsets[row + 1])
    b_lo = int(result.block_runlength_row_offsets[row])
    b_hi = int(result.block_runlength_row_offsets[row + 1])
    i_lo = int(result.insn_runlength_row_offsets[row])
    i_hi = int(result.insn_runlength_row_offsets[row + 1])
    state = _WalkState(
        current_kind=BlockKind.VARIANT_HEADER,
        current_block_idx=-1,
        current_items=[],
        completed=[],
        ct_start_cols=call_target_starts(
            n_axis=n_axis, partial_cut_lengths=partial_cut_lengths,
        ),
        header_trigger_cols=header_trigger_cols(
            n_axis=n_axis,
            partial_cut_lengths=partial_cut_lengths,
            block_runlength_row=result.block_runlength[b_lo:b_hi],
            insn_runlength_row=result.insn_runlength[i_lo:i_hi],
        ),
        row=row, n_axis=n_axis,
    )
    sidecars = _RowSidecars(
        tokens_row=result.tokens[row],
        identities_row=result.identities[id_lo:id_hi],
        numbers_sig=result.numbers_significant[num_lo:num_hi],
        numbers_se=result.numbers_sign_exponent[num_lo:num_hi],
        kind_to_called_idx_per_ct=[
            partition_call_target_kinds(ct.type for ct in cts)
            for cts in call_targets_per_ct
        ],
    )
    return state, sidecars
