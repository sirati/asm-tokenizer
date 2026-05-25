"""Per-col loop driver for the BatchDecodeBackend row walker.

Single concern: walk one row's tokens column-by-column, dispatching
each shifted-id to its band-specific emit path. Section transitions
(VARIANT_HEADER -> FUNCTION_ID, FUNCTION_ID -> BODY) and the
runlength-derived ``pending_header`` latch live here; the per-band
emit primitives are in :mod:`.._band_emitters` and the IDENTITY
dispatch is in :mod:`._dispatch`. Plan reference:
``inspector-render-backends.md`` §6 + decisions #16/#17/#18/#29/#30.
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
from ._state import _WalkState, _init_walk_state


__all__ = ["render_row_blocks"]


def _maybe_open_function_id_section(state: _WalkState, *, col: int) -> None:
    """VARIANT_HEADER -> FUNCTION_ID at the root CT start col.

    Subsequent FUNCTION_ID -> BODY transitions are token-content-
    driven; see :func:`.._sections.enter_body_after_function_id`.
    """
    if (
        not state.ct_start_cols
        or col != state.ct_start_cols[0]
        or state.current_kind is not BlockKind.VARIANT_HEADER
    ):
        return
    enter_variant_header_to_function_id(state)


def _maybe_latch_header_trigger(state: _WalkState, *, col: int) -> None:
    """Latch ``pending_header`` at runlength-derived trigger cols.

    The BODY-block open is deferred to :func:`._dispatch._handle_block`
    when the upcoming BLOCK_V2 consumes the latch (legacy "no Block_V2
    = no new block" rule -- jump-table footer blocks stay folded).
    Skipped while in VARIANT_HEADER / FUNCTION_ID; those own their own
    ``pending_header`` handling.
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

    Section flow: VARIANT_HEADER (cols ``[0, n_axis)``) -> FUNCTION_ID
    (root CT start col) -> BODY blocks (each runlength-derived header
    trigger col; ``Block_Def`` + ``block_v2`` header pair consumed
    silently). Terminates on ``token == 0`` or end-of-row.

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
        _maybe_open_function_id_section(state, col=col)
        _maybe_latch_header_trigger(state, col=col)
        maybe_advance_call_target(state, col=col)
        shifted_id = int(sidecars.tokens_row[col])
        if shifted_id == 0:
            break  # null-content padding tail
        band = classify_shifted_id(shifted_id)
        if band is Band.INSTR_REP:
            if state.pending_header:
                # ``Block_Def`` carrier -- absorbed silently.
                state.last_number_shifted_id = -1
                continue
            emit_instr_rep(
                state.current_items,
                shifted_id=shifted_id,
                vocab_manager=vocab_manager,
                arch_prefixes=arch_prefixes,
            )
            state.last_number_shifted_id = -1
        elif band is Band.NUMBER:
            state.num_cursor = emit_number(
                state.current_items,
                shifted_id=shifted_id,
                numbers_sig=sidecars.numbers_sig,
                numbers_se=sidecars.numbers_se,
                num_cursor=state.num_cursor,
                last_number_shifted_id=state.last_number_shifted_id,
            )
            state.last_number_shifted_id = int(shifted_id)
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
            state.last_number_shifted_id = -1
        else:
            raise ValueError(
                f"render_row_blocks: unexpected Band {band!r} at "
                f"row={row}, col={col} (shifted_id={shifted_id})"
            )

    close_current_section(state)
    return state.completed
