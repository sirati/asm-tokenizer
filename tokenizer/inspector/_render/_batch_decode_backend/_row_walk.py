"""Per-row stream walker for the BatchDecodeBackend.

Single concern: translate one row of :attr:`BatchDecodeResult.tokens`
+ its aligned sidecars into ``[RowBlock, ...]``. Plan
``inspector-render-backends.md`` §6 + decisions #16/#17/#18/#29/#30 +
audits B-CRIT-1..4 / B-HIGH-5..7 / B-MED-9..11 / B-LOW-12,14. Cursors
``id_cursor`` + ``num_cursor`` track per-row sidecar positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    FUNCTION_CATEGORIES,
    _SHIFTED_ID_TO_CATEGORY,
)
from tokenizer.aligned_data.loader.batch_decode._number_decode._band_constants import (
    _NUMBER_BAND_LO_SHIFTED,
    _NUMBER_BLOCK_TOKEN_TYPES,
)
from tokenizer.aligned_data.loader.batch_decode._types import BatchDecodeResult
from tokenizer.aligned_data.loader.decoded.number_hex_format import (
    chunks_to_hex_bits,
)
from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX
from tokenizer.inspector._render._protocol import (
    AsmLine,
    InlineCallEntry,
    InlineJumpEntry,
    LineItem,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category

from ._band import Band, classify_shifted_id
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


# Stage-2 strip-shift (``-256``) applied at token assembly; reverse via
# ``+_V2_RESERVED_DIGIT_COUNT`` to recover the unified-vocab id.
_V2_RESERVED_DIGIT_COUNT: int = VocabularyManager._V2_RESERVED_DIGIT_COUNT


# Shifted ids whose source can span multiple NUMBER-band tokens on the
# wire (VC2 with K_visible chunks, F128 finite with 2 chunks). Derived
# from ``_NUMBER_BLOCK_TOKEN_TYPES`` so a NUMBER-block layout shift
# updates this set without touching the walker. Phase-1 trailing-chunk
# detection (plan #17): a NUMBER token whose shifted id matches the
# immediately-preceding NUMBER token's id AND lives in this set is a
# trailing chunk of the same source -- the walker emits ``"..."`` for
# it instead of feeding the chunk pair to :func:`chunks_to_hex_bits`.
# Single-chunk types (F16/BF16/F32/F64/F80) are excluded; consecutive
# tokens of those types come from distinct sources and each renders.
_MULTI_CHUNK_SHIFTED_IDS: frozenset[int] = frozenset(
    1 + i
    for i, tt in enumerate(_NUMBER_BLOCK_TOKEN_TYPES)
    if tt.name in ("VALUED_CONST_V2", "FLOAT128")
)


@dataclass(frozen=True)
class RowBlock:
    """One rendered block (a block_idx + its ordered LineItems)."""

    block_idx: int
    items: List[LineItem]


@dataclass
class _WalkState:
    """Mutable per-row walk cursors + block accumulator.

    ``pending_header`` is True at call_target boundaries (next BLOCK_V2
    is an entry header). ``header_seen`` is False until the first
    BLOCK_V2 of the row; decision #30 overwrites the pre-allocated
    block_idx in place at that point (no flush). Subsequent
    ``pending_header`` BLOCK_V2s (inlined callees per #29) flush-then-open.

    ``last_number_shifted_id`` carries the most-recently-emitted
    NUMBER-band token's shifted id (``-1`` = "none"). Resets to ``-1``
    on every non-NUMBER token; drives multi-chunk trailing-slot
    detection inside :func:`_emit_number`.
    """

    row: int
    n_axis: int
    id_cursor: int
    num_cursor: int
    current_block_idx: int
    current_items: List[LineItem]
    completed: List[RowBlock]
    current_col: int = 0
    pending_header: bool = False
    header_seen: bool = False
    last_number_shifted_id: int = -1


def _call_target_starts(*, n_axis: int, partial_cut_lengths: list[int]) -> set[int]:
    """Per-call-target start cols (set form for O(1) boundary detect).

    Row layout per ``_token_assembly.py``:
    ``row[0:n_axis] = variant_tokens_prefix``; then each call_target's
    ``expanded_token_ids[:partial_cut_length]`` lands consecutively.
    """
    starts: set[int] = {n_axis}
    running = n_axis
    for pcl in partial_cut_lengths[:-1]:
        running += pcl
        starts.add(running)
    return starts


def _emit_instr_rep(
    state: _WalkState, *, shifted_id: int, vocab_manager: VocabularyManager
) -> None:
    """Instruction-rep band: vocab lookup -> AsmLine."""
    original_id = int(shifted_id) + _V2_RESERVED_DIGIT_COUNT
    state.current_items.append(AsmLine(text=vocab_manager.get_token_str(original_id)))


def _emit_number(
    state: _WalkState,
    *,
    shifted_id: int,
    numbers_sig: np.ndarray,
    numbers_se: np.ndarray,
) -> None:
    """NUMBER band: consume one chunk pair, render hex or trailing slot.

    Phase-1 trailing-chunk detection (plan #17): for shifted ids in
    :data:`_MULTI_CHUNK_SHIFTED_IDS` (VC2, F128), consecutive same-id
    slots on the wire belong to the SAME source and only the lead
    chunk goes through :func:`chunks_to_hex_bits`; every trailing slot
    emits ``AsmLine("...")``. This is the encoder's stream order: a
    multi-chunk source emits K consecutive tokens of the same shifted
    id before any other token interrupts. Phase 2 (plan section 11)
    will replace this stream-position heuristic with the chunk-count
    sidecar so full multi-chunk reconstruction is possible.
    """
    band_index = int(shifted_id) - _NUMBER_BAND_LO_SHIFTED
    token_type = _NUMBER_BLOCK_TOKEN_TYPES[band_index]
    sig = numbers_sig[state.num_cursor]
    se = numbers_se[state.num_cursor]
    is_trailing_chunk = (
        shifted_id == state.last_number_shifted_id
        and shifted_id in _MULTI_CHUNK_SHIFTED_IDS
    )
    if is_trailing_chunk:
        state.current_items.append(AsmLine(text="..."))
    else:
        state.current_items.append(
            AsmLine(text=chunks_to_hex_bits(token_type, sig, se))
        )
    state.num_cursor += 1
    state.last_number_shifted_id = int(shifted_id)


def _close_current_block(state: _WalkState) -> None:
    state.completed.append(
        RowBlock(block_idx=state.current_block_idx, items=state.current_items)
    )


def _open_new_block(state: _WalkState, block_idx: int) -> None:
    state.current_block_idx = block_idx
    state.current_items = []


def _handle_block(state: _WalkState, *, counter: int) -> None:
    """BLOCK_V2 IDENTITY dispatch (header vs inline jump).

    Decision #30: at row start the pre-allocated block absorbs the
    variant_tokens prefix + self-prepend; the first BLOCK_V2 overwrites
    block_idx in place (no flush). Inlined callees (#29) flush+open.
    Non-pending BLOCK_V2s are in-function jumps.

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
) -> None:
    """FUNCTION-Category dispatch: FID lookup + InlineCallEntry."""
    call_kind = _CATEGORY_TO_CALL_TARGET_TYPE[cat]
    fid = fid_table.lookup(row=state.row, cat=cat, counter=counter)
    callee_name = line_to_name.get(fid, "?")
    provider = (
        line_to_provider.get(fid) if call_kind is CallTargetType.EXTERN else None
    )
    state.current_items.append(
        InlineCallEntry(
            kind=call_kind, counter_id=counter, callee_name=callee_name,
            callee_section_pointer=None, variant_idx=MISSING_VARIANT_INDEX,
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
) -> None:
    """IDENTITY band: resolve Category + dispatch per-Category.

    BLOCK -> :func:`_handle_block` (header vs jump per #29 / #30).
    FUNCTION categories -> :func:`_handle_function_category` (one
    InlineCallEntry per token; EXT_FUNC provider keyed by FID per
    decision #28). COUNTER-but-not-BLOCK -> Phase-1 placeholder
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
        )
        return
    state.current_items.append(AsmLine(text=f"<{cat.name.lower()} {counter}>"))


def render_row_blocks(
    *,
    result: BatchDecodeResult,
    row: int,
    n_axis: int,
    partial_cut_lengths: list[int],
    vocab_manager: VocabularyManager,
    fid_table: FidBaseTable,
    line_to_name: Mapping[int, str],
    line_to_provider: Mapping[int, str],
) -> List[RowBlock]:
    """Walk one row and split into blocks.

    Pre-allocates an empty entry block (decision #30); the
    variant-tokens prefix lands in it as instruction-rep AsmLines, the
    first BLOCK_V2 inside the first call_target span overwrites the
    pre-allocated block's index. Terminates on ``token == 0`` (null
    padding tail) or end-of-row. Stage-1 per-call-target start cols
    (decision #29) flip ``pending_header`` so each new span's first
    BLOCK_V2 is the entry header. Empty rows return ``[]``.
    """
    tokens_row = result.tokens[row]
    id_lo, id_hi = int(result.identity_row_offsets[row]), int(result.identity_row_offsets[row + 1])
    num_lo, num_hi = int(result.number_row_offsets[row]), int(result.number_row_offsets[row + 1])
    identities_row = result.identities[id_lo:id_hi]
    numbers_sig = result.numbers_significant[num_lo:num_hi]
    numbers_se = result.numbers_sign_exponent[num_lo:num_hi]

    boundary_set = _call_target_starts(
        n_axis=n_axis, partial_cut_lengths=partial_cut_lengths
    )
    state = _WalkState(
        row=row, n_axis=n_axis, id_cursor=0, num_cursor=0,
        current_block_idx=0, current_items=[], completed=[],
    )

    n_cols = int(tokens_row.shape[0])
    for col in range(n_cols):
        state.current_col = col
        if col in boundary_set:
            state.pending_header = True
        shifted_id = int(tokens_row[col])
        if shifted_id == 0:
            break  # null-content padding tail
        band = classify_shifted_id(shifted_id)
        if band is Band.INSTR_REP:
            _emit_instr_rep(state, shifted_id=shifted_id, vocab_manager=vocab_manager)
            state.last_number_shifted_id = -1
        elif band is Band.NUMBER:
            _emit_number(
                state,
                shifted_id=shifted_id,
                numbers_sig=numbers_sig,
                numbers_se=numbers_se,
            )
        elif band is Band.IDENTITY:
            _emit_identity(
                state,
                shifted_id=shifted_id,
                identities_row=identities_row,
                fid_table=fid_table,
                line_to_name=line_to_name,
                line_to_provider=line_to_provider,
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
