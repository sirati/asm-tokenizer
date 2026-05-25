"""Section accumulator for the BatchDecodeBackend row walker.

Single concern: the per-column section state machine that splits one
row into typed sections (:class:`BlockKind`-discriminated):

* :attr:`BlockKind.VARIANT_HEADER` -- pre-allocated at construction;
  collects the variant_tokens prefix ``[0, n_axis)``.
* :attr:`BlockKind.FUNCTION_ID` -- opens at the root CT start
  (``ct_start_cols[0]``); collects the single LOCAL_FUNC self-prepend
  identity slot.
* :attr:`BlockKind.BODY` -- one section per basic block; opens at
  each runlength-derived body-block start column (see
  :mod:`._boundaries`'s :func:`body_block_starts`).

The walker reads per-col transitions from
:attr:`_WalkState.section_transitions`; opening a BODY section
latches :attr:`_WalkState.pending_header` so the row writer's
``Block_Def`` INSTR_REP + ``block_v2:N`` IDENTITY header pair is
consumed silently (the parent tree row's label already encodes the
block index, so the pair stops polluting block contents).

Owned by the BatchDecode backend; this module knows nothing about
band classification, FID resolution, or callee pointer resolution --
those are :mod:`._row_walk`'s concern. The split keeps each file
under the codebase's ~400-LOC cap per the project convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tokenizer.inspector._render._protocol import BlockKind, LineItem


__all__ = [
    "RowSection",
    "WalkSectionState",
    "close_current_section",
    "enter_body_after_function_id",
    "enter_new_body_block",
    "enter_variant_header_to_function_id",
    "maybe_advance_call_target",
    "open_section",
    "set_current_body_block_idx",
]


@dataclass(frozen=True)
class RowSection:
    """One rendered section -- variant header, function id, or body block.

    ``kind`` discriminates the three section flavours
    (:class:`BlockKind`). ``block_idx`` is the body-block index for
    :attr:`BlockKind.BODY`; the non-body kinds use ``-1`` as a
    sentinel so callers reading ``block_idx`` for body lookups never
    collide with header rows.
    """

    kind: BlockKind
    block_idx: int
    items: List[LineItem]


@dataclass
class WalkSectionState:
    """Mutable section-accumulator slice of the row walker's state.

    Owns only the fields the section state machine needs to read /
    mutate (``current_kind``, ``current_block_idx``, ``current_items``,
    ``completed``, ``ct_start_cols``, ``header_trigger_cols``,
    ``current_call_target_idx``, ``pending_header``); the per-band
    emitter state (token cursors, ``last_number_shifted_id``) lives
    on :class:`_row_walk._WalkState` which composes this state via
    inheritance. The split keeps the section concern self-contained.
    """

    current_kind: BlockKind
    current_block_idx: int
    current_items: List[LineItem]
    completed: List[RowSection]
    ct_start_cols: list[int]
    header_trigger_cols: frozenset[int]
    current_call_target_idx: int = 0
    pending_header: bool = False


def close_current_section(state: WalkSectionState) -> None:
    """Flush the current section into :attr:`WalkSectionState.completed`.

    Empty sections (no items emitted) are suppressed across all
    kinds: an empty VARIANT_HEADER (e.g. ``n_axis=0`` rows), empty
    FUNCTION_ID (a row with no self-prepend slot), or empty BODY
    (clean end-of-stream after a section transition) produce no
    visible tree row. Callers that want a placeholder for an empty
    span can detect the missing section by kind ordering.
    """
    if not state.current_items:
        return
    state.completed.append(
        RowSection(
            kind=state.current_kind,
            block_idx=state.current_block_idx,
            items=state.current_items,
        )
    )


def open_section(
    state: WalkSectionState, *, kind: BlockKind, block_idx: int
) -> None:
    """Set the current section to a fresh ``(kind, block_idx)`` pair."""
    state.current_kind = kind
    state.current_block_idx = block_idx
    state.current_items = []


def enter_variant_header_to_function_id(state: WalkSectionState) -> None:
    """Close VARIANT_HEADER and open FUNCTION_ID at the root CT start.

    Called by the walker exactly once per row, at the column that
    equals ``ct_start_cols[0]`` (the root call_target's start). The
    FUNCTION_ID section captures the LOCAL_FUNC self-prepend IDENTITY
    that the row writer places at this slot; the walker drives the
    next transition (FUNCTION_ID -> BODY) via
    :func:`enter_body_after_function_id` once that IDENTITY has been
    emitted (or immediately, if the first IDENTITY at the slot is
    itself a BLOCK_V2 -- bare-BLOCK_V2 test layouts).
    """
    close_current_section(state)
    open_section(state, kind=BlockKind.FUNCTION_ID, block_idx=-1)


def enter_body_after_function_id(state: WalkSectionState) -> None:
    """Close the FUNCTION_ID section and open BODY block 0.

    Called by the walker once the FUNCTION_ID section's role is
    complete -- either after the self-prepend IDENTITY emits (LOCAL/
    PLT/EXT FUNCTION-category) or when the first IDENTITY at the
    FUNCTION_ID col is itself a BLOCK_V2 (test layouts that skip the
    self-prepend slot). Latches :attr:`pending_header` so the next
    ``Block_Def`` INSTR_REP is suppressed + the next BLOCK_V2 IDENTITY
    is consumed silently as the body block's header.

    The opened BODY block carries ``block_idx == 0`` as a placeholder
    (the pre-allocated entry-block value from the OLD walker
    semantics, retained so jump-table footer blocks without a
    BLOCK_V2 header still get a deterministic block_idx). A
    subsequent BLOCK_V2 header consumed by the walker overwrites it
    with the encoded BLOCK identity counter via
    :func:`set_current_body_block_idx`, which matches the
    InlineJumpEntry target_block_idx scheme (jumps reference the
    BLOCK identity counter, not the positional ordinal).
    """
    close_current_section(state)
    open_section(state, kind=BlockKind.BODY, block_idx=0)
    state.pending_header = True


def enter_new_body_block(state: WalkSectionState) -> None:
    """Close the current BODY block and open a new BODY block.

    Called by the walker at every in-CT runlength-derived block-start
    trigger col (and at non-root CT boundaries). The new BODY block's
    ``block_idx`` carries the encoded BLOCK identity counter once the
    BLOCK_V2 header lands (via :func:`set_current_body_block_idx`);
    pre-overwrite the block keeps the previously-seen ``block_idx``
    so jump-table footer trailers (which never fire a BLOCK_V2) stay
    addressable. Latches :attr:`pending_header` so the upcoming
    header pair is consumed.
    """
    prior_block_idx = state.current_block_idx
    close_current_section(state)
    open_section(state, kind=BlockKind.BODY, block_idx=prior_block_idx)
    state.pending_header = True


def set_current_body_block_idx(
    state: WalkSectionState, *, block_idx: int
) -> None:
    """Overwrite the current BODY section's ``block_idx`` in place.

    Called by the walker when a BLOCK_V2 header is consumed (the
    encoded counter IS the BLOCK identity index used by jumps and
    BLOCK_V2 references). A no-op outside :attr:`BlockKind.BODY`
    (defensive guard against caller drift).
    """
    if state.current_kind is BlockKind.BODY:
        state.current_block_idx = block_idx


def maybe_advance_call_target(state: WalkSectionState, *, col: int) -> None:
    """Advance :attr:`WalkSectionState.current_call_target_idx` at CT boundaries.

    The root CT's start column (``ct_start_cols[0]``) does NOT advance
    the counter -- it stays at 0. Inlined-callee CTs (subsequent
    entries) each advance by one so the FUNCTION-band resolver
    indexes into the right ``call_targets_section`` table.
    """
    if len(state.ct_start_cols) <= 1:
        return
    # Skip the root CT (idx 0); inlined CTs start at idx 1.
    for ct_idx in range(1, len(state.ct_start_cols)):
        if col == state.ct_start_cols[ct_idx]:
            state.current_call_target_idx = ct_idx
            return
