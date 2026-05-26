"""Tests for the inline-jump openable resolvability gate.

Pins the post-walk filter contract that prevents the BatchDecodeBackend
inspector from KeyError'ing on jump-table targets whose addresses have
no body block in the current variant. Two surfaces under test:

* Pure :func:`filter_unresolvable_jump_openables` on hand-built
  :class:`RowSection` lists (unit-level pins for the filter logic).
* :func:`render_row_blocks` end-to-end -- the live TUI scenario
  ``▼ Jump table: 0 <jump_table 0> jump block: 1`` reproduced via a
  synthetic jump-table footer whose target id has no body block.

Plan + project rules: single-concern (filter logic lives in
:mod:`.._jump_validity`, plumbed through :mod:`.._row_walk._driver`'s
end-of-walk return); no parallel indexing (body block_idxs come from
the walk's own sections, never from a side cache); typed
discriminator (``isinstance(InlineJumpEntry)`` on the Openable union,
no string kind).
"""

from __future__ import annotations

import numpy as np

from tokenizer.inspector._render._batch_decode_backend._jump_validity import (
    filter_unresolvable_jump_openables,
)
from tokenizer.inspector._render._batch_decode_backend._row_walk import (
    render_row_blocks,
)
from tokenizer.inspector._render._batch_decode_backend._sections import (
    RowSection,
)
from tokenizer.inspector._render._protocol import (
    AsmLine,
    BlockKind,
    InlineCallEntry,
    InlineJumpEntry,
)
from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX

from ._row_walk_fixtures import (
    BLOCK_V2,
    EMPTY_FID_COUNTS,
    EMPTY_FID_SIDECAR,
    EMPTY_NUMBERS,
    INSTR_REP_TOKEN,
    NULL_CALLEE_RESOLVER,
    make_fid_table,
    make_result,
    vocab_stub,
)
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    _CATEGORY_TO_SHIFTED_ID,
)
from tokenizer.tokens import Category


JUMP_TABLE = _CATEGORY_TO_SHIFTED_ID[Category.JUMP_TABLE]


# ---------------------------------------------------------------------------
# Unit-level pins on filter_unresolvable_jump_openables
# ---------------------------------------------------------------------------


def _body(block_idx: int, items) -> RowSection:
    return RowSection(kind=BlockKind.BODY, block_idx=block_idx, items=list(items))


def test_filter_drops_inline_jump_to_missing_body_block() -> None:
    """The live TUI failure: a jump_table footer references a BLOCK
    identity that has no body block in this row. The pre-fix code
    would propagate the openable; the gate now drops it so the
    expand path never lands on :meth:`render_block(BODY, missing)`.
    """
    sections = [
        _body(
            0,
            [AsmLine(text="jump block: 1", openables=(InlineJumpEntry(1),))],
        ),
    ]
    out = filter_unresolvable_jump_openables(sections)
    assert len(out) == 1
    asm = out[0].items[0]
    assert isinstance(asm, AsmLine)
    # Text preserved (so the row's diagnostic content stays visible);
    # openable dropped because target_block_idx=1 has no BODY section.
    assert asm.text == "jump block: 1"
    assert asm.openables == ()


def test_filter_preserves_inline_jump_to_existing_body_block() -> None:
    """Jumps whose target matches a BODY section in the same row
    survive the gate -- this is the normal intra-function jump
    case and MUST not regress.
    """
    sections = [
        _body(
            0,
            [AsmLine(text="jump block: 0", openables=(InlineJumpEntry(0),))],
        ),
        _body(
            1,
            [AsmLine(text="jump block: 0", openables=(InlineJumpEntry(0),))],
        ),
    ]
    out = filter_unresolvable_jump_openables(sections)
    # Identity-preserving when nothing changes (no rebuilt copies).
    assert out is sections
    # And the openables tuple is intact.
    assert out[0].items[0].openables == (InlineJumpEntry(0),)


def test_filter_preserves_inline_call_openables_untouched() -> None:
    """Non-jump openables (:class:`InlineCallEntry`,
    :class:`InlineNumberPrecisionEntry`) are out of scope for the
    gate -- it filters jump targets only. A jump-to-missing in the
    SAME line drops the jump but keeps the call untouched.
    """
    call = InlineCallEntry(
        kind=CallTargetType.LOCAL, counter_id=0, callee_name="x",
        callee_section_pointer=None, variant_idx=MISSING_VARIANT_INDEX,
        provider=None,
    )
    sections = [
        _body(
            0,
            [
                AsmLine(
                    text="call x; jump block: 99",
                    openables=(call, InlineJumpEntry(99)),
                ),
            ],
        ),
    ]
    out = filter_unresolvable_jump_openables(sections)
    asm = out[0].items[0]
    assert asm.openables == (call,)


def test_filter_mixed_resolvable_and_phantom_targets_on_same_line() -> None:
    """A jump-table footer with mixed valid + phantom targets: the
    gate filters each entry independently; the surviving openables
    keep their order matching the original tuple ordering.
    """
    sections = [
        _body(0, []),
        _body(2, []),
        RowSection(
            kind=BlockKind.JUMP_TABLE, block_idx=7,
            items=[
                AsmLine(
                    text="<jump_table 7> jump block: 0 jump block: 1 jump block: 2",
                    openables=(
                        InlineJumpEntry(0),  # resolvable -- body 0 exists
                        InlineJumpEntry(1),  # phantom    -- no body 1
                        InlineJumpEntry(2),  # resolvable -- body 2 exists
                    ),
                ),
            ],
        ),
    ]
    out = filter_unresolvable_jump_openables(sections)
    asm = out[2].items[0]
    assert asm.openables == (InlineJumpEntry(0), InlineJumpEntry(2))


def test_filter_handles_empty_sections() -> None:
    """The degenerate input (no sections) must short-circuit cleanly."""
    assert filter_unresolvable_jump_openables([]) == []


def test_filter_handles_section_with_no_jump_openables() -> None:
    """An :class:`AsmLine` carrying no openables (or only non-jump
    openables) is returned by reference -- no rebuilt frozen
    dataclass copy.
    """
    line = AsmLine(text="add rax, 1", openables=())
    sections = [_body(0, [line])]
    out = filter_unresolvable_jump_openables(sections)
    assert out is sections


# ---------------------------------------------------------------------------
# End-to-end: render_row_blocks applies the gate
# ---------------------------------------------------------------------------


def _walk(
    *, tokens: np.ndarray, identities: np.ndarray, n_axis: int,
    partial_cut_lengths: list[int],
    block_runlength: np.ndarray | None = None,
    insn_runlength: np.ndarray | None = None,
):
    numbers_sig, numbers_se = EMPTY_NUMBERS
    return render_row_blocks(
        result=make_result(
            tokens_row=tokens, identities=identities,
            numbers_sig=numbers_sig, numbers_se=numbers_se,
            block_runlength=block_runlength,
            insn_runlength=insn_runlength,
        ),
        row=0, n_axis=n_axis,
        partial_cut_lengths=partial_cut_lengths,
        call_targets_per_ct=[[] for _ in partial_cut_lengths],
        vocab_manager=vocab_stub(),
        fid_table=make_fid_table(
            per_category_counts=EMPTY_FID_COUNTS,
            sidecar=EMPTY_FID_SIDECAR,
        ),
        line_to_name={}, line_to_provider={},
        callee_arm_resolver=NULL_CALLEE_RESOLVER,
    )


def test_live_tui_failure_repro_jump_table_target_with_no_body_block() -> None:
    """Reproduces the live TUI KeyError exactly:

    ``▼ Jump table: 0   <jump_table 0> jump block: 1``
    ``└── [*] jump block: 1``
    ``    └── KeyError("...no section kind=BlockKind.BODY block_idx=1...")``

    A jump_table footer whose single BLOCK_V2 target (id=1) has no
    corresponding body block in this variant. Layout: one body block
    (id=0) + one jump-table footer (jt_id=0) referencing target id=1.

    Pre-fix: the row walker emits :class:`InlineJumpEntry(1)` on the
    footer's AsmLine; clicking it constructs
    :class:`InlineJumpNode(target_block_idx=1)`; expand calls
    :meth:`render_block(variant_idx, BODY, 1)` which has no such
    section and KeyError'd.

    Post-fix: the post-walk resolvability gate drops the openable
    (BODY block_idx=1 is not in the row's BODY set
    ``{0}``); the AsmLeaf's ``can_expand`` flips to False so the UI
    never asks the backend to render a missing block.
    """
    # Layout mirrors test_jump_table_footer_opens_jump_table_section_via_runlength_boundary
    # but with the JT target id (1) NOT matching any body block in the
    # row. block_runlength=[1, 1] gives two blocks; the second block's
    # JUMP_TABLE IDENTITY opens it as the footer section.
    blocks = _walk(
        tokens=np.asarray(
            [
                BLOCK_V2,                  # body header (c=0) -- 1 slot
                INSTR_REP_TOKEN,           # body insn        -- 1 slot
                INSTR_REP_TOKEN,           # Block_Def carrier (silent header)
                JUMP_TABLE, BLOCK_V2,      # JT footer (jt_id=0, target=1)
                0,
            ],
            dtype=np.uint16,
        ),
        identities=np.asarray([0, 0, 1], dtype=np.uint16),
        n_axis=0, partial_cut_lengths=[5],
        block_runlength=np.asarray([1, 1], dtype=np.uint32),
        # body insn = 1 slot; JT footer insn = 3 slots (Block_Def
        # silent-header carrier + JUMP_TABLE + 1 BLOCK_V2 target).
        insn_runlength=np.asarray([1, 3], dtype=np.uint32),
    )
    # Two sections: BODY(block_idx=0) + JUMP_TABLE(block_idx=0).
    kinds_and_idxs = [(b.kind, b.block_idx) for b in blocks]
    assert (BlockKind.BODY, 0) in kinds_and_idxs
    assert (BlockKind.JUMP_TABLE, 0) in kinds_and_idxs
    # The JUMP_TABLE section's AsmLine carries the placeholder text
    # but NO openable (target_block_idx=1 has no body block in {0}).
    jt_section = next(b for b in blocks if b.kind is BlockKind.JUMP_TABLE)
    asm_lines = [it for it in jt_section.items if isinstance(it, AsmLine)]
    assert len(asm_lines) == 1
    line = asm_lines[0]
    assert "<jump_table 0>" in line.text
    assert "jump block: 1" in line.text
    # Critical post-fix invariant: no openable references missing block.
    for openable in line.openables:
        if isinstance(openable, InlineJumpEntry):
            assert openable.target_block_idx == 0, (
                f"InlineJumpEntry(target_block_idx={openable.target_block_idx})"
                f" survived the resolvability gate but has no BODY section "
                f"in this row's sections; the gate should have dropped it."
            )


def test_jump_table_footer_with_resolvable_target_preserves_openable() -> None:
    """Companion to the live-TUI repro: same layout shape but the JT
    target id MATCHES a body block in the row, so the gate keeps the
    openable. Pins that the gate's "drop" path is not over-eager.
    """
    blocks = _walk(
        tokens=np.asarray(
            [
                BLOCK_V2,                  # body header (c=0) -- 1 slot
                INSTR_REP_TOKEN,           # body insn        -- 1 slot
                INSTR_REP_TOKEN,           # Block_Def carrier (silent header)
                JUMP_TABLE, BLOCK_V2,      # JT footer (jt_id=0, target=0)
                0,
            ],
            dtype=np.uint16,
        ),
        identities=np.asarray([0, 0, 0], dtype=np.uint16),
        n_axis=0, partial_cut_lengths=[5],
        block_runlength=np.asarray([1, 1], dtype=np.uint32),
        insn_runlength=np.asarray([1, 3], dtype=np.uint32),
    )
    jt_section = next(b for b in blocks if b.kind is BlockKind.JUMP_TABLE)
    asm_lines = [it for it in jt_section.items if isinstance(it, AsmLine)]
    assert len(asm_lines) == 1
    # The openable survives because target_block_idx=0 == BODY(0).
    assert asm_lines[0].openables == (InlineJumpEntry(target_block_idx=0),)
