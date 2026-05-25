"""Subpackage-level invariants of the row-walk submodule.

Pins:

* :class:`_InsnEmitPolicy` enum surface (R2a pre-pave for R2c).
* :func:`_join_instruction_text` bracket-aware spacing rules
  (W3-11 W4-AMENDED).
* Jump-table footer dispatch: trailing ``Block_V2`` targets emit
  :class:`InlineJumpEntry`, NOT new block headers
  (W3-16 W4-AMENDED + cluster #21 H-3).

Plan reference: ``inspector-followup.md`` Phase R2a + W3-11 + W3-16.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    _CATEGORY_TO_SHIFTED_ID,
)
from tokenizer.inspector._render._batch_decode_backend._row_walk import (
    render_row_blocks,
)
from tokenizer.inspector._render._batch_decode_backend._row_walk._instruction import (
    _join_instruction_text,
)
from tokenizer.inspector._render._batch_decode_backend._row_walk._state import (
    _InsnEmitPolicy,
)
from tokenizer.inspector._render._protocol import (
    AsmLine,
    BlockKind,
    InlineJumpEntry,
)
from tokenizer.tokens import Category

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


JUMP_TABLE = _CATEGORY_TO_SHIFTED_ID[Category.JUMP_TABLE]  # 13


# ---------------------------------------------------------------------------
# _InsnEmitPolicy enum surface
# ---------------------------------------------------------------------------


def test_insn_emit_policy_has_three_members() -> None:
    """The R2a pre-paving carries exactly three policy states; R2c's
    per-instruction collector consumes this surface verbatim.
    """
    assert set(_InsnEmitPolicy) == {
        _InsnEmitPolicy.SILENT_HEADER,
        _InsnEmitPolicy.JUMP_TABLE_FOOTER,
        _InsnEmitPolicy.REAL,
    }


def test_insn_emit_policy_values_are_lower_snake() -> None:
    """Pin the enum string-values so test fixtures can spell them
    without importing the enum.
    """
    assert _InsnEmitPolicy.SILENT_HEADER.value == "silent_header"
    assert _InsnEmitPolicy.JUMP_TABLE_FOOTER.value == "jump_table_footer"
    assert _InsnEmitPolicy.REAL.value == "real"


# ---------------------------------------------------------------------------
# _join_instruction_text: bracket-aware spacing (W3-11 W4-AMENDED)
# ---------------------------------------------------------------------------


def test_join_empty_atoms_returns_empty_string() -> None:
    assert _join_instruction_text([]) == ""


def test_join_simple_tokens_inserts_single_spaces() -> None:
    """Two non-bracket atoms produce ``a b`` (one separator)."""
    assert _join_instruction_text(["mov", "rax"]) == "mov rax"


def test_join_no_space_before_close_bracket() -> None:
    """``]`` consumes no leading space; ``[`` consumes no trailing
    space. ``mov rax , [ rbx ]`` -> ``mov rax, [rbx]``.
    """
    atoms = ["mov", "rax", ",", "[", "rbx", "]"]
    assert _join_instruction_text(atoms) == "mov rax, [rbx]"


def test_join_comma_attaches_to_prior_atom() -> None:
    """``,`` is in ``_NO_SPACE_BEFORE``: it never carries a leading
    space.
    """
    assert _join_instruction_text(["a", ",", "b"]) == "a, b"


def test_join_open_bracket_suppresses_trailing_space() -> None:
    """``[`` is in ``_NO_SPACE_AFTER``: the next atom never carries a
    leading space.
    """
    assert _join_instruction_text(["mem", "[", "x", "]"]) == "mem [x]"


def test_join_leading_position_does_not_emit_leading_space() -> None:
    """The first atom never gets a leading space, regardless of its
    position-in-set memberships.
    """
    assert _join_instruction_text(["[", "x", "]"]) == "[x]"


# ---------------------------------------------------------------------------
# Jump-table footer dispatch (W3-16 W4-AMENDED + cluster #21 H-3)
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


def test_jump_table_footer_block_v2_targets_emit_inline_jump_entries() -> None:
    """Synthetic footer instruction: ``[BLOCK_V2(c=5), JUMP_TABLE(n=7),
    BLOCK_V2(t0=10), BLOCK_V2(t1=11), BLOCK_V2(t2=12), 0]`` with
    ``partial_cut_lengths=[5]``. With ``insn_runlength=[1, 4]`` the
    leading BLOCK_V2 opens block 5 as a single-slot silent-header
    instruction, and the next 4-slot instruction IS the jump-table
    footer (``JUMP_TABLE`` + 3 ``BLOCK_V2`` targets). Per W3-16
    W4-AMENDED + cluster #21 H-3, the JUMP_TABLE IDENTITY arms both
    the per-instruction ``saw_jump_table_this_insn`` flag and the
    block-level ``inside_jump_table_footer_block`` flag; the finalize
    flips :attr:`_InsnEmitPolicy.SILENT_HEADER` -> JUMP_TABLE_FOOTER so
    the footer instruction emits a real AsmLine; the trailing BLOCK_V2
    tokens route as :class:`InlineJumpEntry` openables on THAT same
    AsmLine, NOT as new block headers.
    """
    blocks = _walk(
        tokens=np.asarray(
            [BLOCK_V2, JUMP_TABLE, BLOCK_V2, BLOCK_V2, BLOCK_V2, 0],
            dtype=np.uint16,
        ),
        identities=np.asarray([5, 7, 10, 11, 12], dtype=np.uint16),
        n_axis=0, partial_cut_lengths=[5],
        # BLOCK_V2(c=5) is consumed under FUNCTION_ID -> BODY transition
        # (bare-BLOCK_V2 test layout; no Block_Def precedes it) so it
        # spends 1 slot under FUNCTION_ID and DOES NOT consume an
        # insn_runlength entry. The BODY-section instructions then start
        # at col 1: ONE 4-slot footer instruction (JUMP_TABLE + 3
        # BLOCK_V2 targets).
        insn_runlength=np.asarray([4], dtype=np.uint32),
    )
    assert len(blocks) == 1
    block = blocks[0]
    assert block.kind is BlockKind.BODY
    assert block.block_idx == 5  # the leading BLOCK_V2(c=5) header
    # One AsmLine for the footer instruction; its openables carry the
    # 3 jump-table targets (the silent-header BLOCK_V2 emitted no row).
    items = block.items
    assert len(items) == 1
    line = items[0]
    assert isinstance(line, AsmLine)
    assert line.text == (
        "<jump_table 7> jump block: 10 jump block: 11 jump block: 12"
    )
    assert line.openables == (
        InlineJumpEntry(target_block_idx=10),
        InlineJumpEntry(target_block_idx=11),
        InlineJumpEntry(target_block_idx=12),
    )


def test_jump_table_flag_resets_on_new_body_block() -> None:
    """A subsequent body-block transition clears
    :attr:`WalkSectionState.inside_jump_table_footer_block` so the
    next block's BLOCK_V2 is consumed as a header normally, not as a
    jump-table target.

    Row: ``[BLOCK_V2(c=0), JUMP_TABLE(n=7), BLOCK_V2(t=10)]`` then a
    second CT span ``[BLOCK_V2(c=20), INSTR_REP]``. The flag set in
    the first block must NOT leak into the second block.
    """
    blocks = _walk(
        tokens=np.asarray(
            [BLOCK_V2, JUMP_TABLE, BLOCK_V2, BLOCK_V2, INSTR_REP_TOKEN, 0],
            dtype=np.uint16,
        ),
        identities=np.asarray([0, 7, 10, 20], dtype=np.uint16),
        n_axis=0, partial_cut_lengths=[3, 2],
    )
    # Two CT spans -> two blocks. First block (0) emits the footer
    # AsmLine carrying ``<jump_table 7>`` + InlineJumpEntry openable.
    # Second block (20) carries the INSTR_REP AsmLine -- BLOCK_V2(c=20)
    # was consumed as a header, not silently dropped or routed as
    # InlineJumpEntry.
    assert len(blocks) == 2
    first = blocks[0]
    assert first.block_idx == 0
    # At least one AsmLine in the first block has the footer's text.
    assert any(
        isinstance(it, AsmLine) and "<jump_table 7>" in it.text
        for it in first.items
    )
    second = blocks[1]
    assert second.block_idx == 20
    asm_lines = [it for it in second.items if isinstance(it, AsmLine)]
    assert len(asm_lines) >= 1
