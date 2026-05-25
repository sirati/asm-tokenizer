"""Block-dispatch invariants of :func:`render_row_blocks`.

Pre-allocated entry block (decision #30), mid-row BLOCK_V2 -> inline
jump, n_axis prefix guard, padding-row early-out.

Plan reference: ``inspector-render-backends.md`` §6 + decisions #29 +
#30 + audits B-CRIT-2 / B-CRIT-3 / B-HIGH-5.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.inspector._render._batch_decode_backend._row_walk import (
    render_row_blocks,
)
from tokenizer.inspector._render._protocol import AsmLine, InlineJumpEntry

from ._row_walk_fixtures import (
    BLOCK_V2,
    EMPTY_FID_COUNTS,
    EMPTY_FID_SIDECAR,
    EMPTY_NUMBERS,
    INSTR_REP_TOKEN,
    make_fid_table,
    make_result,
    vocab_stub,
)


def _walk(
    *, tokens: np.ndarray, identities: np.ndarray, n_axis: int,
    partial_cut_lengths: list[int],
):
    """Shorthand: ``render_row_blocks`` with empty number sidecars and
    empty FID table -- the default for block-only dispatch tests.
    """
    numbers_sig, numbers_se = EMPTY_NUMBERS
    return render_row_blocks(
        result=make_result(
            tokens_row=tokens, identities=identities,
            numbers_sig=numbers_sig, numbers_se=numbers_se,
        ),
        row=0, n_axis=n_axis,
        partial_cut_lengths=partial_cut_lengths,
        vocab_manager=vocab_stub(),
        fid_table=make_fid_table(
            per_category_counts=EMPTY_FID_COUNTS,
            sidecar=EMPTY_FID_SIDECAR,
        ),
        line_to_name={}, line_to_provider={},
    )


def test_padding_row_returns_empty_list() -> None:
    """An all-zero token row terminates on the first ``shifted_id == 0``;
    no blocks accumulated -> empty result.
    """
    blocks = _walk(
        tokens=np.zeros(8, dtype=np.uint16),
        identities=np.zeros(0, dtype=np.uint16),
        n_axis=0, partial_cut_lengths=[0],
    )
    assert blocks == []


def test_first_block_v2_overwrites_pre_allocated_entry_block() -> None:
    """``[BLOCK_V2(c=7), INSTR_REP, INSTR_REP, 0]`` with n_axis=0:
    the BLOCK_V2 overwrites the pre-allocated block's index (no flush)
    -> ONE block, block_idx=7, two AsmLines after the header overwrite.
    """
    blocks = _walk(
        tokens=np.asarray(
            [BLOCK_V2, INSTR_REP_TOKEN, INSTR_REP_TOKEN, 0], dtype=np.uint16,
        ),
        identities=np.asarray([7], dtype=np.uint16),
        n_axis=0, partial_cut_lengths=[3],
    )
    assert len(blocks) == 1
    assert blocks[0].block_idx == 7
    assert len(blocks[0].items) == 2
    assert all(isinstance(item, AsmLine) for item in blocks[0].items)


def test_second_block_v2_inside_same_call_target_emits_inline_jump() -> None:
    """A non-``pending_header`` BLOCK_V2 (second in a single call-target
    span) emits :class:`InlineJumpEntry` -- it does NOT open a new
    block; it appends under the current block.
    """
    blocks = _walk(
        tokens=np.asarray(
            [BLOCK_V2, INSTR_REP_TOKEN, BLOCK_V2, INSTR_REP_TOKEN, 0],
            dtype=np.uint16,
        ),
        identities=np.asarray([0, 2], dtype=np.uint16),
        n_axis=0, partial_cut_lengths=[4],
    )
    assert len(blocks) == 1
    assert blocks[0].block_idx == 0
    items = blocks[0].items
    assert len(items) == 3
    assert isinstance(items[0], AsmLine)
    assert isinstance(items[1], InlineJumpEntry)
    assert items[1].target_block_idx == 2
    assert isinstance(items[2], AsmLine)


def test_block_v2_inside_variant_prefix_raises_loud() -> None:
    """BLOCK_V2 IDENTITY tokens NEVER land inside ``[0, n_axis)`` --
    the variant_tokens prefix is pure instruction-rep by row-writer
    contract. A BLOCK_V2 at col < n_axis is a data-integrity violation;
    the walker asserts loud.
    """
    with pytest.raises(AssertionError, match="variant_tokens prefix"):
        _walk(
            tokens=np.asarray(
                [INSTR_REP_TOKEN, BLOCK_V2, INSTR_REP_TOKEN, 0],
                dtype=np.uint16,
            ),
            identities=np.asarray([0], dtype=np.uint16),
            n_axis=2, partial_cut_lengths=[1],
        )


def test_second_call_target_block_v2_flushes_then_opens() -> None:
    """Per decision #29: a per-call-target boundary flips
    ``pending_header``; the NEXT BLOCK_V2 inside the new span is the
    entry header for the inlined callee -> flush the prior block + open
    a new block.

    Row: ``[BLOCK_V2(c=0), INSTR_REP] | [BLOCK_V2(c=5), INSTR_REP, 0]``
    with partial_cut_lengths=[2, 3] (so the boundary lands at col 2).
    """
    blocks = _walk(
        tokens=np.asarray(
            [BLOCK_V2, INSTR_REP_TOKEN, BLOCK_V2, INSTR_REP_TOKEN, 0],
            dtype=np.uint16,
        ),
        identities=np.asarray([0, 5], dtype=np.uint16),
        n_axis=0, partial_cut_lengths=[2, 3],
    )
    # Two blocks: first one (block_idx=0, one AsmLine), second
    # (block_idx=5, one AsmLine).
    assert len(blocks) == 2
    assert blocks[0].block_idx == 0
    assert len(blocks[0].items) == 1
    assert isinstance(blocks[0].items[0], AsmLine)
    assert blocks[1].block_idx == 5
    assert len(blocks[1].items) == 1
    assert isinstance(blocks[1].items[0], AsmLine)


def test_variant_prefix_lands_in_pre_allocated_block() -> None:
    """A row with n_axis=2 variant-tokens prefix followed by a
    BLOCK_V2 header: the prefix's two instr-rep AsmLines accumulate
    into the pre-allocated block (block_idx=0); the BLOCK_V2 then
    overwrites it to the real header counter. AsmLines emitted before
    the overwrite remain attached to the block (decision #30: in-place
    overwrite, no flush).
    """
    blocks = _walk(
        tokens=np.asarray(
            [INSTR_REP_TOKEN, INSTR_REP_TOKEN, BLOCK_V2, INSTR_REP_TOKEN, 0],
            dtype=np.uint16,
        ),
        identities=np.asarray([99], dtype=np.uint16),
        n_axis=2, partial_cut_lengths=[3],
    )
    assert len(blocks) == 1
    assert blocks[0].block_idx == 99
    # 3 AsmLines: 2 prefix + 1 post-header instr-rep.
    assert len(blocks[0].items) == 3
    assert all(isinstance(item, AsmLine) for item in blocks[0].items)
