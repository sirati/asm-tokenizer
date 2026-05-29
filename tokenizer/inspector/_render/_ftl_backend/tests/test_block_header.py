"""Header-pair absorption + typed-header read for FTL rendering.

Pins four invariants the user-observed mislabeling / crash violated:

* The sibling positional index of a block in
  :meth:`FunctionTokenList.iter_blocks` is NOT in general the
  ``block_v2:N`` identity printed by the writer; the inspector's
  ``Block: <i>`` label MUST come from the latter.
* The opening ``[Block_Def, <header>:N]`` instruction pair is a
  header, not body content -- it must NOT leak into the rendered
  asm-line stream OR into the preview text.
* The lookup that turns a ``RenderedBlock.block_idx`` back into a
  block (in :meth:`FtlBackend.render_block`) keys by ``(kind, N)``,
  so jump-target wires + tree-row IDs share one address space per
  section kind.
* The gate accepts BOTH ``BLOCK_V2`` and ``JUMP_TABLE`` as valid
  second-token kinds: production functions have jump-table footer
  blocks (``[Block_Def, Jump_Table:N, Block_V2:t0, ...]``) emitted
  as full siblings of body blocks (see
  :mod:`tokenizer.fill_constant_candidates`).

Mirrors BatchDecode's :attr:`WalkSectionState.pending_header` policy
on the FTL path.
"""

from __future__ import annotations

from tokenizer.inspector._render._ftl_backend._block_header import (
    BodyBlockView,
    block_header,
    body_block_view,
)
from tokenizer.inspector._render._protocol import BlockKind
from tokenizer.token_lists import BlockTokenList
from tokenizer.token_manager import VocabularyManager


def _build_block_with_header(
    vm: VocabularyManager, n: int, body_insns: list[tuple[str, list]]
) -> BlockTokenList:
    """Build a v2-style block: ``[Block_Def, block_v2:N]`` then body insns.

    ``body_insns`` is a list of ``(insn_str, [Tokens])`` pairs that
    follow the header. Mirrors the writer-side layout in
    :mod:`tokenizer.fill_constant_candidates` so the resulting block
    matches what the FTL reconstructor would parse out of a real CSV.
    """
    blk = BlockTokenList(len(body_insns) + 1, vocab_manager=vm)
    blk.append_as_insn(
        insn_str=f"block 0x{n:x}",
        tokens=[vm.Block_Def(), vm.Block_V2(n)],
    )
    for insn_str, toks in body_insns:
        blk.append_as_insn(insn_str=insn_str, tokens=toks)
    return blk


def _build_jump_table_footer_block(
    vm: VocabularyManager, jt_id: int, target_ns: list[int]
) -> BlockTokenList:
    """Build a writer-shaped jump-table footer block.

    Mirrors :func:`tokenizer.fill_constant_candidates._emit_jump_table_footer_for`:
    one synthetic instruction carrying
    ``[Block_Def, Jump_Table(jt_id), Block_V2(t0), Block_V2(t1), ...]``.
    The trailing ``Block_V2`` tokens are jump-table TARGETS, not
    body-block headers -- the per-block JUMP_TABLE flag (W3-16 on
    BatchDecode; structural in FTL via the gate's typed header) keeps
    them addressed as :class:`InlineJumpEntry` openables downstream.
    """
    blk = BlockTokenList(1, vocab_manager=vm)
    target_tokens = [vm.Block_V2(t) for t in target_ns]
    blk.append_as_insn(
        insn_str=f"jump_table 0x{jt_id:x}",
        tokens=[vm.Block_Def(), vm.Jump_Table(jt_id), *target_tokens],
    )
    return blk


def _vm() -> VocabularyManager:
    """v2 unified vocab; needed because Block_V2 / Block_Def are v2 inners."""
    return VocabularyManager(platform=None, format_version=2)


# ---------------------------------------------------------------------------
# block_header -- typed (kind, N) read from the opening pair
# ---------------------------------------------------------------------------


def test_block_header_reads_kind_and_n_from_body_block_header_pair() -> None:
    vm = _vm()
    blk = _build_block_with_header(vm, n=7, body_insns=[("nop", [vm.Block_Def()])])
    # The body insn here doesn't need to be a "real" non-header insn for
    # this assertion; we're only reading the FIRST insn's tokens.
    assert block_header(blk) == (BlockKind.BODY, 7)


def test_block_header_reads_jump_table_kind_and_id() -> None:
    """User-observed crash on real arm32 exit@exit:thunk:
    ``ValueError("block does not open with [BLOCK_DEF, BLOCK_V2] ...")``
    fired because the gate rejected jump-table footer blocks. The
    gate must accept both BLOCK_V2 + JUMP_TABLE as valid second-token
    kinds and discriminate the section namespace via the returned
    :class:`BlockKind`.
    """
    vm = _vm()
    blk = _build_jump_table_footer_block(vm, jt_id=11, target_ns=[3, 4, 5])
    assert block_header(blk) == (BlockKind.JUMP_TABLE, 11)


def test_block_header_handles_non_sequential_n() -> None:
    """User-observed shape: block_v2 IDs go [0, 1, 2, 4, 3, 5, ...].

    The writer assigns block_v2 identities via
    ``ConstantHandler._emit_block`` / ``resolver.get_block_id`` --
    insertion-ordered into the ``Category.BLOCK`` cache. Forward jumps
    in the source binary can therefore allocate a higher N before a
    later-walked basic block claims a lower N, producing exactly the
    non-monotone sequence observed in the inspector.
    """
    vm = _vm()
    ids = [0, 1, 2, 4, 3, 5]
    blocks = [
        _build_block_with_header(vm, n=n, body_insns=[("nop", [vm.Block_Def()])])
        for n in ids
    ]
    assert [block_header(b) for b in blocks] == [(BlockKind.BODY, n) for n in ids]


def test_block_header_raises_on_missing_header_pair() -> None:
    """A block whose first insn is NOT ``[BLOCK_DEF, <accepted>]`` is
    a corrupt FTL stream -- surface it loudly, don't mislabel.
    """
    import pytest

    vm = _vm()
    blk = BlockTokenList(1, vocab_manager=vm)
    blk.append_as_insn(insn_str="bare", tokens=[vm.Block_Def()])  # no header
    with pytest.raises(ValueError, match="header pair"):
        block_header(blk)


def test_block_header_error_message_lists_accepted_kinds() -> None:
    """The diagnostic message must surface BOTH accepted second-token
    kinds so a future producer can self-diagnose without code-reading.
    """
    import pytest

    vm = _vm()
    blk = BlockTokenList(1, vocab_manager=vm)
    blk.append_as_insn(insn_str="bare", tokens=[vm.Block_Def()])
    with pytest.raises(ValueError) as exc_info:
        block_header(blk)
    msg = str(exc_info.value)
    assert "BLOCK_V2" in msg
    assert "JUMP_TABLE" in msg


# ---------------------------------------------------------------------------
# BodyBlockView (iter_insn header absorption)
# ---------------------------------------------------------------------------


def test_body_block_view_skips_header_insn() -> None:
    """``iter_insn`` yields only the body insns; the header pair is
    consumed silently."""
    vm = _vm()
    blk = _build_block_with_header(
        vm,
        n=3,
        body_insns=[("nop", [vm.Block_Def()]), ("ret", [vm.Block_Def()])],
    )
    view = body_block_view(blk)

    insns = list(view.iter_insn(transient=False))
    # 2 body insns; header insn absorbed.
    assert len(insns) == 2


def test_body_block_view_absorbs_jump_table_header_too() -> None:
    """The header-absorption contract is kind-agnostic: a jump-table
    footer's synthetic 1-insn header must also be skipped so the
    preview / rendered stream does not leak the ``_def jump_table:N``
    fragment.

    A jump-table footer typically has ONLY the header insn (no body
    insns); the view's ``iter_insn`` must therefore yield zero items.
    """
    vm = _vm()
    blk = _build_jump_table_footer_block(vm, jt_id=4, target_ns=[1, 2])
    view = body_block_view(blk)
    assert list(view.iter_insn(transient=False)) == []


def test_body_block_view_holds_bodyblockview_type() -> None:
    """Public constructor returns the documented wrapper class."""
    vm = _vm()
    blk = _build_block_with_header(
        vm, n=0, body_insns=[("nop", [vm.Block_Def()])]
    )
    assert isinstance(body_block_view(blk), BodyBlockView)
