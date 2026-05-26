"""Header-pair absorption + block_v2 id resolution for FTL rendering.

Pins three invariants the user-observed mislabeling violated:

* The sibling positional index of a block in
  :meth:`FunctionTokenList.iter_blocks` is NOT in general the
  ``block_v2:N`` identity printed by the writer; the inspector's
  ``Block: <i>`` label MUST come from the latter.
* The opening ``[Block_Def, block_v2:N]`` instruction pair is a
  header, not body content -- it must NOT leak into the rendered
  asm-line stream OR into the preview text.
* The lookup that turns a ``RenderedBlock.block_idx`` back into a
  block (in :meth:`FtlBackend.render_block`) keys by ``N``, so
  jump-target wires and tree-row IDs share one address space.

Mirrors BatchDecode's :attr:`WalkSectionState.pending_header` policy
on the FTL path.
"""

from __future__ import annotations

from tokenizer.inspector._render._ftl_backend._block_header import (
    BodyBlockView,
    block_v2_id,
    body_block_view,
)
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


def _vm() -> VocabularyManager:
    """v2 unified vocab; needed because Block_V2 / Block_Def are v2 inners."""
    return VocabularyManager(platform=None, format_version=2)


# ---------------------------------------------------------------------------
# block_v2_id
# ---------------------------------------------------------------------------


def test_block_v2_id_reads_n_from_header_pair() -> None:
    vm = _vm()
    blk = _build_block_with_header(vm, n=7, body_insns=[("nop", [vm.Block_Def()])])
    # The body insn here doesn't need to be a "real" non-header insn for
    # this assertion; we're only reading the FIRST insn's tokens.
    assert block_v2_id(blk) == 7


def test_block_v2_id_handles_non_sequential_n() -> None:
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
    assert [block_v2_id(b) for b in blocks] == ids


def test_block_v2_id_raises_on_missing_header_pair() -> None:
    """A block whose first insn is NOT ``[BLOCK_DEF, BLOCK_V2]`` is
    a corrupt FTL stream -- surface it loudly, don't mislabel.
    """
    import pytest

    vm = _vm()
    blk = BlockTokenList(1, vocab_manager=vm)
    blk.append_as_insn(insn_str="bare", tokens=[vm.Block_Def()])  # no BLOCK_V2
    with pytest.raises(ValueError, match="header pair"):
        block_v2_id(blk)


# ---------------------------------------------------------------------------
# BodyBlockView (iter_insn + to_asm_like absorption)
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


def test_body_block_view_to_asm_like_omits_header_text() -> None:
    """``to_asm_like`` must not include the ``_def block_v2:N`` fragment."""
    vm = _vm()
    blk = _build_block_with_header(
        vm,
        n=4,
        body_insns=[("nop", [vm.Block_Def()])],
    )
    asm = body_block_view(blk).to_asm_like()
    # Single body insn carrying a BLOCK_DEF token -> renders as "_def"
    # but the header's "_def block_v2:4" is gone. (The body insn's
    # "_def" content here is just a placeholder; the assertion is that
    # ``block_v2:4`` does NOT appear at all because the header was the
    # only carrier of the BLOCK_V2 token.)
    assert "block_v2:" not in asm


def test_body_block_view_is_duck_compatible_with_block_preview() -> None:
    """The renderer-agnostic :func:`block_preview` reads only
    ``to_asm_like``; a BodyBlockView must satisfy that slice so the
    inspector's preview path stays one-concern."""
    from tokenizer.inspector._label import block_preview

    vm = _vm()
    blk = _build_block_with_header(
        vm,
        n=9,
        body_insns=[("nop", [vm.Block_Def()])],
    )
    preview = block_preview(body_block_view(blk))
    # Same omission contract as the to_asm_like test, but routed via
    # the production preview helper so the wrapper's duck-typed shape
    # stays pinned.
    assert "block_v2:9" not in preview


def test_body_block_view_holds_bodyblockview_type() -> None:
    """Public constructor returns the documented wrapper class."""
    vm = _vm()
    blk = _build_block_with_header(
        vm, n=0, body_insns=[("nop", [vm.Block_Def()])]
    )
    assert isinstance(body_block_view(blk), BodyBlockView)
