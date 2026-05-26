"""Header-pair absorption for FTL block rendering.

Single concern: every v2 basic block opens with a
``[Block_Def, block_v2:N]`` instruction pair emitted by the writer
(see :mod:`tokenizer.fill_constant_candidates`). The N is the block's
true identity (jump targets, BLOCK_V2 references key off it); the
sibling positional index of a block in
:meth:`FunctionTokenList.iter_blocks` only matches N for the simplest
straight-line functions. Wherever the FTL renderer surfaces a block
to the UI (label, preview, line-item stream) it must read N from the
header pair and consume the pair silently so it never bleeds into
the rendered body — mirroring BatchDecode's
:attr:`WalkSectionState.pending_header` latch
(:mod:`._batch_decode_backend._sections`).

The helpers live in the FTL backend (not on :class:`BlockTokenList`)
because the format-version coupling is FTL-specific: FTL is v2-only
(the unified vocab + CSV stream both carry v2 framing), while
:class:`BlockTokenList` itself stays format-agnostic for v1
reconstruction paths.

API surface crossing the boundary:

* :func:`block_v2_id` -- read N from a block.
* :func:`body_block_view` -- wrap a block so ``iter_insn`` /
  ``to_asm_like`` skip the header instruction. The renderer + the
  :func:`tokenizer.inspector._label.block_preview` helper both consume
  this view; neither knows about the underlying header pair.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from tokenizer.token_lists import BlockTokenList, InsnTokenList
from tokenizer.tokens import TokenType


__all__ = [
    "BodyBlockView",
    "block_v2_id",
    "body_block_view",
]


def block_v2_id(block: BlockTokenList) -> int:
    """Read the v2 block identity N from the block's opening header pair.

    The writer emits ``[Block_Def, block_v2:N]`` as the first
    instruction of every block (see
    :mod:`tokenizer.fill_constant_candidates`). The BLOCK_V2 metatoken
    exposes its identity via :attr:`IdentifierToken.id`; that integer
    IS the block index jump-table footers and ``BLOCK_V2`` references
    key off, so this is the authoritative source for tree labels too.

    Raises :class:`ValueError` when the opening insn does not have
    the expected ``BLOCK_DEF`` + ``BLOCK_V2`` shape -- the FTL stream
    is corrupt, surface it loudly rather than mislabel rows.
    """
    insn_iter = block.iter_insn(transient=True)
    try:
        first_insn = next(iter(insn_iter))
    except StopIteration:
        raise ValueError(
            "block has no instructions; cannot read block_v2 header pair"
        )
    tokens = list(first_insn.iter_tokens())
    if (
        len(tokens) < 2
        or tokens[0].token_type is not TokenType.BLOCK_DEF
        or tokens[1].token_type is not TokenType.BLOCK_V2
    ):
        observed = [t.token_type.name for t in tokens[:2]]
        raise ValueError(
            f"block does not open with [BLOCK_DEF, BLOCK_V2] header pair "
            f"(saw {observed!r})"
        )
    return int(tokens[1].id)


class BodyBlockView:
    """Block view whose ``iter_insn`` skips the v2 header instruction.

    Duck-typed to satisfy the slice of
    :class:`BlockTokenList` that
    :func:`tokenizer.inspector._render._render_block.render_block`
    and :func:`tokenizer.inspector._label.block_preview` consume
    (``iter_insn`` + ``to_asm_like``). The wrapper holds a reference
    to the inner :class:`BlockTokenList`; lifetime + read-only
    semantics carry through unchanged.

    Mirrors BatchDecode's silent-header policy
    (:attr:`WalkSectionState.pending_header`): the BODY section
    starts at the first REAL instruction, not at the
    ``_def block_v2:N`` pair.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: BlockTokenList) -> None:
        self._inner = inner

    def iter_insn(self, transient: bool = False) -> Iterator[InsnTokenList]:
        """Yield every instruction except the leading header insn."""
        first = True
        for insn in self._inner.iter_insn(transient=transient):
            if first:
                first = False
                continue
            yield insn

    def to_asm_like(self) -> str:
        """Re-implement :meth:`BlockTokenList.to_asm_like` over the body.

        Matches the production join (``"; "``) so the preview text the
        UI sees is identical to a header-less block's
        :meth:`to_asm_like` output -- only the leading
        ``"_def block_v2:N; "`` fragment is missing.
        """
        return "; ".join(t.to_asm_like() for t in self.iter_insn(True))


def body_block_view(block: BlockTokenList) -> BodyBlockView:
    """Construct the header-absorbing view; convenience constructor."""
    return BodyBlockView(block)
