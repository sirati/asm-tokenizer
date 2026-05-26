"""Header-pair absorption for FTL block rendering.

Single concern: every v2 basic block opens with a
``[Block_Def, block_v2:N]`` instruction pair AND every v2
jump-table footer opens with a ``[Block_Def, jump_table:N]`` pair
(see :mod:`tokenizer.fill_constant_candidates`). Both are section
headers consumed silently by the renderer; the second-token kind
discriminates a body block from a jump-table footer block. The N
is the section's identity (jump targets, BLOCK_V2 references key
off body N; jump-table cross-refs key off jump-table N); the
sibling positional index of a block in
:meth:`FunctionTokenList.iter_blocks` only matches N for the
simplest straight-line functions. Wherever the FTL renderer
surfaces a block to the UI (label, preview, line-item stream) it
must read the typed kind + N from the header pair and consume the
pair silently so it never bleeds into the rendered body --
mirroring BatchDecode's :attr:`WalkSectionState.pending_header`
latch (:mod:`._batch_decode_backend._sections`).

The helpers live in the FTL backend (not on :class:`BlockTokenList`)
because the format-version coupling is FTL-specific: FTL is v2-only
(the unified vocab + CSV stream both carry v2 framing), while
:class:`BlockTokenList` itself stays format-agnostic for v1
reconstruction paths.

API surface crossing the boundary:

* :func:`block_header` -- read ``(BlockKind, N)`` from a block. The
  kind is :attr:`BlockKind.BODY` for ``[BLOCK_DEF, BLOCK_V2:N]``
  pairs and :attr:`BlockKind.JUMP_TABLE` for
  ``[BLOCK_DEF, JUMP_TABLE:N]`` pairs. Any other shape raises --
  the gate accepts those two kinds and only those two.
* :func:`body_block_view` -- wrap a block so ``iter_insn`` skips
  the header instruction. The shared row renderer
  (:func:`tokenizer.inspector._render._render_block.render_block`)
  consumes this view through ``iter_insn``; both the FTL backend's
  expand path AND its block-preview path route through that single
  walker so the preview text matches the expanded body byte-for-byte
  (including :func:`substitute_display_chars` substitutions). The
  view works identically for body + jump-table-footer blocks (both
  open with a 1-insn header that must be absorbed).
"""

from __future__ import annotations

from typing import Iterator, Tuple

from tokenizer.inspector._render._protocol import BlockKind
from tokenizer.token_lists import BlockTokenList, InsnTokenList
from tokenizer.tokens import TokenType


__all__ = [
    "BodyBlockView",
    "block_header",
    "body_block_view",
]


# Mapping from the second-token TokenType to the section kind it
# introduces. Single source of truth for the gate's accepted kinds;
# adding a new structural-element kind in the IDENTITY band (e.g. a
# future "global_data_block") only requires extending this table and
# the matching :class:`BlockKind` enum -- no fork in the gate body
# or in downstream label code.
_HEADER_TOKEN_TYPE_TO_KIND: dict[TokenType, BlockKind] = {
    TokenType.BLOCK_V2: BlockKind.BODY,
    TokenType.JUMP_TABLE: BlockKind.JUMP_TABLE,
}


def block_header(block: BlockTokenList) -> Tuple[BlockKind, int]:
    """Read the typed ``(kind, N)`` from the block's opening header pair.

    The writer emits ``[Block_Def, block_v2:N]`` for body blocks and
    ``[Block_Def, jump_table:N]`` for jump-table footer blocks (see
    :mod:`tokenizer.fill_constant_candidates`). Both header tokens
    expose their identity via :attr:`IdentifierToken.id`; the kind
    discriminates which identity namespace ``N`` belongs to (body
    blocks share their N space with :class:`InlineJumpEntry`
    targets, jump-table footers share theirs with
    :class:`Category.JUMP_TABLE` callers).

    Raises :class:`ValueError` when the opening insn does not carry
    a ``BLOCK_DEF`` followed by a recognised structural header token
    -- the FTL stream is corrupt, surface it loudly rather than
    mislabel rows.
    """
    insn_iter = block.iter_insn(transient=True)
    try:
        first_insn = next(iter(insn_iter))
    except StopIteration:
        raise ValueError(
            "block has no instructions; cannot read block header pair"
        )
    tokens = list(first_insn.iter_tokens())
    if (
        len(tokens) < 2
        or tokens[0].token_type is not TokenType.BLOCK_DEF
        or tokens[1].token_type not in _HEADER_TOKEN_TYPE_TO_KIND
    ):
        observed = [t.token_type.name for t in tokens[:2]]
        accepted = sorted(t.name for t in _HEADER_TOKEN_TYPE_TO_KIND)
        raise ValueError(
            f"block does not open with [BLOCK_DEF, {{{', '.join(accepted)}}}] "
            f"header pair (saw {observed!r})"
        )
    kind = _HEADER_TOKEN_TYPE_TO_KIND[tokens[1].token_type]
    return kind, int(tokens[1].id)


class BodyBlockView:
    """Block view whose ``iter_insn`` skips the structural header insn.

    Duck-typed to satisfy the slice of
    :class:`BlockTokenList` that
    :func:`tokenizer.inspector._render._render_block.render_block`
    consumes (``iter_insn``). The wrapper holds a reference to the
    inner :class:`BlockTokenList`; lifetime + read-only semantics
    carry through unchanged.

    Mirrors BatchDecode's silent-header policy
    (:attr:`WalkSectionState.pending_header`): the section content
    starts at the first REAL instruction, not at the
    ``_def block_v2:N`` / ``_def jump_table:N`` pair. The view
    works for both BLOCK_V2 + JUMP_TABLE headers (the structural
    discriminator is :func:`block_header`'s job; absorption is
    kind-agnostic).

    Intentionally does NOT expose a ``to_asm_like`` method: every
    asm-text production in the inspector flows through the shared
    row walker
    (:func:`tokenizer.inspector._render._render_block.render_block`)
    so MEM-bracket / register-list display substitution is applied
    in ONE place. A second raw-token join here would skip that
    substitution and re-introduce the ``mem[`` / ``]mem`` shape into
    the preview.
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


def body_block_view(block: BlockTokenList) -> BodyBlockView:
    """Construct the header-absorbing view; convenience constructor."""
    return BodyBlockView(block)
