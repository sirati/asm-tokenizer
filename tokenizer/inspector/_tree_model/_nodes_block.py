"""``BlockNode`` + ``InlineJumpNode`` -- block-level tree nodes.

Both consume the parent variant's prebuilt :class:`VariantContext`
(blocks list + section invariants) so no re-parse happens at expand
time. :class:`InlineJumpNode` renders the target block in-place by
routing through the same body-expansion helper as
:class:`BlockNode`, so jump-target rendering shares its
implementation with a regular block expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._context import DecodeContext
from ._nodes_variant import VariantContext


if TYPE_CHECKING:
    from tokenizer.aligned_data.loader.session import BinarySession
    from tokenizer.token_manager import VocabularyManager


__all__ = [
    "BlockNode",
    "InlineJumpNode",
]


@dataclass(frozen=True)
class BlockNode:
    """One per block within a variant body.

    Expansion lists the block's asm-like lines, inline calls, and
    inline jumps. NO ``batch_decode`` call -- the parent variant's
    :class:`VariantContext` already holds the parsed
    :class:`BlockTokenList`.
    """

    block_idx: int
    batch_row_idx: int
    preview: str
    decode_context: DecodeContext
    variant_context: VariantContext
    is_failed: bool = False
    can_expand: bool = field(default=True, init=False)

    def expand(
        self,
        session: "BinarySession",
        *,
        vocab_manager: "VocabularyManager | None" = None,
    ) -> list:
        """Map this block's render items into model nodes."""
        return _expand_block_body(
            block_idx=self.block_idx,
            batch_row_idx=self.batch_row_idx,
            decode_context=self.decode_context,
            variant_context=self.variant_context,
        )


@dataclass(frozen=True)
class InlineJumpNode:
    """Inline jump to another block in the SAME variant.

    Expansion renders the target block in place -- no decode. Routes
    through :func:`_expand_block_body` so jump-target rendering shares
    its implementation with :class:`BlockNode`.
    """

    batch_row_idx: int
    target_block_idx: int
    decode_context: DecodeContext
    variant_context: VariantContext
    is_failed: bool = False
    can_expand: bool = field(default=True, init=False)

    def expand(
        self,
        session: "BinarySession",
        *,
        vocab_manager: "VocabularyManager | None" = None,
    ) -> list:
        return _expand_block_body(
            block_idx=self.target_block_idx,
            batch_row_idx=self.batch_row_idx,
            decode_context=self.decode_context,
            variant_context=self.variant_context,
        )


def _expand_block_body(
    *,
    block_idx: int,
    batch_row_idx: int,
    decode_context: DecodeContext,
    variant_context: VariantContext,
) -> list:
    """Render one block + lift its line items into model nodes.

    Shared by :meth:`BlockNode.expand` and :meth:`InlineJumpNode.expand`
    so the rendering path is single-sourced (an InlineJump is just
    "render the target block").
    """
    from .._render import (
        AsmLine,
        InlineCallEntry,
        InlineJumpEntry,
        render_block,
    )

    from ._nodes_call import InlineCallNode
    from ._nodes_leaf import AsmLeaf

    block = variant_context.blocks[block_idx]
    items = render_block(
        block=block,
        section=variant_context.section,
        kind_to_called_idx=variant_context.kind_to_called_idx,
        variant_pins=variant_context.variant_pins,
        line_to_name=decode_context.line_to_name,
        callee_arm_resolver=decode_context.callee_arm_resolver,
    )

    out: list = []
    for item in items:
        if isinstance(item, AsmLine):
            out.append(AsmLeaf(text=item.text))
        elif isinstance(item, InlineCallEntry):
            out.append(
                InlineCallNode(
                    kind=item.kind,
                    counter_id=item.counter_id,
                    callee_name=item.callee_name,
                    callee_section_pointer=item.callee_section_pointer,
                    variant_idx=item.variant_idx,
                    provider=item.provider,
                    decode_context=decode_context,
                )
            )
        elif isinstance(item, InlineJumpEntry):
            out.append(
                InlineJumpNode(
                    batch_row_idx=batch_row_idx,
                    target_block_idx=item.target_block_idx,
                    decode_context=decode_context,
                    variant_context=variant_context,
                )
            )
        else:
            # Unknown line-item kind = render/model contract drift.
            raise TypeError(
                f"unknown render line item type: {type(item).__name__}"
            )
    return out
