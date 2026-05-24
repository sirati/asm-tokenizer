"""``VariantNode`` -- one per variant of a function.

Owns the per-variant single-parse pipeline (plan D2 + the "no
re-parsing in call chains" rule from CLAUDE.md): when expanded, parses
the :class:`FunctionTokenList` exactly once, materialises every block
as a distinct :class:`BlockTokenList` (``transient=False`` because
each :class:`BlockNode` stashes its own block), and prebuilds the
two section-level invariants the render layer consumes
(``kind_to_called_idx`` from the :class:`Section`, ``variant_pins``
from the :class:`VariantBlock`). The prebuilt state lives in
:class:`VariantContext`, threaded to every descendant
:class:`BlockNode` / :class:`InlineJumpNode` so :func:`render_block`
sees ONE pre-parsed block + ONE ready table per call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from tokenizer.aligned_data.call_target_type import CallTargetType

from ._context import DecodeContext


if TYPE_CHECKING:
    from tokenizer.aligned_data.loader.function_data import FunctionData
    from tokenizer.aligned_data.loader.session import BinarySession
    from tokenizer.aligned_data.matched_sections_bin import Section, VariantBlock
    from tokenizer.token_lists import BlockTokenList
    from tokenizer.token_manager import VocabularyManager

    from ._nodes_block import BlockNode


__all__ = [
    "VariantContext",
    "VariantNode",
]


@dataclass(frozen=True)
class VariantContext:
    """Per-variant invariants prebuilt once at :meth:`VariantNode.expand`.

    The render layer used to rebuild ``kind_to_called_idx`` (per-kind
    index lists into :attr:`Section.call_targets`) and ``variant_pins``
    (per-:class:`VariantBlock` ``called_idx -> section_variant_index``
    map) on every block expansion -- once per block per variant. These
    are variant-level invariants; building them once and threading the
    same Mapping references to every descendant
    :class:`BlockNode` / :class:`InlineJumpNode` collapses the work to
    one build per variant-open.

    :attr:`blocks` is the materialised list of :class:`BlockTokenList`
    views over the variant's body, parsed once with
    ``transient=False`` so each entry is a distinct stash-safe view
    (the ``transient=True`` form reuses one mutable object across
    iterations; storing those in a list would point every entry at the
    same final-block content).
    """

    section: "Section"
    variant_block: "VariantBlock"
    blocks: tuple["BlockTokenList", ...]
    kind_to_called_idx: Mapping[CallTargetType, list[int]]
    variant_pins: Mapping[int, int]


def _build_variant_context(
    function_data: "FunctionData",
    section: "Section",
    variant_block: "VariantBlock",
    vocab_manager: "VocabularyManager",
) -> VariantContext:
    """One-shot parse: FTL -> blocks list + section invariants.

    Lazy-imports the renderer helpers
    (:func:`_kind_to_called_idx`,
    :func:`_variant_index_for_called_idx`) so the tree-model package
    loads even before ``_render.py`` does.
    """
    from tokenizer.function_token_list import FunctionTokenList

    from .._render import (
        _kind_to_called_idx,
        _variant_index_for_called_idx,
    )

    ftl = FunctionTokenList.reconstruct_func_from_raw_bytes(
        function_data.tokens,
        function_data.block_runlength,
        function_data.insn_runlength,
        vocab_manager=vocab_manager,
    )
    # ``transient=False`` -- each BlockNode stashes its own block, so
    # we need distinct objects (the ``transient=True`` form reuses one
    # mutable view across iterations).
    blocks = tuple(ftl.iter_blocks(transient=False))
    return VariantContext(
        section=section,
        variant_block=variant_block,
        blocks=blocks,
        kind_to_called_idx=_kind_to_called_idx(section),
        variant_pins=_variant_index_for_called_idx(variant_block),
    )


@dataclass(frozen=True)
class VariantNode:
    """One per variant of a function -- wraps the per-variant FunctionData.

    No ``batch_decode`` call: the body was loaded as part of the parent
    :class:`FunctionNode`'s batch.
    """

    function_data: "FunctionData"
    section: "Section"
    variant_block: "VariantBlock"
    batch_row_idx: int
    label: str
    decode_context: DecodeContext
    is_failed: bool = False
    can_expand: bool = field(default=True, init=False)

    def expand(
        self,
        session: "BinarySession",
        *,
        vocab_manager: "VocabularyManager | None" = None,
    ) -> list["BlockNode"]:
        """Parse FTL + section invariants once, then enumerate blocks.

        The render layer below this point sees ONE pre-parsed
        :class:`BlockTokenList` + ONE pair of ready tables per call --
        no re-parsing, no per-block ``iter_blocks`` walk.
        """
        from .._label import block_preview
        from ._nodes_block import BlockNode

        context = _build_variant_context(
            self.function_data,
            self.section,
            self.variant_block,
            self.decode_context.vocab_manager,
        )
        blocks: list[BlockNode] = []
        for block_idx, block in enumerate(context.blocks):
            blocks.append(
                BlockNode(
                    block_idx=block_idx,
                    batch_row_idx=self.batch_row_idx,
                    preview=block_preview(block),
                    decode_context=self.decode_context,
                    variant_context=context,
                )
            )
        return blocks
