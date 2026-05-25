"""``BlockNode`` + ``InlineJumpNode`` -- block-level tree nodes.

Both expansions call :meth:`RenderBackend.render_block` on the parent
backend and translate the returned :class:`LineItem` stream into the
model's leaf/child nodes. :class:`InlineJumpNode` reuses the same
translation by targeting a different ``block_idx`` within the SAME
variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


from tokenizer.inspector._render._protocol import BlockKind


if TYPE_CHECKING:
    from tokenizer.inspector._render._protocol import (
        BackendFactory,
        RenderBackend,
    )


__all__ = [
    "BlockNode",
    "InlineJumpNode",
]


@dataclass
class BlockNode:
    """One per section within a variant.

    A section is one of :attr:`BlockKind.VARIANT_HEADER`,
    :attr:`BlockKind.FUNCTION_ID`, or :attr:`BlockKind.BODY` (per the
    :class:`RenderedBlock` Protocol contract). Carries the factory ref
    so inline-call descendants can spawn callee :class:`FunctionNode`
    s against the same factory; carries the backend ref so the render
    call lands on the parent's per-function backend instance.
    """

    factory: "BackendFactory"
    backend: "RenderBackend"
    variant_idx: int
    kind: BlockKind
    block_idx: int
    preview: str
    is_failed: bool = False
    can_expand: bool = field(default=True, init=False)
    # Per-row horizontal scroll memory; see :mod:`tokenizer.inspector._app`.
    remembered_scroll_x: int = field(default=0, init=False)

    def expand(self) -> list:
        return _translate_line_items(
            factory=self.factory,
            backend=self.backend,
            variant_idx=self.variant_idx,
            kind=self.kind,
            block_idx=self.block_idx,
        )


@dataclass
class InlineJumpNode:
    """Inline jump to another block in the SAME variant.

    Expansion renders the target BODY block in place via
    :meth:`RenderBackend.render_block` (jumps always target BODY
    sections); the line-item translation is shared with
    :class:`BlockNode` so jump-target rendering reuses the
    block-expansion code path.
    """

    factory: "BackendFactory"
    backend: "RenderBackend"
    variant_idx: int
    target_block_idx: int
    is_failed: bool = False
    can_expand: bool = field(default=True, init=False)
    # Per-row horizontal scroll memory; see :mod:`tokenizer.inspector._app`.
    remembered_scroll_x: int = field(default=0, init=False)

    def expand(self) -> list:
        return _translate_line_items(
            factory=self.factory,
            backend=self.backend,
            variant_idx=self.variant_idx,
            kind=BlockKind.BODY,
            block_idx=self.target_block_idx,
        )


def _translate_line_items(
    *,
    factory: "BackendFactory",
    backend: "RenderBackend",
    variant_idx: int,
    kind: BlockKind,
    block_idx: int,
) -> list:
    """Translate one block's :class:`AsmLine` stream into model leaves.

    Shared by :meth:`BlockNode.expand` and :meth:`InlineJumpNode.expand`.
    Post-R2 contract (plan W3-2 W4-amended cluster #3): the backend's
    ``render_block`` stream contains ONLY :class:`AsmLine` items;
    per-instruction call / jump / number-precision sidecars flow
    through ``AsmLine.openables`` and are surfaced by
    :meth:`AsmLeaf.expand`, not as sibling top-level rows. The parent
    BlockNode's ``factory`` / ``backend`` / ``variant_idx`` are
    threaded onto each leaf so its lazy expand can spawn an
    :class:`InlineCallNode` / :class:`InlineJumpNode` with the same
    model-graph context.

    Non-AsmLine items raise -- that signals a drift between the
    Protocol's narrowed LineItem and the model layer; loud crash is
    the right contract.
    """
    # Lazy import so the package can be split by concern without
    # tripping circular imports.
    from tokenizer.inspector._render._protocol import AsmLine

    from ._nodes_leaf import AsmLeaf

    items = backend.render_block(variant_idx, kind, block_idx)

    out: list = []
    for item in items:
        if not isinstance(item, AsmLine):
            # Post-R2 the LineItem stream is AsmLine-only; any other
            # type signals a producer/consumer contract drift.
            raise TypeError(
                f"unknown render line item type: {type(item).__name__}"
            )
        out.append(
            AsmLeaf(
                text=item.text,
                openables=item.openables,
                factory=factory,
                backend=backend,
                variant_idx=variant_idx,
            )
        )
    return out
