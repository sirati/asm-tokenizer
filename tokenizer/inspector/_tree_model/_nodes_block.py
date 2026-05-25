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
    """Translate one block's :class:`LineItem` stream into model nodes.

    Shared by :meth:`BlockNode.expand` and :meth:`InlineJumpNode.expand`.
    The discriminator is the dataclass type (per plan section 4: no
    string-typed prefix). Unknown item types raise -- that signals a
    drift between the protocol union and the model layer that should
    crash loud rather than render an empty leaf.
    """
    # Lazy imports so the package can be split by concern without
    # tripping circular imports (InlineCallNode imports FunctionNode).
    from tokenizer.inspector._render._protocol import (
        AsmLine,
        FunctionHandle,
        InlineCallEntry,
        InlineJumpEntry,
    )

    from ._nodes_call import InlineCallNode
    from ._nodes_leaf import AsmLeaf

    items = backend.render_block(variant_idx, kind, block_idx)

    out: list = []
    for item in items:
        if isinstance(item, AsmLine):
            out.append(AsmLeaf(text=item.text))
            continue
        if isinstance(item, InlineCallEntry):
            # Derive the callee FunctionHandle from the typed pointer
            # spec on the entry. When the pointer is None the call is
            # non-expandable; InlineCallNode.can_expand gates on that.
            spec = item.callee_section_pointer
            callee_handle = (
                None
                if spec is None
                else FunctionHandle(
                    arm=spec.arm, idx=spec.idx, name=item.callee_name
                )
            )
            out.append(
                InlineCallNode(
                    factory=factory,
                    kind=item.kind,
                    counter_id=item.counter_id,
                    callee_name=item.callee_name,
                    callee_handle=callee_handle,
                    variant_idx=item.variant_idx,
                    provider=item.provider,
                )
            )
            continue
        if isinstance(item, InlineJumpEntry):
            out.append(
                InlineJumpNode(
                    factory=factory,
                    backend=backend,
                    variant_idx=variant_idx,
                    target_block_idx=item.target_block_idx,
                )
            )
            continue
        # Closed LineItem union; any miss is a render/model drift.
        raise TypeError(
            f"unknown render line item type: {type(item).__name__}"
        )
    return out
