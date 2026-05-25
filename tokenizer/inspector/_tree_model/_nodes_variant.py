"""``VariantNode`` -- one per variant of a function.

Carries the parent :class:`RenderBackend` reference + the typed
``variant_idx`` it threads into :meth:`RenderBackend.blocks` /
:meth:`RenderBackend.render_block`. The display label is rendered
on the UI side from :attr:`label_axes` (the typed Mapping shape from
:class:`RenderedVariant`) so the model carries pre-flattened typed
data only and the string-formatting concern lives in :mod:`._app`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional


if TYPE_CHECKING:
    from tokenizer.inspector._render._protocol import (
        BackendFactory,
        RenderBackend,
    )

    from ._nodes_block import BlockNode


__all__ = ["VariantNode"]


@dataclass
class VariantNode:
    """One per variant of a function.

    The ``factory`` ref is propagated downward so any inline-call
    descendant can construct a callee :class:`FunctionNode` against
    the same factory; the backend ref handles the render path.
    ``label_axes`` is the typed positional-axis Mapping the UI layer
    formats into the row label.
    """

    factory: "BackendFactory"
    backend: "RenderBackend"
    variant_idx: int
    label_axes: Mapping[str, Optional[str]]
    is_failed: bool = False
    can_expand: bool = field(default=True, init=False)
    # Per-row horizontal scroll memory; see :mod:`tokenizer.inspector._app._tree_widget`.
    remembered_scroll_x: int = field(default=0, init=False)

    def expand(self) -> list["BlockNode"]:
        """Enumerate the variant's blocks; no parse happens here.

        :meth:`RenderBackend.blocks` is the cache-aware boundary: on
        first call it triggers the variant parse + memoises, subsequent
        calls are cheap. Per plan section 4 the cached result is
        invalidated on :meth:`RenderBackend.close`.
        """
        from ._nodes_block import BlockNode

        return [
            BlockNode(
                factory=self.factory,
                backend=self.backend,
                variant_idx=self.variant_idx,
                kind=rb.kind,
                block_idx=rb.block_idx,
                preview=rb.preview,
            )
            for rb in self.backend.blocks(self.variant_idx)
        ]
