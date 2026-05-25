"""``FunctionNode`` -- the top-level matched/unmatched function row.

Owns the ONE :meth:`BackendFactory.make` call per function-open. The
returned :class:`RenderBackend` is cached on the node so descendant
:class:`VariantNode` / :class:`BlockNode` / :class:`InlineCallNode`
instances share the same instance for the lifetime of the expansion
(per plan section 4: one backend instance per ``FunctionNode.expand``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from tokenizer.inspector._render._protocol import (
        BackendFactory,
        FunctionHandle,
        RenderBackend,
    )

    from ._nodes_variant import VariantNode


__all__ = ["FunctionNode"]


@dataclass
class FunctionNode:
    """Top-level node: one per matched function.

    Holds the :class:`BackendFactory` reference + the typed
    :class:`FunctionHandle`. :meth:`expand` calls
    :meth:`BackendFactory.make` exactly once per open and caches the
    resulting :class:`RenderBackend` on the instance; the descendant
    nodes hold a ref to the same backend.
    """

    factory: "BackendFactory"
    handle: "FunctionHandle"
    is_failed: bool = False
    can_expand: bool = field(default=True, init=False)
    # Per-row horizontal scroll memory; the UI saves the row's current
    # ``scroll_offset.x`` here on manual pan and restores it when the
    # cursor returns to this row. See :mod:`tokenizer.inspector._app`.
    remembered_scroll_x: int = field(default=0, init=False)
    _backend: "Optional[RenderBackend]" = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        """Display name from the typed handle.

        ``_compose_label`` reads this for the row text; the typed
        :class:`FunctionHandle` is the single source of truth.
        """
        return self.handle.name

    def expand(self) -> list["VariantNode"]:
        """Open a fresh :class:`RenderBackend` and surface one
        :class:`VariantNode` per variant the backend reports.

        Idempotent at the UI dispatcher level: re-expansion (after
        collapse) constructs a NEW backend, since the previous one's
        cached descendants belong to a stale expand tree. Plan section
        4 forbids backend reuse across collapse/re-expand.
        """
        from ._nodes_variant import VariantNode

        if self._backend is not None:
            self._backend.close()
        backend = self.factory.make(self.handle)
        self._backend = backend
        return [
            VariantNode(
                factory=self.factory,
                backend=backend,
                variant_idx=rv.variant_idx,
                label_axes=rv.label_axes,
            )
            for rv in backend.variants()
        ]
