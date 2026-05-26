"""Post-mount :class:`NodePath` matcher driving capture-on-rebuild
re-expansion.

Single concern: decide -- given a freshly-mounted :class:`TreeNode`
and a :class:`CapturedExpandState` snapshot -- whether the node
should be auto-expanded so its previously-open descendants surface
again. The matcher has two tracks:

* **Full-path equality** -- the node's full :class:`NodePath`
  (including every :class:`VariantGroupKey` ancestor) hits the
  captured ``full_paths`` set. This is the "regroup preserved this
  exact group structure" channel.
* **Logical-path projection** -- content rows match by their
  group-elided path. :class:`VariantGroupNode` matches are
  additionally gated by a model-descendancy check: an unrelated
  sibling group at the same logical level (e.g. a different
  arch bucket after a regroup) does NOT auto-expand. The
  descendancy walker reads off the in-memory
  :attr:`VariantGroupNode.children` so it works BEFORE any tree
  node has been mounted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokenizer.inspector._tree_model import VariantNode

from .._order import VariantGroupNode
from ._keys import NodeKey, NodePath
from ._walk import (
    CapturedExpandState,
    logical_path_of,
    node_path_for,
    variant_key_via_backend,
)


if TYPE_CHECKING:
    from textual.widgets._tree import TreeNode

    from tokenizer.inspector._tree_model import Node


__all__ = ["should_expand_for_capture"]


def should_expand_for_capture(
    tree_node: "TreeNode[Node]",
    captured: CapturedExpandState,
) -> bool:
    """Decide whether ``tree_node`` should be auto-expanded post-mount.

    Two-track match:

    1. **Full-path equality** -- if the new tree node's full path
       (including every :class:`VariantGroupKey` ancestor) is in
       :attr:`captured.full_paths`, the user previously had this
       EXACT row expanded (e.g. a regroup with the same axes
       preserved the group structure). Re-expand.

    2. **Logical-path projection** -- content rows (variants,
       blocks, calls, ...) survive regroups that re-route them
       under a different group ancestry, so the logical-path
       comparison runs against the captured set with group keys
       elided. For a :class:`VariantGroupNode` this match is
       gated by a model-descendancy check (see
       :func:`_group_leads_to_captured`): the group expands only
       when one of its model descendants is on the path to a
       captured deeper row, so unrelated groups at the same
       logical level do NOT auto-expand.

    Non-group nodes have unique-per-parent identity, so the
    logical-prefix match alone suffices for them.
    """
    full = node_path_for(tree_node)
    if full is None:
        return False
    if full in captured.full_paths:
        return True
    logical = logical_path_of(full)
    if logical in captured.logical_paths:
        # Non-group: identity is unique within its parent's child
        # list, so a logical hit means THIS row was previously
        # expanded. Groups need the extra descendancy gate (the
        # logical of every group at the same parent level is
        # identical, so identity alone over-matches).
        model = tree_node.data
        if not isinstance(model, VariantGroupNode):
            return True
    # Group: only auto-expand when a captured deeper logical path
    # has a non-empty intersection with this group's model
    # subtree (i.e. the path continues into a row this group
    # actually wraps).
    model = tree_node.data
    if isinstance(model, VariantGroupNode):
        return _group_leads_to_captured(model, logical, captured)
    return False


def _group_leads_to_captured(
    group: VariantGroupNode,
    group_logical_path: NodePath,
    captured: CapturedExpandState,
) -> bool:
    """``True`` iff some captured logical path passes through this group.

    A captured logical path ``C`` "passes through" the group when
    ``group_logical_path`` is a strict prefix of ``C`` AND the next
    key after ``group_logical_path`` matches the logical identity
    of one of the group's model descendants (recursive over nested
    :class:`VariantGroupNode` children).

    Reads off the in-memory model -- works BEFORE the group's tree
    node has been expanded.
    """
    prefix_len = len(group_logical_path)
    deeper = [
        C
        for C in captured.logical_paths
        if len(C) > prefix_len and C[:prefix_len] == group_logical_path
    ]
    if not deeper:
        return False
    next_keys = {C[prefix_len] for C in deeper}
    return _group_subtree_has_any_next_key(group, next_keys)


def _group_subtree_has_any_next_key(
    group: VariantGroupNode,
    next_keys: "set[NodeKey]",
) -> bool:
    """Does any model descendant of ``group`` carry a key in ``next_keys``?

    Recursive scan over :attr:`VariantGroupNode.children`: each
    :class:`VariantNode` contributes its :class:`VariantKey` (via
    the backend lookup) and each nested :class:`VariantGroupNode`
    is descended into. The first hit returns ``True``.
    """
    for child in group.children:
        if isinstance(child, VariantNode):
            key = variant_key_via_backend(child)
            if key is not None and key in next_keys:
                return True
        elif isinstance(child, VariantGroupNode):
            if _group_subtree_has_any_next_key(child, next_keys):
                return True
    return False
