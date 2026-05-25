"""Auto-expand "selection rows" that carry only one option.

Single concern: implement the universal tree-navigation rule -- when
a parent's :meth:`Node.expand` returned exactly one child, the
intermediate selection row carries no information and we click
through it for the user.

The function lives at the UI layer (called from
:meth:`InspectorApp._on_node_expanded` after children mount), NOT on
each model node's :meth:`Node.expand`. Keeping the policy here makes
the model layer selection-shape-neutral; the UI alone decides which
levels to skip.

Recursion is implicit through Textual's event loop:
:meth:`TreeNode.expand` posts a :class:`Tree.NodeExpanded` that
re-enters the dispatcher, which re-applies the same rule one level
deeper. The base case is the same as the cutoff in this function --
zero, two-plus, or non-expandable -- so a chain of 1-child wrappers
naturally unfolds end-to-end without an explicit recursion loop.

The auto-skipped intermediate nodes remain in the tree (just initially
expanded) so the user can still collapse to navigate back; this
preserves the "user can climb the tree" affordance.
"""

from __future__ import annotations


__all__ = ["auto_expand_lone_child"]


def auto_expand_lone_child(parent_tree_node) -> None:
    """If ``parent_tree_node`` has exactly one expandable child, expand it.

    "Exactly one" is the strict interpretation: not "1 expandable + N
    non-expandable" but a single child in total. Heterogeneous sets
    (e.g. several :class:`AsmLeaf` rows + one
    :class:`ShowAllVariantsNode`) keep their content visible at the
    parent's level instead of disappearing under the lone expandable
    sibling. The pinned-variant fix that splices the callee's blocks
    PLUS a ``show all variants`` sibling under an
    :class:`InlineCallNode` therefore stays as-rendered: multiple
    children -> no auto-skip.

    The function is a no-op when the lone child is already expanded;
    that guard makes the call idempotent against capture-on-rebuild's
    own restore walk (which may have already expanded the same child).
    """
    if len(parent_tree_node.children) != 1:
        return
    only_child = parent_tree_node.children[0]
    only_model = only_child.data
    if only_model is None:
        # Container without a model payload (the tree root or the
        # error-leaf branch). Nothing to expand against.
        return
    if not getattr(only_model, "can_expand", False):
        # Terminal child -- nothing further to surface.
        return
    if only_child.is_expanded:
        # Already expanded by another path (e.g. capture-on-rebuild's
        # restore walk). Avoid posting a duplicate NodeExpanded.
        return
    only_child.expand()
