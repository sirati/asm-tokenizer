"""Collapse 1-expandable-child wrapper chains out of the tree.

Single concern: implement the universal tree-navigation rule -- when
a parent's :meth:`Node.expand` returns exactly one expandable child,
that child is a content-free "selection of one" wrapper. We REMOVE
the wrapper from the rendered tree and promote its own expand result
in its place, recursively, until the lone-child chain ends (zero,
two-or-more, or a terminal child).

The policy lives at the UI layer (called from
:meth:`InspectorApp._on_node_expanded` between the parent's
``model.expand()`` and the actual mounting of children), NOT on each
model node's :meth:`Node.expand`. Keeping the policy here makes the
model layer selection-shape-neutral; the UI alone decides which
levels to skip.

Contrast with the prior in-tree-auto-expand variant: that flavour
kept wrappers visible (just initially expanded). The user-facing
spec is now stricter -- wrappers must be GONE so the deepest content
sits directly under the parent who actually carries information.

A recursion depth limit prevents pathological loops from defective
``Node.expand`` implementations (e.g. a node that returns itself).
On any error during an intermediate wrapper's expand the chain is
cut at that level + the wrapper is kept visible with
``is_failed=True`` so the dispatcher's normal error-leaf attach
re-fires when the user expands the wrapper manually.

Heterogeneous sibling sets (e.g. several :class:`AsmLeaf` rows + one
:class:`ShowAllVariantsNode`) do NOT trigger collapse. The
pinned-variant fix that splices the callee's blocks PLUS a ``show
all variants`` sibling under an :class:`InlineCallNode` therefore
stays as-rendered: multiple children -> no collapse.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, TYPE_CHECKING


if TYPE_CHECKING:
    from tokenizer.inspector._tree_model import Node


__all__ = ["collapse_single_child_chains"]


# Defensive cap on chain depth. A well-formed inspector tree never
# stacks more than a handful of 1-child wrappers; 20 is comfortably
# over the natural maximum (AsmLeaf -> InlineCallNode -> [variant?] ->
# block ~ 4 levels) while bounding the worst-case recursion.
_DEFAULT_DEPTH_LIMIT = 20


def collapse_single_child_chains(
    children: Sequence["Node"],
    *,
    expand_one: Callable[["Node"], "Optional[Sequence[Node]]"],
    depth_limit: int = _DEFAULT_DEPTH_LIMIT,
) -> "list[Node]":
    """Walk a single-expandable-child chain and return the deepest list.

    ``children`` is the parent's already-expanded children list (the
    output of :meth:`Node.expand`). ``expand_one`` is a callback that
    takes ONE model node and returns either its own ``expand`` result
    (success) or ``None`` (failure -- the caller has logged + marked
    ``is_failed`` so the wrapper stays visible). The callback is the
    seam where the dispatcher's try/except + logging policy lives;
    this module stays selection-shape-only.

    The loop is iterative (not real Python recursion) so the depth
    bound is exact + stack-safe. ``depth_limit`` defaults to the
    module constant; tests may pass a smaller value to assert the
    bound fires.

    Returns the FINAL list to mount under the original parent. May be
    empty (a 1-child chain that bottoms out in ``expand`` returning
    ``[]``) or the input list unchanged (when no collapse applies).
    """
    current = list(children)
    for _ in range(depth_limit):
        if len(current) != 1:
            return current
        only = current[0]
        if not getattr(only, "can_expand", False):
            # Terminal lone child -- nothing deeper to surface; keep
            # the leaf as-is so the user still sees its content.
            return current
        grandchildren = expand_one(only)
        if grandchildren is None:
            # Wrapper expand failed; the callback has flagged the
            # wrapper for the dispatcher's error UI. Stop collapsing
            # so the wrapper itself is mounted with the failure flag.
            return current
        current = list(grandchildren)
    # Depth bound exhausted -- defensive cutoff. Whatever we have is
    # what we mount; the dispatcher's normal expand handles the rest.
    return current
