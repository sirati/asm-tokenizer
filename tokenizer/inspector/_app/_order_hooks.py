"""Order modal + grouping integration hooks for :class:`InspectorApp`.

Single concern: every Order-modal-specific code path the
:class:`InspectorApp` invokes -- dialog construction, result
dispatch, grouping pass, capture-on-rebuild expand-state. The
:class:`InspectorApp` exposes thin method shims that delegate here;
the heavy lifting (axis discovery, recursive auto-expand walk,
sub-tree rebuild) lives at module scope so :mod:`._application` stays
focused on the tree dispatcher + search.

The functions take the live :class:`InspectorApp` instance + the
relevant Textual / tree-model handles explicitly so unit tests can
drive them without subclassing the App (one-concern: this module
owns the Order surface, not the App).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from tokenizer.inspector._tree_model import (
    FunctionNode,
    Node,
    ShowAllVariantsNode,
    VariantNode,
)
from tokenizer.variant_info import VariantIdentity

from ._order import (
    OrderAccepted,
    OrderCancelled,
    OrderConfig,
    OrderDialog,
    OrderResult,
    VariantGroupNode,
    build_canonical_axes,
    build_extra_meta_axis,
    discover_extra_meta_keys,
    group_variants,
)


if TYPE_CHECKING:
    from textual.widgets._tree import TreeNode

    from tokenizer.inspector._render._protocol import (
        RenderBackend,
        RenderedVariant,
    )

    from ._application import InspectorApp
    from ._tree_widget import _InspectorTree


__all__ = [
    "open_order_dialog",
    "on_order_dialog_dismissed",
    "apply_grouping",
    "consume_auto_expand_post_mount",
]


# ---------------------------------------------------------------------------
# Public hook entry points (called from InspectorApp method shims)
# ---------------------------------------------------------------------------


def open_order_dialog(app: "InspectorApp") -> None:
    """Open the :class:`OrderDialog` against the app's current state.

    Discovers EXTRA_META axes lazily across every currently-expanded
    :class:`FunctionNode`'s :meth:`RenderBackend.variants` (W3-20);
    seeds the dialog with the app's prior :class:`OrderConfig` so
    re-press resurrects the previous ordering + grouping checks.
    """
    candidates = _candidate_axes(app)
    dialog = OrderDialog(
        candidate_axes=candidates,
        prior_config=app._order_config,
    )
    app.push_screen(dialog, lambda r: on_order_dialog_dismissed(app, r))


def on_order_dialog_dismissed(
    app: "InspectorApp", result: Optional[OrderResult]
) -> None:
    """Pattern-match :class:`OrderResult` (cluster #14, B-L1 H4):

    * :class:`OrderAccepted` -> stash new config + rebuild every
      currently-expanded :class:`FunctionNode` subtree.
    * :class:`OrderCancelled` -> no-op.
    """
    if result is None or isinstance(result, OrderCancelled):
        return
    if not isinstance(result, OrderAccepted):
        raise TypeError(
            f"unexpected OrderResult type: {type(result).__name__}"
        )

    new_config = result.config
    if new_config == app._order_config:
        return
    app._order_config = new_config
    _rebuild_expanded_subtrees(app)


def apply_grouping(
    app: "InspectorApp",
    model: Node,
    children: Sequence[Node],
) -> Sequence[Node]:
    """Route variant-producing nodes through :func:`group_variants`.

    Other node kinds pass through untouched. The pass needs a
    ``variant_idx -> RenderedVariant`` lookup; the variants list is
    :meth:`RenderBackend.variants`-memoised so re-fetching it here is
    cheap (no second walk).
    """
    config = app._order_config
    if config is None:
        return children
    if not isinstance(model, (FunctionNode, ShowAllVariantsNode)):
        return children
    if not children:
        return children
    first = children[0]
    if not isinstance(first, VariantNode):
        return children
    rendered_by_variant = _rendered_by_variant_lookup(first.backend)
    grouped = group_variants(list(children), rendered_by_variant, config)
    return list(grouped)


def consume_auto_expand_post_mount(
    app: "InspectorApp",
    mounted_node: "TreeNode[Node]",
    model: Node,
) -> None:
    """If a capture-on-rebuild set is pending for this subtree, walk
    the freshly-mounted children + expand the matching ones.

    Fires on both :class:`FunctionNode` and :class:`VariantGroupNode`
    mounts so the chain descends across an arbitrarily-deep group
    tree -- each :class:`VariantGroupNode` expand posts a
    :class:`Tree.NodeExpanded` that re-enters the dispatcher, which
    re-enters this function for the nested level.
    """
    fn_tree_node = _enclosing_function_tree_node(mounted_node)
    if fn_tree_node is None:
        return
    fn_model = fn_tree_node.data
    if not isinstance(fn_model, FunctionNode):
        return
    pending = app._pending_auto_expand.get(fn_model.handle)
    if not pending:
        return
    backend = getattr(fn_model, "_backend", None)
    if backend is None:
        return
    idx_to_identity: Mapping[int, VariantIdentity] = {
        rv.variant_idx: rv.variant_identity for rv in backend.variants()
    }

    for child in mounted_node.children:
        data = child.data
        if isinstance(data, VariantNode):
            identity = idx_to_identity.get(data.variant_idx)
            if identity is not None and identity in pending:
                if not child.is_expanded:
                    child.expand()
                pending.discard(identity)
        elif isinstance(data, VariantGroupNode):
            if _group_contains_target(data, pending, idx_to_identity):
                if not child.is_expanded:
                    # The group's expand posts NodeExpanded; the
                    # dispatcher fires for it next, re-entering this
                    # consumer to handle the nested level.
                    child.expand()
    if not pending:
        app._pending_auto_expand.pop(fn_model.handle, None)


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _candidate_axes(app: "InspectorApp") -> tuple:
    """Canonical-5 + EXTRA_META axes discovered across loaded fns."""
    canonical = build_canonical_axes()
    extra_keys = _discover_extra_meta_across_loaded_functions(app)
    extras = tuple(build_extra_meta_axis(k) for k in extra_keys)
    return canonical + extras


def _discover_extra_meta_across_loaded_functions(
    app: "InspectorApp",
) -> Sequence[str]:
    """Union of EXTRA_META keys across every currently-expanded
    FunctionNode's ``backend.variants()`` (W3-20)."""
    from ._tree_widget import _InspectorTree

    tree = app.query_one("#tree", _InspectorTree)
    keys: set[str] = set()
    for fn_node in _iter_expanded_function_tree_nodes(tree.root):
        model = fn_node.data
        if not isinstance(model, FunctionNode):
            continue
        backend = getattr(model, "_backend", None)
        if backend is None:
            continue
        for k in discover_extra_meta_keys(backend.variants()):
            keys.add(k)
    return sorted(keys)


def _iter_expanded_function_tree_nodes(
    root: "TreeNode[Node]",
) -> "list[TreeNode[Node]]":
    """Every top-level :class:`FunctionNode` tree node currently
    expanded."""
    out: list = []
    for child in root.children:
        if isinstance(child.data, FunctionNode) and child.is_expanded:
            out.append(child)
    return out


def _rebuild_expanded_subtrees(app: "InspectorApp") -> None:
    """Capture expand-state, collapse + re-expand every
    currently-expanded :class:`FunctionNode` subtree. The dispatcher's
    grouping integration + :func:`consume_auto_expand_post_mount` walk
    handle the rest."""
    from ._tree_widget import _InspectorTree

    tree = app.query_one("#tree", _InspectorTree)
    for fn_tree_node in list(_iter_expanded_function_tree_nodes(tree.root)):
        fn_model = fn_tree_node.data
        if not isinstance(fn_model, FunctionNode):
            continue
        identities = _snapshot_expanded_variants(fn_tree_node)
        if identities:
            app._pending_auto_expand[fn_model.handle] = identities
        fn_tree_node.collapse()
        fn_tree_node.expand()


def _snapshot_expanded_variants(
    fn_tree_node: "TreeNode[Node]",
) -> "set[VariantIdentity]":
    """Every currently-expanded :class:`VariantNode`'s
    :class:`VariantIdentity`."""
    fn_model = fn_tree_node.data
    if not isinstance(fn_model, FunctionNode):
        return set()
    backend = getattr(fn_model, "_backend", None)
    if backend is None:
        return set()
    idx_to_identity: dict[int, VariantIdentity] = {
        rv.variant_idx: rv.variant_identity for rv in backend.variants()
    }
    identities: set[VariantIdentity] = set()
    for variant_tree_node in _iter_variant_tree_nodes(fn_tree_node):
        if not variant_tree_node.is_expanded:
            continue
        v_model = variant_tree_node.data
        if not isinstance(v_model, VariantNode):
            continue
        identity = idx_to_identity.get(v_model.variant_idx)
        if identity is not None:
            identities.add(identity)
    return identities


def _iter_variant_tree_nodes(
    fn_tree_node: "TreeNode[Node]",
) -> "list[TreeNode[Node]]":
    """DFS over the subtree; returns every :class:`VariantNode` tree
    node. Stops at variants -- deeper layers aren't part of the
    capture set."""
    out: list = []
    stack = list(fn_tree_node.children)
    while stack:
        node = stack.pop()
        if isinstance(node.data, VariantNode):
            out.append(node)
            continue
        if isinstance(node.data, VariantGroupNode):
            stack.extend(node.children)
    return out


def _enclosing_function_tree_node(
    tree_node: "TreeNode[Node]",
) -> "Optional[TreeNode[Node]]":
    """Walk up until a :class:`FunctionNode` ancestor is found."""
    cursor = tree_node
    while cursor is not None:
        if isinstance(cursor.data, FunctionNode):
            return cursor
        cursor = cursor.parent
    return None


def _rendered_by_variant_lookup(
    backend: "RenderBackend",
) -> Mapping[int, "RenderedVariant"]:
    """Map ``variant_idx -> RenderedVariant`` off ``backend.variants()``."""
    return {rv.variant_idx: rv for rv in backend.variants()}


def _group_contains_target(
    group: VariantGroupNode,
    targets: "set[VariantIdentity]",
    idx_to_identity: Mapping[int, "VariantIdentity"],
) -> bool:
    """Does this group (or any nested subgroup) wrap a target identity?

    Reads off the in-memory model -- no tree-node lookup, so it works
    BEFORE the group's tree node has been expanded. Used to decide
    whether the auto-expand walk should descend into this branch.
    """
    for child in group.children:
        if isinstance(child, VariantNode):
            if idx_to_identity.get(child.variant_idx) in targets:
                return True
        elif isinstance(child, VariantGroupNode):
            if _group_contains_target(child, targets, idx_to_identity):
                return True
    return False
