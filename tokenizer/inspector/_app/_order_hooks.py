"""Order modal + grouping integration hooks for :class:`InspectorApp`.

Single concern: every Order-modal-specific code path the
:class:`InspectorApp` invokes -- dialog construction, result
dispatch, grouping pass, capture-on-rebuild expand-state +
cursor restoration. The :class:`InspectorApp` exposes thin method
shims that delegate here; the heavy lifting (axis discovery,
recursive auto-expand walk, sub-tree rebuild) lives at module scope
so :mod:`._application` stays focused on the tree dispatcher + search.

Capture-on-rebuild keys on the typed :class:`NodePath` defined in
:mod:`._node_path` -- one set of paths covers every expanded tree
node (function-level, variant-level, block-level, inline-call-level,
...) and a separate captured path drives cursor restoration. See
:mod:`._node_path` for the per-kind key dispatch.

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
    VariantNode,
)

from ._node_path import (
    CapturedExpandState,
    NodePath,
    capture_expand_state,
    logical_path_of,
    node_path_for,
    should_expand_for_capture,
)
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
    sort_variants_flat,
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
    "consume_node_path_post_mount",
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
    """Route raw variant-sibling sets through the sort + grouping pass.

    The gate is structural ("are these children a homogeneous
    :class:`VariantNode` sibling set?") with one negative exclusion
    (``model`` is a :class:`VariantGroupNode`). The exclusion is the
    "already organised" guard: a :class:`VariantGroupNode`'s ``expand``
    surfaces its pre-grouped children verbatim, so re-applying the
    sort + group pass would wrap them in an extra group level. Every
    OTHER parent whose ``expand`` returns a :class:`VariantNode` list
    -- currently :class:`FunctionNode`, :class:`ShowAllVariantsNode`,
    and the no-pin / missing-variant fallback path on
    :class:`InlineCallNode` -- gets the same sort + group treatment.
    Mixed lists (e.g. :class:`InlineCallNode`'s D2 pinned path
    returning blocks + a :class:`ShowAllVariantsNode` tail) have
    ``first`` as a non-variant and short-circuit here.

    When :attr:`InspectorApp._order_config` is ``None``, the variants are
    still natsort-ordered via :func:`sort_variants_flat` so the user-
    visible sibling order remains ``v9`` before ``v10`` (etc.) even
    before the user opens the Order dialog. When an :class:`OrderConfig`
    is active, :func:`group_variants` owns both sort + partition.
    Single sort path (cluster M-2 / M1 audit findings — no longer a
    duplicate hand-rolled regex in :mod:`tokenizer.inspector._label`).
    """
    if isinstance(model, VariantGroupNode):
        # Already organised by a prior :func:`group_variants` call --
        # the group's own ``expand`` is the verbatim hand-off.
        return children
    if not children:
        return children
    first = children[0]
    if not isinstance(first, VariantNode):
        return children
    config = app._order_config
    if config is None:
        return list(sort_variants_flat(children))  # type: ignore[arg-type]
    rendered_by_variant = _rendered_by_variant_lookup(first.backend)
    grouped = group_variants(list(children), rendered_by_variant, config)
    return list(grouped)


def consume_node_path_post_mount(
    app: "InspectorApp",
    mounted_node: "TreeNode[Node]",
    model: Node,
) -> None:
    """Re-expand + reposition the cursor against the captured state.

    Fires on every freshly-mounted parent (called from the
    dispatcher AFTER ``add(child, ...)`` for the parent's children).
    Each child is matched against the captured
    :class:`CapturedExpandState` via :func:`should_expand_for_capture`:

    * **Content rows** (variants, blocks, calls, ...) match by
      logical-path equality (group ancestry elided), so a regroup
      that re-routes a row under a different group still
      auto-expands it.
    * **Group rows** match by full-path equality OR by the
      "ancestor of a captured logical path" rule, both gated on a
      model-descendancy check so unrelated sibling groups do not
      auto-expand.

    Auto-expanding a child posts a :class:`Tree.NodeExpanded` that
    re-enters the dispatcher + re-enters this consumer for the
    deeper level. The cursor path is restored when the matching
    tree node lands (logical-path match for the same regroup-
    robustness).
    """
    captured = app._captured_expand_state
    if captured is None:
        return

    for child in mounted_node.children:
        if should_expand_for_capture(child, captured):
            if not child.is_expanded:
                child.expand()
        if captured.cursor_path is not None:
            child_full = node_path_for(child)
            if child_full is not None and _cursor_matches(
                child_full, captured.cursor_path
            ):
                _restore_cursor_to(app, child)
                # Clear the cursor path so subsequent mounts don't
                # re-position; the rest of the expand pass still
                # runs against the same captured state.
                app._captured_expand_state = CapturedExpandState(
                    full_paths=captured.full_paths,
                    logical_paths=captured.logical_paths,
                    cursor_path=None,
                )
                captured = app._captured_expand_state


def _cursor_matches(child_full: "NodePath", cursor_path: "NodePath") -> bool:
    """``True`` iff ``child_full`` is the cursor's captured target.

    Match by full path first (exact regroup-preserved row) then
    by logical projection (regroup-rewrap-aware -- the cursor
    survives moves into / out of group wrappers).
    """
    if child_full == cursor_path:
        return True
    return logical_path_of(child_full) == logical_path_of(cursor_path)


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
    """Capture the typed :class:`CapturedExpandState`, collapse +
    re-expand every currently-expanded :class:`FunctionNode` subtree.

    The dispatcher's grouping integration +
    :func:`consume_node_path_post_mount` handle the rest: each newly-
    mounted child whose :class:`NodePath` matches the captured state
    is auto-expanded, propagating through the chain via subsequent
    :class:`Tree.NodeExpanded` events. The cursor is moved to the
    captured cursor-path's tree node as soon as that node mounts.
    """
    from ._tree_widget import _InspectorTree

    tree = app.query_one("#tree", _InspectorTree)
    app._captured_expand_state = capture_expand_state(
        tree.root, tree.cursor_node
    )
    for fn_tree_node in list(_iter_expanded_function_tree_nodes(tree.root)):
        fn_model = fn_tree_node.data
        if not isinstance(fn_model, FunctionNode):
            continue
        fn_tree_node.collapse()
        fn_tree_node.expand()


def _restore_cursor_to(
    app: "InspectorApp", target_node: "TreeNode[Node]"
) -> None:
    """Move the inspector tree's cursor to ``target_node``.

    Deferred via :meth:`MessagePump.call_after_refresh`: Textual's
    :meth:`Tree.move_cursor` reads the node's :attr:`_line` attribute
    which is ``-1`` until the next render pass. Calling ``move_cursor``
    during the dispatcher's mount loop -- before the new tree lines
    are rendered -- silently positions the cursor at the root.
    Deferring the move until after the next refresh lets Textual
    compute the target's line number first.
    """
    from ._tree_widget import _InspectorTree

    tree = app.query_one("#tree", _InspectorTree)
    tree.call_after_refresh(tree.move_cursor, target_node, animate=False)


def _rendered_by_variant_lookup(
    backend: "RenderBackend",
) -> Mapping[int, "RenderedVariant"]:
    """Map ``variant_idx -> RenderedVariant`` off ``backend.variants()``."""
    return {rv.variant_idx: rv for rv in backend.variants()}
