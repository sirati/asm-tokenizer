"""Filter modal integration hooks for :class:`InspectorApp`.

Single concern: every Filter-modal-specific code path the
:class:`InspectorApp` invokes -- dialog construction, result dispatch.
The :class:`InspectorApp` exposes a thin method shim that delegates
here; the heavy lifting (axis discovery + value discovery + rebuild
trigger) lives at module scope so :mod:`._application` stays focused on
the tree dispatcher + search.

The functions take the live :class:`InspectorApp` instance + the
relevant tree-model handles explicitly so unit tests can drive them
without subclassing the App (one-concern: this module owns the Filter
surface, not the App).

Capture-on-rebuild expand-state preservation is delegated to the
existing :func:`._order_hooks._rebuild_expanded_subtrees` helper: the
filter pass shares the same rebuild trigger as the order pass (every
currently-expanded function subtree gets collapsed + re-expanded), so a
filter change resurfaces previously-expanded variants the same way an
order change does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tokenizer.inspector._tree_model import FunctionNode

from ._filter import (
    FilterAccepted,
    FilterCancelled,
    FilterConfig,
    FilterDialog,
    FilterResult,
    discover_all_axis_values,
    function_has_passing_variants,
)
from ._labels import _compose_label, _compose_label_filtered_out


if TYPE_CHECKING:
    from ._application import InspectorApp


__all__ = [
    "open_filter_dialog",
    "on_filter_dialog_dismissed",
]


# ---------------------------------------------------------------------------
# Public hook entry points (called from InspectorApp method shims)
# ---------------------------------------------------------------------------


def open_filter_dialog(app: "InspectorApp") -> None:
    """Open the :class:`FilterDialog` against the app's current state.

    Discovers candidate axes lazily across every currently-expanded
    :class:`FunctionNode`'s :meth:`RenderBackend.variants` (W3-20 idiom
    shared with the order modal); seeds the dialog with the app's prior
    :class:`FilterConfig` so re-press resurrects the previous
    enable/disable state per axis value.

    Empty value sets for an axis mean "no loaded variant has reported a
    value here yet" -- the dialog skips that axis row (an empty
    SelectionList would invite confusion). When more variants get
    loaded later, the next ``f``-press surfaces the new values.
    """
    from ._order_hooks import _candidate_axes  # noqa: PLC0415

    candidates = _candidate_axes(app)
    variants = _all_loaded_rendered_variants(app)
    axis_values = discover_all_axis_values(candidates, variants)
    dialog = FilterDialog(
        axis_values=axis_values,
        prior_config=app._filter_config,
    )
    # Route the dismiss through the App-level method shim so the App's
    # post-dismiss work (status bar refresh) fires after this module's
    # config-update + rebuild trigger. The App's shim calls
    # :func:`on_filter_dialog_dismissed` first, then handles its own
    # side-effects (mirrors the order modal's wiring).
    app.push_screen(dialog, app._on_filter_dialog_dismissed)


def on_filter_dialog_dismissed(
    app: "InspectorApp", result: Optional[FilterResult]
) -> None:
    """Pattern-match :class:`FilterResult` (same idiom as the order dispatcher).

    * :class:`FilterAccepted` -> stash new config + rebuild every
      currently-expanded :class:`FunctionNode` subtree.
    * :class:`FilterCancelled` -> no-op.
    """
    if result is None or isinstance(result, FilterCancelled):
        return
    if not isinstance(result, FilterAccepted):
        raise TypeError(
            f"unexpected FilterResult type: {type(result).__name__}"
        )

    new_config = result.config
    if _configs_equal(new_config, app._filter_config):
        return
    # Normalise the empty filter back to ``None`` so the status bar /
    # apply-pass short-circuits use a single sentinel.
    app._filter_config = None if new_config.is_empty() else new_config
    _refresh_after_change(app)


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _configs_equal(
    a: Optional[FilterConfig], b: Optional[FilterConfig]
) -> bool:
    """``None`` is equivalent to an empty :class:`FilterConfig`."""
    if a is None:
        return b is None or b.is_empty()
    if b is None:
        return a.is_empty()
    return a == b


def _all_loaded_rendered_variants(app: "InspectorApp"):
    """Iterate over every :class:`RenderedVariant` from every currently-
    expanded :class:`FunctionNode`'s backend.

    Same lazy-discovery surface :mod:`._order_hooks` uses for EXTRA_META
    keys: a function the user has never opened doesn't contribute yet.
    Yielded as a list so callers that traverse twice (once per axis)
    don't re-walk the tree.
    """
    from ._order_hooks import _iter_expanded_function_tree_nodes  # noqa: PLC0415
    from ._tree_widget import _InspectorTree  # noqa: PLC0415

    tree = app.query_one("#tree", _InspectorTree)
    rvs = []
    for fn_node in _iter_expanded_function_tree_nodes(tree.root):
        model = fn_node.data
        if not isinstance(model, FunctionNode):
            continue
        backend = getattr(model, "_backend", None)
        if backend is None:
            continue
        rvs.extend(backend.variants())
    return rvs


def _refresh_after_change(app: "InspectorApp") -> None:
    """Trigger the same expand-state-preserving rebuild the order pass uses.

    Reuses :func:`._order_hooks._rebuild_expanded_subtrees` so the
    capture-on-rebuild auto-expand walk fires for the filter change
    too -- a previously-opened variant that survives the filter stays
    expanded after the rebuild, mirroring the order modal's behaviour.

    Then re-applies :func:`function_has_passing_variants` across every
    root :class:`FunctionNode` so a function that was previously
    expandable becomes greyed + non-expandable (and vice-versa) when
    its variants flip in/out of the surviving set under the new
    config. The walk uses the same predicate the initial mount path
    consults, so the two render paths cannot diverge.
    """
    from ._order_hooks import _rebuild_expanded_subtrees  # noqa: PLC0415

    _rebuild_expanded_subtrees(app)
    _refresh_root_function_row_styles(app)


def _refresh_root_function_row_styles(app: "InspectorApp") -> None:
    """Walk root :class:`FunctionNode` rows and re-apply the filter predicate.

    For each root function, recompute :func:`function_has_passing_variants`
    against the app's active :class:`FilterConfig`; rewrite the row's
    label + ``allow_expand`` to match. A function whose surviving variant
    set is empty under the new filter loses its triangle and renders
    dim; a function that gained back at least one surviving variant
    (e.g. the user re-enabled an axis value) is restored to normal.
    """
    from ._tree_widget import _InspectorTree  # noqa: PLC0415

    tree = app.query_one("#tree", _InspectorTree)
    for tree_node in tree.root.children:
        model = tree_node.data
        if not isinstance(model, FunctionNode):
            continue
        passing = function_has_passing_variants(model, app._filter_config)
        tree_node.allow_expand = passing
        tree_node.set_label(
            _compose_label(model) if passing
            else _compose_label_filtered_out(model)
        )
