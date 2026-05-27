"""Textual ``App`` driving the inspector TUI.

Single concern: wire the tree widget to the tree-model ``expand``
calls and centralise the expand-time error policy. Hosts the
:class:`InspectorApp` (textual ``App[None]``), the expand dispatcher
that wraps every model ``expand()`` call, the ``s`` / ``/`` action
shim that opens the inline :class:`SearchBar`, and the file-only
``ERROR``-level inspector logger.

The tree widget itself lives in :mod:`tokenizer.inspector._app._tree_widget`;
Node-typed label composition lives in :mod:`tokenizer.inspector._app._labels`;
the inline search-bar widget lives in
:mod:`tokenizer.inspector._app._search_bar`.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional

from rich.text import Text

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import Tree

from shared.logging_utils import setup_file_logger
from tokenizer.inspector._label import aligned_variant_labels
from tokenizer.inspector._tree_model import (
    BlockNode,
    FunctionNode,
    Node,
    VariantNode,
)

from . import _filter_hooks, _order_hooks
from ._auto_expand import collapse_single_child_chains
from ._filter import FilterConfig, FilterResult, function_has_passing_variants
from ._help_dialog import HelpScreen
from ._labels import (
    _BLOCK_KIND_INDEXED_PREFIXES,
    _ERR_STYLE,
    _compose_label,
    _compose_label_filtered_out,
)
from ._menu_bar import Alignment, MenuBar, MenuItem
from ._node_path import CapturedExpandState
from ._order import AxisKind, OrderConfig, OrderResult, VariantGroupNode
from ._search_bar import SearchBar
from ._search_match import iter_function_matches, next_match_index
from ._status_bar import StatusBar
from ._tree_widget import _InspectorTree


if TYPE_CHECKING:
    from textual.widgets._tree import TreeNode

    from tokenizer.inspector._render._protocol import (
        BackendFactory,
    )

    from ._binary_switcher._provider import LoaderProvider


__all__ = ["InspectorApp", "run_inspector"]

# Logger name -- dedicated logger (not root) so pytest's caplog and
# the asm-tokenizer's existing logging config don't fight over
# handlers. The file handler is attached in ``_setup_inspector_log``.
_LOGGER_NAME = "tokenizer.inspector"


# ---------------------------------------------------------------------------
# Variant sibling-set column alignment (sort lives in _order_hooks.apply_grouping)
# ---------------------------------------------------------------------------


def _stamp_aligned_variant_labels(
    children: list[Node],
    suppressed_axes: frozenset[str] = frozenset(),
) -> None:
    """Stamp ``VariantNode.aligned_label`` across a sibling set.

    Single concern: when an expand handler hands back a homogeneous
    sibling set of :class:`VariantNode` (the FunctionNode -> variants
    case today; grouped sub-sets when a future :class:`VariantGroupNode`
    lands), compute the per-axis-column-padded label for each variant
    and stamp it onto the node so :func:`_compose_label` reads the
    pre-aligned form. Mixed sibling sets are a no-op (column-alignment
    is only well-defined within a single set of variants).

    Threading the aligned string through a node-side field keeps the
    label dispatcher (:func:`_compose_label`) per-node and unaware of
    the sibling set — preserving its single-concern shape.

    ``suppressed_axes`` (collected from the expanded parent's
    :class:`VariantGroupNode` ancestor chain) is forwarded to
    :func:`aligned_variant_labels` so axis values already disclosed by
    a grouping-row above are dropped from the variant label. Empty set
    (no grouping ancestors) preserves the un-grouped layout verbatim.
    """
    variants = [c for c in children if isinstance(c, VariantNode)]
    if len(variants) != len(children) or not variants:
        # Heterogeneous (or empty) sibling set: nothing to align.
        return
    aligned = aligned_variant_labels(
        [v.label_axes for v in variants], suppressed_axes
    )
    for variant, label in zip(variants, aligned):
        variant.aligned_label = label


def _stamp_aligned_block_prefix_width(children: list[Node]) -> None:
    """Stamp ``BlockNode.aligned_prefix_width`` across a sibling set.

    Single concern: when an expand handler hands back a sibling set
    containing :class:`BlockNode` rows of the indexed kinds (BODY /
    JUMP_TABLE), compute the maximum width of the
    ``"<prefix>: <idx>"`` chunk across the indexed siblings and stamp
    it onto each indexed node so the label composer left-pads to that
    width — guaranteeing every row's preview suffix starts at the
    same column. Heterogeneous sibling sets are tolerated: the
    non-indexed kinds (VARIANT_HEADER / FUNCTION_ID) carry their own
    fixed-string labels and are skipped here (they keep
    ``aligned_prefix_width = None``).

    Threading the width through a node-side typed int field keeps the
    label composer (:func:`_block_node_label`) per-node and unaware of
    the sibling set — preserving its single-concern shape (mirror of
    the variant-axis :func:`_stamp_aligned_variant_labels` pattern).

    Empty or all-non-indexed sibling sets are a no-op.
    """
    indexed_blocks = [
        c
        for c in children
        if isinstance(c, BlockNode) and c.kind in _BLOCK_KIND_INDEXED_PREFIXES
    ]
    if not indexed_blocks:
        return
    width = max(
        len(f"{_BLOCK_KIND_INDEXED_PREFIXES[b.kind]}: {b.block_idx}")
        for b in indexed_blocks
    )
    for block in indexed_blocks:
        block.aligned_prefix_width = width


def _collect_suppressed_axes(
    expanded_tree_node: "TreeNode[Node]",
) -> frozenset[str]:
    """Positional axis keys disclosed by :class:`VariantGroupNode` ancestors.

    Single concern: walk the tree-node's parent chain at the App
    boundary and collect the canonical positional-prefix key of each
    enclosing :class:`VariantGroupNode` whose grouping axis is
    :attr:`AxisKind.POSITIONAL`. The returned frozenset is passed
    through to the rendering helpers so a variant label drops any
    axis-value already visible on a group row above.

    BITWIDTH + EXTRA_META grouping ancestors do NOT contribute (the
    variant's positional label does not redundantly display those
    axes anyway — BITWIDTH is a derived 32/64 with no direct slot in
    the positional label, EXTRA_META isn't part of the canonical
    positional label at all). When no :class:`VariantGroupNode`
    ancestors exist (un-grouped tree) the result is the empty
    frozenset, preserving the legacy layout exactly.
    """
    suppressed: set[str] = set()
    cursor: Optional["TreeNode[Node]"] = expanded_tree_node
    while cursor is not None:
        data = cursor.data
        if isinstance(data, VariantGroupNode):
            if data.axis.kind is AxisKind.POSITIONAL:
                suppressed.add(data.axis.key)
        cursor = cursor.parent
    return frozenset(suppressed)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class InspectorApp(App[None]):
    """Inspector app: tree + inline search bar.

    Vertical layout: tree (a ``ScrollView`` by inheritance) + a
    :class:`SearchBar` widget hidden by default, revealed on ``s``
    (or the legacy ``/`` alias). Horizontal-scroll actions delegate
    to the tree's built-in ``scroll_*`` methods.

    The app holds ONE :class:`BackendFactory` reference; every root
    :class:`FunctionNode` is constructed against that factory + the
    typed :class:`FunctionHandle` published in ``factory.handles``.
    """

    CSS: ClassVar[str] = """
    Screen { layout: vertical; }
    #menubar { height: 1; }
    #tree { height: 1fr; }
    #status-bar { height: 1; }
    """

    BINDING_GROUP_TITLE: ClassVar[str] = "Application"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "confirm_quit", "Quit"),
        # Horizontal-pan bindings live on :class:`_InspectorTree` (the
        # widget that owns the cursor + viewport) so the action runs
        # before the ScrollableContainer's built-in pan-only bindings
        # would otherwise capture ``left`` / ``right``.
        # ``s`` is the documented hotkey; ``/`` is kept as a legacy
        # alias so existing muscle memory + test coverage stay green.
        # Both route to :meth:`action_open_search`, which delegates to
        # the :class:`SearchBar` widget. Escape handling lives on the
        # search bar itself (one-concern).
        Binding("s,slash", "open_search", "Search", show=True),
        # ``n`` / ``shift+n`` step the tree cursor through every
        # match of the most-recent saved search needle (the head of
        # ``_search_history``). When the search bar is OPEN, the
        # Input's printable-key handler swallows ``n`` before the
        # binding dispatch reaches the App (see :class:`Input._on_key`
        # in textual), so the action only fires while the tree has
        # focus -- which is the only context where stepping makes
        # sense anyway.
        Binding("n", "search_next", "Next match", show=True),
        # Capital ``N`` is the literal key Textual delivers when the
        # user holds Shift; the binding system keys off the raw key
        # string (terminals don't send a "shift+n" prefix for letter
        # keys), so we register against ``N`` directly. Mirrors the
        # vim ``N`` previous-search convention.
        Binding("N", "search_prev", "Previous match", show=True),
        Binding("h", "open_help", "Help", show=True),
        Binding("o", "open_order_dialog", "Order", show=True),
        Binding("f", "open_filter_dialog", "Filter", show=True),
        Binding("b", "open_binary_switcher", "Switch binary", show=True),
        Binding("p", "toggle_preview", "Toggle block preview", show=True),
    ]

    def __init__(
        self,
        *,
        factory: "BackendFactory",
        log_path: Path,
        path: Optional[Path] = None,
        provider: "Optional[LoaderProvider]" = None,
        binary: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._factory = factory
        self._log = _setup_inspector_log(log_path)
        # Current-source tracking for the binary-switcher dialog: the
        # active provider (memmap vs csv) and the path the factory was
        # opened against. ``__main__`` passes the typed pair the CLI
        # consumed (positional ``PATH`` + ``--memmap``/``--stage1``).
        # Both being ``None`` is the mock-factory-in-tests case where
        # the switch dialog is not exercised against a real source.
        if (path is None) != (provider is None):
            raise ValueError(
                "InspectorApp: path and provider must be supplied "
                "together (both None for the mock-factory test case, "
                "both set otherwise)."
            )
        self._current_path: Optional[Path] = path
        self._current_provider: "Optional[LoaderProvider]" = provider
        self._current_binary: Optional[str] = binary
        # Current variant ordering + grouping. ``None`` means
        # "default-sorted, no grouping" -- mirrors the legacy
        # backend-order rendering until the user opens the Order modal
        # at least once. One ``OrderConfig`` per binary (W3-21); no
        # per-function override.
        self._order_config: Optional[OrderConfig] = None
        # Active filter config; ``None`` means "no filter" (the most
        # common case — the filter modal hasn't been opened yet OR every
        # value is enabled). The filter pass in
        # :func:`_order_hooks.apply_grouping` short-circuits on ``None``
        # so the variant tree's default path is untouched.
        self._filter_config: Optional[FilterConfig] = None
        # Captured typed :class:`CapturedExpandState` driving capture-
        # on-rebuild: every expanded row's :class:`NodePath` lands in
        # ``full_paths`` (group-bearing) + ``logical_paths`` (group-
        # elided) and the cursor's path in ``cursor_path``. The
        # dispatcher consults this state post-mount on every freshly-
        # mounted parent so the chain auto-expands + the cursor lands
        # on the same logical row across arbitrarily-deep regroup
        # rebuilds. ``None`` means "no rebuild pending"; see
        # :mod:`._node_path` for the per-kind key dispatch.
        self._captured_expand_state: Optional[CapturedExpandState] = None
        # Global block-row preview toggle: when True (default) collapsed
        # body / jump-table-footer rows carry the muted-style asm preview
        # suffix; toggled off via the ``p`` action so the user can hide
        # all previews simultaneously. Per-row "is the block expanded?"
        # gating is handled in :meth:`_block_label_show_preview`, not
        # here -- this flag is the GLOBAL discriminator only.
        self._preview_enabled: bool = True
        # Append-only history of needles the user pressed Enter on
        # inside the :class:`SearchBar`. Each Enter (hit OR miss)
        # appends -- the bar guards against a duplicate of the most-
        # recent entry. Read by the bar's Tab / Up / Down handlers and
        # by :meth:`action_search_next` / :meth:`action_search_prev`
        # (the App-level ``n`` / ``shift+n`` walker). Lives on the App
        # so the bar's open/close lifecycle does not lose it.
        self._search_history: list[str] = []

    # --- compose ---------------------------------------------------

    def compose(self) -> ComposeResult:
        yield MenuBar(items=self._menu_items(), id="menubar")
        tree: _InspectorTree = _InspectorTree("inspector", id="tree")
        # Seed the root with one FunctionNode per handle the factory
        # published. The factory owns discovery; the UI just iterates.
        for handle in self._factory.handles:
            fn_node = self._build_root_function_node(handle)
            passing = function_has_passing_variants(fn_node, self._filter_config)
            tree.root.add(
                _compose_label(fn_node) if passing else _compose_label_filtered_out(fn_node),
                data=fn_node,
                allow_expand=passing,
            )
        tree.root.expand()
        yield tree
        # Inline search bar -- hidden by default, revealed by the ``s``
        # binding. Lives BETWEEN the tree and the status bar (the
        # search bar's CSS docks itself to the bottom; the status bar
        # also docks to the bottom but is composed last so it lands
        # below the search bar in the dock order).
        yield SearchBar(id="search-bar")
        status_bar = StatusBar(id="status-bar")
        status_bar.set_tree(tree)
        yield status_bar

    def _menu_items(self) -> tuple[MenuItem, ...]:
        """Static set of top-bar menu items.

        Kept in lockstep with the App-level BINDINGS so the hotkey hint
        the bar shows matches the actual binding. Future menu entries
        (filter, view, ...) extend this tuple — the bar widget is
        item-list-driven and does not require changes.
        """
        return (
            MenuItem(
                label="Switch binary",
                action_name="open_binary_switcher",
                hotkey="b",
                alignment=Alignment.LEFT,
            ),
            MenuItem(
                label="help",
                action_name="open_help",
                hotkey="h",
                alignment=Alignment.RIGHT,
            ),
        )

    def _build_root_function_node(
        self, handle: "FunctionHandle"
    ) -> FunctionNode:
        """Construct one root :class:`FunctionNode` for a handle.

        The typed handle carries the display ``name``, the per-arm
        ``idx``, and the canonical ``arm`` :class:`SectionKind`; the
        factory's ``make(handle)`` opens the matching backend on
        first :meth:`FunctionNode.expand` call.
        """
        return FunctionNode(factory=self._factory, handle=handle)

    # --- expand dispatcher -----------------------------------------

    @on(Tree.NodeExpanded, "#tree")
    def _on_node_expanded(self, event: Tree.NodeExpanded[Node]) -> None:
        """Central expand dispatcher (the ONE try/except wrapping model
        ``expand``).

        Flow: clear prior children + reset ``is_failed`` (collapse-
        then-expand retries the decode), wrap ONLY the model
        ``expand()`` call in ``try/except Exception``, on failure log
        the traceback + attach a dim-red error-child carrying
        ``repr(exc)`` + flip ``is_failed`` + refresh so the prefix
        paints as ``[*]``. On success: attach one child per returned
        model node gated on ``can_expand``.
        """
        event.stop()
        node = event.node
        model = node.data
        if model is None:
            # Root or container-without-payload -- nothing to expand
            # against the model layer.
            return

        node.remove_children()
        if model.is_failed:
            model.is_failed = False
            node.refresh()

        try:
            children = model.expand()
        except Exception as exc:
            self._log.error(
                "expand failed for %r: %s",
                model,
                traceback.format_exc(),
            )
            model.is_failed = True
            err_label = Text(repr(exc), style=_ERR_STYLE)
            node.add_leaf(err_label, data=None)
            # Force re-render so the prefix glyph swaps to the [*] form.
            node.refresh()
            return

        # Universal "no selection when only one option" rule: delegated
        # to :mod:`._auto_expand`. When the parent's expand result is
        # a chain of single-expandable-child wrappers, those wrappers
        # are REMOVED from the rendered tree and the deepest content
        # is mounted under this parent directly. The chain expansion
        # threads through ``_safe_expand_one`` so an intermediate
        # wrapper failure stops the collapse + marks the wrapper for
        # the dispatcher's normal error-leaf flow.
        children = collapse_single_child_chains(
            children, expand_one=self._safe_expand_one
        )

        # Sort + group pass: :func:`_order_hooks.apply_grouping` natsort-
        # orders the variant siblings (always, with or without an active
        # ``OrderConfig``) and optionally restructures them into
        # :class:`VariantGroupNode`s when grouping axes are configured.
        # :class:`FunctionNode.expand` stays unchanged (cluster #6 W4-
        # AMENDED). The alignment stamp below runs AFTER grouping so it
        # acts on the per-group sibling set when grouping is active and
        # on the flat sorted set otherwise. Collapse runs FIRST so a
        # variant list surfaced from beneath a collapsed wrapper still
        # gets the same sort + group treatment.
        children = _order_hooks.apply_grouping(self, model, children)
        suppressed_axes = _collect_suppressed_axes(node)
        _stamp_aligned_variant_labels(children, suppressed_axes)
        # Block-row sibling-set column alignment: when ``model`` is a
        # :class:`VariantNode` its expand hands back a mix of block
        # kinds; the indexed kinds (BODY / JUMP_TABLE) carry numeric
        # suffixes of variable width. Stamping the per-sibling-set max
        # width onto each indexed node lets the label composer left-
        # pad the ``"<prefix>: <idx>"`` chunk so every row's preview
        # suffix starts at the same column. Mirrors the variant-axis
        # stamp above; the helper is a no-op for non-block / non-
        # indexed sibling sets.
        _stamp_aligned_block_prefix_width(children)

        for child in children:
            node.add(
                self._compose_child_label(child, is_expanded=False),
                data=child,
                allow_expand=getattr(child, "can_expand", False),
            )

        # Refresh the JUST-expanded node's own label: when ``node.data``
        # is a :class:`BlockNode`, dropping the preview suffix now that
        # the content is visible. Per Fix #3 (user directive): "when a
        # block has been opened that we can disable the preview, as we
        # are now looking at the content." The flip is one-way -- the
        # mirror NodeCollapsed handler restores the preview suffix when
        # the user re-collapses the row.
        self._refresh_block_label(node)

        # Capture-on-rebuild expand-state + cursor restoration. A
        # prior rebuild may have stashed a typed :class:`NodePath`
        # set + cursor path; the consumer matches each freshly-
        # mounted child against the captured set, auto-expands hits
        # (whose own :class:`Tree.NodeExpanded` re-enters here for
        # the deeper level), and moves the cursor when the matching
        # tree node lands.
        _order_hooks.consume_node_path_post_mount(self, node, model)

    @on(Tree.NodeCollapsed, "#tree")
    def _on_node_collapsed(self, event: Tree.NodeCollapsed[Node]) -> None:
        """Restore the block-row preview suffix when the user re-collapses
        a :class:`BlockNode` row.

        Mirror of the JUST-expanded refresh inside :meth:`_on_node_expanded`:
        the preview is hidden while the block's content is being viewed and
        restored once the row collapses. No-op for non-Block nodes.
        """
        # Do NOT call ``event.stop`` -- Textual's default collapse handler
        # still needs to fire to actually fold the children. We only
        # piggyback to refresh the label.
        self._refresh_block_label(event.node)

    # --- block-preview policy --------------------------------------

    def _block_label_show_preview(self, *, is_expanded: bool) -> bool:
        """Single-source policy for the BlockNode preview suffix.

        Two gates compose: the GLOBAL :attr:`_preview_enabled` flag
        (toggled by ``p``) AND the PER-ROW expansion state (an
        already-expanded block hides its own preview -- the content is
        visible below the row). Both gates must be True for the
        preview to render; either being False elides it. Centralising
        the policy keeps every call site agreeing without per-call
        branching.
        """
        return self._preview_enabled and not is_expanded

    def _compose_child_label(
        self, child: "Node", *, is_expanded: bool
    ) -> Text:
        """Compose the label for a newly-mounted child row.

        Threads the per-row expansion state + the App's global preview
        flag through :func:`_compose_label`. Non-BlockNode children
        ignore the ``show_block_preview`` argument (per the helper's
        contract); routing it through here uniformly keeps the call
        site free of node-type branching.
        """
        return _compose_label(
            child,
            show_block_preview=self._block_label_show_preview(
                is_expanded=is_expanded
            ),
        )

    def _refresh_block_label(self, tree_node: "TreeNode[Node]") -> None:
        """Recompose a tree node's label when its expansion state changes.

        No-op for non-BlockNode payloads. Reads the tree node's
        current ``is_expanded`` state + the App's global preview flag
        and writes the recomposed :class:`Text` back onto the row.
        Centralising the refresh keeps the per-row policy in lockstep
        with :meth:`_compose_child_label`.
        """
        model = tree_node.data
        if not isinstance(model, BlockNode):
            return
        tree_node.set_label(
            _compose_label(
                model,
                show_block_preview=self._block_label_show_preview(
                    is_expanded=tree_node.is_expanded
                ),
            )
        )

    def action_toggle_preview(self) -> None:
        """Flip the global block-preview flag + repaint every visible
        :class:`BlockNode` row.

        Walks the tree and recomposes every BlockNode row's label;
        already-expanded blocks stay preview-less regardless of the
        flag (the per-row gate in :meth:`_block_label_show_preview`
        keeps that policy in one place).
        """
        self._preview_enabled = not self._preview_enabled
        try:
            tree = self.query_one("#tree", _InspectorTree)
        except Exception:
            return
        self._refresh_all_block_labels(tree.root)

    def _refresh_all_block_labels(self, root: "TreeNode[Node]") -> None:
        """Recurse into ``root`` and refresh every BlockNode row's label.

        Iterative DFS over the tree's existing children -- the App
        only mounts tree nodes during expansion, so the walk reaches
        exactly the rows currently visible to the user. Non-Block
        rows are skipped silently by :meth:`_refresh_block_label`.
        """
        stack: list["TreeNode[Node]"] = list(root.children)
        while stack:
            node = stack.pop()
            self._refresh_block_label(node)
            stack.extend(node.children)

    def _safe_expand_one(self, model: "Node") -> "Optional[list[Node]]":
        """Wrapper-level ``model.expand`` with the dispatcher's error policy.

        Returns the model's own child list on success, ``None`` on
        failure. Failure path mirrors :meth:`_on_node_expanded`'s
        try/except: log the traceback at ERROR + flip ``is_failed``
        so the wrapper (which collapse will now KEEP instead of skip)
        paints with the ``[*]`` prefix and re-runs the failing expand
        when the user opens it manually.

        Lives on :class:`InspectorApp` -- not in
        :mod:`._auto_expand` -- because the error policy (logger,
        ``is_failed`` flag) is the dispatcher's concern; the collapse
        module stays selection-shape-only.
        """
        try:
            return list(model.expand())
        except Exception:
            self._log.error(
                "wrapper expand failed for %r during collapse: %s",
                model,
                traceback.format_exc(),
            )
            model.is_failed = True
            return None

    # Horizontal-scroll concerns (editor-like per-row scroll memory +
    # cursor-aware auto-adjust + conditional right-arrow expand) live
    # on :class:`_InspectorTree`. The tree owns the cursor, viewport,
    # and the per-row model nodes, so keeping the keyboard logic
    # there avoids the App brokering tree state through actions.

    # --- status bar --------------------------------------------------

    @on(Tree.NodeHighlighted, "#tree")
    def _on_node_highlighted(self, event: Tree.NodeHighlighted[Node]) -> None:
        """Refresh the status bar's breadcrumb when the cursor moves.

        Listens for Textual's :class:`Tree.NodeHighlighted` post (fired
        synchronously from :meth:`Tree.watch_cursor_line` after the
        :class:`_InspectorTree`'s own override-side scroll-restore
        completes). The status bar reads ``cursor_node`` off the tree
        it was handed at compose-time, so no event payload is needed
        here.
        """
        del event  # The widget reads cursor_node directly.
        try:
            status_bar = self.query_one("#status-bar", StatusBar)
        except Exception:
            # Status bar may not be mounted yet during the very first
            # tree-root highlight that fires before compose finishes.
            return
        status_bar.refresh_state()

    # --- modals ----------------------------------------------------

    def action_open_binary_switcher(self) -> None:
        """Open the binary-switcher modal.

        The heavy lifting (provider tree, folder picker, switch) lives
        in :mod:`tokenizer.inspector._app._binary_switcher` so this
        module's single concern stays the tree dispatcher. Imported
        lazily inside the action so the Phase-1 menu-bar landing does
        not require the switcher subpackage to exist before it ships.
        """
        from ._binary_switcher import open_binary_switcher

        open_binary_switcher(self)

    def action_confirm_quit(self) -> None:
        """Push the quit-confirmation modal; exit only on Accept."""
        from ._quit_dialog import QuitConfirmScreen

        def _on_dismiss(confirmed: bool | None) -> None:
            if confirmed:
                self.exit()

        self.push_screen(QuitConfirmScreen(), _on_dismiss)

    def action_open_help(self) -> None:
        """Push the help modal listing every active binding.

        The modal's :class:`BindingsTable` reads the screen stacked
        below itself (this app's root screen), so the rendered table
        auto-tracks any future addition to :class:`InspectorApp` or
        :class:`_InspectorTree` BINDINGS without a hand-maintained
        help string.
        """
        self.push_screen(HelpScreen())

    # --- search ----------------------------------------------------

    def action_open_search(self) -> None:
        """Reveal the inline :class:`SearchBar` + focus its Input.

        The bar owns the Escape binding, the Input.Submitted handler,
        and the cursor-jump logic; this method is the one-line shim
        that bridges the App-level ``s`` / ``/`` BINDINGS to the
        widget's :meth:`SearchBar.open` API.
        """
        self.query_one("#search-bar", SearchBar).open()

    def action_search_next(self) -> None:
        """Step the tree cursor to the next match of the most-recent needle.

        "Next" is the FunctionNode row immediately after the cursor
        row in declared tree order; wraps to the first match when the
        cursor sits on or past the last match. No-op when the search
        history is empty (no needle to step through). Delegates the
        FunctionNode iteration to :func:`iter_function_matches` so the
        same matcher backs both the bar's live preview and this
        walker.
        """
        self._step_search_match(forward=True)

    def action_search_prev(self) -> None:
        """Step the tree cursor to the previous match (mirror of ``n``).

        "Previous" is the FunctionNode row immediately before the
        cursor row in declared tree order; wraps to the last match
        when the cursor sits on or before the first match. No-op when
        the search history is empty.
        """
        self._step_search_match(forward=False)

    def _step_search_match(self, *, forward: bool) -> None:
        """Move the cursor one match forward or backward.

        Reads the latest needle from :attr:`_search_history`, builds
        the in-tree-order match list, and delegates the
        "where-to-land" decision to :func:`next_match_index`
        (which handles cursor-on-match vs cursor-on-non-match vs
        cursor-on-root + wrap-around). No-op when history is empty
        or the latest needle has no matches.
        """
        if not self._search_history:
            return
        needle = self._search_history[-1].strip().lower()
        if not needle:
            return
        try:
            tree = self.query_one("#tree", _InspectorTree)
        except Exception:
            return
        matches = list(iter_function_matches(tree.root.children, needle))
        if not matches:
            return
        cursor = tree.cursor_node
        cursor_line = cursor._line if cursor is not None else -1
        target_idx = next_match_index(
            matches, cursor, cursor_line, forward=forward
        )
        tree.call_after_refresh(
            tree.move_cursor, matches[target_idx], animate=False
        )

    # --- order dialog ----------------------------------------------

    def action_open_order_dialog(self) -> None:
        """Open the Order modal.

        Heavy lifting (axis discovery + reorder-state preservation
        across regroup) lives in :mod:`tokenizer.inspector._app._order_hooks`
        so this module's single concern stays the tree dispatcher.
        """
        _order_hooks.open_order_dialog(self)

    def _on_order_dialog_dismissed(
        self, result: Optional[OrderResult]
    ) -> None:
        """Dispatcher for the :class:`OrderDialog` result; delegates to
        :mod:`._order_hooks`."""
        _order_hooks.on_order_dialog_dismissed(self, result)

    # --- filter dialog ---------------------------------------------

    def action_open_filter_dialog(self) -> None:
        """Open the Filter modal.

        Heavy lifting (axis + value discovery, rebuild trigger) lives
        in :mod:`._filter_hooks` so this module's single concern stays
        the tree dispatcher.
        """
        _filter_hooks.open_filter_dialog(self)

    def _on_filter_dialog_dismissed(
        self, result: Optional[FilterResult]
    ) -> None:
        """Dispatcher for the :class:`FilterDialog` result; delegates to
        :mod:`._filter_hooks`. Also refreshes the status bar so its
        filter-summary segment reflects the new config."""
        _filter_hooks.on_filter_dialog_dismissed(self, result)
        try:
            status_bar = self.query_one("#status-bar", StatusBar)
        except Exception:
            return
        status_bar.set_filter_config(self._filter_config)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_inspector_log(log_path: Path) -> logging.Logger:
    """Dedicated inspector logger with a FileHandler attached.

    Idempotent: re-entry (two ``App`` instances in one process) reuses
    the existing handler for the same path. Level ERROR -- the
    inspector only writes the failure-traceback path here. Delegates
    the file-handler wiring to :func:`shared.logging_utils.setup_file_logger`
    (no console handler -- the TUI owns the terminal).
    """
    logger = logging.getLogger(_LOGGER_NAME)
    already_attached = any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == str(log_path)
        for h in logger.handlers
    )
    if not already_attached:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        setup_file_logger(
            name=_LOGGER_NAME,
            log_file_path=log_path,
            level=logging.ERROR,
            console=False,
        )
    logger.setLevel(logging.ERROR)
    return logger


# ---------------------------------------------------------------------------
# Entry helper
# ---------------------------------------------------------------------------


def run_inspector(
    *,
    factory: "BackendFactory",
    log_path: Path,
    path: Optional[Path] = None,
    provider: "Optional[LoaderProvider]" = None,
    binary: Optional[str] = None,
) -> int:
    """Construct + run the app; return ``0`` on clean quit.

    Every backend the factory mints is opened lazily on first
    ``FunctionNode.expand`` call; the caller (``__main__``) owns the
    factory + any session it wraps via ``with stack:``. The optional
    ``path`` / ``provider`` arguments seed the binary-switcher
    dialog's "current path" / "current provider" indicators -- pass
    the same typed pair the CLI's ``PATH`` positional +
    ``--memmap``/``--stage1`` flag yielded.
    """
    app = InspectorApp(
        factory=factory,
        log_path=log_path,
        path=path,
        provider=provider,
        binary=binary,
    )
    app.run()
    return 0
