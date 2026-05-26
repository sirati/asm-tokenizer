"""Textual ``App`` driving the inspector TUI.

Single concern: wire the tree widget to the tree-model ``expand``
calls and centralise the expand-time error policy. Hosts the
:class:`InspectorApp` (textual ``App[None]``), the expand dispatcher
that wraps every model ``expand()`` call, the ``/`` search input, and
the file-only ``ERROR``-level inspector logger.

The tree widget itself lives in :mod:`tokenizer.inspector._app._tree_widget`;
Node-typed label composition lives in :mod:`tokenizer.inspector._app._labels`.
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
from textual.widgets import Input, Tree

from shared.logging_utils import setup_file_logger
from tokenizer.inspector._label import aligned_variant_labels
from tokenizer.inspector._tree_model import (
    FunctionNode,
    Node,
    VariantNode,
)
from tokenizer.variant_info import VariantIdentity

from . import _order_hooks
from ._auto_expand import collapse_single_child_chains
from ._help_dialog import HelpScreen
from ._labels import _ERR_STYLE, _compose_label
from ._menu_bar import Alignment, MenuBar, MenuItem
from ._order import AxisKind, OrderConfig, OrderResult, VariantGroupNode
from ._tree_widget import _InspectorTree


if TYPE_CHECKING:
    from textual.widgets._tree import TreeNode

    from tokenizer.inspector._render._protocol import (
        BackendFactory,
        FunctionHandle,
    )


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
    """Inspector app: tree + search input.

    Vertical layout: tree (a ``ScrollView`` by inheritance) + a one-
    line search input hidden by default, revealed on ``/``. Horizontal-
    scroll actions delegate to the tree's built-in ``scroll_*`` methods.

    The app holds ONE :class:`BackendFactory` reference; every root
    :class:`FunctionNode` is constructed against that factory + the
    typed :class:`FunctionHandle` published in ``factory.handles``.
    """

    CSS: ClassVar[str] = """
    Screen { layout: vertical; }
    #menubar { height: 1; }
    #tree { height: 1fr; }
    #search { display: none; height: 3; }
    #search.visible { display: block; }
    """

    BINDING_GROUP_TITLE: ClassVar[str] = "Application"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        # Horizontal-pan bindings live on :class:`_InspectorTree` (the
        # widget that owns the cursor + viewport) so the action runs
        # before the ScrollableContainer's built-in pan-only bindings
        # would otherwise capture ``left`` / ``right``.
        Binding("slash", "focus_search", "Search"),
        Binding("h", "open_help", "Help", show=True),
        Binding("escape", "hide_search", "Hide search", show=False),
        Binding("o", "open_order_dialog", "Order", show=True),
        Binding("b", "open_binary_switcher", "Switch binary", show=True),
    ]

    def __init__(
        self,
        *,
        factory: "BackendFactory",
        log_path: Path,
    ) -> None:
        super().__init__()
        self._factory = factory
        self._log = _setup_inspector_log(log_path)
        # Current variant ordering + grouping. ``None`` means
        # "default-sorted, no grouping" -- mirrors the legacy
        # backend-order rendering until the user opens the Order modal
        # at least once. One ``OrderConfig`` per binary (W3-21); no
        # per-function override.
        self._order_config: Optional[OrderConfig] = None
        # Per-:class:`FunctionHandle` pending auto-expand set, populated
        # by :meth:`_rebuild_expanded_subtrees` capture-on-rebuild. The
        # dispatcher consumes the set after the FunctionNode re-expand
        # mounts its children so previously-open variant rows surface
        # under their new group ancestors. The same set is also
        # consulted whenever a descendant :class:`VariantGroupNode`
        # mounts its children (the group's expand posts a NodeExpanded
        # asynchronously), so the chain auto-expands across the
        # potentially-many-deep group tree without polling.
        self._pending_auto_expand: dict["FunctionHandle", "set[VariantIdentity]"] = {}

    # --- compose ---------------------------------------------------

    def compose(self) -> ComposeResult:
        yield MenuBar(items=self._menu_items(), id="menubar")
        tree: _InspectorTree = _InspectorTree("inspector", id="tree")
        # Seed the root with one FunctionNode per handle the factory
        # published. The factory owns discovery; the UI just iterates.
        for handle in self._factory.handles:
            fn_node = self._build_root_function_node(handle)
            tree.root.add(
                _compose_label(fn_node),
                data=fn_node,
                allow_expand=True,
            )
        tree.root.expand()
        yield tree
        yield Input(placeholder="/ search function name", id="search")

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

    @on(Tree.NodeExpanded)
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

        for child in children:
            node.add(
                _compose_label(child),
                data=child,
                allow_expand=getattr(child, "can_expand", False),
            )

        # Capture-on-rebuild expand-state restoration: a prior
        # rebuild may have stashed an auto-expand identity set for
        # this FunctionNode. The set is consulted post-mount HERE
        # (on the FunctionNode itself) AND on every descendant
        # :class:`VariantGroupNode` mount (the group's expand posts a
        # NodeExpanded asynchronously, so the walk descends across the
        # potentially-many-deep group tree without polling).
        _order_hooks.consume_auto_expand_post_mount(self, node, model)

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

    def action_focus_search(self) -> None:
        search = self.query_one("#search", Input)
        search.add_class("visible")
        search.focus()

    def action_hide_search(self) -> None:
        search = self.query_one("#search", Input)
        search.remove_class("visible")
        search.value = ""
        self.query_one("#tree", _InspectorTree).focus()

    @on(Input.Submitted, "#search")
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        """Jump-to-function-by-name-substring.

        First substring hit against the composed function-row label
        becomes the cursor row + is auto-expanded. Subsequent
        searches start scanning anew from the top (no "find next"
        cursor in this Phase).
        """
        needle = event.value.strip().lower()
        if not needle:
            return
        tree = self.query_one("#tree", _InspectorTree)
        for child in tree.root.children:
            model = child.data
            if not isinstance(model, FunctionNode):
                continue
            label_plain = _compose_label(model).plain.lower()
            if needle in label_plain:
                tree.move_cursor(child)
                child.expand()
                self.action_hide_search()
                return

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
) -> int:
    """Construct + run the app; return ``0`` on clean quit.

    Every backend the factory mints is opened lazily on first
    ``FunctionNode.expand`` call; the caller (``__main__``) owns the
    factory + any session it wraps via ``with stack:``.
    """
    app = InspectorApp(factory=factory, log_path=log_path)
    app.run()
    return 0
