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

from rich.style import Style
from rich.text import Text

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import Input, Tree

from shared.logging_utils import setup_file_logger
from tokenizer.inspector._label import (
    aligned_variant_labels,
    variant_natural_sort_key,
)
from tokenizer.inspector._tree_model import (
    FunctionNode,
    Node,
    VariantNode,
)
from tokenizer.variant_info import VariantIdentity

from . import _order_hooks
from ._help_dialog import HelpScreen
from ._labels import _compose_label
from ._order import OrderConfig, OrderResult
from ._tree_widget import _InspectorTree


if TYPE_CHECKING:
    from tokenizer.inspector._render._protocol import (
        BackendFactory,
        FunctionHandle,
    )


__all__ = ["InspectorApp", "run_inspector"]


# Dim red style for the error-child leaf placed under a failed node.
_ERR_STYLE = Style(color="red", dim=True)

# Logger name -- dedicated logger (not root) so pytest's caplog and
# the asm-tokenizer's existing logging config don't fight over
# handlers. The file handler is attached in ``_setup_inspector_log``.
_LOGGER_NAME = "tokenizer.inspector"


# ---------------------------------------------------------------------------
# Variant sibling-set natural sort + column alignment
# ---------------------------------------------------------------------------


def _sort_variants_naturally(children: list[Node]) -> None:
    """Sort a homogeneous :class:`VariantNode` sibling set in-place by
    :func:`variant_natural_sort_key` (so ``v10`` sorts AFTER ``v9``).

    Single concern: natural sort across the immediate sibling set. The
    sort runs BEFORE :func:`_stamp_aligned_variant_labels` so the
    aligned-label column widths are computed in display order. Mixed
    sibling sets are a no-op (sort order across heterogeneous kinds
    is not well-defined here).
    """
    if not children or not all(isinstance(c, VariantNode) for c in children):
        return
    children.sort(
        key=lambda node: variant_natural_sort_key(
            node.label_axes  # type: ignore[union-attr]  # narrowed by isinstance guard
        )
    )


def _stamp_aligned_variant_labels(children: list[Node]) -> None:
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
    """
    variants = [c for c in children if isinstance(c, VariantNode)]
    if len(variants) != len(children) or not variants:
        # Heterogeneous (or empty) sibling set: nothing to align.
        return
    aligned = aligned_variant_labels([v.label_axes for v in variants])
    for variant, label in zip(variants, aligned):
        variant.aligned_label = label


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

        _sort_variants_naturally(children)
        _stamp_aligned_variant_labels(children)
        # Grouping pass: when the expanded model owns a variant list
        # (FunctionNode / ShowAllVariantsNode) and an OrderConfig is
        # active, route the variants through :func:`group_variants`
        # at the :mod:`._order` boundary -- :class:`FunctionNode.expand`
        # stays unchanged (cluster #6 W4-AMENDED). The sort + align
        # passes above act on the flat variant list before grouping
        # restructures it; the variant objects retain their stamped
        # ``aligned_label`` across grouping (groups wrap the same
        # variant instances).
        children = _order_hooks.apply_grouping(self, model, children)

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

    # Horizontal-scroll concerns (editor-like per-row scroll memory +
    # cursor-aware auto-adjust + conditional right-arrow expand) live
    # on :class:`_InspectorTree`. The tree owns the cursor, viewport,
    # and the per-row model nodes, so keeping the keyboard logic
    # there avoids the App brokering tree state through actions.

    # --- modals ----------------------------------------------------

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
