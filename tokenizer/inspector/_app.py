"""Textual ``App`` driving the inspector TUI.

Single concern: wire the tree widget to the tree-model ``expand``
calls and centralise the expand-time error policy. ONLY module in
:mod:`tokenizer.inspector` that imports :mod:`textual` -- everything
else is pure model + label/render helpers, keeping the default
``nix develop`` shell free of the ``textual`` dep.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from rich.style import Style
from rich.text import Text

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import Input, Tree
from textual.widgets._tree import TOGGLE_STYLE, TreeNode

from tokenizer.inspector._horizontal_scroll import (
    apply_truncation_marker,
    assemble_failed_glyph,
)
from tokenizer.inspector._label import (
    function_label,
    inline_call_label,
    inline_jump_label,
)
from tokenizer.inspector._tree_model import (
    AsmLeaf,
    BlockNode,
    FunctionNode,
    InlineCallNode,
    InlineJumpNode,
    Node,
    ShowAllVariantsNode,
    VariantNode,
)
from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
    POSITIONAL_PREFIXES,
)


if TYPE_CHECKING:
    from typing import Mapping, Optional

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


# Per-axis label prefix used to format the variant row text from
# :attr:`RenderedVariant.label_axes`. Mirrors the per-axis policy in
# :func:`tokenizer.inspector._label.variant_label` (``v`` for cver,
# ``-`` for opt; bare value for arch + compiler) but operates on the
# typed prefix-keyed Mapping instead of the legacy ``function_data.
# metadata`` shape. Anchored on POSITIONAL_PREFIXES so adding a new
# axis trips the assert below in lockstep with :mod:`_label`.
_AXIS_LABEL_PREFIX: dict[str, str] = {
    ARCH_PREFIX: "",
    COMP_PREFIX: "",
    CVER_PREFIX: "v",
    OPT_PREFIX: "-",
}
assert set(_AXIS_LABEL_PREFIX) == set(POSITIONAL_PREFIXES), (
    "_AXIS_LABEL_PREFIX must mirror POSITIONAL_PREFIXES"
)


def _variant_label_from_axes(
    label_axes: "Mapping[str, Optional[str]]",
) -> str:
    """Format the variant row label from the typed ``label_axes``.

    Both backends pre-flatten ``label_axes`` over
    :data:`POSITIONAL_PREFIXES` (plan decision 1); reading the Mapping
    in that canonical order yields a stable axis ordering. ``None``
    values render as ``"?"`` to match
    :func:`tokenizer.inspector._label.variant_label`'s legacy policy
    without re-deriving the unrelated ``metadata`` key shape.
    """
    parts: list[str] = []
    for prefix in POSITIONAL_PREFIXES:
        value = label_axes.get(prefix)
        value_str = "?" if value is None else str(value)
        parts.append(f"{_AXIS_LABEL_PREFIX[prefix]}{value_str}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Label composition (UI side -- plan-drift audit explicitly accepts that
# the BlockNode label composes ``"Block: <i>   <preview>"`` here, since
# the model carries ``block_idx`` + ``preview`` separately).
# ---------------------------------------------------------------------------


def _compose_label(node: Node) -> Text:
    """Translate one model node into its visible label text.

    Dispatch is by ``isinstance`` on the seven concrete node types
    (no string compares on type names). The BlockNode label composes
    its two model fields (``block_idx`` + ``preview``) into the row
    text here on the UI side -- the model deliberately splits them.
    """
    if isinstance(node, FunctionNode):
        return Text(function_label(node.name))
    if isinstance(node, VariantNode):
        return Text(_variant_label_from_axes(node.label_axes))
    if isinstance(node, BlockNode):
        return Text(f"Block: {node.block_idx}   {node.preview}")
    if isinstance(node, InlineCallNode):
        return Text(
            inline_call_label(
                node.kind, node.counter_id, node.callee_name, node.provider
            )
        )
    if isinstance(node, InlineJumpNode):
        return Text(inline_jump_label(node.target_block_idx))
    if isinstance(node, AsmLeaf):
        return Text(node.text)
    if isinstance(node, ShowAllVariantsNode):
        return Text(node.label)
    # Closed Node union; any miss is a model/UI contract drift.
    raise TypeError(f"unsupported node type: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Tree widget specialised for inspector node payloads.
# ---------------------------------------------------------------------------


class _InspectorTree(Tree[Node]):
    """Tree widget that paints the ``[*]`` failed glyph + ``>>`` marker.

    Render policy only -- expand / error / search logic lives on
    :class:`InspectorApp`. Per repaint: (a) pick prefix glyph
    (default Tree icons vs ``[*]`` red on a failed node), (b) route
    through :func:`apply_truncation_marker` so the right-edge ``>>``
    marker tracks the horizontal scroll position.
    """

    def render_label(
        self,
        node: TreeNode[Node],
        base_style: Style,
        style: Style,
    ) -> Text:
        node_label = node._label.copy()
        node_label.stylize(style)

        failed = node.data is not None and node.data.is_failed
        if failed:
            glyph_text, glyph_style = assemble_failed_glyph(base_style)
            prefix = Text(glyph_text, style=glyph_style)
        elif node._allow_expand:
            prefix = Text(
                self.ICON_NODE_EXPANDED if node.is_expanded else self.ICON_NODE,
                style=base_style + TOGGLE_STYLE,
            )
        else:
            prefix = Text("", style=base_style)

        text = prefix + node_label
        return apply_truncation_marker(
            text, self.size.width, self.scroll_offset.x
        )


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

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("h,left", "tree_scroll_left", "Pan left"),
        Binding("l,right", "tree_scroll_right", "Pan right"),
        Binding("0", "tree_scroll_x_home", "Line start"),
        Binding("dollar_sign", "tree_scroll_x_end", "Line end"),
        Binding("slash", "focus_search", "Search"),
        Binding("escape", "hide_search", "Hide search", show=False),
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

        for child in children:
            node.add(
                _compose_label(child),
                data=child,
                allow_expand=getattr(child, "can_expand", False),
            )

    # --- horizontal-scroll actions ---------------------------------

    def action_tree_scroll_left(self) -> None:
        self.query_one("#tree", _InspectorTree).scroll_left(animate=False)

    def action_tree_scroll_right(self) -> None:
        self.query_one("#tree", _InspectorTree).scroll_right(animate=False)

    def action_tree_scroll_x_home(self) -> None:
        self.query_one("#tree", _InspectorTree).scroll_home(
            animate=False, x_axis=True, y_axis=False
        )

    def action_tree_scroll_x_end(self) -> None:
        self.query_one("#tree", _InspectorTree).scroll_end(
            animate=False, x_axis=True, y_axis=False
        )

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


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_inspector_log(log_path: Path) -> logging.Logger:
    """Dedicated inspector logger with a FileHandler attached.

    Idempotent: re-entry (two ``App`` instances in one process) reuses
    the existing handler for the same path. Level ERROR -- the
    inspector only writes the failure-traceback path here.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    already_attached = any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == str(log_path)
        for h in logger.handlers
    )
    if not already_attached:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
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
