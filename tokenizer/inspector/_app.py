"""Textual ``App`` driving the inspector TUI.

Single concern: wire the tree widget to the tree-model ``expand``
calls and centralise the expand-time error policy. ONLY module in
:mod:`tokenizer.inspector` that imports :mod:`textual` -- everything
else is pure model + label/render helpers, keeping the default
``nix develop`` shell free of the ``textual`` dep.

The model is fully frozen (every node dataclass uses
``frozen=True``). The dispatcher uses :func:`object.__setattr__` to
flip ``is_failed`` -- the canonical pattern for post-init mutation
on a frozen dataclass without unfreezing the whole model.
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
from textual.widgets._tree import TreeNode

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


if TYPE_CHECKING:
    from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
    from tokenizer.aligned_data.loader.session import BinarySession


__all__ = ["InspectorApp", "run_inspector"]


# Dim red style for the error-child leaf placed under a failed node.
_ERR_STYLE = Style(color="red", dim=True)

# Logger name -- dedicated logger (not root) so pytest's caplog and
# the asm-tokenizer's existing logging config don't fight over
# handlers. The file handler is attached in ``_setup_inspector_log``.
_LOGGER_NAME = "tokenizer.inspector"


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
        return Text(node.label)
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

        failed = node.data is not None and getattr(
            node.data, "is_failed", False
        )
        if failed:
            glyph_text, glyph_style = assemble_failed_glyph(base_style)
            prefix = Text(glyph_text, style=glyph_style)
        elif node._allow_expand:
            prefix = Text(
                self.ICON_NODE_EXPANDED if node.is_expanded else self.ICON_NODE,
                style=base_style,
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
        dataset: "BinaryDataset",
        session: "BinarySession",
        log_path: Path,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._session = session
        self._log = _setup_inspector_log(log_path)

    # --- compose ---------------------------------------------------

    def compose(self) -> ComposeResult:
        tree: _InspectorTree = _InspectorTree("inspector", id="tree")
        # Seed the root with one matched-arm function per FID (plan D3).
        # ``matched_count`` is the published count; FID ordering equals
        # encounter order from the disassembler so iterating ``range(N)``
        # is the canonical seed walk.
        for idx in range(self._dataset.matched_count):
            fn_node = self._build_root_function_node(idx)
            tree.root.add(
                _compose_label(fn_node),
                data=fn_node,
                allow_expand=True,
            )
        tree.root.expand()
        yield tree
        yield Input(placeholder="/ search function name", id="search")

    def _build_root_function_node(self, idx: int) -> FunctionNode:
        """Construct one root :class:`FunctionNode` for a matched idx.

        Reads the function name from the dataset's published
        ``matched_func_names`` array (one entry per matched FID,
        ordered with ``range(matched_count)``); arm is fixed to
        ``SectionKind.MATCHED`` since the top-level tree only seeds
        matched functions.
        """
        # Lazy import keeps this module's top-level imports textual-
        # focused; SectionKind is already in memory via the dataset.
        from tokenizer.aligned_data.loader.metadata_loader import SectionKind

        func_names = self._dataset.matched_func_names
        name = func_names[idx] if idx < len(func_names) else "?"
        return FunctionNode(arm=SectionKind.MATCHED, idx=idx, name=name)

    # --- expand dispatcher (plan D8) -------------------------------

    @on(Tree.NodeExpanded)
    def _on_node_expanded(self, event: Tree.NodeExpanded[Node]) -> None:
        """Central expand dispatcher (the ONE try/except per plan D8).

        Flow: clear prior children + reset ``is_failed`` (collapse-
        then-expand retries the decode), wrap ONLY the model
        ``expand`` call in ``try/except Exception``, on failure log
        the traceback + attach a dim-red error-child carrying
        ``repr(exc)`` + flip ``is_failed`` (via ``object.__setattr__``
        -- frozen dataclass) + refresh so the prefix paints as
        ``[*]``. On success: attach one child per returned model node
        gated on ``can_expand``.
        """
        event.stop()
        node = event.node
        model = node.data
        if model is None:
            # Root or container-without-payload -- nothing to expand
            # against the model layer.
            return

        node.remove_children()
        # Reset prior failed state (frozen dataclass -- bypass via
        # ``object.__setattr__``; canonical post-init mutation pattern).
        if getattr(model, "is_failed", False):
            object.__setattr__(model, "is_failed", False)
            node.refresh()

        try:
            children = model.expand(
                self._session,
                vocab_manager=self._dataset.vocab_manager,
            )
        except Exception as exc:
            self._log.error(
                "expand failed for %r: %s",
                model,
                traceback.format_exc(),
            )
            object.__setattr__(model, "is_failed", True)
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
    dataset: "BinaryDataset",
    session: "BinarySession",
    log_path: Path,
) -> int:
    """Construct + run the app; return ``0`` on clean quit.

    The session's ``__enter__`` MUST already have run -- every
    ``FunctionNode.expand`` calls :meth:`BinarySession.load_matched`,
    which requires an open session. :mod:`tokenizer.inspector.__main__`
    owns the ``with session:`` wrap.
    """
    app = InspectorApp(dataset=dataset, session=session, log_path=log_path)
    app.run()
    return 0
