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

from textual import events, on
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
from tokenizer.inspector._render._protocol import BlockKind
from tokenizer.inspector._tree_model import (
    AsmLeaf,
    BlockNode,
    FunctionNode,
    InlineCallNode,
    InlineJumpNode,
    Node,
    NumberPrecisionLeaf,
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


# Per-kind label policy for :class:`BlockNode` rows. The model node
# carries the typed :class:`BlockKind` discriminator; the UI side
# composes the row label per-kind without ``isinstance``-on-string
# discriminators.
_BLOCK_KIND_LABELS: dict[BlockKind, str] = {
    BlockKind.VARIANT_HEADER: "Variant Header",
    BlockKind.FUNCTION_ID: "Function ID",
}


def _block_node_label(node: "BlockNode") -> str:
    """Compose the row text for a :class:`BlockNode`.

    Dispatches off :attr:`BlockNode.kind`: the two non-body kinds
    render their fixed section name; :attr:`BlockKind.BODY` composes
    ``Block: <idx>   <preview>`` (the legacy shape).
    """
    fixed = _BLOCK_KIND_LABELS.get(node.kind)
    if fixed is not None:
        return fixed
    return f"Block: {node.block_idx}   {node.preview}"


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
        return Text(_block_node_label(node))
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
    if isinstance(node, NumberPrecisionLeaf):
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

    Render policy + tree-local keyboard behaviour. Expand / error /
    search logic still lives on :class:`InspectorApp`. Per repaint:
    (a) pick prefix glyph (default Tree icons vs ``[*]`` red on a
    failed node), (b) route through :func:`apply_truncation_marker`
    so the right-edge ``>>`` marker tracks the horizontal scroll
    position.

    Tree-local keyboard concerns owned here:

    * Editor-like per-row horizontal scroll memory (each
      :class:`Node` model carries ``remembered_scroll_x``; the tree
      saves on manual pan + restores on cursor change).
    * Cursor-aware auto-adjust when the destination row's content
      would otherwise land mostly off-screen.
    * Conditional ``→`` (and vim ``l``): pan when the cursor row
      overflows the viewport, expand a collapsed node otherwise.
    * Conditional ``←``: pan while ``scroll_x > 0``, else
      :meth:`action_cursor_parent`.
    * One-shot undo for the ``←``-to-parent climb: when the left arrow
      just escaped a child (cursor moved to its parent), the very next
      ``→`` keypress restores the cursor to that child instead of
      panning / expanding. Any other key invalidates this undo state.

    The tree's :class:`textual.containers.ScrollableContainer`
    ancestor binds ``left`` / ``right`` to its built-in
    ``scroll_left`` / ``scroll_right`` actions; the BINDINGS override
    below routes them through our save-on-pan + conditional-expand
    actions instead.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        # Override the ScrollableContainer's pan-only ``left`` / ``right``
        # with the editor-style conditional variants. ``h`` retains the
        # pure pan-left affordance for vim users; ``l`` mirrors ``→`` so
        # both stay symmetric.
        Binding("h", "pan_left", "Pan left", show=False),
        Binding("left", "pan_left_or_parent", "Pan left / parent", show=False),
        Binding("l,right", "pan_right_or_expand", "Pan right / expand", show=False),
        Binding("0", "pan_x_home", "Line start", show=False),
        Binding("dollar_sign", "pan_x_end", "Line end", show=False),
    ]

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # One-shot undo target for the ``←``-to-parent climb. Set in
        # :meth:`action_pan_left_or_parent` when the cursor actually
        # moved up to its parent; consumed by the next ``→`` keypress
        # via :meth:`on_key` or cleared by any other key.
        self._undo_left_child: "Optional[TreeNode[Node]]" = None

    def _compose_full_label(
        self,
        node: TreeNode[Node],
        base_style: Style,
        style: Style,
    ) -> Text:
        """Assemble prefix + node label *without* the truncation marker.

        Shared by :meth:`render_label` and :meth:`label_cell_len`. The
        marker is a viewport-dependent visual hint; the underlying row
        width MUST be measured against the marker-free label so cursor-
        aware scroll decisions (see :meth:`watch_cursor_line`) don't
        bake the cosmetic ' >>' suffix into their math.
        """
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

        return prefix + node_label

    def render_label(
        self,
        node: TreeNode[Node],
        base_style: Style,
        style: Style,
    ) -> Text:
        text = self._compose_full_label(node, base_style, style)
        return apply_truncation_marker(
            text, self.size.width, self.scroll_offset.x
        )

    def label_cell_len(self, node: TreeNode[Node]) -> int:
        """Cell-width of ``node``'s composed label (no truncation marker).

        The scroll-memory + auto-adjust logic measures rows against
        the actual content; bypass :meth:`get_label_width` which would
        re-enter :func:`apply_truncation_marker` and inflate the
        result when the row spills the current viewport.
        """
        from rich.style import NULL_STYLE

        return self._compose_full_label(node, NULL_STYLE, NULL_STYLE).cell_len

    # --- editor-like per-row scroll memory actions ---------------------
    #
    # Manual pan saves the cursor row's new ``scroll_x`` onto its model
    # node (``Node.remembered_scroll_x``). Cursor moves restore the
    # destination row's saved value; a temporary auto-adjustment kicks
    # in when the restored column would leave less than half a viewport
    # of content visible.

    def _save_cursor_scroll_x(self) -> None:
        """Persist ``scroll_offset.x`` onto the cursor row's model.

        No-op when the cursor sits on the root or a stray node without
        model data (e.g. the dim-red error child attached on failed
        expansion).
        """
        node = self.cursor_node
        if node is None or node.data is None:
            return
        node.data.remembered_scroll_x = self.scroll_offset.x

    def action_pan_left(self) -> None:
        """Unconditional pan-left; saves the new ``scroll_x`` onto the row."""
        self.scroll_left(animate=False)
        self._save_cursor_scroll_x()

    def action_pan_left_or_parent(self) -> None:
        """Pan left while ``scroll_x > 0``, else climb to the parent row.

        Standard file-tree TUI affordance: once the row is already
        flush-left there is nothing more to pan, so the arrow becomes a
        cursor-to-parent step. The unconditional pan binding survives
        on ``h`` for vim users who want pure horizontal scroll. When
        the action does pan, it saves the new ``scroll_x`` onto the
        cursor row's remembered value.

        When the action takes the climb-to-parent branch and the
        cursor actually moved, the just-evacuated child is stashed in
        :attr:`_undo_left_child` so the next ``→`` press can restore
        the cursor to it (see :meth:`on_key`).
        """
        if self.scroll_offset.x > 0:
            self.scroll_left(animate=False)
            self._save_cursor_scroll_x()
            return
        # Climb branch: capture the pre-move cursor so a follow-up
        # ``→`` can undo the jump. ``action_cursor_parent`` is a no-op
        # at the tree root; comparing the cursor before/after is the
        # only contract-stable way to detect whether the climb fired.
        pre_climb = self.cursor_node
        self.action_cursor_parent()
        if pre_climb is not None and self.cursor_node is not pre_climb:
            self._undo_left_child = pre_climb

    def action_pan_right_or_expand(self) -> None:
        """Pan right when the cursor row overflows, else expand the node.

        Editor-like ``→``: only pan when there is content past the
        rightmost visible column of the cursor row. If the row fits
        within the viewport already, fall through to expanding a
        collapsed node with children (``can_expand and not is_expanded``);
        on an already-expanded or terminal row the action is a no-op.
        When the action does pan, it saves the new ``scroll_x`` onto
        the cursor row's remembered value.
        """
        node = self.cursor_node
        if node is None:
            return
        cell_len = self.label_cell_len(node)
        if cell_len > self.size.width + self.scroll_offset.x:
            self.scroll_right(animate=False)
            self._save_cursor_scroll_x()
            return
        # Row fits: fall through to the expand affordance.
        model = node.data
        if (
            model is not None
            and getattr(model, "can_expand", False)
            and not node.is_expanded
        ):
            node.expand()
        # else: already expanded or leaf -- no-op.

    def action_pan_x_home(self) -> None:
        self.scroll_home(animate=False, x_axis=True, y_axis=False)
        self._save_cursor_scroll_x()

    def action_pan_x_end(self) -> None:
        self.scroll_end(animate=False, x_axis=True, y_axis=False)
        self._save_cursor_scroll_x()

    # --- one-shot undo for the ``←``-to-parent climb -------------------
    #
    # The undo overlay lives at the key-event layer rather than inside
    # ``action_pan_right_or_expand``: this way the literal ``right``
    # key is the only trigger, while vim ``l`` (which shares the same
    # action) is treated as "any other key" per the UX spec.

    async def on_key(self, event: events.Key) -> None:
        """One-shot undo for the ``←``-to-parent climb.

        Runs before the non-priority binding chain fires (see
        ``App._on_key`` in textual): if the previous keypress was a
        ``←`` that climbed to a parent and the next key is ``→``,
        restore the cursor to the just-evacuated child and short-circuit
        the binding. Any other key clears the undo state and falls
        through to the normal binding dispatch.
        """
        if self._undo_left_child is None:
            return
        if event.key == "right":
            target = self._undo_left_child
            self._undo_left_child = None
            # ``move_cursor`` triggers the per-row scroll restore via
            # :meth:`watch_cursor_line`, mirroring a normal cursor move.
            self.move_cursor(target, animate=False)
            event.stop()
            event.prevent_default()
            return
        # Any other key invalidates the undo state without consuming
        # the event.
        self._undo_left_child = None

    # --- cursor-move scroll restore + auto-adjust ----------------------

    def watch_cursor_line(self, previous_line: int, line: int) -> None:
        """Restore the destination row's remembered ``scroll_x`` on cursor move.

        Hooks Textual's reactive watcher pattern (see
        ``widgets/_tree.py``'s ``watch_cursor_line``) rather than the
        ``Tree.NodeHighlighted`` message, because the message is
        deferred and gets coalesced; the watcher runs synchronously
        right after the cursor-line update, giving us the cleanest
        moment to apply the restore + auto-adjust before any render.

        Behaviour:

        * scroll to the destination's stored ``remembered_scroll_x``;
        * if less than half a viewport of content sits past that
          column, pull the right edge in so the row stays visible.
          The adjustment is NOT saved -- the row's remembered value
          is the user's last manual choice, the auto-adjust is purely
          ergonomic.

        Stays compatible with the superclass: the parent
        :meth:`Tree.watch_cursor_line` is invoked first so the
        per-row repaint + ``NodeHighlighted`` post fire normally.
        """
        super().watch_cursor_line(previous_line, line)
        node = self.cursor_node
        if node is None or node.data is None:
            return
        remembered = getattr(node.data, "remembered_scroll_x", 0)
        viewport_width = self.size.width
        cell_len = self.label_cell_len(node)

        effective = remembered
        if viewport_width > 0 and cell_len - remembered < viewport_width // 2:
            # Less than half a viewport of content sits past the
            # restored column; pull the right edge in so the row stays
            # visible. ``max(0, ...)`` clamps short rows to flush-left.
            effective = max(0, cell_len - viewport_width)

        self.scroll_to(x=effective, animate=False)


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
        # Horizontal-pan bindings live on :class:`_InspectorTree` (the
        # widget that owns the cursor + viewport) so the action runs
        # before the ScrollableContainer's built-in pan-only bindings
        # would otherwise capture ``left`` / ``right``.
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

    # Horizontal-scroll concerns (editor-like per-row scroll memory +
    # cursor-aware auto-adjust + conditional right-arrow expand) live
    # on :class:`_InspectorTree`. The tree owns the cursor, viewport,
    # and the per-row model nodes, so keeping the keyboard logic
    # there avoids the App brokering tree state through actions.

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
