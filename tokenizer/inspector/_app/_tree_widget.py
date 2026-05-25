"""Inspector tree widget — Textual ``Tree`` specialised for ``Node`` payloads.

Single concern: render policy (failed-node glyph + truncation marker)
plus tree-local keyboard behaviour (editor-style horizontal-scroll
memory, conditional ``left``/``right`` arrows, one-shot undo for the
``left``-to-parent climb). Expand / error / search logic lives on
:class:`InspectorApp` in :mod:`tokenizer.inspector._app._application`.

The vim ``h`` key is intentionally NOT bound here -- it is reserved as
an App-level binding that opens the help modal. Pure horizontal pan
remains reachable via ``←`` (conditional) and the editor-style
``0`` / ``$`` line-start / line-end actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.style import Style
from rich.text import Text

from textual import events
from textual.binding import Binding, BindingType
from textual.widgets import Tree
from textual.widgets._tree import TOGGLE_STYLE, TreeNode

from tokenizer.inspector._horizontal_scroll import (
    apply_truncation_marker,
    assemble_failed_glyph,
)
from tokenizer.inspector._tree_model import Node


if TYPE_CHECKING:
    from typing import Optional


__all__ = ["_InspectorTree"]


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

    BINDING_GROUP_TITLE: ClassVar[str] = "Tree navigation"

    BINDINGS: ClassVar[list[BindingType]] = [
        # Override the ScrollableContainer's pan-only ``left`` / ``right``
        # with the editor-style conditional variants. ``l`` mirrors ``→``
        # so both stay symmetric. ``h`` is deliberately absent -- the
        # App owns it as the help-modal trigger; pure pan-left is
        # reachable via ``←`` when ``scroll_x > 0``.
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

    def action_pan_left_or_parent(self) -> None:
        """Pan left while ``scroll_x > 0``, else climb to the parent row.

        Standard file-tree TUI affordance: once the row is already
        flush-left there is nothing more to pan, so the arrow becomes a
        cursor-to-parent step. When the action does pan, it saves the
        new ``scroll_x`` onto the cursor row's remembered value.

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

