"""Inspector tree widget — Textual ``Tree`` specialised for ``Node`` payloads.

Single concern: render policy (failed-node glyph + truncation marker)
plus tree-local keyboard behaviour (editor-style horizontal-scroll
memory, conditional ``left``/``right`` arrows, chained undo for the
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
    * Chained undo for the ``←``-to-parent climb: each successful
      climb pushes the just-evacuated child onto a return stack;
      each subsequent ``→`` keypress pops one and restores the cursor
      to that child instead of panning / expanding. The stack is
      invalidated whenever the cursor moves off the parent of the
      top-of-stack entry (i.e. any other navigation moves the cursor
      elsewhere); a pure modal trip that doesn't move the tree cursor
      keeps the stack intact, so ``o``/``Esc`` round-trips don't break
      the undo chain.

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
        # Return stack for the chained ``←``-to-parent undo. Each
        # successful left-to-parent climb pushes the evacuated child;
        # each subsequent ``→`` pops one and restores that cursor.
        # Cleared by :meth:`watch_cursor_line` whenever the cursor
        # leaves the parent of the top-of-stack entry without our own
        # internal moves driving the change (see ``_pending_internal_move``).
        self._undo_left_stack: "list[TreeNode[Node]]" = []
        # Counter incremented around our own ``move_cursor`` /
        # ``action_cursor_parent`` calls so the resulting
        # :meth:`watch_cursor_line` notification skips the
        # stack-invalidation check. A counter (not a bool) keeps nested
        # safe should a future change wrap one internal move inside
        # another.
        self._pending_internal_move: int = 0

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
        # Textual paints the line as ``guides + label``; the right
        # edge of the LABEL region sits at
        # ``viewport_width - guide_width``. Shrink the effective
        # viewport so the marker fires when the label spills past the
        # label region, not when it spills past the full row.
        return apply_truncation_marker(
            text,
            max(0, self.size.width - self._row_guide_width(node)),
            self.scroll_offset.x,
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

    def _row_guide_width(self, node: TreeNode[Node]) -> int:
        """Cell-width of the indent-guide prefix Textual paints ahead of ``node``.

        Textual's ``Tree`` renders each row as ``guides + label``
        where the guides cost ``len(path) * guide_depth`` cells (see
        :meth:`textual.widgets._tree._TreeLine._get_guide_width`).
        Horizontal scroll and clipping address the FULL line, so any
        row-overflow check that ignores the guide width misses the
        case where the label cell-len alone fits the viewport but
        the rendered row does not. Look up the cursor row's cached
        tree-line and ask it; fall back to ``0`` when the row's line
        has not been built yet (e.g. detached node, pre-mount).
        """
        line_index = node._line
        if line_index < 0:
            return 0
        try:
            tree_line = self._tree_lines[line_index]
        except IndexError:
            return 0
        return tree_line._get_guide_width(self.guide_depth, self.show_root)

    def _row_cell_len(self, node: TreeNode[Node]) -> int:
        """Total cell-width of the row Textual paints for ``node``.

        Combines :meth:`_row_guide_width` (the indent-guide prefix)
        with :meth:`label_cell_len` (the composed label) so callers
        comparing against ``scroll_offset.x + viewport_width`` see
        the same column the user does on screen.
        """
        return self._row_guide_width(node) + self.label_cell_len(node)

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
        cursor actually moved, the just-evacuated child is pushed onto
        :attr:`_undo_left_stack` so a following sequence of ``→``
        presses can pop the chain back down (see :meth:`on_key`).
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
        self._pending_internal_move += 1
        try:
            self.action_cursor_parent()
        finally:
            self._pending_internal_move -= 1
        if pre_climb is not None and self.cursor_node is not pre_climb:
            self._undo_left_stack.append(pre_climb)

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
        # Compare against the FULL rendered row (indent guides + label)
        # so a label whose cell_len alone fits the viewport but whose
        # row-with-guides spills past the right edge still pans
        # instead of falling through to expand.
        row_cell_len = self._row_cell_len(node)
        if row_cell_len > self.size.width + self.scroll_offset.x:
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

    # --- chained undo for the ``←``-to-parent climb --------------------
    #
    # The undo overlay lives at the key-event layer rather than inside
    # ``action_pan_right_or_expand``: this way the literal ``right``
    # key is the only trigger, while vim ``l`` (which shares the same
    # action) is treated as "not undo" per the UX spec. Stack
    # invalidation is driven by :meth:`watch_cursor_line` rather than
    # by ``on_key`` so a keypress that doesn't move the tree cursor
    # (e.g. opening a modal, search input focus) leaves the chain
    # intact for resumption.

    async def on_key(self, event: events.Key) -> None:
        """Pop one return-stack entry on ``→`` while the chain is live.

        Runs before the non-priority binding chain fires (see
        ``App._on_key`` in textual): while the return stack is
        non-empty and the next key is ``→``, pop the most-recent
        evacuated child and move the cursor there, short-circuiting
        the binding. Any other key falls through to the normal
        dispatch -- the stack is invalidated by
        :meth:`watch_cursor_line` only when the cursor actually leaves
        the expected parent row.
        """
        if not self._undo_left_stack:
            return
        if event.key != "right":
            return
        target = self._undo_left_stack.pop()
        if not self._is_attached(target):
            # Tree was rebuilt while the entry sat on the stack; the
            # popped child is no longer reachable. Consume the right-
            # arrow silently (the user pressed an undo step; we just
            # can't fulfill this one), preserving the rest of the chain
            # for the next press.
            event.stop()
            event.prevent_default()
            return
        # ``move_cursor`` triggers the per-row scroll restore via
        # :meth:`watch_cursor_line`, mirroring a normal cursor move.
        # Suppress the stack-invalidation check for this internal move.
        self._pending_internal_move += 1
        try:
            self.move_cursor(target, animate=False)
        finally:
            self._pending_internal_move -= 1
        event.stop()
        event.prevent_default()

    def _is_attached(self, node: "TreeNode[Node]") -> bool:
        """Whether ``node`` is still part of this tree's live structure.

        Walks parent pointers up to the tree root; if any link is
        missing along the way the node was detached (e.g. its subtree
        was collapsed and re-built) and we must not move the cursor
        onto it.
        """
        cursor: "Optional[TreeNode[Node]]" = node
        while cursor is not None:
            if cursor is self.root:
                return True
            cursor = cursor.parent
        return False

    def _maybe_invalidate_undo_stack(self) -> None:
        """Clear the undo stack if the cursor wandered off the chain head.

        Called from :meth:`watch_cursor_line` after each cursor move.
        Our own internal cursor moves (left-to-parent climb + right-
        arrow pop) wrap their ``move_cursor`` calls in
        :attr:`_pending_internal_move`, so this clearing logic only
        fires on user-driven moves -- arrows, page-up/down, mouse
        click, etc. Once the cursor leaves the parent of the top-of-
        stack entry, the chain semantics are broken and the stack is
        emptied wholesale (a partial-clear policy would inflict
        non-obvious undo behaviour on the user).
        """
        if self._pending_internal_move:
            return
        if not self._undo_left_stack:
            return
        expected_parent = self._undo_left_stack[-1].parent
        if self.cursor_node is not expected_parent:
            self._undo_left_stack.clear()

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
        self._maybe_invalidate_undo_stack()
        node = self.cursor_node
        if node is None or node.data is None:
            return
        remembered = getattr(node.data, "remembered_scroll_x", 0)
        viewport_width = self.size.width
        # Measure against the full rendered row (guides + label) since
        # horizontal scroll addresses the full line; a label whose
        # cell_len alone falls short of the viewport may still be
        # clipped by indent-guide overhead on a deeply-nested row.
        row_cell_len = self._row_cell_len(node)

        effective = remembered
        if viewport_width > 0 and row_cell_len - remembered < viewport_width // 2:
            # Less than half a viewport of content sits past the
            # restored column; pull the right edge in so the row stays
            # visible. ``max(0, ...)`` clamps short rows to flush-left.
            effective = max(0, row_cell_len - viewport_width)

        self.scroll_to(x=effective, animate=False)

