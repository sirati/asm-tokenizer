"""Inline search-bar widget for jumping the tree cursor to a function row.

Single concern: an :class:`Input`-backed search bar that lives between
the tree and the status bar, hidden by default. Pressing ``s`` (or the
legacy ``/``) reveals + focuses the bar; the user types a name substring;
Enter jumps the tree cursor to the first matching top-level
:class:`FunctionNode` row; Escape hides the bar and restores focus to
the tree (the tree's cursor row is untouched while the Input had focus,
so returning focus is enough to restore the prior cursor context).

The widget owns:

* its own keyboard binding for Escape (close + restore tree focus);
* its own :class:`Input.Submitted` handler (substring scan + cursor
  jump);
* the visibility toggling via the CSS ``visible`` class on its
  internal Input.

The App layer wires :meth:`InspectorApp.action_open_search` to
:meth:`SearchBar.open` so the App stays focused on the tree-dispatcher
concern. The previously-hand-rolled ``#search`` Input + the inline
``_on_search_submitted`` handler on :class:`InspectorApp` are gone --
the search surface is one widget, in one module.

The name-substring match runs against :attr:`FunctionNode.name`
(the typed handle's display name) rather than the composed row label,
so a user typing ``"calloc"`` does not accidentally match the literal
``"function"`` prefix every row carries. The match is case-insensitive
to mirror how the previous ``/`` search behaved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.widgets import Input

from tokenizer.inspector._tree_model import FunctionNode


if TYPE_CHECKING:
    from textual.widgets._tree import TreeNode

    from tokenizer.inspector._tree_model import Node

    from ._application import InspectorApp
    from ._tree_widget import _InspectorTree


__all__ = ["SearchBar"]


# Widget IDs used by both the widget itself and the App's CSS rules.
# Kept module-private; callers reach the search bar through the typed
# :class:`SearchBar` query rather than these strings.
_SEARCH_BAR_ID = "search-bar"
_SEARCH_INPUT_ID = "search-input"
_VISIBLE_CLASS = "visible"


class SearchBar(Container):
    """Inline name-search bar mounted between the tree and the status bar.

    The widget is a thin :class:`Container` wrapper around a single
    :class:`Input`. It stays hidden until :meth:`open` flips the
    ``visible`` CSS class on, focuses the Input, and (per Textual's
    focus-pull semantics) gives the next keystrokes to the bar.
    Pressing Escape inside the Input fires :meth:`action_close`;
    pressing Enter posts an :class:`Input.Submitted` which the bar's
    own handler routes into :meth:`_jump_to_function_name`.
    """

    BINDING_GROUP_TITLE: ClassVar[str] = "Search bar"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Cancel search", show=True),
    ]

    DEFAULT_CSS: ClassVar[str] = """
    SearchBar {
        display: none;
        height: 3;
        dock: bottom;
        background: $panel;
    }
    SearchBar.visible {
        display: block;
    }
    SearchBar > Input {
        height: 3;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(
            placeholder="search function name (Enter to jump, Esc to cancel)",
            id=_SEARCH_INPUT_ID,
        )

    # --- public API: called from the App's action shim -----------------

    def open(self) -> None:
        """Reveal the bar + focus the Input.

        Clears any stale text from a previous search so the user starts
        each invocation with an empty buffer. Idempotent: re-opening
        while already visible just re-focuses the Input (the typical
        case after the user dismissed the bar via Enter).
        """
        self.add_class(_VISIBLE_CLASS)
        input_widget = self._input()
        input_widget.value = ""
        input_widget.focus()

    def close(self) -> None:
        """Hide the bar + return focus to the tree.

        Restores focus to the inspector tree so the user's prior cursor
        row regains keyboard input. The tree's cursor row was not
        touched while the Input held focus (the search bar only moves
        the cursor when the user presses Enter and hits a match; an
        Escape dismissal leaves the cursor exactly where the user left
        it), so returning focus is sufficient to restore context.
        """
        self.remove_class(_VISIBLE_CLASS)
        self._input().value = ""
        tree = self._tree()
        if tree is not None:
            tree.focus()

    # --- action: Escape inside the Input -------------------------------

    def action_close(self) -> None:
        """Bound to Escape: dismiss the bar without jumping."""
        self.close()

    # --- Input.Submitted: jump to first matching FunctionNode row ------

    @on(Input.Submitted, f"#{_SEARCH_INPUT_ID}")
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        """Scan top-level FunctionNode rows; cursor-jump the first hit.

        Single concern: route the submitted needle through the typed
        match helper; on hit, expand the row + move the cursor via the
        canonical post-refresh path (mirrors
        :func:`._order_hooks._restore_cursor_to`); on miss, leave the
        bar open so the user can refine.

        Empty needles are a no-op (Enter with no text simply keeps the
        bar open) -- matches the previous ``/`` search semantics.
        """
        event.stop()
        needle = event.value.strip().lower()
        if not needle:
            return
        tree = self._tree()
        if tree is None:
            return
        match = _first_function_match(tree.root.children, needle)
        if match is None:
            # Miss: leave the bar visible + focused so the user can
            # refine. Returning here does NOT dismiss the bar.
            return
        if not match.is_expanded:
            match.expand()
        # Cursor-jump via the post-refresh hook: Textual's
        # ``move_cursor`` reads ``node._line`` which is ``-1`` until
        # the next render pass. ``call_after_refresh`` defers the
        # actual move so the target node's line index is computed
        # first. Mirrors :func:`._order_hooks._restore_cursor_to`.
        tree.call_after_refresh(tree.move_cursor, match, animate=False)
        self.close()

    # --- internal helpers ----------------------------------------------

    def _input(self) -> Input:
        return self.query_one(f"#{_SEARCH_INPUT_ID}", Input)

    def _tree(self) -> "_InspectorTree | None":
        """Look up the inspector tree without a hard dependency.

        Imports :class:`_InspectorTree` lazily so this module stays
        cheap to import (the App-level ``_application`` already pays
        the textual+tree-widget import cost). Returns ``None`` when
        the tree is not mounted yet (e.g. test fixtures that drive
        the bar in isolation).
        """
        from ._tree_widget import _InspectorTree

        try:
            return self.app.query_one("#tree", _InspectorTree)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Pure match helper (kept at module scope so unit tests drive it without
# instantiating the widget).
# ---------------------------------------------------------------------------


def _first_function_match(
    children: "list[TreeNode[Node]]", needle_lower: str
) -> "TreeNode[Node] | None":
    """Return the first top-level FunctionNode row whose name matches.

    Substring match against :attr:`FunctionNode.name` (typed handle's
    display name -- the single source of truth for the function row's
    identity). Case-insensitive: ``needle_lower`` is the already-
    lowered query, and each candidate name is lowered before the
    ``in`` check.

    Non-FunctionNode rows (the root has none today, but a defensive
    isinstance keeps the helper robust if a future addition mounts
    non-function payloads directly under the root) are skipped.
    """
    for child in children:
        model = child.data
        if not isinstance(model, FunctionNode):
            continue
        if needle_lower in model.name.lower():
            return child
    return None
