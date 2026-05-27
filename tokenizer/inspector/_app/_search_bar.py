"""Inline search-bar widget for jumping the tree cursor to a function row.

Single concern: an :class:`Input`-backed search bar that lives between
the tree and the status bar, hidden by default. Pressing ``s`` (or the
legacy ``/``) reveals + focuses the bar; the user types a name
substring; each keystroke LIVE-PREVIEWS the first-matching
:class:`FunctionNode` row by moving the tree cursor onto it; Enter
commits the preview position + appends the needle to the App's search
history; Escape restores the cursor to the position it had at open-
time and dismisses the bar. Tab restores the last query into an empty
Input; Up/Down walk the App's search history.

The widget owns visibility toggling (the CSS ``visible`` class on its
internal Input), the stashed-cursor cookie + history-walk pointer
scoped to the current open/close cycle, the :class:`Input.Changed`
handler (live-preview jump), and the :class:`Input.Submitted` handler
(commit + history append). The App layer wires
:meth:`InspectorApp.action_open_search` to :meth:`SearchBar.open` so
the App stays focused on the tree-dispatcher concern. The pure name-
substring matchers live in :mod:`._search_match` so the App's
``n``/``shift+n`` walker reaches them without instantiating the widget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.widgets import Input, Label

from ._search_match import first_function_match


if TYPE_CHECKING:
    from textual.widgets._tree import TreeNode

    from tokenizer.inspector._tree_model import Node

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
    ``visible`` CSS class on, focuses the Input, stashes the tree's
    current cursor for an eventual Escape-restore, and (per Textual's
    focus-pull semantics) gives the next keystrokes to the bar. Each
    keystroke fires :class:`Input.Changed`, which routes through
    :meth:`_on_search_changed` for live-preview cursor jumping.
    Pressing Escape inside the Input fires :meth:`action_close`;
    pressing Enter posts an :class:`Input.Submitted` which the bar's
    own handler routes into :meth:`_commit_search`.

    Tab + Up/Down operate on the App's search history (a list[str]
    appended-to on every Enter): Tab populates an empty Input with the
    most-recent query, Up/Down walk older/newer entries. The history
    pointer is bar-scoped (reset on every :meth:`open` call) so a
    fresh open always starts a fresh walk.
    """

    BINDING_GROUP_TITLE: ClassVar[str] = "Search bar"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Cancel search", show=True),
        # Tab restores the last query when the input buffer is empty;
        # the action itself is a no-op when the buffer is non-empty so
        # the user's partially-typed text is preserved.
        Binding("tab", "restore_last_query", "Restore last query", show=False),
        # History walk: Up = older, Down = newer. Bar-scoped pointer
        # lives on this widget; resets on each :meth:`open` call.
        Binding("up", "history_prev", "Previous history entry", show=False),
        Binding("down", "history_next", "Next history entry", show=False),
        # ctrl+n / ctrl+shift+n cycle matches while the bar is focused;
        # plain n / shift+n would conflict with typeable text in the
        # Input. The App-level bindings accept both forms when the bar
        # is closed.
        Binding("ctrl+n", "search_next", "Next match", show=True),
        Binding("ctrl+shift+n", "search_prev", "Previous match", show=True),
    ]

    DEFAULT_CSS: ClassVar[str] = """
    SearchBar {
        display: none;
        height: 4;
        dock: bottom;
        background: $panel;
    }
    SearchBar.visible {
        display: block;
    }
    SearchBar > Input {
        height: 3;
    }
    SearchBar > #search-hint {
        color: $text-muted;
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # The tree row the cursor sat on when :meth:`open` was called.
        # Used by :meth:`action_close` (Escape) to revert any live-
        # preview cursor moves the user introduced while typing. Reset
        # to ``None`` on close so a stale handle never outlives its
        # search session.
        self._stashed_cursor: "Optional[TreeNode[Node]]" = None
        # Index into the App's search-history list during an Up/Down
        # walk. ``None`` means "no walk active" (the buffer reflects
        # the user's current typing, not a replayed history entry).
        # Reset on each :meth:`open` call so a fresh session always
        # starts un-walked.
        self._history_index: Optional[int] = None

    def compose(self) -> ComposeResult:
        yield Input(
            placeholder="search function name (Enter to jump, Esc to cancel)",
            id=_SEARCH_INPUT_ID,
        )
        yield Label(
            "Ctrl+N next match  Ctrl+Shift+N previous  (n / N also work outside the search)",
            id="search-hint",
        )

    # --- public API: called from the App's action shim -----------------

    def open(self) -> None:
        """Reveal the bar + focus the Input + stash the tree cursor.

        Clears any stale text from a previous search so the user starts
        each invocation with an empty buffer. Captures the tree's
        current cursor onto :attr:`_stashed_cursor` so an Escape
        dismissal can revert any live-preview cursor moves the user
        introduces while typing. Resets :attr:`_history_index` so a
        fresh open always starts a fresh Up/Down walk.

        Idempotent: re-opening while already visible refreshes the
        stash + re-focuses the Input. (Typical case is opening fresh
        from the tree; the re-open path mainly matters in tests.)
        """
        tree = self._tree()
        self._stashed_cursor = tree.cursor_node if tree is not None else None
        self._history_index = None
        self.add_class(_VISIBLE_CLASS)
        input_widget = self._input()
        self._set_input_value("")
        input_widget.focus()

    def close(self, *, restore_cursor: bool = False) -> None:
        """Hide the bar + return focus to the tree.

        ``restore_cursor=True`` (Escape path) moves the tree cursor
        back to :attr:`_stashed_cursor` -- the row that held the
        cursor when the bar was opened. ``restore_cursor=False``
        (Enter / programmatic close path) leaves the cursor on the
        currently-previewed row.

        Resets :attr:`_stashed_cursor` + :attr:`_history_index` so a
        stale cookie never outlives its session.
        """
        self.remove_class(_VISIBLE_CLASS)
        self._set_input_value("")
        tree = self._tree()
        if restore_cursor and tree is not None and self._stashed_cursor is not None:
            tree.call_after_refresh(
                tree.move_cursor, self._stashed_cursor, animate=False
            )
        self._stashed_cursor = None
        self._history_index = None
        if tree is not None:
            tree.focus()

    # --- actions -------------------------------------------------------

    def action_close(self) -> None:
        """Bound to Escape: dismiss the bar + restore the stashed cursor."""
        self.close(restore_cursor=True)

    def action_restore_last_query(self) -> None:
        """Bound to Tab: populate an empty Input with the last query.

        The "last" query is the head of the App's search history (most-
        recent entry first; see :meth:`InspectorApp.search_history`).
        No-op when the Input is non-empty or the history is empty,
        preserving the user's current buffer + falling through to
        Textual's default Tab handling when nothing to restore.
        """
        input_widget = self._input()
        if input_widget.value:
            return
        history = self._app_search_history()
        if not history:
            return
        last = history[-1]
        self._set_input_value(last)

    def action_history_prev(self) -> None:
        """Bound to Up: replace the Input with the previous (older) entry.

        Walks :attr:`_history_index` one step back through the App's
        search history. At the oldest entry, further Up presses are
        no-ops (the user can press Down to walk back forward). Empty
        history is a no-op.
        """
        history = self._app_search_history()
        if not history:
            return
        if self._history_index is None:
            # Fresh walk: land on the most-recent entry.
            self._history_index = len(history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            # Already at the oldest entry -- no-op.
            return
        self._set_input_value(history[self._history_index])

    def action_search_next(self) -> None:
        """Bound to ctrl+n: cycle to the next search match (App helper)."""
        action = getattr(self.app, "action_search_next", None)
        if callable(action):
            action()

    def action_search_prev(self) -> None:
        """Bound to ctrl+shift+n: cycle to the previous match (App helper)."""
        action = getattr(self.app, "action_search_prev", None)
        if callable(action):
            action()

    def action_history_next(self) -> None:
        """Bound to Down: replace the Input with the next (newer) entry.

        Walks :attr:`_history_index` one step forward through the App's
        search history. Stepping past the most-recent entry clears the
        Input (the conventional shell-history affordance: Down past
        the end returns to "current line"). No-op when no walk is in
        progress (``_history_index is None``) or when history is empty.
        """
        if self._history_index is None:
            return
        history = self._app_search_history()
        if not history:
            return
        if self._history_index < len(history) - 1:
            self._history_index += 1
            self._set_input_value(history[self._history_index])
        else:
            # Past the most-recent entry: clear + leave the walk.
            self._history_index = None
            self._set_input_value("")

    # --- Input.Changed: live-preview cursor jump -----------------------

    @on(Input.Changed, f"#{_SEARCH_INPUT_ID}")
    def _on_search_changed(self, event: Input.Changed) -> None:
        """Live-preview: move the tree cursor to the first match.

        Fires on every keystroke AND on programmatic value writes
        (Tab restore, Up/Down history walk, the value-clear inside
        :meth:`open` / :meth:`close`). The handler is the SINGLE entry
        point for live-preview cursor jumps; the action methods just
        set the Input value and let this handler take care of the
        cursor.

        Empty needles + no-match needles leave the cursor on its
        current row (a no-match leaves the previously-previewed match
        in place, since reverting on every miss-after-hit would be
        more jarring than helpful). Escape is the explicit revert
        path.
        """
        event.stop()
        self._preview_jump(event.value)

    # --- Input.Submitted: commit preview + append to history -----------

    @on(Input.Submitted, f"#{_SEARCH_INPUT_ID}")
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        """Commit the current preview + append the needle to history.

        On hit (cursor already sits on the matching row from the live
        preview): append the needle to the App's search history (skip
        duplicate-of-last) and close the bar without restoring the
        stashed cursor. On miss: still append the needle to history
        (every Enter saves, per the user's spec) but leave the bar
        visible so the user can refine. Empty needle: no-op.
        """
        event.stop()
        needle = event.value.strip()
        if not needle:
            return
        self._app_append_to_history(needle)
        match = self._first_match(needle)
        if match is None:
            # Miss: leave the bar visible + focused so the user can
            # refine. Stashed cursor is preserved -- a subsequent
            # Escape still reverts to the open-time position.
            return
        # Hit: the live preview already moved the cursor; just dismiss
        # the bar (without reverting).
        self.close(restore_cursor=False)

    # --- internal helpers ----------------------------------------------

    def _preview_jump(self, raw_needle: str) -> None:
        """Move the tree cursor to the first match for ``raw_needle``.

        Empty or whitespace-only needle is a no-op (the open-time
        cursor stays put). No-match needle is also a no-op (preserve
        the previously-previewed cursor row -- the spec calls out
        ``keep cursor where it is`` as less jarring than reverting on
        every miss-after-hit).

        Defers the cursor move via ``call_after_refresh`` so Textual
        has a chance to compute ``node._line`` after any pending render.
        Mirrors :func:`._order_hooks._restore_cursor_to`.
        """
        match = self._first_match(raw_needle)
        if match is None:
            return
        tree = self._tree()
        if tree is None:
            return
        tree.call_after_refresh(tree.move_cursor, match, animate=False)

    def _first_match(self, raw_needle: str) -> "Optional[TreeNode[Node]]":
        """Return the first matching FunctionNode row for ``raw_needle``.

        Shared by the live-preview path (:meth:`_preview_jump`) and
        the Enter-commit path (:meth:`_on_search_submitted`). Returns
        ``None`` for an empty/whitespace-only needle, an unmounted
        tree, or no match in the FunctionNode rows under the root.
        """
        needle = raw_needle.strip().lower()
        if not needle:
            return None
        tree = self._tree()
        if tree is None:
            return None
        return first_function_match(tree.root.children, needle)

    def _set_input_value(self, value: str) -> None:
        """Set the Input's value.

        The assignment posts an asynchronous :class:`Input.Changed`
        which the bar's own :meth:`_on_search_changed` handler turns
        into the live-preview cursor jump. Keeping the write path in
        one place keeps the Tab + Up + Down + open + close call sites
        symmetric.
        """
        self._input().value = value

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

    def _app_search_history(self) -> "list[str]":
        """Look up the App's search-history list.

        Returns an empty list when the App doesn't expose a history
        attribute (defensive fallback for the rare test fixture that
        mounts the bar against a vanilla :class:`App`).
        """
        history = getattr(self.app, "_search_history", None)
        if not isinstance(history, list):
            return []
        return history

    def _app_append_to_history(self, needle: str) -> None:
        """Append ``needle`` to the App's search history (skip dup-of-last).

        The history list lives on the App so :meth:`InspectorApp.action_search_next`
        / :meth:`InspectorApp.action_search_prev` (the ``n`` / ``shift+n``
        bindings) can read it when the bar is closed. The skip-dup-of-
        last rule keeps the list focused on distinct queries: pressing
        Enter twice on the same buffer doesn't duplicate.
        """
        history = getattr(self.app, "_search_history", None)
        if not isinstance(history, list):
            return
        if history and history[-1] == needle:
            return
        history.append(needle)


