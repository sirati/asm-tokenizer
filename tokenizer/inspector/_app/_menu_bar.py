"""Top menu bar widget: data-driven item list with click + hotkey access.

Single concern: render a one-line bar of clickable menu items (left- and
right-aligned columns) and route a click into the App via
:meth:`textual.app.App.run_action`. The bar knows NOTHING about which
modal each action opens — items are pure data (``label``,
``action_name``, ``hotkey``, ``alignment``) and the App owns every
``action_*`` method. The hotkey shown on each item is for display
only; the actual key dispatch is handled by Textual ``BINDINGS`` on
:class:`tokenizer.inspector._app._application.InspectorApp`. This lets
the menu items stay in lockstep with the help-modal binding list (both
read off the same source of truth, the App-level BINDINGS).

API surface crossing the module boundary:

* :class:`MenuItem` — pure data record describing one item.
* :class:`MenuBar` — Textual widget; constructed with a list of
  :class:`MenuItem`. On click, looks up the item under the click
  coordinates and calls ``self.app.run_action(item.action_name)``.

The App composes a :class:`MenuBar` with its initial items, then
yields the tree + search widgets after it. Future menu items (filter,
view, etc.) add to the same item list without touching the bar's
rendering logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Sequence

from rich.text import Text

from textual import events
from textual.reactive import reactive
from textual.widgets import Static


__all__ = [
    "Alignment",
    "MenuBar",
    "MenuItem",
]


class Alignment(Enum):
    """Discriminator for menu-item placement on the bar.

    The bar splits its render space into a left-justified column (LEFT
    items in declared order) and a right-justified column (RIGHT items
    in declared order). The split avoids any layout-engine wrapping —
    one bar = one line.
    """

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class MenuItem:
    """One clickable menu-bar entry.

    ``label`` is the human-visible text (e.g. ``"Switch binary"``).
    ``action_name`` is the App-side action method name without the
    ``action_`` prefix (e.g. ``"open_binary_switcher"``); the bar
    routes the click through :meth:`textual.app.App.run_action` which
    looks up ``action_<action_name>`` on the active screen / app.
    ``hotkey`` is the single-character key bound to the same action
    on :class:`InspectorApp`'s BINDINGS — displayed as ``"b: <label>"``
    so the user sees both ways to trigger it. ``alignment`` discriminates
    whether the item goes on the left or right side of the bar.
    """

    label: str
    action_name: str
    hotkey: str
    alignment: Alignment = Alignment.LEFT


class MenuBar(Static):
    """One-line menu bar rendering :class:`MenuItem` records.

    Static widget (no expansion / scroll); mouse clicks are routed via
    :meth:`on_click` -- the bar inspects the click x-coordinate against
    the rendered segment offsets and dispatches the matching item's
    action through ``self.app.run_action``. Items with no hit zone
    under the click are no-ops.

    Re-renders on :attr:`items` change so callers may swap the item
    list dynamically (future filter / view items add via this same
    path).
    """

    DEFAULT_CSS: ClassVar[str] = """
    MenuBar {
        height: 1;
        background: $panel;
        color: $text;
    }
    """

    # Reactive item tuple: assignment triggers re-render.
    items: reactive[tuple[MenuItem, ...]] = reactive(tuple(), recompose=False)

    def __init__(
        self,
        items: Sequence[MenuItem] = (),
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        # Use object.__setattr__-style direct assignment so the watch
        # fires once compose has mounted; the reactive default is the
        # empty tuple so a stale-frame render is well-defined.
        self.items = tuple(items)
        # Per-render hit map: list of (start_col, end_col_exclusive,
        # action_name). Populated by :meth:`_render_text`.
        self._hit_map: list[tuple[int, int, str]] = []

    # --- reactive watcher ----------------------------------------------

    def watch_items(self, _old: tuple[MenuItem, ...], _new: tuple[MenuItem, ...]) -> None:
        """Re-render the bar when the item list changes."""
        self.update(self._render_text())

    def on_mount(self) -> None:
        """Initial render once the widget is mounted (size known)."""
        self.update(self._render_text())

    def on_resize(self, _event: events.Resize) -> None:
        """Re-render on resize so the right-column reflows to the new
        width and the hit map stays accurate."""
        self.update(self._render_text())

    # --- click dispatch ------------------------------------------------

    async def on_click(self, event: events.Click) -> None:
        """Look up the menu item under the click + run its action.

        The hit map is in display-column coordinates; ``event.x`` is
        the click column within the widget. A click outside any item
        is silently ignored (the bar has no other interaction).

        :meth:`textual.app.App.run_action` is a coroutine, so the
        click handler awaits it — Textual dispatches async ``on_click``
        handlers normally.
        """
        x = event.x
        for start, end, action_name in self._hit_map:
            if start <= x < end:
                await self.app.run_action(action_name)
                return

    # --- internal render ----------------------------------------------

    def _render_text(self) -> Text:
        """Compose the bar's :class:`rich.text.Text` and populate the
        hit map as a side effect.

        Left items render at columns ``[0..left_width)`` separated by a
        single space; right items render right-justified, separated by
        a single space, ending at the widget's right edge. The middle
        gap is space-padded.
        """
        self._hit_map = []
        left_items = [item for item in self.items if item.alignment is Alignment.LEFT]
        right_items = [item for item in self.items if item.alignment is Alignment.RIGHT]
        width = max(1, self.size.width or 80)

        left_segments: list[tuple[str, str]] = []  # (rendered, action_name)
        for idx, item in enumerate(left_items):
            if idx > 0:
                left_segments.append(("  ", ""))
            left_segments.append((_format_item(item), item.action_name))

        right_segments: list[tuple[str, str]] = []
        for idx, item in enumerate(right_items):
            if idx > 0:
                right_segments.append(("  ", ""))
            right_segments.append((_format_item(item), item.action_name))

        left_text = "".join(s for s, _ in left_segments)
        right_text = "".join(s for s, _ in right_segments)

        # Pad the gap so right items land flush against the widget edge.
        gap = max(1, width - len(left_text) - len(right_text))
        full_text = left_text + (" " * gap) + right_text

        # Build the hit map: walk left segments from column 0, right
        # segments from ``len(left_text) + gap``.
        col = 0
        for seg, action in left_segments:
            seg_len = len(seg)
            if action:
                self._hit_map.append((col, col + seg_len, action))
            col += seg_len
        col = len(left_text) + gap
        for seg, action in right_segments:
            seg_len = len(seg)
            if action:
                self._hit_map.append((col, col + seg_len, action))
            col += seg_len

        return Text(full_text)


def _format_item(item: MenuItem) -> str:
    """Format a :class:`MenuItem` as ``"<hotkey>: <label>"``.

    Single concern: the on-screen string form. The bar's hit map keys
    on this string's column extents, so any future styling (e.g.
    bracketing the hotkey) needs to keep the string-length contract
    intact or the click coordinates will drift.
    """
    return f"{item.hotkey}: {item.label}"
