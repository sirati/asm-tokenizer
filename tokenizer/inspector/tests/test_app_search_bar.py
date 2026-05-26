"""Pilot tests for the inline :class:`SearchBar` widget.

Structural confirmation that:

* ``s`` reveals the inline search bar + gives the Input focus.
* Each keystroke LIVE-PREVIEWS the first matching FunctionNode row
  (the bar moves the tree cursor onto the match without committing).
* Enter on a name-substring hit moves the tree cursor to the first
  matching :class:`FunctionNode` row + appends the needle to the App's
  search history.
* Escape restores the cursor to the row it occupied at open-time and
  hides the bar.
* The App's ``n`` / ``shift+n`` walker steps through matches with
  wrap-around.
* Tab restores the most-recent saved query into an empty Input.
* Up / Down arrow keys walk older / newer entries of the App's
  search history; Down past the most-recent entry clears the Input.
* The new ``s`` / ``n`` / ``shift+n`` bindings show up in the help
  modal's bindings table.

The file is gated on ``pytest.importorskip("textual")`` so the default
``nix develop`` shell (textual-free) shows it as SKIPPED rather than as
an import failure.
"""

from __future__ import annotations

import pytest


pytest.importorskip("textual")

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._app import InspectorApp
from tokenizer.inspector._app._help_dialog import (
    HelpScreen,
    _UnderlyingScreenBindingsTable,
)
from tokenizer.inspector._app._search_bar import SearchBar
from tokenizer.inspector._render._protocol import (
    BackendFactory,
    FunctionHandle,
    RenderBackend,
)


def _make_factory(names: list[str]) -> MagicMock:
    """Build a minimal mock :class:`BackendFactory` with one handle per name."""
    handles = [
        FunctionHandle(arm=SectionKind.MATCHED, idx=i, name=name)
        for i, name in enumerate(names)
    ]
    factory = MagicMock(spec=BackendFactory)
    factory.handles = handles
    backend = MagicMock(spec=RenderBackend)
    backend.handle = handles[0] if handles else None
    backend.closed = False
    backend.variants.return_value = []
    backend.blocks.return_value = []
    backend.render_block.return_value = ()
    factory.make.return_value = backend
    return factory


def _build_app(names: list[str], log_path: Path) -> InspectorApp:
    return InspectorApp(factory=_make_factory(names), log_path=log_path)


# ---------------------------------------------------------------------------
# Pilot tests
# ---------------------------------------------------------------------------


def test_s_opens_search_bar_and_focuses_input():
    """``s`` flips the SearchBar to visible and focuses the inner Input."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["alpha", "beta"], log_path)
            async with app.run_test() as pilot:
                from textual.widgets import Input

                bar = app.query_one("#search-bar", SearchBar)
                assert "visible" not in bar.classes

                await pilot.press("s")
                await pilot.pause()

                assert "visible" in bar.classes
                # The Input nested inside the SearchBar holds focus
                # immediately after open, so the next keystrokes land
                # in the search buffer (not the tree).
                input_widget = bar.query_one(Input)
                assert app.focused is input_widget

    asyncio.run(runner())


def test_typing_live_previews_first_match():
    """Each keystroke moves the cursor to the first matching row (live preview)."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["calloc", "malloc", "free"], log_path)
            async with app.run_test() as pilot:
                from textual.widgets import Input

                from tokenizer.inspector._app import _InspectorTree

                tree = app.query_one("#tree", _InspectorTree)
                # Park the cursor on the LAST function row so the
                # preview jump to "calloc" (first row) is observable.
                tree.move_cursor(tree.root.children[-1], animate=False)
                await pilot.pause()

                await pilot.press("s")
                await pilot.pause()

                bar = app.query_one("#search-bar", SearchBar)
                # Type a substring matching the FIRST function. Each
                # keystroke previews the first match; we let Textual
                # pump events between presses.
                for key in "cal":
                    await pilot.press(key)
                # call_after_refresh defers the cursor move; pause
                # multiple times so the queued moves all land.
                await pilot.pause()
                await pilot.pause()

                input_widget = bar.query_one(Input)
                assert input_widget.value == "cal"
                # Bar stays open; the user hasn't committed yet.
                assert "visible" in bar.classes
                # Live preview moved the cursor to the matching row.
                calloc_row = tree.root.children[0]
                assert tree.cursor_node is calloc_row

    asyncio.run(runner())


def test_enter_jumps_cursor_to_first_matching_function():
    """Enter on a substring hit moves the cursor to the matching row."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            # Three names; substring "loc" matches "calloc" + "malloc"
            # in declared order. The bar's first-match policy picks
            # "calloc" (the earlier sibling).
            app = _build_app(["calloc", "malloc", "free"], log_path)
            async with app.run_test() as pilot:
                from tokenizer.inspector._app import _InspectorTree

                tree = app.query_one("#tree", _InspectorTree)
                free_row = tree.root.children[2]
                tree.move_cursor(free_row, animate=False)
                await pilot.pause()
                assert tree.cursor_node is free_row

                await pilot.press("s")
                await pilot.pause()
                for key in "loc":
                    await pilot.press(key)
                await pilot.press("enter")
                # call_after_refresh defers move_cursor by one frame;
                # pause twice so the deferred call lands.
                await pilot.pause()
                await pilot.pause()

                calloc_row = tree.root.children[0]
                assert tree.cursor_node is calloc_row

                # Bar dismissed on successful jump.
                bar = app.query_one("#search-bar", SearchBar)
                assert "visible" not in bar.classes

    asyncio.run(runner())


def test_enter_on_miss_keeps_bar_visible():
    """Enter with a needle that doesn't match anything leaves the bar open."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["calloc", "malloc"], log_path)
            async with app.run_test() as pilot:
                from textual.widgets import Input

                await pilot.press("s")
                await pilot.pause()
                for key in "zzz":
                    await pilot.press(key)
                await pilot.press("enter")
                await pilot.pause()

                bar = app.query_one("#search-bar", SearchBar)
                # Miss: bar stays visible so the user can refine.
                assert "visible" in bar.classes
                assert bar.query_one(Input).value == "zzz"

    asyncio.run(runner())


def test_escape_dismisses_bar_and_restores_tree_focus():
    """Escape inside the SearchBar closes it and returns focus to the tree."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["alpha", "beta"], log_path)
            async with app.run_test() as pilot:
                from tokenizer.inspector._app import _InspectorTree

                tree = app.query_one("#tree", _InspectorTree)
                cursor_before = tree.cursor_node

                await pilot.press("s")
                await pilot.pause()
                # Type something so we can also assert the buffer
                # clears on dismissal.
                for key in "alp":
                    await pilot.press(key)
                await pilot.pause()

                bar = app.query_one("#search-bar", SearchBar)
                assert "visible" in bar.classes

                await pilot.press("escape")
                await pilot.pause()

                assert "visible" not in bar.classes
                # Tree regained focus; cursor unchanged.
                assert app.focused is tree
                assert tree.cursor_node is cursor_before

    asyncio.run(runner())


def test_case_insensitive_substring_match():
    """Mixed-case needle still matches a lower-case function name."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["Calloc", "free"], log_path)
            async with app.run_test() as pilot:
                from tokenizer.inspector._app import _InspectorTree

                tree = app.query_one("#tree", _InspectorTree)
                tree.move_cursor(tree.root.children[1], animate=False)
                await pilot.pause()

                await pilot.press("s")
                await pilot.pause()
                # Lower-case needle should still match "Calloc".
                for key in "ca":
                    await pilot.press(key)
                await pilot.press("enter")
                await pilot.pause()
                await pilot.pause()

                assert tree.cursor_node is tree.root.children[0]

    asyncio.run(runner())


def test_help_modal_lists_search_binding():
    """The ``s: Search`` binding surfaces in the help modal's bindings table."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test(size=(140, 60)) as pilot:
                await pilot.press("h")
                await pilot.pause()

                modal = app.screen_stack[-1]
                assert isinstance(modal, HelpScreen)
                table_widget = modal.query_one(_UnderlyingScreenBindingsTable)

                from rich.console import Console

                console = Console(record=True, file=None, width=200)
                with console.capture() as capture:
                    console.print(table_widget.render_bindings_table())
                rendered = capture.get()
                # The binding's description is "Search"; the help modal
                # renders one row per non-system binding action.
                assert "Search" in rendered
                # The ``n`` / ``shift+n`` bindings landed alongside the
                # search bar refinements (next / previous match).
                assert "Next match" in rendered
                assert "Previous match" in rendered

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Refinements landed alongside the live-preview / history / n-step rollout
# ---------------------------------------------------------------------------


def test_escape_restores_cursor_after_live_preview():
    """Escape rewinds the cursor that the live preview moved away from."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["calloc", "malloc", "free"], log_path)
            async with app.run_test() as pilot:
                from tokenizer.inspector._app import _InspectorTree

                tree = app.query_one("#tree", _InspectorTree)
                # Park on the LAST function so the preview jumps off
                # observably + the restore returns there.
                free_row = tree.root.children[-1]
                tree.move_cursor(free_row, animate=False)
                await pilot.pause()
                assert tree.cursor_node is free_row

                await pilot.press("s")
                await pilot.pause()
                for key in "cal":
                    await pilot.press(key)
                await pilot.pause()
                await pilot.pause()
                # Live preview moved off ``free`` to ``calloc``.
                assert tree.cursor_node is tree.root.children[0]

                await pilot.press("escape")
                await pilot.pause()
                await pilot.pause()

                bar = app.query_one("#search-bar", SearchBar)
                assert "visible" not in bar.classes
                # Cursor restored to the open-time row.
                assert tree.cursor_node is free_row

    asyncio.run(runner())


def test_enter_appends_to_search_history_dedup_of_last():
    """Each Enter appends to history; consecutive identical Enters dedup."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["calloc", "malloc", "free"], log_path)
            async with app.run_test() as pilot:
                assert app._search_history == []

                # First search: "cal" -> hit
                await pilot.press("s")
                await pilot.pause()
                for key in "cal":
                    await pilot.press(key)
                await pilot.press("enter")
                await pilot.pause()
                assert app._search_history == ["cal"]

                # Second search: same needle -> dedup-of-last skip.
                await pilot.press("s")
                await pilot.pause()
                for key in "cal":
                    await pilot.press(key)
                await pilot.press("enter")
                await pilot.pause()
                assert app._search_history == ["cal"]

                # Third search: distinct needle -> append.
                await pilot.press("s")
                await pilot.pause()
                for key in "mal":
                    await pilot.press(key)
                await pilot.press("enter")
                await pilot.pause()
                assert app._search_history == ["cal", "mal"]

                # Fourth search: a MISS still appends to history (per
                # the user's spec -- saved every time Enter is pressed).
                await pilot.press("s")
                await pilot.pause()
                for key in "zzz":
                    await pilot.press(key)
                await pilot.press("enter")
                await pilot.pause()
                assert app._search_history == ["cal", "mal", "zzz"]

    asyncio.run(runner())


def test_n_steps_through_matches_with_wraparound():
    """``n`` walks forward through every match of the most-recent needle."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            # "loc" matches "calloc" + "malloc" (declared rows 0, 1).
            app = _build_app(["calloc", "malloc", "free"], log_path)
            async with app.run_test() as pilot:
                from tokenizer.inspector._app import _InspectorTree

                tree = app.query_one("#tree", _InspectorTree)

                # Seed the history via a real search session.
                await pilot.press("s")
                await pilot.pause()
                for key in "loc":
                    await pilot.press(key)
                await pilot.press("enter")
                await pilot.pause()
                await pilot.pause()

                calloc = tree.root.children[0]
                malloc = tree.root.children[1]
                # Enter parked the cursor on the first match.
                assert tree.cursor_node is calloc

                # ``n`` -> next match (malloc).
                await pilot.press("n")
                await pilot.pause()
                await pilot.pause()
                assert tree.cursor_node is malloc

                # ``n`` -> wrap to first match (calloc).
                await pilot.press("n")
                await pilot.pause()
                await pilot.pause()
                assert tree.cursor_node is calloc

                # ``shift+n`` -> previous (wraps to malloc).
                await pilot.press("N")
                await pilot.pause()
                await pilot.pause()
                assert tree.cursor_node is malloc

    asyncio.run(runner())


def test_n_is_noop_when_history_is_empty():
    """``n`` does nothing when no search has been committed yet."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["calloc", "malloc"], log_path)
            async with app.run_test() as pilot:
                from tokenizer.inspector._app import _InspectorTree

                tree = app.query_one("#tree", _InspectorTree)
                cursor_before = tree.cursor_node
                assert app._search_history == []

                await pilot.press("n")
                await pilot.pause()

                # Cursor untouched; history still empty.
                assert tree.cursor_node is cursor_before
                assert app._search_history == []

    asyncio.run(runner())


def test_tab_restores_last_query_into_empty_input():
    """Tab populates an empty Input with the most-recent history entry."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["calloc", "malloc"], log_path)
            async with app.run_test() as pilot:
                from textual.widgets import Input

                # Seed history.
                await pilot.press("s")
                await pilot.pause()
                for key in "cal":
                    await pilot.press(key)
                await pilot.press("enter")
                await pilot.pause()
                assert app._search_history == ["cal"]

                # Re-open: Input is empty; Tab should restore "cal".
                await pilot.press("s")
                await pilot.pause()
                bar = app.query_one("#search-bar", SearchBar)
                input_widget = bar.query_one(Input)
                assert input_widget.value == ""

                await pilot.press("tab")
                await pilot.pause()
                assert input_widget.value == "cal"

    asyncio.run(runner())


def test_up_down_walk_search_history():
    """Up/Down walk previous/next entries in the App's search history."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["calloc", "malloc", "free"], log_path)
            async with app.run_test() as pilot:
                from textual.widgets import Input

                # Seed history with two entries: "cal" then "mal".
                for needle in ("cal", "mal"):
                    await pilot.press("s")
                    await pilot.pause()
                    for key in needle:
                        await pilot.press(key)
                    await pilot.press("enter")
                    await pilot.pause()
                assert app._search_history == ["cal", "mal"]

                # Open fresh: Up = most-recent ("mal"), Up = older
                # ("cal"), Down = newer ("mal"), Down = clear.
                await pilot.press("s")
                await pilot.pause()
                bar = app.query_one("#search-bar", SearchBar)
                input_widget = bar.query_one(Input)
                assert input_widget.value == ""

                await pilot.press("up")
                await pilot.pause()
                assert input_widget.value == "mal"

                await pilot.press("up")
                await pilot.pause()
                assert input_widget.value == "cal"

                # Past the oldest entry: no-op.
                await pilot.press("up")
                await pilot.pause()
                assert input_widget.value == "cal"

                await pilot.press("down")
                await pilot.pause()
                assert input_widget.value == "mal"

                # Past the newest: input clears + walk resets.
                await pilot.press("down")
                await pilot.pause()
                assert input_widget.value == ""

    asyncio.run(runner())
