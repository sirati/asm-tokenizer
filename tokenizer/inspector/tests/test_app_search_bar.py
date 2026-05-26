"""Pilot tests for the inline :class:`SearchBar` widget.

Structural confirmation that:

* ``s`` reveals the inline search bar + gives the Input focus.
* The bar stays visible while the user types (no immediate jump).
* Enter on a name-substring hit moves the tree cursor to the first
  matching :class:`FunctionNode` row.
* Escape inside the bar hides it again (and the tree regains focus).
* The new ``s`` binding shows up in the help modal's bindings table.

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


def test_typing_does_not_jump_immediately():
    """Substring keystrokes accumulate in the Input without moving the cursor."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["calloc", "malloc", "free"], log_path)
            async with app.run_test() as pilot:
                from textual.widgets import Input

                from tokenizer.inspector._app import _InspectorTree

                tree = app.query_one("#tree", _InspectorTree)
                # Park the cursor on the LAST function row so a
                # potential jump-to-first-match would visibly move it.
                tree.move_cursor(tree.root.children[-1], animate=False)
                await pilot.pause()
                cursor_before = tree.cursor_node

                await pilot.press("s")
                await pilot.pause()

                bar = app.query_one("#search-bar", SearchBar)
                # Type a substring that matches the FIRST function
                # ("calloc"); the cursor must NOT jump until Enter.
                for key in "cal":
                    await pilot.press(key)
                await pilot.pause()

                input_widget = bar.query_one(Input)
                assert input_widget.value == "cal"
                assert "visible" in bar.classes
                # Cursor unchanged -- the jump only happens on Enter.
                assert tree.cursor_node is cursor_before

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

    asyncio.run(runner())
