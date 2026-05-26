"""Pilot tests for the inspector top menu bar.

Covers:

* :class:`MenuBar` renders one item per declared :class:`MenuItem`.
* Click on the help item dispatches to ``action_open_help`` — modal
  appears in the screen stack.
* Click on the switch-binary item dispatches to
  ``action_open_binary_switcher`` — invocation goes through, even if
  the resulting modal is just the Phase-1 stub.
* Hotkey hint string for each item is present in the rendered bar
  (so the user sees both ways to trigger the action).

Gated on ``pytest.importorskip("textual")`` so the default ``nix
develop`` shell shows it as SKIPPED rather than as an import failure.
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
from tokenizer.inspector._app._help_dialog import HelpScreen
from tokenizer.inspector._app._menu_bar import (
    Alignment,
    MenuBar,
    MenuItem,
)
from tokenizer.inspector._render._protocol import (
    BackendFactory,
    FunctionHandle,
    RenderBackend,
)


def _build_app(log_path: Path) -> InspectorApp:
    handles = [FunctionHandle(arm=SectionKind.MATCHED, idx=0, name="main")]
    factory = MagicMock(spec=BackendFactory)
    factory.handles = handles
    backend = MagicMock(spec=RenderBackend)
    backend.handle = handles[0]
    backend.closed = False
    backend.variants.return_value = []
    backend.blocks.return_value = []
    backend.render_block.return_value = ()
    factory.make.return_value = backend
    return InspectorApp(factory=factory, log_path=log_path)


def test_menu_bar_renders_items():
    """:class:`MenuBar` renders ``Switch binary`` + ``help`` rows."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(log_path)
            async with app.run_test() as pilot:
                bar = app.query_one("#menubar", MenuBar)
                rendered = bar.render()
                # The renderable is a rich Text; convert to plain str.
                text = (
                    rendered.plain
                    if hasattr(rendered, "plain")
                    else str(rendered)
                )
                assert "Switch binary" in text
                assert "help" in text
                # Hotkey hints rendered alongside labels.
                assert "b:" in text
                assert "h:" in text

    asyncio.run(runner())


def test_menu_bar_help_click_opens_help_modal():
    """Click on the ``help`` item invokes ``open_help`` action and pushes
    the :class:`HelpScreen` modal."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(log_path)
            async with app.run_test() as pilot:
                # Find the help item in the bar's hit map and click its
                # midpoint column.
                bar = app.query_one("#menubar", MenuBar)
                # Force a layout pass so the hit map populates.
                bar.update(bar._render_text())
                help_hit = next(
                    (h for h in bar._hit_map if h[2] == "open_help"),
                    None,
                )
                assert help_hit is not None, "help hit not found in bar"
                mid_col = (help_hit[0] + help_hit[1]) // 2
                await pilot.click("#menubar", offset=(mid_col, 0))
                await pilot.pause()
                assert isinstance(app.screen_stack[-1], HelpScreen)

    asyncio.run(runner())


def test_menu_bar_switch_binary_click_runs_action():
    """Click on the ``Switch binary`` item invokes ``open_binary_switcher``.

    The Phase-1 landing routes through the action method which may push
    a modal or no-op; the assertion is just that the action ran (the
    bar dispatched the click successfully and did not raise).
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(log_path)
            async with app.run_test() as pilot:
                bar = app.query_one("#menubar", MenuBar)
                bar.update(bar._render_text())
                switch_hit = next(
                    (
                        h
                        for h in bar._hit_map
                        if h[2] == "open_binary_switcher"
                    ),
                    None,
                )
                assert switch_hit is not None, (
                    "switch-binary hit not found in bar"
                )
                mid_col = (switch_hit[0] + switch_hit[1]) // 2
                # The action should run without raising; what it does
                # (push modal, no-op) is the later phases' concern.
                await pilot.click("#menubar", offset=(mid_col, 0))
                await pilot.pause()

    asyncio.run(runner())


def test_menu_item_alignment_split():
    """Items are partitioned by :class:`Alignment` — LEFT items render
    flush left, RIGHT items render flush against the widget's right edge.

    Runs inside a :class:`textual.app.App` test harness so the bar's
    ``self.size`` is populated by Textual's layout pass.
    """

    from textual.app import App, ComposeResult

    items = (
        MenuItem(label="a", action_name="noop_a", hotkey="1", alignment=Alignment.LEFT),
        MenuItem(label="z", action_name="noop_z", hotkey="9", alignment=Alignment.RIGHT),
    )

    class _MenuApp(App[None]):
        def compose(self) -> ComposeResult:
            yield MenuBar(items=items, id="bar")

    async def runner() -> None:
        async with _MenuApp().run_test(size=(40, 24)) as pilot:
            bar = pilot.app.query_one("#bar", MenuBar)
            # Force a layout pass.
            await pilot.pause()
            text = bar._render_text().plain
            assert text.lstrip().startswith("1: a")
            assert text.rstrip().endswith("9: z")
            assert len(text) == 40

    asyncio.run(runner())
