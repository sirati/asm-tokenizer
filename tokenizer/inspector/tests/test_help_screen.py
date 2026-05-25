"""Pilot tests for the help-modal screen.

Structural confirmation that ``h`` opens :class:`HelpScreen` (a
:class:`textual.screen.ModalScreen`) embedding a :class:`BindingsTable`
that renders one row per active binding under the modal -- so adding a
binding to :class:`InspectorApp` or :class:`_InspectorTree` surfaces
here automatically. The drift-detection role of the previously-planned
``test_help_modal_contains_every_binding_description`` is structurally
subsumed by ``BindingsTable`` reading
:meth:`textual.screen.Screen.active_bindings` -- per W3-10 W4-AMENDED
that test is gone in favour of the structural assertions below.
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
from tokenizer.inspector._render._protocol import (
    BackendFactory,
    FunctionHandle,
    RenderBackend,
)


def _build_app(log_path: Path) -> InspectorApp:
    """Minimal :class:`InspectorApp` against a mock backend factory."""
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


def test_h_opens_help_modal():
    """``h`` while the tree has focus pushes a :class:`HelpScreen` modal."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(log_path)
            async with app.run_test() as pilot:
                # Tree has focus by default; the App-level ``h`` binding
                # fires because the tree no longer consumes it.
                stack_before = len(app.screen_stack)
                await pilot.press("h")
                await pilot.pause()
                assert len(app.screen_stack) == stack_before + 1
                assert isinstance(app.screen_stack[-1], HelpScreen)

    asyncio.run(runner())


def test_help_modal_escape_dismisses():
    """``escape`` on the help modal pops it via ``Screen.action_dismiss``."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(log_path)
            async with app.run_test() as pilot:
                await pilot.press("h")
                await pilot.pause()
                assert isinstance(app.screen_stack[-1], HelpScreen)

                await pilot.press("escape")
                await pilot.pause()
                # The modal is gone; the inspector screen is on top.
                assert not isinstance(app.screen_stack[-1], HelpScreen)

    asyncio.run(runner())


def test_help_screen_renders_bindings_table():
    """The modal hosts a :class:`BindingsTable` and renders the underlying
    screen's active bindings (one row per non-system action).

    Structural test: walk the rendered Rich table's rows and confirm at
    least one row per App-level + tree-level binding action description
    appears. The expected actions are pulled from the live BINDINGS
    lists at runtime so any future addition is automatically asserted
    on.
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(log_path)
            async with app.run_test() as pilot:
                await pilot.press("h")
                await pilot.pause()

                modal = app.screen_stack[-1]
                assert isinstance(modal, HelpScreen)
                table_widget = modal.query_one(_UnderlyingScreenBindingsTable)

                # Source screen MUST be the inspector screen (stack[-2]),
                # not the modal itself. Without the retarget the table
                # would only show the modal's own ``escape`` binding.
                source = table_widget._source_screen()
                assert source is app.screen_stack[-2]

                # Render produces a Rich Table; confirm it carries at
                # least one row for each non-modal action description.
                from rich.table import Table as RichTable

                rich_table = table_widget.render_bindings_table()
                assert isinstance(rich_table, RichTable)

                # Walk the table's row cells, concatenate into a single
                # haystack, and assert each expected action description
                # appears at least once. Use the action descriptions as
                # listed in the live BINDINGS so this test stays in
                # lockstep with the binding inventory.
                rendered_text = _flatten_table_text(rich_table)
                expected_descriptions = _collect_active_binding_descriptions(source)
                assert expected_descriptions, "no source bindings to render"
                for description in expected_descriptions:
                    assert description in rendered_text, (
                        f"binding description {description!r} missing from "
                        f"rendered help table"
                    )

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_table_text(table) -> str:
    """Concatenate every cell in ``table`` into a single string for assertion."""
    from rich.console import Console

    console = Console(record=True, file=None, width=200)
    with console.capture() as capture:
        console.print(table)
    return capture.get()


def _collect_active_binding_descriptions(source) -> list[str]:
    """Return the non-empty descriptions of every non-system binding."""
    out: list[str] = []
    seen: set[str] = set()
    for _key, active in source.active_bindings.items():
        binding = active.binding
        if binding.system:
            continue
        if not binding.description:
            continue
        if binding.description in seen:
            continue
        seen.add(binding.description)
        out.append(binding.description)
    return out
