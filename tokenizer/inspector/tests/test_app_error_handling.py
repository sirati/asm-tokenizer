"""Pilot tests for the inspector app's expand dispatcher + failed glyph.

This file closes the Phase 3b coverage gap that let the
``assemble_failed_glyph`` signature mismatch ship before being caught by
a hotfix commit. The tests:

* drive ``InspectorApp`` via :meth:`textual.app.App.run_test` (a Pilot)
* force a root :class:`FunctionNode`'s ``expand`` to raise
* assert the dispatcher marks the node failed, paints ``[*]``, attaches
  the exception repr as a dim-red error child, logs a traceback, and
  retries on collapse + re-expand

The file is gated on ``pytest.importorskip("textual")`` so the default
``nix develop`` shell (textual-free) shows it as SKIPPED rather than as
an import failure. It runs green under the ``tui-inspector`` flake-app
python env where textual is installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from rich.style import Style
from rich.text import Text

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._app import InspectorApp
from tokenizer.inspector._render._protocol import (
    BackendFactory,
    FunctionHandle,
    RenderBackend,
)
from tokenizer.inspector._tree_model import FunctionNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_factory(names: list[str]) -> MagicMock:
    """Build a minimal mock :class:`BackendFactory`.

    Only ``.handles`` + ``.make`` + ``.close`` are spec'd; ``make``
    returns a RenderBackend-spec'd mock with no variants so any
    accidental expand path produces an empty (but valid) tree row.
    """
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
    """Construct an :class:`InspectorApp` for tests against a mock factory."""
    return InspectorApp(factory=_make_factory(names), log_path=log_path)


def _root_tree_node(app: InspectorApp):
    """Reach the first FunctionNode tree-node mounted under the tree root."""
    from tokenizer.inspector._app import _InspectorTree

    tree = app.query_one("#tree", _InspectorTree)
    return tree, tree.root.children[0]


def _make_raising_expand(exc: Exception):
    """Spy that records its call args + raises ``exc`` when invoked."""
    spy = MagicMock(side_effect=exc)
    return spy


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_expand_raises_marks_node_failed():
    """Audit-mandated case (a): expand raises -> node flips is_failed=True."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                _tree, fn_tree_node = _root_tree_node(app)
                fn_model: FunctionNode = fn_tree_node.data
                # Force this node's bound expand to raise.
                fn_model.expand = _make_raising_expand(RuntimeError("boom"))
                fn_tree_node.expand()
                await pilot.pause()

                assert fn_model.is_failed is True
                assert len(fn_tree_node.children) == 1
                err_leaf = fn_tree_node.children[0]
                plain = err_leaf.label.plain
                assert "RuntimeError" in plain
                assert "boom" in plain

    asyncio.run(runner())


def test_failed_node_paints_marker_glyph():
    """Audit-mandated case (b): render_label paints ``[*]`` without TypeError.

    This is the test that would have caught the original
    ``assemble_failed_glyph`` signature mismatch hotfixed post-merge.
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                tree, fn_tree_node = _root_tree_node(app)
                fn_model: FunctionNode = fn_tree_node.data
                fn_model.expand = _make_raising_expand(RuntimeError("boom"))
                fn_tree_node.expand()
                await pilot.pause()

                # Direct render_label call -- mirrors the per-paint code
                # path in the Tree widget.
                rendered = tree.render_label(
                    fn_tree_node, base_style=Style(), style=Style()
                )
                assert isinstance(rendered, Text)
                assert "[*]" in rendered.plain

    asyncio.run(runner())


def test_collapse_expand_retries_after_failure():
    """Audit-mandated case (c): collapse + expand clears is_failed + reruns."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                _tree, fn_tree_node = _root_tree_node(app)
                fn_model: FunctionNode = fn_tree_node.data

                # Step 1: fail.
                fn_model.expand = _make_raising_expand(RuntimeError("boom"))
                fn_tree_node.expand()
                await pilot.pause()
                assert fn_model.is_failed is True
                assert len(fn_tree_node.children) == 1

                # Step 2: swap expand for a no-fail spy returning empty,
                # then collapse + re-expand to drive the retry path.
                retry_spy = MagicMock(return_value=[])
                fn_model.expand = retry_spy

                fn_tree_node.collapse()
                await pilot.pause()
                fn_tree_node.expand()
                await pilot.pause()

                assert fn_model.is_failed is False
                assert len(fn_tree_node.children) == 0
                retry_spy.assert_called_once()
                # The dispatcher calls expand() arg-less.
                call = retry_spy.call_args
                assert call.args == ()
                assert call.kwargs == {}

    asyncio.run(runner())


def test_log_file_carries_traceback():
    """The log file must capture the exception class + a traceback marker."""

    async def runner() -> Path:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "nested" / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                _tree, fn_tree_node = _root_tree_node(app)
                fn_model: FunctionNode = fn_tree_node.data
                fn_model.expand = _make_raising_expand(RuntimeError("boom"))
                fn_tree_node.expand()
                await pilot.pause()

                # Flush handlers so the file content is on disk by the
                # time we read it.
                for handler in app._log.handlers:
                    handler.flush()

                assert log_path.exists(), f"log file not created at {log_path}"
                contents = log_path.read_text(encoding="utf-8")
                assert "RuntimeError" in contents
                assert (
                    "Traceback (most recent call last)" in contents
                    or 'File "' in contents
                )
                return log_path

        return log_path  # unreachable, keeps mypy happy

    asyncio.run(runner())


def test_quit_binding_returns_cleanly():
    """``q`` exits the pilot with ``app.return_value is None``."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app([], log_path)
            async with app.run_test() as pilot:
                await pilot.press("q")
                await pilot.pause()
            # App[None] -- return_value should be None on clean quit.
            assert app.return_value is None

    asyncio.run(runner())


def test_search_input_focus_and_clear():
    """``/`` reveals the search Input and ``escape`` hides it again."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                from textual.widgets import Input

                search = app.query_one("#search", Input)
                # Hidden by default per the inspector CSS.
                assert "visible" not in search.classes

                await pilot.press("slash")
                await pilot.pause()
                assert "visible" in search.classes

                await pilot.press("escape")
                await pilot.pause()
                assert "visible" not in search.classes

    asyncio.run(runner())
