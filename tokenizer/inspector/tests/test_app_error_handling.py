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

Sync wrappers (``asyncio.run``) are used in lieu of ``@pytest.mark.asyncio``
because pytest-asyncio is not in the tui-inspector env either; the plan
does not require it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from rich.style import Style
from rich.text import Text

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._app import InspectorApp
from tokenizer.inspector._tree_model import FunctionNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(*, matched_count: int, names: list[str]) -> Any:
    """Build a minimal mock ``BinaryDataset`` that drives ``compose``.

    Only the attributes the inspector's ``compose`` + dispatcher touch
    are populated: ``matched_count``, ``matched_func_names``,
    ``vocab_manager``. Everything else routes through the (un-called)
    expand path.
    """
    dataset = MagicMock()
    dataset.matched_count = matched_count
    dataset.matched_func_names = names
    dataset.vocab_manager = MagicMock(name="vocab_manager")
    return dataset


def _build_app(matched_count: int, names: list[str], log_path: Path) -> InspectorApp:
    """Construct an ``InspectorApp`` for tests with a mock dataset + session."""
    dataset = _make_dataset(matched_count=matched_count, names=names)
    session = MagicMock(name="session")
    return InspectorApp(dataset=dataset, session=session, log_path=log_path)


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
    """Audit-mandated case (a): expand raises → node flips is_failed=True."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(1, ["main"], log_path)
            async with app.run_test() as pilot:
                tree, fn_tree_node = _root_tree_node(app)
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
            app = _build_app(1, ["main"], log_path)
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
            app = _build_app(1, ["main"], log_path)
            async with app.run_test() as pilot:
                tree, fn_tree_node = _root_tree_node(app)
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
                # The dispatcher passes session positionally + vocab_manager
                # as a keyword (see InspectorApp._on_node_expanded).
                call = retry_spy.call_args
                assert call.args == (app._session,)
                assert call.kwargs == {"vocab_manager": app._dataset.vocab_manager}

    asyncio.run(runner())


def test_log_file_carries_traceback():
    """The log file must capture the exception class + a traceback marker."""

    async def runner() -> Path:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "nested" / "tui.log"
            app = _build_app(1, ["main"], log_path)
            async with app.run_test() as pilot:
                tree, fn_tree_node = _root_tree_node(app)
                fn_model: FunctionNode = fn_tree_node.data
                fn_model.expand = _make_raising_expand(RuntimeError("boom"))
                fn_tree_node.expand()
                await pilot.pause()

                # Flush handlers so the file content is on disk by the
                # time we read it. The logger is dedicated, so iterating
                # ``app._log.handlers`` is sufficient.
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
            app = _build_app(0, [], log_path)
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
            app = _build_app(1, ["main"], log_path)
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
