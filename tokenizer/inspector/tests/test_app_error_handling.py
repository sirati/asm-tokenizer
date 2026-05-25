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
    BlockKind,
    FunctionHandle,
    RenderBackend,
)
from tokenizer.inspector._tree_model import AsmLeaf, BlockNode, FunctionNode


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


def test_left_arrow_at_scroll_zero_moves_cursor_to_parent():
    """Left arrow at ``scroll_offset.x == 0`` climbs to the parent row.

    Standard file-tree TUI affordance: the arrow pans while there is
    horizontal slack, then becomes cursor-to-parent once the row is
    flush-left.
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                tree, fn_tree_node = _root_tree_node(app)
                fn_model: FunctionNode = fn_tree_node.data
                # Stub the function's expand to attach one synthetic
                # AsmLeaf child so we have a non-root cursor target.
                child_leaf = AsmLeaf(text="stub-line")
                fn_model.expand = MagicMock(return_value=[child_leaf])
                fn_tree_node.expand()
                await pilot.pause()
                assert len(fn_tree_node.children) == 1

                # Park the cursor on the child + confirm scroll_x is 0.
                child_tree_node = fn_tree_node.children[0]
                tree.move_cursor(child_tree_node, animate=False)
                await pilot.pause()
                assert tree.scroll_offset.x == 0
                assert tree.cursor_node is child_tree_node

                # Left arrow at scroll-zero -> parent function row.
                await pilot.press("left")
                await pilot.pause()
                assert tree.cursor_node is fn_tree_node

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


# ---------------------------------------------------------------------------
# Per-row horizontal-scroll memory + cursor-aware auto-adjust + conditional
# right-arrow expand. The tests below treat the tree's rows like editor
# lines: each row owns its own ``remembered_scroll_x``; the cursor move
# restores it; manual pan saves it; the right arrow only pans when the
# row would otherwise hide content past the viewport.
# ---------------------------------------------------------------------------


async def _expand_function_with_children(
    app: "InspectorApp", pilot, children: list
):
    """Stub the root function's expand to return ``children`` + drive expand.

    The dispatcher attaches children asynchronously (it runs in response
    to the :class:`Tree.NodeExpanded` message); the helper yields once
    via ``pilot.pause()`` so the returned ``child_tree_nodes`` tuple
    contains the attached :class:`textual.widgets._tree.TreeNode`
    instances.

    Returns ``(tree, fn_tree_node, child_tree_nodes)``.
    """
    tree, fn_tree_node = _root_tree_node(app)
    fn_model: FunctionNode = fn_tree_node.data
    fn_model.expand = MagicMock(return_value=children)
    fn_tree_node.expand()
    await pilot.pause()
    return tree, fn_tree_node, list(fn_tree_node.children)


def test_remembered_scroll_x_default_is_zero():
    """Every model node ships with ``remembered_scroll_x == 0`` by default."""

    long_leaf = AsmLeaf(text="x" * 200)
    short_leaf = AsmLeaf(text="hi")
    assert long_leaf.remembered_scroll_x == 0
    assert short_leaf.remembered_scroll_x == 0


def test_pan_right_on_long_row_saves_remembered_scroll_x():
    """Manual pan on the cursor row persists its ``scroll_x`` onto the model."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                long_leaf = AsmLeaf(text="x" * 200)
                tree, _fn, child_tree_nodes = (
                    await _expand_function_with_children(app, pilot, [long_leaf])
                )

                tree.move_cursor(child_tree_nodes[0], animate=False)
                await pilot.pause()
                # Restore path leaves scroll_x at the remembered value (0).
                assert tree.scroll_offset.x == 0
                assert long_leaf.remembered_scroll_x == 0

                # Right arrow on a long row -> pans + saves.
                await pilot.press("right")
                await pilot.pause()
                assert tree.scroll_offset.x == 1
                assert long_leaf.remembered_scroll_x == 1

                # A few more pans -> the saved value tracks each step.
                await pilot.press("right")
                await pilot.press("right")
                await pilot.pause()
                assert tree.scroll_offset.x == 3
                assert long_leaf.remembered_scroll_x == 3

    asyncio.run(runner())


def test_cursor_move_restores_per_row_scroll_memory():
    """Moving cursor across rows restores each row's remembered ``scroll_x``."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                long_a = AsmLeaf(text="a" * 200)
                long_b = AsmLeaf(text="b" * 200)
                tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [long_a, long_b]
                )
                row_a, row_b = children

                # Pan row A right by 5 columns.
                tree.move_cursor(row_a, animate=False)
                await pilot.pause()
                for _ in range(5):
                    await pilot.press("right")
                await pilot.pause()
                assert long_a.remembered_scroll_x == 5

                # Move to row B -- its remembered_scroll_x is 0, viewport
                # restores to 0 (long row, auto-adjust does not fire).
                tree.move_cursor(row_b, animate=False)
                await pilot.pause()
                assert tree.scroll_offset.x == 0
                assert long_b.remembered_scroll_x == 0

                # Back to row A -- viewport restores to row A's remembered 5.
                tree.move_cursor(row_a, animate=False)
                await pilot.pause()
                assert tree.scroll_offset.x == 5
                # Saved value persists across the round-trip.
                assert long_a.remembered_scroll_x == 5

    asyncio.run(runner())


def test_cursor_move_auto_adjusts_when_destination_row_off_screen():
    """Auto-adjust pulls a short destination row into view without saving.

    Pan row A far right (scroll_x large); cursor to a short row B whose
    cell_len is much smaller than the current scroll_x. The viewport
    auto-adjusts so row B's content stays visible (effective scroll_x =
    0 when the label fits entirely). Row B's remembered_scroll_x stays
    at the model default 0 (the user never panned on B).
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                long_a = AsmLeaf(text="a" * 200)
                short_b = AsmLeaf(text="hi")
                tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [long_a, short_b]
                )
                row_a, row_b = children

                # Pan row A by 50 columns -- saved as A's remembered.
                tree.move_cursor(row_a, animate=False)
                await pilot.pause()
                for _ in range(50):
                    await pilot.press("right")
                await pilot.pause()
                assert long_a.remembered_scroll_x == 50

                # Move to short row B. cell_len("hi") == 2 << viewport
                # width (80), so the auto-adjust clamps to flush-left.
                tree.move_cursor(row_b, animate=False)
                await pilot.pause()
                # Effective scroll_x is 0 (short row fits at flush-left).
                assert tree.scroll_offset.x == 0
                # Auto-adjust does NOT save -- the row's remembered stays 0.
                assert short_b.remembered_scroll_x == 0

                # Move back to row A -- the saved 50 is honoured.
                tree.move_cursor(row_a, animate=False)
                await pilot.pause()
                assert tree.scroll_offset.x == 50
                assert long_a.remembered_scroll_x == 50

    asyncio.run(runner())


def test_right_arrow_on_fitting_row_expands_collapsed_node():
    """``→`` on a row that fits + ``can_expand=True`` expands the node."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # A BlockNode with a short preview fits within the 80-col
                # viewport; can_expand is True so right-arrow must expand.
                short_block = BlockNode(
                    factory=MagicMock(),
                    backend=MagicMock(),
                    variant_idx=0,
                    kind=BlockKind.BODY,
                    block_idx=1,
                    preview="x",
                )
                short_block.expand = MagicMock(return_value=[])
                tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [short_block]
                )
                row = children[0]
                tree.move_cursor(row, animate=False)
                await pilot.pause()

                assert row.is_expanded is False
                scroll_before = tree.scroll_offset.x

                await pilot.press("right")
                await pilot.pause()

                # Did NOT pan -- the row fit, so the action fell through
                # to expand. The model's expand spy was invoked.
                assert tree.scroll_offset.x == scroll_before
                assert row.is_expanded is True
                short_block.expand.assert_called_once()

    asyncio.run(runner())


def test_right_arrow_on_already_expanded_fitting_row_is_noop():
    """``→`` on a row that fits + already expanded: scroll unchanged + no extra expand."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                short_block = BlockNode(
                    factory=MagicMock(),
                    backend=MagicMock(),
                    variant_idx=0,
                    kind=BlockKind.BODY,
                    block_idx=1,
                    preview="x",
                )
                short_block.expand = MagicMock(return_value=[])
                tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [short_block]
                )
                row = children[0]
                tree.move_cursor(row, animate=False)
                await pilot.pause()

                # Pre-expand the row so the action's expand-fallback
                # branch must take the no-op path.
                row.expand()
                await pilot.pause()
                assert row.is_expanded is True
                short_block.expand.reset_mock()
                scroll_before = tree.scroll_offset.x

                await pilot.press("right")
                await pilot.pause()

                assert tree.scroll_offset.x == scroll_before
                # No new expand call -- the action is a no-op on an
                # already-expanded fitting row.
                short_block.expand.assert_not_called()

    asyncio.run(runner())


def test_right_arrow_on_overflowing_collapsed_node_pans_instead_of_expand():
    """``→`` on an overflowing collapsed expandable row: scroll wins over expand.

    When the cursor row's content extends past the viewport's right
    edge (the ``>>`` truncation marker is visible) and the model also
    advertises ``can_expand=True``, the user's first ``→`` keystroke
    must reveal more of the current row's content, not collapse the
    user's mental ``>>``-driven intent into an irreversible expand.
    The expand affordance is reached once enough panning has flushed
    the overflow (covered separately by the fitting-row test above).
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # A BlockNode whose preview spills past the 80-col viewport
                # AND can_expand=True. Both the scroll path and the expand
                # path are reachable; the priority order must pick scroll.
                long_block = BlockNode(
                    factory=MagicMock(),
                    backend=MagicMock(),
                    variant_idx=0,
                    kind=BlockKind.BODY,
                    block_idx=1,
                    preview="x" * 200,
                )
                long_block.expand = MagicMock(return_value=[])
                tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [long_block]
                )
                row = children[0]
                tree.move_cursor(row, animate=False)
                await pilot.pause()
                # Sanity: the row actually overflows the viewport.
                assert tree.label_cell_len(row) > tree.size.width
                assert row.is_expanded is False
                scroll_before = tree.scroll_offset.x

                await pilot.press("right")
                await pilot.pause()

                # Scroll advanced + expand spy NOT called -- priority
                # respected.
                assert tree.scroll_offset.x == scroll_before + 1
                assert long_block.remembered_scroll_x == scroll_before + 1
                assert row.is_expanded is False
                long_block.expand.assert_not_called()

    asyncio.run(runner())


def test_right_arrow_on_fitting_leaf_is_noop():
    """``→`` on a terminal (``can_expand=False``) fitting row: no-op."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                short_leaf = AsmLeaf(text="hi")
                tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [short_leaf]
                )
                row = children[0]
                tree.move_cursor(row, animate=False)
                await pilot.pause()
                scroll_before = tree.scroll_offset.x

                await pilot.press("right")
                await pilot.pause()

                # No pan + leaf cannot expand -> action is a no-op.
                assert tree.scroll_offset.x == scroll_before
                assert row.is_expanded is False


    asyncio.run(runner())


def test_vim_l_mirrors_right_arrow():
    """``l`` behaves like ``→``: pan on overflow, expand on fitting."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                long_leaf = AsmLeaf(text="x" * 200)
                tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [long_leaf]
                )
                tree.move_cursor(children[0], animate=False)
                await pilot.pause()
                assert tree.scroll_offset.x == 0

                # ``l`` on overflow -> pan + save remembered.
                await pilot.press("l")
                await pilot.pause()
                assert tree.scroll_offset.x == 1
                assert long_leaf.remembered_scroll_x == 1

    asyncio.run(runner())


def test_tree_h_no_longer_pans():
    """``h`` is reserved for the App-level help modal -- the tree drops it.

    Positive confirmation that pressing ``h`` while the tree has focus
    no longer triggers a horizontal pan: ``scroll_offset.x`` stays
    untouched and the model's ``remembered_scroll_x`` stays at its
    pre-press value. The App-level binding (covered separately) takes
    over the keystroke.
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                long_leaf = AsmLeaf(text="x" * 200)
                tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [long_leaf]
                )
                tree.move_cursor(children[0], animate=False)
                await pilot.pause()
                # Pan right by 3 with the standard binding so we land
                # at a non-zero scroll position; any future tree-level
                # ``h``-binding regression would pan back to 2.
                for _ in range(3):
                    await pilot.press("right")
                await pilot.pause()
                assert tree.scroll_offset.x == 3
                assert long_leaf.remembered_scroll_x == 3

                await pilot.press("h")
                await pilot.pause()
                # Scroll position + remembered value MUST be unchanged
                # -- the tree no longer consumes ``h``.
                assert tree.scroll_offset.x == 3
                assert long_leaf.remembered_scroll_x == 3

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# One-shot undo for the ``←``-to-parent climb. After a left-arrow that
# moved the cursor up to its parent (scroll_x already 0 path), the very
# next keypress is special: ``→`` restores the cursor to the just-
# evacuated child; any other key invalidates the undo state.
# ---------------------------------------------------------------------------


def test_right_arrow_after_left_to_parent_restores_child_cursor():
    """``→`` immediately after ``←``-to-parent restores the prior cursor."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # Short child so the right-arrow undo case can't be
                # masked by a pan: a short row has no content past the
                # viewport, so the binding's fallback would be "expand"
                # (and AsmLeaf is can_expand=False -> no-op).
                short_leaf = AsmLeaf(text="hi")
                tree, fn_tree_node, children = (
                    await _expand_function_with_children(app, pilot, [short_leaf])
                )
                child_tree_node = children[0]

                # Park cursor on the child.
                tree.move_cursor(child_tree_node, animate=False)
                await pilot.pause()
                assert tree.cursor_node is child_tree_node
                assert tree.scroll_offset.x == 0

                # ``←`` at scroll-zero: cursor climbs to the parent
                # FunctionNode row.
                await pilot.press("left")
                await pilot.pause()
                assert tree.cursor_node is fn_tree_node

                # ``→`` immediately after: restore the child cursor.
                await pilot.press("right")
                await pilot.pause()
                assert tree.cursor_node is child_tree_node
                # The child is a non-expandable leaf, so the restore
                # path did NOT also expand anything along the way.
                assert child_tree_node.is_expanded is False

    asyncio.run(runner())


def test_undo_state_consumed_after_one_use():
    """A second ``→`` after a successful undo falls back to normal behavior."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # Use a collapsed expandable child so the post-undo
                # second ``→`` exercises the expand fallback. A second
                # sentinel sibling AsmLeaf keeps the FunctionNode's
                # child count at 2 so the universal "auto-expand on
                # exactly one child" rule does not pre-expand
                # ``short_block`` and defeat the collapsed-state setup.
                short_block = BlockNode(
                    factory=MagicMock(),
                    backend=MagicMock(),
                    variant_idx=0,
                    kind=BlockKind.BODY,
                    block_idx=1,
                    preview="x",
                )
                short_block.expand = MagicMock(return_value=[])
                sentinel_sibling = AsmLeaf(text="sentinel")
                tree, fn_tree_node, children = (
                    await _expand_function_with_children(
                        app, pilot, [short_block, sentinel_sibling]
                    )
                )
                child_tree_node = children[0]

                tree.move_cursor(child_tree_node, animate=False)
                await pilot.pause()
                assert tree.cursor_node is child_tree_node
                assert child_tree_node.is_expanded is False

                # left -> parent.
                await pilot.press("left")
                await pilot.pause()
                assert tree.cursor_node is fn_tree_node

                # First right -> undo, cursor back to child.
                await pilot.press("right")
                await pilot.pause()
                assert tree.cursor_node is child_tree_node
                short_block.expand.assert_not_called()

                # Second right -> undo state already consumed; the
                # binding's normal expand path fires (row fits, node is
                # collapsed + can_expand).
                await pilot.press("right")
                await pilot.pause()
                assert child_tree_node.is_expanded is True
                short_block.expand.assert_called_once()

    asyncio.run(runner())


def test_intervening_key_invalidates_undo():
    """Any key other than ``→`` between ``←``-to-parent and ``→`` clears undo."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # Collapsed expandable so the post-invalidation ``→``
                # exercises the expand fallback (proving it's NOT undo).
                short_block = BlockNode(
                    factory=MagicMock(),
                    backend=MagicMock(),
                    variant_idx=0,
                    kind=BlockKind.BODY,
                    block_idx=1,
                    preview="x",
                )
                short_block.expand = MagicMock(return_value=[])
                tree, fn_tree_node, children = (
                    await _expand_function_with_children(app, pilot, [short_block])
                )
                child_tree_node = children[0]

                tree.move_cursor(child_tree_node, animate=False)
                await pilot.pause()

                # left -> parent.
                await pilot.press("left")
                await pilot.pause()
                assert tree.cursor_node is fn_tree_node

                # ``down`` invalidates the undo state. (There is no
                # sibling FunctionNode in this fixture, so the cursor
                # position after ``down`` is allowed to be anywhere;
                # what matters is that the next ``→`` is NOT an undo.)
                await pilot.press("down")
                await pilot.pause()

                # Move back to the parent fn_tree_node so the next
                # ``→`` has a deterministic expand target.
                tree.move_cursor(fn_tree_node, animate=False)
                await pilot.pause()
                # fn_tree_node is already expanded (children present),
                # so the right-arrow falls through to the no-op branch
                # on an already-expanded fitting row; assert the cursor
                # did NOT return to the child (which would be the undo
                # outcome).
                short_block.expand.reset_mock()
                await pilot.press("right")
                await pilot.pause()
                assert tree.cursor_node is fn_tree_node
                # Undo would have restored child_tree_node; it did not.
                assert tree.cursor_node is not child_tree_node

    asyncio.run(runner())


def test_vim_l_does_not_trigger_undo():
    """Vim ``l`` is "any other key" w.r.t. the undo overlay; it clears state.

    The spec singles out the literal ``right`` key as the only trigger;
    vim ``l`` (which shares the conditional pan/expand action) is
    treated as an invalidating key.
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                short_block = BlockNode(
                    factory=MagicMock(),
                    backend=MagicMock(),
                    variant_idx=0,
                    kind=BlockKind.BODY,
                    block_idx=1,
                    preview="x",
                )
                short_block.expand = MagicMock(return_value=[])
                tree, fn_tree_node, children = (
                    await _expand_function_with_children(app, pilot, [short_block])
                )
                child_tree_node = children[0]

                tree.move_cursor(child_tree_node, animate=False)
                await pilot.pause()
                await pilot.press("left")
                await pilot.pause()
                assert tree.cursor_node is fn_tree_node

                # ``l`` clears the undo state. The parent FunctionNode
                # is already expanded (children present) + fits the
                # viewport; the binding's expand fallback is a no-op
                # on an already-expanded row.
                await pilot.press("l")
                await pilot.pause()
                # Did NOT undo (cursor stayed on the parent).
                assert tree.cursor_node is fn_tree_node

                # A subsequent ``→`` is now normal: no undo state, so
                # the binding falls through to the no-op expand on an
                # already-expanded fitting row.
                await pilot.press("right")
                await pilot.pause()
                assert tree.cursor_node is fn_tree_node

    asyncio.run(runner())


def test_pan_left_branch_does_not_arm_undo():
    """The ``←`` pan branch (scroll_x > 0) does NOT arm the undo state.

    Only the cursor-to-parent climb arms undo. Confirm by panning right
    so scroll_x > 0, then pressing ``←`` (pure pan-left), then ``→``:
    the right-arrow must perform its normal pan / expand action, NOT a
    cursor restore.
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                long_leaf = AsmLeaf(text="x" * 200)
                tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [long_leaf]
                )
                row = children[0]
                tree.move_cursor(row, animate=False)
                await pilot.pause()

                # Pan right twice -> scroll_x == 2.
                await pilot.press("right")
                await pilot.press("right")
                await pilot.pause()
                assert tree.scroll_offset.x == 2
                cursor_before_left = tree.cursor_node

                # ``←`` while scroll_x > 0: pure pan-left, cursor unchanged.
                await pilot.press("left")
                await pilot.pause()
                assert tree.cursor_node is cursor_before_left
                assert tree.scroll_offset.x == 1

                # ``→`` after a pan-left ``←``: must perform its normal
                # pan-right action (NOT a cursor restore), so scroll_x
                # increments back to 2.
                await pilot.press("right")
                await pilot.pause()
                assert tree.cursor_node is cursor_before_left
                assert tree.scroll_offset.x == 2

    asyncio.run(runner())
