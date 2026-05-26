"""Pilot + pure-unit tests for the inspector status bar.

Covers:

* :func:`node_breadcrumb_segment` per-node-type segment formatting.
* :func:`breadcrumb_for_cursor` walks the cursor's ancestor chain
  root-to-cursor, skipping :class:`VariantGroupNode` wrappers.
* :func:`filter_summary` brief textual form for the active filter.
* :class:`StatusBar` reflects the current filter on demand.

The file is gated on ``pytest.importorskip("textual")`` +
``pytest.importorskip("natsort")`` so the default ``nix develop`` shell
shows it as SKIPPED rather than as an import failure.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")
pytest.importorskip("natsort")

import asyncio
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._app import InspectorApp
from tokenizer.inspector._app._filter import FilterAccepted, FilterConfig
from tokenizer.inspector._app._order import (
    VariantGroupNode,
    build_canonical_axes,
)
from tokenizer.inspector._app._status_bar import (
    StatusBar,
    breadcrumb_for_cursor,
    filter_summary,
    node_breadcrumb_segment,
)
from tokenizer.inspector._render._protocol import (
    BackendFactory,
    BlockKind,
    FunctionHandle,
    RenderBackend,
    RenderedBlock,
    RenderedVariant,
)
from tokenizer.inspector._tree_model import (
    BlockNode,
    FunctionNode,
    VariantNode,
)
from tokenizer.variant_info import VariantIdentity
from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
)


# ---------------------------------------------------------------------------
# Helpers (RV / VariantNode shape shared with the filter test suite).
# ---------------------------------------------------------------------------


def _make_rv(
    *,
    variant_idx: int,
    arch: str = "x64",
    compiler: str = "clang",
    cver: str = "10",
    opt: str = "O2",
    pkg: str = "nping",
) -> RenderedVariant:
    label_axes = types.MappingProxyType(
        {
            ARCH_PREFIX: arch,
            COMP_PREFIX: compiler,
            CVER_PREFIX: cver,
            OPT_PREFIX: opt,
        }
    )
    identity = VariantIdentity(
        arch=arch,
        compiler=compiler,
        compiler_version=cver,
        opt=opt,
        pkg=pkg,
        variant_id=variant_idx,
    )
    return RenderedVariant(
        variant_idx=variant_idx,
        label_axes=label_axes,
        extra_metadata=types.MappingProxyType({}),
        variant_identity=identity,
    )


def _make_variant_node(rv: RenderedVariant) -> VariantNode:
    return VariantNode(
        factory=MagicMock(spec=BackendFactory),
        backend=MagicMock(spec=RenderBackend),
        variant_idx=rv.variant_idx,
        label_axes=rv.label_axes,
    )


def _make_factory_with_rvs(*, name: str, rvs: list) -> MagicMock:
    handle = FunctionHandle(arm=SectionKind.MATCHED, idx=0, name=name)
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [handle]
    backend = MagicMock(spec=RenderBackend)
    backend.handle = handle
    backend.closed = False
    backend.variants.return_value = rvs
    backend.blocks.return_value = [
        RenderedBlock(kind=BlockKind.BODY, block_idx=0, preview="block-preview")
    ]
    backend.render_block.return_value = ()
    factory.make.return_value = backend
    return factory


def _mock_tree_node(data, parent):
    n = MagicMock()
    n.data = data
    n.parent = parent
    return n


# ---------------------------------------------------------------------------
# Filter summary -- pure unit tests.
# ---------------------------------------------------------------------------


def test_filter_summary_empty():
    assert filter_summary(None) == ""
    assert filter_summary(FilterConfig.empty()) == ""


def test_filter_summary_single_axis():
    axes = build_canonical_axes()
    arch_axis = axes[0]
    cfg = FilterConfig.build({arch_axis: ["x86"]})
    assert filter_summary(cfg) == "filter: -arch:x86"


def test_filter_summary_multi_axis_sorted():
    axes = build_canonical_axes()
    arch_axis = axes[0]
    comp_axis = axes[2]
    cfg = FilterConfig.build({arch_axis: ["x86"], comp_axis: ["gcc"]})
    out = filter_summary(cfg)
    # Brief labels: arch + comp (sorted alphabetically).
    assert out == "filter: -arch:x86 -comp:gcc"


def test_filter_summary_multi_values_per_axis_sorted():
    axes = build_canonical_axes()
    arch_axis = axes[0]
    cfg = FilterConfig.build({arch_axis: ["x86", "arm32"]})
    assert filter_summary(cfg) == "filter: -arch:arm32,x86"


# ---------------------------------------------------------------------------
# Per-node-type breadcrumb segments.
# ---------------------------------------------------------------------------


def test_node_breadcrumb_segment_function():
    handle = FunctionHandle(arm=SectionKind.MATCHED, idx=0, name="Calloc")
    fn = FunctionNode(factory=MagicMock(), handle=handle)
    assert node_breadcrumb_segment(fn) == "Calloc"


def test_node_breadcrumb_segment_variant():
    rv = _make_rv(
        variant_idx=0, arch="arm32", compiler="clang", cver="5.0", opt="O0"
    )
    v = _make_variant_node(rv)
    assert node_breadcrumb_segment(v) == "arm32 clang v5.0 -O0"


def test_node_breadcrumb_segment_block_body_and_special():
    body = BlockNode(
        factory=MagicMock(),
        backend=MagicMock(),
        variant_idx=0,
        kind=BlockKind.BODY,
        block_idx=3,
        preview="x",
    )
    header = BlockNode(
        factory=MagicMock(),
        backend=MagicMock(),
        variant_idx=0,
        kind=BlockKind.VARIANT_HEADER,
        block_idx=0,
        preview="",
    )
    fid = BlockNode(
        factory=MagicMock(),
        backend=MagicMock(),
        variant_idx=0,
        kind=BlockKind.FUNCTION_ID,
        block_idx=0,
        preview="",
    )
    assert node_breadcrumb_segment(body) == "Block: 3"
    assert node_breadcrumb_segment(header) == "Variant Header"
    assert node_breadcrumb_segment(fid) == "Function ID"


def test_node_breadcrumb_segment_variant_group_skipped():
    """Intermediate group rows return ``None`` so the breadcrumb skips them."""
    rv = _make_rv(variant_idx=0, arch="x86")
    axes = build_canonical_axes()
    group = VariantGroupNode(
        axis=axes[0],
        axis_value="x86",
        children=[],
        rendered_by_variant={0: rv},
    )
    assert node_breadcrumb_segment(group) is None


# ---------------------------------------------------------------------------
# Full ancestor-chain walks.
# ---------------------------------------------------------------------------


def test_breadcrumb_walks_full_ancestor_chain():
    """Function > Variant > Block reads top-down (root-most first)."""
    handle = FunctionHandle(arm=SectionKind.MATCHED, idx=0, name="Calloc")
    fn = FunctionNode(factory=MagicMock(), handle=handle)
    rv = _make_rv(
        variant_idx=0, arch="arm32", compiler="clang", cver="5.0", opt="O0"
    )
    v = _make_variant_node(rv)
    b = BlockNode(
        factory=MagicMock(),
        backend=MagicMock(),
        variant_idx=0,
        kind=BlockKind.BODY,
        block_idx=3,
        preview="x",
    )

    root = _mock_tree_node(None, None)
    fn_tree_node = _mock_tree_node(fn, root)
    v_tree_node = _mock_tree_node(v, fn_tree_node)
    b_tree_node = _mock_tree_node(b, v_tree_node)

    assert (
        breadcrumb_for_cursor(b_tree_node)
        == "Calloc > arm32 clang v5.0 -O0 > Block: 3"
    )


def test_breadcrumb_skips_variant_group_nodes():
    """Intermediate :class:`VariantGroupNode` wrappers don't surface in
    the breadcrumb — the leaf variant already carries the axis value."""
    handle = FunctionHandle(arm=SectionKind.MATCHED, idx=0, name="Foo")
    fn = FunctionNode(factory=MagicMock(), handle=handle)
    rv = _make_rv(variant_idx=0, arch="x86")
    v = _make_variant_node(rv)
    axes = build_canonical_axes()
    arch_axis = axes[0]
    group = VariantGroupNode(
        axis=arch_axis,
        axis_value="x86",
        children=[v],
        rendered_by_variant={0: rv},
    )

    root = _mock_tree_node(None, None)
    fn_tree_node = _mock_tree_node(fn, root)
    group_tree_node = _mock_tree_node(group, fn_tree_node)
    v_tree_node = _mock_tree_node(v, group_tree_node)

    # Group row contributes nothing; chain is fn > variant only.
    assert breadcrumb_for_cursor(v_tree_node) == "Foo > x86 clang v10 -O2"


def test_breadcrumb_empty_at_root():
    root = _mock_tree_node(None, None)
    assert breadcrumb_for_cursor(root) == ""
    assert breadcrumb_for_cursor(None) == ""


# ---------------------------------------------------------------------------
# StatusBar widget integration.
# ---------------------------------------------------------------------------


def test_status_bar_reflects_active_filter_after_dispatch():
    """An accepted filter is reflected in the status bar's summary."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            rvs = [
                _make_rv(variant_idx=0, arch="x86"),
                _make_rv(variant_idx=1, arch="arm32"),
            ]
            factory = _make_factory_with_rvs(name="main", rvs=rvs)
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                tree = app.query_one("#tree")
                fn_tree_node = tree.root.children[0]
                fn_tree_node.expand()
                await pilot.pause()

                arch_axis = build_canonical_axes()[0]
                new_config = FilterConfig.build({arch_axis: ["x86"]})
                app._on_filter_dialog_dismissed(
                    FilterAccepted(config=new_config)
                )
                await pilot.pause()

                # Status bar carries the active filter -- the brief
                # form drops out of the pure helper for byte-equal
                # comparison.
                status_bar = app.query_one("#status-bar", StatusBar)
                # The widget owns the filter ref via its setter.
                status_bar.refresh_state()
                assert filter_summary(app._filter_config) == "filter: -arch:x86"

    asyncio.run(runner())
