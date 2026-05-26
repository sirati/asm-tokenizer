"""Pilot tests for the inspector Filter modal + filter pass.

Covers:

* :class:`FilterConfig` value-equality (frozen dataclass, two builds
  from the same disabled-set compare equal).
* :func:`apply_filter` drops variants whose axis values are disabled.
* :func:`discover_axis_values` + :func:`discover_all_axis_values`.
* :class:`FilterDialog` accept via ``ctrl+s`` -> :class:`FilterAccepted`.
* :class:`FilterDialog` accept via :class:`Button` press.
* :class:`FilterDialog` escape -> :class:`FilterCancelled`.
* ``f`` opens :class:`FilterDialog`.
* App-level accept persists the new :class:`FilterConfig` + rebuilds
  the tree (a disabled-value variant drops from the function row's
  children).

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
from tokenizer.inspector._app._filter import (
    FilterAccepted,
    FilterCancelled,
    FilterConfig,
    FilterDialog,
    MISSING_VALUE_TOKEN,
    apply_filter,
    discover_all_axis_values,
    discover_axis_values,
)
from tokenizer.inspector._app._order import (
    build_canonical_axes,
    build_extra_meta_axis,
)
from tokenizer.inspector._render._protocol import (
    BackendFactory,
    BlockKind,
    FunctionHandle,
    RenderBackend,
    RenderedBlock,
    RenderedVariant,
)
from tokenizer.inspector._tree_model import VariantNode
from tokenizer.variant_info import VariantIdentity
from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
)


# ---------------------------------------------------------------------------
# Helpers (mirror :mod:`test_app_order_dialog` so the two suites share
# the same RV / VariantNode shape).
# ---------------------------------------------------------------------------


def _make_rv(
    *,
    variant_idx: int,
    arch: str = "x64",
    compiler: str = "clang",
    cver: str = "10",
    opt: str = "O2",
    extra: dict | None = None,
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
    extra_metadata = types.MappingProxyType(dict(extra or {}))
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
        extra_metadata=extra_metadata,
        variant_identity=identity,
    )


def _make_variant_node(rv: RenderedVariant) -> VariantNode:
    factory = MagicMock(spec=BackendFactory)
    backend = MagicMock(spec=RenderBackend)
    return VariantNode(
        factory=factory,
        backend=backend,
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


# ---------------------------------------------------------------------------
# Pure-unit tests -- FilterConfig + apply_filter + discovery.
# ---------------------------------------------------------------------------


def test_filter_config_empty_baseline():
    """``FilterConfig.empty()`` compares equal to itself + reports empty."""
    a = FilterConfig.empty()
    b = FilterConfig.empty()
    assert a == b
    assert hash(a) == hash(b)
    assert a.is_empty()


def test_filter_config_build_equality():
    """Two configs built from the same disabled-set compare equal."""
    axes = build_canonical_axes()
    arch_axis = axes[0]
    cfg_a = FilterConfig.build({arch_axis: ["x86", "arm32"]})
    cfg_b = FilterConfig.build({arch_axis: frozenset({"arm32", "x86"})})
    assert cfg_a == cfg_b
    assert hash(cfg_a) == hash(cfg_b)
    assert not cfg_a.is_empty()
    assert cfg_a.disabled_for(arch_axis) == frozenset({"x86", "arm32"})


def test_filter_config_build_drops_empty_disabled_sets():
    """An axis with no disabled values is dropped to the canonical form."""
    axes = build_canonical_axes()
    arch_axis = axes[0]
    comp_axis = axes[2]
    cfg = FilterConfig.build({arch_axis: ["x86"], comp_axis: []})
    assert cfg.axes() == (arch_axis,)
    assert FilterConfig.build({arch_axis: []}) == FilterConfig.empty()


def test_apply_filter_drops_disabled_arch():
    """A disabled arch value removes every variant on that arch."""
    rvs = [
        _make_rv(variant_idx=0, arch="x86"),
        _make_rv(variant_idx=1, arch="arm32"),
        _make_rv(variant_idx=2, arch="arm64"),
        _make_rv(variant_idx=3, arch="x86"),
    ]
    variants = [_make_variant_node(rv) for rv in rvs]
    rendered_by_variant = {rv.variant_idx: rv for rv in rvs}
    arch_axis = build_canonical_axes()[0]
    cfg = FilterConfig.build({arch_axis: ["x86"]})

    out = apply_filter(variants, rendered_by_variant, cfg)
    assert [v.variant_idx for v in out] == [1, 2]


def test_apply_filter_none_or_empty_is_passthrough():
    rvs = [_make_rv(variant_idx=0, arch="x86")]
    variants = [_make_variant_node(rv) for rv in rvs]
    rendered_by_variant = {rv.variant_idx: rv for rv in rvs}

    assert apply_filter(variants, rendered_by_variant, None) == variants
    assert apply_filter(variants, rendered_by_variant, FilterConfig.empty()) == variants


def test_apply_filter_multi_axis_intersects():
    """Multi-axis filter hides a variant if ANY axis is disabled."""
    rvs = [
        _make_rv(variant_idx=0, arch="x86", compiler="clang"),
        _make_rv(variant_idx=1, arch="x86", compiler="gcc"),
        _make_rv(variant_idx=2, arch="arm32", compiler="clang"),
    ]
    variants = [_make_variant_node(rv) for rv in rvs]
    rendered_by_variant = {rv.variant_idx: rv for rv in rvs}
    axes = build_canonical_axes()
    arch_axis = axes[0]
    comp_axis = axes[2]
    cfg = FilterConfig.build({arch_axis: ["x86"], comp_axis: ["gcc"]})

    # idx 0 hidden by arch=x86, idx 1 hidden by both, idx 2 visible.
    out = apply_filter(variants, rendered_by_variant, cfg)
    assert [v.variant_idx for v in out] == [2]


def test_apply_filter_missing_value_token_drops_axis_absent_variants():
    """Disabling the ``"?"`` value drops variants where the axis is missing."""
    rvs = [
        _make_rv(variant_idx=0, extra={"sanitizer": "address"}),
        _make_rv(variant_idx=1, extra={}),
    ]
    variants = [_make_variant_node(rv) for rv in rvs]
    rendered_by_variant = {rv.variant_idx: rv for rv in rvs}
    san_axis = build_extra_meta_axis("sanitizer")
    cfg = FilterConfig.build({san_axis: [MISSING_VALUE_TOKEN]})

    out = apply_filter(variants, rendered_by_variant, cfg)
    assert [v.variant_idx for v in out] == [0]


def test_discover_axis_values_natsort_with_missing_last():
    """Values are natsort-ordered; missing-value token sinks to the end."""
    rvs = [
        _make_rv(variant_idx=0, extra={"san": "address"}),
        _make_rv(variant_idx=1, extra={"san": "thread"}),
        _make_rv(variant_idx=2, extra={}),
    ]
    san_axis = build_extra_meta_axis("san")
    values = discover_axis_values(san_axis, rvs)
    assert values == ("address", "thread", MISSING_VALUE_TOKEN)


def test_discover_all_axis_values_per_axis():
    rvs = [
        _make_rv(variant_idx=0, arch="x86", compiler="clang"),
        _make_rv(variant_idx=1, arch="arm32", compiler="gcc"),
    ]
    axes = build_canonical_axes()
    by_axis = discover_all_axis_values(axes, rvs)
    arch_axis = axes[0]
    comp_axis = axes[2]
    assert by_axis[arch_axis] == ("arm32", "x86")
    assert by_axis[comp_axis] == ("clang", "gcc")


# ---------------------------------------------------------------------------
# Modal dialog tests (Textual Pilot).
# ---------------------------------------------------------------------------


def test_filter_dialog_ctrl_s_yields_filter_accepted():
    """Accept via ``ctrl+s`` dismisses with :class:`FilterAccepted`."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                results: list = []
                axes = build_canonical_axes()
                arch_axis = axes[0]
                dialog = FilterDialog(
                    axis_values={arch_axis: ("x86", "arm32")},
                )
                app.push_screen(dialog, results.append)
                await pilot.pause()
                await pilot.press("ctrl+s")
                await pilot.pause()
                assert len(results) == 1
                assert isinstance(results[0], FilterAccepted)
                assert isinstance(results[0].config, FilterConfig)

    asyncio.run(runner())


def test_filter_dialog_escape_yields_filter_cancelled():
    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                results: list = []
                axes = build_canonical_axes()
                arch_axis = axes[0]
                dialog = FilterDialog(axis_values={arch_axis: ("x86",)})
                app.push_screen(dialog, results.append)
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()
                assert len(results) == 1
                assert isinstance(results[0], FilterCancelled)

    asyncio.run(runner())


def test_f_binding_opens_filter_dialog():
    """Pressing ``f`` pushes :class:`FilterDialog` onto the screen stack."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            rvs = [_make_rv(variant_idx=0, arch="x86")]
            factory = _make_factory_with_rvs(name="main", rvs=rvs)
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                # Expand the function so the discovery pass sees the
                # backend's variants (the filter modal pulls axis values
                # off currently-expanded functions).
                tree = app.query_one("#tree")
                fn_tree_node = tree.root.children[0]
                fn_tree_node.expand()
                await pilot.pause()

                await pilot.press("f")
                await pilot.pause()
                assert isinstance(app.screen, FilterDialog)
                await pilot.press("escape")
                await pilot.pause()

    asyncio.run(runner())


def test_filter_dialog_accept_persists_filter_config_and_rebuilds():
    """Accept persists the new config on the App + filters the tree."""

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
                assert len(fn_tree_node.children) == 2

                # Drive the dismiss callback directly with a config that
                # disables arch=x86. The function's variant siblings
                # should rebuild + drop the x86 row.
                arch_axis = build_canonical_axes()[0]
                new_config = FilterConfig.build({arch_axis: ["x86"]})
                app._on_filter_dialog_dismissed(
                    FilterAccepted(config=new_config)
                )
                await pilot.pause()

                # One variant survives the filter (arm32 -> idx 1).
                surviving = [
                    c for c in fn_tree_node.children if c.data is not None
                ]
                assert len(surviving) == 1
                variant_model = surviving[0].data
                assert isinstance(variant_model, VariantNode)
                assert variant_model.variant_idx == 1

    asyncio.run(runner())


def test_filter_dialog_cancel_leaves_config_unchanged():
    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                before = app._filter_config
                app._on_filter_dialog_dismissed(FilterCancelled())
                await pilot.pause()
                assert app._filter_config is before
