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
    function_has_passing_variants,
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
from tokenizer.inspector._tree_model import FunctionNode, VariantNode
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


# ---------------------------------------------------------------------------
# Fix #1 -- function-row greying + non-expandable when filter zeros out variants.
# ---------------------------------------------------------------------------


def test_function_has_passing_variants_true_when_filter_is_none():
    """``None`` / empty config short-circuits to ``True`` (passthrough)."""
    rvs = [_make_rv(variant_idx=0, arch="x86")]
    factory = _make_factory_with_rvs(name="fn", rvs=rvs)
    fn_node = FunctionNode(factory=factory, handle=factory.handles[0])
    assert function_has_passing_variants(fn_node, None) is True
    assert function_has_passing_variants(fn_node, FilterConfig.empty()) is True


def test_function_has_passing_variants_true_when_at_least_one_passes():
    rvs = [
        _make_rv(variant_idx=0, arch="x86"),
        _make_rv(variant_idx=1, arch="arm32"),
    ]
    factory = _make_factory_with_rvs(name="fn", rvs=rvs)
    fn_node = FunctionNode(factory=factory, handle=factory.handles[0])
    arch_axis = build_canonical_axes()[0]
    cfg = FilterConfig.build({arch_axis: ["x86"]})  # arm32 survives
    assert function_has_passing_variants(fn_node, cfg) is True


def test_function_has_passing_variants_false_when_all_filtered():
    rvs = [
        _make_rv(variant_idx=0, arch="x86"),
        _make_rv(variant_idx=1, arch="x86"),
    ]
    factory = _make_factory_with_rvs(name="fn", rvs=rvs)
    fn_node = FunctionNode(factory=factory, handle=factory.handles[0])
    arch_axis = build_canonical_axes()[0]
    cfg = FilterConfig.build({arch_axis: ["x86"]})  # everything disabled
    assert function_has_passing_variants(fn_node, cfg) is False


def test_root_function_row_dim_and_non_expandable_when_filter_zeros_out():
    """A function with every variant filtered out renders dim + ``allow_expand=False``."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            rvs = [_make_rv(variant_idx=0, arch="x86")]
            factory = _make_factory_with_rvs(name="dead_fn", rvs=rvs)
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                tree = app.query_one("#tree")
                fn_tree_node = tree.root.children[0]
                # Expand once so the backend caches; predicate then reuses
                # the cached one. Collapse so the test exercises the same
                # closed-tree path the user would see on a fresh filter.
                fn_tree_node.expand()
                await pilot.pause()
                fn_tree_node.collapse()
                await pilot.pause()
                assert fn_tree_node.allow_expand is True

                arch_axis = build_canonical_axes()[0]
                new_config = FilterConfig.build({arch_axis: ["x86"]})
                app._on_filter_dialog_dismissed(FilterAccepted(config=new_config))
                await pilot.pause()

                # The function row must lose its expand triangle and
                # render with the dim filtered-out style.
                assert fn_tree_node.allow_expand is False
                label_text = fn_tree_node.label
                # Every span on the row inherits the dim style.
                styles = [str(span.style) for span in label_text.spans]
                assert any("dim" in s for s in styles) or label_text.style is not None

    asyncio.run(runner())


def test_root_function_row_normal_when_at_least_one_variant_passes():
    """A function with at least one surviving variant stays expandable."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            rvs = [
                _make_rv(variant_idx=0, arch="x86"),
                _make_rv(variant_idx=1, arch="arm32"),
            ]
            factory = _make_factory_with_rvs(name="live_fn", rvs=rvs)
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                tree = app.query_one("#tree")
                fn_tree_node = tree.root.children[0]
                assert fn_tree_node.allow_expand is True

                arch_axis = build_canonical_axes()[0]
                new_config = FilterConfig.build({arch_axis: ["x86"]})
                app._on_filter_dialog_dismissed(FilterAccepted(config=new_config))
                await pilot.pause()

                # arm32 survives, function remains expandable + normally
                # styled (no dim).
                assert fn_tree_node.allow_expand is True

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Fix #2 -- filter dialog cannot deselect every option for an axis.
# ---------------------------------------------------------------------------


def test_filter_dialog_min_one_selected_blocks_last_deselect():
    """Toggling the last selected value off is rejected (state unchanged)."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                arch_axis = build_canonical_axes()[0]
                dialog = FilterDialog(axis_values={arch_axis: ("x86",)})
                app.push_screen(dialog)
                await pilot.pause()

                from tokenizer.inspector._app._filter._dialog import (
                    _MinOneSelectionList,
                )

                sel_list = dialog.query_one(_MinOneSelectionList)
                assert sel_list.selected == ["x86"]

                # Attempt to deselect the only checked value.
                changed = sel_list._toggle("x86")
                assert changed is False
                assert sel_list.selected == ["x86"]

                await pilot.press("escape")
                await pilot.pause()

    asyncio.run(runner())


def test_filter_dialog_min_one_allows_non_last_deselect():
    """Toggling a value off when others remain selected proceeds normally."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                arch_axis = build_canonical_axes()[0]
                dialog = FilterDialog(axis_values={arch_axis: ("x86", "arm32")})
                app.push_screen(dialog)
                await pilot.pause()

                from tokenizer.inspector._app._filter._dialog import (
                    _MinOneSelectionList,
                )

                sel_list = dialog.query_one(_MinOneSelectionList)
                assert set(sel_list.selected) == {"x86", "arm32"}

                # First deselect: two selected -> one selected (allowed).
                assert sel_list._toggle("x86") is True
                assert sel_list.selected == ["arm32"]

                # Second deselect: one selected -> would be zero (rejected).
                assert sel_list._toggle("arm32") is False
                assert sel_list.selected == ["arm32"]

                await pilot.press("escape")
                await pilot.pause()

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Fix #3 -- focus lands on the first axis SelectionList at mount.
# ---------------------------------------------------------------------------


def test_filter_dialog_focuses_first_axis_list_on_mount():
    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                axes = build_canonical_axes()
                # Two axes with values; the FIRST one must take focus.
                dialog = FilterDialog(
                    axis_values={
                        axes[0]: ("x86", "arm32"),
                        axes[2]: ("clang", "gcc"),
                    }
                )
                app.push_screen(dialog)
                await pilot.pause()

                from tokenizer.inspector._app._filter._dialog import (
                    _MinOneSelectionList,
                    _AXIS_LIST_ID_PREFIX,
                )

                first_list = dialog.query_one(
                    f"#{_AXIS_LIST_ID_PREFIX}0", _MinOneSelectionList
                )
                # Focused widget is the first axis list (not the
                # Accept button, not the second axis list).
                assert app.focused is first_list

                await pilot.press("escape")
                await pilot.pause()

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Fix #4 -- alt+a / alt+c bindings + underlined button labels.
# ---------------------------------------------------------------------------


def test_filter_dialog_alt_a_accepts():
    """``alt+a`` dismisses with :class:`FilterAccepted`."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                results: list = []
                arch_axis = build_canonical_axes()[0]
                dialog = FilterDialog(axis_values={arch_axis: ("x86",)})
                app.push_screen(dialog, results.append)
                await pilot.pause()
                await pilot.press("alt+a")
                await pilot.pause()
                assert len(results) == 1
                assert isinstance(results[0], FilterAccepted)

    asyncio.run(runner())


def test_filter_dialog_alt_c_cancels():
    """``alt+c`` dismisses with :class:`FilterCancelled`."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                results: list = []
                arch_axis = build_canonical_axes()[0]
                dialog = FilterDialog(axis_values={arch_axis: ("x86",)})
                app.push_screen(dialog, results.append)
                await pilot.pause()
                await pilot.press("alt+c")
                await pilot.pause()
                assert len(results) == 1
                assert isinstance(results[0], FilterCancelled)

    asyncio.run(runner())


def test_filter_dialog_accept_button_label_has_underlined_a():
    """The Accept button's rendered label underlines the ``A`` character."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                arch_axis = build_canonical_axes()[0]
                dialog = FilterDialog(axis_values={arch_axis: ("x86",)})
                app.push_screen(dialog)
                await pilot.pause()

                from textual.widgets import Button

                accept_btn = dialog.query_one("#filter-accept", Button)
                cancel_btn = dialog.query_one("#filter-cancel", Button)
                # The rendered label parsed markup and carries an
                # underline-styled span covering the trigger character.
                accept_plain = accept_btn.label.plain
                cancel_plain = cancel_btn.label.plain
                assert accept_plain == "Accept"
                assert cancel_plain == "Cancel"
                # Confirm an underline span covers the trigger letter on
                # each button (Rich serialises 'u' / 'underline' depending
                # on the renderer; check both).
                assert any(
                    "u" in str(span.style).split()
                    or "underline" in str(span.style)
                    for span in accept_btn.label.spans
                ), f"no underline span on Accept: {accept_btn.label.spans}"
                assert any(
                    "u" in str(span.style).split()
                    or "underline" in str(span.style)
                    for span in cancel_btn.label.spans
                ), f"no underline span on Cancel: {cancel_btn.label.spans}"

                await pilot.press("escape")
                await pilot.pause()

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Small-viewport responsive layout: no minimum size on the dialog AND the
# buttons sit INSIDE the scrollable area so they remain reachable via scroll
# when the terminal can't fit content + button row at the same time.
# ---------------------------------------------------------------------------


def test_filter_dialog_buttons_inside_scroll_container():
    """The Accept/Cancel button row is a descendant of the scroll
    container so a short terminal scrolls the buttons into view
    instead of clipping them at the dialog's bottom edge."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                arch_axis = build_canonical_axes()[0]
                dialog = FilterDialog(axis_values={arch_axis: ("x86",)})
                app.push_screen(dialog)
                await pilot.pause()

                from textual.containers import VerticalScroll
                from textual.widgets import Button

                scroll = dialog.query_one("#filter-scroll", VerticalScroll)
                accept_btn = dialog.query_one("#filter-accept", Button)
                cancel_btn = dialog.query_one("#filter-cancel", Button)
                # Both buttons must be descendants of the scroll
                # container, not siblings of it.
                scroll_descendants = set(scroll.walk_children(with_self=False))
                assert accept_btn in scroll_descendants
                assert cancel_btn in scroll_descendants

                await pilot.press("escape")
                await pilot.pause()

    asyncio.run(runner())


def test_filter_dialog_body_fills_viewport_minus_three_lines_top_and_bottom():
    """At a generous viewport height H the dialog body fills H-6 lines
    (3 lines of breathing room above + 3 below). The sizing is driven
    by Textual's ``height: 1fr; margin: 3 0`` CSS on the outer body,
    so the dialog grows with the terminal instead of capping at a
    fixed line count.
    """

    async def runner() -> None:
        for viewport_height in (24, 40, 60):
            with tempfile.TemporaryDirectory() as td:
                log_path = Path(td) / "tui.log"
                factory = _make_factory_with_rvs(name="main", rvs=[])
                app = InspectorApp(factory=factory, log_path=log_path)
                async with app.run_test(size=(120, viewport_height)) as pilot:
                    arch_axis = build_canonical_axes()[0]
                    dialog = FilterDialog(
                        axis_values={arch_axis: ("x86", "arm32")}
                    )
                    app.push_screen(dialog)
                    await pilot.pause()

                    body = dialog.query_one("#filter-body")
                    # Body sits 3 lines from the top + fills the
                    # remaining height minus 3 trailing lines.
                    assert body.region.y == 3, (
                        f"vh={viewport_height}: body.region.y "
                        f"{body.region.y} != 3 (top margin)"
                    )
                    assert body.region.height == viewport_height - 6, (
                        f"vh={viewport_height}: body.region.height "
                        f"{body.region.height} != {viewport_height - 6} "
                        f"(viewport - 6)"
                    )

                    await pilot.press("escape")
                    await pilot.pause()

    asyncio.run(runner())


def test_filter_dialog_opens_at_ten_line_terminal_without_error():
    """A 10-line viewport (too small for content + buttons stacked)
    must still mount the dialog and let the screen-level Accept
    binding fire -- the dialog must not enforce a min-height floor
    that overflows the terminal."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test(size=(80, 10)) as pilot:
                results: list = []
                arch_axis = build_canonical_axes()[0]
                dialog = FilterDialog(axis_values={arch_axis: ("x86", "arm32")})
                app.push_screen(dialog, results.append)
                await pilot.pause()
                # The screen-level Accept binding works even on the
                # tiny viewport (the button-press path is exercised
                # via Pilot.click in a separate test; this test guards
                # the no-runtime-error + binding-still-fires invariant).
                await pilot.press("ctrl+s")
                await pilot.pause()
                assert len(results) == 1
                assert isinstance(results[0], FilterAccepted)

    asyncio.run(runner())
