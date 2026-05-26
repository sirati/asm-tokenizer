"""Tests for the block-row preview suffix (5-fix bundle).

Pins the user-facing contract of the block-row preview text:

* Fix #1 -- the preview is sourced from the SAME data
  :meth:`RenderBackend.render_block` returns, so the asm text shown
  next to ``Block: <i>`` is guaranteed to match the items the row
  yields on expand (no off-by-one or variant-threading drift).
* Fix #2 -- the preview lists EVERY instruction in the block joined
  with ``"; "``, not just the first.
* Fix #3 -- expanding a block row drops its preview suffix (the user
  is now looking at the content); collapsing restores it.
* Fix #4 -- the ``p`` key (and :meth:`InspectorApp.action_toggle_preview`)
  flips a global flag and repaints every visible :class:`BlockNode`
  row's label.
* Fix #5 -- the preview chunk is rendered with the muted
  :data:`_BLOCK_PREVIEW_STYLE` (``dim``) so the user visually
  distinguishes it from the row's identity prefix.

Most tests are pure unit tests against the label composer; the
``p``-binding test uses :class:`InspectorApp.run_test` and so is
gated on ``pytest.importorskip("textual")``.
"""

from __future__ import annotations

import asyncio
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._app._labels import (
    _BLOCK_PREVIEW_STYLE,
    _block_node_label,
    _compose_label,
)
from tokenizer.inspector._label import block_preview_from_asm_texts
from tokenizer.inspector._render._batch_decode_backend._backend import (
    _preview_for_section,
)
from tokenizer.inspector._render._batch_decode_backend._sections import RowSection
from tokenizer.inspector._render._protocol import (
    AsmLine,
    BackendFactory,
    BlockKind,
    FunctionHandle,
    RenderBackend,
    RenderedBlock,
    RenderedVariant,
)
from tokenizer.inspector._tree_model import BlockNode
from tokenizer.variant_info import VariantIdentity
from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block_node(*, block_idx: int, preview: str) -> BlockNode:
    """One :class:`BlockNode` against mock factory + backend.

    Sufficient for label-composer tests; the descendant expand path
    is exercised separately (see :mod:`.test_tree_model`).
    """
    return BlockNode(
        factory=MagicMock(spec=BackendFactory),
        backend=MagicMock(spec=RenderBackend),
        variant_idx=0,
        kind=BlockKind.BODY,
        block_idx=block_idx,
        preview=preview,
    )


def _label_axes() -> dict:
    return types.MappingProxyType(
        {
            ARCH_PREFIX: "x86",
            COMP_PREFIX: "clang",
            CVER_PREFIX: "8.0",
            OPT_PREFIX: "O3",
        }
    )


def _variant_identity() -> VariantIdentity:
    return VariantIdentity(
        arch="x86",
        compiler="clang",
        compiler_version="8.0",
        opt="O3",
        pkg="",
        variant_id=0,
    )


def _build_app(rendered_blocks: list[RenderedBlock]):
    """Minimal :class:`InspectorApp` wired to a mock backend with the
    supplied :class:`RenderedBlock` list -- one variant, one function."""
    from tokenizer.inspector._app import InspectorApp

    handle = FunctionHandle(arm=SectionKind.MATCHED, idx=0, name="main")
    backend = MagicMock(spec=RenderBackend)
    backend.handle = handle
    backend.closed = False
    backend.variants.return_value = [
        RenderedVariant(
            variant_idx=0,
            label_axes=_label_axes(),
            extra_metadata=types.MappingProxyType({}),
            variant_identity=_variant_identity(),
        )
    ]
    backend.blocks.return_value = list(rendered_blocks)
    backend.render_block.return_value = ()
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [handle]
    factory.make.return_value = backend
    return InspectorApp, factory


# ---------------------------------------------------------------------------
# Fix #1 -- preview sourced from the same items render_block returns
# ---------------------------------------------------------------------------


def test_preview_for_section_matches_render_block_items_first_line():
    """The BatchDecode preview helper reads the SAME ``section.items``
    list :meth:`render_block` returns -- the joined-and-truncated text
    starts with the first AsmLine's text by construction. Pins the
    contract that an expand of the block shows asm whose first line
    matches the preview's leading fragment."""
    section = RowSection(
        kind=BlockKind.BODY,
        block_idx=151,
        items=[
            AsmLine(text="ldr r0 [r11 + v:20 (32)]"),
            AsmLine(text="cmp r0 0"),
            AsmLine(text="b ne jump block: 522"),
        ],
    )

    preview = _preview_for_section(section)

    # The preview is computed from the EXACT items render_block would
    # return for (BODY, 151). The first instruction's text MUST appear
    # at the head of the preview.
    assert preview.startswith("ldr r0 [r11 + v:20 (32)]")
    # And by construction every item appears (until the max_chars cap):
    assert "cmp r0 0" in preview
    assert "b ne jump block: 522" in preview


def test_preview_for_section_skips_function_id_section_with_no_asmlines():
    """The FUNCTION_ID section commonly carries no AsmLines (its single
    entry is an :class:`InlineCallEntry` for the self-prepend); the
    preview falls through to the empty string -- the UI labels that
    section with a fixed name, not a preview suffix."""
    # AsmLine-less section: simulates the FUNCTION_ID case.
    section = RowSection(
        kind=BlockKind.FUNCTION_ID,
        block_idx=-1,
        items=[],
    )
    assert _preview_for_section(section) == ""


# ---------------------------------------------------------------------------
# Fix #2 -- preview joins ALL asm-line texts with "; "
# ---------------------------------------------------------------------------


def test_block_preview_from_asm_texts_joins_with_semicolon_space():
    """Multi-instruction join contract: ``"; "`` separator matches
    :meth:`BlockTokenList.to_asm_like` (FTL preview) so both backends'
    previews share the same visual shape."""
    out = block_preview_from_asm_texts(
        ["mov eax 0", "add eax 1", "ret"], max_chars=80
    )
    assert out == "mov eax 0; add eax 1; ret"


def test_block_preview_from_asm_texts_truncates_to_max_chars():
    """The 80-char cap policy lives in :mod:`tokenizer.inspector._label`
    (single source of truth across backends); inputs that exceed it
    truncate without an overflow marker (the UI's
    :func:`apply_truncation_marker` owns the ``>>`` suffix)."""
    texts = ["instr" + str(i) for i in range(100)]
    out = block_preview_from_asm_texts(texts, max_chars=80)
    assert len(out) == 80
    assert ">>" not in out


def test_block_preview_from_asm_texts_empty_iterable_returns_empty():
    """Empty input yields the empty string; the UI then suppresses the
    preview suffix entirely (no leading whitespace gap)."""
    assert block_preview_from_asm_texts([]) == ""


def test_preview_for_section_joins_all_asmlines():
    """Three-instruction body -> all three texts join in the preview;
    pins Fix #2 (the bug observation was "preview is only the first
    instruction; should be every instruction in the block")."""
    section = RowSection(
        kind=BlockKind.BODY,
        block_idx=3,
        items=[
            AsmLine(text="push r1"),
            AsmLine(text="mov eax 7"),
            AsmLine(text="pop r1"),
        ],
    )
    preview = _preview_for_section(section)
    assert preview == "push r1; mov eax 7; pop r1"


# ---------------------------------------------------------------------------
# Fix #5 -- preview chunk styled with the muted _BLOCK_PREVIEW_STYLE
# ---------------------------------------------------------------------------


def test_block_node_label_preview_chunk_is_dim_styled():
    """The label composes a two-chunk :class:`Text`: a plain prefix
    (``"Block: <i>   "``) + a styled preview chunk; the muted style
    visually distinguishes the row's identity from its asm preview."""
    node = _make_block_node(block_idx=42, preview="mov eax 0; ret")
    text = _block_node_label(node, show_preview=True)

    # The plain string mirrors the legacy "Block: i   preview" form.
    assert text.plain == "Block: 42   mov eax 0; ret"

    # Walk the styled spans and confirm at least one span over the
    # preview region carries the muted style. ``Text.spans`` is the
    # rich.text representation -- our composer calls ``Text.append``
    # with the dim style, so one span MUST match.
    preview_start = len("Block: 42   ")
    styled_spans = [
        span for span in text.spans if span.style == _BLOCK_PREVIEW_STYLE
    ]
    assert styled_spans, (
        f"no preview span styled {_BLOCK_PREVIEW_STYLE!r}; "
        f"spans={text.spans!r}"
    )
    # The styled span covers the preview text -- starting AT the
    # preview boundary, not before it (the prefix stays unstyled).
    assert any(span.start == preview_start for span in styled_spans)


def test_block_node_label_empty_preview_renders_no_suffix():
    """An empty preview string (e.g. the FUNCTION_ID fallthrough) emits
    JUST the prefix, with no trailing whitespace / no styled chunk."""
    node = _make_block_node(block_idx=0, preview="")
    text = _block_node_label(node, show_preview=True)
    assert text.plain == "Block: 0"
    # No styled span when the preview is empty.
    assert not [
        span for span in text.spans if span.style == _BLOCK_PREVIEW_STYLE
    ]


# ---------------------------------------------------------------------------
# Fix #3 -- expanded block row drops its preview suffix
# ---------------------------------------------------------------------------


def test_block_node_label_show_preview_false_drops_suffix():
    """``show_preview=False`` (the "this block is currently expanded"
    case) elides the preview suffix entirely; only the bare
    ``"Block: <i>"`` prefix remains."""
    node = _make_block_node(block_idx=7, preview="mov eax 0; ret")
    text = _block_node_label(node, show_preview=False)
    assert text.plain == "Block: 7"
    assert not [
        span for span in text.spans if span.style == _BLOCK_PREVIEW_STYLE
    ]


def test_compose_label_threads_show_block_preview_through():
    """:func:`_compose_label` forwards ``show_block_preview`` to the
    BlockNode-specific path; non-BlockNode rows ignore the flag (their
    label has no preview concept)."""
    node = _make_block_node(block_idx=5, preview="add r0 1")
    with_preview = _compose_label(node, show_block_preview=True)
    without_preview = _compose_label(node, show_block_preview=False)

    assert with_preview.plain.endswith("add r0 1")
    assert without_preview.plain == "Block: 5"


def test_block_label_refreshes_on_expand_and_collapse():
    """The App's :meth:`_on_node_expanded` refreshes the just-expanded
    block-row's label dropping the preview suffix; the mirror
    :meth:`_on_node_collapsed` restores it. End-to-end pilot."""

    async def runner() -> None:
        # Two blocks so the single-child-chain collapser doesn't elide
        # the variant + sole-block wrapper away from beneath the
        # FunctionNode; the second block keeps both wrappers in place.
        rendered_blocks = [
            RenderedBlock(
                kind=BlockKind.BODY,
                block_idx=151,
                preview="ldr r0 [r11]; cmp r0 0",
            ),
            RenderedBlock(
                kind=BlockKind.BODY,
                block_idx=152,
                preview="nop",
            ),
        ]
        InspectorApp, factory = _build_app(rendered_blocks)

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test(size=(140, 50)) as pilot:
                tree = app.query_one("#tree")
                # Expand the FunctionNode -> VariantNode chain so the
                # BlockNode row mounts. The first FunctionNode is the
                # only child off the root.
                fn_node = tree.root.children[0]
                fn_node.expand()
                await pilot.pause()
                # The lone-child collapser elides the single variant
                # wrapper away (two blocks keep the variant in place
                # though); walk down to the BlockNode row whose
                # ``block_idx`` matches the row we want to refresh.
                target = _find_block_tree_node(tree.root, block_idx=151)
                assert target is not None, "BlockNode row never mounted"

                # Pre-expand: the label carries the preview suffix.
                assert "ldr r0" in target.label.plain

                # Expand triggers the dispatcher; even with empty
                # ``render_block`` the post-expand label refresh fires.
                target.expand()
                await pilot.pause()
                assert target.label.plain == "Block: 151"

                # Collapse restores the preview suffix.
                target.collapse()
                await pilot.pause()
                assert "ldr r0" in target.label.plain

    asyncio.run(runner())


def _find_block_tree_node(root, *, block_idx: int | None = None):
    """Iterative DFS for the first tree node whose payload is a
    :class:`BlockNode`. When ``block_idx`` is supplied the search
    targets the specific block id; otherwise returns the first match
    in DFS order."""
    # Iterative BFS so the children scan preserves insertion order
    # (DFS with stack.pop would surface block_idx=13 before 12 in the
    # 2-block test fixtures, masking which row was refreshed).
    queue: list = list(root.children)
    while queue:
        node = queue.pop(0)
        data = node.data
        if isinstance(data, BlockNode):
            if block_idx is None or data.block_idx == block_idx:
                return node
        queue.extend(node.children)
    return None


# ---------------------------------------------------------------------------
# Fix #4 -- `p` toggles preview globally; all BlockNode labels refresh
# ---------------------------------------------------------------------------


def test_action_toggle_preview_flips_flag_and_refreshes_block_rows():
    """Calling :meth:`InspectorApp.action_toggle_preview` flips the
    flag AND repaints every visible :class:`BlockNode` row's label;
    pressing ``p`` invokes the same action via the binding."""

    async def runner() -> None:
        # Two blocks so the auto-collapse doesn't elide the variant
        # + sole-block wrappers away from beneath the FunctionNode.
        rendered_blocks = [
            RenderedBlock(kind=BlockKind.BODY, block_idx=88, preview="nop"),
            RenderedBlock(kind=BlockKind.BODY, block_idx=89, preview="ret"),
        ]
        InspectorApp, factory = _build_app(rendered_blocks)

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test(size=(140, 50)) as pilot:
                tree = app.query_one("#tree")
                tree.root.children[0].expand()
                await pilot.pause()
                target = _find_block_tree_node(tree.root, block_idx=88)
                assert target is not None
                assert "nop" in target.label.plain
                assert app._preview_enabled is True

                # Toggle off -- the BlockNode row label loses its preview.
                await pilot.press("p")
                await pilot.pause()
                assert app._preview_enabled is False
                assert target.label.plain == "Block: 88"

                # Toggle on again -- the preview suffix returns.
                await pilot.press("p")
                await pilot.pause()
                assert app._preview_enabled is True
                assert "nop" in target.label.plain

    asyncio.run(runner())


def test_action_toggle_preview_keeps_expanded_block_preview_hidden():
    """The global toggle composes with the per-row "is expanded"
    gate: an already-expanded block stays preview-less even when the
    flag is True, because :meth:`_block_label_show_preview` AND-s
    both gates."""

    async def runner() -> None:
        rendered_blocks = [
            RenderedBlock(kind=BlockKind.BODY, block_idx=12, preview="ret"),
            RenderedBlock(kind=BlockKind.BODY, block_idx=13, preview="nop"),
        ]
        InspectorApp, factory = _build_app(rendered_blocks)

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test(size=(140, 50)) as pilot:
                tree = app.query_one("#tree")
                tree.root.children[0].expand()
                await pilot.pause()
                target = _find_block_tree_node(tree.root, block_idx=12)
                assert target is not None

                # Expand the block -- preview hidden.
                target.expand()
                await pilot.pause()
                assert target.label.plain == "Block: 12"

                # Toggle global flag off + on; expanded block stays
                # preview-less throughout (per-row gate dominates).
                await pilot.press("p")
                await pilot.pause()
                assert target.label.plain == "Block: 12"
                await pilot.press("p")
                await pilot.pause()
                assert target.label.plain == "Block: 12"

    asyncio.run(runner())
