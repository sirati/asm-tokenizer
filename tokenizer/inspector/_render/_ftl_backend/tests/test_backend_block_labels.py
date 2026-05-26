"""Pin :meth:`FtlBackend.blocks` + :meth:`FtlBackend.render_block`
against the user-observed mislabel: when the writer assigned block_v2
identities ``[0, 1, 2, 4, 3, 5, ...]`` the inspector displayed them
sequentially (``Block: 0..N``), losing the swap and showing the header
text inside the preview.

These tests inject a hand-built :class:`VariantState` whose
``blocks`` tuple carries explicit non-sequential ``block_v2:N``
headers + body content; we then assert that
:meth:`FtlBackend.blocks` echoes the N values verbatim and that
:meth:`FtlBackend.render_block` looks blocks up by N (not sibling
position) and yields zero header text in the rendered stream.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._render._ftl_backend._backend import FtlBackend
from tokenizer.inspector._render._ftl_backend._ftl_section_view import (
    FtlSectionView,
)
from tokenizer.inspector._render._ftl_backend._variant_state import (
    VariantState,
)
from tokenizer.inspector._render._protocol import (
    AsmLine,
    BlockKind,
    FunctionHandle,
)
from tokenizer.token_lists import BlockTokenList
from tokenizer.token_manager import VocabularyManager


def _vm() -> VocabularyManager:
    return VocabularyManager(platform=None, format_version=2)


def _block_with_header_and_body(
    vm: VocabularyManager, *, n: int, body_text: str
) -> BlockTokenList:
    """Build a v2-style block whose body is one plain instruction.

    The body insn carries a single ``Block_Def`` token (re-used as a
    cheap stand-in for any non-call/non-jump token, since the
    renderer treats anything outside the LOCAL/PLT/EXT/BLOCK_V2 set as
    plain text via ``to_asm_like``). ``body_text`` is the human-
    readable insn string that round-trips through ``insn_strs``.
    """
    blk = BlockTokenList(2, vocab_manager=vm)
    blk.append_as_insn(
        insn_str=f"block 0x{n:x}",
        tokens=[vm.Block_Def(), vm.Block_V2(n)],
    )
    blk.append_as_insn(insn_str=body_text, tokens=[vm.Block_Def()])
    return blk


def _stub_variant_state(blocks: List[BlockTokenList]) -> VariantState:
    """Hand-built :class:`VariantState` carrying only what the
    backend's ``blocks`` / ``render_block`` paths actually read.

    Other fields (``record``, ``view``, ``vocab``, ``line_to_name``,
    ``line_to_provider``, ``kind_to_called_idx``) are filled with
    minimum-surface stand-ins; the tests only assert on per-block
    labels + the body-text stream.
    """
    return VariantState(
        record=MagicMock(name="parsed_record"),
        vocab=MagicMock(name="vocab"),
        ftl=MagicMock(name="function_token_list"),
        view=FtlSectionView(call_targets=()),
        blocks=tuple(blocks),
        kind_to_called_idx={k: [] for k in CallTargetType},
        line_to_name={},
        line_to_provider={},
    )


def _make_backend_with_state(state: VariantState) -> FtlBackend:
    """Construct a backend whose variant-0 state is pre-populated.

    Bypasses :class:`CsvIndex` (no real CSV on disk) by injecting
    the state directly into the backend's per-variant cache; the
    ``_csv_index`` MagicMock satisfies the constructor's reference
    but is never consulted on the exercised paths.
    """
    handle = FunctionHandle(arm=SectionKind.MATCHED, idx=0, name="fn")
    backend = FtlBackend(csv_index=MagicMock(name="csv_index"), handle=handle)
    backend._variant_states[0] = state
    return backend


# ---------------------------------------------------------------------------
# blocks(): block_idx mirrors block_v2 N
# ---------------------------------------------------------------------------


def test_blocks_block_idx_matches_block_v2_not_sibling_position() -> None:
    """User-observed bug: the FTL backend labelled blocks by sibling
    position (``[0, 1, 2, 3, 4, ...]``) even when the underlying
    ``block_v2:N`` sequence was non-monotone. The fix wires
    :attr:`RenderedBlock.block_idx` to the header pair's N value.
    """
    vm = _vm()
    block_v2_ns = [0, 1, 2, 4, 3, 5]
    blocks = [
        _block_with_header_and_body(vm, n=n, body_text=f"insn-{n}")
        for n in block_v2_ns
    ]
    backend = _make_backend_with_state(_stub_variant_state(blocks))

    rendered = backend.blocks(variant_idx=0)

    assert [rb.block_idx for rb in rendered] == block_v2_ns
    # The two swapped slots prove the bug: in the old code
    # ``rendered[3].block_idx`` was 3 (position), now it is 4 (N).
    assert rendered[3].block_idx == 4
    assert rendered[4].block_idx == 3


def test_blocks_preview_does_not_include_header_pair() -> None:
    """The ``_def block_v2:N`` prefix must NOT bleed into the
    per-block preview text. Mirrors BatchDecode's
    :attr:`WalkSectionState.pending_header` policy on the FTL side.
    """
    vm = _vm()
    blocks = [_block_with_header_and_body(vm, n=2, body_text="real-body")]
    backend = _make_backend_with_state(_stub_variant_state(blocks))

    rendered = backend.blocks(variant_idx=0)
    preview = rendered[0].preview

    assert "block_v2:2" not in preview
    assert "_def block_v2" not in preview


# ---------------------------------------------------------------------------
# render_block(): lookup by N + body-only stream
# ---------------------------------------------------------------------------


def test_render_block_looks_up_block_by_v2_n_not_position() -> None:
    """``render_block(N)`` must find the block whose header is
    ``block_v2:N``, regardless of where it sits in
    ``state.blocks``. The pre-fix code did ``state.blocks[N]`` and
    rendered the wrong block.
    """
    vm = _vm()
    # block_v2 ids [0, 4, 3] -- positional idx 1 holds N=4, idx 2 holds N=3.
    blocks = [
        _block_with_header_and_body(vm, n=0, body_text="body-zero"),
        _block_with_header_and_body(vm, n=4, body_text="body-four"),
        _block_with_header_and_body(vm, n=3, body_text="body-three"),
    ]
    backend = _make_backend_with_state(_stub_variant_state(blocks))

    items_for_3 = list(
        backend.render_block(variant_idx=0, kind=BlockKind.BODY, block_idx=3)
    )
    items_for_4 = list(
        backend.render_block(variant_idx=0, kind=BlockKind.BODY, block_idx=4)
    )

    # Each block has exactly ONE body insn (the header is absorbed).
    assert len(items_for_3) == 1
    assert len(items_for_4) == 1
    # The body insn carries a single ``Block_Def`` token rendered as
    # ``"_def"`` by :meth:`BlockDefInner.to_asm_like`; this proves we
    # rendered ONE body line (not the header pair).
    assert isinstance(items_for_3[0], AsmLine)
    assert items_for_3[0].text == "_def"
    assert items_for_4[0].text == "_def"


def test_render_block_raises_on_unknown_v2_id() -> None:
    """A stale block_idx must surface as a :class:`KeyError`; silent
    fallthrough would let a corrupt jump target render a wrong block.
    """
    import pytest

    vm = _vm()
    blocks = [_block_with_header_and_body(vm, n=0, body_text="body")]
    backend = _make_backend_with_state(_stub_variant_state(blocks))

    with pytest.raises(KeyError, match="BODY, 99"):
        list(backend.render_block(variant_idx=0, kind=BlockKind.BODY, block_idx=99))


def _jump_table_footer_block(
    vm: VocabularyManager, *, jt_id: int, target_ns: list[int]
) -> BlockTokenList:
    """Build a writer-shaped jump-table footer block.

    Mirrors :func:`tokenizer.fill_constant_candidates._emit_jump_table_footer_for`:
    one synthetic instruction carrying
    ``[Block_Def, Jump_Table(jt_id), Block_V2(t0), ...]``. The whole
    block has exactly one instruction (the writer never adds bodies
    to footer blocks).
    """
    blk = BlockTokenList(1, vocab_manager=vm)
    target_tokens = [vm.Block_V2(t) for t in target_ns]
    blk.append_as_insn(
        insn_str=f"jump_table 0x{jt_id:x}",
        tokens=[vm.Block_Def(), vm.Jump_Table(jt_id), *target_tokens],
    )
    return blk


def test_blocks_emits_jump_table_kind_for_footer_blocks() -> None:
    """User-reported crash: a function with a jump-table footer block
    crashed the inspector with
    ``ValueError("block does not open with [BLOCK_DEF, BLOCK_V2] ...")``
    because the gate rejected the JUMP_TABLE header. Post-fix the
    backend must emit a :attr:`BlockKind.JUMP_TABLE` RenderedBlock per
    footer block, with ``block_idx`` carrying the JUMP_TABLE id (not
    a BLOCK_V2 id).

    The body block + jump-table footer sit as siblings in the
    variant's blocks tuple -- mirroring writer order via
    :func:`func_tokens.add_block`. The renderer must yield both as
    separate :class:`RenderedBlock` entries.
    """
    vm = _vm()
    blocks = [
        _block_with_header_and_body(vm, n=0, body_text="body"),
        _jump_table_footer_block(vm, jt_id=7, target_ns=[0, 0, 0]),
    ]
    backend = _make_backend_with_state(_stub_variant_state(blocks))

    rendered = backend.blocks(variant_idx=0)

    assert [(rb.kind, rb.block_idx) for rb in rendered] == [
        (BlockKind.BODY, 0),
        (BlockKind.JUMP_TABLE, 7),
    ]


def test_render_block_supports_jump_table_kind() -> None:
    """``render_block(kind=JUMP_TABLE, block_idx=N)`` must locate the
    footer block by N in the JUMP_TABLE namespace -- distinct from
    the BLOCK_V2 namespace so a coincidental id collision (e.g. a
    body block N=7 + a jump-table N=7 in the same function) does
    not cross-resolve.
    """
    vm = _vm()
    # Body block N=7 + footer N=7 -- same int N, different namespace.
    blocks = [
        _block_with_header_and_body(vm, n=7, body_text="body-seven"),
        _jump_table_footer_block(vm, jt_id=7, target_ns=[7]),
    ]
    backend = _make_backend_with_state(_stub_variant_state(blocks))

    body_items = list(
        backend.render_block(variant_idx=0, kind=BlockKind.BODY, block_idx=7)
    )
    jt_items = list(
        backend.render_block(
            variant_idx=0, kind=BlockKind.JUMP_TABLE, block_idx=7
        )
    )

    # Body block has one body insn (the "_def" placeholder); the
    # jump-table footer has zero body insns (the synthetic 1-insn
    # header is fully absorbed by BodyBlockView).
    assert len(body_items) == 1
    assert isinstance(body_items[0], AsmLine)
    assert body_items[0].text == "_def"
    # Jump-table footer has no body insn so the rendered stream is
    # empty (mirrors BatchDecode's silent-header-only emission).
    assert jt_items == []


def test_render_block_stream_omits_header_pair() -> None:
    """The rendered AsmLine stream must NOT carry an item whose text
    starts with ``_def block_v2:``. Mirrors BatchDecode's silent-
    header policy.
    """
    vm = _vm()
    blocks = [_block_with_header_and_body(vm, n=7, body_text="body")]
    backend = _make_backend_with_state(_stub_variant_state(blocks))

    items = list(
        backend.render_block(variant_idx=0, kind=BlockKind.BODY, block_idx=7)
    )

    assert all("block_v2:7" not in it.text for it in items)
