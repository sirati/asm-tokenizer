"""Tests for the FTL-side inline-jump openable resolvability gate.

Mirrors :mod:`tokenizer.inspector._render._batch_decode_backend.tests.test_jump_validity`
on the FtlBackend surface. Pins the contract that prevents the FTL
inspector from KeyError'ing when a jump-table footer (or any
intra-instruction BLOCK_V2 token) names a target id with no body
block in this variant -- the same writer-side root cause documented
in :mod:`tokenizer.inspector._render._jump_validity`.

Two surfaces under test:

* Pure :func:`filter_unresolvable_jump_openables_in_lines` on
  hand-built :class:`AsmLine` sequences (unit-level pins for the
  shared per-line filter -- the same helper the BatchDecode side
  composes through its section wrapper).
* :meth:`FtlBackend.render_block` end-to-end -- a synthetic block
  whose body carries a BLOCK_V2(N) token (the FTL renderer emits
  this as an :class:`InlineJumpEntry`) where N is OR is NOT a body
  block id in this variant; the rendered stream's openables tuple
  reflects the gate's drop / keep decision.

Plan + project rules: single-concern (filter logic lives in
:mod:`.._jump_validity` -- the shared module -- and is composed by
both backends); no parallel indexing (body block_idxs come from
the variant's own ``state.blocks`` tuple, computed once at
:func:`build_variant_state` time, never from a side cache); typed
discriminator (``isinstance(InlineJumpEntry)`` on the Openable union,
no string kind).
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX
from tokenizer.inspector._render._ftl_backend._backend import FtlBackend
from tokenizer.inspector._render._ftl_backend._ftl_section_view import (
    FtlSectionView,
)
from tokenizer.inspector._render._ftl_backend._variant_state import (
    VariantState,
    body_block_idxs_for_blocks,
)
from tokenizer.inspector._render._jump_validity import (
    filter_unresolvable_jump_openables_in_lines,
)
from tokenizer.inspector._render._protocol import (
    AsmLine,
    BlockKind,
    FunctionHandle,
    InlineCallEntry,
    InlineJumpEntry,
)
from tokenizer.token_lists import BlockTokenList
from tokenizer.token_manager import VocabularyManager


# ---------------------------------------------------------------------------
# Unit-level pins on filter_unresolvable_jump_openables_in_lines
# ---------------------------------------------------------------------------


def test_filter_drops_inline_jump_to_missing_body_block() -> None:
    """The live failure shape on the FTL side: a body block carries a
    BLOCK_V2 token whose N has no matching body block header in the
    variant. The pre-fix code would propagate the openable; the gate
    now drops it so the expand path never lands on
    :meth:`FtlBackend.render_block(BODY, missing)` (which would
    :class:`KeyError`).
    """
    lines = (
        AsmLine(text="jump block: 1", openables=(InlineJumpEntry(1),)),
    )
    out = filter_unresolvable_jump_openables_in_lines(lines, frozenset({0}))
    assert len(out) == 1
    asm = out[0]
    assert isinstance(asm, AsmLine)
    # Text preserved (so the row's diagnostic content stays visible);
    # openable dropped because target_block_idx=1 is not in {0}.
    assert asm.text == "jump block: 1"
    assert asm.openables == ()


def test_filter_preserves_inline_jump_to_existing_body_block() -> None:
    """Jumps whose target matches an addressable body block_idx
    survive the gate -- this is the normal intra-function jump case
    and MUST not regress.
    """
    line_a = AsmLine(text="jump block: 0", openables=(InlineJumpEntry(0),))
    line_b = AsmLine(text="jump block: 0", openables=(InlineJumpEntry(0),))
    lines = (line_a, line_b)
    out = filter_unresolvable_jump_openables_in_lines(
        lines, frozenset({0, 1}),
    )
    # Identity-preserving when nothing changes.
    assert out[0] is line_a
    assert out[1] is line_b


def test_filter_preserves_inline_call_openables_untouched() -> None:
    """Non-jump openables (:class:`InlineCallEntry`,
    :class:`InlineNumberPrecisionEntry`) are out of scope for the
    gate -- it filters jump targets only. A jump-to-missing in the
    SAME line drops the jump but keeps the call untouched.
    """
    call = InlineCallEntry(
        kind=CallTargetType.LOCAL, counter_id=0, callee_name="x",
        callee_section_pointer=None, variant_idx=MISSING_VARIANT_INDEX,
        provider=None,
    )
    lines = (
        AsmLine(
            text="call x; jump block: 99",
            openables=(call, InlineJumpEntry(99)),
        ),
    )
    out = filter_unresolvable_jump_openables_in_lines(lines, frozenset({0}))
    asm = out[0]
    assert isinstance(asm, AsmLine)
    assert asm.openables == (call,)


def test_filter_mixed_resolvable_and_phantom_targets_on_same_line() -> None:
    """A jump-table footer line with mixed valid + phantom targets:
    the gate filters each entry independently; the surviving
    openables keep their order matching the original tuple ordering.
    """
    lines = (
        AsmLine(
            text="<jump_table 7> jump block: 0 jump block: 1 jump block: 2",
            openables=(
                InlineJumpEntry(0),  # resolvable -- body 0 exists
                InlineJumpEntry(1),  # phantom    -- no body 1
                InlineJumpEntry(2),  # resolvable -- body 2 exists
            ),
        ),
    )
    out = filter_unresolvable_jump_openables_in_lines(
        lines, frozenset({0, 2}),
    )
    asm = out[0]
    assert isinstance(asm, AsmLine)
    assert asm.openables == (InlineJumpEntry(0), InlineJumpEntry(2))


def test_filter_handles_empty_line_sequence() -> None:
    """The degenerate input (no lines) must short-circuit cleanly."""
    assert filter_unresolvable_jump_openables_in_lines((), frozenset()) == ()


def test_filter_handles_line_with_no_jump_openables() -> None:
    """An :class:`AsmLine` carrying no openables (or only non-jump
    openables) is returned by reference -- no rebuilt frozen
    dataclass copy.
    """
    line = AsmLine(text="add rax, 1", openables=())
    out = filter_unresolvable_jump_openables_in_lines((line,), frozenset({0}))
    assert out[0] is line


# ---------------------------------------------------------------------------
# End-to-end: FtlBackend.render_block applies the gate
# ---------------------------------------------------------------------------


def _vm() -> VocabularyManager:
    """Per-CSV vocab stand-in -- v2 format, no platform binding needed
    for the renderer's per-token discrimination."""
    return VocabularyManager(platform=None, format_version=2)


def _body_block_with_jump(
    vm: VocabularyManager, *, n: int, jump_target_n: int,
) -> BlockTokenList:
    """Build a v2 body block whose body insn carries a BLOCK_V2
    token (= :class:`InlineJumpEntry` after rendering).

    Mirrors the live tokenizer flow: a body block opens with
    ``[Block_Def, Block_V2(n)]`` (silent header pair absorbed by
    :class:`BodyBlockView`) and is followed by body instructions.
    One of those body instructions carries a ``Block_V2(jump_target_n)``
    token to model an intra-function jump. The shared row walker
    (:func:`tokenizer.inspector._render._render_block.render_block`)
    emits this token as an :class:`InlineJumpEntry(jump_target_n)`
    openable on the AsmLine.
    """
    blk = BlockTokenList(2, vocab_manager=vm)
    blk.append_as_insn(
        insn_str=f"block 0x{n:x}",
        tokens=[vm.Block_Def(), vm.Block_V2(n)],
    )
    # Body insn carrying a BLOCK_V2 jump-target token. The renderer
    # treats any BLOCK_V2 outside the header position as an inline
    # jump target (see :data:`_JUMP_TOKEN_TYPES` in _render_block).
    blk.append_as_insn(
        insn_str=f"jmp 0x{jump_target_n:x}",
        tokens=[vm.Block_V2(jump_target_n)],
    )
    return blk


def _stub_variant_state(blocks: List[BlockTokenList]) -> VariantState:
    """Hand-built :class:`VariantState` -- mirrors the helper in
    :mod:`test_backend_block_labels` but with ``body_block_idxs``
    derived from the supplied blocks (matches the real
    :func:`build_variant_state` flow).
    """
    block_tuple = tuple(blocks)
    return VariantState(
        record=MagicMock(name="parsed_record"),
        vocab=MagicMock(name="vocab"),
        ftl=MagicMock(name="function_token_list"),
        view=FtlSectionView(call_targets=()),
        blocks=block_tuple,
        kind_to_called_idx={k: [] for k in CallTargetType},
        line_to_name={},
        line_to_provider={},
        body_block_idxs=body_block_idxs_for_blocks(block_tuple),
    )


def _make_backend_with_state(state: VariantState) -> FtlBackend:
    """Construct a backend whose variant-0 state is pre-populated.

    Bypasses :class:`CsvIndex` (no real CSV on disk) by injecting the
    state directly into the backend's per-variant cache; the
    ``_csv_index`` MagicMock satisfies the constructor's reference
    but is never consulted on the exercised paths.
    """
    handle = FunctionHandle(arm=SectionKind.MATCHED, idx=0, name="fn")
    backend = FtlBackend(csv_index=MagicMock(name="csv_index"), handle=handle)
    backend._variant_states[0] = state
    return backend


def test_render_block_drops_jump_openable_with_no_matching_body_block() -> None:
    """End-to-end repro of the FTL-side KeyError shape.

    A body block N=0 carries a body insn whose BLOCK_V2(99) token
    references a block that doesn't exist (no other block has
    ``block_v2:99`` header). Pre-fix: the renderer emits
    :class:`InlineJumpEntry(99)` on the AsmLine; clicking it builds an
    :class:`InlineJumpNode(target_block_idx=99)`; expand calls
    :meth:`FtlBackend.render_block(BODY, 99)` which
    :class:`KeyError`s in :meth:`FtlBackend._block_for_header`.

    Post-fix: the resolvability gate drops the openable (BODY 99 is
    not in ``state.body_block_idxs == {0}``); the AsmLeaf's
    ``can_expand`` flips to False so the UI never asks the backend
    to render a missing block. The line's text is preserved so the
    diagnostic placeholder content survives.
    """
    vm = _vm()
    # One body block (N=0) whose body carries a jump to N=99.
    blocks = [_body_block_with_jump(vm, n=0, jump_target_n=99)]
    backend = _make_backend_with_state(_stub_variant_state(blocks))

    items = list(
        backend.render_block(variant_idx=0, kind=BlockKind.BODY, block_idx=0)
    )
    # The body insn rendered into one AsmLine (the header pair is
    # absorbed by :class:`BodyBlockView`).
    assert len(items) == 1
    asm = items[0]
    assert isinstance(asm, AsmLine)
    # Critical post-fix invariant: no openable references the missing
    # block. The text placeholder (`jump block: 99`) survives so the
    # diagnostic content is still visible in the row.
    for openable in asm.openables:
        if isinstance(openable, InlineJumpEntry):
            assert openable.target_block_idx == 0, (
                f"InlineJumpEntry(target_block_idx={openable.target_block_idx})"
                f" survived the resolvability gate but has no BODY block "
                f"in this variant's body_block_idxs={{0}}; the gate "
                f"should have dropped it."
            )


def test_render_block_preserves_jump_openable_with_matching_body_block() -> None:
    """Companion to the missing-target test: same shape but the jump
    target ID MATCHES an addressable body block in the variant. Pins
    that the gate's "drop" path is not over-eager -- the legitimate
    intra-function jump case MUST not regress.
    """
    vm = _vm()
    # Two body blocks (N=0, N=1); block 0's body jumps to N=1.
    blocks = [
        _body_block_with_jump(vm, n=0, jump_target_n=1),
        # Use a different jump_target_n in block 1 just to keep the
        # blocks distinct; we never render block 1 in this test.
        _body_block_with_jump(vm, n=1, jump_target_n=0),
    ]
    backend = _make_backend_with_state(_stub_variant_state(blocks))

    items = list(
        backend.render_block(variant_idx=0, kind=BlockKind.BODY, block_idx=0)
    )
    assert len(items) == 1
    asm = items[0]
    assert isinstance(asm, AsmLine)
    # The openable survives because target_block_idx=1 IS in
    # body_block_idxs={0, 1}.
    jump_openables = [
        o for o in asm.openables if isinstance(o, InlineJumpEntry)
    ]
    assert jump_openables == [InlineJumpEntry(target_block_idx=1)]


def test_body_block_idxs_for_blocks_excludes_jump_table_namespace() -> None:
    """:func:`body_block_idxs_for_blocks` collects ONLY
    :attr:`BlockKind.BODY` Ns -- the JUMP_TABLE namespace is
    distinct (a JUMP_TABLE id is not a valid InlineJumpEntry target).

    A function with body block N=0 + jump-table footer N=0 must yield
    ``frozenset({0})`` (the body N), not ``frozenset({0, 0})`` /
    cross-contaminated with the JT id.
    """
    vm = _vm()
    body = _body_block_with_jump(vm, n=0, jump_target_n=0)
    # Build a writer-shaped jump-table footer (jt_id=0).
    jt = BlockTokenList(1, vocab_manager=vm)
    jt.append_as_insn(
        insn_str="jump_table 0x0",
        tokens=[vm.Block_Def(), vm.Jump_Table(0), vm.Block_V2(0)],
    )
    block_tuple = (body, jt)
    body_idxs = body_block_idxs_for_blocks(block_tuple)
    assert body_idxs == frozenset({0})
