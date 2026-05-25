"""Stream-order tests for :func:`render_block`.

Pins the per-instruction emit shape: one :class:`AsmLine` per
instruction whose :attr:`AsmLine.openables` tuple carries the inline
call/jump entries in token-stream order; empty blocks emit nothing;
the wrapper return type is a flat ``list[AsmLine]``.
"""

from __future__ import annotations

from .conftest import (
    _EMPTY_KIND_MAP,
    _EMPTY_SECTION,
    _NO_NAMES,
    _NO_PINS,
    _NO_PROVIDERS,
    _StubBlock,
    _StubInsn,
    _StubToken,
    _kind_to_idx,
    _make_call_target,
    _make_section,
    _resolver_never_called,
)

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionKind,
    SectionPointerSpec,
)
from tokenizer.inspector._render import (
    AsmLine,
    InlineCallEntry,
    InlineJumpEntry,
    render_block,
)
from tokenizer.tokens import TokenType


def test_render_block_returns_list_of_asm_lines():
    """Smoke: a non-empty block returns a list of :class:`AsmLine` s.

    Post-R2g the FTL renderer emits ONLY :class:`AsmLine` items; inline
    call/jump entries hang off :attr:`AsmLine.openables` instead of
    siblings.
    """
    block = _StubBlock(insns=[_StubInsn(asm="nop", tokens=[])])

    items = render_block(
        block=block,
        section=_EMPTY_SECTION,
        kind_to_called_idx=_EMPTY_KIND_MAP,
        variant_pins=_NO_PINS,
        line_to_name=_NO_NAMES,
        line_to_provider=_NO_PROVIDERS,
        callee_arm_resolver=_resolver_never_called,
    )

    assert isinstance(items, list)
    assert len(items) == 1
    assert isinstance(items[0], AsmLine)
    assert items[0].openables == ()


def test_render_block_emits_asm_lines_for_plain_instructions():
    """No call/jump tokens -> exactly one AsmLine per instruction, in order."""
    block = _StubBlock(
        insns=[
            _StubInsn(asm="mov rax, rbx", tokens=[]),
            _StubInsn(asm="add rax, 1", tokens=[]),
            _StubInsn(asm="ret", tokens=[]),
        ]
    )

    items = render_block(
        block=block,
        section=_EMPTY_SECTION,
        kind_to_called_idx=_EMPTY_KIND_MAP,
        variant_pins=_NO_PINS,
        line_to_name=_NO_NAMES,
        line_to_provider=_NO_PROVIDERS,
        callee_arm_resolver=_resolver_never_called,
    )

    assert items == [
        AsmLine(text="mov rax, rbx"),
        AsmLine(text="add rax, 1"),
        AsmLine(text="ret"),
    ]


def test_render_block_handles_empty_block():
    """A block with zero instructions emits zero items."""
    block = _StubBlock(insns=[])

    items = render_block(
        block=block,
        section=_EMPTY_SECTION,
        kind_to_called_idx=_EMPTY_KIND_MAP,
        variant_pins=_NO_PINS,
        line_to_name=_NO_NAMES,
        line_to_provider=_NO_PROVIDERS,
        callee_arm_resolver=_resolver_never_called,
    )

    assert items == []


def test_inline_jump_entry_target_block_idx():
    """BLOCK_V2 tokens lift to :class:`InlineJumpEntry` carrying the
    token's ``.id`` as the target block index (no section / variant /
    name lookup; jumps are purely intra-function references). The entry
    attaches to its owning instruction's :attr:`AsmLine.openables`.
    """
    block = _StubBlock(
        insns=[
            _StubInsn(
                asm="jmp .L3",
                tokens=[_StubToken(TokenType.BLOCK_V2, id=3)],
            ),
        ]
    )

    items = render_block(
        block=block,
        section=_EMPTY_SECTION,
        kind_to_called_idx=_EMPTY_KIND_MAP,
        variant_pins=_NO_PINS,
        line_to_name=_NO_NAMES,
        line_to_provider=_NO_PROVIDERS,
        callee_arm_resolver=_resolver_never_called,
    )

    assert len(items) == 1
    assert isinstance(items[0], AsmLine)
    assert items[0].openables == (InlineJumpEntry(target_block_idx=3),)


def test_inline_entries_attached_to_asm_line_in_stream_order():
    """One instruction can carry multiple metatokens (e.g. a call site
    with a fallthrough jump); the renderer attaches them to the owning
    instruction's :attr:`AsmLine.openables` in token-stream order, with
    NO sibling Inline*Entry items at the block level.
    """
    call_targets = [
        _make_call_target(
            type_=CallTargetType.LOCAL, function_name_ptr=1, function_section_ptr=500
        ),
    ]
    section = _make_section(call_targets)
    kind_to_idx = _kind_to_idx(call_targets)

    block = _StubBlock(
        insns=[
            _StubInsn(
                asm="call+jump",
                tokens=[
                    _StubToken(TokenType.LOCAL_FUNC, id=0),
                    _StubToken(TokenType.BLOCK_V2, id=7),
                ],
            ),
        ]
    )

    items = render_block(
        block=block,
        section=section,
        kind_to_called_idx=kind_to_idx,
        variant_pins={0: 2},
        line_to_name={1: "loc_fn"},
        line_to_provider={},
        callee_arm_resolver=lambda _o: SectionPointerSpec(
            arm=SectionKind.MATCHED, idx=99
        ),
    )

    # Block-level stream is AsmLine-ONLY post-R2g.
    assert len(items) == 1
    assert isinstance(items[0], AsmLine)
    assert items[0].text == "call+jump"
    # Openables carry the inline entries in token-stream order.
    openables = items[0].openables
    assert len(openables) == 2
    assert isinstance(openables[0], InlineCallEntry)
    assert openables[0].kind == CallTargetType.LOCAL
    assert isinstance(openables[1], InlineJumpEntry)
    assert openables[1].target_block_idx == 7


def test_block_lines_contain_only_asm_lines_post_r2g():
    """Cross-instruction shape pin: the block's emit stream contains
    ONLY :class:`AsmLine` items even when inline entries exist on
    multiple instructions. Sibling Inline*Entry items at the block
    level are gone post-R2g; the tree-model consumes ONE item kind.
    """
    call_targets = [
        _make_call_target(
            type_=CallTargetType.LOCAL, function_name_ptr=1, function_section_ptr=500
        ),
    ]
    block = _StubBlock(
        insns=[
            _StubInsn(asm="nop", tokens=[]),
            _StubInsn(
                asm="call loc_fn",
                tokens=[_StubToken(TokenType.LOCAL_FUNC, id=0)],
            ),
            _StubInsn(
                asm="jmp .L7",
                tokens=[_StubToken(TokenType.BLOCK_V2, id=7)],
            ),
        ]
    )

    items = render_block(
        block=block,
        section=_make_section(call_targets),
        kind_to_called_idx=_kind_to_idx(call_targets),
        variant_pins={},
        line_to_name={1: "loc_fn"},
        line_to_provider={},
        callee_arm_resolver=lambda _o: None,
    )

    # One AsmLine per instruction, never a sibling Inline*Entry.
    assert len(items) == 3
    assert all(isinstance(it, AsmLine) for it in items)
    assert all(
        not isinstance(it, (InlineCallEntry, InlineJumpEntry)) for it in items
    )
    # Openables hang off the owning instruction.
    assert items[0].openables == ()
    assert len(items[1].openables) == 1 and isinstance(
        items[1].openables[0], InlineCallEntry
    )
    assert items[2].openables == (InlineJumpEntry(target_block_idx=7),)


def test_mem_substitution_applied_on_ftl_path():
    """W3-3 W4-amended: the FTL path also substitutes MEM-operand
    asm-value forms (``mem[`` / ``]mem``) to the polished display
    chars (``[`` / ``]``). The substitution applies BEFORE the
    openables migration touches the line; pinning here ensures the
    R2g rewrite did not regress the W3-3 integration.
    """
    block = _StubBlock(insns=[_StubInsn(asm="mov rax , mem[ rbp ]mem", tokens=[])])

    items = render_block(
        block=block,
        section=_EMPTY_SECTION,
        kind_to_called_idx=_EMPTY_KIND_MAP,
        variant_pins=_NO_PINS,
        line_to_name=_NO_NAMES,
        line_to_provider=_NO_PROVIDERS,
        callee_arm_resolver=_resolver_never_called,
    )

    assert len(items) == 1
    assert isinstance(items[0], AsmLine)
    assert items[0].text == "mov rax , [ rbp ]"
