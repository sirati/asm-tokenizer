"""Stream-order tests for :func:`render_block`.

Pins the per-instruction emit order: one :class:`AsmLine` followed by
any inline call/jump entries in token-stream order; empty blocks emit
nothing; the wrapper return type is a flat ``list[LineItem]``.
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


def test_render_block_returns_list_of_line_items():
    """Smoke: a non-empty block returns a list of typed LineItem s."""
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
    # LineItem is a union; just confirm the runtime type is one of the three.
    assert isinstance(items[0], (AsmLine, InlineCallEntry, InlineJumpEntry))


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
    name lookup; jumps are purely intra-function references)."""
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

    jumps = [it for it in items if isinstance(it, InlineJumpEntry)]
    assert jumps == [InlineJumpEntry(target_block_idx=3)]


def test_inline_entries_interleave_with_asm_lines_in_stream_order():
    """One instruction can carry multiple metatokens (e.g. a call site
    with a fallthrough jump); the renderer must emit them in
    token-stream order AFTER the instruction's AsmLine."""
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

    assert isinstance(items[0], AsmLine)
    assert items[0].text == "call+jump"
    assert isinstance(items[1], InlineCallEntry)
    assert items[1].kind == CallTargetType.LOCAL
    assert isinstance(items[2], InlineJumpEntry)
    assert items[2].target_block_idx == 7
