"""Unit tests for :func:`tokenizer.inspector._render.render_block`.

The renderer's contract crosses ONE boundary: it consumes a parsed
:class:`~tokenizer.token_lists.BlockTokenList` + the parent section's
parsed :class:`~tokenizer.aligned_data.matched_sections_bin.Section`
+ pre-built per-variant invariants (``kind_to_called_idx`` +
``variant_pins``) + three closures/maps (``line_to_name``,
``line_to_provider``, ``callee_arm_resolver``), and emits an ordered
:class:`list` of typed :class:`LineItem` s.

These tests stub the BlockTokenList walk via minimal lookalike
classes that match the slice of the API ``render_block`` actually
touches (``iter_insn(transient=True)`` on the block, ``to_asm_like()``
+ ``iter_tokens()`` on each instruction, ``.token_type`` + ``.id`` on
each metatoken). This keeps the tests focused on the renderer's own
classification logic without dragging in the full vocab + numpy
buffer machinery (covered by token_lists' own tests). The
:class:`Section` and :class:`VariantBlock` inputs are constructed as
real frozen dataclasses because they have no behaviour beyond field
access.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Iterable, List

import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionKind,
    SectionPointerSpec,
)
from tokenizer.aligned_data.matched_sections_bin import (
    MISSING_VARIANT_INDEX,
    CallTarget,
    Section,
    VariantBlock,
)
from tokenizer.inspector._render import (
    AsmLine,
    InlineCallEntry,
    InlineJumpEntry,
    LineItem,
    render_block,
)
from tokenizer.tokens import TokenType


# ---------------------------------------------------------------------------
# Block/instruction stubs (the renderer touches a thin slice of these APIs)
# ---------------------------------------------------------------------------


@dataclass
class _StubToken:
    """Minimal metatoken stand-in: just ``.token_type`` + ``.id``.

    The renderer reads ``token.token_type`` to discriminate (LOCAL_FUNC
    / PLT_FUNC / EXT_FUNC / BLOCK_V2) and ``int(token.id)`` for the
    per-Category counter (calls) or jump-target block index (BLOCK_V2).
    No other attributes are touched, so a flat dataclass suffices.
    """

    token_type: TokenType
    id: int = 0


@dataclass
class _StubInsn:
    """Stand-in for :class:`InsnTokenList`.

    Implements the two methods :func:`_walk_block_instructions` calls:
    ``to_asm_like()`` for the AsmLine text and ``iter_tokens()`` for
    the inline-call/jump scan.
    """

    asm: str
    tokens: List[_StubToken] = field(default_factory=list)

    def to_asm_like(self) -> str:
        return self.asm

    def iter_tokens(self) -> Iterable[_StubToken]:
        return iter(self.tokens)


@dataclass
class _StubBlock:
    """Stand-in for :class:`BlockTokenList`.

    The renderer only calls ``block.iter_insn(transient=True)``;
    ``transient`` is opaque from the renderer's side (the real impl
    uses it to reuse an InsnTokenList shell -- correctness here only
    requires we yield each insn).
    """

    insns: List[_StubInsn] = field(default_factory=list)

    def iter_insn(self, transient: bool = False) -> Iterable[_StubInsn]:
        return iter(self.insns)


# ---------------------------------------------------------------------------
# Section / VariantBlock builders
# ---------------------------------------------------------------------------


def _make_call_target(
    *,
    type_: CallTargetType,
    function_name_ptr: int,
    function_section_ptr: int,
    is_matched: bool = True,
) -> CallTarget:
    return CallTarget(
        function_name_ptr=function_name_ptr,
        function_section_ptr=function_section_ptr,
        type=type_,
        is_matched=is_matched,
    )


def _make_section(call_targets: List[CallTarget]) -> Section:
    return Section(
        function_name_ptr=0,
        section_offset=0,
        call_targets=call_targets,
        variants=[],
    )


def _kind_to_idx(call_targets: List[CallTarget]) -> dict:
    out: dict[CallTargetType, list[int]] = {k: [] for k in CallTargetType}
    for i, ct in enumerate(call_targets):
        out[ct.type].append(i)
    return out


def _make_resolver(spec_for_offset: dict[int, SectionPointerSpec]):
    def resolver(offset: int) -> SectionPointerSpec | None:
        return spec_for_offset.get(int(offset))

    return resolver


# ---------------------------------------------------------------------------
# Common scaffolding for "no calls / jumps" cases
# ---------------------------------------------------------------------------


_EMPTY_SECTION = _make_section([])
_EMPTY_KIND_MAP = _kind_to_idx([])
_NO_PINS: dict[int, int] = {}
_NO_NAMES: dict[int, str] = {}
_NO_PROVIDERS: dict[int, str] = {}


def _resolver_never_called(_offset: int) -> SectionPointerSpec | None:
    raise AssertionError("callee_arm_resolver should not have been invoked")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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


def test_inline_call_entry_kind_matches_call_target_type():
    """Each call-token kind drives one InlineCallEntry whose ``kind``
    equals the looked-up ``CallTarget.type``. The renderer routes
    LOCAL/PLT/EXTERN through one typed dispatch table; this test
    pins all three kinds in one block."""
    call_targets = [
        _make_call_target(
            type_=CallTargetType.LOCAL, function_name_ptr=1, function_section_ptr=100
        ),
        _make_call_target(
            type_=CallTargetType.PLT, function_name_ptr=2, function_section_ptr=200
        ),
        _make_call_target(
            type_=CallTargetType.EXTERN, function_name_ptr=3, function_section_ptr=42
        ),
    ]
    section = _make_section(call_targets)
    kind_to_idx = _kind_to_idx(call_targets)

    block = _StubBlock(
        insns=[
            _StubInsn(
                asm="call local",
                tokens=[_StubToken(TokenType.LOCAL_FUNC, id=0)],
            ),
            _StubInsn(
                asm="call plt",
                tokens=[_StubToken(TokenType.PLT_FUNC, id=0)],
            ),
            _StubInsn(
                asm="call ext",
                tokens=[_StubToken(TokenType.EXT_FUNC, id=0)],
            ),
        ]
    )

    spec = SectionPointerSpec(arm=SectionKind.MATCHED, idx=7)
    items = render_block(
        block=block,
        section=section,
        kind_to_called_idx=kind_to_idx,
        variant_pins=_NO_PINS,
        line_to_name={1: "loc_fn", 2: "plt_fn", 3: "ext_fn"},
        line_to_provider={42: "libc.so.6"},
        callee_arm_resolver=lambda _offset: spec,
    )

    calls = [it for it in items if isinstance(it, InlineCallEntry)]
    assert [c.kind for c in calls] == [
        CallTargetType.LOCAL,
        CallTargetType.PLT,
        CallTargetType.EXTERN,
    ]
    # And counter_id round-trips the token's payload.
    assert [c.counter_id for c in calls] == [0, 0, 0]
    # callee_name resolves via line_to_name.
    assert [c.callee_name for c in calls] == ["loc_fn", "plt_fn", "ext_fn"]


def test_inline_call_entry_provider_set_for_extern_only():
    """EXTERN call_targets key into ``line_to_provider``;
    LOCAL/PLT call_targets unconditionally see ``provider=None`` --
    the renderer routes per-kind via a dispatch table that maps
    LOCAL/PLT to an empty mapping (see ``_provider_sources``)."""
    call_targets = [
        _make_call_target(
            type_=CallTargetType.LOCAL, function_name_ptr=1, function_section_ptr=200
        ),
        _make_call_target(
            type_=CallTargetType.PLT, function_name_ptr=2, function_section_ptr=200
        ),
        _make_call_target(
            type_=CallTargetType.EXTERN, function_name_ptr=3, function_section_ptr=42
        ),
        _make_call_target(
            type_=CallTargetType.EXTERN, function_name_ptr=4, function_section_ptr=99
        ),
    ]
    section = _make_section(call_targets)
    kind_to_idx = _kind_to_idx(call_targets)

    block = _StubBlock(
        insns=[
            _StubInsn(
                asm="call local",
                tokens=[_StubToken(TokenType.LOCAL_FUNC, id=0)],
            ),
            _StubInsn(
                asm="call plt",
                tokens=[_StubToken(TokenType.PLT_FUNC, id=0)],
            ),
            _StubInsn(
                asm="call ext_known",
                tokens=[_StubToken(TokenType.EXT_FUNC, id=0)],
            ),
            _StubInsn(
                asm="call ext_unknown",
                tokens=[_StubToken(TokenType.EXT_FUNC, id=1)],
            ),
        ]
    )

    items = render_block(
        block=block,
        section=section,
        kind_to_called_idx=kind_to_idx,
        variant_pins=_NO_PINS,
        line_to_name={1: "a", 2: "b", 3: "c", 4: "d"},
        line_to_provider={42: "libc.so.6"},
        callee_arm_resolver=lambda _offset: None,
    )

    calls = [it for it in items if isinstance(it, InlineCallEntry)]
    # LOCAL + PLT -> None even though function_section_ptr=200 sits in
    # the same numeric range; the dispatch table never reads
    # line_to_provider for non-EXTERN kinds.
    assert calls[0].provider is None
    assert calls[1].provider is None
    # EXTERN with provider entry -> the mapped string.
    assert calls[2].provider == "libc.so.6"
    # EXTERN whose function_section_ptr is missing from the sidecar -> None.
    assert calls[3].provider is None


def test_inline_call_entry_callee_section_pointer_via_resolver():
    """For LOCAL/PLT, ``callee_section_pointer`` is whatever the
    caller-supplied resolver returns for the call_target's
    ``function_section_ptr``. The renderer NEVER inspects the
    SectionPointerSpec's internals; it threads the closure's
    return value through. EXTERN call_targets aren't bodies the
    inspector can expand into, so the resolver -- given the
    same byte offset -- typically returns ``None`` and that
    None flows straight through."""
    call_targets = [
        _make_call_target(
            type_=CallTargetType.LOCAL, function_name_ptr=1, function_section_ptr=500
        ),
        _make_call_target(
            type_=CallTargetType.PLT, function_name_ptr=2, function_section_ptr=600
        ),
        _make_call_target(
            type_=CallTargetType.EXTERN, function_name_ptr=3, function_section_ptr=42
        ),
    ]
    section = _make_section(call_targets)
    kind_to_idx = _kind_to_idx(call_targets)

    local_spec = SectionPointerSpec(arm=SectionKind.MATCHED, idx=11)
    plt_spec = SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=22)
    resolver = _make_resolver({500: local_spec, 600: plt_spec})

    block = _StubBlock(
        insns=[
            _StubInsn(
                asm="call local",
                tokens=[_StubToken(TokenType.LOCAL_FUNC, id=0)],
            ),
            _StubInsn(
                asm="call plt",
                tokens=[_StubToken(TokenType.PLT_FUNC, id=0)],
            ),
            _StubInsn(
                asm="call ext",
                tokens=[_StubToken(TokenType.EXT_FUNC, id=0)],
            ),
        ]
    )

    items = render_block(
        block=block,
        section=section,
        kind_to_called_idx=kind_to_idx,
        variant_pins=_NO_PINS,
        line_to_name={1: "loc_fn", 2: "plt_fn", 3: "ext_fn"},
        line_to_provider={},
        callee_arm_resolver=resolver,
    )

    calls = [it for it in items if isinstance(it, InlineCallEntry)]
    assert calls[0].callee_section_pointer == local_spec
    assert calls[1].callee_section_pointer == plt_spec
    # EXTERN: resolver returns None for offset=42 (not in the map).
    assert calls[2].callee_section_pointer is None


def test_inline_call_entry_variant_idx_from_pins_or_sentinel():
    """``variant_idx`` is read out of ``variant_pins`` keyed by the
    LOCAL/PLT call_target's section-level ``called_idx``. When the
    pin is missing (e.g. EXTERN call_targets, which the encoder
    filters out of ``per_call_entries`` entirely), the renderer
    falls back to :data:`MISSING_VARIANT_INDEX`."""
    call_targets = [
        _make_call_target(
            type_=CallTargetType.LOCAL, function_name_ptr=1, function_section_ptr=500
        ),
        _make_call_target(
            type_=CallTargetType.EXTERN, function_name_ptr=2, function_section_ptr=42
        ),
    ]
    section = _make_section(call_targets)
    kind_to_idx = _kind_to_idx(call_targets)

    # Pin called_idx=0 (the LOCAL) to section_variant_index=5; leave the
    # EXTERN at called_idx=1 absent so it falls back to MISSING.
    pins = {0: 5}

    block = _StubBlock(
        insns=[
            _StubInsn(
                asm="call local",
                tokens=[_StubToken(TokenType.LOCAL_FUNC, id=0)],
            ),
            _StubInsn(
                asm="call ext",
                tokens=[_StubToken(TokenType.EXT_FUNC, id=0)],
            ),
        ]
    )

    items = render_block(
        block=block,
        section=section,
        kind_to_called_idx=kind_to_idx,
        variant_pins=pins,
        line_to_name={1: "loc_fn", 2: "ext_fn"},
        line_to_provider={},
        callee_arm_resolver=lambda _offset: None,
    )

    calls = [it for it in items if isinstance(it, InlineCallEntry)]
    assert calls[0].variant_idx == 5
    assert calls[1].variant_idx == MISSING_VARIANT_INDEX


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


def test_unknown_callee_name_falls_back_to_question_mark():
    """When ``line_to_name`` has no entry for the call_target's
    ``function_name_ptr``, the renderer surfaces ``"?"`` (the
    documented fallback in :func:`_emit_call_entry`)."""
    call_targets = [
        _make_call_target(
            type_=CallTargetType.LOCAL,
            function_name_ptr=999,
            function_section_ptr=500,
        ),
    ]
    section = _make_section(call_targets)
    kind_to_idx = _kind_to_idx(call_targets)

    block = _StubBlock(
        insns=[
            _StubInsn(
                asm="call ?",
                tokens=[_StubToken(TokenType.LOCAL_FUNC, id=0)],
            ),
        ]
    )

    items = render_block(
        block=block,
        section=section,
        kind_to_called_idx=kind_to_idx,
        variant_pins={},
        line_to_name={},
        line_to_provider={},
        callee_arm_resolver=lambda _o: None,
    )

    calls = [it for it in items if isinstance(it, InlineCallEntry)]
    assert calls[0].callee_name == "?"


def test_render_block_signature_no_unused_params():
    """Pin the cleaned-up keyword-only signature.

    The renderer was scoped down during the inspector-plan audit to
    exactly the parameters it needs. Re-introducing any of
    ``function_data`` / ``block_idx`` / ``fid_sidecar`` /
    ``fid_row_offsets`` / ``batch_row_idx`` / ``vocab_manager`` /
    ``arm`` would reintroduce coupling and re-parse antipatterns
    (see CLAUDE.md "no re-parsing in call chains" + the per-call
    invariant build is the tree-model layer's job).
    """
    sig = inspect.signature(render_block)
    params = set(sig.parameters.keys())
    assert params == {
        "block",
        "section",
        "kind_to_called_idx",
        "variant_pins",
        "line_to_name",
        "line_to_provider",
        "callee_arm_resolver",
    }
    # And every parameter is keyword-only -- positional ordering would
    # turn a call-site rename into a silent semantic shift across this
    # 7-arg boundary.
    for name, p in sig.parameters.items():
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"parameter {name!r} is not keyword-only ({p.kind})"
        )


def test_line_item_union_covers_all_render_output_classes():
    """The :data:`LineItem` union is what downstream consumers
    (the tree model's ``_lift_render_items_to_nodes``) match against.
    Pin it to exactly the three dataclasses the renderer yields so a
    silent expansion of the union surfaces as a test break."""
    # NB: LineItem is a `X | Y | Z` PEP-604 union; the runtime form is
    # ``types.UnionType`` whose ``__args__`` enumerates the members.
    members = set(LineItem.__args__)
    assert members == {AsmLine, InlineCallEntry, InlineJumpEntry}


def test_inline_call_entry_dataclass_field_set():
    """Pin the wire-shape of :class:`InlineCallEntry`.

    Tree-model nodes (``InlineCallNode``) consume these fields by name;
    a silent rename here would break the downstream label / expand
    paths without surfacing in this module's own tests. The set is
    small enough to enumerate explicitly."""
    expected = {
        "kind",
        "counter_id",
        "callee_name",
        "callee_section_pointer",
        "variant_idx",
        "provider",
    }
    assert {f.name for f in InlineCallEntry.__dataclass_fields__.values()} == expected


# ---------------------------------------------------------------------------
# Defensive: VariantBlock construction stays compatible with the parser
# ---------------------------------------------------------------------------


def test_variant_block_roundtrip_via_kind_to_called_idx_helper():
    """Indirect smoke for the module-private helpers
    :func:`_kind_to_called_idx` and :func:`_variant_index_for_called_idx`:
    the public ``render_block`` only sees the BUILT tables, but the
    same writer ordering (LOCAL -> PLT -> EXTERN, encounter-order
    within each) is what makes ``counter_id == index into kind_to_idx``
    correct. Replicating the helper's logic here pins the ordering
    contract."""
    cts = [
        _make_call_target(
            type_=CallTargetType.LOCAL, function_name_ptr=1, function_section_ptr=100
        ),
        _make_call_target(
            type_=CallTargetType.LOCAL, function_name_ptr=2, function_section_ptr=110
        ),
        _make_call_target(
            type_=CallTargetType.PLT, function_name_ptr=3, function_section_ptr=200
        ),
        _make_call_target(
            type_=CallTargetType.EXTERN, function_name_ptr=4, function_section_ptr=42
        ),
    ]
    section = _make_section(cts)

    # Use a VariantBlock as the encoder would emit it (EXTERN filtered
    # out; per_call_entries pin one variant_idx per LOCAL/PLT called).
    variant_block = VariantBlock(
        variant_ref_offset=0,
        data_offset_shifted=0,
        per_call_entries=[(0, 7), (1, 8), (2, 9)],
    )

    kind_to_idx = _kind_to_idx(cts)
    pins = {ci: vi for ci, vi in variant_block.per_call_entries}

    # counter_id=0 for LOCAL -> called_idx 0 -> pin 7
    # counter_id=1 for LOCAL -> called_idx 1 -> pin 8
    # counter_id=0 for PLT   -> called_idx 2 -> pin 9
    block = _StubBlock(
        insns=[
            _StubInsn(asm="l0", tokens=[_StubToken(TokenType.LOCAL_FUNC, id=0)]),
            _StubInsn(asm="l1", tokens=[_StubToken(TokenType.LOCAL_FUNC, id=1)]),
            _StubInsn(asm="p0", tokens=[_StubToken(TokenType.PLT_FUNC, id=0)]),
            _StubInsn(asm="e0", tokens=[_StubToken(TokenType.EXT_FUNC, id=0)]),
        ]
    )

    items = render_block(
        block=block,
        section=section,
        kind_to_called_idx=kind_to_idx,
        variant_pins=pins,
        line_to_name={1: "a", 2: "b", 3: "c", 4: "d"},
        line_to_provider={42: "libc"},
        callee_arm_resolver=lambda _o: None,
    )

    calls = [it for it in items if isinstance(it, InlineCallEntry)]
    assert [c.variant_idx for c in calls] == [7, 8, 9, MISSING_VARIANT_INDEX]
    assert [c.kind for c in calls] == [
        CallTargetType.LOCAL,
        CallTargetType.LOCAL,
        CallTargetType.PLT,
        CallTargetType.EXTERN,
    ]
