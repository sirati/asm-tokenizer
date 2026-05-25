"""Kind-dispatch tests for :func:`render_block`.

Pins how the renderer routes each call-token kind (LOCAL / PLT /
EXTERN) through its typed dispatch table: ``kind``, ``provider``,
``callee_section_pointer``, ``variant_idx`` and the ``"?"``
fallback for missing callee names.
"""

from __future__ import annotations

from .conftest import (
    _StubBlock,
    _ct,
    _flatten_openables,
    _insn_with_call,
    _kind_to_idx,
    _make_resolver,
    _make_section,
    _three_kind_block,
    _three_kind_call_targets,
)

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionKind,
    SectionPointerSpec,
)
from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX
from tokenizer.inspector._render import InlineCallEntry, render_block
from tokenizer.tokens import TokenType


def test_inline_call_entry_kind_matches_call_target_type():
    """Each call-token kind drives one InlineCallEntry whose ``kind``
    equals the looked-up ``CallTarget.type``. The renderer routes
    LOCAL/PLT/EXTERN through one typed dispatch table; this test
    pins all three kinds in one block."""
    call_targets = _three_kind_call_targets(
        local_section_ptr=100, plt_section_ptr=200, extern_section_ptr=42
    )
    section = _make_section(call_targets)

    spec = SectionPointerSpec(arm=SectionKind.MATCHED, idx=7)
    items = render_block(
        block=_three_kind_block(),
        section=section,
        kind_to_called_idx=_kind_to_idx(call_targets),
        variant_pins={},
        line_to_name={1: "loc_fn", 2: "plt_fn", 3: "ext_fn"},
        line_to_provider={42: "libc.so.6"},
        callee_arm_resolver=lambda _offset: spec,
    )

    calls = [it for it in _flatten_openables(items) if isinstance(it, InlineCallEntry)]
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
    # Two EXTERNs (one resolvable, one not) need a hand-rolled CT list
    # since the canonical triple only carries a single EXTERN slot.
    call_targets = [
        _ct(CallTargetType.LOCAL, 1, 200),
        _ct(CallTargetType.PLT, 2, 200),
        _ct(CallTargetType.EXTERN, 3, 42),
        _ct(CallTargetType.EXTERN, 4, 99),
    ]
    block = _StubBlock(
        insns=[
            _insn_with_call("call local", TokenType.LOCAL_FUNC),
            _insn_with_call("call plt", TokenType.PLT_FUNC),
            _insn_with_call("call ext_known", TokenType.EXT_FUNC),
            _insn_with_call("call ext_unknown", TokenType.EXT_FUNC, id=1),
        ]
    )

    items = render_block(
        block=block,
        section=_make_section(call_targets),
        kind_to_called_idx=_kind_to_idx(call_targets),
        variant_pins={},
        line_to_name={1: "a", 2: "b", 3: "c", 4: "d"},
        line_to_provider={42: "libc.so.6"},
        callee_arm_resolver=lambda _offset: None,
    )

    calls = [it for it in _flatten_openables(items) if isinstance(it, InlineCallEntry)]
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
    call_targets = _three_kind_call_targets(
        local_section_ptr=500, plt_section_ptr=600, extern_section_ptr=42
    )
    local_spec = SectionPointerSpec(arm=SectionKind.MATCHED, idx=11)
    plt_spec = SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=22)
    resolver = _make_resolver({500: local_spec, 600: plt_spec})

    items = render_block(
        block=_three_kind_block(),
        section=_make_section(call_targets),
        kind_to_called_idx=_kind_to_idx(call_targets),
        variant_pins={},
        line_to_name={1: "loc_fn", 2: "plt_fn", 3: "ext_fn"},
        line_to_provider={},
        callee_arm_resolver=resolver,
    )

    calls = [it for it in _flatten_openables(items) if isinstance(it, InlineCallEntry)]
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
        _ct(CallTargetType.LOCAL, 1, 500),
        _ct(CallTargetType.EXTERN, 2, 42),
    ]
    block = _StubBlock(
        insns=[
            _insn_with_call("call local", TokenType.LOCAL_FUNC),
            _insn_with_call("call ext", TokenType.EXT_FUNC),
        ]
    )

    # Pin called_idx=0 (the LOCAL) to section_variant_index=5; leave the
    # EXTERN at called_idx=1 absent so it falls back to MISSING.
    items = render_block(
        block=block,
        section=_make_section(call_targets),
        kind_to_called_idx=_kind_to_idx(call_targets),
        variant_pins={0: 5},
        line_to_name={1: "loc_fn", 2: "ext_fn"},
        line_to_provider={},
        callee_arm_resolver=lambda _offset: None,
    )

    calls = [it for it in _flatten_openables(items) if isinstance(it, InlineCallEntry)]
    assert calls[0].variant_idx == 5
    assert calls[1].variant_idx == MISSING_VARIANT_INDEX


def test_unknown_callee_name_falls_back_to_question_mark():
    """When ``line_to_name`` has no entry for the call_target's
    ``function_name_ptr``, the renderer surfaces ``"?"`` (the
    documented fallback in :func:`_emit_call_entry`)."""
    call_targets = [_ct(CallTargetType.LOCAL, 999, 500)]
    block = _StubBlock(insns=[_insn_with_call("call ?", TokenType.LOCAL_FUNC)])

    items = render_block(
        block=block,
        section=_make_section(call_targets),
        kind_to_called_idx=_kind_to_idx(call_targets),
        variant_pins={},
        line_to_name={},
        line_to_provider={},
        callee_arm_resolver=lambda _o: None,
    )

    calls = [it for it in _flatten_openables(items) if isinstance(it, InlineCallEntry)]
    assert calls[0].callee_name == "?"
