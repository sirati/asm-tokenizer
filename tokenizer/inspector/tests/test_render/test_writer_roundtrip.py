"""Writer-ordering roundtrip test for the renderer.

Indirect smoke for the module-private helpers
:func:`_kind_to_called_idx` and :func:`_variant_index_for_called_idx`:
the public ``render_block`` only sees the BUILT tables, but the
same writer ordering (LOCAL -> PLT -> EXTERN, encounter-order within
each) is what makes ``counter_id == index into kind_to_idx`` correct.
Replicating the helper's logic here pins the ordering contract.
"""

from __future__ import annotations

from .conftest import (
    _StubBlock,
    _StubInsn,
    _StubToken,
    _flatten_openables,
    _kind_to_idx,
    _make_call_target,
    _make_section,
)

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_bin import (
    MISSING_VARIANT_INDEX,
    VariantBlock,
)
from tokenizer.inspector._render import InlineCallEntry, render_block
from tokenizer.tokens import TokenType


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
            _StubInsn(tokens=[_StubToken(TokenType.LOCAL_FUNC, id=0)]),
            _StubInsn(tokens=[_StubToken(TokenType.LOCAL_FUNC, id=1)]),
            _StubInsn(tokens=[_StubToken(TokenType.PLT_FUNC, id=0)]),
            _StubInsn(tokens=[_StubToken(TokenType.EXT_FUNC, id=0)]),
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

    calls = [it for it in _flatten_openables(items) if isinstance(it, InlineCallEntry)]
    assert [c.variant_idx for c in calls] == [7, 8, 9, MISSING_VARIANT_INDEX]
    assert [c.kind for c in calls] == [
        CallTargetType.LOCAL,
        CallTargetType.LOCAL,
        CallTargetType.PLT,
        CallTargetType.EXTERN,
    ]
