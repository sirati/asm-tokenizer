"""Per-call-site variant_idx pin reading by the row walker.

Pins the BatchDecode emit path's contract: every emitted
:class:`InlineCallEntry` carries the dataloader-recorded per-call-site
pin from :attr:`VariantBlock.per_call_entries` (threaded via
``variant_pins_per_ct``). NO content-similarity matching anywhere;
the pin is either a direct caller-call-site -> callee-variant link
or it is :data:`MISSING_VARIANT_INDEX` (the dataloader recorded no
specific link) and downstream :meth:`InlineCallNode.expand` falls
through to the all-variants surface.

The FUNCTION_ID self-prepend is special-cased: the encoder never
emits a per-call entry for a function's own ID, so the walker stamps
the current row's variant_idx directly (a function's own ID always
resolves to the SAME variant by construction).
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_bin import (
    MISSING_VARIANT_INDEX,
    CallTarget,
)
from tokenizer.inspector._render._batch_decode_backend._row_walk import (
    render_row_blocks,
)
from tokenizer.inspector._render._protocol import (
    AsmLine,
    BlockKind,
    InlineCallEntry,
)

from ._row_walk_fixtures import (
    BLOCK_V2,
    EMPTY_FID_COUNTS,
    EMPTY_FID_SIDECAR,
    EMPTY_NUMBERS,
    INSTR_REP_TOKEN,
    LOCAL_FUNC,
    NULL_CALLEE_RESOLVER,
    make_fid_table,
    make_result,
    vocab_stub,
)


def _local_call_target(*, name_ptr: int, section_ptr: int) -> CallTarget:
    return CallTarget(
        function_name_ptr=name_ptr,
        function_section_ptr=section_ptr,
        type=CallTargetType.LOCAL,
        is_matched=True,
    )


def test_body_local_call_reads_variant_pin_from_per_call_entries() -> None:
    """A LOCAL_FUNC body call site lands on the variant_idx the
    dataloader recorded in :attr:`VariantBlock.per_call_entries` for
    that ``called_idx``. The walker reads off
    ``variant_pins_per_ct[current_call_target_idx]``; this is the
    direct caller-call-site -> callee-variant link, NOT a content
    match.
    """
    sig, se = EMPTY_NUMBERS
    # Row layout: [BLOCK_V2 (header trigger), LOCAL_FUNC (body call),
    # null]. n_axis=0 so no variant-prefix section; partial_cut_length
    # spans the whole row; one CT with one call_target (counter 0 ->
    # called_idx 0). The variant_pins map ``called_idx 0 -> 7``.
    blocks = render_row_blocks(
        result=make_result(
            tokens_row=np.asarray(
                [BLOCK_V2, LOCAL_FUNC, 0], dtype=np.uint16,
            ),
            identities=np.asarray([0, 0], dtype=np.uint16),
            numbers_sig=sig, numbers_se=se,
        ),
        row=0, n_axis=0,
        row_variant_idx=3,
        variant_pins_per_ct=[{0: 7}],
        partial_cut_lengths=[3],
        call_targets_per_ct=[[_local_call_target(name_ptr=42, section_ptr=0)]],
        vocab_manager=vocab_stub(),
        fid_table=make_fid_table(
            per_category_counts=np.asarray([[1, 0, 0]], dtype=np.uint32),
            sidecar=np.asarray([42], dtype=np.uint32),
        ),
        line_to_name={42: "callee"},
        line_to_provider={},
        callee_arm_resolver=NULL_CALLEE_RESOLVER,
    )
    body_lines = [
        item
        for blk in blocks
        if blk.kind is BlockKind.BODY
        for item in blk.items
        if isinstance(item, AsmLine)
    ]
    assert len(body_lines) == 1
    openables = body_lines[0].openables
    assert len(openables) == 1
    call = openables[0]
    assert isinstance(call, InlineCallEntry)
    assert call.variant_idx == 7, (
        f"call site's variant_idx must equal the dataloader's pin "
        f"(per_call_entries[0]=7); got {call.variant_idx}"
    )


def test_body_local_call_missing_pin_yields_missing_variant_index() -> None:
    """The dataloader recorded no specific callee-variant link for
    this call site (the ``called_idx`` is absent from
    :attr:`VariantBlock.per_call_entries`). The walker stamps
    :data:`MISSING_VARIANT_INDEX` so :meth:`InlineCallNode.expand`
    falls through to the all-variants surface — NO content matching.
    """
    sig, se = EMPTY_NUMBERS
    blocks = render_row_blocks(
        result=make_result(
            tokens_row=np.asarray(
                [BLOCK_V2, LOCAL_FUNC, 0], dtype=np.uint16,
            ),
            identities=np.asarray([0, 0], dtype=np.uint16),
            numbers_sig=sig, numbers_se=se,
        ),
        row=0, n_axis=0,
        row_variant_idx=3,
        # Empty pin table for the only CT.
        variant_pins_per_ct=[{}],
        partial_cut_lengths=[3],
        call_targets_per_ct=[[_local_call_target(name_ptr=42, section_ptr=0)]],
        vocab_manager=vocab_stub(),
        fid_table=make_fid_table(
            per_category_counts=np.asarray([[1, 0, 0]], dtype=np.uint32),
            sidecar=np.asarray([42], dtype=np.uint32),
        ),
        line_to_name={42: "callee"},
        line_to_provider={},
        callee_arm_resolver=NULL_CALLEE_RESOLVER,
    )
    body_lines = [
        item
        for blk in blocks
        if blk.kind is BlockKind.BODY
        for item in blk.items
        if isinstance(item, AsmLine)
    ]
    assert len(body_lines) == 1
    call = body_lines[0].openables[0]
    assert isinstance(call, InlineCallEntry)
    assert call.variant_idx == MISSING_VARIANT_INDEX


def test_function_id_self_reference_uses_row_variant_idx() -> None:
    """The FUNCTION_ID synthetic section's LOCAL_FUNC self-prepend
    has no entry in :attr:`VariantBlock.per_call_entries` (the
    encoder doesn't emit one — by construction a function's own ID
    references its own variant). The walker stamps the current row's
    :attr:`_WalkState.row_variant_idx` directly, NOT the per-call
    pin map.
    """
    sig, se = EMPTY_NUMBERS
    # Production-layout layout: variant-prefix (n_axis=2) +
    # LOCAL_FUNC self-prepend + Block_Def + block_v2 + body.
    blocks = render_row_blocks(
        result=make_result(
            tokens_row=np.asarray(
                [INSTR_REP_TOKEN, INSTR_REP_TOKEN, LOCAL_FUNC,
                 INSTR_REP_TOKEN, BLOCK_V2, INSTR_REP_TOKEN, 0],
                dtype=np.uint16,
            ),
            identities=np.asarray([0, 7], dtype=np.uint16),
            numbers_sig=sig, numbers_se=se,
        ),
        row=0, n_axis=2,
        row_variant_idx=5,
        # IMPORTANT: empty pin table — the self-prepend MUST NOT read
        # from here; if it did, it would land on MISSING_VARIANT_INDEX
        # and this test would fail.
        variant_pins_per_ct=[{}],
        partial_cut_lengths=[5],
        call_targets_per_ct=[[]],
        vocab_manager=vocab_stub(),
        fid_table=make_fid_table(
            per_category_counts=np.asarray([[1, 0, 0]], dtype=np.uint32),
            sidecar=np.asarray([42], dtype=np.uint32),
        ),
        line_to_name={42: "my_func"},
        line_to_provider={},
        callee_arm_resolver=NULL_CALLEE_RESOLVER,
    )
    function_id_block = next(
        blk for blk in blocks if blk.kind is BlockKind.FUNCTION_ID
    )
    assert len(function_id_block.items) == 1
    func_id_line = function_id_block.items[0]
    assert isinstance(func_id_line, AsmLine)
    assert len(func_id_line.openables) == 1
    self_call = func_id_line.openables[0]
    assert isinstance(self_call, InlineCallEntry)
    assert self_call.variant_idx == 5, (
        f"FUNCTION_ID self-prepend must stamp row_variant_idx (5); "
        f"got {self_call.variant_idx}"
    )


def test_pin_picks_specific_variant_regardless_of_callee_axes() -> None:
    """Regression: pre-revert the inspector matched the caller's
    canonical-4 build axes against the callee's variant list, which
    was wrong because (a) cross-arm calls could land on
    arch-incompatible variants and (b) the dataloader's pin is the
    direct source of truth for which variant this call site
    references. Pin says "variant 13"; walker stamps 13. End of
    story — the inspector never inspects the callee's axes at row-
    walk time.
    """
    sig, se = EMPTY_NUMBERS
    blocks = render_row_blocks(
        result=make_result(
            tokens_row=np.asarray(
                [BLOCK_V2, LOCAL_FUNC, 0], dtype=np.uint16,
            ),
            identities=np.asarray([0, 0], dtype=np.uint16),
            numbers_sig=sig, numbers_se=se,
        ),
        row=0, n_axis=0,
        row_variant_idx=0,
        variant_pins_per_ct=[{0: 13}],
        partial_cut_lengths=[3],
        call_targets_per_ct=[[_local_call_target(name_ptr=42, section_ptr=0)]],
        vocab_manager=vocab_stub(),
        fid_table=make_fid_table(
            per_category_counts=np.asarray([[1, 0, 0]], dtype=np.uint32),
            sidecar=np.asarray([42], dtype=np.uint32),
        ),
        line_to_name={42: "callee"},
        line_to_provider={},
        callee_arm_resolver=NULL_CALLEE_RESOLVER,
    )
    body_lines = [
        item
        for blk in blocks
        if blk.kind is BlockKind.BODY
        for item in blk.items
        if isinstance(item, AsmLine)
    ]
    assert len(body_lines) == 1
    call = body_lines[0].openables[0]
    assert isinstance(call, InlineCallEntry)
    assert call.variant_idx == 13
