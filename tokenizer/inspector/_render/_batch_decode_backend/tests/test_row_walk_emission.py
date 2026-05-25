"""Emission-band invariants of :func:`render_row_blocks`.

NUMBER-band trailing-chunk placeholder (decision #17), FUNCTION-band
FID resolution via :class:`FidBaseTable`, EXTERN provider routing,
counter-but-not-BLOCK Phase-1 placeholder.

Plan reference: ``inspector-render-backends.md`` §6 + decisions
#16/#17/#28 + audits B-CRIT-1 / B-CRIT-4 / B-HIGH-6 / B-HIGH-7.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.decoded.custom_float import (
    from_float128,
    from_float32,
    from_int,
)
from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX
from tokenizer.inspector._render._batch_decode_backend._row_walk import (
    render_row_blocks,
)
from tokenizer.inspector._render._protocol import AsmLine, InlineCallEntry

from ._row_walk_fixtures import (
    BLOCK_V2,
    EMPTY_FID_COUNTS,
    EMPTY_FID_SIDECAR,
    EMPTY_NUMBERS,
    EXT_FUNC,
    F128_NUMBER,
    F32_NUMBER,
    LOCAL_FUNC,
    STRING_PTR,
    VC2_NUMBER,
    make_fid_table,
    make_result,
    vocab_stub,
)


def _walk(
    *,
    tokens: np.ndarray,
    identities: np.ndarray,
    numbers_sig: np.ndarray = None,
    numbers_se: np.ndarray = None,
    per_category_counts: np.ndarray = None,
    sidecar: np.ndarray = None,
    line_to_name: dict | None = None,
    line_to_provider: dict | None = None,
    block_runlength: np.ndarray | None = None,
    insn_runlength: np.ndarray | None = None,
    call_targets_per_ct: list | None = None,
    callee_arm_resolver=None,
):
    """Shorthand: ``render_row_blocks`` with n_axis=0 + single-CT span.

    Number / FID arrays default to "empty"; callers populate only what
    their dispatch needs to exercise. The runlength sidecars default
    to empty (every BLOCK_V2 token becomes an InlineJumpEntry) -- tests
    that exercise block-boundary detection supply explicit per-row
    counts.
    """
    from ._row_walk_fixtures import NULL_CALLEE_RESOLVER
    sig_default, se_default = EMPTY_NUMBERS
    return render_row_blocks(
        result=make_result(
            tokens_row=tokens, identities=identities,
            numbers_sig=numbers_sig if numbers_sig is not None else sig_default,
            numbers_se=numbers_se if numbers_se is not None else se_default,
            block_runlength=block_runlength,
            insn_runlength=insn_runlength,
        ),
        row=0, n_axis=0,
        partial_cut_lengths=[int(tokens.shape[0])],
        call_targets_per_ct=(
            call_targets_per_ct if call_targets_per_ct is not None else [[]]
        ),
        vocab_manager=vocab_stub(),
        fid_table=make_fid_table(
            per_category_counts=(
                per_category_counts
                if per_category_counts is not None
                else EMPTY_FID_COUNTS
            ),
            sidecar=sidecar if sidecar is not None else EMPTY_FID_SIDECAR,
        ),
        line_to_name=line_to_name or {},
        line_to_provider=line_to_provider or {},
        callee_arm_resolver=(
            callee_arm_resolver if callee_arm_resolver is not None
            else NULL_CALLEE_RESOLVER
        ),
    )


# ---------------------------------------------------------------------------
# Multi-chunk trailing-slot placeholder (decision #17)
# ---------------------------------------------------------------------------


def test_consecutive_f128_tokens_render_lead_plus_placeholder() -> None:
    """Two consecutive F128 NUMBER tokens belong to the SAME source
    (encoder emits K consecutive same-shifted-id tokens per multi-chunk
    source). Lead chunk -> :func:`chunks_to_hex_bits`; trailing slot ->
    ``AsmLine("...")``.

    Finite F128 -> lead-chunk render is ``"float128:..."`` (Phase-1
    placeholder); the trailing slot still independently emits the
    dedicated ``"..."`` placeholder.
    """
    finite_bits = 0x3FFF0000000000000000000000000000  # 1.0
    chunks = from_float128(finite_bits)
    assert len(chunks) == 2
    numbers_sig = np.asarray([c[0] for c in chunks], dtype=np.uint64)
    numbers_se = np.asarray([c[1] for c in chunks], dtype=np.uint32)
    blocks = _walk(
        tokens=np.asarray(
            [BLOCK_V2, F128_NUMBER, F128_NUMBER, 0], dtype=np.uint16,
        ),
        identities=np.asarray([0], dtype=np.uint16),
        numbers_sig=numbers_sig, numbers_se=numbers_se,
    )
    items = blocks[0].items
    assert len(items) == 2
    assert items[0].text == "float128:..."  # finite F128 placeholder
    assert items[1].text == "..."  # trailing-slot placeholder


def test_consecutive_single_chunk_floats_each_render_independently() -> None:
    """Two consecutive F32 (single-chunk) tokens come from DISTINCT
    sources; each MUST render independently (no trailing-slot dispatch).
    """
    chunks_a = from_float32(0x3F800000)  # +1.0
    chunks_b = from_float32(0x40000000)  # +2.0
    numbers_sig = np.asarray(
        [chunks_a[0][0], chunks_b[0][0]], dtype=np.uint64,
    )
    numbers_se = np.asarray(
        [chunks_a[0][1], chunks_b[0][1]], dtype=np.uint32,
    )
    blocks = _walk(
        tokens=np.asarray(
            [BLOCK_V2, F32_NUMBER, F32_NUMBER, 0], dtype=np.uint16,
        ),
        identities=np.asarray([0], dtype=np.uint16),
        numbers_sig=numbers_sig, numbers_se=numbers_se,
    )
    items = blocks[0].items
    assert len(items) == 2
    assert items[0].text == "float32:3f800000"
    assert items[1].text == "float32:40000000"


def test_single_chunk_vc2_renders_via_inner_text_shape() -> None:
    """A single-chunk VC2 renders via ``reconstruct_chunks`` ->
    ``"v2:<hex>"`` parity with :meth:`ValuedConstV2Inner.to_asm_like`.
    """
    chunks = from_int(0xCAFE)
    numbers_sig = np.asarray([chunks[0][0]], dtype=np.uint64)
    numbers_se = np.asarray([chunks[0][1]], dtype=np.uint32)
    blocks = _walk(
        tokens=np.asarray([BLOCK_V2, VC2_NUMBER, 0], dtype=np.uint16),
        identities=np.asarray([0], dtype=np.uint16),
        numbers_sig=numbers_sig, numbers_se=numbers_se,
    )
    items = blocks[0].items
    assert len(items) == 1
    assert items[0].text == "v2:cafe"


def test_identity_between_number_tokens_resets_trailing_state() -> None:
    """Decision #17 contract: ``last_number_shifted_id`` resets on
    every NON-number token. A row ``[BLOCK_V2(c=0), F32, BLOCK_V2(c=1),
    F32]`` with two identical F32 sources should render BOTH F32s as
    real hex (the IDENTITY in between resets the trailing-chunk state).
    """
    chunks = from_float32(0x3F800000)
    numbers_sig = np.asarray(
        [chunks[0][0], chunks[0][0]], dtype=np.uint64,
    )
    numbers_se = np.asarray(
        [chunks[0][1], chunks[0][1]], dtype=np.uint32,
    )
    blocks = _walk(
        tokens=np.asarray(
            [BLOCK_V2, F32_NUMBER, BLOCK_V2, F32_NUMBER, 0], dtype=np.uint16,
        ),
        identities=np.asarray([0, 1], dtype=np.uint16),
        numbers_sig=numbers_sig, numbers_se=numbers_se,
    )
    # First block: AsmLine (F32 render) + InlineJumpEntry (BLOCK_V2 c=1)
    # Wait: second BLOCK_V2 is an inline-jump under the FIRST block
    # because we have only one partial_cut_length segment. The F32 after
    # the inline jump renders as real hex (state reset by the IDENTITY).
    items = blocks[0].items
    asm_text = [it.text for it in items if isinstance(it, AsmLine)]
    assert "float32:3f800000" in asm_text
    # Both F32 emissions are real hex (no "..." trailing placeholder).
    assert asm_text.count("float32:3f800000") == 2


# ---------------------------------------------------------------------------
# FUNCTION-Category FID resolution
# ---------------------------------------------------------------------------


def test_local_func_resolves_fid_via_line_to_name() -> None:
    """``[BLOCK_V2(c=0), LOCAL_FUNC(c=0)]`` with sidecar=[42] +
    line_to_name={42:'my_func'} -> :class:`InlineCallEntry` carries
    name + LOCAL kind + ``variant_idx=MISSING_VARIANT_INDEX``.
    """
    blocks = _walk(
        tokens=np.asarray([BLOCK_V2, LOCAL_FUNC, 0], dtype=np.uint16),
        identities=np.asarray([0, 0], dtype=np.uint16),
        per_category_counts=np.asarray([[1, 0, 0]], dtype=np.uint32),
        sidecar=np.asarray([42], dtype=np.uint32),
        line_to_name={42: "my_func"},
    )
    call = blocks[0].items[0]
    assert isinstance(call, InlineCallEntry)
    assert call.kind is CallTargetType.LOCAL
    assert call.counter_id == 0
    assert call.callee_name == "my_func"
    assert call.variant_idx == MISSING_VARIANT_INDEX
    assert call.provider is None  # LOCAL never carries provider


def test_ext_func_carries_provider_from_line_to_provider() -> None:
    """EXTERN -> ``provider`` field carries the library name keyed by
    FID. LOCAL leaves provider None even with a matching dict entry
    (provider is EXTERN-only by contract).
    """
    blocks = _walk(
        tokens=np.asarray(
            [BLOCK_V2, EXT_FUNC, LOCAL_FUNC, 0], dtype=np.uint16,
        ),
        identities=np.asarray([0, 0, 0], dtype=np.uint16),
        per_category_counts=np.asarray([[1, 0, 1]], dtype=np.uint32),
        sidecar=np.asarray([100, 200], dtype=np.uint32),  # LOCAL, EXT
        line_to_name={100: "local_callee", 200: "ext_callee"},
        line_to_provider={100: "libc.so", 200: "libm.so"},
    )
    items = blocks[0].items
    assert len(items) == 2
    ext_call, local_call = items
    assert isinstance(ext_call, InlineCallEntry)
    assert ext_call.kind is CallTargetType.EXTERN
    assert ext_call.callee_name == "ext_callee"
    assert ext_call.provider == "libm.so"
    assert isinstance(local_call, InlineCallEntry)
    assert local_call.kind is CallTargetType.LOCAL
    assert local_call.provider is None


def test_unknown_fid_renders_question_mark_via_line_to_name_default() -> None:
    """``line_to_name.get(fid, "?")`` -- a FID missing from the mapping
    falls back to ``"?"``. Pins the default so a name-table miss never
    crashes the walker.
    """
    blocks = _walk(
        tokens=np.asarray([BLOCK_V2, LOCAL_FUNC, 0], dtype=np.uint16),
        identities=np.asarray([0, 0], dtype=np.uint16),
        per_category_counts=np.asarray([[1, 0, 0]], dtype=np.uint32),
        sidecar=np.asarray([999], dtype=np.uint32),
    )
    call = blocks[0].items[0]
    assert isinstance(call, InlineCallEntry)
    assert call.callee_name == "?"


# ---------------------------------------------------------------------------
# Counter-but-not-BLOCK placeholder (Phase-1 plan §11 follow-up)
# ---------------------------------------------------------------------------


def test_counter_category_non_block_emits_placeholder_asmline() -> None:
    """STRING_PTR IDENTITY tokens land on the counter-but-not-BLOCK
    placeholder branch -> ``AsmLine("<category_lower counter>")``. Pins
    the shape so a Phase-2 promotion wiring real label resolution will
    deliberately flip this test.
    """
    blocks = _walk(
        tokens=np.asarray([BLOCK_V2, STRING_PTR, 0], dtype=np.uint16),
        identities=np.asarray([0, 5], dtype=np.uint16),
    )
    item = blocks[0].items[0]
    assert isinstance(item, AsmLine)
    assert item.text == "<string_ptr 5>"
