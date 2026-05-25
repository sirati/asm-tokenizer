"""Emission-band invariants of :func:`render_row_blocks`.

NUMBER-band multi-chunk grouping via :class:`_NumberAccumulator`
(R2c; replaces the legacy ``"..."`` trailing-slot placeholder),
FUNCTION-band FID resolution via :class:`FidBaseTable`, EXTERN
provider routing, counter-but-not-BLOCK Phase-1 placeholder.

Plan reference: ``inspector-render-backends.md`` §6 + decisions
#16/#17/#28 + audits B-CRIT-1 / B-CRIT-4 / B-HIGH-6 / B-HIGH-7;
``inspector-followup.md`` §W3-12 / W3-17 (R2c integration).
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.decoded._number_render import (
    InlineNumberPrecisionEntry,
)
from tokenizer.aligned_data.loader.decoded.custom_float import (
    from_float128,
    from_float32,
    from_float64,
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
# NUMBER-band multi-chunk grouping via _NumberAccumulator (R2c)
# ---------------------------------------------------------------------------


def test_consecutive_f128_tokens_group_into_one_finite_source() -> None:
    """Two consecutive F128 NUMBER tokens with the same shifted id are
    one finite F128 source per encoder discipline (K_visible=2). The
    accumulator buffers BOTH chunks and emits ONE :class:`AsmLine`
    with the short-form rendering when the band switches off NUMBER
    (or the row ends).

    The legacy Phase-1 ``"..."`` trailing-slot placeholder is GONE;
    the accumulator is the SSOT (A-L4 M2).
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
        # Per W3-17 W4-AMENDED, multi-chunk sources MUST live within
        # ONE instruction; supply an insn_runlength sidecar that groups
        # both F128 chunk slots into a single 2-slot instruction (the
        # silent-header pair contains the BLOCK_V2 + an INSTR_REP, but
        # in this bare layout the BLOCK_V2 IS the header so the BODY
        # body contains exactly one 2-slot instruction).
        insn_runlength=np.asarray([2], dtype=np.uint32),
    )
    items = blocks[0].items
    # Exactly ONE AsmLine -- the two chunks reconstructed into a single
    # finite-F128 source rendered at short precision.
    assert len(items) == 1
    assert isinstance(items[0], AsmLine)
    assert items[0].text == "f128:0.1 E1"


def test_consecutive_single_chunk_floats_each_render_independently() -> None:
    """Two consecutive F32 (single-chunk) tokens come from DISTINCT
    sources; each MUST render independently. The accumulator-side
    grouping policy collapses by shifted id, so the row walker
    force-flushes after every non-multi-chunk feed (see
    :data:`MULTI_CHUNK_SHIFTED_IDS` exclusion); without that gate
    these two would mis-group into one mis-rendered source (cluster
    #21 / R1-Audit L-4).
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
    # R1c short form: ``f32:0.<digits> E<exp>``.
    assert items[0].text == "f32:0.1 E1"
    assert items[1].text == "f32:0.2 E1"


def test_consecutive_same_id_single_chunk_floats_render_separately() -> None:
    """Audit gap R1-Audit L-4: two F32 tokens with the SAME shifted id
    AND IDENTICAL chunk payloads ALSO render as two separate AsmLines.

    The accumulator's auto-flush triggers only on shifted_id MISMATCH,
    so without the row walker's force-flush after each single-chunk
    feed (MULTI_CHUNK_SHIFTED_IDS exclusion gate) these two feeds
    would extend ONE source instead of producing two.
    """
    chunks = from_float32(0x3F800000)  # +1.0
    numbers_sig = np.asarray(
        [chunks[0][0], chunks[0][0]], dtype=np.uint64,
    )
    numbers_se = np.asarray(
        [chunks[0][1], chunks[0][1]], dtype=np.uint32,
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
    assert items[0].text == "f32:0.1 E1"
    assert items[1].text == "f32:0.1 E1"


def test_single_chunk_vc2_renders_via_inner_text_shape() -> None:
    """A single-chunk VC2 renders via the R1c short renderer ->
    ``"v:<HEX> (<decimal>)"``.
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
    assert items[0].text == "v:CAFE (51966)"


def test_identity_between_number_tokens_flushes_accumulator() -> None:
    """A non-NUMBER (IDENTITY) token between two same-id NUMBER tokens
    triggers an accumulator flush so the second F32 source renders
    independently of the first.

    Row ``[BLOCK_V2(c=0), F32, BLOCK_V2(c=1), F32]``: the second
    BLOCK_V2 (under the SAME body block) is an inline jump; the F32
    after it MUST render as its own source, not extend the prior.
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
    items = blocks[0].items
    asm_text = [it.text for it in items if isinstance(it, AsmLine)]
    # Both F32 emissions render in the new short form.
    assert asm_text.count("f32:0.1 E1") == 2


def test_float_short_lost_precision_carries_inline_precision_entry() -> None:
    """An F64 value that does NOT fit losslessly in the 4-digit short
    form carries an :class:`InlineNumberPrecisionEntry` on the emitted
    :class:`AsmLine`'s ``openables`` tuple. The tree-model (R2e) later
    expands that entry as a child row.

    Uses pi (irrational): short form ``f64:0.3142 E1`` loses precision
    vs full ``f64:0.<17digits> E1``.
    """
    import math
    # Build a single-chunk F64 of math.pi via from_float64.
    import struct
    pi_bits = struct.unpack(">Q", struct.pack(">d", math.pi))[0]
    chunks = from_float64(pi_bits)
    numbers_sig = np.asarray([chunks[0][0]], dtype=np.uint64)
    numbers_se = np.asarray([chunks[0][1]], dtype=np.uint32)
    blocks = _walk(
        tokens=np.asarray(
            [BLOCK_V2, F32_NUMBER + 1, 0], dtype=np.uint16,
        ),  # F64 shifted id sits at F32_NUMBER + 1 in the NUMBER band.
        identities=np.asarray([0], dtype=np.uint16),
        numbers_sig=numbers_sig, numbers_se=numbers_se,
    )
    items = blocks[0].items
    assert len(items) == 1
    line = items[0]
    assert isinstance(line, AsmLine)
    # The short form truncates pi to 4 digits.
    assert line.text == "f64:0.3142 E1"
    # The precision entry rides along on the openables tuple.
    assert len(line.openables) == 1
    entry = line.openables[0]
    assert isinstance(entry, InlineNumberPrecisionEntry)
    assert entry.full_text == "f64:0.31415926535897931 E1"


def test_multi_chunk_source_spanning_instructions_raises_mid_row() -> None:
    """W3-17 W4-AMENDED encoder-invariant guard (regression pin for
    audit drift D-R2C-1): a multi-chunk NUMBER source MUST live within
    ONE instruction. If the runlength sidecar splits the chunks across
    two adjacent instructions (synthetic encoder-bug scenario), the
    per-instruction finalizer's accumulator-drain MUST raise
    ``AssertionError`` rather than silently emit garbage.

    Construction: a 2-chunk F128 source with
    ``insn_runlength=[1, 1]`` -- the first F128 NUMBER lands in a
    1-slot instruction, finalize fires, the accumulator still has the
    lead chunk pending (because the second chunk has not arrived yet);
    the mid-row assert trips.
    """
    finite_bits = 0x3FFF0000000000000000000000000000  # 1.0
    chunks = from_float128(finite_bits)
    assert len(chunks) == 2
    numbers_sig = np.asarray([c[0] for c in chunks], dtype=np.uint64)
    numbers_se = np.asarray([c[1] for c in chunks], dtype=np.uint32)
    with pytest.raises(AssertionError, match="encoder invariant"):
        _walk(
            tokens=np.asarray(
                [BLOCK_V2, F128_NUMBER, F128_NUMBER, 0], dtype=np.uint16,
            ),
            identities=np.asarray([0], dtype=np.uint16),
            numbers_sig=numbers_sig, numbers_se=numbers_se,
            # Two 1-slot instructions: the lead F128 chunk sits in the
            # first instruction; finalize fires before the trailing
            # chunk arrives -> accumulator has pending mid-row -> raise.
            insn_runlength=np.asarray([1, 1], dtype=np.uint32),
        )


def test_cut_variant_end_of_row_flushes_pending_chunks() -> None:
    """Cluster #21 H-4: a row that ends mid-multi-chunk source MUST
    still emit the lead-chunk contribution (best-effort), not silently
    drop it. Synthetic: a finite F128 source has 2 chunks but the row
    carries only ONE F128 NUMBER token (the second chunk got cut).
    The end-of-row flush emits one :class:`AsmLine` rendering only the
    lead chunk's content.
    """
    finite_bits = 0x3FFF0000000000000000000000000000  # 1.0
    chunks = from_float128(finite_bits)
    assert len(chunks) == 2
    # Only feed the LEAD chunk; trailing chunk is "cut" (not in the row).
    numbers_sig = np.asarray([chunks[0][0]], dtype=np.uint64)
    numbers_se = np.asarray([chunks[0][1]], dtype=np.uint32)
    blocks = _walk(
        tokens=np.asarray([BLOCK_V2, F128_NUMBER, 0], dtype=np.uint16),
        identities=np.asarray([0], dtype=np.uint16),
        numbers_sig=numbers_sig, numbers_se=numbers_se,
    )
    items = blocks[0].items
    # End-of-row flush emits the partial source as best-effort.
    assert len(items) == 1
    assert isinstance(items[0], AsmLine)


# ---------------------------------------------------------------------------
# FUNCTION-Category FID resolution
# ---------------------------------------------------------------------------


def test_local_func_resolves_fid_via_line_to_name() -> None:
    """``[BLOCK_V2(c=0), LOCAL_FUNC(c=0)]`` with sidecar=[42] +
    line_to_name={42:'my_func'} -> :class:`InlineCallEntry` rides on
    the in-flight AsmLine's openables (post-R2 narrowed-LineItem
    contract); carries name + LOCAL kind +
    ``variant_idx=MISSING_VARIANT_INDEX``.
    """
    blocks = _walk(
        tokens=np.asarray([BLOCK_V2, LOCAL_FUNC, 0], dtype=np.uint16),
        identities=np.asarray([0, 0], dtype=np.uint16),
        per_category_counts=np.asarray([[1, 0, 0]], dtype=np.uint32),
        sidecar=np.asarray([42], dtype=np.uint32),
        line_to_name={42: "my_func"},
    )
    line = blocks[0].items[0]
    assert isinstance(line, AsmLine)
    assert len(line.openables) == 1
    call = line.openables[0]
    assert isinstance(call, InlineCallEntry)
    assert call.kind is CallTargetType.LOCAL
    assert call.counter_id == 0
    assert call.callee_name == "my_func"
    assert call.variant_idx == MISSING_VARIANT_INDEX
    assert call.provider is None  # LOCAL never carries provider


def test_ext_func_carries_provider_from_line_to_provider() -> None:
    """EXTERN -> ``provider`` field carries the library name keyed by
    FID. LOCAL leaves provider None even with a matching dict entry
    (provider is EXTERN-only by contract). The InlineCallEntry items
    ride on the per-instruction AsmLines' openables tuples (R2f).
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
    assert all(isinstance(item, AsmLine) for item in items)
    assert len(items) == 2
    ext_line, local_line = items
    assert len(ext_line.openables) == 1
    ext_call = ext_line.openables[0]
    assert isinstance(ext_call, InlineCallEntry)
    assert ext_call.kind is CallTargetType.EXTERN
    assert ext_call.callee_name == "ext_callee"
    assert ext_call.provider == "libm.so"
    assert len(local_line.openables) == 1
    local_call = local_line.openables[0]
    assert isinstance(local_call, InlineCallEntry)
    assert local_call.kind is CallTargetType.LOCAL
    assert local_call.provider is None


def test_unknown_fid_renders_question_mark_via_line_to_name_default() -> None:
    """``line_to_name.get(fid, "?")`` -- a FID missing from the mapping
    falls back to ``"?"``. Pins the default so a name-table miss never
    crashes the walker. The InlineCallEntry rides on the AsmLine's
    openables (R2f narrowed contract).
    """
    blocks = _walk(
        tokens=np.asarray([BLOCK_V2, LOCAL_FUNC, 0], dtype=np.uint16),
        identities=np.asarray([0, 0], dtype=np.uint16),
        per_category_counts=np.asarray([[1, 0, 0]], dtype=np.uint32),
        sidecar=np.asarray([999], dtype=np.uint32),
    )
    line = blocks[0].items[0]
    assert isinstance(line, AsmLine)
    assert len(line.openables) == 1
    call = line.openables[0]
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
