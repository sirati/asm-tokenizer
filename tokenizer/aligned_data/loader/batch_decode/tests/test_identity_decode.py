"""Unit tests for stage 3b -- identity idx_2d construction + view-cast.

Single concern: pin the ALG-5 behavioural contract by constructing
synthetic ``Stage2Batch`` fixtures (one section, one variant, one or two
call_targets) with controlled identity-carrier payload widths and
asserting the produced ``identity_idx_2d`` rows + ``identity_slices``
match the plan-documented mapping byte-for-byte.

The fixtures here build :class:`InlineDecodeState` directly from raw
streams (mirroring ``test_expand_tokens.py``'s pattern) and feed
:func:`expand_tokens` (the real stage-2a implementation) so the
expanded ids, surviving counts, and identity-carrier-vs-prepend split
exactly match what 3b's algorithm relies on.

Stage 3a is not yet implemented in this worktree, so the tests build
``inline_bytes`` + ``inline_byte_slices`` by hand from each fixture's
raw stream. The 3a contract is: ``inline_bytes[0]`` is the leading zero
pad, then each call_target's surviving inline bytes (from
``raw_tokens[number_mask]`` truncated to the surviving prefix) are
concatenated in DFS encounter order; ``inline_byte_slices[i]`` is the
range each call_target occupies.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._expand_tokens import (
    _V2_IDENTITY_BLOCK_START,
    _V2_NUMBER_BLOCK_START,
    _V2_RESERVED_DIGIT_COUNT,
    expand_tokens,
)
from tokenizer.aligned_data.loader.batch_decode._identity_decode import (
    build_identity_idx_2d,
    view_cast_identities,
)
from tokenizer.aligned_data.loader.batch_decode._surviving_counts import (
    count_surviving,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    Stage1Batch,
    Stage1CallTarget,
    Stage1Section,
    Stage1Variant,
    Stage2Batch,
    Stage2CallTarget,
    Stage2Section,
    Stage2Variant,
)
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    InlineDecodeState,
)
from tokenizer.aligned_data.loader.decoded.run_lengths import run_lengths
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import Section
from tokenizer.tokens import Category


# ---------------------------------------------------------------------------
# Vocab anchor constants -- re-derive locally so the tests document the
# concrete numeric ids they construct streams against.
# ---------------------------------------------------------------------------

_BLOCK_V2_ID = _V2_IDENTITY_BLOCK_START + 0  # 264
_LOCAL_FUNC_ID = _V2_IDENTITY_BLOCK_START + 1  # 265
_PLT_FUNC_ID = _V2_IDENTITY_BLOCK_START + 2  # 266
_STRING_PTR_ID = _V2_IDENTITY_BLOCK_START + 4  # 268


# ---------------------------------------------------------------------------
# Fixture builders -- minimal typed dummies for unused fields.
# ---------------------------------------------------------------------------


def _empty_function_data() -> FunctionData:
    return FunctionData(
        func_name="dummy",
        metadata={"arch": "x86_64", "compiler": "gcc", "opt": "O2"},
        tokens=np.zeros(0, dtype=np.uint16),
        insn_runlength=np.zeros(0, dtype=np.uint32),
        block_runlength=np.zeros(0, dtype=np.uint32),
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )


def _make_section() -> Section:
    return Section(
        function_name_ptr=42,
        section_offset=0,
        call_targets=[],
        variants=[],
    )


def _build_state(raw_tokens: np.ndarray) -> InlineDecodeState:
    """Construct an :class:`InlineDecodeState` from a raw stream.

    Computes the run-length + mask arrays exactly the same way the real
    :func:`build_inline_decode_state` would so 3b's algorithm receives
    realistic inputs.
    """
    real_mask = raw_tokens > _V2_RESERVED_DIGIT_COUNT
    number_mask = raw_tokens < _V2_RESERVED_DIGIT_COUNT
    if raw_tokens.shape[0] == 0:
        runlen_number = np.zeros(0, dtype=np.uint16)
        runlen_value = np.zeros(0, dtype=np.uint16)
    else:
        runlen_number = run_lengths(number_mask)
        runlen_value = run_lengths(~real_mask)
    # carries_inline_mask covers the NUMBER + IDENTITY bands (the
    # whole carrier band; vocab IDs in [257, 272)).
    carries_inline_mask = real_mask & (raw_tokens < _V2_IDENTITY_BLOCK_START + 8)
    is_negative_per_position = np.zeros(raw_tokens.shape[0], dtype=bool)
    digit_cumsum = np.zeros(raw_tokens.shape[0] + 1, dtype=np.uint32)
    if raw_tokens.shape[0] > 0:
        np.cumsum(number_mask.view(np.uint8), out=digit_cumsum[1:])
    return InlineDecodeState(
        raw_tokens=raw_tokens,
        real_mask=real_mask,
        number_mask=number_mask,
        runlen_number=runlen_number,
        runlen_value=runlen_value,
        carries_inline_mask=carries_inline_mask,
        is_negative_per_position=is_negative_per_position,
        digit_cumsum=digit_cumsum,
    )


def _make_stage1_ct(
    raw_tokens: np.ndarray,
    *,
    encounter_category: Category = Category.LOCAL_FUNC,
) -> Stage1CallTarget:
    return Stage1CallTarget(
        function_data=_empty_function_data(),
        state=_build_state(raw_tokens),
        call_targets_section=[],
        encounter_category=encounter_category,
        parent_call_target_index=None,
        function_name_ptr=0,
    )


def _make_stage2_ct(
    stage1_ct: Stage1CallTarget,
    *,
    cut_length: int | None = None,
) -> Stage2CallTarget:
    """Run the real stage-2a expand_tokens + surviving-counts to build a
    Stage2CallTarget that matches what 3b expects.

    ``cut_length`` clamps ``surviving_token_count`` for testing the
    cut-call_target path. When ``None``, no cut (fully included).
    """
    expanded = expand_tokens(stage1_ct)
    full_len = expanded.predicted_full_length
    if cut_length is None:
        surviving_token_count = full_len
        is_cut = False
    else:
        assert 0 <= cut_length <= full_len
        surviving_token_count = cut_length
        is_cut = cut_length < full_len
    counts = count_surviving(
        expanded.expanded_token_ids, surviving_token_count
    )
    return Stage2CallTarget(
        stage1=stage1_ct,
        expanded_token_ids=expanded.expanded_token_ids,
        extra_value_v2_mask=expanded.extra_value_v2_mask,
        extra_f128_mask=expanded.extra_f128_mask,
        predicted_full_length=full_len,
        surviving_token_count=surviving_token_count,
        surviving_identity_count=counts.surviving_identity_count,
        surviving_number_chunk_count=counts.surviving_number_chunk_count,
        is_cut=is_cut,
        partial_cut_length=surviving_token_count,
    )


def _wrap_stage2_batch(
    stage2_call_targets: list[Stage2CallTarget],
) -> Stage2Batch:
    """Wrap a flat list of Stage2CallTargets in a single-section/single-
    variant Stage2Batch. The order matches DFS encounter order."""
    # Synthesize the matching Stage1 hierarchy.
    stage1_cts = [ct.stage1 for ct in stage2_call_targets]
    stage1_variant = Stage1Variant(
        variant_idx=0,
        variant_ref_offset=0,
        batch_idx=0,
        call_targets=stage1_cts,
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )
    stage1_section = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=0,
        section=_make_section(),
        variants=[stage1_variant],
    )
    stage1_batch = Stage1Batch(
        sections=[stage1_section],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        batch_size=1,
    )

    stage2_variant = Stage2Variant(
        stage1=stage1_variant,
        call_targets=stage2_call_targets,
        cut_call_target_index=len(stage2_call_targets),
        total_surviving_token_count=sum(
            ct.surviving_token_count for ct in stage2_call_targets
        ),
        total_surviving_identity_count=sum(
            ct.surviving_identity_count for ct in stage2_call_targets
        ),
        total_surviving_number_chunk_count=sum(
            ct.surviving_number_chunk_count for ct in stage2_call_targets
        ),
    )
    stage2_section = Stage2Section(
        stage1=stage1_section,
        variants=[stage2_variant],
    )
    return Stage2Batch(
        stage1=stage1_batch,
        sections=[stage2_section],
        identity_row_offsets=np.zeros(2, dtype=np.uint32),
        number_row_offsets=np.zeros(2, dtype=np.uint32),
    )


def _build_inline_bytes_3a_oracle(
    stage2_call_targets: list[Stage2CallTarget],
) -> tuple[np.ndarray, list[slice]]:
    """Manual ALG-1 reference: concatenate per-call-target surviving
    inline bytes with a leading zero pad.

    For the cut call_target, the "surviving inline bytes" are the
    number_mask=True bytes at raw positions strictly before the cut
    boundary in the raw stream. We derive that boundary from
    :func:`expand_tokens`'s keep_mask + the per-call-target
    ``surviving_token_count``: the surviving raw range is the prefix
    that contains the first ``surviving_token_count - 1`` real tokens
    (the -1 strips the prepended self-token, which has NO raw-stream
    counterpart).
    """
    slices: list[slice] = []
    chunks: list[np.ndarray] = [np.zeros(1, dtype=np.uint8)]
    running = 1  # past the leading zero pad
    for ct in stage2_call_targets:
        state = ct.stage1.state
        raw = state.raw_tokens
        n = int(raw.shape[0])
        surv = ct.surviving_token_count
        if surv == 0:
            slices.append(slice(running, running))
            continue
        # surv includes the prepend (+1 over raw real tokens). The
        # remaining surv-1 real tokens of the function body come from
        # raw_tokens. The surviving raw byte range OWNS the inline-byte
        # payloads of each surviving real carrier, so raw_cut is the
        # position of the (raw_real_surv+1)-th raw real token (i.e.
        # the FIRST dropped real token) -- everything strictly before
        # that position survives. If no real tokens are dropped, the
        # whole raw stream survives (raw_cut = n).
        real_positions = np.nonzero(state.real_mask)[0]
        raw_real_surv = surv - 1  # prepend is not in raw
        if raw_real_surv >= real_positions.shape[0]:
            # Fully included function.
            raw_cut = n
        else:
            # raw_cut = first dropped real-token raw position.
            raw_cut = int(real_positions[raw_real_surv])
        surviving_bytes = raw[: raw_cut][state.number_mask[:raw_cut]].astype(
            np.uint8
        )
        K = int(surviving_bytes.shape[0])
        slices.append(slice(running, running + K))
        chunks.append(surviving_bytes)
        running += K
    inline_bytes = np.concatenate(chunks)
    return inline_bytes, slices


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_zero_identity_tokens_produces_empty_idx_2d() -> None:
    """A stream with no identity-band carriers yields an empty (0, 2)
    idx_2d, an empty u16 view-cast, and a single per-call-target slice
    of length 1 (just the prepend slot)."""
    # Pure VC2 stream with a small payload -- no identity carriers.
    raw = np.array(
        [
            _V2_NUMBER_BLOCK_START,  # VC2
            5,
            7,
        ],
        dtype=np.uint16,
    )
    s1 = _make_stage1_ct(raw)
    s2 = _make_stage2_ct(s1)
    batch = _wrap_stage2_batch([s2])
    inline_bytes, slices = _build_inline_bytes_3a_oracle([s2])
    idx_2d, identity_slices = build_identity_idx_2d(
        batch, inline_bytes, slices
    )
    assert idx_2d.shape == (0, 2)
    assert idx_2d.dtype == np.uint32
    # The call_target survived (prepend + VC2 + promoted continuations)
    # so identity_slice has length 1 (the prepend only).
    assert len(identity_slices) == 1
    assert identity_slices[0].stop - identity_slices[0].start == 1
    u16_view = view_cast_identities(idx_2d, inline_bytes)
    assert u16_view.shape == (0,)
    assert u16_view.dtype.kind == "u"
    assert u16_view.dtype.itemsize == 2


def test_single_2byte_identity_carrier() -> None:
    """A single identity carrier with a 2-byte payload yields one row
    of ``[hi, hi+1]`` whose view-cast equals the big-endian u16 of the
    payload."""
    # raw = [STRING_PTR, 0xAB, 0xCD]
    raw = np.array(
        [_STRING_PTR_ID, 0xAB, 0xCD],
        dtype=np.uint16,
    )
    s1 = _make_stage1_ct(raw)
    s2 = _make_stage2_ct(s1)
    batch = _wrap_stage2_batch([s2])
    inline_bytes, slices = _build_inline_bytes_3a_oracle([s2])
    idx_2d, identity_slices = build_identity_idx_2d(
        batch, inline_bytes, slices
    )

    # Expected inline_bytes layout: [0 (pad), 0xAB, 0xCD]
    assert list(inline_bytes) == [0, 0xAB, 0xCD]

    # One in-stream identity carrier -> one idx_2d row pointing at
    # inline_bytes[1] (high byte = 0xAB) and inline_bytes[2] (low
    # byte = 0xCD).
    assert idx_2d.shape == (1, 2)
    assert idx_2d[0, 0] == 1
    assert idx_2d[0, 1] == 2

    u16_view = view_cast_identities(idx_2d, inline_bytes)
    assert u16_view.shape == (1,)
    # Big-endian u16: 0xAB * 256 + 0xCD = 0xABCD.
    assert int(u16_view[0]) == 0xABCD

    # identity_slice covers prepend + this carrier (length 2).
    assert identity_slices[0].stop - identity_slices[0].start == 2


def test_single_1byte_identity_carrier() -> None:
    """A 1-byte payload yields ``[0, lo_offset]`` -- the leading zero
    pad supplies the high byte and the view-cast u16 equals the payload
    byte itself."""
    raw = np.array(
        [_BLOCK_V2_ID, 0x42],
        dtype=np.uint16,
    )
    s1 = _make_stage1_ct(raw)
    s2 = _make_stage2_ct(s1)
    batch = _wrap_stage2_batch([s2])
    inline_bytes, slices = _build_inline_bytes_3a_oracle([s2])
    idx_2d, identity_slices = build_identity_idx_2d(
        batch, inline_bytes, slices
    )

    assert list(inline_bytes) == [0, 0x42]
    assert idx_2d.shape == (1, 2)
    assert idx_2d[0, 0] == 0
    assert idx_2d[0, 1] == 1

    u16_view = view_cast_identities(idx_2d, inline_bytes)
    assert int(u16_view[0]) == 0x42


def test_single_0byte_identity_carrier() -> None:
    """A 0-byte payload (carrier with no following inline-digit run)
    yields ``[0, 0]`` -- the view-cast reads as the big-endian u16
    formed from the leading zero pad twice = 0. The encoder reserves
    caller-local id 0 for this case."""
    # Identity carrier as the ONLY token -- no payload follows.
    raw = np.array([_BLOCK_V2_ID], dtype=np.uint16)
    s1 = _make_stage1_ct(raw)
    s2 = _make_stage2_ct(s1)
    batch = _wrap_stage2_batch([s2])
    inline_bytes, slices = _build_inline_bytes_3a_oracle([s2])
    idx_2d, identity_slices = build_identity_idx_2d(
        batch, inline_bytes, slices
    )

    # inline_bytes = [0] (just the pad).
    assert inline_bytes.shape == (1,)
    assert idx_2d.shape == (1, 2)
    assert idx_2d[0, 0] == 0
    assert idx_2d[0, 1] == 0

    u16_view = view_cast_identities(idx_2d, inline_bytes)
    assert int(u16_view[0]) == 0


def test_mixed_payload_widths_in_single_function() -> None:
    """Three identity carriers in one function -- 2-byte, 1-byte, then
    0-byte (at the tail) -- map to the three plan-documented patterns."""
    # raw = [BLOCK_V2, 0x12, 0x34, LOCAL_FUNC, 0x77, STRING_PTR]
    # carrier 0 -> 2-byte payload (0x1234)
    # carrier 1 -> 1-byte payload (0x77)
    # carrier 2 -> 0-byte payload (tail)
    raw = np.array(
        [_BLOCK_V2_ID, 0x12, 0x34, _LOCAL_FUNC_ID, 0x77, _STRING_PTR_ID],
        dtype=np.uint16,
    )
    s1 = _make_stage1_ct(raw)
    s2 = _make_stage2_ct(s1)
    batch = _wrap_stage2_batch([s2])
    inline_bytes, slices = _build_inline_bytes_3a_oracle([s2])
    idx_2d, identity_slices = build_identity_idx_2d(
        batch, inline_bytes, slices
    )

    # inline_bytes = [0 (pad), 0x12, 0x34, 0x77]
    assert list(inline_bytes) == [0, 0x12, 0x34, 0x77]

    # idx_2d rows:
    #   row 0 -> [1, 2]  (BLOCK_V2 / 2-byte at offset 1)
    #   row 1 -> [0, 3]  (LOCAL_FUNC / 1-byte at offset 3)
    #   row 2 -> [0, 0]  (STRING_PTR / 0-byte tail)
    assert idx_2d.shape == (3, 2)
    np.testing.assert_array_equal(
        idx_2d,
        np.array(
            [[1, 2], [0, 3], [0, 0]],
            dtype=np.uint32,
        ),
    )

    u16_view = view_cast_identities(idx_2d, inline_bytes)
    assert list(int(v) for v in u16_view) == [0x1234, 0x77, 0]

    # identity_slice covers prepend + 3 carriers = length 4.
    assert identity_slices[0].stop - identity_slices[0].start == 4


def test_multiple_call_targets_slice_continuity() -> None:
    """Two call_targets in DFS order: identity_slices contiguous (no
    overlap, no gap) and idx_2d rows reference the correct call_target's
    byte slice."""
    # Function A: one 2-byte identity -> 2 inline bytes.
    raw_a = np.array(
        [_BLOCK_V2_ID, 0x01, 0x02], dtype=np.uint16
    )
    # Function B: one 1-byte identity -> 1 inline byte.
    raw_b = np.array(
        [_BLOCK_V2_ID, 0x03], dtype=np.uint16
    )

    s1a = _make_stage1_ct(raw_a)
    s1b = _make_stage1_ct(raw_b)
    s2a = _make_stage2_ct(s1a)
    s2b = _make_stage2_ct(s1b)
    batch = _wrap_stage2_batch([s2a, s2b])
    inline_bytes, slices = _build_inline_bytes_3a_oracle([s2a, s2b])

    # inline_bytes = [0 (pad), 0x01, 0x02 (A's payload), 0x03 (B's
    # payload)].
    assert list(inline_bytes) == [0, 0x01, 0x02, 0x03]
    assert slices[0] == slice(1, 3)
    assert slices[1] == slice(3, 4)

    idx_2d, identity_slices = build_identity_idx_2d(
        batch, inline_bytes, slices
    )

    # Row 0 -> A's carrier (offsets 1, 2)
    # Row 1 -> B's carrier (offsets 0, 3)
    assert idx_2d.shape == (2, 2)
    np.testing.assert_array_equal(
        idx_2d,
        np.array([[1, 2], [0, 3]], dtype=np.uint32),
    )

    u16_view = view_cast_identities(idx_2d, inline_bytes)
    assert int(u16_view[0]) == 0x0102
    assert int(u16_view[1]) == 0x0003

    # Per-call-target slices are contiguous with no overlap.
    assert identity_slices[0] == slice(0, 2)  # prepend + 1 in-stream
    assert identity_slices[1] == slice(2, 4)  # prepend + 1 in-stream
    # The next slice's start equals the previous slice's stop.
    assert identity_slices[1].start == identity_slices[0].stop


def test_identity_slice_includes_prepend_reservation() -> None:
    """Per the API contract ``identity_slice`` length =
    ``1 + in_stream_count``. We verify this for a call_target with K=3
    in-stream identity carriers."""
    raw = np.array(
        [
            _BLOCK_V2_ID,  # 2-byte
            0x10,
            0x20,
            _STRING_PTR_ID,  # 2-byte
            0x30,
            0x40,
            _LOCAL_FUNC_ID,  # 1-byte
            0x50,
        ],
        dtype=np.uint16,
    )
    s1 = _make_stage1_ct(raw)
    s2 = _make_stage2_ct(s1)
    batch = _wrap_stage2_batch([s2])
    inline_bytes, slices = _build_inline_bytes_3a_oracle([s2])
    idx_2d, identity_slices = build_identity_idx_2d(
        batch, inline_bytes, slices
    )
    # 3 in-stream + 1 prepend = 4.
    assert identity_slices[0].stop - identity_slices[0].start == 4
    # idx_2d only has the 3 in-stream rows (no prepend).
    assert idx_2d.shape == (3, 2)


def test_cut_call_target_drops_post_cut_identities() -> None:
    """When a call_target is cut, only the identity carriers in the
    surviving expanded prefix contribute idx_2d rows -- post-cut
    identities are excluded."""
    # 3 identity carriers; cut so only the first survives.
    raw = np.array(
        [
            _BLOCK_V2_ID,  # carrier 0 -- 2-byte
            0x11,
            0x22,
            _BLOCK_V2_ID,  # carrier 1 -- 2-byte (will be cut off)
            0x33,
            0x44,
            _BLOCK_V2_ID,  # carrier 2 -- 1-byte (will be cut off)
            0x55,
        ],
        dtype=np.uint16,
    )
    s1 = _make_stage1_ct(raw)
    # expand_tokens yields: [prepend, c0, c1, c2] (length 4).
    # We cut at length 2 -> only [prepend, c0] survive.
    s2 = _make_stage2_ct(s1, cut_length=2)
    assert s2.surviving_identity_count == 2  # prepend + c0
    batch = _wrap_stage2_batch([s2])
    inline_bytes, slices = _build_inline_bytes_3a_oracle([s2])

    # Surviving inline bytes from the cut function = 0x11, 0x22 (c0's
    # payload only). 3a's slice spans those 2 bytes.
    assert list(inline_bytes) == [0, 0x11, 0x22]
    assert slices[0] == slice(1, 3)

    idx_2d, identity_slices = build_identity_idx_2d(
        batch, inline_bytes, slices
    )
    # Only ONE in-stream identity (c0) -> 1 idx_2d row.
    assert idx_2d.shape == (1, 2)
    np.testing.assert_array_equal(
        idx_2d, np.array([[1, 2]], dtype=np.uint32)
    )
    # identity_slice: prepend + c0 = length 2.
    assert identity_slices[0] == slice(0, 2)
    u16_view = view_cast_identities(idx_2d, inline_bytes)
    assert int(u16_view[0]) == 0x1122


def test_dtype_check_u32_idx_2d_u16_view_cast() -> None:
    """idx_2d must be u32; view-cast must produce u16 (big-endian
    byte order)."""
    raw = np.array(
        [_BLOCK_V2_ID, 0x12, 0x34], dtype=np.uint16
    )
    s1 = _make_stage1_ct(raw)
    s2 = _make_stage2_ct(s1)
    batch = _wrap_stage2_batch([s2])
    inline_bytes, slices = _build_inline_bytes_3a_oracle([s2])
    idx_2d, _identity_slices = build_identity_idx_2d(
        batch, inline_bytes, slices
    )
    assert idx_2d.dtype == np.uint32
    u16_view = view_cast_identities(idx_2d, inline_bytes)
    # itemsize 2 + unsigned kind == u16 regardless of endianness label.
    assert u16_view.dtype.itemsize == 2
    assert u16_view.dtype.kind == "u"


def test_view_cast_endianness_is_big_endian() -> None:
    """The gather-then-view-cast step uses ``>u2`` so the first byte
    of each idx_2d row is the HIGH byte of the resulting u16."""
    raw = np.array(
        [_BLOCK_V2_ID, 0xAA, 0xBB], dtype=np.uint16
    )
    s1 = _make_stage1_ct(raw)
    s2 = _make_stage2_ct(s1)
    batch = _wrap_stage2_batch([s2])
    inline_bytes, slices = _build_inline_bytes_3a_oracle([s2])
    idx_2d, _identity_slices = build_identity_idx_2d(
        batch, inline_bytes, slices
    )
    u16_view = view_cast_identities(idx_2d, inline_bytes)
    # 0xAABB -- if the cast were little-endian we'd see 0xBBAA.
    assert int(u16_view[0]) == 0xAABB
    assert int(u16_view[0]) != 0xBBAA


def test_fully_dropped_call_target_empty_identity_slice() -> None:
    """A call_target with ``surviving_token_count == 0`` produces an
    empty identity_slice and contributes no idx_2d rows."""
    # 2 functions: A fully included, B fully dropped via cut_length=0.
    raw_a = np.array(
        [_BLOCK_V2_ID, 0x01, 0x02], dtype=np.uint16
    )
    raw_b = np.array(
        [_BLOCK_V2_ID, 0x03], dtype=np.uint16
    )
    s1a = _make_stage1_ct(raw_a)
    s1b = _make_stage1_ct(raw_b)
    s2a = _make_stage2_ct(s1a)
    s2b = _make_stage2_ct(s1b, cut_length=0)
    assert s2b.surviving_token_count == 0
    assert s2b.surviving_identity_count == 0
    batch = _wrap_stage2_batch([s2a, s2b])
    inline_bytes, slices = _build_inline_bytes_3a_oracle([s2a, s2b])

    # B's slice has length 0; only A contributes inline bytes.
    assert slices[1].stop - slices[1].start == 0

    idx_2d, identity_slices = build_identity_idx_2d(
        batch, inline_bytes, slices
    )
    # Only A's carrier contributes.
    assert idx_2d.shape == (1, 2)
    # A's identity_slice = prepend + 1 in-stream = 2.
    # B's identity_slice = 0 (no prepend either).
    assert identity_slices[0] == slice(0, 2)
    assert identity_slices[1] == slice(2, 2)


def test_plt_func_encounter_category_handled() -> None:
    """The encounter_category affects the prepend's expanded token id
    but not the identity_idx_2d shape -- only the per-call-target
    in-stream identity carriers contribute rows."""
    raw = np.array(
        [_BLOCK_V2_ID, 0x99], dtype=np.uint16
    )
    s1 = _make_stage1_ct(raw, encounter_category=Category.PLT_FUNC)
    s2 = _make_stage2_ct(s1)
    batch = _wrap_stage2_batch([s2])
    inline_bytes, slices = _build_inline_bytes_3a_oracle([s2])
    idx_2d, identity_slices = build_identity_idx_2d(
        batch, inline_bytes, slices
    )
    # Same shape regardless of PLT vs LOCAL prepend: 1 in-stream
    # carrier + prepend reservation.
    assert idx_2d.shape == (1, 2)
    assert identity_slices[0].stop - identity_slices[0].start == 2
    u16_view = view_cast_identities(idx_2d, inline_bytes)
    assert int(u16_view[0]) == 0x99


def test_zero_in_stream_but_surviving_call_target() -> None:
    """A call_target whose only surviving token is the prepend
    contributes 0 idx_2d rows but reserves a length-1 identity_slice
    for stage 4's prepend write."""
    # All identity carriers cut off; only prepend remains.
    raw = np.array(
        [_BLOCK_V2_ID, 0xAA], dtype=np.uint16
    )
    s1 = _make_stage1_ct(raw)
    # expand_tokens yields [prepend, BLOCK_V2_shifted] (length 2).
    # Cut at 1 -> only prepend survives.
    s2 = _make_stage2_ct(s1, cut_length=1)
    assert s2.surviving_identity_count == 1
    batch = _wrap_stage2_batch([s2])
    inline_bytes, slices = _build_inline_bytes_3a_oracle([s2])
    idx_2d, identity_slices = build_identity_idx_2d(
        batch, inline_bytes, slices
    )
    assert idx_2d.shape == (0, 2)
    # Slice length 1 (just the prepend reservation).
    assert identity_slices[0] == slice(0, 1)
