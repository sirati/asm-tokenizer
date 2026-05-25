"""Tests for the metatoken-runlength helper + the
``emit_block_n_insns_runlength`` batch-decode flag.

Pins:

* :func:`compute_metatoken_runlengths` produces the FTL-equivalent
  ``(block_runlength, insn_runlength)`` for a synthetic FunctionData,
  with post-promotion slot scaling for F128 finite / NaN-Inf and
  multi-chunk VC2.
* :func:`batch_decode` emits matching sidecars when the flag is True;
  the per-row offsets cumsum total equals the per-row sum of the flat
  arrays.
* The public alias :data:`manual_calc_block_n_insn_runlengths` returns
  identical output (it is the same function rebound under the user-
  asked name).
* Per-instruction slot counts agree with :func:`expand_tokens`' wire
  output: for every instruction ``i``, ``insn_runlength[i]`` equals the
  number of surviving (post-strip) wire tokens that lie in instruction
  ``i``'s raw range. This cross-checks the two independent slot-count
  computations (helper-side ``_per_metatoken_slot_counts`` + folded via
  ``CA_BArle_to_CBrle`` vs ``expand_tokens``' explicit strip + paint).
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode import (
    SectionPointerSpec,
    VariantPadding,
    batch_decode,
)
from tokenizer.aligned_data.loader.batch_decode._expand_tokens import (
    expand_tokens,
)
from tokenizer.aligned_data.loader.batch_decode._runlengths import (
    compute_metatoken_runlengths,
    manual_calc_block_n_insn_runlengths,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    Stage1CallTarget,
)
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    build_inline_decode_state,
)
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.loader.tests._session_fixture import (
    build_synthetic_binary,
)
from tokenizer.tokens import Category


# Vocab anchors mirror :mod:`._runlengths` so the synthetic streams
# below remain readable.
_RESERVED = 256
_VALUE_NEGATIVE = 256  # protocol-pinned: postfix sign marker at id 256.
_VC2 = 257
_F128 = 263  # last NUMBER carrier: VC2(257) + 6 = F128(263)
_BLOCK_V2 = 264
_LOCAL_FUNC = 265
_FIRST_INSTR_REP = 272  # _V2_IDENTITY_BLOCK_START + 8


def _make_function_data(
    *,
    tokens: np.ndarray,
    insn_runlength: np.ndarray,
    block_runlength: np.ndarray,
) -> FunctionData:
    """Bare-minimum :class:`FunctionData` with the three arrays the
    runlength helper reads. Other fields are inert (an empty metadata
    dict + an empty variant_tokens array)."""
    return FunctionData(
        func_name="synthetic",
        metadata={},
        tokens=tokens.astype(np.uint16),
        insn_runlength=insn_runlength.astype(np.uint32),
        block_runlength=block_runlength.astype(np.uint32),
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )


# ---------------------------------------------------------------------------
# Ordinary metatokens: 1 slot per instruction's metatoken.
# ---------------------------------------------------------------------------


def test_ordinary_single_block_single_insn_single_metatoken() -> None:
    """One block, one instruction, one metatoken (1 raw token, an
    instruction-rep carrier). Expected (block_rl=[1], insn_rl=[1])."""
    fd = _make_function_data(
        tokens=np.array([_FIRST_INSTR_REP], dtype=np.uint16),
        insn_runlength=np.array([1], dtype=np.uint32),
        block_runlength=np.array([1], dtype=np.uint32),
    )
    block_rl, insn_rl = compute_metatoken_runlengths(fd)
    assert block_rl.tolist() == [1]
    assert insn_rl.tolist() == [1]
    # Public alias produces identical output.
    block_rl_alias, insn_rl_alias = manual_calc_block_n_insn_runlengths(fd)
    assert np.array_equal(block_rl, block_rl_alias)
    assert np.array_equal(insn_rl, insn_rl_alias)


def test_ordinary_two_blocks_three_insns_each_one_metatoken() -> None:
    """Two blocks of 3 instructions of 1 metatoken each. Expected
    block_rl=[3, 3], insn_rl=[1, 1, 1, 1, 1, 1].
    """
    fd = _make_function_data(
        tokens=np.full(6, _FIRST_INSTR_REP, dtype=np.uint16),
        insn_runlength=np.ones(6, dtype=np.uint32),
        block_runlength=np.array([3, 3], dtype=np.uint32),
    )
    block_rl, insn_rl = compute_metatoken_runlengths(fd)
    assert block_rl.tolist() == [3, 3]
    assert insn_rl.tolist() == [1, 1, 1, 1, 1, 1]


def test_instruction_with_two_metatokens_two_slots() -> None:
    """One block of one instruction with TWO metatokens (each a 1-token
    instruction-rep carrier). Insn slot count = 2.
    """
    fd = _make_function_data(
        tokens=np.array(
            [_FIRST_INSTR_REP, _FIRST_INSTR_REP + 1], dtype=np.uint16,
        ),
        insn_runlength=np.array([2], dtype=np.uint32),
        block_runlength=np.array([2], dtype=np.uint32),
    )
    block_rl, insn_rl = compute_metatoken_runlengths(fd)
    assert block_rl.tolist() == [1]
    assert insn_rl.tolist() == [2]


# ---------------------------------------------------------------------------
# Multi-chunk promotion: F128 finite (2 slots) + NaN/Inf (1 slot).
# ---------------------------------------------------------------------------


def test_f128_finite_bumps_insn_to_two_slots() -> None:
    """An instruction containing one F128 metatoken with finite payload
    (high u16 != all-ones) renders as 2 post-decode slots. F128 payload
    is 16 inline-digit bytes; high u16 from bytes (p+1, p+2).

    Finite payload: bytes (p+1, p+2) = (0x00, 0x00) -> high u16 = 0
    -> finite -> 2 slots.
    """
    payload = np.zeros(16, dtype=np.uint16)  # all-zero finite
    tokens = np.concatenate(
        [np.array([_F128], dtype=np.uint16), payload],
    )
    # 17 raw tokens for the F128 instruction (1 carrier + 16 digits).
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([17], dtype=np.uint32),
        block_runlength=np.array([17], dtype=np.uint32),
    )
    block_rl, insn_rl = compute_metatoken_runlengths(fd)
    assert block_rl.tolist() == [1]
    assert insn_rl.tolist() == [2]


def test_f128_nan_or_inf_stays_one_slot() -> None:
    """F128 NaN/Inf source: high u16 with the 15-bit exponent all ones
    after sign-strip; 1 slot.

    Construct (p+1, p+2) = (0x7F, 0xFF) -> high u16 = 0x7FFF -> NaN/Inf.
    """
    payload = np.zeros(16, dtype=np.uint16)
    payload[0] = 0x7F  # high byte sign-stripped
    payload[1] = 0xFF  # low byte
    tokens = np.concatenate(
        [np.array([_F128], dtype=np.uint16), payload],
    )
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([17], dtype=np.uint32),
        block_runlength=np.array([17], dtype=np.uint32),
    )
    block_rl, insn_rl = compute_metatoken_runlengths(fd)
    assert block_rl.tolist() == [1]
    assert insn_rl.tolist() == [1]


def test_vc2_two_chunks_bumps_insn_to_two_slots() -> None:
    """VC2 with 9-byte payload: chunk_count = ceil(9/8) = 2. Insn slot
    count = 2.
    """
    payload = np.zeros(9, dtype=np.uint16)
    tokens = np.concatenate(
        [np.array([_VC2], dtype=np.uint16), payload],
    )
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([10], dtype=np.uint32),
        block_runlength=np.array([10], dtype=np.uint32),
    )
    block_rl, insn_rl = compute_metatoken_runlengths(fd)
    assert block_rl.tolist() == [1]
    assert insn_rl.tolist() == [2]


def test_vc2_three_chunks_bumps_insn_to_three_slots() -> None:
    """VC2 with 17-byte payload: chunk_count = ceil(17/8) = 3. Insn
    slot count = 3.
    """
    payload = np.zeros(17, dtype=np.uint16)
    tokens = np.concatenate(
        [np.array([_VC2], dtype=np.uint16), payload],
    )
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([18], dtype=np.uint32),
        block_runlength=np.array([18], dtype=np.uint32),
    )
    block_rl, insn_rl = compute_metatoken_runlengths(fd)
    assert block_rl.tolist() == [1]
    assert insn_rl.tolist() == [3]


def test_vc2_single_byte_payload_one_chunk() -> None:
    """VC2 with 1-byte payload: chunk_count = max(1, ceil(1/8)) = 1.
    Insn slot count = 1 (default).
    """
    tokens = np.array([_VC2, 0x42], dtype=np.uint16)
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([2], dtype=np.uint32),
        block_runlength=np.array([2], dtype=np.uint32),
    )
    block_rl, insn_rl = compute_metatoken_runlengths(fd)
    assert block_rl.tolist() == [1]
    assert insn_rl.tolist() == [1]


# ---------------------------------------------------------------------------
# value_negative (id 256) postfix: counted as a metatoken upstream (FTL's
# ``>= 256`` boundary rule) but stripped post-decode by ``_expand_tokens``'
# ``> 256`` keep mask. Slot count contribution: 0.
# ---------------------------------------------------------------------------


def test_negative_vc2_two_chunks_does_not_count_value_negative() -> None:
    """Audit cluster #21 H-2 pinning case: 9-byte negative VC2 magnitude.

    Wire shape: ``[VC2, 9 payload bytes, value_negative]``. Post-decode
    wire slot count = 2 (VC2 produces ceil(9/8) = 2 chunks; the trailing
    value_negative marker is stripped). The metatoken-runlength helper
    must mirror that count exactly so downstream row-walkers can advance
    one slot per emitted wire token.
    """
    tokens = np.array(
        [_VC2] + [0x42] * 9 + [_VALUE_NEGATIVE], dtype=np.uint16,
    )
    n = int(tokens.size)
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([n], dtype=np.uint32),
        block_runlength=np.array([n], dtype=np.uint32),
    )
    block_rl, insn_rl = compute_metatoken_runlengths(fd)
    assert block_rl.tolist() == [1]
    assert insn_rl.tolist() == [2]


def test_negative_vc2_eight_byte_magnitude_stays_one_chunk() -> None:
    """Eight-byte negative VC2 magnitude (e.g., ``-(1 << 63)``):
    chunk_count = ceil(8/8) = 1, post-decode slot count = 1.

    Edge case: K*8-byte magnitudes are where the naive
    "drop value_negative from the boundary mask" alternative would
    silently inflate the VC2 chunk count (the trailing 256 would be
    absorbed into the VC2 metatoken's run-length, turning
    ``payload_runlen = 8`` into ``9`` and bumping
    ``ceil(payload_runlen / 8)`` from 1 to 2). The correct accountant
    subtracts the value_negative slot AFTER the run-length is computed,
    leaving the VC2 chunk-count derivation intact.
    """
    tokens = np.array(
        [_VC2] + [0x42] * 8 + [_VALUE_NEGATIVE], dtype=np.uint16,
    )
    n = int(tokens.size)
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([n], dtype=np.uint32),
        block_runlength=np.array([n], dtype=np.uint32),
    )
    block_rl, insn_rl = compute_metatoken_runlengths(fd)
    assert block_rl.tolist() == [1]
    assert insn_rl.tolist() == [1]


def test_negative_vc2_sixteen_byte_magnitude_two_chunks() -> None:
    """Sixteen-byte negative VC2 magnitude: chunk_count = ceil(16/8) = 2,
    post-decode slot count = 2. Second K*8-byte edge case (paired with
    the 8-byte test above) to keep both boundaries pinned.
    """
    tokens = np.array(
        [_VC2] + [0x42] * 16 + [_VALUE_NEGATIVE], dtype=np.uint16,
    )
    n = int(tokens.size)
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([n], dtype=np.uint32),
        block_runlength=np.array([n], dtype=np.uint32),
    )
    block_rl, insn_rl = compute_metatoken_runlengths(fd)
    assert block_rl.tolist() == [1]
    assert insn_rl.tolist() == [2]


def test_negative_vc2_single_byte_magnitude_one_chunk() -> None:
    """One-byte negative VC2 magnitude (e.g., ``-1``):
    chunk_count = max(1, ceil(1/8)) = 1, post-decode slot count = 1.
    """
    tokens = np.array(
        [_VC2, 0x42, _VALUE_NEGATIVE], dtype=np.uint16,
    )
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([3], dtype=np.uint32),
        block_runlength=np.array([3], dtype=np.uint32),
    )
    block_rl, insn_rl = compute_metatoken_runlengths(fd)
    assert block_rl.tolist() == [1]
    assert insn_rl.tolist() == [1]


# ---------------------------------------------------------------------------
# Empty / degenerate.
# ---------------------------------------------------------------------------


def test_empty_function_returns_empty_arrays() -> None:
    """An empty function (no tokens) returns ``(empty, empty)`` arrays."""
    fd = _make_function_data(
        tokens=np.zeros(0, dtype=np.uint16),
        insn_runlength=np.zeros(0, dtype=np.uint32),
        block_runlength=np.zeros(0, dtype=np.uint32),
    )
    block_rl, insn_rl = compute_metatoken_runlengths(fd)
    assert block_rl.tolist() == []
    assert insn_rl.tolist() == []
    assert block_rl.dtype == np.uint32
    assert insn_rl.dtype == np.uint32


# ---------------------------------------------------------------------------
# Per-instruction cross-check: helper output vs ``expand_tokens`` body.
#
# The helper's ``insn_runlength`` partitions the post-decode wire stream into
# per-instruction slot ranges. ``expand_tokens`` is the authoritative wire
# producer: its post-strip + post-paint length is the slot count downstream
# walkers see. The two MUST agree per-instruction so any walker that uses
# ``insn_runlength`` to drive instruction boundaries (e.g. the inspector's
# batch-decode row walker) lands on the SAME slot ranges ``expand_tokens``
# emitted. Any disagreement here means one of the two computations is
# wrong; the regression below pins the cross-check across the canonical
# multi-chunk patterns (1-chunk VC2 at end of instruction, multi-chunk VC2
# straddling no boundary, F128 finite + NaN/Inf, value_negative skip).
# ---------------------------------------------------------------------------


def _build_stage1(fd: FunctionData) -> Stage1CallTarget:
    """Wrap a :class:`FunctionData` in a minimal :class:`Stage1CallTarget`
    so :func:`expand_tokens` can run on it.

    Only the fields :func:`expand_tokens` reads (``state`` +
    ``encounter_category``) need to be populated; the rest stay inert.
    """
    state = build_inline_decode_state(fd.tokens, format_version=1)
    return Stage1CallTarget(
        function_data=fd,
        state=state,
        call_targets_section=[],
        encounter_category=Category.LOCAL_FUNC,
        parent_call_target_index=None,
        function_name_ptr=0,
    )


def _per_insn_surviving_count(
    raw_tokens: np.ndarray,
    insn_runlength: np.ndarray,
    expanded_token_ids: np.ndarray,
) -> np.ndarray:
    """Walk the expanded body slot-by-slot and split it at each
    instruction boundary the helper's ``insn_runlength`` declares.
    Returns the per-instruction slot count walked out of the expanded
    body -- which MUST equal ``insn_runlength`` itself when the helper
    + expand_tokens agree.

    Computed independently of the helper: we slice the raw stream at the
    raw-runlength-derived insn boundaries, count ``raw > 256`` survivors
    in each slice (mirrors ``_expand_tokens``' ``keep_mask = working >
    256`` -- but without VC2 / F128 paint, which doesn't shift the
    surviving-count INSIDE one instruction since paint slots are inline-
    digit positions that ``expand_tokens`` paints over THEN keeps).
    """
    # Raw insn boundary positions (exclusive end).
    raw_cum = np.cumsum(insn_runlength.astype(np.int64))
    n_insns = int(insn_runlength.size)
    out = np.empty(n_insns, dtype=np.int64)
    raw_start = 0
    for i in range(n_insns):
        raw_end = int(raw_cum[i])
        chunk = raw_tokens[raw_start:raw_end]
        # Count BOTH the carriers (raw > 256) AND the painted continuation
        # slots: a multi-chunk VC2 / F128 finite carrier paints its
        # continuation slots into ``working_tokens`` at positions p+1, ...
        # which then survive the ``working > 256`` keep mask. Those
        # painted positions sit in the raw range [p+1, p+K) which is
        # inline-digit positions BEFORE paint. So:
        #
        #   surviving_count = real_carriers + painted_continuations
        #                   = (raw > 256).sum() + paint_count_in_chunk
        #
        # The paint count comes from ``expand_tokens`` -- we recompute it
        # by counting per-VC2 ``max(0, K_full - 1)`` + per-finite-F128
        # ``1`` (since F128 paints exactly one continuation slot).
        real_count = int(np.sum(chunk > 256))
        paint_count = 0
        # VC2 paint: each carrier paints ``K_full - 1`` continuation
        # slots, where K_full = max(1, ceil(L/8)) and L = inline-digit
        # run-length immediately after the carrier.
        vc2_in_chunk = np.where(chunk == 257)[0]
        for offset in vc2_in_chunk:
            p_raw = raw_start + int(offset)
            # Count consecutive inline-digits after the carrier within
            # the FULL raw stream (the paint spans inline-digit positions
            # that may extend beyond the current instruction; for the
            # per-instruction count we only care about paints within
            # [raw_start, raw_end), which are bounded by ``raw_end``).
            j = p_raw + 1
            while j < raw_tokens.size and raw_tokens[j] < 256:
                j += 1
            L = j - p_raw - 1
            K_full = max(1, (L + 7) // 8)
            # Paint slots: ``K_full - 1`` continuations starting at
            # ``p_raw + 1``. Count those that fall inside the current
            # instruction's raw range.
            paint_in_insn = max(
                0,
                min(p_raw + K_full, raw_end) - (p_raw + 1),
            )
            paint_count += paint_in_insn
        # F128 paint: each finite F128 carrier paints exactly 1
        # continuation slot at p_raw + 1.
        f128_in_chunk = np.where(chunk == 263)[0]
        for offset in f128_in_chunk:
            p_raw = raw_start + int(offset)
            # NaN/Inf detection: high u16 = (raw[p+1] << 8) | raw[p+2].
            # NaN/Inf when the 15-bit exponent (sign-stripped) is all
            # ones -> 1-chunk source, no paint.
            if p_raw + 2 < raw_tokens.size:
                high_u16 = (int(raw_tokens[p_raw + 1]) << 8) | int(
                    raw_tokens[p_raw + 2]
                )
                is_nan_inf = (high_u16 & 0x7FFF) == 0x7FFF
                if not is_nan_inf:
                    # Finite: paint p+1 if it's within the instruction.
                    if p_raw + 1 < raw_end:
                        paint_count += 1
        out[i] = real_count + paint_count
        raw_start = raw_end
    return out


def _assert_helper_matches_expand_tokens(fd: FunctionData) -> None:
    """Pin the per-instruction agreement between
    :func:`compute_metatoken_runlengths` and :func:`expand_tokens`.

    Two checks:

    1. Total slot count agrees: ``insn_runlength.sum()`` ==
       ``expand_tokens body length`` (= ``expanded.size - 1`` after
       stripping the self-prepend slot).
    2. Per-instruction slot count agrees: for every ``i``,
       ``insn_runlength[i]`` equals the count of surviving wire slots
       (real carriers + painted continuations) that lie in instruction
       ``i``'s raw range.
    """
    _, insn_rl = compute_metatoken_runlengths(fd)
    s1 = _build_stage1(fd)
    expanded = expand_tokens(s1)
    body_len = int(expanded.expanded_token_ids.size) - 1  # self-prepend
    assert int(insn_rl.sum()) == body_len, (
        f"helper sum {int(insn_rl.sum())} != expand_tokens body len {body_len}"
    )
    per_insn_wire = _per_insn_surviving_count(
        fd.tokens, fd.insn_runlength, expanded.expanded_token_ids
    )
    assert np.array_equal(insn_rl.astype(np.int64), per_insn_wire), (
        f"per-instruction mismatch:\n"
        f"  helper: {insn_rl.tolist()}\n"
        f"  wire:   {per_insn_wire.tolist()}"
    )


def test_vc2_at_end_of_instruction_helper_matches_wire() -> None:
    """Cluster-Calloc regression: an ARM-style ``ADD r11, sp, #4``
    instruction renders as ``[INSTR_REP, INSTR_REP, INSTR_REP, VC2,
    inline_4]`` in raw space. The VC2 carrier is the LAST metatoken,
    its 1-byte payload is the LAST raw token. Helper's
    ``insn_runlength`` for this instruction must equal the
    ``expand_tokens`` wire-slot count -- 4 slots (3 carriers + the VC2
    carrier; the inline-digit byte is stripped). Pins the per-
    instruction match across the multi-chunk-capable shifted-id end-
    of-instruction case the inspector's accumulator-drain trips on
    (the dataloader's count is correct; downstream consumers MUST land
    on the same slot ranges).
    """
    tokens = np.array(
        [_FIRST_INSTR_REP, _FIRST_INSTR_REP + 1, _FIRST_INSTR_REP + 2,
         _VC2, 0x04],
        dtype=np.uint16,
    )
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([5], dtype=np.uint32),
        block_runlength=np.array([5], dtype=np.uint32),
    )
    _assert_helper_matches_expand_tokens(fd)
    # Explicit slot count: 3 INSTR_REP + 1 VC2 carrier = 4 slots.
    _, insn_rl = compute_metatoken_runlengths(fd)
    assert insn_rl.tolist() == [4]


def test_vc2_mid_instruction_helper_matches_wire() -> None:
    """Memory-operand-style instruction with VC2 followed by an
    INSTR_REP (e.g. ``MOV r0, [esi + #5]`` where the ``]`` is an
    INSTR_REP after the VC2). Raw: ``[INSTR_REP, INSTR_REP, VC2,
    inline_5, INSTR_REP]``. Helper slot count = 4 (3 carriers + VC2
    carrier; inline stripped). Same agreement check across the VC2-
    not-at-end case.
    """
    tokens = np.array(
        [_FIRST_INSTR_REP, _FIRST_INSTR_REP + 1, _VC2, 0x05,
         _FIRST_INSTR_REP + 2],
        dtype=np.uint16,
    )
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([5], dtype=np.uint32),
        block_runlength=np.array([5], dtype=np.uint32),
    )
    _assert_helper_matches_expand_tokens(fd)
    _, insn_rl = compute_metatoken_runlengths(fd)
    assert insn_rl.tolist() == [4]


def test_two_adjacent_instructions_each_ending_in_vc2_match_wire() -> None:
    """Pair of instructions back-to-back, each ending in a 1-byte VC2.
    Cross-check that the helper splits the wire stream cleanly at the
    insn boundary: insn[0] = 4 slots ending in VC2#1, insn[1] = 4 slots
    ending in VC2#2.
    """
    tokens = np.array(
        [
            # insn 0: [INSTR_REP, INSTR_REP, INSTR_REP, VC2, inline_4]
            _FIRST_INSTR_REP, _FIRST_INSTR_REP + 1, _FIRST_INSTR_REP + 2,
            _VC2, 0x04,
            # insn 1: [INSTR_REP, INSTR_REP, INSTR_REP, VC2, inline_8]
            _FIRST_INSTR_REP + 3, _FIRST_INSTR_REP + 4, _FIRST_INSTR_REP + 5,
            _VC2, 0x08,
        ],
        dtype=np.uint16,
    )
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([5, 5], dtype=np.uint32),
        block_runlength=np.array([10], dtype=np.uint32),
    )
    _assert_helper_matches_expand_tokens(fd)
    _, insn_rl = compute_metatoken_runlengths(fd)
    assert insn_rl.tolist() == [4, 4]


def test_multi_chunk_vc2_within_one_instruction_matches_wire() -> None:
    """Multi-chunk VC2 (9-byte payload, K_full=2) within a single
    instruction. Helper slot count = 2 (carrier + 1 painted
    continuation); the wire-side counter walks the painted-continuation
    slot too. Pins the multi-chunk happy path.
    """
    payload = np.full(9, 0x42, dtype=np.uint16)
    tokens = np.concatenate(
        [
            np.array([_FIRST_INSTR_REP, _VC2], dtype=np.uint16),
            payload,
            np.array([_FIRST_INSTR_REP + 1], dtype=np.uint16),
        ],
    )
    n = int(tokens.size)
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([n], dtype=np.uint32),
        block_runlength=np.array([n], dtype=np.uint32),
    )
    _assert_helper_matches_expand_tokens(fd)
    _, insn_rl = compute_metatoken_runlengths(fd)
    # 1 INSTR_REP + 2 VC2 (1 carrier + 1 painted) + 1 INSTR_REP = 4 slots.
    assert insn_rl.tolist() == [4]


def test_f128_finite_within_one_instruction_matches_wire() -> None:
    """F128 finite source within ONE instruction. Helper slot count = 2
    (carrier + 1 painted continuation); wire-side count agrees.
    """
    payload = np.zeros(16, dtype=np.uint16)
    # finite payload: high u16 != 0x7FFF after sign-strip
    tokens = np.concatenate(
        [
            np.array([_F128], dtype=np.uint16),
            payload,
            np.array([_FIRST_INSTR_REP], dtype=np.uint16),
        ],
    )
    n = int(tokens.size)
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([n], dtype=np.uint32),
        block_runlength=np.array([n], dtype=np.uint32),
    )
    _assert_helper_matches_expand_tokens(fd)
    _, insn_rl = compute_metatoken_runlengths(fd)
    # 1 F128 carrier + 1 painted + 1 INSTR_REP = 3 slots.
    assert insn_rl.tolist() == [3]


def test_f128_nan_inf_within_one_instruction_matches_wire() -> None:
    """F128 NaN/Inf source: 1-chunk (no painted continuation). Helper
    slot count = 1; wire-side count agrees.
    """
    payload = np.zeros(16, dtype=np.uint16)
    payload[0] = 0x7F
    payload[1] = 0xFF
    tokens = np.concatenate(
        [
            np.array([_F128], dtype=np.uint16),
            payload,
            np.array([_FIRST_INSTR_REP], dtype=np.uint16),
        ],
    )
    n = int(tokens.size)
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([n], dtype=np.uint32),
        block_runlength=np.array([n], dtype=np.uint32),
    )
    _assert_helper_matches_expand_tokens(fd)
    _, insn_rl = compute_metatoken_runlengths(fd)
    # 1 F128 (NaN/Inf, no paint) + 1 INSTR_REP = 2 slots.
    assert insn_rl.tolist() == [2]


def test_negative_vc2_value_negative_marker_does_not_drift_wire_count() -> None:
    """Negative-VC2 instruction with value_negative marker: helper
    drops the marker's slot count, ``expand_tokens`` strips the marker
    via ``> 256`` keep mask. Per-instruction count agrees.
    """
    tokens = np.array(
        [_FIRST_INSTR_REP, _VC2, 0x42, _VALUE_NEGATIVE, _FIRST_INSTR_REP + 1],
        dtype=np.uint16,
    )
    fd = _make_function_data(
        tokens=tokens,
        insn_runlength=np.array([5], dtype=np.uint32),
        block_runlength=np.array([5], dtype=np.uint32),
    )
    _assert_helper_matches_expand_tokens(fd)
    _, insn_rl = compute_metatoken_runlengths(fd)
    # 1 INSTR_REP + 1 VC2 carrier (inline stripped, value_negative stripped)
    # + 1 INSTR_REP = 3 slots.
    assert insn_rl.tolist() == [3]


# ---------------------------------------------------------------------------
# End-to-end integration with batch_decode.
# ---------------------------------------------------------------------------


def test_batch_decode_with_flag_emits_runlength_sidecars(tmp_path) -> None:
    """:func:`batch_decode` with ``emit_block_n_insns_runlength=True``
    populates the four sidecars; row offsets are cumsum-monotone and
    end at the flat array's total length.
    """
    fb = build_synthetic_binary(tmp_path)

    section_pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=0),
    ]
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as session:
        result = batch_decode(
            session,
            section_pointers=section_pointers,
            num_variants_per_section=2,
            context_len=128,
            max_depth=0,
            variant_padding=VariantPadding.PAD_NULL,
            emit_block_n_insns_runlength=True,
            rng=np.random.default_rng(seed=42),
        )
    assert result.block_runlength is not None
    assert result.block_runlength_row_offsets is not None
    assert result.insn_runlength is not None
    assert result.insn_runlength_row_offsets is not None

    # Per-row offsets are monotonic non-decreasing.
    block_offsets = result.block_runlength_row_offsets
    insn_offsets = result.insn_runlength_row_offsets
    assert np.all(np.diff(block_offsets) >= 0)
    assert np.all(np.diff(insn_offsets) >= 0)
    # End-of-cumsum equals the flat array's total length.
    assert int(block_offsets[-1]) == int(result.block_runlength.size)
    assert int(insn_offsets[-1]) == int(result.insn_runlength.size)
    # dtype contracts.
    assert result.block_runlength.dtype == np.uint32
    assert result.block_runlength_row_offsets.dtype == np.uint32
    assert result.insn_runlength.dtype == np.uint32
    assert result.insn_runlength_row_offsets.dtype == np.uint32


def test_batch_decode_default_flag_leaves_runlength_none(tmp_path) -> None:
    """When ``emit_block_n_insns_runlength=False`` (default) the four
    sidecar fields are ``None``."""
    fb = build_synthetic_binary(tmp_path)

    section_pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=0),
    ]
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as session:
        result = batch_decode(
            session,
            section_pointers=section_pointers,
            num_variants_per_section=1,
            context_len=64,
            max_depth=0,
            rng=np.random.default_rng(seed=42),
        )
    assert result.block_runlength is None
    assert result.block_runlength_row_offsets is None
    assert result.insn_runlength is None
    assert result.insn_runlength_row_offsets is None
