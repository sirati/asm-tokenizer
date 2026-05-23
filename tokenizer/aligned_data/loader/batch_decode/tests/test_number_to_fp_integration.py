"""3c -> 3d byte-equivalence integration test for the number arm.

Single concern: pin the chained contract between
:func:`build_number_idx_2d` (3c) and :func:`normalize_per_token_type`
(3d) end-to-end. 3c emits per-:class:`TokenType` ``idx_2d`` arrays +
the ``f128_is_nan_or_inf`` + ``vc2_chunk_exponent_sidecar`` sidecars;
3d gathers payload bytes through ``idx_2d`` and produces normalized
``(significand, sign_exp)`` chunks.

Earlier audit found a layout mismatch between the two: 3c emits F128
as ``(n_chunks_total, 8)`` per-chunk rows (2 per finite source, 1 per
NaN/Inf source) while the old 3d expected ``(n_sources, 16)``. This
test exists specifically to pin the chained-output byte-equivalence
against the per-source oracle so the fixed contract can't silently
regress.

For each F128 source we compare the chained output to the explicit
batch-layout helper from :mod:`test_fp_normalize` (always 2 chunks per
finite source; ``.rodata-robustness`` high-mantissa-only NaN/Inf
classification). Mixed-batch tests additionally exercise F32 sources
alongside F128 to ensure other TokenTypes still work through the same
chained call.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._fp_normalize import (
    normalize_per_token_type,
)
from tokenizer.aligned_data.loader.batch_decode._number_decode import (
    build_number_idx_2d,
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
from tokenizer.aligned_data.loader.decoded.custom_float import from_float32
from tokenizer.aligned_data.loader.decoded.run_lengths import run_lengths
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import Section
from tokenizer.tokens import Category, TokenType

# Reuse the batch-layout oracle helper (single source of truth for
# "what 3c -> 3d emits per F128 source") from the unit test file.
from .test_fp_normalize import (
    _f128_batch_expected_chunks,
    _oracle_pairs_to_arrays,
)


# ---------------------------------------------------------------------------
# Vocab constants (mirror test_number_decode.py to keep this file self-
# contained; a vocab-layout shift surfaces here too).
# ---------------------------------------------------------------------------

_F32_RAW = 260
_F128_RAW = 263

_F32_SHIFTED = 4
_F128_SHIFTED = 7
_LOCAL_FUNC_SHIFTED = 9


# ---------------------------------------------------------------------------
# Fixture builders.
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


def _empty_section() -> Section:
    return Section(
        function_name_ptr=0,
        section_offset=0,
        call_targets=[],
        variants=[],
    )


def _build_state(raw_tokens: np.ndarray) -> InlineDecodeState:
    """Build an :class:`InlineDecodeState` from ``raw_tokens``.

    Mirrors :mod:`test_number_decode`'s builder -- only the fields
    :func:`build_number_idx_2d` reads are populated meaningfully.
    """
    real_mask = raw_tokens > 256
    number_mask = raw_tokens < 256
    if raw_tokens.shape[0] == 0:
        runlen_number = np.zeros(0, dtype=np.uint32)
        runlen_value = np.zeros(0, dtype=np.uint32)
    else:
        runlen_number = run_lengths(number_mask)
        runlen_value = run_lengths(~real_mask)
    carries_inline_mask = real_mask & (raw_tokens < 272)
    is_negative_per_position = np.zeros(raw_tokens.shape[0], dtype=bool)
    return InlineDecodeState(
        raw_tokens=raw_tokens,
        real_mask=real_mask,
        number_mask=number_mask,
        runlen_number=runlen_number,
        runlen_value=runlen_value,
        carries_inline_mask=carries_inline_mask,
        is_negative_per_position=is_negative_per_position,
    )


def _wrap_single_call_target(stage2_ct: Stage2CallTarget) -> Stage2Batch:
    """Wrap one Stage2CallTarget into a minimal Stage2Batch."""
    stage1_ct = stage2_ct.stage1
    stage1_variant = Stage1Variant(
        variant_idx=0,
        variant_ref_offset=0,
        batch_idx=0,
        call_targets=[stage1_ct],
    )
    stage1_section = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=0,
        section=_empty_section(),
        variants=[stage1_variant],
    )
    stage1_batch = Stage1Batch(
        sections=[stage1_section],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        batch_size=1,
    )
    stage2_variant = Stage2Variant(
        stage1=stage1_variant,
        call_targets=[stage2_ct],
        cut_call_target_index=0 if stage2_ct.is_cut else 1,
        total_surviving_token_count=stage2_ct.surviving_token_count,
        total_surviving_identity_count=stage2_ct.surviving_identity_count,
        total_surviving_number_chunk_count=stage2_ct.surviving_number_chunk_count,
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


def _make_call_target(
    raw_tokens: np.ndarray,
    expanded_token_ids: np.ndarray,
    extra_value_v2_mask: np.ndarray,
    extra_f128_mask: np.ndarray,
) -> Stage2CallTarget:
    """Build a Stage2CallTarget around hand-crafted expanded stream."""
    stage1_ct = Stage1CallTarget(
        function_data=_empty_function_data(),
        state=_build_state(raw_tokens),
        call_targets_section=[],
        encounter_category=Category.LOCAL_FUNC,
        parent_call_target_index=None,
        function_name_ptr=0,
    )
    predicted_full_length = int(expanded_token_ids.shape[0])
    identity_band_mask = (expanded_token_ids >= 8) & (expanded_token_ids < 16)
    number_band_mask = (expanded_token_ids >= 1) & (expanded_token_ids < 8)
    return Stage2CallTarget(
        stage1=stage1_ct,
        expanded_token_ids=expanded_token_ids,
        extra_value_v2_mask=extra_value_v2_mask,
        extra_f128_mask=extra_f128_mask,
        predicted_full_length=predicted_full_length,
        surviving_token_count=predicted_full_length,
        surviving_identity_count=int(identity_band_mask.sum()),
        surviving_number_chunk_count=int(number_band_mask.sum()),
        is_cut=False,
        partial_cut_length=predicted_full_length,
    )


def _build_inline_bytes_from_raw(
    raw_tokens: np.ndarray,
) -> tuple[np.ndarray, slice]:
    """Synthetic 3a-style inline_bytes -- leading pad + payload bytes."""
    payload = raw_tokens[raw_tokens < 256].astype(np.uint8)
    inline_bytes = np.empty(1 + payload.shape[0], dtype=np.uint8)
    inline_bytes[0] = 0
    inline_bytes[1:] = payload
    return inline_bytes, slice(1, 1 + payload.shape[0])


# ---------------------------------------------------------------------------
# F128 stream builders.
# ---------------------------------------------------------------------------


def _f128_bytes(bits: int) -> bytes:
    """Pack a 128-bit value into 16 big-endian bytes."""
    high = (bits >> 64) & ((1 << 64) - 1)
    low = bits & ((1 << 64) - 1)
    return high.to_bytes(8, "big") + low.to_bytes(8, "big")


def _build_f128_stream(
    bits_list: list[int],
) -> tuple[Stage2Batch, np.ndarray, slice]:
    """Build a single-call_target Stage2Batch + inline_bytes for N F128
    sources.

    For each source we emit:
      raw_tokens: [F128_RAW, b0, b1, ..., b15]
      expanded:   [F128_SHIFTED, F128_SHIFTED?]  (the trailing slot is
                  the painted-continuation chunk-1 for finite sources;
                  absent for NaN/Inf sources per ALG-2).

    All N sources are packed into one raw_tokens / expanded stream so
    the call_target's surviving prefix covers them all.
    """
    raw_chunks: list[np.ndarray] = []
    expanded_chunks: list[np.ndarray] = []
    extra_f128_chunks: list[np.ndarray] = []
    raw_chunks.append(np.array([_LOCAL_FUNC_SHIFTED], dtype=np.uint16))
    # ``raw_chunks`` here is used to grow raw_tokens -- the prepend slot
    # is in EXPANDED, not raw, so reset and rebuild raw separately.
    raw_chunks = []
    # Prepend slot lives in expanded only.
    expanded_chunks.append(np.array([_LOCAL_FUNC_SHIFTED], dtype=np.uint16))
    extra_f128_chunks.append(np.array([False], dtype=bool))

    is_nan_or_inf_per_source: list[bool] = []
    for bits in bits_list:
        payload = _f128_bytes(bits)
        raw_chunks.append(np.array([_F128_RAW], dtype=np.uint16))
        raw_chunks.append(
            np.frombuffer(payload, dtype=np.uint8).astype(np.uint16)
        )
        biased_exp = (bits >> 112) & 0x7FFF
        nan_or_inf = biased_exp == 0x7FFF
        is_nan_or_inf_per_source.append(bool(nan_or_inf))
        if nan_or_inf:
            # NaN/Inf: only the carrier survives in expanded (no painted
            # continuation -- per ALG-2 stage 2 skips painting for
            # ``biased_exp == 0x7FFF``).
            expanded_chunks.append(np.array([_F128_SHIFTED], dtype=np.uint16))
            extra_f128_chunks.append(np.array([False], dtype=bool))
        else:
            # Finite: carrier + 1 painted continuation slot.
            expanded_chunks.append(
                np.array([_F128_SHIFTED, _F128_SHIFTED], dtype=np.uint16)
            )
            extra_f128_chunks.append(np.array([False, True], dtype=bool))

    raw_tokens = np.concatenate(raw_chunks)
    expanded = np.concatenate(expanded_chunks)
    extra_f128 = np.concatenate(extra_f128_chunks)
    extra_vc2 = np.zeros_like(extra_f128, dtype=bool)

    stage2_ct = _make_call_target(raw_tokens, expanded, extra_vc2, extra_f128)
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes, ct_slice = _build_inline_bytes_from_raw(raw_tokens)
    return stage2_batch, inline_bytes, ct_slice


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


# Bit patterns -- mirrors the test_fp_normalize.py F128 vocabulary.
def _f128_finite_bits(sign: int, biased_exp: int, raw_mantissa: int) -> int:
    return (sign << 127) | (biased_exp << 112) | raw_mantissa


F128_POS_ZERO = 0
F128_NEG_ZERO = 1 << 127
F128_ONE = _f128_finite_bits(0, 16383, 0)
F128_NEG_ONE = _f128_finite_bits(1, 16383, 0)
F128_POS_INF = _f128_finite_bits(0, 0x7FFF, 0)
F128_NEG_INF = _f128_finite_bits(1, 0x7FFF, 0)
# NaN with high-mantissa-bit set so the .rodata-robustness classifier
# (high_mantissa == 0 -> Inf) reports NaN, matching the bit pattern.
F128_NAN = _f128_finite_bits(0, 0x7FFF, 1 << 100)
F128_SMALLEST_NORMAL = _f128_finite_bits(0, 1, 0)
F128_LARGEST_FINITE = _f128_finite_bits(0, 0x7FFE, (1 << 112) - 1)
F128_BIG_DENORMAL = _f128_finite_bits(0, 0, (1 << 112) - 1)


@pytest.mark.parametrize(
    "bits_list, label",
    [
        ([F128_ONE], "one_finite"),
        ([F128_POS_INF], "one_pos_inf"),
        ([F128_NAN], "one_nan_high_mantissa"),
        ([F128_ONE, F128_NEG_ONE], "two_finite"),
        ([F128_ONE, F128_POS_INF, F128_NEG_ONE], "finite_inf_finite"),
        ([F128_POS_INF, F128_NEG_INF, F128_NAN], "all_nan_or_inf"),
        (
            [
                F128_POS_ZERO,
                F128_NEG_ZERO,
                F128_SMALLEST_NORMAL,
                F128_LARGEST_FINITE,
                F128_BIG_DENORMAL,
            ],
            "finite_edge_cases",
        ),
        (
            [
                F128_ONE,
                F128_POS_INF,
                F128_NEG_ONE,
                F128_NAN,
                F128_LARGEST_FINITE,
                F128_NEG_INF,
                F128_SMALLEST_NORMAL,
            ],
            "mixed_finite_and_nan_inf",
        ),
    ],
)
def test_f128_3c_to_3d_chain_byte_equivalent(
    bits_list: list[int], label: str
) -> None:
    """3c output piped into 3d MUST be byte-equivalent to running the
    per-source batch-layout oracle.

    Pins the chained contract: ``build_number_idx_2d`` produces
    ``(idx_2d, f128_is_nan_or_inf)`` consistent with what
    ``normalize_f128`` consumes -- finite sources contribute 2 chunks
    (LSB then MSB limb), NaN/Inf sources contribute 1 chunk (MSB limb
    only), with the ``.rodata-robustness`` high-mantissa-only Inf/NaN
    classification.
    """
    stage2_batch, inline_bytes, ct_slice = _build_f128_stream(bits_list)

    idx_2d_per_type, _, f128_is_nan_or_inf, vc2_sidecar = build_number_idx_2d(
        stage2_batch, inline_bytes, [ct_slice]
    )

    # Sanity: f128_is_nan_or_inf length = n_sources; idx_2d row count =
    # sum of chunks_per_source.
    n_sources = len(bits_list)
    assert f128_is_nan_or_inf.shape == (n_sources,)
    expected_row_count = sum(
        1 if _is_nan_or_inf(b) else 2 for b in bits_list
    )
    assert idx_2d_per_type[TokenType.FLOAT128].shape == (expected_row_count, 8)

    out = normalize_per_token_type(
        idx_2d_per_type=idx_2d_per_type,
        inline_bytes=inline_bytes,
        f128_is_nan_or_inf=f128_is_nan_or_inf,
        vc2_chunk_exponent_sidecar=vc2_sidecar,
        is_negative_per_source_per_type={
            TokenType.FLOAT128: np.zeros(n_sources, dtype=bool),
        },
    )
    got_sig, got_sign_exp = out[TokenType.FLOAT128]

    expected_sig, expected_sign_exp = _oracle_pairs_to_arrays(
        [_f128_batch_expected_chunks(b) for b in bits_list]
    )
    np.testing.assert_array_equal(got_sig, expected_sig)
    np.testing.assert_array_equal(got_sign_exp, expected_sign_exp)


def _is_nan_or_inf(bits: int) -> bool:
    return ((bits >> 112) & 0x7FFF) == 0x7FFF


# ---------------------------------------------------------------------------
# Mixed-batch (F32 + F128) -- ensures other TokenTypes still work.
# ---------------------------------------------------------------------------


def _f32_bits(value: float) -> int:
    return int(np.frombuffer(np.float32(value).tobytes(), dtype=np.uint32)[0])


def test_f32_plus_f128_chain_byte_equivalent() -> None:
    """A stream mixing F32 + F128 sources MUST chain through 3c -> 3d
    with each TokenType's output byte-equivalent to its per-source oracle.

    The F32 source goes through the IEEE-narrow normalizer; the F128
    sources go through the per-chunk normalizer; the two TokenTypes
    don't share idx_2d / sidecar state. This test pins that the chain
    keeps them isolated.
    """
    f32_value = _f32_bits(1.5)
    f128_values = [F128_ONE, F128_POS_INF, F128_NEG_ONE]

    # Build raw_tokens: prepend (in expanded only) -> F32 carrier + 4
    # bytes -> 3x (F128 carrier + 16 bytes).
    raw_chunks: list[np.ndarray] = []
    expanded_chunks: list[np.ndarray] = []
    extra_f128_chunks: list[np.ndarray] = []
    expanded_chunks.append(np.array([_LOCAL_FUNC_SHIFTED], dtype=np.uint16))
    extra_f128_chunks.append(np.array([False], dtype=bool))

    # F32 source.
    raw_chunks.append(np.array([_F32_RAW], dtype=np.uint16))
    raw_chunks.append(
        np.frombuffer(
            f32_value.to_bytes(4, "big"), dtype=np.uint8
        ).astype(np.uint16)
    )
    expanded_chunks.append(np.array([_F32_SHIFTED], dtype=np.uint16))
    extra_f128_chunks.append(np.array([False], dtype=bool))

    # F128 sources.
    for bits in f128_values:
        payload = _f128_bytes(bits)
        raw_chunks.append(np.array([_F128_RAW], dtype=np.uint16))
        raw_chunks.append(
            np.frombuffer(payload, dtype=np.uint8).astype(np.uint16)
        )
        if _is_nan_or_inf(bits):
            expanded_chunks.append(np.array([_F128_SHIFTED], dtype=np.uint16))
            extra_f128_chunks.append(np.array([False], dtype=bool))
        else:
            expanded_chunks.append(
                np.array([_F128_SHIFTED, _F128_SHIFTED], dtype=np.uint16)
            )
            extra_f128_chunks.append(np.array([False, True], dtype=bool))

    raw_tokens = np.concatenate(raw_chunks)
    expanded = np.concatenate(expanded_chunks)
    extra_f128 = np.concatenate(extra_f128_chunks)
    extra_vc2 = np.zeros_like(extra_f128, dtype=bool)

    stage2_ct = _make_call_target(raw_tokens, expanded, extra_vc2, extra_f128)
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes, ct_slice = _build_inline_bytes_from_raw(raw_tokens)

    idx_2d_per_type, _, f128_is_nan_or_inf, vc2_sidecar = build_number_idx_2d(
        stage2_batch, inline_bytes, [ct_slice]
    )

    n_f128_sources = len(f128_values)
    out = normalize_per_token_type(
        idx_2d_per_type=idx_2d_per_type,
        inline_bytes=inline_bytes,
        f128_is_nan_or_inf=f128_is_nan_or_inf,
        vc2_chunk_exponent_sidecar=vc2_sidecar,
        is_negative_per_source_per_type={
            T: np.zeros(idx_2d_per_type[T].shape[0], dtype=bool)
            for T in idx_2d_per_type
            if T is not TokenType.FLOAT128
        }
        | {
            # F128's is_negative-per-source matches n_f128_sources (sign
            # in bit pattern; param ignored by F128 normalizer).
            TokenType.FLOAT128: np.zeros(n_f128_sources, dtype=bool),
        },
    )

    # F32: single source, 1 chunk -- bit-equivalent to from_float32.
    expected_f32_sig, expected_f32_sign_exp = _oracle_pairs_to_arrays(
        [from_float32(f32_value)]
    )
    got_f32_sig, got_f32_sign_exp = out[TokenType.FLOAT32]
    np.testing.assert_array_equal(got_f32_sig, expected_f32_sig)
    np.testing.assert_array_equal(got_f32_sign_exp, expected_f32_sign_exp)

    # F128: batch-layout chunks per source.
    expected_f128_sig, expected_f128_sign_exp = _oracle_pairs_to_arrays(
        [_f128_batch_expected_chunks(b) for b in f128_values]
    )
    got_f128_sig, got_f128_sign_exp = out[TokenType.FLOAT128]
    np.testing.assert_array_equal(got_f128_sig, expected_f128_sig)
    np.testing.assert_array_equal(got_f128_sign_exp, expected_f128_sign_exp)
