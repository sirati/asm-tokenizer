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
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode import (
    SectionPointerSpec,
    VariantPadding,
    batch_decode,
)
from tokenizer.aligned_data.loader.batch_decode._runlengths import (
    compute_metatoken_runlengths,
    manual_calc_block_n_insn_runlengths,
)
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.loader.tests._session_fixture import (
    build_synthetic_binary,
)


# Vocab anchors mirror :mod:`._runlengths` so the synthetic streams
# below remain readable.
_RESERVED = 256
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
