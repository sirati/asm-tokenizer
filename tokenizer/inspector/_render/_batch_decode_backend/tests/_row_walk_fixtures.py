"""Shared synthetic fixtures for the row-walk test modules.

Single concern: build minimal :class:`BatchDecodeResult` /
:class:`FidBaseTable` / :class:`VocabularyManager` stubs the
:func:`render_row_blocks` walker reads, without standing up a real
:class:`BinarySession` + 4-stage pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    _CATEGORY_TO_SHIFTED_ID,
)
from tokenizer.aligned_data.loader.batch_decode._number_decode._band_constants import (
    _NUMBER_BAND_LO_SHIFTED,
)
from tokenizer.aligned_data.loader.batch_decode._types import BatchDecodeResult
from tokenizer.inspector._render._batch_decode_backend._fid_table import (
    FidBaseTable,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category


# Shifted-id anchors (derived so a vocab-layout shift updates here implicitly).
BLOCK_V2 = _CATEGORY_TO_SHIFTED_ID[Category.BLOCK]  # 8
LOCAL_FUNC = _CATEGORY_TO_SHIFTED_ID[Category.LOCAL_FUNC]  # 9
PLT_FUNC = _CATEGORY_TO_SHIFTED_ID[Category.PLT_FUNC]  # 10
EXT_FUNC = _CATEGORY_TO_SHIFTED_ID[Category.EXT_FUNC]  # 11
STRING_PTR = _CATEGORY_TO_SHIFTED_ID[Category.STRING_PTR]
VC2_NUMBER = _NUMBER_BAND_LO_SHIFTED  # 1
F32_NUMBER = _NUMBER_BAND_LO_SHIFTED + 3
F128_NUMBER = _NUMBER_BAND_LO_SHIFTED + 6
INSTR_REP_TOKEN = (
    VocabularyManager._V2_EAGER_BLOCK_END
    - VocabularyManager._V2_RESERVED_DIGIT_COUNT
)  # 16 (first instruction-rep shifted id)


def make_result(
    *,
    tokens_row: np.ndarray,
    identities: np.ndarray,
    numbers_sig: np.ndarray,
    numbers_se: np.ndarray,
    block_runlength: np.ndarray | None = None,
    insn_runlength: np.ndarray | None = None,
) -> BatchDecodeResult:
    """Construct a 1-row :class:`BatchDecodeResult` stub via
    :class:`MagicMock`; only fields the walker reads are populated.

    ``block_runlength`` / ``insn_runlength`` default to empty arrays --
    callers exercising the runlength-driven block boundaries supply
    explicit per-row counts.
    """
    stub = MagicMock(spec=BatchDecodeResult)
    stub.tokens = np.asarray([tokens_row], dtype=np.uint16)
    stub.identities = identities.astype(np.uint16)
    stub.identity_row_offsets = np.asarray(
        [0, len(identities)], dtype=np.uint32,
    )
    stub.numbers_significant = numbers_sig.astype(np.uint64)
    stub.numbers_sign_exponent = numbers_se.astype(np.uint32)
    stub.number_row_offsets = np.asarray(
        [0, len(numbers_sig)], dtype=np.uint32,
    )
    br = (
        block_runlength.astype(np.uint32)
        if block_runlength is not None
        else np.zeros(0, dtype=np.uint32)
    )
    ir = (
        insn_runlength.astype(np.uint32)
        if insn_runlength is not None
        else np.zeros(0, dtype=np.uint32)
    )
    stub.block_runlength = br
    stub.block_runlength_row_offsets = np.asarray(
        [0, int(br.size)], dtype=np.uint32,
    )
    stub.insn_runlength = ir
    stub.insn_runlength_row_offsets = np.asarray(
        [0, int(ir.size)], dtype=np.uint32,
    )
    return stub


# Resolver stub for tests that don't exercise callee resolution: every
# pointer resolves to ``None`` (matches EXTERN's no-body behavior).
NULL_CALLEE_RESOLVER = lambda _offset: None


def make_fid_table(
    *,
    per_category_counts: np.ndarray,
    sidecar: np.ndarray,
) -> FidBaseTable:
    """Build a :class:`FidBaseTable` covering one row via
    :meth:`from_result` so the cumsum / offsets are exercised through
    the real :class:`FidBaseTable` construction path.
    """
    row_offsets = np.asarray([0, len(sidecar)], dtype=np.uint32)
    stub = MagicMock(spec=BatchDecodeResult)
    stub.fid_per_category_counts = per_category_counts.astype(np.uint32)
    stub.fid_row_offsets = row_offsets
    stub.fid_sidecar = sidecar.astype(np.uint32)
    return FidBaseTable.from_result(stub)


def vocab_stub() -> MagicMock:
    """A :class:`VocabularyManager`-shaped stub whose
    :meth:`get_token_str` returns ``f"insn_{original_id}"``.
    """
    vm = MagicMock(spec=VocabularyManager)
    vm.get_token_str = lambda token_id: f"insn_{token_id}"
    return vm


EMPTY_NUMBERS = (
    np.zeros(0, dtype=np.uint64),
    np.zeros(0, dtype=np.uint32),
)
"""Convenience pair for rows with no NUMBER-band tokens."""


EMPTY_FID_COUNTS = np.zeros((1, 3), dtype=np.uint32)
EMPTY_FID_SIDECAR = np.zeros(0, dtype=np.uint32)
