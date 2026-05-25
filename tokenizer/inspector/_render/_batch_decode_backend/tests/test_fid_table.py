"""Tests for :class:`FidBaseTable`.

Pins the per-row Category-segment cumulative-offset table that the
:class:`BatchDecodeBackend` row walker uses to translate
``(row, FUNCTION-Category, counter_id) -> FID``. Single concern: the
sidecar-driven base (NOT a stream-position bincount) -- audit
B-CRIT-1's correctness fix for recursive LOCAL_FUNC calls (multiple
in-stream identity references to the same callee collapse to ONE
sidecar entry per Category).

Plan reference: ``inspector-render-backends.md`` §6 + decision #16 +
audit B-CRIT-1.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    FUNCTION_CATEGORIES,
)
from tokenizer.aligned_data.loader.batch_decode._types import BatchDecodeResult
from tokenizer.inspector._render._batch_decode_backend._fid_table import (
    FidBaseTable,
)
from tokenizer.tokens import Category


# FUNCTION_CATEGORIES = (LOCAL_FUNC, PLT_FUNC, EXT_FUNC); pinned here
# so the test stays independent of any future reorder (which would
# also force the sidecar dtype's column layout to change).
assert FUNCTION_CATEGORIES == (
    Category.LOCAL_FUNC,
    Category.PLT_FUNC,
    Category.EXT_FUNC,
)


def _make_result_stub(
    *,
    per_category_counts: np.ndarray,
    row_offsets: np.ndarray,
    sidecar: np.ndarray,
) -> BatchDecodeResult:
    """Build a :class:`BatchDecodeResult` whose only populated fields
    are the three FID sidecars :meth:`FidBaseTable.from_result` reads.

    ``BatchDecodeResult`` is a frozen dataclass; MagicMock with
    explicit attribute pinning is the lowest-overhead synthetic
    fixture (no need to fill every flat tensor).
    """
    stub = MagicMock(spec=BatchDecodeResult)
    stub.fid_per_category_counts = per_category_counts
    stub.fid_row_offsets = row_offsets
    stub.fid_sidecar = sidecar
    return stub


# ---------------------------------------------------------------------------
# from_result invariants
# ---------------------------------------------------------------------------


def test_from_result_raises_when_sidecar_absent() -> None:
    """``include_fid_sidecar=False`` -> all three fields None ->
    typed :class:`ValueError` rather than silent broken bases.
    """
    stub = MagicMock(spec=BatchDecodeResult)
    stub.fid_per_category_counts = None
    stub.fid_row_offsets = None
    stub.fid_sidecar = None
    with pytest.raises(ValueError, match="include_fid_sidecar=True"):
        FidBaseTable.from_result(stub)


def test_cumsum_three_categories_single_row() -> None:
    """Counts (3, 2, 1) -> bases (0, 3, 5).

    Pins the cumsum semantics: column ``c`` holds the START offset of
    the c-th Category's segment (= sum of preceding Category counts),
    NOT the end offset.
    """
    counts = np.asarray([[3, 2, 1]], dtype=np.uint32)
    row_offsets = np.asarray([0, 6], dtype=np.uint32)
    sidecar = np.arange(6, dtype=np.uint32)
    result = _make_result_stub(
        per_category_counts=counts, row_offsets=row_offsets, sidecar=sidecar,
    )
    table = FidBaseTable.from_result(result)
    assert table.bases_by_row.dtype == np.uint32
    assert table.bases_by_row.tolist() == [[0, 3, 5]]


def test_cumsum_multi_row_independent_per_row() -> None:
    """Row 0 counts (2, 1, 0) -> bases (0, 2, 3);
    Row 1 counts (0, 3, 2) -> bases (0, 0, 3).

    Per-row cumsum is independent: no leakage across rows.
    """
    counts = np.asarray([[2, 1, 0], [0, 3, 2]], dtype=np.uint32)
    row_offsets = np.asarray([0, 3, 8], dtype=np.uint32)
    sidecar = np.arange(8, dtype=np.uint32)
    result = _make_result_stub(
        per_category_counts=counts, row_offsets=row_offsets, sidecar=sidecar,
    )
    table = FidBaseTable.from_result(result)
    assert table.bases_by_row.tolist() == [[0, 2, 3], [0, 0, 3]]


def test_first_column_is_always_zero() -> None:
    """Column 0 (= LOCAL_FUNC segment start) is ALWAYS zero -- the
    first Category begins at the row's own start in ``fid_sidecar``.
    """
    counts = np.asarray(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=np.uint32,
    )
    row_offsets = np.asarray([0, 5, 10, 15], dtype=np.uint32)
    sidecar = np.arange(15, dtype=np.uint32)
    result = _make_result_stub(
        per_category_counts=counts, row_offsets=row_offsets, sidecar=sidecar,
    )
    table = FidBaseTable.from_result(result)
    assert (table.bases_by_row[:, 0] == 0).all()


# ---------------------------------------------------------------------------
# lookup() resolution
# ---------------------------------------------------------------------------


def test_lookup_returns_correct_fid_single_category() -> None:
    """Row 0 has 3 LOCAL FIDs (100, 101, 102); lookup(0, LOCAL, K)
    returns ``sidecar[row_offsets[0] + 0 + K]`` for K in 0..2.
    """
    counts = np.asarray([[3, 0, 0]], dtype=np.uint32)
    row_offsets = np.asarray([0, 3], dtype=np.uint32)
    sidecar = np.asarray([100, 101, 102], dtype=np.uint32)
    result = _make_result_stub(
        per_category_counts=counts, row_offsets=row_offsets, sidecar=sidecar,
    )
    table = FidBaseTable.from_result(result)
    assert table.lookup(row=0, cat=Category.LOCAL_FUNC, counter=0) == 100
    assert table.lookup(row=0, cat=Category.LOCAL_FUNC, counter=1) == 101
    assert table.lookup(row=0, cat=Category.LOCAL_FUNC, counter=2) == 102


def test_lookup_across_categories_one_row() -> None:
    """Mixed row: 2 LOCAL + 1 PLT + 1 EXT, sidecar = [L0, L1, P0, E0].

    Per-Category counter starts at 0; lookups land in their own
    segment via the cumulative base.
    """
    counts = np.asarray([[2, 1, 1]], dtype=np.uint32)
    row_offsets = np.asarray([0, 4], dtype=np.uint32)
    sidecar = np.asarray([10, 11, 20, 30], dtype=np.uint32)
    result = _make_result_stub(
        per_category_counts=counts, row_offsets=row_offsets, sidecar=sidecar,
    )
    table = FidBaseTable.from_result(result)
    assert table.lookup(row=0, cat=Category.LOCAL_FUNC, counter=0) == 10
    assert table.lookup(row=0, cat=Category.LOCAL_FUNC, counter=1) == 11
    assert table.lookup(row=0, cat=Category.PLT_FUNC, counter=0) == 20
    assert table.lookup(row=0, cat=Category.EXT_FUNC, counter=0) == 30


def test_lookup_multi_row_uses_row_offset() -> None:
    """Row 1 lookups land in the row-offsets[1] segment of ``sidecar``.

    Pins that ``lookup`` adds ``fid_row_offsets[row]`` before the
    in-row Category base -- the flat sidecar's per-row segmentation is
    the OFFSETS array, not the cumsum.
    """
    counts = np.asarray([[1, 0, 0], [1, 1, 0]], dtype=np.uint32)
    row_offsets = np.asarray([0, 1, 3], dtype=np.uint32)
    sidecar = np.asarray([100, 200, 201], dtype=np.uint32)
    result = _make_result_stub(
        per_category_counts=counts, row_offsets=row_offsets, sidecar=sidecar,
    )
    table = FidBaseTable.from_result(result)
    assert table.lookup(row=0, cat=Category.LOCAL_FUNC, counter=0) == 100
    assert table.lookup(row=1, cat=Category.LOCAL_FUNC, counter=0) == 200
    assert table.lookup(row=1, cat=Category.PLT_FUNC, counter=0) == 201


def test_lookup_recursive_call_uses_dedup_count_not_bincount() -> None:
    """Audit B-CRIT-1: a row whose IDENTITY stream has FIVE in-stream
    LOCAL_FUNC token positions all referencing the SAME callee (a
    self-recursive caller) holds ONE entry in ``fid_sidecar`` for
    that callee.

    The naive "bincount the IDENTITY stream" approach would emit
    ``LOCAL_FUNC count = 5`` and produce an out-of-bounds lookup;
    the sidecar-driven count is ``1`` and looking up counter=0
    correctly returns the single dedup entry. This test pins the
    correctness fix that motivated the entire ``FidBaseTable`` design.
    """
    counts = np.asarray([[1, 0, 0]], dtype=np.uint32)  # NOT 5!
    row_offsets = np.asarray([0, 1], dtype=np.uint32)
    sidecar = np.asarray([42], dtype=np.uint32)  # the single dedup FID
    result = _make_result_stub(
        per_category_counts=counts, row_offsets=row_offsets, sidecar=sidecar,
    )
    table = FidBaseTable.from_result(result)
    assert table.lookup(row=0, cat=Category.LOCAL_FUNC, counter=0) == 42


def test_lookup_raises_on_non_function_category() -> None:
    """The IDENTITY-band Category emitter dispatches FUNCTION
    categories ONLY to :meth:`lookup`; BLOCK / STRING_PTR / etc. go
    through the COUNTER path. Typed contract: pass a non-FUNCTION
    Category and get :class:`KeyError`.
    """
    counts = np.asarray([[1, 1, 1]], dtype=np.uint32)
    row_offsets = np.asarray([0, 3], dtype=np.uint32)
    sidecar = np.asarray([1, 2, 3], dtype=np.uint32)
    result = _make_result_stub(
        per_category_counts=counts, row_offsets=row_offsets, sidecar=sidecar,
    )
    table = FidBaseTable.from_result(result)
    with pytest.raises(KeyError, match="FUNCTION Category"):
        table.lookup(row=0, cat=Category.BLOCK, counter=0)
    with pytest.raises(KeyError, match="FUNCTION Category"):
        table.lookup(row=0, cat=Category.STRING_PTR, counter=0)


def test_lookup_padding_row_zero_counts_raises_indexerror() -> None:
    """A padding row with counts ``(0, 0, 0)`` cannot answer any
    ``lookup`` (the segment is empty); the helper is concerned ONLY
    with offset math, so a stray lookup falls out as an IndexError
    from the flat sidecar indexing -- there is no defensive fallback
    masking a caller bug.

    Construct a row whose sidecar segment is empty (row_offsets[1] ==
    row_offsets[2]) and confirm the loud failure.
    """
    counts = np.asarray([[1, 0, 0], [0, 0, 0]], dtype=np.uint32)
    row_offsets = np.asarray([0, 1, 1], dtype=np.uint32)  # row 1 empty
    sidecar = np.asarray([99], dtype=np.uint32)
    result = _make_result_stub(
        per_category_counts=counts, row_offsets=row_offsets, sidecar=sidecar,
    )
    table = FidBaseTable.from_result(result)
    # row 1, segment empty -- counter=0 lands at sidecar[1] which is OOB
    with pytest.raises(IndexError):
        table.lookup(row=1, cat=Category.LOCAL_FUNC, counter=0)
