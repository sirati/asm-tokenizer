"""Per-row FID-base table for the BatchDecodeBackend walk.

Single concern: convert the
:attr:`BatchDecodeResult.fid_per_category_counts` sidecar (plan
decision #16) into a typed ``(row, Category, counter_id) -> fid_sidecar
index`` lookup. The walker reads this when an IDENTITY token in the
FUNCTION-Category band lands on the wire -- it needs the FID
(``function_name_ptr``) keyed off the per-row Category-segment in
:attr:`BatchDecodeResult.fid_sidecar`.

Why a sidecar-driven base instead of a stream-position bincount
(audit B-CRIT-1): the dedup walk collapses repeated same-callee
references within a row to ONE sidecar entry, so a naive ``bincount``
over the IDENTITY band overcounts under recursive LOCAL_FUNC calls.
The sidecar's per-row per-Category dedup cardinality is the only
correct base.

Plan reference: ``inspector-render-backends.md`` §6 +
``_batch_decode_backend`` audit B-CRIT-1 / B-HIGH-7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    FUNCTION_CATEGORIES,
)
from tokenizer.aligned_data.loader.batch_decode._types import BatchDecodeResult
from tokenizer.tokens import Category


__all__ = ["FidBaseTable"]


# Cache the per-FUNCTION Category column index once -- the
# FUNCTION_CATEGORIES tuple order is the contract pinned by
# ``BatchDecodeResult.fid_per_category_counts`` (plan decision #16,
# see the field docstring).
_CATEGORY_TO_COLUMN: dict[Category, int] = {
    cat: idx for idx, cat in enumerate(FUNCTION_CATEGORIES)
}


@dataclass(frozen=True)
class FidBaseTable:
    """Per-row cumulative-offset table into :attr:`BatchDecodeResult.fid_sidecar`.

    Construction (via :meth:`from_result`) pre-computes the per-row
    cumulative offsets across the FUNCTION-Category partition --
    ``bases_by_row[r, col] = sum(per_category_counts[r, :col])`` --
    so :meth:`lookup` is O(1) per call site.

    The row index used here is the ``fid_row_offsets``-relative row;
    callers add ``fid_row_offsets[row]`` themselves before indexing
    :attr:`BatchDecodeResult.fid_sidecar` so this helper stays
    concerned only with the per-Category-segment offset within the
    row.
    """

    bases_by_row: np.ndarray  # ``u32[batch_size, len(FUNCTION_CATEGORIES)]``
    """Per-row cumulative-segment offsets. Column ``c`` holds
    ``sum(fid_per_category_counts[row, :c])``."""

    fid_row_offsets: np.ndarray
    """``u32[batch_size + 1]`` -- the row offsets into
    :attr:`BatchDecodeResult.fid_sidecar` (passed through so
    :meth:`lookup` can resolve a flat ``fid_sidecar`` index in one call)."""

    fid_sidecar: np.ndarray
    """``u32`` flat array -- :attr:`BatchDecodeResult.fid_sidecar`.
    Indexed by ``fid_row_offsets[row] + bases_by_row[row, col] + counter``."""

    @classmethod
    def from_result(cls, result: BatchDecodeResult) -> "FidBaseTable":
        """Pre-compute the per-row Category cumulative offsets.

        Requires the result to have been produced with
        ``include_fid_sidecar=True`` -- otherwise the three sidecar
        fields are ``None`` and this helper raises a typed error
        rather than emitting silently-broken bases.
        """
        if (
            result.fid_per_category_counts is None
            or result.fid_row_offsets is None
            or result.fid_sidecar is None
        ):
            raise ValueError(
                "FidBaseTable.from_result requires a BatchDecodeResult "
                "produced with include_fid_sidecar=True"
            )
        counts = result.fid_per_category_counts
        # Cumulative sum along the Category axis, then shift right by
        # one column so column ``c`` holds the start offset of the c-th
        # Category's segment (not the end). The first column is always
        # zero (the first Category's segment starts at row offset 0).
        cumulative = np.cumsum(counts, axis=1, dtype=np.uint32)
        bases = np.zeros_like(cumulative)
        bases[:, 1:] = cumulative[:, :-1]
        return cls(
            bases_by_row=bases,
            fid_row_offsets=result.fid_row_offsets,
            fid_sidecar=result.fid_sidecar,
        )

    def lookup(self, row: int, cat: Category, counter: int) -> int:
        """Resolve ``(row, Category, counter_id) -> FID``.

        Reads the row's segment base for ``cat`` from
        :attr:`bases_by_row`, adds the per-row ``fid_row_offsets``
        start to land in the flat :attr:`fid_sidecar`, plus the
        in-segment counter offset. Raises :class:`KeyError` on a
        non-FUNCTION Category (typed contract: the IDENTITY-band
        Category emitter dispatches FUNCTION Categories only).
        """
        col = _CATEGORY_TO_COLUMN.get(cat)
        if col is None:
            raise KeyError(
                f"FidBaseTable.lookup expects a FUNCTION Category "
                f"({FUNCTION_CATEGORIES!r}); got {cat!r}"
            )
        row_start = int(self.fid_row_offsets[row])
        segment_base = int(self.bases_by_row[row, col])
        return int(self.fid_sidecar[row_start + segment_base + counter])
