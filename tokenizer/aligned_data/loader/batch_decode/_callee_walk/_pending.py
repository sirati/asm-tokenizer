"""Pending-row payload + finalisation for the batched callee walk.

Single concern: the data shape of a DFS-emitted call_target row BEFORE
its :class:`InlineDecodeState` is constructed, plus the
mask-construction helper that stages the two :func:`run_lengths` calls
on a shared :class:`BucketedRunLengthCollector`, plus the post-flush
finalisation step that turns pending rows into
:class:`Stage1CallTarget` instances.

The pending pattern is what makes the batched amortisation work: the
DFS produces N pending rows + 2N collector handles in one pass; the
caller flushes the collector once (one 2D ``run_lengths`` per pow2
bucket); :func:`finalise_pending_call_targets` finishes the cheap
``is_negative_per_position`` + ``digit_cumsum`` computations per row
and constructs the frozen :class:`InlineDecodeState`.

Per-row mask construction is intentionally NOT batched: the masks are
single ``uint16`` comparisons (cheap), and batching them would require
allocating a global tokens buffer. The expensive part is the
``run_lengths`` dispatch, which IS batched -- everything else lives in
the cheap per-row path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from tokenizer.aligned_data.loader.decoded._bucketed_run_lengths import (
    BucketedRunLengthCollector,
)
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    InlineDecodeState,
    compute_is_negative_per_position,
)
from tokenizer.aligned_data.matched_sections_bin import CallTarget
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category

from .._types import Stage1CallTarget

if TYPE_CHECKING:  # pragma: no cover -- type-only
    from tokenizer.aligned_data.loader.function_data import FunctionData


__all__ = [
    "PendingCallTarget",
    "build_pending_call_target",
    "finalise_pending_call_targets",
]


# Wire-stream layout constants -- mirrors the locals in
# :mod:`_inline_decode_state`. The vectorized mask construction below
# pins the same constants so the batched path produces byte-identical
# masks to the single-call :func:`build_inline_decode_state` route.
_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT
_V2_VALUE_NEGATIVE_TOKEN_ID = VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID
_V2_EAGER_BLOCK_END = VocabularyManager._V2_EAGER_BLOCK_END


@dataclass(frozen=True)
class PendingCallTarget:
    """One DFS-emitted call_target row WITHOUT its :class:`InlineDecodeState`.

    The :func:`walk_callees_pending` walker emits these rows so the
    caller can batch every row's ``run_lengths`` work into ONE 2D
    dispatch via :class:`BucketedRunLengthCollector`. The masks that
    feed the run-length pass are pre-computed here (they are cheap:
    single uint16 comparisons), and the two collector handles point at
    the pending ``runlen_number`` / ``runlen_value`` arrays.

    :func:`finalise_pending_call_targets` consumes the collector's
    :meth:`flush` result + these pending rows and produces the final
    list of :class:`Stage1CallTarget` objects.
    """

    function_data: "FunctionData"
    raw_tokens: np.ndarray
    real_mask: np.ndarray
    number_mask: np.ndarray
    value_mask: np.ndarray
    carries_inline_mask: np.ndarray
    runlen_number_handle: int
    runlen_value_handle: int

    call_targets_section: List[CallTarget]
    encounter_category: Category
    parent_call_target_index: Optional[int]
    function_name_ptr: int


def build_pending_call_target(
    *,
    function_data: "FunctionData",
    raw_tokens: np.ndarray,
    call_targets_section: List[CallTarget],
    encounter_category: Category,
    parent_call_target_index: Optional[int],
    function_name_ptr: int,
    collector: BucketedRunLengthCollector,
) -> PendingCallTarget:
    """Compute the cheap per-position masks + stage the two
    :func:`run_lengths` calls on the collector.

    The masks (``real_mask``, ``number_mask``, ``value_mask``,
    ``carries_inline_mask``) are single uint16 comparisons -- cheap to
    do per row. The two run-length passes are the expensive part; both
    get staged via :meth:`BucketedRunLengthCollector.add` and the
    handles are stored on the pending row for finalisation.
    """
    real_mask = raw_tokens > _V2_VALUE_NEGATIVE_TOKEN_ID
    number_mask = raw_tokens < _V2_RESERVED_DIGIT_COUNT
    # ``value_mask`` covers numbers AND the optional postfix sign
    # marker; equal to ``~real_mask`` under strict ``> 256``.
    value_mask = ~real_mask
    carries_inline_mask = real_mask & (raw_tokens < _V2_EAGER_BLOCK_END)

    runlen_number_handle = collector.add(number_mask)
    runlen_value_handle = collector.add(value_mask)

    return PendingCallTarget(
        function_data=function_data,
        raw_tokens=raw_tokens,
        real_mask=real_mask,
        number_mask=number_mask,
        value_mask=value_mask,
        carries_inline_mask=carries_inline_mask,
        runlen_number_handle=runlen_number_handle,
        runlen_value_handle=runlen_value_handle,
        call_targets_section=call_targets_section,
        encounter_category=encounter_category,
        parent_call_target_index=parent_call_target_index,
        function_name_ptr=function_name_ptr,
    )


def finalise_pending_call_targets(
    pending: List[PendingCallTarget],
    runlen_results: dict[int, np.ndarray],
) -> List[Stage1CallTarget]:
    """Turn :class:`PendingCallTarget` rows into :class:`Stage1CallTarget`.

    Reads the two run-length arrays per row from ``runlen_results`` (the
    output of :meth:`BucketedRunLengthCollector.flush`), computes
    ``is_negative_per_position`` + ``digit_cumsum`` from the cached
    masks, and constructs the frozen :class:`InlineDecodeState` +
    :class:`Stage1CallTarget` for each row.

    Pure transformation -- no IO, no further numpy heavy lifting per
    row. The expensive run-length work was already amortized across
    pow2 buckets during ``flush``.
    """
    out: List[Stage1CallTarget] = []
    for row in pending:
        runlen_number = runlen_results[row.runlen_number_handle]
        runlen_value = runlen_results[row.runlen_value_handle]

        is_negative_per_position = compute_is_negative_per_position(
            runlen_number=runlen_number,
            runlen_value=runlen_value,
            carries_inline_mask=row.carries_inline_mask,
        )

        # Exclusive-prefix cumsum of ``number_mask``. Same construction
        # as the single-call ``build_inline_decode_state`` path -- the
        # operand view-cast to ``uint8`` keeps the accumulator small.
        n = int(row.raw_tokens.shape[0])
        digit_cumsum = np.zeros(n + 1, dtype=np.uint32)
        if n > 0:
            np.cumsum(row.number_mask.view(np.uint8), out=digit_cumsum[1:])

        state = InlineDecodeState(
            raw_tokens=row.raw_tokens,
            real_mask=row.real_mask,
            number_mask=row.number_mask,
            runlen_number=runlen_number,
            runlen_value=runlen_value,
            carries_inline_mask=row.carries_inline_mask,
            is_negative_per_position=is_negative_per_position,
            digit_cumsum=digit_cumsum,
        )
        out.append(
            Stage1CallTarget(
                function_data=row.function_data,
                state=state,
                call_targets_section=row.call_targets_section,
                encounter_category=row.encounter_category,
                parent_call_target_index=row.parent_call_target_index,
                function_name_ptr=row.function_name_ptr,
            )
        )
    return out
