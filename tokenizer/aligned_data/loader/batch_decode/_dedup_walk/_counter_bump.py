"""COUNTER-Category offset bump (ALG-4).

Single concern: pure offset addition for ONE call_target's in-stream
identity slice for ONE COUNTER Category. No dedup lookup — the offset
is the running total of unique caller-local ids in this Category
across all prior call_targets in the row.

Plan reference: ``batch_decode_plan.md`` ``## Algorithms`` ALG-4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tokenizer.tokens import Category

from ._constants import _CATEGORY_TO_SHIFTED_ID
from ._helpers import _surviving_in_stream_token_ids
from ._row_state import _RowState


if TYPE_CHECKING:
    from .._types import Stage3CallTarget


__all__ = ["_bump_counter_category"]


def _per_call_target_counter_count(
    call_target: "Stage3CallTarget",
    category: Category,
) -> int:
    """Per-call-target unique-id count for a COUNTER Category.

    Plan ALG-4 specifies the value as
    ``call_target.stage2.stage1.function_data.metadata["category_counts"][category]``.
    The metadata key is reserved by the plan but populated by the loader
    (a separate concern); this helper centralizes the lookup so the
    main remap walk stays algorithmic.

    Returns 0 if the function has no ids of this Category. Raises
    ``KeyError`` if the metadata field is absent — that is a loader
    contract violation, not a dedup-walk concern, and surfacing it as a
    typed error from one place keeps the diagnostic crisp.
    """
    metadata = call_target.stage2.stage1.function_data.metadata
    category_counts = metadata["category_counts"]
    return int(category_counts.get(category, 0))


def _bump_counter_category(
    state: _RowState,
    category: Category,
    call_target: "Stage3CallTarget",
    identities_flat: np.ndarray,
) -> None:
    """ALG-4: COUNTER-category offset bump for one call_target.

    Pure offset addition; no dedup lookup. The offset is the running
    total of unique caller-local ids in this Category across all prior
    call_targets in the row.
    """
    offset = state.counter_offset[category]
    per_function_count = _per_call_target_counter_count(call_target, category)

    if per_function_count > 0:
        # Apply offset to in-stream identity positions of this
        # Category. Skip if the offset is zero — the bump would be a
        # no-op.
        if offset > 0:
            in_stream_token_ids = _surviving_in_stream_token_ids(call_target)
            if in_stream_token_ids.size > 0:
                cat_token_id_shifted = np.uint16(
                    _CATEGORY_TO_SHIFTED_ID[category]
                )
                cat_mask = in_stream_token_ids == cat_token_id_shifted
                if cat_mask.any():
                    in_stream_sl = slice(
                        call_target.identity_slice.start + 1,
                        call_target.identity_slice.stop,
                    )
                    target_view = identities_flat[in_stream_sl]
                    target_view[cat_mask] = target_view[cat_mask] + np.uint16(
                        offset
                    )
        state.counter_offset[category] = offset + per_function_count
