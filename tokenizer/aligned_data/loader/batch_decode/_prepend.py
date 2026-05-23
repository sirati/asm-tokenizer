"""Stage 4b -- per-call_target prepend slot writes (ALG-9).

Single concern: emit ONE call_target's synthetic self-prepend into the two
batch-shared tensors. Two writes:

1. ``tokens[row, column]`` <- the calling-category's shifted vocab id
   (``LOCAL_FUNC=9`` or ``PLT_FUNC=10``; ``EXT_FUNC`` is rejected per plan D3:
   externs are never inlined and therefore never get a body to prepend).
2. ``identities_flat_caller_local[identity_slice_start]`` <- the
   caller-supplied dedup counter that THIS call_target's
   ``function_name_ptr`` resolves to in its calling category's space.

The dedup counter lives in the PARENT'S category-space (D3 + ALG-9 design
note): for the root call_target the counter is ``0`` (LOCAL_FUNC's
self-reservation seeded at row start per D4); for an inlined callee the
counter was assigned during the parent's ALG-3 dedup walk when the
callee's FID was first encountered. This module never computes that
counter -- the orchestrator (stage 4c) looks it up from its per-row
dedup state and passes it in.

Why this is its own module
--------------------------
Stage 3 reserves the prepend slot at ``identity_slice.start`` but
deliberately does NOT populate it via the view-cast bulk decode (see plan
ALG-9 rationale: "the prepend's caller-local id lives in the parent's
call_targets space, not the current function's, so applying the current
function's remap_lookup would be wrong"). Stage 4 writes it directly.
The actual write is two array stores; everything else (which row, which
column, which counter, which category) is the orchestrator's concern.
Isolating those two stores here keeps the orchestrator free of vocab-id
arithmetic and keeps the vocab-id mapping in ONE place that re-derives
from the :class:`VocabularyManager` source of truth.

Vocab id source of truth
------------------------
The shifted token ids are derived at import time from
:class:`tokenizer.token_manager.VocabularyManager` (the same anchor used
by ``_surviving_counts.py``). Concretely:

* ``_LOCAL_FUNC_SHIFTED = 265 - 256 = 9``
* ``_PLT_FUNC_SHIFTED   = 266 - 256 = 10``

Both are positions in the canonical IDENTITY block (D5 + vocab layout
docstring in :class:`VocabularyManager`); shifting by the reserved-digit
count moves them out of the wire-format digit band and into the
post-shift token id range used by the decoded ``tokens`` tensor.

See :doc:`batch_decode_plan` ``ALG-9`` for the full design rationale.
"""

from __future__ import annotations

import numpy as np

from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category


__all__ = [
    "write_prepend_slot",
]


# ---------------------------------------------------------------------------
# Shifted vocab ids -- derived from the VocabularyManager source of truth.
#
# The IDENTITY block starts at ``_V2_IDENTITY_BLOCK_START`` (= 264 -- the
# ``BLOCK_V2`` slot). Per :class:`VocabularyManager`'s ``_create_inner_classes``
# canonical layout (vocab id assignments at construction time):
#
#     264 = BLOCK_V2, 265 = LOCAL_FUNC, 266 = PLT_FUNC, 267 = EXT_FUNC, ...
#
# So ``LOCAL_FUNC`` lives at ``_V2_IDENTITY_BLOCK_START + 1`` and
# ``PLT_FUNC`` at ``_V2_IDENTITY_BLOCK_START + 2``. The post-shift token id
# subtracts ``_V2_RESERVED_DIGIT_COUNT`` (= 256) per the post-promotion strip
# rule used everywhere downstream (D2 + Stage 2's strip+shift step).
# ---------------------------------------------------------------------------
_LOCAL_FUNC_SHIFTED = (
    VocabularyManager._V2_IDENTITY_BLOCK_START
    + 1
    - VocabularyManager._V2_RESERVED_DIGIT_COUNT
)  # = 264 + 1 - 256 = 9
_PLT_FUNC_SHIFTED = (
    VocabularyManager._V2_IDENTITY_BLOCK_START
    + 2
    - VocabularyManager._V2_RESERVED_DIGIT_COUNT
)  # = 264 + 2 - 256 = 10


# Category -> shifted self-prepend token id. EXT_FUNC is deliberately
# absent: per plan D3 + ALG-9 EXT_FUNC has no inlined body and therefore
# can never be the calling category of a prepended call_target.
_CATEGORY_TO_SHIFTED_TOKEN: dict[Category, int] = {
    Category.LOCAL_FUNC: _LOCAL_FUNC_SHIFTED,
    Category.PLT_FUNC: _PLT_FUNC_SHIFTED,
}


def write_prepend_slot(
    tokens: np.ndarray,
    identities_flat_caller_local: np.ndarray,
    *,
    row: int,
    column: int,
    identity_slice_start: int,
    encounter_category: Category,
    self_counter: int,
) -> None:
    """Write ONE call_target's prepend slot per ALG-9.

    In-place mutates both arrays. The two writes are:

    * ``tokens[row, column] = SHIFTED[encounter_category]``
    * ``identities_flat_caller_local[identity_slice_start] = self_counter``

    Parameters
    ----------
    tokens:
        ``u16[batch_size, context_len]`` output token tensor. Mutated at
        ``(row, column)``.
    identities_flat_caller_local:
        ``u16[N]`` flat identity sidecar (the stage-3 array; pre-remap
        caller-local). Mutated at ``identity_slice_start``.
    row:
        Output-tensor row for this call_target (the variant's
        ``batch_idx``).
    column:
        The prepend slot's column position in ``tokens[row]``. For the
        root call_target this is ``0`` (the row's first non-skipped
        column); for an inlined callee it is the running column offset
        produced by the orchestrator's row-walk (the sum of prior
        surviving call_targets' lengths in this row). This module does
        NOT compute the position -- the orchestrator (stage 4c) drives
        the loop and supplies it.
    identity_slice_start:
        Position in ``identities_flat_caller_local``. Per stage-3
        allocation, ``identity_slice.start`` IS the prepend slot for
        this call_target (length = 1 + surviving_in_stream_identity_count).
    encounter_category:
        The calling category of THIS call_target per plan D3 +
        ``Stage1CallTarget.encounter_category``: :attr:`Category.LOCAL_FUNC`
        for the root and LOCAL-inlined callees;
        :attr:`Category.PLT_FUNC` for PLT-inlined callees. Passing
        :attr:`Category.EXT_FUNC` (or any other category) raises
        :class:`AssertionError` -- externs have no inlined body and
        therefore can never appear as a call_target's encounter category.
    self_counter:
        The dedup counter that this call_target's ``function_name_ptr``
        resolves to in its calling category's space (see plan ALG-9 for
        derivation). For the root call_target this is ``0`` (LOCAL_FUNC's
        seeded self-reservation per D4). For an inlined callee the
        orchestrator looks it up from the parent's per-row dedup map
        (set during the parent's ALG-3 dedup walk).

    Notes
    -----
    Both writes use ``np.uint16`` to match the dtypes of the two arrays.
    The function is idempotent under identical arguments -- repeated
    invocation with the same ``(row, column, identity_slice_start,
    encounter_category, self_counter)`` leaves the same final values in
    the two arrays.

    See :doc:`batch_decode_plan` ``ALG-9`` for the full design.
    """

    # Map calling category -> shifted token id. The assert message
    # spells out the D3 invariant so a future regression (e.g. a stage-1
    # walker accidentally tagging an EXT_FUNC call_target as inlined)
    # surfaces with the rule it broke, not just a KeyError.
    assert encounter_category in _CATEGORY_TO_SHIFTED_TOKEN, (
        f"prepend slot requires a calling-category with an inlined body "
        f"(LOCAL_FUNC or PLT_FUNC) per plan D3 + ALG-9; got "
        f"{encounter_category!r}. EXT_FUNC has no body to inline and "
        f"therefore can never be a call_target's encounter_category."
    )
    shifted_token_id = _CATEGORY_TO_SHIFTED_TOKEN[encounter_category]

    # Two in-place stores. Casting through np.uint16 here -- not at the
    # caller -- means the orchestrator passes plain ints and this module
    # owns the dtype contract with the two arrays.
    tokens[row, column] = np.uint16(shifted_token_id)
    identities_flat_caller_local[identity_slice_start] = np.uint16(self_counter)
