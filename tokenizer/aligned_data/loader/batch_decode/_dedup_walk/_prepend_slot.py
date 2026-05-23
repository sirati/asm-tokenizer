"""Prepend slot writes (ALG-9) + LOCAL_FUNC root seed (ALG-3).

Single concern: write the per-call-target prepend slot's
``self_counter`` into ``identities_flat_caller_local`` (ALG-9), and
seed the LOCAL_FUNC dedup map with the row's root function (ALG-3 +
ALG-9 boundary). The dedup map at ``encounter_category`` already
holds the counter when the prepend write fires — either the row-level
seed (for the root) or the parent call_target's ALG-3 walk (for
inlined callees) put it there.

Plan reference: ``batch_decode_plan.md`` ``## Algorithms`` ALG-3 +
ALG-9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dedup_hashmap import HashMapU32U16

from tokenizer.tokens import Category

from ._row_state import _RowState


if TYPE_CHECKING:
    from .._types import Stage3CallTarget


__all__ = ["_seed_local_func_root", "_write_prepend_slot"]


def _seed_local_func_root(
    state: _RowState,
    local_map: HashMapU32U16,
    root_fn_name_ptr: int,
) -> None:
    """Per ALG-3, ALG-9: LOCAL_FUNC root takes counter id 0.

    The dedup map gets the root FID -> 0 entry; ``next_fresh_id`` for
    LOCAL_FUNC advances to 1 so subsequent fresh LOCAL_FUNC ids start
    at 1. PLT_FUNC and EXT_FUNC remain empty with ``next_fresh_id`` = 0.
    """
    key = np.uint32(root_fn_name_ptr)
    local_map.insert(key, np.uint16(0))
    state.next_fresh_id[Category.LOCAL_FUNC] = 1
    if state.fid_inverse is not None:
        state.fid_inverse[Category.LOCAL_FUNC].append(root_fn_name_ptr)


def _write_prepend_slot(
    dedup_maps: dict[Category, HashMapU32U16],
    call_target: "Stage3CallTarget",
    identities_flat: np.ndarray,
) -> None:
    """ALG-9: write the prepend slot's self-counter.

    The slot at ``identity_slice.start`` carries the call_target's own
    counter in its ``encounter_category`` space. For the root that is
    0 (seeded at row start). For inlined callees the counter was
    minted by the parent's dedup walk when the callee's FID first
    appeared as a call_target row of its category — so the dedup map
    for ``encounter_category`` already holds it.

    The token id at the prepend position (in ``tokens[row, col]``) is
    written by the 4b prepend stage, NOT here. This function owns
    ONLY the counter id at ``identities_flat_caller_local``.
    """
    stage1 = call_target.stage2.stage1
    dedup_map = dedup_maps[stage1.encounter_category]
    self_counter = dedup_map.lookup(np.uint32(stage1.function_name_ptr))
    if self_counter is None:
        # The callee's FID was not in the parent's call_targets_section
        # — that is a stage-1 walker invariant violation (every inlined
        # callee MUST have been an entry in its parent's call_targets
        # table; that is how stage 1 picked the call_target to inline).
        # Surface as a typed AssertionError so the diagnostic points at
        # the upstream concern rather than producing a wrong identity.
        raise AssertionError(
            "callee prepend slot has no minted self-counter for "
            f"(encounter_category={stage1.encounter_category!r}, "
            f"function_name_ptr={stage1.function_name_ptr}); the "
            "stage-1 walker must inline only callees whose FID lives "
            "in the parent's call_targets table for the matching "
            "Category."
        )
    identities_flat[call_target.identity_slice.start] = np.uint16(self_counter)
