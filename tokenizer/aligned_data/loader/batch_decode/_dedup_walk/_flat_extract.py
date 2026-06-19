"""Object-tree -> flat-int-array extractor for the Rust remap kernel.

Single concern: flatten the referenced :class:`Stage3Variant` subtree
into the flat int arrays the ``dedup_hashmap.apply_remap_walk`` kernel
consumes, in the SAME emission order the Python dedup walk visited
``call_targets``. This is an ADAPTER -- it carries NONE of the ALG-3/4/9
remap logic (which lives only in the Rust kernel); it only reads the
per-call-target fields the walk read (``surviving_token_count``,
``encounter_category``, ``function_name_ptr``, ``identity_slice``,
``call_targets_section``, the surviving in-stream identity-band token
ids, and ``metadata['category_counts']``) and re-expresses them as int
columns.

Boundary crossed (design-first sentence): *given the referenced-variant
subtree + the batch-shared ``identities_flat_caller_local`` array, emit
the per-node + per-call-target + per-in-stream-position flat int arrays
keyed by the kernel's dense FUNCTION/COUNTER slot codes.*

The kernel groups nodes into rows by a contiguous ``node_row`` run; this
extractor emits one row per referenced variant (in section -> slot
order) and one node per ``call_target`` in encounter order (INCLUDING
``surviving_token_count == 0`` call_targets -- the kernel skips them but
the root seed reads ``call_targets[0]`` regardless, so the node sequence
must be complete). Returns the flat arrays plus the per-row
``(section_idx, slot_idx)`` keys so the caller can rebuild the
per-variant FID inverse from the kernel's per-row output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tokenizer.tokens import Category

from ._constants import (
    COUNTER_CATEGORIES,
    FUNCTION_CATEGORIES,
    _CALL_TARGET_TYPE_TO_CATEGORY,
    _CATEGORY_TO_SHIFTED_ID,
    _COUNTER_CATEGORY_TO_SLOT,
    _FUNCTION_CATEGORY_TO_SLOT,
)
from ._helpers import _surviving_in_stream_token_ids


if TYPE_CHECKING:
    from .._types import Stage3Batch, Stage3CallTarget


__all__ = ["FlatRemapInputs", "extract_flat_remap_inputs"]


# Per-FUNCTION-category shifted token id (the in-stream IDENTITY-band id
# that signals a token of that Category). Resolved once.
_FUNC_SHIFTED_TO_SLOT: dict[int, int] = {
    _CATEGORY_TO_SHIFTED_ID[cat]: _FUNCTION_CATEGORY_TO_SLOT[cat]
    for cat in FUNCTION_CATEGORIES
}
_COUNTER_SHIFTED_TO_SLOT: dict[int, int] = {
    _CATEGORY_TO_SHIFTED_ID[cat]: _COUNTER_CATEGORY_TO_SLOT[cat]
    for cat in COUNTER_CATEGORIES
}

# Per-``CallTargetType`` FUNCTION slot (call-target rows are always a
# FUNCTION category). Built off the shared type->Category map so a layout
# change reshapes it automatically.
_CT_TYPE_TO_FUNC_SLOT: dict[object, int] = {
    ct_type: _FUNCTION_CATEGORY_TO_SLOT[cat]
    for ct_type, cat in _CALL_TARGET_TYPE_TO_CATEGORY.items()
}


@dataclass(frozen=True)
class FlatRemapInputs:
    """The flat int arrays + per-row keys feeding the remap kernel.

    All ``node_*`` arrays are length ``n_nodes`` in emission order; rows
    (variants) are contiguous runs of equal ``node_row``. ``ct_off`` /
    ``instream_off`` are CSR offsets of length ``n_nodes + 1``.
    """

    n_rows: int
    node_row: np.ndarray  # i64[n_nodes]
    node_skip: np.ndarray  # bool[n_nodes]
    node_prepend_pos: np.ndarray  # i64[n_nodes]
    node_fid: np.ndarray  # i64[n_nodes]
    node_enc_func_slot: np.ndarray  # i64[n_nodes]
    ct_off: np.ndarray  # i64[n_nodes + 1]
    ct_fid: np.ndarray  # i64[n_ct]
    ct_func_slot: np.ndarray  # i64[n_ct]
    instream_off: np.ndarray  # i64[n_nodes + 1]
    instream_func_slot: np.ndarray  # i64[n_instream]
    instream_counter_slot: np.ndarray  # i64[n_instream]
    counter_counts: np.ndarray  # i64[n_nodes, n_counter]
    #: ``(section_idx, slot_idx)`` for each emitted row, in row order.
    row_keys: list[tuple[int, int]]


def _instream_slots(
    call_target: "Stage3CallTarget",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-in-stream-position FUNCTION + COUNTER slot columns.

    Parallel to the call_target's in-stream identity slice
    (``identity_slice`` minus the prepend slot). Each position's IDENTITY-
    band shifted token id maps to EITHER a FUNCTION slot (``func_slot``,
    COUNTER column ``-1``) or a COUNTER slot (``counter_slot``, FUNCTION
    column ``-1``); the two columns are disjoint by construction (the
    IDENTITY-band partition). Returns ``(func_slot, counter_slot)`` i64
    arrays of equal length.
    """
    token_ids = _surviving_in_stream_token_ids(call_target)
    n = int(token_ids.size)
    func_slot = np.full(n, -1, dtype=np.int64)
    counter_slot = np.full(n, -1, dtype=np.int64)
    for shifted, slot in _FUNC_SHIFTED_TO_SLOT.items():
        func_slot[token_ids == np.uint16(shifted)] = slot
    for shifted, slot in _COUNTER_SHIFTED_TO_SLOT.items():
        counter_slot[token_ids == np.uint16(shifted)] = slot
    return func_slot, counter_slot


def _category_counts_column(call_target: "Stage3CallTarget") -> np.ndarray:
    """Per-COUNTER-category count row for one call_target (i64[n_counter]).

    Reads ``function_data.metadata['category_counts']`` -- the same field
    ALG-4 read -- and projects it onto the kernel's COUNTER slot order.
    """
    metadata = call_target.stage2.stage1.function_data.metadata
    category_counts = metadata["category_counts"]
    return np.asarray(
        [int(category_counts.get(cat, 0)) for cat in COUNTER_CATEGORIES],
        dtype=np.int64,
    )


def extract_flat_remap_inputs(
    stage3_batch: "Stage3Batch",
) -> FlatRemapInputs:
    """Flatten the referenced-variant subtree into the kernel's arrays.

    Walks ``stage3_batch.sections -> variants`` in section -> slot order,
    skipping variants whose ``stage1.batch_idx is None`` (not referenced
    by any batch row -- identical to the Python walk's pass-1 skip). For
    each referenced variant emits one row; for each ``call_target`` in
    encounter order emits one node with its prepend position, fid,
    encounter slot, call-target section, in-stream slot columns, and
    COUNTER count row.
    """
    node_row: list[int] = []
    node_skip: list[bool] = []
    node_prepend_pos: list[int] = []
    node_fid: list[int] = []
    node_enc_func_slot: list[int] = []

    ct_off: list[int] = [0]
    ct_fid: list[int] = []
    ct_func_slot: list[int] = []

    instream_off: list[int] = [0]
    instream_func_pieces: list[np.ndarray] = []
    instream_counter_pieces: list[np.ndarray] = []

    counter_rows: list[np.ndarray] = []
    row_keys: list[tuple[int, int]] = []

    row_idx = 0
    for section_idx, stage3_section in enumerate(stage3_batch.sections):
        for slot_idx, stage3_variant in enumerate(stage3_section.variants):
            if stage3_variant.stage2.stage1.batch_idx is None:
                continue
            row_keys.append((section_idx, slot_idx))
            for call_target in stage3_variant.call_targets:
                stage1 = call_target.stage2.stage1
                node_row.append(row_idx)
                node_skip.append(
                    call_target.stage2.surviving_token_count == 0
                )
                node_prepend_pos.append(int(call_target.identity_slice.start))
                node_fid.append(int(stage1.function_name_ptr))
                node_enc_func_slot.append(
                    _FUNCTION_CATEGORY_TO_SLOT[stage1.encounter_category]
                )

                # Call-target section in section order -> (fid, func slot).
                for ct in stage1.call_targets_section:
                    ct_fid.append(int(ct.function_name_ptr))
                    ct_func_slot.append(_CT_TYPE_TO_FUNC_SLOT[ct.type])
                ct_off.append(len(ct_fid))

                # In-stream identity slot columns.
                func_col, counter_col = _instream_slots(call_target)
                instream_func_pieces.append(func_col)
                instream_counter_pieces.append(counter_col)
                instream_off.append(instream_off[-1] + int(func_col.size))

                counter_rows.append(_category_counts_column(call_target))
            row_idx += 1

    n_counter = len(COUNTER_CATEGORIES)
    counter_counts = (
        np.stack(counter_rows, axis=0)
        if counter_rows
        else np.zeros((0, n_counter), dtype=np.int64)
    )
    instream_func_slot = (
        np.concatenate(instream_func_pieces)
        if instream_func_pieces
        else np.zeros(0, dtype=np.int64)
    )
    instream_counter_slot = (
        np.concatenate(instream_counter_pieces)
        if instream_counter_pieces
        else np.zeros(0, dtype=np.int64)
    )
    return FlatRemapInputs(
        n_rows=row_idx,
        node_row=np.asarray(node_row, dtype=np.int64),
        node_skip=np.asarray(node_skip, dtype=np.bool_),
        node_prepend_pos=np.asarray(node_prepend_pos, dtype=np.int64),
        node_fid=np.asarray(node_fid, dtype=np.int64),
        node_enc_func_slot=np.asarray(node_enc_func_slot, dtype=np.int64),
        ct_off=np.asarray(ct_off, dtype=np.int64),
        ct_fid=np.asarray(ct_fid, dtype=np.int64),
        ct_func_slot=np.asarray(ct_func_slot, dtype=np.int64),
        instream_off=np.asarray(instream_off, dtype=np.int64),
        instream_func_slot=instream_func_slot.astype(np.int64, copy=False),
        instream_counter_slot=instream_counter_slot.astype(
            np.int64, copy=False
        ),
        counter_counts=counter_counts,
        row_keys=row_keys,
    )
