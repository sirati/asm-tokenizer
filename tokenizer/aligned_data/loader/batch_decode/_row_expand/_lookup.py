"""Per-row variant lookup (shared step #1 of every downstream concern).

Single concern: resolve the canonical
``Stage1Batch.batch_idx_to_section_variant`` mapping into the per-row flat
variant index + the padding mask, in a single ``np.repeat`` + fancy-index
step (no Python loop over ``batch_size``).

Multi-row mapping (RESAMPLE / REDISTRIBUTE) is handled naturally: each
referencing row reads the same per-variant payload through the same
``per_row_variant_idx`` value downstream.

Padding rows (sentinel ``(UINT32_MAX, UINT32_MAX)`` in the mapping)
contribute zero length and zero flat bytes. Both columns of the mapping
are checked for the sentinel; ALG-10 pairs them together but the
OR-check is the defensible reading of "is this a real mapping entry".

Plan reference: ``batch_decode_plan.md`` ``## Stages -- algorithm
sketch`` Stage 2 (row-offset cumsum) + Stage 4 (per-row sidecar
concatenation).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .._batch_layout import UINT32_MAX


__all__ = ["build_per_row_variant_lookup"]


def build_per_row_variant_lookup(
    batch_idx_to_section_variant: np.ndarray,
    variants_per_section: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized ``(section_idx, slot_idx) -> flat_variant_idx`` lookup.

    Builds the per-row flat variant index plus the padding mask in a
    single ``np.repeat`` + fancy-index step; no Python loop over
    ``batch_size``.

    Parameters
    ----------
    batch_idx_to_section_variant:
        ``u32[batch_size, 2]`` mapping from :class:`Stage1Batch`. Padding
        rows hold ``(UINT32_MAX, UINT32_MAX)`` per plan ALG-10.
    variants_per_section:
        ``len(stage.sections)``-long list of per-section variant counts
        (i.e. ``len(section.variants)`` for each section, in section
        order). Used to build the flat variant offset table; a section
        with zero variants contributes zero to the offset.

    Returns
    -------
    (per_row_variant_idx, is_padding):
        ``per_row_variant_idx`` is ``u32[batch_size]`` -- the flat index
        into a per-unique-variant array (built by following the same
        section -> variant order). Padding rows are clamped to ``0`` so
        the array stays in-bounds; callers MUST mask via ``is_padding``
        before using the value. ``is_padding`` is ``bool[batch_size]``
        -- True where either column of the mapping is the
        ``UINT32_MAX`` sentinel.
    """

    if batch_idx_to_section_variant.ndim != 2 or (
        batch_idx_to_section_variant.shape[1] != 2
    ):
        raise ValueError(
            "batch_idx_to_section_variant must be u32[batch_size, 2]; "
            f"got shape {batch_idx_to_section_variant.shape!r}"
        )

    sentinel = UINT32_MAX
    section_col = batch_idx_to_section_variant[:, 0]
    slot_col = batch_idx_to_section_variant[:, 1]
    is_padding = (section_col == sentinel) | (slot_col == sentinel)

    # Variant offset table: cumulative count of variants up to (but not
    # including) section ``i``. A ``[0]`` prepend keeps ``offset[0] = 0``
    # so ``offset[section_idx] + slot_idx`` lands the right flat
    # variant index for ANY ``(section_idx, slot_idx)`` pair within
    # bounds. Allocated as int64 to keep the running sum safe across
    # very large batches; downcast to u32 after the per-row lookup.
    variant_counts = np.asarray(variants_per_section, dtype=np.int64)
    variant_section_offsets = np.empty(
        variant_counts.shape[0] + 1, dtype=np.int64
    )
    variant_section_offsets[0] = 0
    np.cumsum(variant_counts, out=variant_section_offsets[1:])

    # Clamp padding rows so the fancy-index stays in bounds; the value
    # is irrelevant (masked) but we still need a valid index. Use a
    # safe section idx of 0 for padding rows.
    safe_section = np.where(is_padding, np.uint32(0), section_col).astype(
        np.int64
    )
    safe_slot = np.where(is_padding, np.uint32(0), slot_col).astype(np.int64)
    per_row_variant_idx = (
        variant_section_offsets[safe_section] + safe_slot
    ).astype(np.uint32)
    return per_row_variant_idx, is_padding
