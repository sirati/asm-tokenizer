"""Stage 4 sub-concern: per-row VariantPadding sentinel enforcement.

This module owns ONE concern: the row-level interpretation of the
``batch_idx_to_section_variant`` sentinel marker produced by stage 1's
:func:`_batch_layout.compute_batch_idx_mapping`. Stage 4's per-row loops
(``_assemble`` + friends) consult these helpers to short-circuit padding
rows so their tokens stay id ``0`` (null-content) and their sidecar
offsets stay equal to the prior offset (zero-length slice).

Per :class:`VariantPadding` policy (plan ALG-10):

* ``PAD_NULL``: padding rows hold ``(UINT32_MAX, UINT32_MAX)`` -- this
  module's helpers detect them.
* ``RESAMPLE_WITHIN_SECTION`` / ``REDISTRIBUTE``: stage 1 already filled
  the deficit slots; every row maps to a real variant; the helpers
  unconditionally return ``True`` / a real :class:`Stage1Variant`.
* ``RAGGED``: ``batch_size == total_real_variants``; no padding rows;
  same as above.

The sentinel value is the single source of truth from
:mod:`._batch_layout`; re-exported here only for symmetry with the
``is_padding_row`` predicate.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ._batch_layout import UINT32_MAX
from ._types import Stage1Section, Stage1Variant


__all__ = [
    "get_real_row_mask",
    "is_padding_row",
    "resolve_row_to_variant",
]


def is_padding_row(
    batch_idx_to_section_variant: np.ndarray,
    row: int,
) -> bool:
    """Return ``True`` iff ``batch_idx_to_section_variant[row]`` is the
    ``(UINT32_MAX, UINT32_MAX)`` sentinel.

    Used by stage-4 per-row loops to short-circuit padding rows so their
    tokens stay id ``0`` and their sidecar offsets stay equal to the
    prior offset (zero-length slice).
    """

    # Reading only the section-idx column is sufficient: stage 1 always
    # writes the pair atomically, so ``section_idx == UINT32_MAX`` iff
    # ``slot_idx == UINT32_MAX``. See ``_batch_layout`` for the writer
    # contract.
    return bool(batch_idx_to_section_variant[row, 0] == UINT32_MAX)


def get_real_row_mask(
    batch_idx_to_section_variant: np.ndarray,
) -> np.ndarray:
    """Return a ``bool[batch_size]`` mask: ``True`` iff the row is real
    (NOT a padding sentinel).

    Vectorized helper for sidecar-size computation:
    ``real_row_mask.sum()`` gives the count of rows that contribute real
    content.
    """

    # Same contract as :func:`is_padding_row`: the section-idx column
    # alone identifies sentinel rows.
    return batch_idx_to_section_variant[:, 0] != UINT32_MAX


def resolve_row_to_variant(
    batch_idx_to_section_variant: np.ndarray,
    sections: List[Stage1Section],
    row: int,
) -> Optional[Stage1Variant]:
    """Return the :class:`Stage1Variant` for ``row``, or ``None`` when
    the row is a padding sentinel.

    Equivalent to::

        section_idx, slot_idx = batch_idx_to_section_variant[row]
        if section_idx == UINT32_MAX:
            return None
        return sections[section_idx].variants[slot_idx]
    """

    section_idx = batch_idx_to_section_variant[row, 0]
    if section_idx == UINT32_MAX:
        return None
    slot_idx = batch_idx_to_section_variant[row, 1]
    return sections[int(section_idx)].variants[int(slot_idx)]
