"""Top-level duplicate handling for the per-section length reduction.

Single concern: decide how a section's top-level per-variant lengths
are fed into a :class:`LengthReduction`. Two strategies, selected once
from the ``--adjust-for-duplicates`` flag and threaded as a typed
object -- never re-decided per section:

* :data:`PLAIN` -- every variant is its own item; the lengths go
  straight into :meth:`LengthReduction.reduce`. This is the historical
  (and ``--adjust-for-duplicates`` off) behaviour, byte-for-byte.
* :data:`DEDUP_BY_DATA_POINTER` -- variants sharing a ``data_offset_
  shifted`` form one duplicate-group; each group collapses to one
  representative (per :meth:`LengthReduction.reduce_groups`) before the
  reduction.

Boundary contract (the design-first sentence):

  *Given one section's per-variant lengths + the matching per-variant
  data-bin pointers + a reduction, return the reduced int key -- the
  strategy object owns the grouping decision so the compute never
  branches on the duplicate flag per section.*

The grouping key is ``data_offset_shifted`` (surfaced by the pre-pass);
equality of that field is the "same content" relation. Pointers must be
parallel to lengths (same variant order); the compute supplies both
from the same Stage 2 walk + pre-pass.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ._types import LengthReduction


__all__ = [
    "DuplicateHandling",
    "PLAIN",
    "DEDUP_BY_DATA_POINTER",
]


def _group_by_pointer(
    lengths: np.ndarray, data_pointers: np.ndarray
) -> List[np.ndarray]:
    """Partition ``lengths`` into duplicate-groups keyed on ``data_pointers``.

    Returns one length vector per distinct pointer value. Variant order
    within a group is preserved; group order follows
    :func:`numpy.unique` (ascending pointer value) -- the reduction is
    order-insensitive so the group ordering is immaterial to the result.
    """
    unique_ptrs = np.unique(data_pointers)
    return [lengths[data_pointers == ptr] for ptr in unique_ptrs]


class DuplicateHandling:
    """Typed top-level duplicate-handling strategy.

    Constructed only as the two module-level singletons :data:`PLAIN`
    and :data:`DEDUP_BY_DATA_POINTER`; ``adjusts`` records which one a
    caller holds without exposing the implementation.
    """

    def __init__(self, *, adjusts: bool) -> None:
        self._adjusts = adjusts

    @property
    def adjusts_for_duplicates(self) -> bool:
        """Whether this strategy collapses top-level duplicate-groups."""
        return self._adjusts

    def reduce_section(
        self,
        reduction: LengthReduction,
        *,
        lengths: np.ndarray,
        data_pointers: np.ndarray,
    ) -> int:
        """Reduce one section's per-variant ``lengths`` to a key.

        ``data_pointers`` is parallel to ``lengths`` (one
        ``data_offset_shifted`` per variant). The PLAIN strategy ignores
        the pointers and reduces the flat vector; the dedup strategy
        groups by pointer first.
        """
        if not self._adjusts:
            return reduction.reduce(lengths)
        return reduction.reduce_groups(
            _group_by_pointer(lengths, data_pointers)
        )


#: Historical behaviour: no top-level dedup (``--adjust-for-duplicates``
#: off). Byte-for-byte identical to the pre-feature reduction path.
PLAIN = DuplicateHandling(adjusts=False)

#: ``--adjust-for-duplicates`` on: collapse top-level duplicate-groups
#: (same ``data_offset_shifted``) before the reduction.
DEDUP_BY_DATA_POINTER = DuplicateHandling(adjusts=True)
