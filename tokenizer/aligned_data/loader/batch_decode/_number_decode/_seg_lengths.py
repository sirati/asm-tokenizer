"""Per-segment lengths from a CSR base array.

Single concern: turn a CSR ``seg_base`` (the start offset of each segment
in a flat concatenation) plus the total flat length into the per-segment
length, so a ``+1`` lookahead can be bounds-guarded against the carrier's
OWN segment rather than bleeding into a neighbour's segment (or off the
end of the flat array entirely).

Shared by the F128 and VC2 emitters, which both read a per-carrier
``expanded_position + 1`` lookahead out of a per-segment flat
concatenation and must clip it to the owning segment.
"""

from __future__ import annotations

import numpy as np


__all__ = ["seg_lengths_from_base"]


def seg_lengths_from_base(seg_base: np.ndarray, flat_len: int) -> np.ndarray:
    """``len[s]`` = number of flat entries owned by segment ``s``.

    ``seg_base`` is the CSR start offset of each segment in the flat
    concatenation (``seg_base[0] == 0``); ``flat_len`` is the total flat
    length. The length of segment ``s`` is ``seg_base[s + 1] -
    seg_base[s]`` with the implicit final boundary at ``flat_len``.
    """
    return np.diff(
        np.concatenate(
            [seg_base, np.array([flat_len], dtype=np.int64)]
        )
    )
