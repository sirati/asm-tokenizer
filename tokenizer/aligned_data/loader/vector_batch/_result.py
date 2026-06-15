"""The vectorized batch path's typed result shape.

Single concern: the :class:`VectorBatchResult` dataclass -- the one
handoff shape the orchestrator (:mod:`._entry`), the per-arm dispatch
(:mod:`._dispatch`), and the per-arm merge (:mod:`._merge`) all share.
Kept on its own module so those three (which would otherwise form an
import cycle through the result type) each depend only on this leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


__all__ = ["VectorBatchResult"]


@dataclass(frozen=True)
class VectorBatchResult:
    """The vectorized path's full result (backfill OFF).

    Mirrors the ``batch_decode`` ``BatchDecodeResult`` fields the
    byte-identity harness compares: the ``u16[B, L]`` token tensor, the
    ``u32[B, 2]`` ``(section_idx, variant_idx)`` mapping (padding rows
    hold the ``(UINT32_MAX, UINT32_MAX)`` sentinel), AND the DENSE
    sidecars -- the post-remap caller-local->counter identity array (+
    offsets), the ``(significand, sign_exp)`` numeric arrays (+ offsets),
    and the optional per-Category FID sidecars. Every array is
    byte-identical to ``batch_decode`` with backfill off.
    """

    tokens: np.ndarray
    batch_idx_to_section_variant: np.ndarray
    identities: np.ndarray
    identity_row_offsets: np.ndarray
    numbers_significant: np.ndarray
    numbers_sign_exponent: np.ndarray
    number_row_offsets: np.ndarray
    fid_sidecar: Optional[np.ndarray]
    fid_row_offsets: Optional[np.ndarray]
    fid_per_category_counts: Optional[np.ndarray]
