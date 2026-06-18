"""Flatten the DFS call_target stream into CSR-shaped flat arrays.

Single concern: given a :class:`Stage2Batch`, gather every level-4
call_target's per-stream numpy fields into FLAT concatenations carrying a
per-call_target CSR (segment offsets), in the SAME DFS encounter order the
stage-3 kernels walk. This is the substrate B-S2 batches over: the
per-call_target Python ``for`` loops in the sign-collection / number-emit /
identity-emit kernels become single vectorised passes over these flat
arrays + a segmented cumsum, instead of one compute pass per call_target.

This module owns ONLY the flattening (array-collection + CSR construction);
it re-implements no decode rule. The carrier identification, byte-offset
arithmetic, and per-:class:`TokenType` emission stay with the owning
kernels -- they consume the flat arrays the same way they consumed the
per-call_target slices, but in one batched pass.

Boundary crossed (design-first sentence): *given the Stage2 DFS call_target
hierarchy, produce the flat per-call_target CSR arrays the stage-3 number /
sign kernels batch over.* The per-call_target compute is the kernels'
concern; this module only laces their inputs into flat carriers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from ._types import Stage2Batch, Stage2CallTarget


__all__ = [
    "FlatCallTargets",
    "flatten_call_targets",
    "surviving_call_targets",
]


@dataclass(frozen=True)
class FlatCallTargets:
    """Flat, CSR-shaped view of every DFS call_target's expanded stream.

    All flat arrays are concatenations over the SURVIVING-prefix slice
    ``expanded_token_ids[:surviving_token_count]`` of each call_target,
    laid out in DFS encounter order. ``seg_offsets`` is the CSR: segment
    ``i`` of the flat arrays spans ``[seg_offsets[i], seg_offsets[i + 1])``
    and corresponds to ``ct_index[i]`` -- the DFS index of the call_target
    that owns it (the position in the full DFS enumeration, including
    fully-dropped call_targets that contribute zero-length segments and so
    are absent from ``ct_index``).

    Only call_targets with ``surviving_token_count > 0`` contribute a
    segment; a fully-dropped call_target carries no carriers and is skipped
    (its DFS index never appears in ``ct_index``).

    Fields
    ------
    expanded_ids:
        ``int64[total_surviving]`` -- concatenated
        ``expanded_token_ids[:surviving]`` (shifted ids).
    is_painted:
        ``bool[total_surviving]`` -- ``extra_value_v2_mask | extra_f128_mask``
        over the same prefix (VC2 / F128 continuation slots).
    seg_offsets:
        ``int64[n_segments + 1]`` -- CSR boundaries over the flat arrays.
    seg_len:
        ``int64[n_segments]`` -- ``diff(seg_offsets)`` (the per-segment
        surviving length); convenience for segment-local arange building.
    ct_index:
        ``int64[n_segments]`` -- the DFS call_target index owning each
        segment.
    """

    expanded_ids: np.ndarray
    is_painted: np.ndarray
    seg_offsets: np.ndarray
    seg_len: np.ndarray
    ct_index: np.ndarray


def surviving_call_targets(
    stage2: "Stage2Batch",
) -> tuple[List["Stage2CallTarget"], np.ndarray]:
    """Enumerate surviving call_targets + their DFS indices.

    Walks ``sections -> variants -> call_targets`` in DFS order (the
    canonical stage-3 linearisation), keeping only call_targets with at
    least one surviving token. Returns the kept call_targets and an
    ``int64`` array of their positions in the FULL DFS enumeration (the
    same ordinal the per-call_target slice lists are keyed by).
    """
    kept: List["Stage2CallTarget"] = []
    kept_idx: List[int] = []
    dfs_idx = 0
    for section in stage2.sections:
        for variant in section.variants:
            for ct in variant.call_targets:
                if int(ct.surviving_token_count) > 0:
                    kept.append(ct)
                    kept_idx.append(dfs_idx)
                dfs_idx += 1
    return kept, np.asarray(kept_idx, dtype=np.int64)


def flatten_call_targets(stage2: "Stage2Batch") -> FlatCallTargets:
    """Build the flat CSR view over surviving call_targets' prefixes.

    The per-call_target gather here is a pure array-collection loop (one
    ``np.concatenate`` at the end), not a per-call_target compute pass:
    the carrier identification + emission the kernels do over these flat
    arrays runs in a SINGLE vectorised pass downstream.
    """
    kept, ct_index = surviving_call_targets(stage2)

    if not kept:
        empty_i = np.empty(0, dtype=np.int64)
        return FlatCallTargets(
            expanded_ids=empty_i,
            is_painted=np.empty(0, dtype=np.bool_),
            seg_offsets=np.zeros(1, dtype=np.int64),
            seg_len=np.empty(0, dtype=np.int64),
            ct_index=ct_index,
        )

    expanded_chunks: List[np.ndarray] = []
    painted_chunks: List[np.ndarray] = []
    seg_len = np.empty(len(kept), dtype=np.int64)
    for i, ct in enumerate(kept):
        surviving = int(ct.surviving_token_count)
        expanded_chunks.append(
            ct.expanded_token_ids[:surviving].astype(np.int64, copy=False)
        )
        painted_chunks.append(
            (
                ct.extra_value_v2_mask[:surviving]
                | ct.extra_f128_mask[:surviving]
            )
        )
        seg_len[i] = surviving

    seg_offsets = np.zeros(len(kept) + 1, dtype=np.int64)
    np.cumsum(seg_len, out=seg_offsets[1:])

    return FlatCallTargets(
        expanded_ids=np.concatenate(expanded_chunks),
        is_painted=np.concatenate(painted_chunks),
        seg_offsets=seg_offsets,
        seg_len=seg_len,
        ct_index=ct_index,
    )
