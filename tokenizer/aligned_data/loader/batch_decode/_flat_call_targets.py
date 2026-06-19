"""Flatten the DFS call_target stream into CSR-shaped flat arrays.

Single concern: given a :class:`Stage2Batch`, walk every level-4
call_target ONCE in DFS encounter order and surface each one's per-stream
numpy column views + scalar metadata, in the SAME order the stage-3
dense-byte-stream kernels walk. This is the one shared collection the
stage-3 sites (inline-byte concat, identity carriers + slices, number
carriers, sign collection, chunk-slice reconstruction) all consume
instead of each re-walking ``sections -> variants -> call_targets`` and
re-grabbing the same overlapping per-call_target columns.

This module owns ONLY the walk + column surfacing (the per-call_target
``state`` field views + scalar metadata); it re-implements no decode rule.
The carrier identification, byte-offset arithmetic, per-segment CSR
construction, and per-:class:`TokenType` emission stay with the owning
kernels -- they consume the shared columns the same way they consumed the
per-call_target ``ct`` / ``ct.stage1.state`` reads, but the DFS walk runs
once.

Boundary crossed (design-first sentence): *given the Stage2 DFS call_target
hierarchy, produce the per-call_target column views + scalar metadata (and
the flat CSR view over surviving prefixes) the stage-3 dense-byte-stream
sites consume.* What each site does with those columns (concat, carrier
mask, segmented cumsum, slice math) is the site's concern; this module
only walks once and surfaces the columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List

import numpy as np

if TYPE_CHECKING:
    from ._types import Stage2Batch, Stage2CallTarget


__all__ = [
    "CallTargetColumns",
    "FlatCallTargets",
    "flatten_call_targets",
    "iter_call_target_columns",
    "surviving_call_targets",
]


@dataclass(frozen=True)
class CallTargetColumns:
    """Per-call_target column views + scalar metadata (DFS order).

    One instance per level-4 call_target in the canonical stage-3 DFS
    enumeration (``sections -> variants -> call_targets``, root-first),
    INCLUDING fully-dropped (``surviving_token_count == 0``) targets so
    the ``dfs_index`` aligns 1:1 with the per-DFS-call_target slice lists
    (``inline_byte_slices`` etc.). Every ``np.ndarray`` field is a VIEW
    into the batched expansion's flat arrays (no copy) -- the same view
    the per-call_target site read off ``ct`` / ``ct.stage1.state``.

    A site that processes only surviving call_targets filters on
    ``surviving_token_count > 0``; a site that processes every target
    (inline bytes, identity slices) reads every record.

    Fields
    ------
    dfs_index:
        Position in the full DFS enumeration (the ordinal the
        per-DFS-call_target slice lists are keyed by).
    call_target:
        The owning :class:`Stage2CallTarget` (back-pointer for the few
        scalar reads not surfaced as columns, e.g. ``is_cut``).
    expanded_token_ids / extra_value_v2_mask / extra_f128_mask:
        Expanded-space columns (parallel to ``expanded_token_ids``).
    raw_tokens / real_mask / number_mask / runlen_number / digit_cumsum /
    is_negative_per_position:
        Raw-space :class:`InlineDecodeState` column views.
    surviving_token_count / surviving_identity_count / partial_cut_length:
        The cutoff-aware scalar counts.
    is_cut:
        Whether the per-row cutoff falls inside this call_target's body.
    """

    dfs_index: int
    call_target: "Stage2CallTarget"

    expanded_token_ids: np.ndarray
    extra_value_v2_mask: np.ndarray
    extra_f128_mask: np.ndarray

    raw_tokens: np.ndarray
    real_mask: np.ndarray
    number_mask: np.ndarray
    runlen_number: np.ndarray
    digit_cumsum: np.ndarray
    is_negative_per_position: np.ndarray

    surviving_token_count: int
    surviving_identity_count: int
    partial_cut_length: int
    is_cut: bool


def iter_call_target_columns(
    stage2: "Stage2Batch",
) -> Iterator[CallTargetColumns]:
    """Yield one :class:`CallTargetColumns` per level-4 call_target.

    The SINGLE shared DFS walk the stage-3 dense-byte-stream sites consume
    instead of each re-walking ``sections -> variants -> call_targets``.
    Yields in DFS encounter order, INCLUDING fully-dropped targets, so a
    consumer's enumeration index equals the ``dfs_index`` the
    per-DFS-call_target slice lists are keyed by.

    Each yielded record carries the per-call_target column VIEWS (no copy)
    + scalar metadata -- the same fields the per-call_target loops read off
    ``ct`` / ``ct.stage1.state``. The records are produced lazily; a
    consumer that needs a list materialises one with ``list(...)``.
    """
    dfs_index = 0
    for section in stage2.sections:
        for variant in section.variants:
            for ct in variant.call_targets:
                state = ct.stage1.state
                yield CallTargetColumns(
                    dfs_index=dfs_index,
                    call_target=ct,
                    expanded_token_ids=ct.expanded_token_ids,
                    extra_value_v2_mask=ct.extra_value_v2_mask,
                    extra_f128_mask=ct.extra_f128_mask,
                    raw_tokens=state.raw_tokens,
                    real_mask=state.real_mask,
                    number_mask=state.number_mask,
                    runlen_number=state.runlen_number,
                    digit_cumsum=state.digit_cumsum,
                    is_negative_per_position=state.is_negative_per_position,
                    surviving_token_count=int(ct.surviving_token_count),
                    surviving_identity_count=int(
                        ct.surviving_identity_count
                    ),
                    partial_cut_length=int(ct.partial_cut_length),
                    is_cut=bool(ct.is_cut),
                )
                dfs_index += 1


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
    for cols in iter_call_target_columns(stage2):
        if cols.surviving_token_count > 0:
            kept.append(cols.call_target)
            kept_idx.append(cols.dfs_index)
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
