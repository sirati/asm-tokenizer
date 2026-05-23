"""Shared per-variant -> per-row expansion helper.

Single concern: take a per-unique-variant payload (one scalar count per
variant, or one ``np.ndarray`` per variant) plus the canonical
``Stage1Batch.batch_idx_to_section_variant`` mapping, and emit either:

* per-row length cumsum (``u32[batch_size + 1]`` ``row_offsets``), or
* per-row flat concatenation (``flat`` + ``row_offsets``).

Used by four downstream stage modules that all share the same per-row
scalar walk pattern -- pre-vectorization, each of them rolled its own
``for row in range(batch_size)`` loop over the mapping, looking up the
per-variant value at ``sections[section_idx].variants[slot_idx]`` and
accumulating into a flat output. Post-vectorization, every such walk
becomes ``np.repeat`` + ``np.cumsum`` + (optionally) ``np.concatenate``,
with the Python loop reduced to a per-unique-variant outer build of the
per-variant payload (length-bounded by ``num_unique_variants``, NOT
``batch_size``).

Multi-row mapping (RESAMPLE / REDISTRIBUTE) is handled naturally: each
referencing row reads the same per-variant payload through the same
``per_row_variant_idx`` value; the resulting flat output duplicates the
variant's contribution once per referencing row, matching the per-row
``row_offsets`` cumsum.

Padding rows (sentinel ``(UINT32_MAX, UINT32_MAX)`` in the mapping)
contribute zero length and zero flat bytes. Both columns of the mapping
are checked for the sentinel; ALG-10 pairs them together but the
OR-check is the defensible reading of "is this a real mapping entry".

Plan reference: ``batch_decode_plan.md`` ``## Stages -- algorithm
sketch`` Stage 2 (row-offset cumsum) + Stage 4 (per-row sidecar
concatenation).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ._batch_layout import UINT32_MAX


__all__ = [
    "build_per_row_variant_lookup",
    "concat_per_row",
    "row_offsets_from_per_variant_lengths",
]


# ---------------------------------------------------------------------------
# Per-row variant lookup (shared step #1 of every downstream concern).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Length-only mode (row_offsets cumsum without materialising a flat array).
# ---------------------------------------------------------------------------


def row_offsets_from_per_variant_lengths(
    per_variant_lengths: np.ndarray,
    per_row_variant_idx: np.ndarray,
    is_padding: np.ndarray,
) -> np.ndarray:
    """Build ``u32[batch_size + 1]`` cumsum-of-per-row-lengths.

    Parameters
    ----------
    per_variant_lengths:
        ``u32[num_unique_variants]`` -- one entry per unique variant in
        the flat order produced by :func:`build_per_row_variant_lookup`.
        For scalar-count concerns (e.g. surviving identity / number
        counts) this IS the per-variant payload; for array concerns it
        is ``[a.shape[0] for a in per_variant_arrays]``.
    per_row_variant_idx:
        ``u32[batch_size]`` -- output of
        :func:`build_per_row_variant_lookup`.
    is_padding:
        ``bool[batch_size]`` -- output of
        :func:`build_per_row_variant_lookup`. Padding rows contribute
        zero length regardless of the (clamped) variant index.

    Returns
    -------
    np.ndarray
        ``u32[batch_size + 1]``; ``[0]`` is always 0; ``[i + 1] =
        [i] + per_row_length_at_i``.
    """

    batch_size = per_row_variant_idx.shape[0]
    if per_variant_lengths.shape[0] == 0:
        # No variants exist at all -- every row is either padding or
        # has clamped index 0 with no backing entry. Per-row length
        # is uniformly 0; skip the fancy-index to avoid an out-of-bounds
        # access on the empty per_variant_lengths array.
        per_row_lengths = np.zeros(batch_size, dtype=np.uint32)
    else:
        per_row_lengths = np.where(
            is_padding,
            np.uint32(0),
            per_variant_lengths[per_row_variant_idx],
        ).astype(np.uint32, copy=False)

    row_offsets = np.empty(batch_size + 1, dtype=np.uint32)
    row_offsets[0] = 0
    np.cumsum(per_row_lengths, out=row_offsets[1:])
    return row_offsets


# ---------------------------------------------------------------------------
# Flat-concatenation mode (per_variant_arrays -> flat per-row stream).
# ---------------------------------------------------------------------------


def concat_per_row(
    per_variant_arrays: List[np.ndarray],
    per_row_variant_idx: np.ndarray,
    is_padding: np.ndarray,
    *,
    dtype: Optional[np.dtype] = None,
    expected_row_offsets: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized per-row concatenation of per-variant payloads.

    For each non-padding row ``r``, copies
    ``per_variant_arrays[per_row_variant_idx[r]]`` into the flat output
    at the position derived from a cumsum over per-row lengths.
    Multi-mapped variants (the same ``per_row_variant_idx`` value on
    multiple rows) emit their payload once per referencing row.

    Parameters
    ----------
    per_variant_arrays:
        One ``np.ndarray`` per unique variant in flat order (same order
        as :func:`build_per_row_variant_lookup`'s flat variant index).
        Entries may be empty arrays; they contribute zero length. All
        non-empty entries must share a common dtype (defaulting to
        ``dtype`` if provided; otherwise inferred from the first
        non-empty entry).
    per_row_variant_idx:
        ``u32[batch_size]`` from :func:`build_per_row_variant_lookup`.
    is_padding:
        ``bool[batch_size]`` from :func:`build_per_row_variant_lookup`.
    dtype:
        Optional explicit dtype for the output flat array. Used when the
        per-variant arrays might all be empty (and so no inference is
        possible). When given, the per-variant arrays must also conform
        to this dtype.
    expected_row_offsets:
        Optional ``u32[batch_size + 1]`` cumsum-of-per-row-lengths from
        an external sizing pass (e.g. stage 2's
        ``number_row_offsets``). When given, the helper asserts that
        the computed per-row lengths match this expectation exactly
        (per-row, including padding-row zero contributions). The check
        surfaces upstream sizing bugs at the call site rather than as
        silent flat-array under/overrun. Padding-row deltas that are
        non-zero against the expectation raise ``AssertionError`` with
        a "padding row" message; non-padding mismatches raise with a
        "row {idx}" message; both messages mirror the prior per-row
        scalar walks' assertions.

    Returns
    -------
    (flat, row_offsets):
        ``flat`` is a 1D ndarray of dtype matching the per-variant
        payload (or ``dtype`` when given). ``row_offsets`` is
        ``u32[batch_size + 1]``; ``row_offsets[-1] == flat.shape[0]``.
    """

    # ----- Per-variant lengths + per-row length expansion. -----
    per_variant_lengths = np.array(
        [a.shape[0] for a in per_variant_arrays], dtype=np.uint32
    )
    row_offsets = row_offsets_from_per_variant_lengths(
        per_variant_lengths, per_row_variant_idx, is_padding
    )

    if expected_row_offsets is not None:
        _assert_row_offsets_match(
            row_offsets, expected_row_offsets, is_padding
        )

    total = int(row_offsets[-1])

    # ----- Dtype inference + empty fast-path. -----
    out_dtype = dtype
    if out_dtype is None:
        for a in per_variant_arrays:
            if a.shape[0] > 0:
                out_dtype = a.dtype
                break
    if out_dtype is None:
        # All per-variant arrays are empty AND no explicit dtype was
        # provided. There is nothing we can write; return an empty
        # array with the default object dtype only if the caller did
        # not require a specific dtype. The downstream code typically
        # expects a typed empty array (e.g. u32), so this branch is
        # only reachable when the caller knows the output is empty.
        out_dtype = np.uint8

    flat = np.empty(total, dtype=out_dtype)
    if total == 0:
        return flat, row_offsets

    # ----- Vectorized flat fill via np.repeat-expanded source indices. -----
    # The approach: for each row r with variant index v_r and length
    # L_r, the flat output positions [row_offsets[r], row_offsets[r] +
    # L_r) take the per-variant array at v_r in order. We need a
    # ``src_idx[k]`` array of length ``total`` whose entries point into
    # a single concatenated per-variant buffer.
    #
    # Build:
    #   variant_buffer = concatenate(per_variant_arrays)  # len = sum(per_variant_lengths)
    #   variant_start_offset = cumsum-prefix(per_variant_lengths)
    #   per_row_variant_start = variant_start_offset[per_row_variant_idx]  # zeroed for padding
    #   src_idx_per_row_base = repeat(per_row_variant_start, per_row_lengths)  # length = total
    #   src_idx_within_row = arange-per-row-pattern             # length = total
    #   src_idx = src_idx_per_row_base + src_idx_within_row
    #   flat[:] = variant_buffer[src_idx]
    #
    # ``np.repeat`` expands the per-row variant start once per row
    # length, producing a length-``total`` array of "starting source
    # offsets per output position". The within-row offsets ``0, 1, ...,
    # L_r - 1`` for each row are built via ``arange(total) -
    # repeat(row_offsets[:-1], per_row_lengths)`` -- standard trick.

    variant_buffer = np.concatenate(per_variant_arrays) if any(
        a.shape[0] > 0 for a in per_variant_arrays
    ) else np.empty(0, dtype=out_dtype)

    variant_start_offset = np.empty(
        per_variant_lengths.shape[0] + 1, dtype=np.int64
    )
    variant_start_offset[0] = 0
    np.cumsum(per_variant_lengths.astype(np.int64), out=variant_start_offset[1:])

    # Per-row variant start offset; padding rows have per_row_lengths==0
    # so their start value never gets repeated (np.repeat with count 0
    # contributes zero entries), making the clamped value harmless.
    per_row_variant_start = variant_start_offset[per_row_variant_idx]
    per_row_lengths = (row_offsets[1:] - row_offsets[:-1]).astype(np.int64)

    src_base = np.repeat(per_row_variant_start, per_row_lengths)
    # Within-row offsets: for each output position k in [0, total),
    # the within-row offset is k - (row_offsets[r] for the r owning k).
    row_offsets_per_position = np.repeat(
        row_offsets[:-1].astype(np.int64), per_row_lengths
    )
    src_within = np.arange(total, dtype=np.int64) - row_offsets_per_position
    src_idx = src_base + src_within
    flat[:] = variant_buffer[src_idx]
    return flat, row_offsets


# ---------------------------------------------------------------------------
# Internal: row_offsets vs expected sizing check.
# ---------------------------------------------------------------------------


def _assert_row_offsets_match(
    actual: np.ndarray,
    expected: np.ndarray,
    is_padding: np.ndarray,
) -> None:
    """Surface per-row sizing mismatches with the message format the
    prior per-row scalar walks produced.

    Padding rows are checked separately so the error message
    distinguishes "padding row N has non-zero ... delta D" from the
    generic "row N: emitted X but expects Y" -- mirroring the
    per-site assertion strings that pre-vectorization tests pin.
    """

    if actual.shape != expected.shape:
        raise AssertionError(
            f"row_offsets shape mismatch: actual {actual.shape!r} "
            f"vs expected {expected.shape!r}"
        )

    actual_deltas = actual[1:].astype(np.int64) - actual[:-1].astype(np.int64)
    expected_deltas = expected[1:].astype(np.int64) - expected[:-1].astype(
        np.int64
    )

    # Padding rows: expected delta must be 0 (the per-row payload contributes
    # nothing). Surface non-zero expected deltas with the "padding row"
    # message that the pre-vectorization sidecar assembler used.
    padding_mismatch = is_padding & (expected_deltas != 0)
    if padding_mismatch.any():
        first = int(np.argmax(padding_mismatch))
        delta = int(expected_deltas[first])
        raise AssertionError(
            f"padding row {first} has non-zero "
            f"row_offsets delta {delta}"
        )

    # Non-padding rows: emitted-vs-expected mismatch.
    real_mismatch = (~is_padding) & (actual_deltas != expected_deltas)
    if real_mismatch.any():
        first = int(np.argmax(real_mismatch))
        emitted = int(actual_deltas[first])
        expected_n = int(expected_deltas[first])
        raise AssertionError(
            f"row {first}: emitted {emitted} chunks but "
            f"row_offsets expects {expected_n}"
        )
