"""Per-row flat concatenation of per-variant payloads.

Single concern: scatter per-unique-variant payloads to a per-row flat
stream via the :func:`build_per_row_variant_lookup` mapping + the per-row
length cumsum. Two input shapes feed the SAME vectorised ``np.repeat`` /
``arange`` fill core (:func:`_concat_per_row_core`):

* :func:`concat_per_row` -- a ``list`` of one ``np.ndarray`` per variant.
* :func:`concat_per_row_from_buffer` -- one already-concatenated
  variant-ordered ``variant_buffer`` + its per-variant CSR offsets (the
  global per-chunk stream's native shape; avoids a redundant re-slice +
  ``np.concatenate`` round-trip).

Multi-row mapping (RESAMPLE / REDISTRIBUTE) is handled naturally: each
referencing row reads the same per-variant payload through the same
``per_row_variant_idx`` value; the resulting flat output duplicates the
variant's contribution once per referencing row, matching the per-row
``row_offsets`` cumsum. Padding rows contribute zero length / zero bytes.

Plan reference: ``batch_decode_plan.md`` ``## Stages -- algorithm
sketch`` Stage 4 (per-row sidecar concatenation).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ._sizing import row_offsets_from_per_variant_lengths


__all__ = [
    "concat_per_row",
    "concat_per_row_from_buffer",
]


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

    # ----- Per-variant lengths + per-variant buffer/CSR. -----
    per_variant_lengths = np.array(
        [a.shape[0] for a in per_variant_arrays], dtype=np.uint32
    )

    # Dtype inference must run on the list (the buffer-input entry takes a
    # ready-typed buffer instead) before concatenation collapses it.
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

    variant_buffer = np.concatenate(per_variant_arrays) if any(
        a.shape[0] > 0 for a in per_variant_arrays
    ) else np.empty(0, dtype=out_dtype)

    variant_offsets = np.empty(
        per_variant_lengths.shape[0] + 1, dtype=np.int64
    )
    variant_offsets[0] = 0
    np.cumsum(per_variant_lengths.astype(np.int64), out=variant_offsets[1:])

    return _concat_per_row_core(
        variant_buffer,
        variant_offsets,
        per_variant_lengths,
        per_row_variant_idx,
        is_padding,
        out_dtype=out_dtype,
        expected_row_offsets=expected_row_offsets,
    )


def concat_per_row_from_buffer(
    variant_buffer: np.ndarray,
    variant_offsets: np.ndarray,
    per_row_variant_idx: np.ndarray,
    is_padding: np.ndarray,
    *,
    dtype: Optional[np.dtype] = None,
    expected_row_offsets: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row concatenation from an ALREADY-concatenated variant buffer.

    Byte-identical to :func:`concat_per_row`, but the caller supplies the
    variant payloads as one flat ``variant_buffer`` (1D) grouped in
    variant order, plus its per-variant CSR ``variant_offsets`` (``int``;
    ``variant_offsets[v + 1] - variant_offsets[v]`` is variant ``v``'s
    length). This is the form the global per-chunk stream already has --
    ``concat_per_row`` would otherwise re-slice the buffer into a list
    and ``np.concatenate`` it straight back, a redundant round-trip.

    Parameters mirror :func:`concat_per_row`; ``variant_buffer`` +
    ``variant_offsets`` replace ``per_variant_arrays``. ``dtype`` overrides
    the output dtype (defaults to ``variant_buffer.dtype``).
    """
    variant_offsets = np.ascontiguousarray(variant_offsets, dtype=np.int64)
    per_variant_lengths = (
        variant_offsets[1:] - variant_offsets[:-1]
    ).astype(np.uint32)
    out_dtype = dtype if dtype is not None else variant_buffer.dtype

    return _concat_per_row_core(
        variant_buffer,
        variant_offsets,
        per_variant_lengths,
        per_row_variant_idx,
        is_padding,
        out_dtype=out_dtype,
        expected_row_offsets=expected_row_offsets,
    )


def _concat_per_row_core(
    variant_buffer: np.ndarray,
    variant_start_offset: np.ndarray,
    per_variant_lengths: np.ndarray,
    per_row_variant_idx: np.ndarray,
    is_padding: np.ndarray,
    *,
    out_dtype: np.dtype,
    expected_row_offsets: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared tail: scatter a variant buffer to per-row flat output.

    Single source of truth for the vectorised ``np.repeat`` / ``arange``
    per-row fill used by both the list-input (:func:`concat_per_row`) and
    buffer-input (:func:`concat_per_row_from_buffer`) entries. Given the
    per-variant lengths, the variant-ordered ``variant_buffer``, and its
    prefix-sum CSR ``variant_start_offset``, produces ``(flat,
    row_offsets)`` identical to the original monolithic body.
    """
    # ----- Per-row length expansion + optional sizing check. -----
    row_offsets = row_offsets_from_per_variant_lengths(
        per_variant_lengths, per_row_variant_idx, is_padding
    )

    if expected_row_offsets is not None:
        _assert_row_offsets_match(
            row_offsets, expected_row_offsets, is_padding
        )

    total = int(row_offsets[-1])

    flat = np.empty(total, dtype=out_dtype)
    if total == 0:
        return flat, row_offsets

    # ----- Vectorized flat fill via np.repeat-expanded source indices. -----
    # The approach: for each row r with variant index v_r and length
    # L_r, the flat output positions [row_offsets[r], row_offsets[r] +
    # L_r) take the per-variant array at v_r in order. We need a
    # ``src_idx[k]`` array of length ``total`` whose entries point into
    # the single variant-ordered ``variant_buffer``.
    #
    # Build:
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
