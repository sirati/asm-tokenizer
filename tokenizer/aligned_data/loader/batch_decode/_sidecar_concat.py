"""Stage 4 helper -- per-row sidecar concatenation.

Single concern of this module: take a finalised :class:`Stage3Batch` and
produce the flat numbers sidecar pair the model consumer wants:

Numbers sidecar: ``(numbers_significant: u64[total_chunks],
numbers_sign_exponent: u32[total_chunks])`` indexed by
:attr:`Stage2Batch.number_row_offsets`. Chunks are concatenated in
row-major stream order -- the per-:class:`TokenType`
``(significand, sign_exp)`` chunk pairs that stage 3 staged in
``Stage3Batch.numbers_per_TokenType`` are interleaved exactly as the
surviving ``expanded_token_ids[:partial_cut_length]`` stream
discovers them.

What this module is NOT:

- It does not write into the token tensor or the identity sidecar --
  those concerns live in subagents 4a/4c.
- It does not renormalise floats -- the ``(significand, sign_exp)``
  arrays come pre-normalised from stage 3 (ALG-7).

Plan reference: ``batch_decode_plan.md`` Stage 4 step 4; D8 pins the
row-offset sizing contract.

Multi-row mapping note (RESAMPLE_WITHIN_SECTION / REDISTRIBUTE): when
one :class:`Stage1Variant` is referenced by multiple batch rows,
:attr:`Stage1Batch.batch_idx_to_section_variant` carries the same
``(section_idx, variant_idx)`` pair on each row. This module resolves
each batch row's source variant through that mapping, so a single
variant's chunks are emitted once per referencing row -- matching the
sizes already cumsum'd into ``stage2.number_row_offsets``.

Batched (B-S2b): the prior per-variant -> per-call_target Python chunk
walk is replaced by a SINGLE pass over the flat
``expanded[:partial_cut_length]`` stream of every call_target in the
batch (DFS encounter order). The per-call_target per-:class:`TokenType`
stream-order scatter becomes a segmented gather: each surviving
number-band slot's source index is ``number_chunk_slices[T][ct].start +
rank-within-(ct, T)``, and a per-type boolean gather fills the global
stream in one shot. The per-variant buffers are then contiguous slices
of that global stream (DFS order groups call_targets by variant), fed to
the shared :func:`._row_expand.concat_per_row` for the variant -> row
expansion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

from tokenizer.tokens import TokenType

from ._row_expand import build_per_row_variant_lookup, concat_per_row

if TYPE_CHECKING:
    from ._types import Stage3Batch


__all__ = [
    "assemble_number_sidecars",
]


# ---------------------------------------------------------------------------
# Vocab constants (post-shift NUMBER block -- plan vocab table + Stage 2 ALG)
# ---------------------------------------------------------------------------


# Shifted ids 1..7 cover the NUMBER block (originals 257..263) in
# source-declaration order: VC2, F16, BF16, F32, F64, F80, F128. The
# original ids are NOT contiguous on :class:`TokenType` (VC2=17, F16=19,
# ...) -- the unified vocab v1 lays them out contiguously starting at 257
# per ``_V2_NUMBER_BLOCK_START`` (see plan "Vocab + wire format
# reference"). After the post-shift (``- 256``) they sit at 1..7.
_SHIFTED_NUMBER_BAND_LO = 1
_SHIFTED_NUMBER_BAND_HI = 8

# Canonical NUMBER-block ordering, indexed by ``shifted_id - 1``
# (block 0 = VC2 = shifted id 1, ..., block 6 = F128 = shifted id 7).
_NUMBER_BLOCK_TOKEN_TYPES: tuple[TokenType, ...] = (
    TokenType.VALUED_CONST_V2,
    TokenType.FLOAT16,
    TokenType.BFLOAT16,
    TokenType.FLOAT32,
    TokenType.FLOAT64,
    TokenType.FLOAT80,
    TokenType.FLOAT128,
)


# ---------------------------------------------------------------------------
# Numbers sidecar
# ---------------------------------------------------------------------------


def _build_global_chunk_stream(
    stage3_batch: "Stage3Batch",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the flat per-chunk ``(significand, sign_exp)`` stream over
    every call_target in DFS encounter order, plus per-variant CSR.

    Walks ``sections -> variants -> call_targets`` once, concatenating
    each call_target's surviving number-band ids
    (``expanded[:partial_cut_length]`` filtered to the band) into a flat
    ``out_block`` stream (block index = ``shifted_id - 1``), tagged with
    each chunk's owning call_target ordinal. The per-chunk SOURCE index
    into ``numbers_per_TokenType[T]`` is then
    ``number_chunk_slices[T][ct].start + rank-within-(ct, T)`` -- a
    per-(call_target, type) segmented arange. A single per-type boolean
    gather fills the global ``(sig, sex)`` stream.

    Returns ``(sig_flat, sex_flat, variant_chunk_offsets)`` where
    ``variant_chunk_offsets`` is the CSR over the global stream keyed by
    unique-variant flat index (DFS order groups call_targets by variant,
    so each variant owns a contiguous run).
    """
    numbers_per_TokenType = stage3_batch.numbers_per_TokenType

    # --- flat per-chunk concatenation over DFS call_targets -----------
    block_chunks: List[np.ndarray] = []
    # Per-chunk source-slice base = the owning call_target's per-type
    # slice start (selected per chunk by its block). Built per type as a
    # flat parallel array so the per-type gather is a single fancy-index.
    slice_start_per_block_chunks: List[np.ndarray] = []
    ct_ordinal_chunks: List[np.ndarray] = []
    # Per-variant chunk counts (CSR over the global stream).
    variant_chunk_counts: List[int] = []

    ct_ordinal = 0
    for stage3_section in stage3_batch.sections:
        for stage3_variant in stage3_section.variants:
            variant_chunks = 0
            for call_target in stage3_variant.call_targets:
                stage2_ct = call_target.stage2
                pcl = int(stage2_ct.partial_cut_length)
                if pcl == 0:
                    ct_ordinal += 1
                    continue
                surviving = stage2_ct.expanded_token_ids[:pcl]
                in_band = (surviving >= _SHIFTED_NUMBER_BAND_LO) & (
                    surviving < _SHIFTED_NUMBER_BAND_HI
                )
                n = int(in_band.sum())
                if n == 0:
                    ct_ordinal += 1
                    continue
                block = (
                    surviving[in_band].astype(np.int64)
                    - _SHIFTED_NUMBER_BAND_LO
                )
                block_chunks.append(block)
                # Per-block slice-start for THIS call_target: the start of
                # ``number_chunk_slices[T]`` for each block (0 when the
                # call_target carries no slice for that type -- such a
                # block never appears among its chunks).
                ct_slice_start = np.zeros(
                    len(_NUMBER_BLOCK_TOKEN_TYPES), dtype=np.int64
                )
                for b, token_type in enumerate(_NUMBER_BLOCK_TOKEN_TYPES):
                    per_type_slice = call_target.number_chunk_slices.get(
                        token_type
                    )
                    if per_type_slice is not None:
                        ct_slice_start[b] = int(per_type_slice.start)
                slice_start_per_block_chunks.append(ct_slice_start[block])
                ct_ordinal_chunks.append(
                    np.full(n, ct_ordinal, dtype=np.int64)
                )
                variant_chunks += n
                ct_ordinal += 1
            variant_chunk_counts.append(variant_chunks)

    n_variants = len(variant_chunk_counts)
    variant_chunk_offsets = np.zeros(n_variants + 1, dtype=np.int64)
    if n_variants:
        np.cumsum(
            np.asarray(variant_chunk_counts, dtype=np.int64),
            out=variant_chunk_offsets[1:],
        )

    if not block_chunks:
        return (
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=np.uint32),
            variant_chunk_offsets,
        )

    out_block = np.concatenate(block_chunks)
    slice_start = np.concatenate(slice_start_per_block_chunks)
    ct_ordinal_flat = np.concatenate(ct_ordinal_chunks)
    total = int(out_block.shape[0])

    # rank-within-(call_target, block): the chunks are in (ct, stream)
    # order, so for a fixed block the type-T subsequence is ordered by
    # call_target; the per-(ct, block) rank is the within-segment arange
    # where a segment = a maximal run of identical (ct_ordinal, block).
    # Build it by a group-key reset: a new segment starts where either
    # the ct ordinal or the block changes vs the previous chunk.
    rank = _segmented_rank(ct_ordinal_flat, out_block)
    src_idx = slice_start + rank

    sig_flat = np.empty(total, dtype=np.uint64)
    sex_flat = np.empty(total, dtype=np.uint32)
    for b, token_type in enumerate(_NUMBER_BLOCK_TOKEN_TYPES):
        type_mask = out_block == b
        if not type_mask.any():
            continue
        sig_for_type, sex_for_type = numbers_per_TokenType[token_type]
        src = src_idx[type_mask]
        sig_flat[type_mask] = sig_for_type[src]
        sex_flat[type_mask] = sex_for_type[src]

    return sig_flat, sex_flat, variant_chunk_offsets


def _segmented_rank(
    ct_ordinal: np.ndarray, block: np.ndarray
) -> np.ndarray:
    """Per-(call_target, block) rank in stream order.

    ``rank[k]`` = how many earlier chunks share chunk ``k``'s
    ``(ct_ordinal, block)`` pair AND precede it in the stream. The global
    stream is in (call_target, stream-position) order, so each
    call_target owns a CONTIGUOUS run of chunks, but WITHIN a
    call_target the per-:class:`TokenType` blocks interleave freely
    (e.g. ``F32 F64 F32``). The rank must therefore count same-block
    chunks across the interleaving, reset at each call_target boundary.

    Computed per block as an inclusive cumcount with the call_target's
    carry-in subtracted: ``cumsum(block == b)`` minus the cumulative
    value carried in at the call_target's first chunk gives the
    per-(ct, block) 0, 1, 2, ... sequence. The call_target run boundaries
    are where ``ct_ordinal`` changes (the stream is contiguous per ct).
    """
    total = int(ct_ordinal.shape[0])
    if total == 0:
        return np.empty(0, dtype=np.int64)

    # First flat index of each call_target run (ct_ordinal change points).
    ct_run_start = np.empty(total, dtype=bool)
    ct_run_start[0] = True
    ct_run_start[1:] = ct_ordinal[1:] != ct_ordinal[:-1]
    # For each chunk, the flat index of its call_target's first chunk.
    arange = np.arange(total, dtype=np.int64)
    ct_first_idx = np.maximum.accumulate(
        np.where(ct_run_start, arange, np.int64(-1))
    )

    rank = np.empty(total, dtype=np.int64)
    n_blocks = len(_NUMBER_BLOCK_TOKEN_TYPES)
    for b in range(n_blocks):
        is_b = block == b
        if not is_b.any():
            continue
        # Inclusive cumcount of block ``b`` over the whole stream, then
        # subtract the exclusive cumcount carried in at the ct's first
        # chunk to reset per call_target.
        cum_incl = np.cumsum(is_b.astype(np.int64))
        cum_excl = cum_incl - is_b.astype(np.int64)
        carry_in = cum_excl[ct_first_idx]
        rank[is_b] = (cum_incl - 1 - carry_in)[is_b]
    return rank


def assemble_number_sidecars(
    stage3_batch: "Stage3Batch",
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-:class:`TokenType` ``(significand, sign_exp)``
    chunks into the global row-major number sidecars.

    Vectorized via a single batched stream build + the shared
    :mod:`._row_expand` variant -> row expansion: the global per-chunk
    stream is built once over every call_target (no Python loop over
    ``batch_size`` or per-:class:`TokenType` re-scatter per call_target),
    sliced per unique variant, then scattered to batch rows by the shared
    lookup + concat primitive. Padding rows (sentinel
    ``(UINT32_MAX, UINT32_MAX)``) contribute zero chunks; their expected
    ``number_row_offsets`` delta is checked against zero by the helper.

    Multi-row mapping: when one :class:`Stage1Variant` is referenced
    by multiple batch rows (RESAMPLE / REDISTRIBUTE), the variant's
    chunks are emitted once per referencing row -- stage 2's
    :attr:`Stage2Batch.number_row_offsets` already accounts for the
    duplication when sizing the output.

    Returns ``(numbers_significant: u64[total_chunks],
    numbers_sign_exponent: u32[total_chunks])``; total length equals
    ``int(stage2_batch.number_row_offsets[-1])``.
    """

    stage2_batch = stage3_batch.stage2
    stage1_batch = stage2_batch.stage1
    number_row_offsets = stage2_batch.number_row_offsets

    sig_flat, sex_flat, variant_chunk_offsets = _build_global_chunk_stream(
        stage3_batch
    )

    # Per-unique-variant ``(sig, sex)`` slices of the global stream, in
    # flat ``section -> slot`` order (the same order DFS produced them).
    variants_per_section: List[int] = []
    per_variant_sig: List[np.ndarray] = []
    per_variant_sex: List[np.ndarray] = []
    v = 0
    for stage3_section in stage3_batch.sections:
        variants_per_section.append(len(stage3_section.variants))
        for _stage3_variant in stage3_section.variants:
            lo = int(variant_chunk_offsets[v])
            hi = int(variant_chunk_offsets[v + 1])
            per_variant_sig.append(sig_flat[lo:hi])
            per_variant_sex.append(sex_flat[lo:hi])
            v += 1

    per_row_variant_idx, is_padding = build_per_row_variant_lookup(
        stage1_batch.batch_idx_to_section_variant, variants_per_section
    )

    sig_out, _ = concat_per_row(
        per_variant_sig,
        per_row_variant_idx,
        is_padding,
        dtype=np.dtype(np.uint64),
        expected_row_offsets=number_row_offsets,
    )
    sex_out, _ = concat_per_row(
        per_variant_sex,
        per_row_variant_idx,
        is_padding,
        dtype=np.dtype(np.uint32),
        expected_row_offsets=number_row_offsets,
    )
    return sig_out, sex_out
