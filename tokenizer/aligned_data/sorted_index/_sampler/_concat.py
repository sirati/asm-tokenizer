"""Cross-binary :class:`BatchDecodeResult` concatenation (plan ALG-6).

:func:`_concat_results` stitches per-binary
:class:`BatchDecodeResult` instances into one
:class:`MultiBinaryBatchDecodeResult`, re-basing per-row cumsums via
:func:`_concat_row_offsets` and stamping a per-row ``binary_id``
sidecar.

The caller MUST supply a list ordered by alphabetical ``binary_name``
(the canonical order
:class:`~tokenizer.aligned_data.sorted_index.MultiBinarySortedIndexSampler`
exposes); :func:`open_length_bucketed_batch` enforces this.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
)

from .._types import (
    MultiBinaryBatchDecodeResult,
    PerBinaryDecodeResult,
)


__all__ = [
    "_concat_results",
    "_concat_row_offsets",
]


def _concat_row_offsets(offsets_list: List[np.ndarray]) -> np.ndarray:
    """Concatenate per-batch cumsum offset arrays into one global cumsum.

    Each input is ``u32[batch_size_i + 1]`` with ``offsets_i[0] == 0``.
    Output is ``u32[sum(batch_size_i) + 1]`` re-based so offsets are
    cumulative across the concatenated batch.

    Computation happens in ``uint64`` so the re-base addition cannot
    overflow when individual offsets are near the u32 limit; the
    final cast back to ``uint32`` is safe under the assumption (held
    by the existing single-binary stages) that no single batch's
    cumulative count exceeds u32 range.
    """
    if not offsets_list:
        raise ValueError("_concat_row_offsets: empty input")
    pieces: List[np.ndarray] = [offsets_list[0].astype(np.uint64)]
    running = int(offsets_list[0][-1])
    for offsets in offsets_list[1:]:
        # Drop offsets[0] (== 0) when appending; re-base by running total.
        pieces.append(
            offsets[1:].astype(np.uint64) + np.uint64(running),
        )
        running += int(offsets[-1])
    return np.concatenate(pieces).astype(np.uint32)


def _concat_results(
    per_binary: List[PerBinaryDecodeResult],
) -> MultiBinaryBatchDecodeResult:
    """Stitch per-binary :class:`PerBinaryDecodeResult`s into one combined result.

    ``per_binary`` MUST be sorted by alphabetical ``binary_name`` (the
    canonical order :class:`MultiBinarySortedIndexSampler` exposes); the
    caller :func:`open_length_bucketed_batch` enforces this. Each entry's
    ``depth_per_row`` is concatenated row-wise into the combined
    ``depth_per_row`` exactly as ``binary_id_per_row`` is built.

    Invariants enforced here:

    * **Shape precondition (plan D-2.1)**: every per-binary
      ``tokens.shape[1]`` must equal the first's; otherwise raise a
      clear :class:`ValueError` rather than letting
      :func:`numpy.concatenate` raise a cryptic shape mismatch.
    * **Row offsets** are re-based via :func:`_concat_row_offsets` so
      the stitched cumsums are globally monotone.
    * **batch_idx_to_section_variant** is NOT re-numbered; the new
      ``binary_id_per_row`` sidecar carries cross-binary identity.
    * **fid sidecar all-or-none**: every per-binary result must
      either supply :attr:`BatchDecodeResult.fid_sidecar` or none of
      them must; mixed inputs raise :class:`ValueError`.
    * **Intermediate is dropped**: the per-binary
      :class:`Stage3Batch` is single-binary-scoped and cannot be
      stitched cross-binary; the returned inner result's
      :attr:`BatchDecodeResult.intermediate` is always ``None``. The
      caller is expected to set ``keep_intermediate=False`` in
      :func:`open_length_bucketed_batch`.
    """
    if not per_binary:
        raise ValueError("_concat_results: empty input")

    # Re-expose the typed per-binary entries as the (name, result) view the
    # cross-binary stitch below operates on; the parallel per-row depth is
    # concatenated alongside (mirroring ``binary_id_per_row``).
    name_result = [(pb.binary_name, pb.result) for pb in per_binary]
    depth_per_row = np.concatenate(
        [np.asarray(pb.depth_per_row, dtype=np.int64) for pb in per_binary]
    )

    # Shape precondition (plan D-2.1).
    context_len = name_result[0][1].tokens.shape[1]
    mismatched = [
        name for name, r in name_result if r.tokens.shape[1] != context_len
    ]
    if mismatched:
        raise ValueError(
            "_concat_results: tokens.shape[1] mismatch across binaries: "
            f"first sees {context_len}, diverging: {mismatched}",
        )

    binary_names = [name for name, _ in name_result]
    name_to_id = {name: i for i, name in enumerate(binary_names)}

    # Token tensor: stack rows.
    tokens = np.concatenate([r.tokens for _, r in name_result], axis=0)

    # Identities + per-row offsets.
    identities = np.concatenate([r.identities for _, r in name_result])
    identity_offsets = _concat_row_offsets(
        [r.identity_row_offsets for _, r in name_result],
    )

    # Numbers + per-row offsets.
    sig = np.concatenate(
        [r.numbers_significant for _, r in name_result],
    )
    sgnexp = np.concatenate(
        [r.numbers_sign_exponent for _, r in name_result],
    )
    number_offsets = _concat_row_offsets(
        [r.number_row_offsets for _, r in name_result],
    )

    # batch_idx_to_section_variant: stack as-is; binary_id sidecar
    # below carries cross-binary identity.
    btv = np.concatenate(
        [r.batch_idx_to_section_variant for _, r in name_result], axis=0,
    )

    # binary_id_per_row: repeat the binary's id once per row.
    binary_id_per_row = np.concatenate([
        np.full(r.tokens.shape[0], name_to_id[name], dtype=np.uint32)
        for name, r in name_result
    ])

    # Optional fid sidecar (all-or-none across inputs).
    fid_present = [r.fid_sidecar is not None for _, r in name_result]
    fid_sidecar: Optional[np.ndarray] = None
    fid_offsets: Optional[np.ndarray] = None
    fid_per_category_counts: Optional[np.ndarray] = None
    if all(fid_present):
        fid_sidecar = np.concatenate(
            [r.fid_sidecar for _, r in name_result],
        )
        fid_offsets = _concat_row_offsets(
            [r.fid_row_offsets for _, r in name_result],
        )
        # Per-row per-Category counts are dense per-row (no cumsum
        # rebase needed); stack along axis 0.
        fid_per_category_counts = np.concatenate(
            [r.fid_per_category_counts for _, r in name_result], axis=0,
        )
    elif any(fid_present):
        raise ValueError(
            "_concat_results: include_fid_sidecar inconsistent across inputs",
        )

    # Optional metatoken-runlength sidecars (all-or-none across inputs).
    # Mirrors the fid_sidecar pattern: per-binary all-or-none, flat
    # array stacked + row offsets re-based.
    block_rl_present = [r.block_runlength is not None for _, r in name_result]
    block_runlength: Optional[np.ndarray] = None
    block_runlength_row_offsets: Optional[np.ndarray] = None
    insn_runlength: Optional[np.ndarray] = None
    insn_runlength_row_offsets: Optional[np.ndarray] = None
    if all(block_rl_present):
        block_runlength = np.concatenate(
            [r.block_runlength for _, r in name_result],
        )
        block_runlength_row_offsets = _concat_row_offsets(
            [r.block_runlength_row_offsets for _, r in name_result],
        )
        insn_runlength = np.concatenate(
            [r.insn_runlength for _, r in name_result],
        )
        insn_runlength_row_offsets = _concat_row_offsets(
            [r.insn_runlength_row_offsets for _, r in name_result],
        )
    elif any(block_rl_present):
        raise ValueError(
            "_concat_results: emit_block_n_insns_runlength inconsistent "
            "across inputs",
        )

    inner = BatchDecodeResult(
        tokens=tokens,
        identities=identities,
        identity_row_offsets=identity_offsets,
        numbers_significant=sig,
        numbers_sign_exponent=sgnexp,
        number_row_offsets=number_offsets,
        batch_idx_to_section_variant=btv,
        fid_sidecar=fid_sidecar,
        fid_row_offsets=fid_offsets,
        fid_per_category_counts=fid_per_category_counts,
        block_runlength=block_runlength,
        block_runlength_row_offsets=block_runlength_row_offsets,
        insn_runlength=insn_runlength,
        insn_runlength_row_offsets=insn_runlength_row_offsets,
        intermediate=None,
    )
    return MultiBinaryBatchDecodeResult(
        inner=inner,
        binary_id_per_row=binary_id_per_row,
        binary_names=binary_names,
        depth_per_row=depth_per_row,
    )
