"""Byte-identity gate for the stage-3 carrier-sign Rust kernel port.

Compares ``_batched_carrier_signs`` (now kernel-backed) against the
ORIGINAL numpy reference (reproduced verbatim below as
``_reference_batched_carrier_signs``) over a fuzz of synthetic
:class:`DenseColumns`, exercising the shape spread the task calls out:
cuts (surviving==1 / fully-cut / mid-cut), every NUMBER band kind, bare
identity slots, dropped call_targets, and multi-node DFS order.

Any ``np.array_equal`` divergence (block_idx OR signs) is a BUG.
"""

from __future__ import annotations

import sys

import numpy as np

from number_kernel_byte_identity import _build_batch, _gen_ct

from tokenizer.aligned_data.loader.batch_decode._bulk_bytes import (
    _NUMBER_BAND_HI_SHIFTED,
    _NUMBER_BAND_LO_SHIFTED,
    _batched_carrier_signs,
)
from tokenizer.aligned_data.loader.batch_decode._dense_columns import (
    DenseColumns,
)
from tokenizer.aligned_data.loader.batch_decode._flat_call_targets import (
    dense_columns_from_stage2,
)


def _reference_batched_carrier_signs(
    dense: DenseColumns,
) -> tuple[np.ndarray, np.ndarray]:
    """The pre-port numpy reference (verbatim from HEAD ea79d57)."""
    kept_idx = np.asarray(dense.kept_node_index, dtype=np.int64).tolist()
    n_kept = len(kept_idx)
    if n_kept == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.bool_))

    expanded_chunks: list[np.ndarray] = []
    painted_chunks: list[np.ndarray] = []
    real_pos_chunks: list[np.ndarray] = []
    is_neg_chunks: list[np.ndarray] = []
    exp_seg_len = np.empty(n_kept, dtype=np.int64)
    real_seg_base = np.empty(n_kept, dtype=np.int64)

    real_running = 0
    for i, e in enumerate(kept_idx):
        surviving = int(dense.surviving_token_count[e])
        raw_slice = dense.node_raw_slice(e)
        expanded_slice = dense.node_expanded_slice(e)
        expanded_chunks.append(
            dense.expanded[expanded_slice][1:surviving].astype(
                np.int64, copy=False
            )
        )
        painted_chunks.append(
            dense.extra_value_v2_mask[expanded_slice][1:surviving]
            | dense.extra_f128_mask[expanded_slice][1:surviving]
        )
        exp_seg_len[i] = max(surviving - 1, 0)
        real_positions = np.nonzero(dense.real_mask[raw_slice])[0]
        real_pos_chunks.append(real_positions.astype(np.int64, copy=False))
        is_neg_chunks.append(
            dense.is_negative_per_position[raw_slice][real_positions]
        )
        real_seg_base[i] = real_running
        real_running += int(real_positions.shape[0])

    expanded_flat = np.concatenate(expanded_chunks)
    is_painted_flat = np.concatenate(painted_chunks)
    is_real_flat = ~is_painted_flat
    is_neg_at_real_flat = np.concatenate(is_neg_chunks)

    if expanded_flat.shape[0] == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.bool_))

    exp_seg_offsets = np.zeros(n_kept + 1, dtype=np.int64)
    np.cumsum(exp_seg_len, out=exp_seg_offsets[1:])
    seg_id = np.repeat(np.arange(n_kept, dtype=np.int64), exp_seg_len)

    is_real_i64 = is_real_flat.astype(np.int64)
    global_cum = np.cumsum(is_real_i64)
    global_cum_excl = global_cum - is_real_i64
    first_idx = np.minimum(
        exp_seg_offsets[:-1], int(global_cum_excl.shape[0]) - 1
    )
    seg_carry_in = global_cum_excl[first_idx]
    real_idx_inclusive = global_cum - 1 - seg_carry_in[seg_id]

    in_number_band = (expanded_flat >= _NUMBER_BAND_LO_SHIFTED) & (
        expanded_flat < _NUMBER_BAND_HI_SHIFTED
    )
    carrier_mask = in_number_band & is_real_flat
    if not carrier_mask.any():
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.bool_))

    carrier_seg = seg_id[carrier_mask]
    carrier_real_global = (
        real_seg_base[carrier_seg] + real_idx_inclusive[carrier_mask]
    )
    carrier_signs = is_neg_at_real_flat[carrier_real_global]
    carrier_block_idx = expanded_flat[carrier_mask] - _NUMBER_BAND_LO_SHIFTED
    return carrier_block_idx, carrier_signs


def _with_random_signs(dense: DenseColumns, rng: np.random.Generator):
    """Stamp random per-position negative flags so sign byte-identity is
    actually exercised (the fuzz builder zeros is_negative)."""
    neg = rng.integers(0, 2, size=dense.is_negative_per_position.shape[0]).astype(
        np.bool_
    )
    return DenseColumns(
        surviving_token_count=dense.surviving_token_count,
        predicted_full_length=dense.predicted_full_length,
        is_cut=dense.is_cut,
        surviving_identity_count=dense.surviving_identity_count,
        surviving_number_chunk_count=dense.surviving_number_chunk_count,
        raw_tokens=dense.raw_tokens,
        real_mask=dense.real_mask,
        number_mask=dense.number_mask,
        runlen_number=dense.runlen_number,
        is_negative_per_position=neg,
        raw_offsets=dense.raw_offsets,
        digit_cumsum=dense.digit_cumsum,
        digit_offsets=dense.digit_offsets,
        expanded=dense.expanded,
        extra_value_v2_mask=dense.extra_value_v2_mask,
        extra_f128_mask=dense.extra_f128_mask,
        node_offsets=dense.node_offsets,
        kept_node_index=dense.kept_node_index,
    )


def main() -> int:
    rng = np.random.default_rng(20260619)
    div = 0
    cases = 0
    for _ in range(4000):
        n_cts = int(rng.integers(0, 9))
        cts = [_gen_ct(rng) for _ in range(n_cts)]
        if not cts:
            cts = [_gen_ct(rng)]
        stage2, _ib, _sl = _build_batch(cts)
        dense = _with_random_signs(dense_columns_from_stage2(stage2), rng)
        ref_b, ref_s = _reference_batched_carrier_signs(dense)
        got_b, got_s = _batched_carrier_signs(dense)
        cases += 1
        if not (
            np.array_equal(ref_b, got_b)
            and ref_b.dtype == np.asarray(got_b).dtype
        ):
            print(f"DIVERGE block_idx: ref {ref_b!r} got {np.asarray(got_b)!r}")
            div += 1
        if not (
            np.array_equal(ref_s, got_s)
            and ref_s.dtype == np.asarray(got_s).dtype
        ):
            print(f"DIVERGE signs: ref {ref_s!r} got {np.asarray(got_s)!r}")
            div += 1
    print(f"compared {cases} cases; {div} divergence(s)")
    return 1 if div else 0


if __name__ == "__main__":
    sys.exit(main())
