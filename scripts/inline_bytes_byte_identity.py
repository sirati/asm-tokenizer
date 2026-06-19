"""Byte-identity gate for the stage-3a inline-byte Rust kernel port.

Compares ``build_inline_bytes`` (now kernel-backed) against the ORIGINAL
numpy reference (reproduced verbatim below as ``_reference_*``) over a
fuzz of synthetic :class:`DenseColumns`, exercising the shape spread the
task calls out: cuts (surviving==1 / fully-cut / mid-cut), VC2 multi-
chunk mid-cut (full-payload retention), F128 finite/NaN/Inf, fixed-width
FP, bare identity carriers, dropped call_targets, and multi-node DFS
order. The CUT path (the sharp edge) is exercised hard via ``_gen_ct``'s
random cut.

Any ``np.array_equal`` divergence (inline_bytes OR slices) is a BUG.
"""

from __future__ import annotations

import sys

import numpy as np

from number_kernel_byte_identity import _build_batch, _gen_ct

from tokenizer.aligned_data.loader.batch_decode._dense_columns import (
    DenseColumns,
)
from tokenizer.aligned_data.loader.batch_decode._flat_call_targets import (
    dense_columns_from_stage2,
)
from tokenizer.aligned_data.loader.batch_decode._inline_bytes import (
    build_inline_bytes,
)


def _reference_surviving_bytes(dense: DenseColumns, e: int) -> np.ndarray:
    """The pre-port numpy ``_surviving_bytes`` (verbatim from HEAD b66441e)."""
    raw_slice = dense.node_raw_slice(e)
    raw_tokens = dense.raw_tokens[raw_slice]
    number_mask = dense.number_mask[raw_slice]

    if not bool(dense.is_cut[e]):
        return raw_tokens[number_mask].astype(np.uint8)

    partial_cut_length = int(dense.surviving_token_count[e])
    if partial_cut_length <= 1:
        return np.empty(0, dtype=np.uint8)

    expanded_slice = dense.node_expanded_slice(e)
    extra_vc2_mask = dense.extra_value_v2_mask[expanded_slice]
    extra_f128_mask = dense.extra_f128_mask[expanded_slice]

    visible_extra_vc2 = extra_vc2_mask[1:partial_cut_length]
    visible_extra_f128 = extra_f128_mask[1:partial_cut_length]
    visible_is_painted = visible_extra_vc2 | visible_extra_f128
    visible_is_real_carrier = ~visible_is_painted

    n_carriers_consumed = int(visible_is_real_carrier.sum())
    if n_carriers_consumed == 0:
        return np.empty(0, dtype=np.uint8)

    real_mask = dense.real_mask[raw_slice]
    carrier_positions = np.nonzero(real_mask)[0]
    p_last = int(carrier_positions[n_carriers_consumed - 1])

    runlen_number = dense.runlen_number[raw_slice]
    if p_last + 1 < runlen_number.shape[0]:
        L_last = int(runlen_number[p_last + 1])
    else:
        L_last = 0

    number_mask_keep = number_mask.copy()
    number_mask_keep[p_last + 1 + L_last :] = False
    bytes_kept = raw_tokens[number_mask_keep]

    return bytes_kept.astype(np.uint8)


def _reference_build_inline_bytes(
    dense: DenseColumns,
) -> tuple[np.ndarray, list[slice]]:
    """The pre-port numpy ``build_inline_bytes`` (verbatim from HEAD b66441e)."""
    per_call_target_bytes: list[np.ndarray] = [
        _reference_surviving_bytes(dense, e) for e in range(dense.n_nodes)
    ]
    per_call_target_counts = np.array(
        [arr.shape[0] for arr in per_call_target_bytes], dtype=np.int64
    )
    total_bytes = int(per_call_target_counts.sum())
    inline_bytes = np.zeros(1 + total_bytes, dtype=np.uint8)

    inline_byte_slices: list[slice] = []
    cursor = 1
    for arr, count in zip(per_call_target_bytes, per_call_target_counts):
        n = int(count)
        sl = slice(cursor, cursor + n)
        if n > 0:
            inline_bytes[sl] = arr
        inline_byte_slices.append(sl)
        cursor += n

    return inline_bytes, inline_byte_slices


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
        dense = dense_columns_from_stage2(stage2)

        ref_bytes, ref_slices = _reference_build_inline_bytes(dense)
        got_bytes, got_slices = build_inline_bytes(dense)
        cases += 1

        got_bytes = np.asarray(got_bytes)
        if not (
            np.array_equal(ref_bytes, got_bytes)
            and ref_bytes.dtype == got_bytes.dtype
        ):
            print(
                f"DIVERGE inline_bytes: ref {ref_bytes!r} got {got_bytes!r}"
            )
            div += 1
        if ref_slices != got_slices:
            print(
                f"DIVERGE slices: ref {ref_slices!r} got {got_slices!r}"
            )
            div += 1
    print(f"compared {cases} cases; {div} divergence(s)")
    return 1 if div else 0


if __name__ == "__main__":
    sys.exit(main())
