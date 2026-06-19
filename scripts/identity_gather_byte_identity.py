"""Byte-identity gate for the stage-3b identity-gather Rust kernel port.

Compares ``_gather_identity_carriers`` (now kernel-backed) against the
ORIGINAL numpy reference (reproduced verbatim below) over a fuzz of
synthetic :class:`DenseColumns` + the 3a inline_byte_slices, exercising
identity carriers of width {0,1,2}, terminal carriers, cuts, dropped
nodes, and DFS multi-node order.

Any ``np.array_equal`` divergence (offsets / L / positions) is a BUG.
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
from tokenizer.aligned_data.loader.batch_decode._identity_decode import (
    _V2_EAGER_BLOCK_END,
    _V2_IDENTITY_BLOCK_START,
    _gather_identity_carriers,
)
from tokenizer.aligned_data.loader.batch_decode._inline_bytes import (
    build_inline_bytes,
)


def _reference_gather(
    dense: DenseColumns, inline_byte_slices
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-port numpy reference (verbatim from HEAD ea79d57)."""
    offset_chunks = []
    L_chunks = []
    pos_chunks = []
    for e in range(dense.n_nodes):
        inline_byte_slice = inline_byte_slices[e]
        if int(dense.surviving_token_count[e]) == 0:
            continue
        in_stream_id_count = int(dense.surviving_identity_count[e]) - 1
        if in_stream_id_count <= 0:
            continue
        raw_slice = dense.node_raw_slice(e)
        raw_tokens = dense.raw_tokens[raw_slice]
        identity_carrier_mask = dense.real_mask[raw_slice] & (
            (raw_tokens >= _V2_IDENTITY_BLOCK_START)
            & (raw_tokens < _V2_EAGER_BLOCK_END)
        )
        identity_carrier_positions = np.nonzero(identity_carrier_mask)[0]
        p = identity_carrier_positions[:in_stream_id_count].astype(np.int64)
        n = int(raw_tokens.shape[0])
        runlen_number = dense.runlen_number[raw_slice]
        has_p1 = p < (n - 1)
        safe_p1 = np.where(has_p1, p + 1, np.int64(0))
        L_raw = runlen_number[safe_p1].astype(np.int64)
        L = np.where(has_p1, L_raw, np.int64(0))
        first_payload_offset = dense.digit_cumsum[dense.node_digit_slice(e)][
            p + 1
        ].astype(np.int64) + np.int64(inline_byte_slice.start)
        offset_chunks.append(first_payload_offset)
        L_chunks.append(L)
        pos_chunks.append(p)
    if not offset_chunks:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), empty_i.copy()
    return (
        np.concatenate(offset_chunks),
        np.concatenate(L_chunks),
        np.concatenate(pos_chunks),
    )


def main() -> int:
    rng = np.random.default_rng(20260620)
    div = 0
    cases = 0
    for _ in range(4000):
        n_cts = int(rng.integers(0, 9))
        cts = [_gen_ct(rng) for _ in range(n_cts)]
        if not cts:
            cts = [_gen_ct(rng)]
        stage2, _ib, _sl = _build_batch(cts)
        dense = dense_columns_from_stage2(stage2)
        # 3a slices off the SAME dense (production order).
        _ibytes, slices = build_inline_bytes(dense)
        ref = _reference_gather(dense, slices)
        got = _gather_identity_carriers(dense, slices)
        cases += 1
        for name, r, g in zip(("offset", "L", "pos"), ref, got):
            g = np.asarray(g)
            if not (np.array_equal(r, g) and r.dtype == g.dtype):
                print(f"DIVERGE {name}: ref {r!r} got {g!r}")
                div += 1
    print(f"compared {cases} cases; {div} divergence(s)")
    return 1 if div else 0


if __name__ == "__main__":
    sys.exit(main())
