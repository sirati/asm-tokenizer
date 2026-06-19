"""Micro-benchmark for stage-3c ``build_number_idx_2d``.

Times the number-stage emission on a large synthetic batch with a
realistic mix of NUMBER-band carriers (VC2 multi-chunk, F128, fixed-width)
across many call_targets, so the BEFORE (numpy emitters) vs AFTER (Rust
kernel) wall-clock delta of the number stage is measurable without a live
corpus (the real corpora are gc'd at this HEAD).

Reuses the fuzz fixture builders from ``number_kernel_byte_identity``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from number_kernel_byte_identity import _build_batch, _gen_ct

from tokenizer.aligned_data.loader.batch_decode._number_decode import (
    build_number_idx_2d,
)


def _make_batches(seed: int, n_batches: int, cts_per_batch: int):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(n_batches):
        cts = [_gen_ct(rng) for _ in range(cts_per_batch)]
        batches.append(_build_batch(cts))
    return batches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--cts", type=int, default=400)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    batches = _make_batches(args.seed, args.batches, args.cts)
    # warm-up
    for stage2, inline_bytes, slices in batches:
        build_number_idx_2d(stage2, inline_bytes, slices)

    best = float("inf")
    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        for stage2, inline_bytes, slices in batches:
            build_number_idx_2d(stage2, inline_bytes, slices)
        dt = time.perf_counter() - t0
        times.append(dt)
        best = min(best, dt)
    per_batch_ms = best / args.batches * 1e3
    print(
        f"number-stage: {args.batches} batches x {args.cts} CTs, "
        f"reps={args.reps}; best total {best * 1e3:.2f} ms, "
        f"per-batch {per_batch_ms:.4f} ms "
        f"(median total {np.median(times) * 1e3:.2f} ms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
