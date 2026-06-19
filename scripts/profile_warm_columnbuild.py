"""WARM steady-state cProfile of load_batch_cross_depth over the real corpus.

Drives many cross-depth batches AFTER a warm-up pass so the per-binary lazy
section parse (`parse_sections_columnar`) is amortised. The point is to rank
the GIL-held front-matter frames (column-build np ops vs ensure_inclusion_
closure vs _scatter_variants vs numpy glue) on a sustained run, NOT a cold
single-batch profile.

Run under nix develop with PYTHONPATH=$PWD. Synchronous, ~tens of s.
"""

from __future__ import annotations

import cProfile
import io
import logging
import pstats
import time
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.sorted_index import (
    IndexSpec,
    IndexedMemmapCollection,
    LengthReduction,
    MissingIndexPolicy,
    ReductionKind,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    make_test_vocab_manager,
)

_BUILD_MEMMAP = Path("/home/sirati/devel/python/asm-tokenizer/out/build_memmap")
_P75 = LengthReduction(ReductionKind.PERCENTILE, 75)
_SPECS = [IndexSpec(reduction=_P75, depth=d) for d in (0, 1, 3)]
_WIDE_BAND = (1, 10_000_000)


def _drive(coll, n_batches, batch_size, context_len, num_variants, seed0):
    rng = np.random.default_rng(seed0)
    rows = 0
    for _ in range(n_batches):
        res = coll.load_batch_cross_depth(
            0,
            batch_size,
            rng=rng,
            band=_WIDE_BAND,
            context_len=context_len,
            num_variants_per_section=num_variants,
            include_fid_sidecar=True,
        )
        rows += int(res.inner.tokens.shape[0])
    return rows


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-batches", type=int, default=30)
    ap.add_argument("--prof-batches", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--context-len", type=int, default=512)
    ap.add_argument("--num-variants", type=int, default=7)
    ap.add_argument("--top", type=int, default=45)
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    coll = IndexedMemmapCollection.discover(
        [_BUILD_MEMMAP],
        specs=_SPECS,
        on_missing=MissingIndexPolicy.SKIP_WITH_ERROR_LOG,
        vocab_manager=make_test_vocab_manager(),
    )
    try:
        # WARM: amortise every per-binary lazy section parse + closure caches.
        t0 = time.perf_counter()
        _drive(
            coll, args.warm_batches, args.batch_size, args.context_len,
            args.num_variants, seed0=12345,
        )
        warm_s = time.perf_counter() - t0

        # Timed warm steady-state wall (no profiler overhead).
        t0 = time.perf_counter()
        rows = _drive(
            coll, args.prof_batches, args.batch_size, args.context_len,
            args.num_variants, seed0=999,
        )
        wall_s = time.perf_counter() - t0
        print(
            f"WARM wall: warm={warm_s:.3f}s "
            f"steady={wall_s:.3f}s over {args.prof_batches} batches "
            f"({wall_s / args.prof_batches * 1e3:.2f} ms/batch, {rows} rows)"
        )

        # Profiled warm steady-state.
        pr = cProfile.Profile()
        pr.enable()
        _drive(
            coll, args.prof_batches, args.batch_size, args.context_len,
            args.num_variants, seed0=4242,
        )
        pr.disable()
    finally:
        coll.close()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(args.top)
    print(s.getvalue())


if __name__ == "__main__":
    main()
