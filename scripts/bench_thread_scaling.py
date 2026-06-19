"""Multi-core thread-scaling bench for the vector_batch decode hot path.

Single concern: measure how the per-batch ``vector_batch_tokens`` decode
scales when fanned across N worker THREADS sharing one process (one GIL).
This is the GIL-contention probe: if the decode is GIL-held numpy/Python
orchestration, throughput stays flat (~1x) as threads rise; as residual
ops move into ``py.detach`` Rust kernels, aggregate b/s should climb
toward N x.

Each worker thread owns its OWN ``BinaryDataset`` session + arm-set
handles (independent decode workers -- the same shape a thread-pool CPU
preloader uses; no shared mutable decode state across threads). All
threads decode the SAME pre-sampled pointer set so the per-batch work is
identical across the sweep.

Reported per thread-count T:
  * wall seconds for ``iters`` batches PER thread (T*iters batches total)
  * aggregate batches/sec = (T*iters) / wall
  * scaling = aggregate(T) / aggregate(1)

A perfectly GIL-bound workload gives scaling ~1.0 at every T; a fully
GIL-free workload gives ~T (until memory-bandwidth / core count caps).

Run UNDER A MEMORY CAP (a concurrent ML job holds ~28GB on this box):

    PYTHONPATH=/tmp/dh_shadow:$PWD systemd-run --user --scope \
        -p MemoryMax=20G -p MemorySwapMax=0 --quiet \
        bash -c 'python scripts/bench_thread_scaling.py \
            --binary nping --shape 256x256 --depth 1 --threads 1 3 6 --iters 8'

Use a MODEST batch (B256/B512); never B4096 here (T independent copies =
memory bomb).
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import List

import numpy as np

DEFAULT_MEMMAP_DIR = Path(
    "/home/sirati/devel/python/asm-tokenizer/out/build_memmap"
)
DEFAULT_SEED = 42


def _load_vocab(memmap_dir: Path):
    from tokenizer.aligned_data.loader.unified_vocab_gate import (
        load_and_validate_unified_vocab,
        resolve_unified_vocab_path,
    )

    vocab_path = resolve_unified_vocab_path(memmap_dir)
    return load_and_validate_unified_vocab(vocab_path)


def _collect_pointers(memmap_dir: Path, binary: str, vocab_manager):
    """Matched SectionPointerSpecs with >=1 variant (mirrors bench_decode)."""
    from tokenizer.aligned_data.loader.batch_decode import SectionPointerSpec
    from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
    from tokenizer.aligned_data.loader.metadata_loader import SectionKind

    dataset = BinaryDataset(memmap_dir, binary, vocab_manager=vocab_manager)
    pointers = []
    n = 0
    with dataset.open_session() as session:
        while True:
            try:
                matched = session.load_matched(n)
            except IndexError:
                break
            if len(matched.variants) > 0:
                pointers.append(
                    SectionPointerSpec(arm=SectionKind.MATCHED, idx=n)
                )
            n += 1
    return pointers


def _sample_pointers(pointers, rng: np.random.Generator, B: int):
    idx = rng.integers(0, len(pointers), size=B)
    return [pointers[i] for i in idx]


def _decode_once(session, sampled, handles, *, L, depth, num_variants, seed):
    from tokenizer.aligned_data.loader.batch_decode import VariantPadding
    from tokenizer.aligned_data.loader.vector_batch._entry import (
        vector_batch_tokens,
    )

    rng = np.random.default_rng(seed)
    return vector_batch_tokens(
        session,
        sampled,
        handles=handles,
        num_variants_per_section=num_variants,
        context_len=L,
        max_depth=depth,
        variant_padding=VariantPadding.PAD_NULL,
        rng=rng,
        unmatched_inline=False,
    )


def _worker(
    memmap_dir, binary, vocab_manager, sampled, *,
    L, depth, num_variants, iters, start_barrier, durations, slot,
):
    """One thread: own session+handles, warm once, then time ``iters`` decodes."""
    from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
    from tokenizer.aligned_data.loader.vector_batch.session_handles import (
        open_vector_batch_arm_set,
    )

    dataset = BinaryDataset(memmap_dir, binary, vocab_manager=vocab_manager)
    with dataset.open_session() as session:
        with open_vector_batch_arm_set(memmap_dir, binary) as handles:
            # warm (build native modules, fill caches) BEFORE the timed barrier
            _decode_once(
                session, sampled, handles,
                L=L, depth=depth, num_variants=num_variants, seed=DEFAULT_SEED,
            )
            start_barrier.wait()  # all threads start the timed region together
            t0 = time.perf_counter()
            for _ in range(iters):
                _decode_once(
                    session, sampled, handles,
                    L=L, depth=depth, num_variants=num_variants,
                    seed=DEFAULT_SEED,
                )
            durations[slot] = time.perf_counter() - t0


def _run_thread_count(
    memmap_dir, binary, vocab_manager, sampled, *,
    L, depth, num_variants, iters, n_threads,
):
    start_barrier = threading.Barrier(n_threads)
    durations: List[float] = [0.0] * n_threads
    threads = [
        threading.Thread(
            target=_worker,
            args=(memmap_dir, binary, vocab_manager, sampled),
            kwargs=dict(
                L=L, depth=depth, num_variants=num_variants, iters=iters,
                start_barrier=start_barrier, durations=durations, slot=i,
            ),
        )
        for i in range(n_threads)
    ]
    wall0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall0
    # aggregate throughput across the timed region: each thread did `iters`
    # batches; the timed region spans from the barrier release to the last
    # thread finishing. Use the max per-thread duration as the timed window
    # (all started together at the barrier), batches = n_threads * iters.
    timed_window = max(durations) if durations else wall
    total_batches = n_threads * iters
    agg_bps = total_batches / timed_window if timed_window > 0 else float("nan")
    return agg_bps, timed_window, wall


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--memmap-dir", type=Path, default=DEFAULT_MEMMAP_DIR)
    p.add_argument("--binary", default="nping")
    p.add_argument("--shape", default="256x256", metavar="BxL")
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--num-variants", type=int, default=7)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 3, 6])
    p.add_argument("--iters", type=int, default=8)
    args = p.parse_args(argv)

    B, L = (int(x) for x in args.shape.lower().split("x"))

    print(
        f"[scaling] binary={args.binary} B={B} L={L} depth={args.depth} "
        f"nvar={args.num_variants} iters={args.iters} threads={args.threads}",
        flush=True,
    )

    vocab_manager = _load_vocab(args.memmap_dir)
    all_pointers = _collect_pointers(args.memmap_dir, args.binary, vocab_manager)
    if not all_pointers:
        print(f"[scaling] {args.binary}: no sections with variants -- abort")
        return 1
    sampled = _sample_pointers(
        all_pointers, np.random.default_rng(DEFAULT_SEED), B
    )
    print(f"[scaling] {len(all_pointers)} sections; sampled B={B}", flush=True)

    base_bps = None
    for n_threads in args.threads:
        agg_bps, window, wall = _run_thread_count(
            args.memmap_dir, args.binary, vocab_manager, sampled,
            L=L, depth=args.depth, num_variants=args.num_variants,
            iters=args.iters, n_threads=n_threads,
        )
        if base_bps is None:
            base_bps = agg_bps
        scaling = agg_bps / base_bps if base_bps else float("nan")
        print(
            f"SCALING {args.binary} threads={n_threads} "
            f"agg_bps={agg_bps:.1f} window_s={window:.3f} wall_s={wall:.3f} "
            f"scaling={scaling:.2f}x",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
