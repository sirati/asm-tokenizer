"""Memory-bandwidth scaling microbench — the GIL-free reference ceiling.

Single concern: measure how a PURELY memory-bandwidth-bound, GIL-FREE
workload scales across N worker THREADS on this box, to give Round 4 a
reference curve for disentangling "the decode's 6T ratio is GIL-bound"
(→ keep porting) from "it's memory-bandwidth-bound" (→ physical ceiling
reached). If the decode's GIL-free kernels saturate DRAM bandwidth at 6
threads, the decode's after-port scaling should track THIS curve's
shape, not the ideal N x line.

The kernel is a large strided gather + sum over a working set sized to
spill L2/L3 (so it hits DRAM, like the decode's body gather over
``_data.bin``). It runs entirely in numpy C (the GIL is released across
the big ufuncs / fancy-index), so its thread-scaling is governed by
memory bandwidth + core count, NOT the GIL — exactly the regime the
decode's ported kernels live in.

Reported per thread-count T (mirrors bench_thread_scaling.py):
  * aggregate ops/sec = (T*iters) / max-per-thread-window
  * scaling = aggregate(T) / aggregate(1)

A bandwidth-SATURATED workload plateaus (scaling < N, flattening as T
rises); a bandwidth-UNCONSTRAINED one tracks ~N. Compare the decode's
after-port 6T scaling against this curve at the SAME thread counts on
the SAME box (run both under the same cgroup cap).

Run UNDER THE MEMORY CAP (concurrent ML job holds ~28GB):

    systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \
        --quiet bash -c 'python scripts/bench_mem_bandwidth.py \
            --threads 1 3 6 --mb 64 --iters 40'
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import List

import numpy as np

DEFAULT_SEED = 42


def _make_workload(mb: int, seed: int):
    """A working set of ~``mb`` MiB of u16 + a random gather index.

    u16 mirrors the decode body stream's element width; the gather index
    is a fixed random permutation-like scatter so the access pattern is
    DRAM-bound (no cache reuse), the same shape as the decode's
    ``data_u16[word_idx]`` body gather.
    """
    rng = np.random.default_rng(seed)
    n = (mb * 1024 * 1024) // 2  # u16 elements
    data = rng.integers(0, 1 << 16, size=n, dtype=np.uint16)
    idx = rng.integers(0, n, size=n, dtype=np.int64)
    return data, idx


def _op_once(data: np.ndarray, idx: np.ndarray) -> int:
    """One bandwidth-bound gather + reduce (GIL released across the C ops)."""
    gathered = data[idx]  # DRAM-bound fancy gather (the decode's hot shape)
    return int(gathered.sum(dtype=np.int64))


def _worker(data, idx, *, iters, start_barrier, durations, slot):
    # warm once before the timed barrier
    _op_once(data, idx)
    start_barrier.wait()
    t0 = time.perf_counter()
    acc = 0
    for _ in range(iters):
        acc ^= _op_once(data, idx)
    durations[slot] = time.perf_counter() - t0
    # keep acc alive so the optimizer can't elide the loop
    if acc == 0x5F5F5F5F:
        print("", end="")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 3, 6])
    p.add_argument("--mb", type=int, default=64, help="working set MiB/thread")
    p.add_argument("--iters", type=int, default=40)
    args = p.parse_args(argv)

    print(
        f"[membw] mb={args.mb} iters={args.iters} threads={args.threads}",
        flush=True,
    )
    # Each thread owns its OWN working set (independent gathers, the same
    # shape as the decode's thread-private datasets) so the aggregate
    # bandwidth demand scales with T.
    workloads = [
        _make_workload(args.mb, DEFAULT_SEED + i) for i in range(max(args.threads))
    ]

    base = None
    for n_threads in args.threads:
        # give thread i its own workload copy
        agg, window = _run_thread_count_multi(
            workloads, iters=args.iters, n_threads=n_threads
        )
        if base is None:
            base = agg
        scaling = agg / base if base else float("nan")
        print(
            f"MEMBW threads={n_threads} agg_ops={agg:.1f} "
            f"window_s={window:.3f} scaling={scaling:.2f}x",
            flush=True,
        )
    return 0


def _run_thread_count_multi(workloads, *, iters, n_threads):
    start_barrier = threading.Barrier(n_threads)
    durations: List[float] = [0.0] * n_threads
    threads = [
        threading.Thread(
            target=_worker,
            args=(workloads[i][0], workloads[i][1]),
            kwargs=dict(
                iters=iters, start_barrier=start_barrier,
                durations=durations, slot=i,
            ),
        )
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    window = max(durations) if durations else float("nan")
    total_ops = n_threads * iters
    agg = total_ops / window if window > 0 else float("nan")
    return agg, window


if __name__ == "__main__":
    raise SystemExit(main())
