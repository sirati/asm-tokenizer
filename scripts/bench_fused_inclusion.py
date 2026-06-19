"""Stage-3 fused inclusion-BFS: single-thread delta + multi-thread scaling.

Measures (1) the fused GIL-released kernel vs the Stage-2 per-level Python
drive (single-thread delta) and (2) the multi-core scaling of N threads each
calling the fused kernel concurrently -- the direct test of the multi-core
unlock (Stage 2 was GIL-bound; one py.detach over the whole BFS should now
scale).

Run under ``nix develop`` with ``PYTHONPATH=$PWD``. Synchronous, ~tens of s.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.loader.tests._corpus import (
    MatchedFunctionSpec,
    VariantSpec,
    build_corpus,
    make_simple_variant,
)
from tokenizer.aligned_data.loader.vector_batch._inclusion._bfs import (
    _ROOT_EDGE_TYPE,
    _bfs_emit,
    _bfs_full_included,
)
from tokenizer.aligned_data.loader.vector_batch._inclusion import (
    compute_row_inclusions,
)
from tokenizer.aligned_data.matched_sections_columnar import (
    parse_sections_columnar,
)
from tokenizer.aligned_data.sorted_index._graph_lengths._adjacency import (
    LiveNodeAdjacency,
)
from tokenizer.aligned_data.splice_inclusion import OnceOnlyInclusion


def _variant(vkey, seed, called):
    base = make_simple_variant(vkey, token_seed=seed, n_tokens=6)
    return VariantSpec(
        vkey=base.vkey,
        tokens=base.tokens,
        block_rl=base.block_rl,
        insn_rl=base.insn_rl,
        called=tuple(called),
    )


def _wide_corpus(tmp: Path, n_funcs: int, fan: int):
    """A deep, wide call graph: each function calls the next `fan` funcs."""
    names = [f"f{k}" for k in range(n_funcs)]
    specs = []
    for k, name in enumerate(names):
        callees = [names[(k + 1 + j) % n_funcs] for j in range(fan)]
        variants = tuple(
            _variant(("V", v), (k * 3 + v) % 600, callees) for v in range(3)
        )
        specs.append(
            MatchedFunctionSpec(
                func_name=name, variants=variants, called=tuple(callees)
            )
        )
    build_corpus(tmp, "bench", matched=specs)
    starts, lens = read_csv_section_index_arrays(tmp / "bench_index.bin")
    blob = np.fromfile(tmp / "bench_sections.bin", dtype=np.uint8)
    cols = parse_sections_columnar(blob, starts, lens)
    adj = LiveNodeAdjacency(cols, starts, cols.sec_of_var)
    return cols, starts, adj


def _python_drive(cols, starts, adj, sec, smp, grp, max_depth):
    """The Stage-2 per-level Python drive (one root group), GIL-bound."""
    decider = OnceOnlyInclusion()
    rows_by_group: dict = {}
    for r in range(sec.size):
        rows_by_group.setdefault(int(grp[r]), []).append(r)
    total = 0
    for batch_rows in rows_by_group.values():
        section_idx = int(sec[batch_rows[0]])
        sampled = smp[batch_rows]
        e_rows, e_types, _inc = _bfs_emit(
            section_idx=section_idx,
            sampled_variants=sampled,
            cols=cols,
            adjacency=adj,
            decider=decider,
            max_depth=max_depth,
        )
        inc_full, _et = _bfs_full_included(
            section_idx=section_idx,
            cols=cols,
            adjacency=adj,
            decider=decider,
            max_depth=max_depth,
        )
        total += sum(len(e) for e in e_rows) + int(inc_full.size)
    return total


def _fused(cols, starts, adj, sec, smp, grp, max_depth):
    incs = compute_row_inclusions(
        cols,
        starts,
        root_sections=sec,
        root_sampled_variants=smp,
        root_groups=grp,
        max_depth=max_depth,
        need_excluded_pool=True,
        adjacency=adj,
    )
    return int(incs.emitted_nodes.size)


def _time(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main():
    with TemporaryDirectory() as d:
        tmp = Path(d)
        cols, starts, adj = _wide_corpus(tmp, n_funcs=200, fan=4)
        n_rows = 60
        rng = np.random.default_rng(7)
        # 20 root groups (sections), each 3 co-sampled variants.
        groups = rng.integers(0, 60, size=20)
        sec_list, smp_list, grp_list = [], [], []
        g = 0
        for s in groups.tolist():
            for v in range(3):
                sec_list.append(s)
                smp_list.append(v)
                grp_list.append(g)
            g += 1
        sec = np.asarray(sec_list, dtype=np.int64)
        smp = np.asarray(smp_list, dtype=np.int64)
        grp = np.asarray(grp_list, dtype=np.int64)
        max_depth = 6

        # warm the lazy/eager catalog + fallback caches identically.
        _python_drive(cols, starts, adj, sec, smp, grp, max_depth)
        _fused(cols, starts, adj, sec, smp, grp, max_depth)

        reps = 200
        t_py = _time(
            lambda: _python_drive(cols, starts, adj, sec, smp, grp, max_depth),
            reps,
        )
        t_fu = _time(
            lambda: _fused(cols, starts, adj, sec, smp, grp, max_depth), reps
        )
        print(f"rows={sec.size} groups={int(grp.max())+1} max_depth={max_depth}")
        print(f"single-thread  python-drive : {t_py*1e3:8.3f} ms")
        print(f"single-thread  fused-kernel : {t_fu*1e3:8.3f} ms")
        print(f"single-thread  speedup      : {t_py/t_fu:6.2f}x")

        # Multi-thread scaling of the GIL-RELEASED KERNEL in isolation. We
        # time `compute_row_inclusions_csr` directly (no RowInclusion object
        # assembly, no per-call closure pre-fill -- the closure is warmed
        # once below), so the measured region is overwhelmingly the single
        # py.detach. Each thread uses its OWN adjacency (the kernel mutates a
        # per-instance fallback cache + decider), so there is no shared
        # mutable state -- the only question is whether the GIL is held.
        def kernel_call(a):
            a.compute_row_inclusions_csr(
                root_sections=sec,
                root_sampled_variants=smp,
                root_groups=grp,
                max_depth=max_depth,
                need_excluded_pool=True,
                root_edge_type=_ROOT_EDGE_TYPE,
            )

        def worker(adj_local, reps_local, out, idx):
            t0 = time.perf_counter()
            for _ in range(reps_local):
                kernel_call(adj_local)
            out[idx] = time.perf_counter() - t0

        per = 400
        single_wall = None
        for n_threads in (1, 2, 4):
            adjs = [
                LiveNodeAdjacency(cols, starts, cols.sec_of_var)
                for _ in range(n_threads)
            ]
            for a in adjs:  # warm each thread's closure + caches
                kernel_call(a)
            out = [0.0] * n_threads
            threads = [
                threading.Thread(target=worker, args=(adjs[i], per, out, i))
                for i in range(n_threads)
            ]
            t0 = time.perf_counter()
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            wall = time.perf_counter() - t0
            total_calls = n_threads * per
            throughput = total_calls / wall
            if n_threads == 1:
                single_wall = wall
            scaling = (single_wall / wall) * n_threads if single_wall else 1.0
            print(
                f"threads={n_threads}  wall={wall*1e3:8.2f} ms  "
                f"calls={total_calls}  throughput={throughput:9.1f} calls/s  "
                f"scaling={scaling:5.2f}x"
            )


if __name__ == "__main__":
    main()
