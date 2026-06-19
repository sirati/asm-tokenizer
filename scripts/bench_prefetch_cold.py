"""Cold-regime measurement for the _data.bin MADV_WILLNEED prefetch.

Single concern: PROVE the page-prefetch does what it claims in the COLD
page-cache regime (the only regime where an advisory WILLNEED hint can
help) and quantify the win, honestly, as a normal user (no root / no
drop_caches).

Cold-cache lever (no root): ``posix_fadvise(POSIX_FADV_DONTNEED)`` evicts
a file's clean pages from the page cache for the current user. We mmap a
real ``_data.bin``, evict it cold, then compare the body-read phase WITH
vs WITHOUT the WILLNEED prefetch over the SAME sampled record set.

Two measurements, both per binary:

1. RESIDENCY PROOF (``mincore``): after eviction, how many of the pages
   the sample will read are resident (should be ~0)?  Then after
   ``prefetch_willneed``, how many are resident (should jump toward full
   coverage)?  This proves the mechanism independent of any timing noise.

2. COLD READ-PHASE TIMING: time the actual gather of every sampled
   record's bytes from a freshly-evicted mmap, WITHOUT prefetch (each
   first-touch faults synchronously) vs WITH prefetch issued first (pages
   stream in ahead).  Re-evict + re-mmap between trials so each trial
   starts cold.  Report the read-phase wall delta.

Run UNDER A MEMORY CAP (a concurrent ML job holds ~28GB on this box):

    PYTHONPATH=/tmp/dh_shadow:$PWD systemd-run --user --scope \
        -p MemoryMax=20G -p MemorySwapMax=0 --quiet \
        bash -c 'python scripts/bench_prefetch_cold.py \
            --binary libcrypto.so.3 --batch 256 --trials 5'
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import mmap
import os
import time
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.loader.vector_batch.session_handles import (
    open_vector_batch_handles,
)
from tokenizer.aligned_data.loader.vector_batch._scatter._locator import (
    RECORD_OFFSET_SHIFT,
)
from tokenizer.aligned_data.loader.vector_batch._scatter._prefetch_spans import (
    estimate_body_prefetch_ranges,
)
from tokenizer.aligned_data.loader.vector_batch._prefetch import prefetch_willneed
from tokenizer.aligned_data.binary_format._bulk_geometry import bulk_token_spans


_PAGE = mmap.PAGESIZE
_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)


def _evict(fd: int) -> None:
    """Evict a file's clean pages from the page cache (no root needed)."""
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def _resident_pages(
    data_u8: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> tuple:
    """(resident_pages, total_pages) over the [start,end) byte ranges.

    Uses ``mincore`` on the whole map once, then counts the page bits the
    sampled ranges touch. The map base address is the read-only numpy
    view's buffer pointer (``.ctypes.data``) -- ``mincore`` only reads.
    """
    size = int(data_u8.nbytes)
    n_pages = (size + _PAGE - 1) // _PAGE
    vec = (ctypes.c_ubyte * n_pages)()
    addr = data_u8.ctypes.data
    rc = _libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec)
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"mincore failed: {os.strerror(err)}")
    resident = np.frombuffer(vec, dtype=np.uint8) & 1
    # Mark the pages each range covers, count how many are resident.
    touched = np.zeros(n_pages, dtype=bool)
    p0 = (starts // _PAGE).astype(np.int64)
    p1 = ((ends + _PAGE - 1) // _PAGE).astype(np.int64)
    for a, b in zip(p0, p1):
        touched[a:b] = True
    total = int(touched.sum())
    res = int((resident[:n_pages][touched]).sum())
    return res, total


def _gather(data_u8: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> int:
    """Touch every sampled record's token bytes; return a checksum.

    Forces the actual reads (the page-faults the prefetch front-runs).
    """
    acc = 0
    for s, c in zip(starts.tolist(), counts.tolist()):
        if c > 0:
            acc += int(data_u8[s : s + 2 * c].sum())
    return acc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--memmap-dir",
        type=Path,
        default=Path("/home/sirati/devel/python/asm-tokenizer/out/build_memmap"),
    )
    p.add_argument("--binary", default="libcrypto.so.3")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    h = open_vector_batch_handles(args.memmap_dir, args.binary)
    cols = h.cols
    geom = h.geometry
    n_nodes = int(geom.body_lengths.size)
    rng = np.random.default_rng(args.seed)
    nodes = rng.integers(0, n_nodes, size=args.batch).astype(np.int64)
    cols.ensure_sections(cols.sec_of_var[nodes])

    # Build a stand-in emission carrying just what the estimator reads.
    class _Emi:
        node = nodes
        own_length = geom.body_lengths[nodes].astype(np.int64) + 1
        id_total = geom.id_counts[nodes].astype(np.int64)
        value_total = geom.value_counts[nodes].astype(np.int64)

    starts_pf, spans_pf = estimate_body_prefetch_ranges(cols, _Emi)

    data_path = str(h.data_u8.filename)
    fd = os.open(data_path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
    data_u8 = np.frombuffer(mm, dtype=np.uint8)

    # Exact record spans (for the gather + the residency footprint).
    rec_off = (
        cols.var_data_offset_shifted[nodes].astype(np.int64) << RECORD_OFFSET_SHIFT
    )
    tstart, tcount = bulk_token_spans(np.asarray(h.data_u8), rec_off)

    print(
        f"[cold] binary={args.binary} batch={args.batch} nodes/{n_nodes} "
        f"data={size // (1024 * 1024)}MB trials={args.trials}",
        flush=True,
    )

    # --- (1) residency proof -------------------------------------------
    _evict(fd)
    res0, tot = _resident_pages(data_u8, tstart, tstart + 2 * tcount)
    prefetch_willneed(mm, starts_pf, spans_pf)
    time.sleep(0.05)  # let async readahead land
    res1, _ = _resident_pages(data_u8, tstart, tstart + 2 * tcount)
    print(
        f"[residency] sample touches {tot} pages | "
        f"after evict: {res0}/{tot} resident ({100*res0/max(tot,1):.1f}%) | "
        f"after WILLNEED: {res1}/{tot} resident ({100*res1/max(tot,1):.1f}%)",
        flush=True,
    )

    # --- (2) cold read-phase timing ------------------------------------
    def trial(do_prefetch: bool) -> float:
        _evict(fd)
        t0 = time.perf_counter()
        if do_prefetch:
            prefetch_willneed(mm, starts_pf, spans_pf)
        _gather(data_u8, tstart, tcount)
        return time.perf_counter() - t0

    base = [trial(False) for _ in range(args.trials)]
    pref = [trial(True) for _ in range(args.trials)]
    bm = float(np.median(base))
    pm = float(np.median(pref))
    print(
        f"[cold-read] median no-prefetch={bm*1e3:.2f}ms "
        f"prefetch={pm*1e3:.2f}ms  speedup={bm/pm:.2f}x  "
        f"(base={[f'{x*1e3:.1f}' for x in base]} "
        f"pref={[f'{x*1e3:.1f}' for x in pref]})",
        flush=True,
    )

    # ``data_u8`` is a frombuffer view over ``mm``; drop it before closing
    # the map so no exported pointer blocks ``mm.close()``.
    del data_u8
    mm.close()
    os.close(fd)
    geom.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
