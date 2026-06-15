"""End-to-end batch_decode vs vector_batch benchmark on real memmaps.

Usage
-----
    python scripts/bench_decode.py
    python scripts/bench_decode.py --binaries nping openssl --B 64 --L 512 --depth 3
    python scripts/bench_decode.py --cprofile

Before/after comparison recipe (e.g. after a perf change)
----------------------------------------------------------
    git stash
    python scripts/bench_decode.py 2>&1 | tee /tmp/before.txt
    git stash pop
    python scripts/bench_decode.py 2>&1 | tee /tmp/after.txt
    grep -E "^BENCH|^SPEEDUP" /tmp/before.txt /tmp/after.txt

Output lines (machine-greppable)
---------------------------------
    BENCH <binary> <loader> B=.. L=.. D=.. median_ms=.. min_ms=..
    SPEEDUP <binary> vb/bd=..   (< 1 means vb is slower)
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

WORKTREE = Path(__file__).parent.parent

DEFAULT_MEMMAP_DIR = Path("/home/sirati/devel/python/asm-tokenizer/out/build_memmap")
DEFAULT_BINARIES = ["nping", "openssl"]
DEFAULT_B = 64
DEFAULT_L = 512
DEFAULT_DEPTH = 3
DEFAULT_ITERS = 7
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_realized_sidecar(memmap_dir: Path, binary: str) -> None:
    """Generate *_realized.bin sidecar if absent (vector_batch requires it)."""
    realized = memmap_dir / f"{binary}_realized.bin"
    if realized.exists():
        return
    print(f"[bench] generating realized-lengths sidecar for {binary} ...", flush=True)
    cmd = [
        sys.executable, "-m",
        "tokenizer.aligned_data.realized_lengths",
        "--only", binary,
        "--memmap-dir", str(memmap_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"realized_lengths generation failed for {binary}:\n"
            f"{result.stderr}"
        )
    if not realized.exists():
        raise RuntimeError(
            f"realized_lengths ran OK but {realized} still missing"
        )


def _load_vocab(memmap_dir: Path):
    from tokenizer.aligned_data.loader.unified_vocab_gate import (
        load_and_validate_unified_vocab,
        resolve_unified_vocab_path,
    )
    vocab_path = resolve_unified_vocab_path(memmap_dir)
    return load_and_validate_unified_vocab(vocab_path)


def _collect_pointers(memmap_dir: Path, binary: str, vocab_manager):
    """Return list of MATCHED SectionPointerSpecs with at least one variant."""
    from tokenizer.aligned_data.loader.batch_decode import SectionPointerSpec
    from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
    from tokenizer.aligned_data.loader.metadata_loader import SectionKind

    dataset = BinaryDataset(memmap_dir, binary, vocab_manager=vocab_manager)
    pointers: List[SectionPointerSpec] = []
    n = 0
    with dataset.open_session() as session:
        while True:
            try:
                matched = session.load_matched(n)
            except IndexError:
                break
            if len(matched.variants) > 0:
                pointers.append(SectionPointerSpec(arm=SectionKind.MATCHED, idx=n))
            n += 1
    return pointers, dataset


def _sample_pointers(pointers, rng: np.random.Generator, B: int):
    """Draw B pointers (with replacement if needed) so both runs get the same set."""
    indices = rng.integers(0, len(pointers), size=B)
    return [pointers[int(i)] for i in indices]


# ---------------------------------------------------------------------------
# Single-iteration runners
# ---------------------------------------------------------------------------


def _run_batch_decode(session, sampled_pointers, *, L: int, depth: int, seed: int):
    from tokenizer.aligned_data.loader.batch_decode import (
        VariantPadding,
        batch_decode,
    )
    rng = np.random.default_rng(seed)
    result = batch_decode(
        session,
        sampled_pointers,
        num_variants_per_section=1,
        context_len=L,
        max_depth=depth,
        variant_padding=VariantPadding.PAD_NULL,
        rng=rng,
    )
    return result


def _run_vector_batch(session, sampled_pointers, handles, *, L: int, depth: int, seed: int):
    from tokenizer.aligned_data.loader.batch_decode import VariantPadding
    from tokenizer.aligned_data.loader.vector_batch._entry import vector_batch_tokens

    rng = np.random.default_rng(seed)
    result = vector_batch_tokens(
        session,
        sampled_pointers,
        handles=handles,
        num_variants_per_section=1,
        context_len=L,
        max_depth=depth,
        variant_padding=VariantPadding.PAD_NULL,
        rng=rng,
    )
    return result


# ---------------------------------------------------------------------------
# Correctness gate
# ---------------------------------------------------------------------------


def _assert_byte_identical(bd_result, vb_result, binary: str) -> None:
    """Fail loudly if the two loaders diverge on tokens or mapping."""
    bd_tokens = bd_result.tokens
    vb_tokens = vb_result.tokens

    assert bd_tokens.shape == vb_tokens.shape, (
        f"{binary}: shape mismatch bd={bd_tokens.shape} vb={vb_tokens.shape}"
    )
    assert bd_tokens.dtype == vb_tokens.dtype, (
        f"{binary}: dtype mismatch bd={bd_tokens.dtype} vb={vb_tokens.dtype}"
    )
    if not np.array_equal(bd_tokens, vb_tokens):
        diff_rows = np.nonzero((bd_tokens != vb_tokens).any(axis=1))[0]
        r = int(diff_rows[0])
        raise AssertionError(
            f"{binary}: tokens differ in {diff_rows.size} row(s); "
            f"first row {r}:\n"
            f"  bd={bd_tokens[r].tolist()}\n"
            f"  vb={vb_tokens[r].tolist()}"
        )
    if not np.array_equal(
        bd_result.batch_idx_to_section_variant,
        vb_result.batch_idx_to_section_variant,
    ):
        raise AssertionError(
            f"{binary}: batch_idx_to_section_variant diverged"
        )


# ---------------------------------------------------------------------------
# Timing loop
# ---------------------------------------------------------------------------


def _time_loader(fn, iters: int):
    """Return (median_ms, min_ms) across ``iters`` timed calls, discarding warmup."""
    # warmup: call once without recording
    fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1000.0
    return float(np.median(arr)), float(np.min(arr))


def _profile_loader(fn, top_n: int = 10, label: str = "") -> None:
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(top_n)
    print(f"\n--- cProfile top-{top_n} cumulative [{label}] ---")
    print(buf.getvalue())


# ---------------------------------------------------------------------------
# Per-binary benchmark
# ---------------------------------------------------------------------------


def bench_binary(
    binary: str,
    *,
    memmap_dir: Path,
    vocab_manager,
    B: int,
    L: int,
    depth: int,
    iters: int,
    do_cprofile: bool,
) -> None:
    from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
    from tokenizer.aligned_data.loader.vector_batch.session_handles import (
        open_vector_batch_arm_set,
    )

    print(f"\n[bench] === {binary} B={B} L={L} D={depth} ===", flush=True)

    # Sample a fixed reproducible set of pointers (same for both loaders).
    all_pointers, _dataset = _collect_pointers(memmap_dir, binary, vocab_manager)
    if not all_pointers:
        print(f"[bench] {binary}: no matched sections with variants -- skipping")
        return

    rng_sample = np.random.default_rng(DEFAULT_SEED)
    sampled = _sample_pointers(all_pointers, rng_sample, B)
    print(f"[bench] {binary}: {len(all_pointers)} sections, sampled {len(sampled)} pointers")

    dataset = BinaryDataset(memmap_dir, binary, vocab_manager=vocab_manager)

    with dataset.open_session() as session:
        with open_vector_batch_arm_set(memmap_dir, binary) as handles:

            # --- correctness gate (run once at seed=0 to verify identity) ---
            bd_ref = _run_batch_decode(session, sampled, L=L, depth=depth, seed=0)
            vb_ref = _run_vector_batch(session, sampled, handles, L=L, depth=depth, seed=0)
            _assert_byte_identical(bd_ref, vb_ref, binary)
            print(f"[bench] {binary}: byte-identity OK (shape={bd_ref.tokens.shape})")

            # --- timing ---
            bd_fn = lambda: _run_batch_decode(session, sampled, L=L, depth=depth, seed=DEFAULT_SEED)
            vb_fn = lambda: _run_vector_batch(session, sampled, handles, L=L, depth=depth, seed=DEFAULT_SEED)

            if do_cprofile:
                _profile_loader(bd_fn, label=f"{binary}/batch_decode")
                _profile_loader(vb_fn, label=f"{binary}/vector_batch")

            bd_median, bd_min = _time_loader(bd_fn, iters)
            vb_median, vb_min = _time_loader(vb_fn, iters)

    speedup = vb_median / bd_median if bd_median > 0 else float("nan")

    print(
        f"BENCH {binary} batch_decode B={B} L={L} D={depth} "
        f"median_ms={bd_median:.1f} min_ms={bd_min:.1f}"
    )
    print(
        f"BENCH {binary} vector_batch B={B} L={L} D={depth} "
        f"median_ms={vb_median:.1f} min_ms={vb_min:.1f}"
    )
    print(
        f"SPEEDUP {binary} vb/bd={speedup:.3f}  "
        f"({'vb faster' if speedup < 1 else 'vb slower'})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark batch_decode vs vector_batch on real memmaps."
    )
    p.add_argument(
        "--binaries", nargs="+", default=DEFAULT_BINARIES,
        help=f"Binary stems to bench (default: {DEFAULT_BINARIES})",
    )
    p.add_argument("--B", type=int, default=DEFAULT_B, help="Batch size (default 64)")
    p.add_argument("--L", type=int, default=DEFAULT_L, help="Context length (default 512)")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Splice depth (default 3)")
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="Timing iterations (default 7)")
    p.add_argument(
        "--cprofile", action="store_true",
        help="Print top-10 cProfile frames per loader",
    )
    p.add_argument(
        "--memmap-dir", type=Path, default=DEFAULT_MEMMAP_DIR,
        help=f"Directory containing the memmap bins (default: {DEFAULT_MEMMAP_DIR})",
    )
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    memmap_dir: Path = args.memmap_dir

    print(f"[bench] memmap_dir={memmap_dir}")
    print(f"[bench] binaries={args.binaries} B={args.B} L={args.L} D={args.depth} iters={args.iters}")

    vocab_manager = _load_vocab(memmap_dir)
    print(f"[bench] vocab loaded (format_version={vocab_manager.format_version})")

    for binary in args.binaries:
        _ensure_realized_sidecar(memmap_dir, binary)
        bench_binary(
            binary,
            memmap_dir=memmap_dir,
            vocab_manager=vocab_manager,
            B=args.B,
            L=args.L,
            depth=args.depth,
            iters=args.iters,
            do_cprofile=args.cprofile,
        )

    print("\n[bench] done.")


if __name__ == "__main__":
    main()
