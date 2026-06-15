"""End-to-end batch_decode vs vector_batch benchmark on real memmaps.

Defaults exercise the NON-degenerate path: num_variants=7, B=70, depths 3/1/0
(depth 3, deepest call-target inclusion, is the most important byte-identity
case and runs first). Never run the byte-identity gate at num_variants=1 -- at
nvar=1 FLAG-A fires and single-variant roots splice nothing, so the entire
multi-variant inclusion/splice path is short-circuited and untested.

Usage
-----
    python scripts/bench_decode.py
    python scripts/bench_decode.py --binaries nping openssl --shapes 70x4096 1120x256 --depths 3 1 0 --num-variants 7
    python scripts/bench_decode.py --cprofile

Before/after comparison recipe (e.g. after a perf change)
----------------------------------------------------------
    # Do NOT use ``git stash`` -- the stash stack is global across sibling
    # worktrees and collides with parallel agents. Use a file-patch instead:
    python scripts/bench_decode.py 2>&1 | tee /tmp/after.txt
    git diff > /tmp/change.patch && git checkout -- .
    python scripts/bench_decode.py 2>&1 | tee /tmp/before.txt
    git apply /tmp/change.patch
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
# (batch_size, seq_len) shapes swept per binary. Roughly constant token budget
# (B*L ~ 287k, except the last) so we cover the few-long-rows <-> many-short-rows
# spectrum where shape-dependent divergences (e.g. #68 at B=256) hide.
DEFAULT_SHAPES = [
    (70, 4096),
    (140, 2048),
    (280, 1024),
    (560, 512),
    (1120, 256),
    (1120, 128),
]
# Depths run in this order; depth 3 (deepest call-target inclusion) is the most
# important byte-identity case and is run first.
DEFAULT_DEPTHS = [3, 1, 0]
# Variants sampled per section. NEVER default this to 1: at nvar=1 the
# columnwise-ALL exclusion (FLAG-A) fires and single-variant roots splice
# nothing, short-circuiting the entire multi-variant inclusion/splice path the
# byte-identity gate is meant to exercise.
DEFAULT_NUM_VARIANTS = 7
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


def _run_batch_decode(session, sampled_pointers, *, L: int, depth: int, num_variants: int, seed: int):
    from tokenizer.aligned_data.loader.batch_decode import (
        VariantPadding,
        batch_decode,
    )
    rng = np.random.default_rng(seed)
    result = batch_decode(
        session,
        sampled_pointers,
        num_variants_per_section=num_variants,
        context_len=L,
        max_depth=depth,
        variant_padding=VariantPadding.PAD_NULL,
        rng=rng,
    )
    return result


def _run_vector_batch(session, sampled_pointers, handles, *, L: int, depth: int, num_variants: int, seed: int):
    from tokenizer.aligned_data.loader.batch_decode import VariantPadding
    from tokenizer.aligned_data.loader.vector_batch._entry import vector_batch_tokens

    rng = np.random.default_rng(seed)
    result = vector_batch_tokens(
        session,
        sampled_pointers,
        handles=handles,
        num_variants_per_section=num_variants,
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
    shapes: List[tuple],
    depths: List[int],
    num_variants: int,
    iters: int,
    do_cprofile: bool,
) -> None:
    from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
    from tokenizer.aligned_data.loader.vector_batch.session_handles import (
        open_vector_batch_arm_set,
    )

    all_pointers, _dataset = _collect_pointers(memmap_dir, binary, vocab_manager)
    if not all_pointers:
        print(f"[bench] {binary}: no matched sections with variants -- skipping")
        return
    print(f"[bench] {binary}: {len(all_pointers)} sections with variants")

    dataset = BinaryDataset(memmap_dir, binary, vocab_manager=vocab_manager)

    with dataset.open_session() as session:
        with open_vector_batch_arm_set(memmap_dir, binary) as handles:
            for B, L in shapes:
                # Sample B pointers per shape (reproducible: fixed seed per draw),
                # same set for both loaders + all depths of this shape.
                sampled = _sample_pointers(all_pointers, np.random.default_rng(DEFAULT_SEED), B)
                for depth in depths:
                    print(
                        f"\n[bench] === {binary} B={B} L={L} D={depth} nvar={num_variants} ===",
                        flush=True,
                    )

                    # --- correctness gate (seed=0) ---
                    bd_ref = _run_batch_decode(
                        session, sampled, L=L, depth=depth, num_variants=num_variants, seed=0
                    )
                    vb_ref = _run_vector_batch(
                        session, sampled, handles, L=L, depth=depth, num_variants=num_variants, seed=0
                    )
                    _assert_byte_identical(bd_ref, vb_ref, binary)
                    print(f"[bench] {binary} B={B} L={L} D={depth}: byte-identity OK (shape={bd_ref.tokens.shape})")

                    # --- timing (b/l/d default-args pin the loop vars per lambda) ---
                    bd_fn = lambda b=B, l=L, d=depth: _run_batch_decode(
                        session, sampled, L=l, depth=d, num_variants=num_variants, seed=DEFAULT_SEED
                    )
                    vb_fn = lambda b=B, l=L, d=depth: _run_vector_batch(
                        session, sampled, handles, L=l, depth=d, num_variants=num_variants, seed=DEFAULT_SEED
                    )

                    if do_cprofile:
                        _profile_loader(bd_fn, label=f"{binary}/batch_decode/B{B}L{L}D{depth}")
                        _profile_loader(vb_fn, label=f"{binary}/vector_batch/B{B}L{L}D{depth}")

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
                        f"SPEEDUP {binary} B={B} L={L} D={depth} vb/bd={speedup:.3f}  "
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
    p.add_argument(
        "--shapes", nargs="+", default=None, metavar="BxL",
        help="(batch, seq_len) shapes as BxL, e.g. 70x4096 1120x256 (default: built-in sweep)",
    )
    p.add_argument(
        "--depths", type=int, nargs="+", default=DEFAULT_DEPTHS,
        help="Splice depths to run, in order (default: 3 1 0; depth 3 matters most)",
    )
    p.add_argument(
        "--num-variants", type=int, default=DEFAULT_NUM_VARIANTS,
        help="num_variants_per_section sampled (default 7; do NOT use 1 -- degenerate)",
    )
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


def _parse_shapes(tokens) -> List[tuple]:
    """Parse ['70x4096', '1120x256'] -> [(70, 4096), (1120, 256)]."""
    shapes = []
    for tok in tokens:
        b_str, _, l_str = tok.lower().partition("x")
        shapes.append((int(b_str), int(l_str)))
    return shapes


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    memmap_dir: Path = args.memmap_dir
    shapes = _parse_shapes(args.shapes) if args.shapes else DEFAULT_SHAPES

    print(f"[bench] memmap_dir={memmap_dir}")
    print(
        f"[bench] binaries={args.binaries} shapes={shapes} "
        f"depths={args.depths} nvar={args.num_variants} iters={args.iters}"
    )

    vocab_manager = _load_vocab(memmap_dir)
    print(f"[bench] vocab loaded (format_version={vocab_manager.format_version})")

    for binary in args.binaries:
        _ensure_realized_sidecar(memmap_dir, binary)
        bench_binary(
            binary,
            memmap_dir=memmap_dir,
            vocab_manager=vocab_manager,
            shapes=shapes,
            depths=args.depths,
            num_variants=args.num_variants,
            iters=args.iters,
            do_cprofile=args.cprofile,
        )

    print("\n[bench] done.")


if __name__ == "__main__":
    main()
