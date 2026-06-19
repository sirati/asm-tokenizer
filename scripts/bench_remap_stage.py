"""Per-batch remap-stage micro-benchmark (task #91 perf gate).

Builds the staged ``batch_decode`` pipeline up to ``Stage3Batch`` once on
a real binary/shape (via ``keep_intermediate=True``), snapshots the
caller-local identity buffer, and times ONLY
``apply_per_row_remap`` -- restoring the snapshot before each timed call
so every iteration remaps the same fresh caller-local ids. Reports the
median wall-clock of the remap stage.

Run on the CURRENT branch (kernel) and the base (old Python walk, via
file-patch) and diff the medians.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.loader.batch_decode import (
    SectionPointerSpec,
    VariantPadding,
    batch_decode,
)
from tokenizer.aligned_data.loader.batch_decode._dedup_walk import (
    apply_per_row_remap,
)
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.unified_vocab_gate import (
    load_and_validate_unified_vocab,
    resolve_unified_vocab_path,
)


def _matched_section_idxs(base: Path, binary: str, limit: int) -> list[int]:
    dataset = BinaryDataset(base, binary, vocab_manager=None)
    out: list[int] = []
    n = 0
    with dataset.open_session() as session:
        while len(out) < limit:
            try:
                matched = session.load_matched(n)
            except IndexError:
                break
            if len(matched.variants) > 0:
                out.append(n)
            n += 1
    return out


def _bench_one(
    base: Path,
    binary: str,
    vocab_manager,
    *,
    depth: int,
    batch_cap: int,
    context_len: int,
    seed: int,
    iters: int,
) -> tuple[float, int]:
    """Return ``(median_seconds, n_rows)`` for the remap stage."""
    section_idxs = _matched_section_idxs(base, binary, limit=64)
    pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=int(i))
        for i in section_idxs
    ]
    num_variants = max(1, batch_cap // max(1, len(section_idxs)))
    dataset = BinaryDataset(base, binary, vocab_manager=vocab_manager)
    with dataset.open_session() as session:
        result = batch_decode(
            session,
            pointers,
            num_variants_per_section=num_variants,
            context_len=context_len,
            max_depth=depth,
            variant_padding=VariantPadding.PAD_NULL,
            include_fid_sidecar=True,
            keep_intermediate=True,
            rng=np.random.default_rng(seed),
        )
    stage3 = result.intermediate
    assert stage3 is not None, "keep_intermediate must yield a Stage3Batch"
    snapshot = stage3.identities_flat_caller_local.copy()

    # Signature-agnostic call: the BEFORE (old Python walk) build requires
    # a caller-provided ``dedup_maps`` pool; the AFTER (kernel) build does
    # not. Probe once to pick the right call shape so this harness can run
    # against EITHER tree via file-patch.
    import inspect

    needs_dedup_maps = (
        "dedup_maps" in inspect.signature(apply_per_row_remap).parameters
    )
    dedup_maps = None
    if needs_dedup_maps:
        from dedup_hashmap import HashMapU32U16

        from tokenizer.aligned_data.loader.batch_decode._dedup_walk import (
            FUNCTION_CATEGORIES,
        )

        dedup_maps = {
            cat: HashMapU32U16(capacity=256) for cat in FUNCTION_CATEGORIES
        }

    samples: list[float] = []
    for _ in range(iters):
        # Restore the fresh caller-local ids (the remap mutates in place).
        stage3.identities_flat_caller_local[:] = snapshot
        t0 = time.perf_counter()
        if needs_dedup_maps:
            apply_per_row_remap(
                stage3, dedup_maps=dedup_maps, collect_fid_sidecar=True
            )
        else:
            apply_per_row_remap(stage3, collect_fid_sidecar=True)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), int(snapshot.size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()

    vocab_manager = load_and_validate_unified_vocab(
        resolve_unified_vocab_path(args.out_dir)
    )
    cells = [
        ("openssl", 3, 280, 1024),
        ("z3", 3, 1024, 4096),
    ]
    for binary, depth, cap, ctx in cells:
        if not (args.out_dir / f"{binary}_data.bin").exists():
            print(f"skip {binary}: no data.bin", file=sys.stderr)
            continue
        med, n_ids = _bench_one(
            args.out_dir,
            binary,
            vocab_manager,
            depth=depth,
            batch_cap=cap,
            context_len=ctx,
            seed=7,
            iters=args.iters,
        )
        print(
            f"{binary} D{depth} B{cap} L{ctx}: remap median "
            f"{med * 1e3:.2f} ms  (identity_tokens={n_ids})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
