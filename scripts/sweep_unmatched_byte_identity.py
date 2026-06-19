"""Unmatched-arm byte-identity sweep for the build-once resolve restructure.

Drives ``batch_decode`` on UNMATCHED section-base roots of real
``out/build_memmap`` binaries across a small (depth, B) grid and dumps
every returned array to an ``.npz``. Run once on the AFTER branch and
once on the BEFORE base (via file-patch, NOT git stash); ``--compare``
asserts byte-identity of every captured array. Any divergence is a hard
failure. Unmatched-heavy binaries maximise the per-section variant slot
count that the O(V^2)->O(V) hoist targets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.loader.batch_decode import (
    SectionPointerSpec,
    VariantPadding,
    batch_decode,
)
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.unified_vocab_gate import (
    load_and_validate_unified_vocab,
    resolve_unified_vocab_path,
)


_RESULT_ARRAYS = (
    "tokens",
    "batch_idx_to_section_variant",
    "identities",
    "identity_row_offsets",
    "numbers_significant",
    "numbers_sign_exponent",
    "number_row_offsets",
    "fid_sidecar",
    "fid_row_offsets",
    "fid_per_category_counts",
)


def _unmatched_section_base_idxs(
    base: Path, binary: str, vocab_manager, limit: int
) -> list[int]:
    """Top-``limit`` HIGHEST-variant unmatched section-base record idxs.

    The arm's ``record_to_section_idx`` maps per-record idx -> section;
    the base record of section ``k`` is the first record with that
    section idx. Roots MUST be section bases (the per-section loaders pin
    the first-record idx). Sections are ranked by their variant count
    (descending) so the sweep exercises the high-V sections where the
    per-slot whole-section rebuild (O(V^2)->O(V)) actually bites; V=1
    sections have nothing to hoist."""
    import collections

    dataset = BinaryDataset(base, binary, vocab_manager=vocab_manager)
    out: list[int] = []
    with dataset.open_session() as session:
        arm = session._meta_get("unmatched_arm")
        if arm is None:
            return out
        mapping = getattr(arm, "record_to_section_idx", None)
        if mapping is None or len(mapping) == 0:
            return out
        base_idx: dict[int, int] = {}
        counts: collections.Counter = collections.Counter()
        for record_idx in range(len(mapping)):
            section_idx = int(mapping[record_idx])
            counts[section_idx] += 1
            if section_idx not in base_idx:
                base_idx[section_idx] = record_idx
        # Highest variant count first (ties broken by lowest base idx for
        # determinism), then take the top ``limit`` section bases.
        ranked = sorted(
            base_idx,
            key=lambda s: (-counts[s], base_idx[s]),
        )
        out = [base_idx[s] for s in ranked[:limit]]
    return out


def _capture_one(
    base: Path,
    binary: str,
    vocab_manager,
    section_idxs: list[int],
    *,
    depth: int,
    batch_size_cap: int,
    context_len: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run the staged batch_decode on unmatched roots for one shape."""
    pointers = [
        SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=int(i))
        for i in section_idxs
    ]
    num_variants = max(1, batch_size_cap // max(1, len(section_idxs)))
    dataset = BinaryDataset(base, binary, vocab_manager=vocab_manager)
    captured: dict[str, np.ndarray] = {}
    with dataset.open_session() as session:
        ref = batch_decode(
            session,
            pointers,
            num_variants_per_section=num_variants,
            context_len=context_len,
            max_depth=depth,
            variant_padding=VariantPadding.PAD_NULL,
            include_fid_sidecar=True,
            rng=np.random.default_rng(seed),
        )
    for name in _RESULT_ARRAYS:
        arr = getattr(ref, name, None)
        if arr is None:
            arr = np.zeros(0, dtype=np.uint8)
        captured[f"oracle.{name}"] = np.asarray(arr)
    return captured


def _run_sweep(
    out_dir: Path, npz_path: Path, binaries: list[str] | None = None
) -> None:
    vocab_path = resolve_unified_vocab_path(out_dir)
    vocab_manager = load_and_validate_unified_vocab(vocab_path)

    if binaries is None:
        binaries = ["nmap", "curl", "sigtool", "freshclam"]
    depths = [1, 3]
    batch_caps = [64, 256, 1024]
    context_len = 1024
    seed = 1234

    bundle: dict[str, np.ndarray] = {}
    for binary in binaries:
        if not (out_dir / f"{binary}_unmatched_data.bin").exists():
            print(f"  skip {binary}: no unmatched_data.bin", file=sys.stderr)
            continue
        section_idxs = _unmatched_section_base_idxs(
            out_dir, binary, vocab_manager, limit=24
        )
        if not section_idxs:
            print(f"  skip {binary}: no unmatched sections", file=sys.stderr)
            continue
        for depth in depths:
            for cap in batch_caps:
                key = f"{binary}|d{depth}|B{cap}"
                cap_dict = _capture_one(
                    out_dir,
                    binary,
                    vocab_manager,
                    section_idxs,
                    depth=depth,
                    batch_size_cap=cap,
                    context_len=context_len,
                    seed=seed,
                )
                for name, arr in cap_dict.items():
                    bundle[f"{key}|{name}"] = arr
                print(f"  captured {key} ({len(cap_dict)} arrays)")
    np.savez(npz_path, **bundle)
    print(f"wrote {npz_path} ({len(bundle)} arrays)")


def _compare(before: Path, after: Path) -> int:
    a = np.load(before, allow_pickle=False)
    b = np.load(after, allow_pickle=False)
    keys = sorted(set(a.files) | set(b.files))
    divergences = 0
    for k in keys:
        if k not in a.files or k not in b.files:
            print(f"DIVERGE {k}: present in only one capture")
            divergences += 1
            continue
        if not np.array_equal(a[k], b[k]):
            print(
                f"DIVERGE {k}: shapes {a[k].shape} vs {b[k].shape}; "
                f"dtypes {a[k].dtype} vs {b[k].dtype}"
            )
            divergences += 1
    print(f"compared {len(keys)} arrays; {divergences} divergence(s)")
    return divergences


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--binaries", type=str, default=None)
    args = ap.parse_args()
    if args.compare is not None:
        return 1 if _compare(args.compare, args.npz) else 0
    binaries = (
        [b for b in args.binaries.split(",") if b]
        if args.binaries is not None
        else None
    )
    _run_sweep(args.out_dir, args.npz, binaries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
