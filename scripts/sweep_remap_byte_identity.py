"""Real-binary byte-identity sweep for the remap-kernel port (task #91).

Drives BOTH the staged ``batch_decode`` AND the dense vector path
(``vector_batch_tokens``) on real ``out/build_memmap`` binaries across a
small shape grid, with ``include_fid_sidecar=True`` so the dedup walk's
identity + FID sidecars are exercised + captured. Dumps every returned
array to an ``.npz`` keyed by ``(binary, arm, depth, B, path)``.

Run it once on the CURRENT branch (AFTER, kernel) and once on the base
commit (BEFORE, old Python walk, applied via file-patch -- NOT git stash
in this shared worktree); ``--compare`` then asserts byte-identity of
every captured array. Any divergence is a hard failure.
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
from tokenizer.aligned_data.loader.vector_batch._entry import (
    vector_batch_tokens,
)
from tokenizer.aligned_data.loader.vector_batch.session_handles import (
    open_vector_batch_handles,
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


def _matched_section_idxs(base: Path, binary: str, limit: int) -> list[int]:
    """First ``limit`` matched section idxs carrying >= 1 variant."""
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
    """Run both paths for one shape; return ``{f'{path}.{name}': arr}``."""
    pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=int(i))
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
        with open_vector_batch_handles(base, binary) as handles:
            new = vector_batch_tokens(
                session,
                pointers,
                handles=handles,
                num_variants_per_section=num_variants,
                context_len=context_len,
                max_depth=depth,
                variant_padding=VariantPadding.PAD_NULL,
                include_fid_sidecar=True,
                rng=np.random.default_rng(seed),
            )
    for path_name, result in (("oracle", ref), ("dense", new)):
        for name in _RESULT_ARRAYS:
            arr = getattr(result, name, None)
            if arr is None:
                arr = np.zeros(0, dtype=np.uint8)
            captured[f"{path_name}.{name}"] = np.asarray(arr)
    return captured


def _run_sweep(out_dir: Path, npz_path: Path) -> None:
    vocab_path = resolve_unified_vocab_path(out_dir)
    vocab_manager = load_and_validate_unified_vocab(vocab_path)

    binaries = ["nping", "openssl", "minigzip64", "libcrypto.so.3", "z3"]
    depths = [1, 3]
    batch_caps = [64, 256]
    context_len = 1024
    seed = 1234

    bundle: dict[str, np.ndarray] = {}
    for binary in binaries:
        if not (out_dir / f"{binary}_data.bin").exists():
            print(f"  skip {binary}: no data.bin", file=sys.stderr)
            continue
        section_idxs = _matched_section_idxs(out_dir, binary, limit=24)
        if not section_idxs:
            print(f"  skip {binary}: no matched sections", file=sys.stderr)
            continue
        for depth in depths:
            for cap in batch_caps:
                key = f"{binary}|d{depth}|B{cap}"
                try:
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
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERROR {key}: {exc!r}", file=sys.stderr)
                    raise
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
    print(
        f"compared {len(keys)} arrays; {divergences} divergence(s)"
    )
    return divergences


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None)
    args = ap.parse_args()
    if args.compare is not None:
        return 1 if _compare(args.compare, args.npz) else 0
    _run_sweep(args.out_dir, args.npz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
