"""Byte-identity harness for the deferred-callee-body load change.

Runs ``batch_decode`` end-to-end over the synthetic combined fixture
(its ``caller_fn -> callee_fn`` edge + multi-variant callee drive the
per-edge callee resolution + once-only prune that the load-scheduling
change touches), with EVERY sidecar flag on and a fixed rng seed, then
serialises all output arrays to an ``.npz``.

Usage:
    python scripts/repro_callee_load_schedule.py <out.npz>

Compare two runs (old vs new code) with ``--compare a.npz b.npz``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.batch_decode import batch_decode
from tokenizer.aligned_data.loader.batch_decode._resolve_pointers import (
    SectionPointerSpec,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture,
    make_test_vocab_manager,
)

_BINARY_NAME = "sortbin"

# The combined fixture's matched sections, in matched_index.bin order:
# 0 func_zero (0 var), 1 solo_a, 2 multi_fn (4 var), 3 caller_fn ->
# callee_fn, 4 callee_fn (2 var). Sampling all of them (and caller_fn in
# particular) exercises the callee walk's per-edge resolve + prune.
_POINTERS = [
    SectionPointerSpec(arm=SectionKind.MATCHED, idx=i) for i in range(1, 5)
]

_SIDECAR_FIELDS = (
    "tokens",
    "identities",
    "identity_row_offsets",
    "numbers_significant",
    "numbers_sign_exponent",
    "number_row_offsets",
    "batch_idx_to_section_variant",
    "fid_sidecar",
    "fid_row_offsets",
    "fid_per_category_counts",
    "block_runlength",
    "block_runlength_row_offsets",
    "insn_runlength",
    "insn_runlength_row_offsets",
)


def _run() -> dict:
    with tempfile.TemporaryDirectory() as td:
        base = build_combined_fixture(Path(td))
        vocab = make_test_vocab_manager()
        dataset = BinaryDataset(base, _BINARY_NAME, vocab_manager=vocab)
        with dataset.open_session() as session:
            result = batch_decode(
                session,
                _POINTERS,
                num_variants_per_section=4,
                context_len=64,
                max_depth=4,
                include_fid_sidecar=True,
                emit_block_n_insns_runlength=True,
                rng=np.random.default_rng(0xBADC0DE),
            )
    out = {}
    for f in _SIDECAR_FIELDS:
        v = getattr(result, f)
        out[f] = np.asarray(v) if v is not None else np.array([], dtype=np.uint8)
        out[f + "__present"] = np.array([getattr(result, f) is not None])
    return out


def _compare(a_path: str, b_path: str) -> int:
    a = np.load(a_path, allow_pickle=False)
    b = np.load(b_path, allow_pickle=False)
    ok = True
    for f in _SIDECAR_FIELDS:
        if not np.array_equal(a[f], b[f]) or not np.array_equal(
            a[f + "__present"], b[f + "__present"]
        ):
            ok = False
            print(f"MISMATCH: {f}")
            print(f"  a={a[f]!r}")
            print(f"  b={b[f]!r}")
    if ok:
        print("BYTE-IDENTICAL: all sidecars match across the two runs.")
        return 0
    return 1


def main(argv: list) -> int:
    if len(argv) >= 3 and argv[0] == "--compare":
        return _compare(argv[1], argv[2])
    out_path = argv[0] if argv else "out.npz"
    arrays = _run()
    np.savez(out_path, **arrays)
    print(f"wrote {out_path} ({len(_SIDECAR_FIELDS)} sidecars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
