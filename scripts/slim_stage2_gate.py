"""Step-5 slim gate: vector-path byte-identity capture + dense-stage perf.

Two modes:

* ``capture <out.npz>`` -- run the vector path over the shape spread
  (depths {1,3} x variants {1,8,32,128} x context_len {16,64,256} incl.
  empty / fully-cut / surviving==1) on the rich splice fixture and the
  unmatched corpus, dumping every returned array. Run once at HEAD, once
  after the change; compare the two npz for byte-identity.
* ``perf`` -- time ``build_dense_sidecars`` (which calls
  ``build_stage2_batch``) on a wide cross-depth unmatched-heavy batch.

Drives the SAME fixtures the byte-identity harness uses, so the staged
oracle is identical to the gated tests.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.vector_batch.tests._byte_identity_harness import (
    _nonempty_matched_idxs,
    _prepare,
    _run_both,
    _unmatched_section_base_idxs,
)
from tokenizer.aligned_data.loader.vector_batch.tests._rich_corpus import (
    build_rich_splice_fixture,
)
from tokenizer.aligned_data.loader.vector_batch.tests._rich_unmatched_corpus import (
    build_rich_unmatched_fixture,
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


def _flatten_result(tag: str, res) -> dict:
    out = {}
    for name in _RESULT_ARRAYS:
        val = getattr(res, name, None)
        if val is None:
            out[f"{tag}::{name}"] = np.array([np.nan], dtype=np.float64)
        else:
            out[f"{tag}::{name}"] = np.asarray(val)
    return out


def _spread():
    # (depths, variants, context_lens) -- includes empty (ctx 0 handled by
    # tiny ctx), fully-cut (ctx 16 tight on max_depth 3), surviving==1.
    for max_depth in (1, 3):
        for nvar in (1, 8, 32, 128):
            for ctx in (16, 64, 256):
                yield max_depth, nvar, ctx


def capture(out_path: Path, tmp_root: Path) -> None:
    rich = _prepare(build_rich_splice_fixture, tmp_root / "rich")
    rich_idxs = _nonempty_matched_idxs(rich)
    unm = _prepare(build_rich_unmatched_fixture, tmp_root / "unm")
    unm_idxs = _unmatched_section_base_idxs(unm)

    dump: dict = {}
    for max_depth, nvar, ctx in _spread():
        _, new = _run_both(
            rich,
            section_idxs=rich_idxs,
            num_variants_per_section=nvar,
            context_len=ctx,
            max_depth=max_depth,
            seed=0,
            arm=SectionKind.MATCHED,
        )
        tag = f"matched/d{max_depth}/v{nvar}/c{ctx}"
        dump.update(_flatten_result(tag, new))

        _, new_u = _run_both(
            unm,
            section_idxs=unm_idxs,
            num_variants_per_section=nvar,
            context_len=ctx,
            max_depth=max_depth,
            seed=0,
            arm=SectionKind.UNMATCHED,
        )
        tag_u = f"unmatched/d{max_depth}/v{nvar}/c{ctx}"
        dump.update(_flatten_result(tag_u, new_u))

    np.savez(out_path, **dump)
    print(f"captured {len(dump)} arrays -> {out_path}")


def compare(a_path: Path, b_path: Path) -> int:
    a = np.load(a_path, allow_pickle=False)
    b = np.load(b_path, allow_pickle=False)
    keys_a, keys_b = set(a.files), set(b.files)
    if keys_a != keys_b:
        print(f"KEY MISMATCH: only-a={keys_a - keys_b} only-b={keys_b - keys_a}")
        return 1
    diverged = 0
    for k in sorted(keys_a):
        if not np.array_equal(a[k], b[k]):
            diverged += 1
            print(f"DIVERGE {k}: shapes {a[k].shape} vs {b[k].shape}")
    if diverged == 0:
        print(f"BYTE-IDENTICAL: all {len(keys_a)} arrays match")
    else:
        print(f"{diverged} arrays diverged")
    return 1 if diverged else 0


def perf(tmp_root: Path) -> None:
    from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
    from tokenizer.aligned_data.loader.batch_decode import (
        SectionPointerSpec,
        VariantPadding,
    )
    from tokenizer.aligned_data.loader.vector_batch.session_handles import (
        open_vector_batch_arm_set,
    )
    from tokenizer.aligned_data.loader.vector_batch import _entry as ventry
    from tokenizer.aligned_data.sorted_index.tests.fixtures import (
        make_test_vocab_manager,
    )

    # Build a large cross-depth unmatched-heavy batch by timing the dense
    # stage in isolation. We reuse the public vector entry but wrap the
    # dense builder with a timer by monkey-instrumenting build_dense_sidecars.
    import tokenizer.aligned_data.loader.vector_batch._scatter._dense as densemod

    rich = _prepare(build_rich_unmatched_fixture, tmp_root / "perf_unm")
    unm_idxs = _unmatched_section_base_idxs(rich)

    timings: list[float] = []
    orig = densemod.build_dense_sidecars

    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        r = orig(*args, **kwargs)
        timings.append(time.perf_counter() - t0)
        return r

    densemod.build_dense_sidecars = timed
    # Re-point the symbol the dispatch module imported (it does
    # ``from ._scatter import build_dense_sidecars`` -> a module-level name).
    import tokenizer.aligned_data.loader.vector_batch._dispatch as dispmod
    dispmod.build_dense_sidecars = timed

    dataset = BinaryDataset(
        rich, "sortbin", vocab_manager=make_test_vocab_manager()
    )
    pointers = [
        SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=int(i))
        for i in unm_idxs
    ]
    n_iter = 8
    with open_vector_batch_arm_set(rich, "sortbin") as handles:
        with dataset.open_session() as session:
            for it in range(n_iter):
                ventry.vector_batch_tokens(
                    session,
                    pointers,
                    handles=handles,
                    num_variants_per_section=128,
                    context_len=512,
                    max_depth=3,
                    variant_padding=VariantPadding.PAD_NULL,
                    include_fid_sidecar=False,
                    rng=np.random.default_rng(it),
                )

    arr = np.asarray(timings)
    # drop the first (warmup)
    warm = arr[1:] if arr.size > 1 else arr
    print(
        f"build_dense_sidecars: n={warm.size} "
        f"mean={warm.mean()*1e3:.2f}ms median={np.median(warm)*1e3:.2f}ms "
        f"min={warm.min()*1e3:.2f}ms"
    )


def microperf() -> None:
    """Isolated B1024/L512-scale ``build_stage2_batch`` per-node-loop delta.

    The synthetic on-disk fixtures cap node count far below corpus scale,
    so this drives the per-node adapter loop DIRECTLY on a large
    ``BatchedExpansion`` (the same ``bench_batched_expand`` synthetic body
    generator). It times the HEAD per-node materialisation (``_slice_per_node``
    + a per-node :class:`InlineDecodeState` / :class:`FunctionData` /
    promotion-mask slice per call target) against the SLIM loop (only the
    ``expanded_token_ids`` slice + scalars + the kept ``category_counts``),
    on the SAME flats -- the exact work step-5 removed from the dense stage.
    """
    from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
        InlineDecodeState,
    )
    from tokenizer.aligned_data.loader.function_data import FunctionData
    from tokenizer.aligned_data.loader.vector_batch._scatter._batched_expand import (
        batched_expand,
    )
    from tokenizer.aligned_data.loader.vector_batch._scatter._expand import (
        _SELF_TOKEN_LUT,
        _slice_per_node,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_batched_expand import _build  # synthetic body generator

    # B1024 rows x ~1 root + cross-depth callees -> ~3000 emitted nodes is a
    # representative L512 unmatched-heavy cross-depth dense stage.
    n_nodes = 3072
    raw, rec, edge_types = _build(n_nodes, seed=0)
    self_ids = _SELF_TOKEN_LUT[edge_types].astype(np.uint16)
    raw_flat = np.asarray(raw, dtype=np.uint16)
    batched = batched_expand(raw, rec, self_ids)
    node_off = np.asarray(batched.node_offsets, dtype=np.int64)
    rec64 = np.asarray(rec, dtype=np.int64)
    surviving = np.diff(node_off)  # full survival

    def head_loop():
        # HEAD: slice the per-node states + masks, then a per-node
        # InlineDecodeState / FunctionData / mask Stage2CallTarget body.
        states, vc2_masks, f128_masks = _slice_per_node(batched, raw_flat, rec64)
        out = []
        for e in range(n_nodes):
            st = states[e]
            fd = FunctionData(
                func_name="",
                metadata={"category_counts": {}},
                tokens=st.raw_tokens,
                insn_runlength=np.zeros(0, dtype=np.int64),
                block_runlength=np.zeros(0, dtype=np.int64),
                variant_tokens=np.zeros(0, dtype=np.uint16),
            )
            out.append(
                (
                    fd,
                    st,
                    batched.expanded[node_off[e] : node_off[e + 1]],
                    vc2_masks[e],
                    f128_masks[e],
                    int(surviving[e]),
                )
            )
        return out

    def slim_loop():
        # SLIM: only the expanded slice + scalars + shared empty singletons.
        empty_b = np.zeros(0, dtype=np.bool_)
        empty_state = InlineDecodeState(
            raw_tokens=np.zeros(0, dtype=np.uint16),
            real_mask=empty_b,
            number_mask=empty_b,
            runlen_number=np.zeros(0, dtype=np.uint16),
            runlen_value=np.zeros(0, dtype=np.uint16),
            carries_inline_mask=empty_b,
            is_negative_per_position=empty_b,
            digit_cumsum=np.zeros(0, dtype=np.int64),
        )
        out = []
        for e in range(n_nodes):
            out.append(
                (
                    empty_state,
                    batched.expanded[node_off[e] : node_off[e + 1]],
                    empty_b,
                    empty_b,
                    int(surviving[e]),
                )
            )
        return out

    for fn in (head_loop, slim_loop):  # warm
        fn()
    reps = 50

    def best(fn):
        b = float("inf")
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            b = min(b, time.perf_counter() - t)
        return b

    t_head = best(head_loop)
    t_slim = best(slim_loop)
    print(
        f"build_stage2 per-node loop (n_nodes={n_nodes}, raw={raw.size}): "
        f"HEAD={t_head*1e3:.2f}ms SLIM={t_slim*1e3:.2f}ms "
        f"delta={(t_head-t_slim)*1e3:.2f}ms ({(1-t_slim/t_head)*100:.1f}% faster)"
    )


def main(argv):
    import tempfile

    mode = argv[0]
    if mode == "capture":
        out = Path(argv[1])
        with tempfile.TemporaryDirectory() as td:
            capture(out, Path(td))
    elif mode == "compare":
        sys.exit(compare(Path(argv[1]), Path(argv[2])))
    elif mode == "perf":
        with tempfile.TemporaryDirectory() as td:
            perf(Path(td))
    elif mode == "microperf":
        microperf()
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
