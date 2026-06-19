"""Synthetic byte-identity harness for the 3c number ``idx_2d`` kernel port.

Single concern: generate a randomized fuzz of multi-call_target
:class:`Stage2Batch` fixtures that exercise every NUMBER-band emission
corner (VC2 multi-chunk incl ``L=0`` + ``L % 8 != 0`` MSB pad, F128
finite + NaN/Inf + mid-cut, fixed-width FP, terminal carriers, dropped
call_targets), run :func:`build_number_idx_2d` over each, and dump every
returned array to an ``.npz``.

Run once on the BASE Python emitters (golden) and once on the
kernel-backed path; ``--compare`` asserts byte-identity of every captured
array (idx_2d rows per TokenType, per-CT chunk slices, the
``f128_is_nan_or_inf`` flag array, the ``vc2_chunk_exponent_sidecar``).

The fixtures are built around real :class:`InlineDecodeState` so the
carrier-recovery byte-offset arithmetic (digit_cumsum gather) is exercised
exactly as production does, not faked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._number_decode import (
    _NUMBER_BLOCK_TOKEN_TYPES,
    build_number_idx_2d,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    Stage1Batch,
    Stage1CallTarget,
    Stage1Section,
    Stage1Variant,
    Stage2Batch,
    Stage2CallTarget,
    Stage2Section,
    Stage2Variant,
)
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    InlineDecodeState,
)
from tokenizer.aligned_data.loader.decoded.run_lengths import run_lengths
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import Section
from tokenizer.tokens import Category, TokenType


# Raw vocab ids for the NUMBER block (unified layout).
_RAW = {
    "VC2": 257,
    "F16": 258,
    "BF16": 259,
    "F32": 260,
    "F64": 261,
    "F80": 262,
    "F128": 263,
}
_SHIFTED = {k: v - 256 for k, v in _RAW.items()}
_LOCAL_FUNC_SHIFTED = 9
_BLOCK_V2_SHIFTED = 8  # an identity-band token (id 264 raw)
_FIXED_WIDTH = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "F80": 10}


def _empty_function_data() -> FunctionData:
    return FunctionData(
        func_name="dummy",
        metadata={"arch": "x86_64", "compiler": "gcc", "opt": "O2"},
        tokens=np.zeros(0, dtype=np.uint16),
        insn_runlength=np.zeros(0, dtype=np.uint32),
        block_runlength=np.zeros(0, dtype=np.uint32),
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )


def _empty_section() -> Section:
    return Section(
        function_name_ptr=0, section_offset=0, call_targets=[], variants=[]
    )


def _build_state(raw_tokens: np.ndarray) -> InlineDecodeState:
    real_mask = raw_tokens > 256
    number_mask = raw_tokens < 256
    if raw_tokens.shape[0] == 0:
        runlen_number = np.zeros(0, dtype=np.uint32)
        runlen_value = np.zeros(0, dtype=np.uint32)
    else:
        runlen_number = run_lengths(number_mask)
        runlen_value = run_lengths(~real_mask)
    carries_inline_mask = real_mask & (raw_tokens < 272)
    is_negative_per_position = np.zeros(raw_tokens.shape[0], dtype=bool)
    digit_cumsum = np.zeros(raw_tokens.shape[0] + 1, dtype=np.uint32)
    if raw_tokens.shape[0] > 0:
        np.cumsum(number_mask.view(np.uint8), out=digit_cumsum[1:])
    return InlineDecodeState(
        raw_tokens=raw_tokens,
        real_mask=real_mask,
        number_mask=number_mask,
        runlen_number=runlen_number,
        runlen_value=runlen_value,
        carries_inline_mask=carries_inline_mask,
        is_negative_per_position=is_negative_per_position,
        digit_cumsum=digit_cumsum,
    )


def _make_call_target(
    raw_tokens: np.ndarray,
    expanded: np.ndarray,
    extra_vc2: np.ndarray,
    extra_f128: np.ndarray,
    *,
    surviving_token_count: int | None = None,
) -> Stage2CallTarget:
    stage1_ct = Stage1CallTarget(
        function_data=_empty_function_data(),
        state=_build_state(raw_tokens),
        call_targets_section=[],
        encounter_category=Category.LOCAL_FUNC,
        parent_call_target_index=None,
        function_name_ptr=0,
    )
    predicted_full_length = int(expanded.shape[0])
    if surviving_token_count is None:
        surviving_token_count = predicted_full_length
    s = surviving_token_count
    id_mask = (expanded[:s] >= 8) & (expanded[:s] < 16)
    num_mask = (expanded[:s] >= 1) & (expanded[:s] < 8)
    return Stage2CallTarget(
        stage1=stage1_ct,
        expanded_token_ids=expanded,
        extra_value_v2_mask=extra_vc2,
        extra_f128_mask=extra_f128,
        predicted_full_length=predicted_full_length,
        surviving_token_count=s,
        surviving_identity_count=int(id_mask.sum()),
        surviving_number_chunk_count=int(num_mask.sum()),
        is_cut=s < predicted_full_length,
        partial_cut_length=s,
    )


class _CTSpec:
    """One call_target's generated raw + expanded stream + cut."""

    def __init__(self, raw, expanded, extra_vc2, extra_f128, surviving):
        self.raw = raw
        self.expanded = expanded
        self.extra_vc2 = extra_vc2
        self.extra_f128 = extra_f128
        self.surviving = surviving


def _gen_source(rng: np.random.Generator, kind: str):
    """Generate (raw_tokens chunk, expanded ids, vc2 mask, f128 mask) for one
    NUMBER source, plus the count of expanded slots it occupies."""
    if kind == "VC2":
        L = int(rng.integers(0, 24))  # incl 0, incl L%8 != 0
        payload = (
            rng.integers(1, 256, size=L, dtype=np.uint16)
            if L
            else np.zeros(0, dtype=np.uint16)
        )
        raw = np.concatenate(
            [np.array([_RAW["VC2"]], dtype=np.uint16), payload]
        )
        k_full = max(1, (L + 7) // 8)
        paints = k_full - 1
        expanded = np.array(
            [_SHIFTED["VC2"]] * (1 + paints), dtype=np.uint16
        )
        vc2 = np.array([False] + [True] * paints, dtype=bool)
        f128 = np.zeros(1 + paints, dtype=bool)
        return raw, expanded, vc2, f128
    if kind == "F128":
        payload = rng.integers(0, 256, size=16, dtype=np.uint16)
        raw = np.concatenate(
            [np.array([_RAW["F128"]], dtype=np.uint16), payload]
        )
        finite = bool(rng.integers(0, 2))
        if finite:
            expanded = np.array(
                [_SHIFTED["F128"], _SHIFTED["F128"]], dtype=np.uint16
            )
            vc2 = np.zeros(2, dtype=bool)
            f128 = np.array([False, True], dtype=bool)
        else:
            expanded = np.array([_SHIFTED["F128"]], dtype=np.uint16)
            vc2 = np.zeros(1, dtype=bool)
            f128 = np.zeros(1, dtype=bool)
        return raw, expanded, vc2, f128
    # fixed width
    w = _FIXED_WIDTH[kind]
    payload = rng.integers(0, 256, size=w, dtype=np.uint16)
    raw = np.concatenate([np.array([_RAW[kind]], dtype=np.uint16), payload])
    expanded = np.array([_SHIFTED[kind]], dtype=np.uint16)
    return raw, expanded, np.zeros(1, dtype=bool), np.zeros(1, dtype=bool)


def _gen_ct(rng: np.random.Generator) -> _CTSpec:
    """Build one call_target: a prepend identity slot, then a random mix of
    number sources (+ occasional bare identity tokens), with a random cut."""
    raw_chunks = [np.zeros(0, dtype=np.uint16)]  # no raw for prepend slot
    expanded = [np.array([_LOCAL_FUNC_SHIFTED], dtype=np.uint16)]
    vc2 = [np.array([False], dtype=bool)]
    f128 = [np.array([False], dtype=bool)]
    n_sources = int(rng.integers(0, 5))
    kinds = ["VC2", "F128", "F16", "BF16", "F32", "F64", "F80", "ID"]
    for _ in range(n_sources):
        kind = kinds[int(rng.integers(0, len(kinds)))]
        if kind == "ID":
            # a bare identity token (no inline payload) -> id 264 raw,
            # shifted 8. Must NOT contribute a number row.
            raw_chunks.append(np.array([264], dtype=np.uint16))
            expanded.append(np.array([_BLOCK_V2_SHIFTED], dtype=np.uint16))
            vc2.append(np.array([False], dtype=bool))
            f128.append(np.array([False], dtype=bool))
            continue
        r, e, v, f = _gen_source(rng, kind)
        raw_chunks.append(r)
        expanded.append(e)
        vc2.append(v)
        f128.append(f)
    raw = np.concatenate(raw_chunks)
    expanded_a = np.concatenate(expanded)
    vc2_a = np.concatenate(vc2)
    f128_a = np.concatenate(f128)
    total = int(expanded_a.shape[0])
    # random cut: sometimes full, sometimes a partial prefix (>=1).
    if total > 1 and rng.integers(0, 3) == 0:
        surviving = int(rng.integers(1, total + 1))
    else:
        surviving = total
    return _CTSpec(raw, expanded_a, vc2_a, f128_a, surviving)


def _build_batch(cts: list[_CTSpec]):
    """Stitch generated call_targets into one variant/section/batch +
    build the matching inline_bytes buffer + per-CT slices."""
    stage2_cts = [
        _make_call_target(
            c.raw, c.expanded, c.extra_vc2, c.extra_f128,
            surviving_token_count=c.surviving,
        )
        for c in cts
    ]
    stage1_variant = Stage1Variant(
        variant_idx=0,
        variant_ref_offset=0,
        batch_idx=0,
        call_targets=[ct.stage1 for ct in stage2_cts],
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )
    stage1_section = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=0,
        section=_empty_section(),
        variants=[stage1_variant],
    )
    stage1_batch = Stage1Batch(
        sections=[stage1_section],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        batch_size=1,
    )
    stage2_variant = Stage2Variant(
        stage1=stage1_variant,
        call_targets=stage2_cts,
        cut_call_target_index=len(stage2_cts),
        total_surviving_token_count=sum(
            ct.surviving_token_count for ct in stage2_cts
        ),
        total_surviving_identity_count=0,
        total_surviving_number_chunk_count=0,
    )
    stage2_section = Stage2Section(
        stage1=stage1_section, variants=[stage2_variant]
    )
    stage2_batch = Stage2Batch(
        stage1=stage1_batch,
        sections=[stage2_section],
        identity_row_offsets=np.zeros(2, dtype=np.uint32),
        number_row_offsets=np.zeros(2, dtype=np.uint32),
    )
    # inline_bytes: pad at 0, then each CT's inline-digit bytes contiguous.
    payloads = [c.raw[c.raw < 256].astype(np.uint8) for c in cts]
    total_bytes = 1 + sum(p.shape[0] for p in payloads)
    inline_bytes = np.zeros(total_bytes, dtype=np.uint8)
    slices = []
    cursor = 1
    for p in payloads:
        inline_bytes[cursor : cursor + p.shape[0]] = p
        slices.append(slice(cursor, cursor + p.shape[0]))
        cursor += p.shape[0]
    return stage2_batch, inline_bytes, slices


def _capture(seed: int, n_cases: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    bundle: dict[str, np.ndarray] = {}
    for case in range(n_cases):
        n_cts = int(rng.integers(1, 6))
        cts = [_gen_ct(rng) for _ in range(n_cts)]
        stage2, inline_bytes, slices = _build_batch(cts)
        (
            idx_2d_per_type,
            chunk_slices_per_type,
            f128_flag,
            vc2_sidecar,
        ) = build_number_idx_2d(stage2, inline_bytes, slices)
        pre = f"case{case:04d}"
        for T in _NUMBER_BLOCK_TOKEN_TYPES:
            bundle[f"{pre}|rows|{T.name}"] = np.asarray(idx_2d_per_type[T])
            sl = chunk_slices_per_type[T]
            bundle[f"{pre}|slices|{T.name}"] = np.array(
                [[s.start, s.stop] for s in sl], dtype=np.int64
            )
        bundle[f"{pre}|f128flag"] = np.asarray(f128_flag)
        bundle[f"{pre}|vc2sidecar"] = np.asarray(vc2_sidecar)
    return bundle


def _compare(before: Path, after: Path) -> int:
    a = np.load(before, allow_pickle=False)
    b = np.load(after, allow_pickle=False)
    keys = sorted(set(a.files) | set(b.files))
    div = 0
    for k in keys:
        if k not in a.files or k not in b.files:
            print(f"DIVERGE {k}: present in only one capture")
            div += 1
            continue
        if a[k].dtype != b[k].dtype or not np.array_equal(a[k], b[k]):
            print(
                f"DIVERGE {k}: shapes {a[k].shape} vs {b[k].shape}; "
                f"dtypes {a[k].dtype} vs {b[k].dtype}"
            )
            div += 1
    print(f"compared {len(keys)} arrays; {div} divergence(s)")
    return div


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=20260619)
    ap.add_argument("--cases", type=int, default=2000)
    args = ap.parse_args()
    if args.compare is not None:
        return 1 if _compare(args.compare, args.npz) else 0
    bundle = _capture(args.seed, args.cases)
    np.savez(args.npz, **bundle)
    print(f"wrote {args.npz} ({len(bundle)} arrays, {args.cases} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
