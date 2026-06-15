"""Byte-identical golden for the sorted-index build (HARD GATE 1).

Pins the full wire-format index bytes across a battery of fixtures that
exercise the linear/mmap-streaming refactor's touched paths -- the three
sorted_index fixtures, a fallback-J cycle corpus, and a z3-style
big-root-then-many-small-roots corpus (high n_cols + fallback-J). The
aggregate sha256 was frozen on base 4086dbb (pre-refactor); it must stay
identical so the HOW (linear time, memmap streaming, precomputed
fallback) never perturbs the WHAT.

Any change here means the build produced different bytes -- a semantic
regression. Re-freeze ONLY after proving the new bytes are correct
against the dataloader oracle (``test_graph_lengths.py``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from tokenizer.aligned_data.loader.tests._corpus import (
    MatchedFunctionSpec,
    VariantSpec,
    build_corpus,
    make_simple_variant,
)
from tokenizer.aligned_data.sorted_index import (
    LengthReduction,
    ReductionKind,
    build_sorted_index_bytes,
)

from ._length_helpers import ensure_sidecar
from .fixtures import (
    build_combined_fixture,
    build_many_variant_section_fixture,
    build_missing_variant_index_fixture,
)


_BINARY = "sortbin"
_REDS = [
    LengthReduction(kind=ReductionKind.MAX),
    LengthReduction(kind=ReductionKind.PERCENTILE, percentile=50),
    LengthReduction(kind=ReductionKind.PERCENTILE, percentile=95),
]
_DEPTHS = [0, 1, 2, 3]

#: Frozen on base 4086dbb (pre linear/mmap refactor). DO NOT edit without
#: re-validating against the dataloader oracle.
_GOLDEN_AGGREGATE = (
    "e9dbfc7868f1025b4da3f53207d675dcb1b8e3822f157c21f01c935786067c00"
)


def _callset(seed: int, per_variant_called) -> tuple:
    out = []
    for i, called in enumerate(per_variant_called):
        b = make_simple_variant(("V", i), token_seed=seed + i, n_tokens=6 + i)
        out.append(
            VariantSpec(
                vkey=b.vkey, tokens=b.tokens, block_rl=b.block_rl,
                insn_rl=b.insn_rl, called=tuple(called),
            )
        )
    return tuple(out)


def _shared(n: int, seed: int) -> tuple:
    return tuple(
        make_simple_variant(("V", i), token_seed=seed + i, n_tokens=6 + i)
        for i in range(n)
    )


def _cycle_fallback_corpus(base: Path) -> str:
    shared = ("shared", 0)
    specs = [
        MatchedFunctionSpec(
            func_name="alpha",
            variants=(
                make_simple_variant(("only_alpha", 0), token_seed=91, n_tokens=6),
                make_simple_variant(shared, token_seed=92, n_tokens=7),
            ),
            called=("beta",),
        ),
        MatchedFunctionSpec(
            func_name="beta",
            variants=(make_simple_variant(shared, token_seed=93, n_tokens=8),),
            called=("alpha",),
        ),
    ]
    base.mkdir(parents=True, exist_ok=True)
    build_corpus(base, "cycfb", matched=specs)
    return "cycfb"


def _big_then_small_corpus(base: Path) -> str:
    specs = []
    chain = [f"c{i}" for i in range(20)]
    specs.append(
        MatchedFunctionSpec(
            func_name="big",
            variants=_callset(1, [tuple(chain[:8]), ()]),
            called=tuple(chain[:8]),
        )
    )
    for i, c in enumerate(chain):
        nxt = chain[i + 1] if i + 1 < len(chain) else None
        if i % 3 == 0:
            specs.append(
                MatchedFunctionSpec(
                    func_name=c,
                    variants=(
                        make_simple_variant(("odd", i), token_seed=100 + i, n_tokens=6),
                        make_simple_variant(("V", 1), token_seed=130 + i, n_tokens=7),
                    ),
                    called=(nxt,) if nxt else (),
                )
            )
        else:
            specs.append(
                MatchedFunctionSpec(
                    func_name=c,
                    variants=_callset(160 + 2 * i, [(nxt,) if nxt else (), ()]),
                    called=(nxt,) if nxt else (),
                )
            )
    for k in range(30):
        specs.append(
            MatchedFunctionSpec(
                func_name=f"tiny{k}",
                variants=_shared(2, seed=300 + 5 * k),
                called=(),
            )
        )
    base.mkdir(parents=True, exist_ok=True)
    build_corpus(base, "bigsmall", matched=specs)
    return "bigsmall"


def _aggregate(tmp_path: Path) -> str:
    fixtures = [
        (_BINARY, build_combined_fixture(tmp_path / "a")),
        (_BINARY, build_many_variant_section_fixture(tmp_path / "b")),
        (_BINARY, build_missing_variant_index_fixture(tmp_path / "c")),
    ]
    d = tmp_path / "d"
    fixtures.append((_cycle_fallback_corpus(d), d))
    e = tmp_path / "e"
    fixtures.append((_big_then_small_corpus(e), e))

    lines: List[str] = []
    for name, base in fixtures:
        # Seed the Phase-4a realized-length sidecar the build now
        # consumes. The golden hash must NOT change: the sidecar body
        # lengths are byte-identical to the retired _data.bin recompute.
        ensure_sidecar(base, name)
        blobs = build_sorted_index_bytes(
            base, name, reductions=_REDS, depths=_DEPTHS
        )
        for spec in sorted(
            blobs, key=lambda s: (s.depth, repr(s.reduction))
        ):
            h = hashlib.sha256(bytes(blobs[spec])).hexdigest()
            lines.append(
                f"{name} depth={spec.depth} {spec.reduction!r} "
                f"sha256={h} len={len(blobs[spec])}"
            )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def test_sorted_index_build_byte_identical_golden(tmp_path: Path) -> None:
    assert _aggregate(tmp_path) == _GOLDEN_AGGREGATE, (
        "sorted-index build bytes drifted from the frozen base-4086dbb "
        "golden; the linear/mmap refactor must not change the output"
    )
