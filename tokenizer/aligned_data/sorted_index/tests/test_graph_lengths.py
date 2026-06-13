"""Parity gate: sorted-index lengths == dataloader emitted-token counts.

For every corpus shape, :func:`.._graph_lengths.compute_node_lengths`
must produce, per (section, variant) node and per depth, EXACTLY the
included-token count the dataloader emits: the section-level callee walk
(:func:`walk_callees`) + stage-2 expansion (:func:`expand_tokens`),
summing ``predicted_full_length`` over the call targets with
``path_depth <= depth``. Both consumers drive the SAME shared once-only
inclusion decider, so the equality is structural -- this test pins them
together on real-J corpora.

The once-only + all-variants-equivalence semantics (owner's spec): a
function body is included once per variant on first encounter; a
function reached by EVERY variant at a level is excluded (+ pruned).
This SUPERSEDES the legacy active-path DAG semantics (diamonds spliced
twice). Corpus shapes exercise self-recursion, mutual recursion,
diamond, root+branch-shared, two-L1-same-L2, all-variants-shared callee
(excluded + pruned), partially-shared callee, single-variant root
(FLAG-A: splices nothing), and late convergence (FLAG-B).

Every parity corpus uses CROSS-MATCHING vkeys (callee sections carry
the callers' vkeys) so per-call Js resolve to real edges -- a corpus
whose vkeys don't match across sections leaves every J at
MISSING_VARIANT_INDEX and the splice tree is vacuously root-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pytest

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.loader.batch_decode._callee_walk import (
    walk_callees,
)
from tokenizer.aligned_data.loader.batch_decode._expand_tokens import (
    expand_tokens,
)
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.tests._corpus import (
    MatchedFunctionSpec,
    VariantSpec,
    build_corpus,
    make_simple_variant,
)
from tokenizer.aligned_data.matched_sections_columnar import (
    parse_sections_columnar,
)
from tokenizer.aligned_data.sorted_index._graph_lengths import (
    compute_node_lengths,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture,
    build_many_variant_section_fixture,
    build_missing_variant_index_fixture,
)


DEPTHS = [0, 1, 2, 3]


def _variants(name: str, n: int, *, seed: int) -> tuple:
    return tuple(
        make_simple_variant((name, i), token_seed=seed + i, n_tokens=6 + i)
        for i in range(n)
    )


def _shared(n: int, *, seed: int) -> tuple:
    """``n`` variants under the SHARED vkey namespace ``("V", i)``.

    Every section built with :func:`_shared` carries variants keyed
    ``("V", 0..n-1)``; a caller variant ``("V", i)`` therefore resolves
    its per-call J against the callee section's matching ``("V", i)``
    variant (or the ascending-sibling fallback). Cross-section edges
    resolve to real splices -- the precondition for a non-vacuous parity
    test (see the module docstring).
    """
    return tuple(
        make_simple_variant(("V", i), token_seed=seed + i, n_tokens=6 + i)
        for i in range(n)
    )


def _callset(seed: int, per_variant_called: Sequence[Sequence[str]]) -> tuple:
    """Shared-vkey variants with PER-VARIANT call sets.

    ``per_variant_called[i]`` is the callee-name subset variant ``i``
    actually calls. Differentiating the per-variant call sets is what
    makes a callee "reached by SOME but not ALL variants" -- the only
    shape the all-variants-equivalence rule INCLUDES (a callee every
    variant reaches is excluded + pruned). A section whose variants all
    call the same set splices nothing.
    """
    out = []
    for i, called in enumerate(per_variant_called):
        base = make_simple_variant(("V", i), token_seed=seed + i, n_tokens=6 + i)
        out.append(
            VariantSpec(
                vkey=base.vkey,
                tokens=base.tokens,
                block_rl=base.block_rl,
                insn_rl=base.insn_rl,
                called=tuple(called),
            )
        )
    return tuple(out)


def _assert_non_vacuous(base: Path, name: str) -> None:
    """Guard that the corpus actually splices (depth-3 > depth-0 for some
    node) -- else an all-MISSING-J or all-excluded corpus would pass the
    parity check vacuously without exercising the inclusion logic.
    """
    _cols, got = _new_path(base, name)
    assert int(got[3].sum()) > int(got[0].sum()), (
        f"corpus {name!r} splices nothing (depth-3 == depth-0); the "
        "parity test would be vacuous"
    )


def _new_path(base: Path, name: str) -> Dict[int, np.ndarray]:
    starts, lengths = read_csv_section_index_arrays(
        base / f"{name}_index.bin"
    )
    blob = np.fromfile(base / f"{name}_sections.bin", dtype=np.uint8)
    data = np.fromfile(base / f"{name}_data.bin", dtype=np.uint8)
    cols = parse_sections_columnar(blob, starts, lengths)
    return cols, compute_node_lengths(cols, starts, data, DEPTHS)


def _oracle_lengths(base: Path, name: str) -> List[List[Dict[int, int]]]:
    """Legacy walk + expand per (section, variant): {depth -> length}."""
    starts, _lengths = read_csv_section_index_arrays(
        base / f"{name}_index.bin"
    )
    out: List[List[Dict[int, int]]] = []
    dataset = BinaryDataset(base, name, vocab_manager=None)
    with dataset.open_session() as session:
        for idx in range(len(starts)):
            section, _off, matched = (
                session._load_matched_section_and_variants(idx)
            )
            per_variant: List[Dict[int, int]] = []
            for v, fd in enumerate(matched.variants):
                cts = walk_callees(
                    session,
                    root_arm=SectionKind.MATCHED,
                    root_section=section,
                    root_variant_idx=v,
                    root_function_data=fd,
                    root_function_name_ptr=section.function_name_ptr,
                    max_depth=max(DEPTHS),
                )
                expanded = [
                    (ct.path_depth, expand_tokens(ct).predicted_full_length)
                    for ct in cts
                ]
                per_variant.append(
                    {
                        d: sum(n for pd, n in expanded if pd <= d)
                        for d in DEPTHS
                    }
                )
            out.append(per_variant)
    return out


def _assert_equivalent(base: Path) -> None:
    (idx_file,) = (
        p
        for p in base.glob("*_index.bin")
        if not p.name.endswith("_unmatched_index.bin")
    )
    name = idx_file.name[: -len("_index.bin")]
    cols, got = _new_path(base, name)
    oracle = _oracle_lengths(base, name)

    for s, per_variant in enumerate(oracle):
        v0 = int(cols.var_offsets[s])
        assert cols.n_variants[s] == len(per_variant)
        for v, depth_map in enumerate(per_variant):
            for d, expected in depth_map.items():
                assert got[d][v0 + v] == expected, (
                    f"section {s} variant {v} depth {d}: "
                    f"got {int(got[d][v0 + v])}, oracle {expected}"
                )


@pytest.mark.parametrize(
    "builder",
    [
        build_many_variant_section_fixture,
        build_missing_variant_index_fixture,
        build_combined_fixture,
    ],
)
def test_matches_oracle_on_fixtures(builder, tmp_path: Path) -> None:
    _assert_equivalent(builder(tmp_path))


def test_matches_oracle_on_deep_chain(tmp_path: Path) -> None:
    # a -> b -> c -> d -> e: depth caps bite at every requested depth.
    specs = [
        MatchedFunctionSpec(
            func_name=n,
            variants=_variants(n, 1, seed=10 * i + 1),
            called=(nxt,) if nxt else (),
        )
        for i, (n, nxt) in enumerate(
            [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", None)]
        )
    ]
    build_corpus(tmp_path, "chain", matched=specs)
    _assert_equivalent(tmp_path)


def test_matches_oracle_on_diamond(tmp_path: Path) -> None:
    # top -> {left, right} -> bottom. Once-only: bottom is included ONCE
    # per variant (deduped across the two branches), NOT twice as the
    # legacy active-path walk emitted it. top v0 calls both branches; v1
    # calls neither, so left/right/bottom are reached by SOME (not all)
    # variants -> included (else the equivalence rule would exclude all).
    specs = [
        MatchedFunctionSpec(
            func_name="top",
            variants=_callset(1, [("left", "right"), ()]),
            called=("left", "right"),
        ),
        MatchedFunctionSpec(
            func_name="left", variants=_callset(21, [("bottom",), ()]),
            called=("bottom",),
        ),
        MatchedFunctionSpec(
            func_name="right", variants=_callset(31, [("bottom",), ()]),
            called=("bottom",),
        ),
        MatchedFunctionSpec(
            func_name="bottom", variants=_shared(2, seed=41), called=(),
        ),
    ]
    build_corpus(tmp_path, "diamond", matched=specs)
    _assert_non_vacuous(tmp_path, "diamond")
    _assert_equivalent(tmp_path)


def test_matches_oracle_on_recursion(tmp_path: Path) -> None:
    # Direct recursion (self) and mutual recursion (ping <-> pong).
    # Once-only inclusion makes a revisit impossible: the recursive edge
    # back to an already-included section is deduped (no re-splice), so
    # the cycle never blows up. Variant 0 drives the calls; variant 1
    # stays quiet so the callees are SOME-not-all (included).
    specs = [
        MatchedFunctionSpec(
            func_name="self_rec",
            variants=_callset(51, [("self_rec",), ()]),
            called=("self_rec",),
        ),
        MatchedFunctionSpec(
            func_name="ping", variants=_callset(61, [("pong",), ()]),
            called=("pong",),
        ),
        MatchedFunctionSpec(
            func_name="pong",
            variants=_callset(71, [("ping", "tail"), ()]),
            called=("ping", "tail"),
        ),
        MatchedFunctionSpec(
            func_name="tail", variants=_shared(2, seed=81), called=(),
        ),
    ]
    build_corpus(tmp_path, "recursion", matched=specs)
    _assert_non_vacuous(tmp_path, "recursion")
    _assert_equivalent(tmp_path)


def test_matches_oracle_on_two_l1_same_l2(tmp_path: Path) -> None:
    # root calls A and B (both L1); A and B both call C (L2). C is
    # reached via two L1 parents at L2 -- deduped to ONE inclusion per
    # variant. Variant 0 drives every edge; variant 1 stays quiet.
    specs = [
        MatchedFunctionSpec(
            func_name="root", variants=_callset(1, [("A", "B"), ()]),
            called=("A", "B"),
        ),
        MatchedFunctionSpec(
            func_name="A", variants=_callset(21, [("C",), ()]), called=("C",),
        ),
        MatchedFunctionSpec(
            func_name="B", variants=_callset(31, [("C",), ()]), called=("C",),
        ),
        MatchedFunctionSpec(
            func_name="C", variants=_shared(2, seed=41), called=(),
        ),
    ]
    build_corpus(tmp_path, "twol1", matched=specs)
    _assert_non_vacuous(tmp_path, "twol1")
    _assert_equivalent(tmp_path)


def test_matches_oracle_on_root_and_branch_shared(tmp_path: Path) -> None:
    # root calls S (L1) and M (L1); M also calls S (L2). S is reached at
    # L1 directly AND at L2 via M -- the L1 inclusion wins, the L2
    # encounter is a dedup no-op. Variant 0 drives the edges.
    specs = [
        MatchedFunctionSpec(
            func_name="root", variants=_callset(1, [("S", "M"), ()]),
            called=("S", "M"),
        ),
        MatchedFunctionSpec(
            func_name="S", variants=_shared(2, seed=21), called=(),
        ),
        MatchedFunctionSpec(
            func_name="M", variants=_callset(31, [("S",), ()]), called=("S",),
        ),
    ]
    build_corpus(tmp_path, "rootshared", matched=specs)
    _assert_non_vacuous(tmp_path, "rootshared")
    _assert_equivalent(tmp_path)


def test_matches_oracle_on_single_variant_root_flag_a(tmp_path: Path) -> None:
    # FLAG-A: a single-variant section's all-variants-equivalence test
    # is over ONE row, so every callee is "reached by all variants" and
    # is excluded -- the splice tree is root-only at every depth. The
    # parity check still holds (both consumers agree on root-only); this
    # corpus is INTENTIONALLY vacuous (no non-vacuous assert).
    specs = [
        MatchedFunctionSpec(
            func_name="solo", variants=_shared(1, seed=1), called=("leaf",),
        ),
        MatchedFunctionSpec(
            func_name="leaf", variants=_shared(1, seed=21), called=(),
        ),
    ]
    build_corpus(tmp_path, "flaga", matched=specs)
    cols, got = _new_path(tmp_path, "flaga")
    # solo is section 0 with 1 variant; FLAG-A -> depth-3 == depth-0.
    assert int(got[3][0]) == int(got[0][0])
    _assert_equivalent(tmp_path)


def test_matches_oracle_on_partially_shared_callee(tmp_path: Path) -> None:
    # 3-variant root; variants 0,1 call P, variant 2 does not. P is
    # reached by SOME but not ALL -> included for v0,v1, never excluded.
    # Q is called by every variant -> excluded (equivalence). The two
    # outcomes coexist in one section.
    specs = [
        MatchedFunctionSpec(
            func_name="root",
            variants=_callset(1, [("P", "Q"), ("P", "Q"), ("Q",)]),
            called=("P", "Q"),
        ),
        MatchedFunctionSpec(
            func_name="P", variants=_shared(3, seed=21), called=(),
        ),
        MatchedFunctionSpec(
            func_name="Q", variants=_shared(3, seed=41), called=(),
        ),
    ]
    build_corpus(tmp_path, "partial", matched=specs)
    _assert_non_vacuous(tmp_path, "partial")
    _assert_equivalent(tmp_path)


def test_matches_oracle_on_all_variants_shared_excluded(tmp_path: Path) -> None:
    # Every variant of root calls X at L1 -> columnwise ALL -> X is
    # excluded AND pruned (its callee Y never appears either). Root-only
    # at every depth; INTENTIONALLY vacuous, parity still holds.
    specs = [
        MatchedFunctionSpec(
            func_name="root", variants=_shared(2, seed=1), called=("X",),
        ),
        MatchedFunctionSpec(
            func_name="X", variants=_callset(21, [("Y",), ()]), called=("Y",),
        ),
        MatchedFunctionSpec(
            func_name="Y", variants=_shared(2, seed=41), called=(),
        ),
    ]
    build_corpus(tmp_path, "allshared", matched=specs)
    cols, got = _new_path(tmp_path, "allshared")
    # root (section 0) calls X by all variants -> X excluded+pruned.
    assert int(got[3][0]) == int(got[0][0])
    assert int(got[3][1]) == int(got[0][1])
    _assert_equivalent(tmp_path)


def test_matches_oracle_on_late_convergence_flag_b(tmp_path: Path) -> None:
    # FLAG-B: F is reached by v0 at L1 (included, expands) but by v1 only
    # at L2, making F's column all-True at L2 -> v1 does NOT include F,
    # and F prunes. root v0 calls F directly; v1 calls G which calls F.
    specs = [
        MatchedFunctionSpec(
            func_name="root", variants=_callset(1, [("F",), ("G",)]),
            called=("F", "G"),
        ),
        MatchedFunctionSpec(
            func_name="G", variants=_callset(21, [("F",), ()]), called=("F",),
        ),
        MatchedFunctionSpec(
            func_name="F", variants=_callset(41, [("leaf",), ()]),
            called=("leaf",),
        ),
        MatchedFunctionSpec(
            func_name="leaf", variants=_shared(2, seed=61), called=(),
        ),
    ]
    build_corpus(tmp_path, "lateconv", matched=specs)
    _assert_non_vacuous(tmp_path, "lateconv")
    _assert_equivalent(tmp_path)


def test_matches_oracle_on_cycle_with_missing_vkey_fallback(
    tmp_path: Path,
) -> None:
    # alpha <-> beta cycle where alpha's variant 0 carries a vkey beta
    # lacks: its per-call slot is MISSING-stamped, so the walker resolves
    # the splice via the ascending-sibling FALLBACK (variant 1's entry)
    # while ON a cycle -- the exact-replay path must consume
    # fallback-resolved edges, not just primary overrides.
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
            variants=(
                make_simple_variant(shared, token_seed=93, n_tokens=8),
            ),
            called=("alpha",),
        ),
    ]
    build_corpus(tmp_path, "cycfb", matched=specs)
    _assert_equivalent(tmp_path)


def test_depth0_equals_own_body_plus_one(tmp_path: Path) -> None:
    # Depth-0 spliced length is byte-identical to the legacy build:
    # 1 self-token + the contributing body length per variant
    # (_body_lengths), with NO splice contribution. Pins that the BFS
    # rewrite did not perturb the depth-0 path.
    from tokenizer.aligned_data.sorted_index._graph_lengths._resolve import (
        _body_lengths,
    )

    specs = [
        MatchedFunctionSpec(
            func_name="root", variants=_callset(1, [("leaf",), ()]),
            called=("leaf",),
        ),
        MatchedFunctionSpec(
            func_name="leaf", variants=_shared(2, seed=21), called=(),
        ),
    ]
    build_corpus(tmp_path, "d0id", matched=specs)
    starts, _l = read_csv_section_index_arrays(tmp_path / "d0id_index.bin")
    blob = np.fromfile(tmp_path / "d0id_sections.bin", dtype=np.uint8)
    data = np.fromfile(tmp_path / "d0id_data.bin", dtype=np.uint8)
    cols = parse_sections_columnar(blob, starts, _l)
    got = compute_node_lengths(cols, starts, data, [0])
    expected = _body_lengths(cols, data) + 1
    np.testing.assert_array_equal(got[0], expected)


def test_buffer_reuse_one_decider_many_roots(tmp_path: Path) -> None:
    # The indexer drives ONE OnceOnlyInclusion across every section of
    # a catalog; this catalog has many roots so the per-root reset path
    # (begin_root) is exercised repeatedly through one instance. A leaked
    # mask cell or hashmap entry across roots would corrupt later
    # sections' lengths -- caught by the oracle equality.
    specs = []
    for k in range(12):
        nxt = f"f{k + 1}"
        specs.append(
            MatchedFunctionSpec(
                func_name=f"f{k}",
                variants=_callset(10 * k + 1, [(nxt,), ()]),
                called=(nxt,),
            )
        )
    specs.append(
        MatchedFunctionSpec(
            func_name="f12", variants=_shared(2, seed=130), called=(),
        )
    )
    build_corpus(tmp_path, "manyroots", matched=specs)
    _assert_non_vacuous(tmp_path, "manyroots")
    _assert_equivalent(tmp_path)
