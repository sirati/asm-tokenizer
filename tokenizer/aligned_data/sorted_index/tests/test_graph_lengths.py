"""Oracle equivalence for the graph-DP length compute.

For every corpus shape, :func:`.._graph_lengths.compute_node_lengths`
must produce, per (section, variant) node and per depth, EXACTLY the
length the legacy machinery computes: the stage-1 callee walk
(:func:`walk_callees`) + stage-2 expansion
(:func:`expand_tokens`), summing ``predicted_full_length`` over call
targets with ``path_depth <= depth`` (the historical
``_variant_lengths_at_depth`` contract under the no-cutoff budget).

Corpus shapes: the production sorted_index fixtures (0-variant,
many-variant, MISSING sentinel) plus purpose-built graphs -- a deep
chain (depth caps bite), a diamond (same callee spliced via two
branches), and direct + mutual recursion (the walker's active-path
cycle skip, exercising the exact-DFS fallback).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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
                    inlined_equivalent_call_targets_only=False,
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
    # top -> {left, right} -> bottom: bottom splices TWICE at depth 2
    # (the walker discards the cycle key on backtrack).
    specs = [
        MatchedFunctionSpec(
            func_name="top",
            variants=_variants("top", 2, seed=1),
            called=("left", "right"),
        ),
        MatchedFunctionSpec(
            func_name="left",
            variants=_variants("left", 1, seed=21),
            called=("bottom",),
        ),
        MatchedFunctionSpec(
            func_name="right",
            variants=_variants("right", 1, seed=31),
            called=("bottom",),
        ),
        MatchedFunctionSpec(
            func_name="bottom",
            variants=_variants("bottom", 1, seed=41),
            called=(),
        ),
    ]
    build_corpus(tmp_path, "diamond", matched=specs)
    _assert_equivalent(tmp_path)


def test_matches_oracle_on_recursion(tmp_path: Path) -> None:
    # Direct recursion (self) and mutual recursion (ping <-> pong):
    # the DP alone would over-count; the cycle-exact fallback must kick
    # in and reproduce the walker's active-path skip.
    specs = [
        MatchedFunctionSpec(
            func_name="self_rec",
            variants=_variants("self_rec", 1, seed=51),
            called=("self_rec",),
        ),
        MatchedFunctionSpec(
            func_name="ping",
            variants=_variants("ping", 1, seed=61),
            called=("pong",),
        ),
        MatchedFunctionSpec(
            func_name="pong",
            variants=_variants("pong", 2, seed=71),
            called=("ping", "tail"),
        ),
        MatchedFunctionSpec(
            func_name="tail",
            variants=_variants("tail", 1, seed=81),
            called=(),
        ),
    ]
    build_corpus(tmp_path, "recursion", matched=specs)
    _assert_equivalent(tmp_path)
