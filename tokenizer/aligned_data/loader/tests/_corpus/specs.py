"""Per-function fixture descriptors + convenience constructors.

Single concern: dataclass shapes that callers compose into a corpus
build, plus tiny constructors that pre-seed common shapes (matched
function with N variants, unmatched function with N versions). The
builder layer (:mod:`.builder`) consumes these specs and drives the
production pass-2 writers. Records are self-describing in
``_data.bin`` so there is no "overlong" variant concept here -- the
record-header encoder handles every record size up to its (much
larger) cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class VariantSpec:
    """One matched-function variant or one unmatched-function version.

    ``vkey`` is the opaque hashable key the pass-2 writers use to look
    up a variant ref via the (stubbed) ``VariantRegistry``; a plain
    tuple keeps the fixture independent of the real ``VersionKey``
    dataclass.

    ``tokens`` / ``block_rl`` / ``insn_rl`` go through
    ``write_function_binary_data`` so the on-disk ``_data.bin`` records
    are real wire bytes; consumers exercise the same record-parser path
    production goes through.
    """

    vkey: Tuple
    tokens: np.ndarray
    block_rl: np.ndarray
    insn_rl: np.ndarray


@dataclass(frozen=True)
class MatchedFunctionSpec:
    """One matched function = N variants of the same function name.

    Pass-1 production drops the function if all variants share a
    ``data_offset`` (the dedup-result heuristic); fixtures must keep
    the variants' ``tokens`` distinct enough that
    ``write_function_binary_data`` produces distinct offsets.
    """

    func_name: str
    variants: Tuple[VariantSpec, ...]
    called: Tuple[str, ...] = ()


@dataclass(frozen=True)
class UnmatchedFunctionSpec:
    """One unmatched function = N versions written sequentially.

    The production grouping logic (``group_unmatched_entries_by_function``)
    coalesces multiple entries that share a function name into one
    section row; ``versions`` mirrors that list.
    """

    func_name: str
    versions: Tuple[VariantSpec, ...]
    called: Tuple[str, ...] = ()


def make_simple_variant(
    vkey: Tuple, token_seed: int, *, n_tokens: int = 8
) -> VariantSpec:
    """A tiny variant body: distinct tokens per ``token_seed``.

    Uniqueness across variants of one function comes from feeding a
    different ``token_seed``; the dedup cache path is unused (writers
    invoked without a cache).
    """
    tokens = np.arange(
        token_seed * 100, token_seed * 100 + n_tokens, dtype=np.uint16
    )
    block_rl = np.array([n_tokens], dtype=np.uint8)
    insn_rl = np.array([2, n_tokens - 2 if n_tokens > 2 else 1], dtype=np.uint8)
    return VariantSpec(vkey=vkey, tokens=tokens, block_rl=block_rl, insn_rl=insn_rl)


def matched_spec(
    func_name: str,
    *,
    n_variants: int = 2,
    called: Sequence[str] = (),
) -> MatchedFunctionSpec:
    """A 2+ variant matched function."""
    variants: List[VariantSpec] = [
        make_simple_variant((func_name, i), token_seed=i + 1)
        for i in range(n_variants)
    ]
    return MatchedFunctionSpec(
        func_name=func_name,
        variants=tuple(variants),
        called=tuple(called),
    )


def unmatched_spec(
    func_name: str,
    *,
    n_versions: int = 1,
    called: Sequence[str] = (),
) -> UnmatchedFunctionSpec:
    """A 1+ version unmatched function."""
    versions: List[VariantSpec] = [
        make_simple_variant((func_name, "u", i), token_seed=i + 1)
        for i in range(n_versions)
    ]
    return UnmatchedFunctionSpec(
        func_name=func_name,
        versions=tuple(versions),
        called=tuple(called),
    )
