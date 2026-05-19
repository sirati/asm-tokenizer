"""Per-function fixture descriptors + convenience constructors.

Single concern: dataclass shapes that callers compose into a corpus
build, plus tiny constructors that pre-seed common shapes (normal
matched function, overlong-record matched variant, unmatched function).
The builder layer (:mod:`.builder`) consumes these specs and drives
the production pass-2 writers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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


def make_overlong_variant(vkey: Tuple, *, token_count: int = 16) -> VariantSpec:
    """A variant whose data record lands in the overlong band
    (real_length > 256 KiB).

    The bulk lives in ``insn_runlength`` because the u16 ``block_len``
    cap (64 KiB) cannot accommodate a >256 KiB record on its own.
    Triggers the writer's sentinel + u24 overlong-field path; the
    inline-indexer encoder emits ``length_field == SENTINEL_LENGTH``
    so the consumer reads the real length from the data record.
    """
    insn_len = (1 << 18) + 5  # 256 KiB + tail -> definitely overlong
    insn = np.zeros(insn_len, dtype=np.uint8)
    block = np.array([1, 2], dtype=np.uint8)
    tokens = np.arange(token_count, dtype=np.uint16)
    return VariantSpec(vkey=vkey, tokens=tokens, block_rl=block, insn_rl=insn)


def matched_spec(
    func_name: str,
    *,
    n_variants: int = 2,
    called: Sequence[str] = (),
    overlong_variant_idx: Optional[int] = None,
) -> MatchedFunctionSpec:
    """A 2+ variant matched function.

    ``overlong_variant_idx`` -- when set, the variant at that position
    uses :func:`make_overlong_variant` so the test exercises the
    inline indexer's sentinel path under the matched arm
    (cross-product of "matched" x "overlong-sentinel").
    """
    variants: List[VariantSpec] = []
    for i in range(n_variants):
        vkey = (func_name, i)
        if overlong_variant_idx is not None and i == overlong_variant_idx:
            variants.append(make_overlong_variant(vkey))
        else:
            variants.append(make_simple_variant(vkey, token_seed=i + 1))
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
    overlong_version_idx: Optional[int] = None,
) -> UnmatchedFunctionSpec:
    """A 1+ version unmatched function."""
    versions: List[VariantSpec] = []
    for i in range(n_versions):
        vkey = (func_name, "u", i)
        if overlong_version_idx is not None and i == overlong_version_idx:
            versions.append(make_overlong_variant(vkey))
        else:
            versions.append(make_simple_variant(vkey, token_seed=i + 1))
    return UnmatchedFunctionSpec(
        func_name=func_name,
        versions=tuple(versions),
        called=tuple(called),
    )
