"""Per-arm token-equality compare loops.

Single concern: given one :class:`Matched` / :class:`Unmatched` from the
lockstep merge plus the ``BinaryDataset`` + the per-arm name-to-index
lookups, fold the matched arm's or unmatched arm's comparison into the
running ``ValidationStats``.

Extracted from ``validator.py`` to keep the orchestrator focused on
flow-control + setup; the per-arm comparison logic was originally an
inlined ~150-LOC block. Token-mismatch formatting itself stays in
``_validator_mismatch_report`` (single concern: pretty-printing one
diff block).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from ..aligned_data.parsed_record_iter import Matched, ParsedRecord, Unmatched
from ..memmap_builder import VersionKey
from ._validator_mismatch_report import format_token_mismatch

logger = logging.getLogger(__name__)


def _pack_records(
    version_keys: List[VersionKey],
    records: Dict[int, ParsedRecord],
) -> List[dict]:
    """Pack per-variant :class:`ParsedRecord`s into validator-shaped dicts.

    Returns one dict per surviving variant index. Each dict carries the
    vkey + the three ndarrays the comparators byte-compare against the
    memmap-loaded counterparts.
    """
    packed: List[dict] = []
    for variant_index, rec in records.items():
        packed.append(
            {
                "vkey": version_keys[variant_index],
                "tokens": rec.tokens,
                "block_rl": rec.block_runlength,
                "insn_rl": rec.insn_runlength,
            }
        )
    return packed


def compare_matched_arm(
    matched: Matched,
    *,
    version_keys: List[VersionKey],
    has_unique_offsets,
    matched_func_name_to_idx: Dict[str, int],
    dataset,
    vocab_manager,
    stats,
) -> None:
    """Compare one matched-arm function's per-variant tokens/runlengths
    against the memmap-loaded counterpart; mutate ``stats`` in place.

    Mirrors the previous inlined ``count >= 2`` block; the bytes-on-the-
    wire path through ``dataset.load_matched_function`` is unchanged.
    """
    func_name = matched.func_name
    version_data_csv = _pack_records(version_keys, matched.records)

    if not has_unique_offsets(version_data_csv):
        stats.matched_skipped += 1
        return

    if func_name not in matched_func_name_to_idx:
        logger.warning(f"Matched function {func_name} in CSV but not in memmap")
        stats.csv_only_matched += 1
        return

    matched_idx = matched_func_name_to_idx[func_name]
    matched_func = dataset.load_matched_function(matched_idx)

    csv_version_keys = {vdata["vkey"] for vdata in version_data_csv}

    for memmap_version in matched_func.versions:
        vkey = VersionKey(
            arch=memmap_version.metadata["arch"],
            compiler=memmap_version.metadata["compiler"],
            compilerversion=memmap_version.metadata["compilerversion"],
            opt=memmap_version.metadata["opt"],
        )

        if vkey not in csv_version_keys:
            continue

        csv_version = None
        for vdata in version_data_csv:
            if vdata["vkey"] == vkey:
                csv_version = vdata
                break

        if csv_version is None:
            continue

        if not np.array_equal(memmap_version.tokens, csv_version["tokens"]):
            mismatch_details = format_token_mismatch(
                memmap_version.tokens, csv_version["tokens"], vocab_manager
            )
            error_msg = f"Tokens mismatch for {func_name} version {vkey}\n{mismatch_details}"
            stats.errors.append(error_msg)
            continue

        if not np.array_equal(memmap_version.block_runlength, csv_version["block_rl"]):
            error_msg = (
                f"Block runlength mismatch for {func_name} version {vkey}\n"
                f"  Memmap: {memmap_version.block_runlength}\n"
                f"  CSV: {csv_version['block_rl']}"
            )
            stats.errors.append(error_msg)
            continue

        if not np.array_equal(memmap_version.insn_runlength, csv_version["insn_rl"]):
            error_msg = (
                f"Instruction runlength mismatch for {func_name} version {vkey}\n"
                f"  Memmap: {memmap_version.insn_runlength}\n"
                f"  CSV: {csv_version['insn_rl']}"
            )
            stats.errors.append(error_msg)
            continue

    stats.matched_validated += 1


def compare_unmatched_arm(
    unmatched: Unmatched,
    *,
    version_keys: List[VersionKey],
    unmatched_data_by_name_and_vkey: Dict[tuple, int],
    dataset,
    vocab_manager,
    stats,
) -> None:
    """Compare one unmatched-arm function's per-variant tokens against
    the memmap-loaded counterpart; mutate ``stats`` in place.
    """
    func_name = unmatched.func_name
    rec = unmatched.record
    vkey = version_keys[unmatched.variant_index]

    lookup_key = (func_name, vkey)
    if lookup_key not in unmatched_data_by_name_and_vkey:
        logger.warning(f"Unmatched function {func_name} version {vkey} in CSV but not in memmap")
        stats.csv_only_unmatched += 1
        return

    unmatched_idx = unmatched_data_by_name_and_vkey[lookup_key]
    unmatched_func = dataset.load_unmatched_function(unmatched_idx)

    if not np.array_equal(unmatched_func.tokens, rec.tokens):
        mismatch_details = format_token_mismatch(unmatched_func.tokens, rec.tokens, vocab_manager)
        error_msg = f"Tokens mismatch for unmatched function {func_name} version {vkey}\n{mismatch_details}"
        stats.errors.append(error_msg)
        return

    if not np.array_equal(unmatched_func.block_runlength, rec.block_runlength):
        error_msg = (
            f"Block runlength mismatch for unmatched function {func_name} version {vkey}\n"
            f"  Memmap: {unmatched_func.block_runlength}\n"
            f"  CSV: {rec.block_runlength}"
        )
        stats.errors.append(error_msg)
        return

    if not np.array_equal(unmatched_func.insn_runlength, rec.insn_runlength):
        error_msg = (
            f"Instruction runlength mismatch for unmatched function {func_name} version {vkey}\n"
            f"  Memmap: {unmatched_func.insn_runlength}\n"
            f"  CSV: {rec.insn_runlength}"
        )
        stats.errors.append(error_msg)
        return

    stats.unmatched_validated += 1
    del unmatched_data_by_name_and_vkey[lookup_key]
