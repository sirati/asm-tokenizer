"""Per-arm token-equality compare loops.

Single concern: given one match-data record from
``lockstep_function_match`` plus the per-version CSV decoders + the
``BinaryDataset`` + the per-arm name-to-index lookups, fold the
matched (count >= 2) or unmatched (count == 1) arm's comparison into
the running ``ValidationStats``.

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

from ..memmap_builder import VersionKey
from ._validator_mismatch_report import format_token_mismatch

logger = logging.getLogger(__name__)


def _decode_per_version_rows(
    version_keys: List[VersionKey],
    rows: List[Optional[dict]],
    mapping_dict: Dict[VersionKey, Optional[np.ndarray]],
    decode_csv_row_data,
) -> List[dict]:
    """Decode each non-None row into ``{vkey, tokens, block_rl, insn_rl}``.

    Skips rows that fail to decode (per-CSV corruption is tolerated --
    one bad version does not poison the others' comparison).
    """
    decoded: List[dict] = []
    for vkey, row in zip(version_keys, rows):
        if row is None:
            continue
        mapping = mapping_dict.get(vkey)
        try:
            tokens_csv, block_rl_csv, insn_rl_csv = decode_csv_row_data(row, mapping)
        except Exception:
            continue
        decoded.append(
            {
                "vkey": vkey,
                "tokens": tokens_csv,
                "block_rl": block_rl_csv,
                "insn_rl": insn_rl_csv,
            }
        )
    return decoded


def compare_matched_arm(
    func_name: str,
    rows: List[Optional[dict]],
    *,
    version_keys: List[VersionKey],
    mapping_dict: Dict[VersionKey, Optional[np.ndarray]],
    decode_csv_row_data,
    has_unique_offsets,
    should_skip_matched_function,
    matched_func_name_to_idx: Dict[str, int],
    dataset,
    vocab_manager,
    stats,
) -> None:
    """Compare one matched-arm function's per-version tokens/runlengths
    against the memmap-loaded counterpart; mutate ``stats`` in place.

    Mirrors the previous inlined ``count >= 2`` block in
    ``validate_memmap_output``; the bytes-on-the-wire path through
    ``dataset.load_matched_function`` is unchanged.
    """
    if should_skip_matched_function(rows):
        stats.matched_skipped += 1
        return

    version_data_csv = _decode_per_version_rows(
        version_keys, rows, mapping_dict, decode_csv_row_data
    )

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
    func_name: str,
    rows: List[Optional[dict]],
    *,
    version_keys: List[VersionKey],
    mapping_dict: Dict[VersionKey, Optional[np.ndarray]],
    decode_csv_row_data,
    should_skip_unmatched_function,
    unmatched_data_by_name_and_vkey: Dict[tuple, int],
    dataset,
    vocab_manager,
    stats,
) -> None:
    """Compare one unmatched-arm function's per-version tokens against
    the memmap-loaded counterpart; mutate ``stats`` in place.

    Mirrors the previous inlined ``count == 1`` block in
    ``validate_memmap_output``; entries are consumed from
    ``unmatched_data_by_name_and_vkey`` on success so the caller can
    track leftovers if needed (current orchestrator does not).
    """
    for vkey, row in zip(version_keys, rows):
        if row is None:
            continue

        if should_skip_unmatched_function(row):
            stats.unmatched_skipped += 1
            continue

        mapping = mapping_dict.get(vkey)

        try:
            tokens_csv, block_rl_csv, insn_rl_csv = decode_csv_row_data(row, mapping)
        except Exception:
            continue

        lookup_key = (func_name, vkey)
        if lookup_key not in unmatched_data_by_name_and_vkey:
            logger.warning(f"Unmatched function {func_name} version {vkey} in CSV but not in memmap")
            stats.csv_only_unmatched += 1
            continue

        unmatched_idx = unmatched_data_by_name_and_vkey[lookup_key]
        unmatched_func = dataset.load_unmatched_function(unmatched_idx)

        if not np.array_equal(unmatched_func.tokens, tokens_csv):
            mismatch_details = format_token_mismatch(unmatched_func.tokens, tokens_csv, vocab_manager)
            error_msg = f"Tokens mismatch for unmatched function {func_name} version {vkey}\n{mismatch_details}"
            stats.errors.append(error_msg)
            continue

        if not np.array_equal(unmatched_func.block_runlength, block_rl_csv):
            error_msg = (
                f"Block runlength mismatch for unmatched function {func_name} version {vkey}\n"
                f"  Memmap: {unmatched_func.block_runlength}\n"
                f"  CSV: {block_rl_csv}"
            )
            stats.errors.append(error_msg)
            continue

        if not np.array_equal(unmatched_func.insn_runlength, insn_rl_csv):
            error_msg = (
                f"Instruction runlength mismatch for unmatched function {func_name} version {vkey}\n"
                f"  Memmap: {unmatched_func.insn_runlength}\n"
                f"  CSV: {insn_rl_csv}"
            )
            stats.errors.append(error_msg)
            continue

        stats.unmatched_validated += 1
        del unmatched_data_by_name_and_vkey[lookup_key]
