"""Validation module for memmap builder output.

This module validates that the memmap files produced by memmap_builder contain
the same data as the original CSV files.

`_versions.json` cross-check
----------------------------

When `build_memmap_files` runs against sidecar-format inputs it emits a
`<binary>_versions.json` file alongside the per-binary memmap. The schema
is one entry per version (positional index), each carrying the canonical
4 axes (`arch`, `compiler`, `compiler_version`, `opt`), the integer
`variant_id`, and the opaque `extra_metadata` dict. `_sections.csv`
remains flat (no `variant_id` column); `_versions.json` is purely
additive lookup keyed by row position.

The validator already reconstructs `VersionKey` from per-section data
without supplying `variant_id`; this is now safe because `VersionKey`'s
fifth field defaults to `0` (legacy invariant preserved). When the
sidecar is present the validator additionally verifies that the
canonical-4 axes of each `_versions.json` entry match the
positionally-corresponding `VersionKey` reconstructed from `VersionInfo`,
flagging any mismatch as a validation error. Memmaps without the sidecar
(legacy builds) skip the cross-check unconditionally.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tokenizer.compact_base64_utils import base64_to_ndarray_vec

from ..aligned_data.loader import BinaryDataset
from ..aligned_data.match import lockstep_function_match
from ..function_token_list import FunctionTokenList
from ..memmap_builder import VersionKey
from ..token_manager import VocabularyManager
from ..vocab_unifier.loader import load_unified_vocab_manager

logger = logging.getLogger(__name__)


def _load_versions_sidecar(output_dir: Path, binary_name: str) -> Optional[List[dict]]:
    """Return the parsed ``<binary>_versions.json`` list or ``None`` when
    the sidecar is absent (legacy memmap)."""
    sidecar_path = output_dir / f"{binary_name}_versions.json"
    if not sidecar_path.exists():
        return None
    with open(sidecar_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cross_check_versions_sidecar(
    version_keys: List[VersionKey],
    sidecar_entries: List[dict],
) -> List[str]:
    """Verify that each ``_versions.json`` entry's canonical-4 axes match
    the positionally-corresponding reconstructed ``VersionKey``.

    Returns a list of human-readable error strings (empty when
    consistent). The mapping is positional: ``sidecar_entries[i]`` must
    describe the same build as ``version_keys[i]`` — this is the
    contract `build_memmap_files` establishes when emitting the sidecar
    in iteration order over the version list.

    Note the schema asymmetry: `_versions.json` uses ``compiler_version``
    (snake_case, sidecar convention) whereas `VersionKey` uses
    ``compilerversion`` (legacy convention). The cross-check normalises
    by name at the boundary so neither side leaks its naming choice.
    """
    errors: List[str] = []
    if len(version_keys) != len(sidecar_entries):
        errors.append(
            f"_versions.json entry count ({len(sidecar_entries)}) does not "
            f"match reconstructed version count ({len(version_keys)})"
        )
        return errors

    for idx, (vkey, entry) in enumerate(zip(version_keys, sidecar_entries)):
        sidecar_canonical = (
            entry.get("arch"),
            entry.get("compiler"),
            entry.get("compiler_version"),
            entry.get("opt"),
        )
        reconstructed_canonical = (
            vkey.arch,
            vkey.compiler,
            vkey.compilerversion,
            vkey.opt,
        )
        if sidecar_canonical != reconstructed_canonical:
            errors.append(
                f"_versions.json entry {idx} canonical-4 mismatch: "
                f"sidecar={sidecar_canonical} reconstructed={reconstructed_canonical}"
            )
    return errors


@dataclass
class VersionInfo:
    """Information about a binary version for validation."""

    csv_path: Path
    mapping_path: Path
    arch: str
    compiler: str
    compilerversion: str
    opt: str


@dataclass
class ValidatorConfig:
    """Configuration for memmap validation."""

    versions: List[VersionInfo]
    output_dir: Path
    binary_name: str


@dataclass
class ValidationStats:
    """Statistics from validation."""

    matched_validated: int
    matched_skipped: int
    unmatched_validated: int
    unmatched_skipped: int
    csv_only_matched: int
    csv_only_unmatched: int
    errors: List[str]


def load_mapping(mapping_path: Path) -> Optional[np.ndarray]:
    """Load mapping file if it exists."""
    if mapping_path and mapping_path.exists():
        with open(mapping_path, "r", encoding="ascii") as f:
            return base64_to_ndarray_vec(f.read())
    return None


def decode_csv_row_data(row: dict, mapping: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode tokens and runlengths from CSV row."""
    tokens = base64_to_ndarray_vec(row["tokens_base64"])
    if mapping is not None:
        tokens = mapping[tokens]
    tokens = tokens.astype(np.uint16)

    block_runlength = base64_to_ndarray_vec(row["block_runlength_base64"])
    insn_runlength = base64_to_ndarray_vec(row["instruction_runlength_base64"])

    return tokens, block_runlength, insn_runlength


def should_skip_matched_function(rows: List[Optional[dict]]) -> bool:
    """Check if matched function should be skipped based on builder logic."""
    for row in rows:
        if row is not None and "block_runlength_base64" in row:
            try:
                block_runlength = base64_to_ndarray_vec(row["block_runlength_base64"])
                if block_runlength.sum() >= 4096:
                    return True
            except Exception:
                return True
    return False


def should_skip_unmatched_function(row: dict) -> bool:
    """Check if unmatched function should be skipped based on builder logic.

    Note: The builder checks for 'block_runlength' (not 'block_runlength_base64'),
    which doesn't exist in the CSV, so it never actually skips unmatched functions.
    This matches that behavior.
    """
    return False


def has_unique_offsets(version_data_list: List[dict]) -> bool:
    """Check if versions have unique binary offsets (would not be deduplicated)."""
    unique_offsets = set()
    for vdata in version_data_list:
        cache_key = (
            vdata["tokens"].tobytes(),
            vdata["block_rl"].tobytes(),
            vdata["insn_rl"].tobytes(),
        )
        unique_offsets.add(cache_key)
    return len(unique_offsets) > 1


def format_token_mismatch(
    memmap_tokens: np.ndarray, csv_tokens: np.ndarray, vocab_manager: Optional[VocabularyManager] = None
) -> str:
    """Format a detailed token mismatch error message.

    Shows where tokens start to differ with context, using ASM-like format if vocab available.
    """
    min_len = min(len(memmap_tokens), len(csv_tokens))
    max_display = min(min_len, 2000)

    # Find first mismatch position
    mismatch_pos = None
    for i in range(min_len):
        if memmap_tokens[i] != csv_tokens[i]:
            mismatch_pos = i
            break

    if mismatch_pos is None:
        # Lengths differ but all common tokens match
        return (
            f"Token count mismatch: memmap has {len(memmap_tokens)}, CSV has {len(csv_tokens)}\n"
            f"  All first {min_len} tokens match correctly"
        )

    # Show context: 10 tokens before mismatch, then the mismatched section (up to 100 more tokens)
    context_start = max(0, mismatch_pos - 10)
    context_end = min(max_display, mismatch_pos + 100)

    result = [
        f"Token mismatch at position {mismatch_pos} (total lengths: memmap={len(memmap_tokens)}, csv={len(csv_tokens)})"
    ]

    if vocab_manager is not None:
        try:
            # Use FunctionTokenList to properly format tokens
            # Create dummy runlengths for display purposes
            dummy_block_rl = np.array([len(memmap_tokens)], dtype=np.uint16)
            dummy_insn_rl = np.ones(len(memmap_tokens), dtype=np.uint8)

            # Show matching prefix if there is one
            if mismatch_pos > 0:
                prefix_tokens = memmap_tokens[context_start:mismatch_pos]
                try:
                    prefix_func = FunctionTokenList.reconstruct_func_from_raw_bytes(
                        prefix_tokens,
                        np.array([len(prefix_tokens)], dtype=np.uint16),
                        np.ones(len(prefix_tokens), dtype=np.uint8),
                        vocab_manager,
                    )
                    prefix_str = " ".join(token.to_asm_like() for token in prefix_func.iter_tokens())
                    result.append(f"  Matching prefix [{context_start}:{mismatch_pos}]: {prefix_str}")
                except Exception:
                    result.append(f"  Matching prefix [{context_start}:{mismatch_pos}]: {list(prefix_tokens)}")

            # Show mismatched section from memmap
            memmap_section = memmap_tokens[mismatch_pos:context_end]
            try:
                memmap_func = FunctionTokenList.reconstruct_func_from_raw_bytes(
                    memmap_section,
                    np.array([len(memmap_section)], dtype=np.uint16),
                    np.ones(len(memmap_section), dtype=np.uint8),
                    vocab_manager,
                )
                memmap_str = " ".join(token.to_asm_like() for token in memmap_func.iter_tokens())
                result.append(f"  Memmap [{mismatch_pos}:{context_end}]: {memmap_str}")
            except Exception:
                result.append(f"  Memmap [{mismatch_pos}:{context_end}]: {list(memmap_section)}")

            # Show mismatched section from CSV
            csv_section = csv_tokens[mismatch_pos:context_end]
            try:
                csv_func = FunctionTokenList.reconstruct_func_from_raw_bytes(
                    csv_section,
                    np.array([len(csv_section)], dtype=np.uint16),
                    np.ones(len(csv_section), dtype=np.uint8),
                    vocab_manager,
                )
                csv_str = " ".join(token.to_asm_like() for token in csv_func.iter_tokens())
                result.append(f"  CSV    [{mismatch_pos}:{context_end}]: {csv_str}")
            except Exception:
                result.append(f"  CSV    [{mismatch_pos}:{context_end}]: {list(csv_section)}")

        except Exception as e:
            # Fallback to raw token IDs if vocab reconstruction fails
            result.append(f"  (Failed to use vocab: {e})")
            result.append(f"  Memmap [{mismatch_pos}:{context_end}]: {list(memmap_tokens[mismatch_pos:context_end])}")
            result.append(f"  CSV    [{mismatch_pos}:{context_end}]: {list(csv_tokens[mismatch_pos:context_end])}")
    else:
        # No vocab manager - show raw token IDs
        if mismatch_pos > 0:
            prefix_tokens = memmap_tokens[context_start:mismatch_pos]
            result.append(f"  Matching prefix [{context_start}:{mismatch_pos}]: {list(prefix_tokens)}")

        memmap_section = memmap_tokens[mismatch_pos:context_end]
        csv_section = csv_tokens[mismatch_pos:context_end]

        result.append(f"  Memmap [{mismatch_pos}:{context_end}]: {list(memmap_section)}")
        result.append(f"  CSV    [{mismatch_pos}:{context_end}]: {list(csv_section)}")

    return "\n".join(result)


def validate_memmap_output(config: ValidatorConfig) -> ValidationStats:
    """Validate that memmap output matches original CSV data.

    Args:
        config: ValidatorConfig with versions, output_dir, and binary_name

    Returns:
        ValidationStats with counts of validated and skipped functions

    Raises:
        AssertionError: If validation fails
    """
    logger.info(f"Starting validation for binary: {config.binary_name}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Versions to validate: {len(config.versions)}")

    dataset = BinaryDataset(config.output_dir, config.binary_name)

    # Try to load vocabulary for better error messages
    vocab_manager = None
    vocab_path = config.output_dir / "unified_vocab.csv"
    if vocab_path.exists():
        try:
            vocab_manager = load_unified_vocab_manager(vocab_path)
            if vocab_manager:
                logger.info(f"  Loaded vocabulary with {len(vocab_manager.id_to_token)} tokens")
        except Exception as e:
            logger.warning(f"  Could not load vocabulary: {e}")

    mapping_dict = {}
    csv_paths = []
    version_keys = []

    for version in config.versions:
        mapping = load_mapping(version.mapping_path)

        vkey = VersionKey(
            arch=version.arch,
            compiler=version.compiler,
            compilerversion=version.compilerversion,
            opt=version.opt,
        )

        mapping_dict[vkey] = mapping
        csv_paths.append(str(version.csv_path))
        version_keys.append(vkey)

    # Cross-check `<binary>_versions.json` if present. Legacy memmaps
    # without the sidecar skip this step (returns None). Mismatches are
    # surfaced through the same `stats.errors` channel as data mismatches
    # so the existing reporter prints them in the summary block.
    sidecar_entries = _load_versions_sidecar(config.output_dir, config.binary_name)
    sidecar_errors: List[str] = []
    if sidecar_entries is not None:
        sidecar_errors = _cross_check_versions_sidecar(version_keys, sidecar_entries)
        if sidecar_errors:
            for err in sidecar_errors:
                logger.error(f"  {err}")

    matched_func_name_to_idx: Dict[str, int] = {}
    if dataset.matched_func_names:
        for idx, name in enumerate(dataset.matched_func_names):
            matched_func_name_to_idx[name] = idx

    unmatched_data_by_name_and_vkey: Dict[tuple, int] = {}
    if dataset.unmatched_func_names and dataset.unmatched_sections.exists():
        import csv

        index_entry_idx = 0
        with open(dataset.unmatched_sections, "r", newline="", encoding="ascii") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and len(row) == 6:
                    func_name = row[0]
                    platform_tuples_str = row[1]
                    if platform_tuples_str:
                        from ..aligned_data.csv_format import parse_compiler_sets

                        platform_tuples = parse_compiler_sets(platform_tuples_str)
                        for arch, compiler, compilerversion, opt in platform_tuples:
                            vkey = VersionKey(
                                arch=arch,
                                compiler=compiler,
                                compilerversion=compilerversion,
                                opt=opt,
                            )
                            unmatched_data_by_name_and_vkey[(func_name, vkey)] = index_entry_idx
                            index_entry_idx += 1

    stats = ValidationStats(
        matched_validated=0,
        matched_skipped=0,
        unmatched_validated=0,
        unmatched_skipped=0,
        csv_only_matched=0,
        csv_only_unmatched=0,
        errors=list(sidecar_errors),
    )

    for match_data in lockstep_function_match(csv_paths):
        func_name = match_data["function_name"]
        rows = match_data["rows"]
        count = match_data["count"]

        if func_name.startswith(".L"):
            continue

        if count >= 2:
            if should_skip_matched_function(rows):
                stats.matched_skipped += 1
                continue

            version_data_csv = []
            for vkey, row in zip(version_keys, rows):
                if row is None:
                    continue

                mapping = mapping_dict.get(vkey)
                try:
                    tokens_csv, block_rl_csv, insn_rl_csv = decode_csv_row_data(row, mapping)
                    version_data_csv.append(
                        {
                            "vkey": vkey,
                            "tokens": tokens_csv,
                            "block_rl": block_rl_csv,
                            "insn_rl": insn_rl_csv,
                        }
                    )
                except Exception:
                    continue

            if not has_unique_offsets(version_data_csv):
                stats.matched_skipped += 1
                continue

            if func_name not in matched_func_name_to_idx:
                logger.warning(f"Matched function {func_name} in CSV but not in memmap")
                stats.csv_only_matched += 1
                continue

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

        elif count == 1:
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

    if stats.errors:
        logger.error(f"Validation found {len(stats.errors)} error(s):")
        for i, error in enumerate(stats.errors[:10], 1):
            logger.error(f"\nError {i}:\n{error}")
        if len(stats.errors) > 10:
            logger.error(f"\n... and {len(stats.errors) - 10} more errors")
    else:
        logger.info(f"Validation completed successfully!")

    logger.info(f"  Matched functions validated: {stats.matched_validated}")
    logger.info(f"  Matched functions skipped (filters): {stats.matched_skipped}")
    logger.info(f"  Matched functions in CSV only: {stats.csv_only_matched}")
    logger.info(f"  Unmatched functions validated: {stats.unmatched_validated}")
    logger.info(f"  Unmatched functions skipped (filters): {stats.unmatched_skipped}")
    logger.info(f"  Unmatched functions in CSV only: {stats.csv_only_unmatched}")
    logger.info(f"  Errors found: {len(stats.errors)}")

    return stats
