"""Per-binary validation orchestrator.

Single concern: drive the per-binary validation flow -- vocab load +
prelude probes + ``BinaryDataset`` open + per-arm v1 invariant checks
+ ``_variants.bin`` cross-check + lockstep per-version token-equality
loop -- and fold every helper's results into one ``ValidationStats``
the caller renders.

Sub-concerns owned by sibling modules:
    * Token-equality compare loops .... :mod:`_validator_arms`
    * Token-mismatch report formatter . :mod:`_validator_mismatch_report`
    * Per-version CSV decoders + skip
      predicates ...................... :mod:`_validator_csv`
    * End-of-run punch-list reporter .. :mod:`_validator_summary`
    * Unmatched ``(name, vkey) -> idx``
      lookup builder .................. :mod:`_unmatched_lookup`
    * v1 invariant checks ............. :mod:`_v1_checks`
    * ``_variants.bin`` cross-check ... :mod:`variants_bin_check`

The unified vocab is loaded ONCE at validator entry (via the shared v1
gate in ``aligned_data.loader.unified_vocab_gate``) so the cross-check
and the dataloader-side error formatter both share the same instance.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import contextlib

from ..aligned_data.loader import BinaryDataset
from ..aligned_data.loader.unified_vocab_gate import load_and_validate_unified_vocab
from ..aligned_data.parsed_record_iter import (
    Matched,
    Unmatched,
    lockstep_records,
    open_parsed_record_iter,
)
from ..memmap_builder import VersionKey
from ._v1_checks import (
    check_csv_prelude,
    check_index_prelude,
    run_v1_post_checks,
)
from ._unmatched_lookup import build_unmatched_index_lookup
from ._validator_arms import compare_matched_arm, compare_unmatched_arm
from ._validator_csv import (
    detect_csv_format_version,
    has_unique_offsets,
    load_mapping,
)
from ._validator_summary import log_validation_summary
from .variants_bin_check import cross_check_variants_bin

logger = logging.getLogger(__name__)


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
    """Configuration for memmap validation.

    ``unified_vocab_path`` defaults to ``<output_dir>/unified_vocab.csv``
    (mirrors ``AlignedDataLoader``'s convention) so the CLI does not need
    to thread an extra arg; an explicit override is supported for tests
    and out-of-tree vocab layouts.
    """

    versions: List[VersionInfo]
    output_dir: Path
    binary_name: str
    unified_vocab_path: Optional[Path] = None


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

    # Unified vocab is a hard dependency of the variant-bin cross-check
    # AND of the dataloader-side variant resolver. Loading + gating once
    # here gives the BinaryDataset and the cross-check the same instance,
    # and surfaces a v2 / missing dataset as a single clear ValueError
    # before any per-binary state materialises.
    vocab_path = (
        config.unified_vocab_path
        if config.unified_vocab_path is not None
        else config.output_dir / "unified_vocab.csv"
    )
    vocab_manager = load_and_validate_unified_vocab(vocab_path)
    logger.info(
        f"  Loaded vocabulary v{vocab_manager.format_version} with "
        f"{len(vocab_manager.id_to_token)} tokens from {vocab_path}"
    )

    # Prelude / magic / version checks run BEFORE ``BinaryDataset``
    # construction: a bad prelude raises ``ValueError`` deep in the
    # metadata loader, so probing first keeps the failure in the
    # error-list path the caller iterates over. ``_index.bin`` (matched
    # arm) is pre-v1 layout (no IDX1 magic; structurally validated by
    # ``read_csv_section_index_arrays`` at ``BinaryDataset`` open
    # below) so only ``_unmatched_index.bin`` gets the IDX1 probe here.
    base = config.output_dir
    bn = config.binary_name
    matched_sections = base / f"{bn}_sections.csv"
    unmatched_sections = base / f"{bn}_unmatched_sections.csv"
    variants_csv = base / f"{bn}_variants.csv"
    unmatched_index = base / f"{bn}_unmatched_index.bin"
    prelude_errors: List[str] = []
    prelude_errors.extend(check_csv_prelude(matched_sections, str(matched_sections)))
    prelude_errors.extend(check_csv_prelude(unmatched_sections, str(unmatched_sections)))
    prelude_errors.extend(check_csv_prelude(variants_csv, str(variants_csv)))
    prelude_errors.extend(check_index_prelude(unmatched_index, str(unmatched_index)))
    if prelude_errors:
        for err in prelude_errors:
            logger.error(f"  {err}")
        return ValidationStats(
            matched_validated=0,
            matched_skipped=0,
            unmatched_validated=0,
            unmatched_skipped=0,
            csv_only_matched=0,
            csv_only_unmatched=0,
            errors=list(prelude_errors),
        )

    dataset = BinaryDataset(
        config.output_dir, config.binary_name, vocab_manager=vocab_manager
    )

    csv_paths = []
    version_keys = []
    mappings = []
    detected_formats: List[int] = []

    for version in config.versions:
        vkey = VersionKey(
            arch=version.arch,
            compiler=version.compiler,
            compilerversion=version.compilerversion,
            opt=version.opt,
        )

        csv_paths.append(str(version.csv_path))
        version_keys.append(vkey)
        mappings.append(load_mapping(version.mapping_path))
        detected_formats.append(detect_csv_format_version(version.csv_path))

    # Refuse mixed v1/v2 inputs: lockstep validation joins per-function
    # rows across all per-version CSVs, so a single mismatched wire
    # format would mean comparing semantically different header
    # schemas. ``open_csv_skip_vocab`` already normalises the prelude,
    # but a mixed set still indicates a broken build pipeline.
    unique_formats = set(detected_formats)
    if len(unique_formats) > 1:
        raise ValueError(
            f"per-version CSVs have inconsistent format versions {sorted(unique_formats)}: "
            f"{list(zip(csv_paths, detected_formats))}"
        )
    csv_format_version = detected_formats[0] if detected_formats else 1
    logger.info(f"  Per-version CSV format: v{csv_format_version}")

    # Cross-check ``<binary>_variants.bin`` against the per-variant CSVs.
    # The cross-check rebuilds each variant's expected axis-string list
    # from the CSV (via VariantInfo.from_csv) and asserts the bin record
    # at the slim CSV's offset decodes (through the unified vocab) to the
    # same list. Errors flow through ``stats.errors`` so the existing
    # reporter prints them in the summary block.
    variants_bin_errors = cross_check_variants_bin(
        config.versions,
        config.output_dir,
        config.binary_name,
        vocab_manager,
    )
    if variants_bin_errors:
        for err in variants_bin_errors:
            logger.error(f"  {err}")

    # Per-record v1 invariant checks (16-byte alignment + zero-pad +
    # pad-placement consistency + record bounds). Run after
    # ``BinaryDataset`` loaded the section arms so the checks reuse the
    # already-decoded starts arrays instead of re-opening ``_index.bin``;
    # records are self-describing so no per-record length array is
    # threaded through (the checks parse the header at each start).
    v1_check_errors = run_v1_post_checks(
        matched_index=dataset.matched_index,
        unmatched_index=dataset.unmatched_index,
        matched_data=dataset.matched_data,
        unmatched_data=dataset.unmatched_data,
        matched_starts=dataset.matched_starts,
        unmatched_starts=dataset.unmatched_starts,
    )
    if v1_check_errors:
        for err in v1_check_errors:
            logger.error(f"  {err}")

    matched_func_name_to_idx: Dict[str, int] = {}
    if dataset.matched_func_names:
        for idx, name in enumerate(dataset.matched_func_names):
            matched_func_name_to_idx[name] = idx

    if dataset.unmatched_func_names and dataset.unmatched_sections.exists():
        unmatched_data_by_name_and_vkey, sidecar_resolution_errors = (
            build_unmatched_index_lookup(
                dataset.unmatched_sections,
                dataset.variants_sidecar,
                version_keys,
                dataset.line_to_name,
            )
        )
    else:
        unmatched_data_by_name_and_vkey = {}
        sidecar_resolution_errors = []

    stats = ValidationStats(
        matched_validated=0,
        matched_skipped=0,
        unmatched_validated=0,
        unmatched_skipped=0,
        csv_only_matched=0,
        csv_only_unmatched=0,
        errors=list(variants_bin_errors)
        + list(v1_check_errors)
        + list(sidecar_resolution_errors),
    )

    with contextlib.ExitStack() as stack:
        wrappers = []
        per_csv_iters = []
        for csv_path, mapping in zip(csv_paths, mappings):
            wrapper, it, _header = open_parsed_record_iter(csv_path, mapping)
            wrappers.append(wrapper)
            per_csv_iters.append(it)
            stack.callback(wrapper.close)

        for item in lockstep_records(per_csv_iters, wrappers):
            if item.func_name.startswith(".L"):
                continue

            if isinstance(item, Matched):
                compare_matched_arm(
                    item,
                    version_keys=version_keys,
                    has_unique_offsets=has_unique_offsets,
                    matched_func_name_to_idx=matched_func_name_to_idx,
                    dataset=dataset,
                    vocab_manager=vocab_manager,
                    stats=stats,
                )
            elif isinstance(item, Unmatched):
                compare_unmatched_arm(
                    item,
                    version_keys=version_keys,
                    unmatched_data_by_name_and_vkey=unmatched_data_by_name_and_vkey,
                    dataset=dataset,
                    vocab_manager=vocab_manager,
                    stats=stats,
                )

    log_validation_summary(stats)
    return stats
