import contextlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION
from tokenizer.compact_base64_utils import base64_to_ndarray_vec
from tokenizer.vocab_unifier.loader import load_unified_vocab_manager

from ..aligned_data.match import lockstep_function_match
from ._output_files import open_section_outputs
from .function_names import FunctionNamesRegistry
from .passes import (
    build_function_lookup_table,
    process_matched_function_pass1,
    process_unmatched_function_pass1,
    write_matched_sections_pass2,
    write_unmatched_sections_pass2,
)
from .variants import VariantRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VersionKey:
    """Identity of a single build of a binary used as the pairing key
    inside ``build_memmap_files`` and as the lookup key over which
    matched/unmatched function tables are indexed.

    The canonical 4 axes (`arch`, `compiler`, `compilerversion`, `opt`)
    are augmented with `variant_id` so that two builds sharing the
    canonical 4 but differing in opaque sidecar metadata
    (`flag_set`, `hardening`, ...) are *not* coalesced into a single
    version. Legacy callers that don't supply `variant_id` get the
    default `0` and remain behaviorally identical.

    Frozen: hash and equality are auto-derived over all 5 fields.
    """

    arch: str
    compiler: str
    compilerversion: str
    opt: str
    variant_id: int = 0


@dataclass
class BinaryVersionInfo:
    """Information about a specific binary version.

    `extra_metadata` is the opaque pass-through dict reconstructed from
    the per-variant sidecar (`<basename>_meta.json`) during worker
    decoding. Legacy versions without a sidecar carry an empty dict —
    the metadata is forwarded to the per-binary `_variants.bin` record
    encoder (via the unified vocab's variant-axis tokens) and is never
    inspected by the builder itself.
    """

    path: Path
    mapping_path: Path
    arch: str
    compiler: str
    compilerversion: str
    opt: str
    pkg: str = ""
    variant_id: int = 0
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    # ``filename`` is the variant's stable on-disk identifier — the
    # parent folder name of its per-variant CSV (variant folder name
    # for sidecar variants, binary basename for legacy 4-axis). Empty
    # when the caller didn't populate it (e.g. legacy export.py
    # entry-point that has no folder-name notion). Surfaces in the
    # per-group ``_variants.csv`` so consumers can recover the original
    # build's filesystem identity.
    filename: str = ""


def get_mapping(mapping_path: Path):
    """Load mapping file if it exists."""
    if mapping_path and mapping_path.exists():
        with open(mapping_path, "r", encoding="ascii") as f:
            return base64_to_ndarray_vec(f.read())
    return None


def build_memmap_files(
    versions: List[BinaryVersionInfo],
    output_dir: Path,
    binary_name: str,
    unified_vocab_path: Path,
) -> None:
    """Build memory-mapped binary files from aligned CSV data.

    `unified_vocab_path` points at the corpus-wide ``unified_vocab.csv``
    produced by ``tokenizer.vocab_unifier``. It is loaded once here
    (``MEMMAP_FORMAT_VERSION`` required) and threaded into
    ``VariantRegistry`` so the per-variant token records emitted into
    ``<binary>_variants.bin`` can resolve each axis string (``arch:*``,
    ``comp:*``, ``cver:*``, ``opt:*``, plus per-metadata-pair tokens) to
    its assigned uint16 ID. Any non-current-version (or missing) unified
    vocab is rejected loudly here rather than silently corrupting the bin
    via stub IDs.
    """

    logger.info(f"  Output directory: {output_dir}")

    # Unified vocab is a hard dependency of the variant-token encoder
    # (every axis string must already have an assigned uint16 ID before
    # the registry walks its records). Load before constructing the
    # registry so a vocab-shape mismatch fails this group up front
    # instead of mid-record-write.
    unified_vocab = load_unified_vocab_manager(unified_vocab_path)
    if unified_vocab is None:
        raise ValueError(
            f"build_memmap_files: failed to load unified vocab from "
            f"{unified_vocab_path}; cannot encode variant-axis tokens."
        )
    if unified_vocab.format_version != MEMMAP_FORMAT_VERSION:
        raise ValueError(
            f"build_memmap_files: unified vocab at {unified_vocab_path} "
            f"reports format_version={unified_vocab.format_version}; "
            f"v{MEMMAP_FORMAT_VERSION} required for the memmap-output "
            f"chain. Re-run tokenizer.vocab_unifier against the per-binary "
            f"CSV inputs to regenerate."
        )

    # Variant registry: single authority on the `vkey -> 0x<hex>` ref
    # used by every section-CSV row and the warn-log. Built up front
    # so the assignment is fixed before any matched/unmatched pass
    # consults it. ``write_sidecar`` MUST run before the section passes
    # since `ref(vkey)` only returns a usable byte offset once the bin
    # has been written; today's ordering (sidecar -> section passes)
    # already satisfies that invariant.
    variants = VariantRegistry.from_versions(versions, unified_vocab=unified_vocab)
    variants_path = variants.write_sidecar(output_dir, binary_name)
    logger.info(f"  Wrote: {variants_path}")

    mapping_dict = {}
    csv_paths = []
    version_keys = []

    for version in versions:
        mapping = get_mapping(version.mapping_path)

        vkey = VersionKey(
            arch=version.arch,
            compiler=version.compiler,
            compilerversion=version.compilerversion,
            opt=version.opt,
            variant_id=version.variant_id,
        )

        mapping_dict[vkey] = mapping
        csv_paths.append(str(version.path))
        version_keys.append(vkey)

    prefix = binary_name
    unmatched_prefix = f"{binary_name}_unmatched"

    matched_data_entries = []
    unmatched_data_entries = []

    matched_data_path = output_dir / f"{prefix}_data.bin"
    unmatched_data_path = output_dir / f"{unmatched_prefix}_data.bin"
    error_log_path = output_dir / f"{binary_name}.error.log"
    warn_log_path = output_dir / f"{binary_name}.warn.log"

    # ExitStack owns every file handle the build opens so an exception
    # in any phase (pass 1, sidecar emission, pass 2) reliably closes
    # them in reverse open order. Partial output is intentionally left
    # on disk — clean-up on retry is the caller's responsibility, as
    # it has been pre-refactor; only the close-on-exception behaviour
    # changes here.
    function_names_registry = FunctionNamesRegistry()

    with contextlib.ExitStack() as stack:
        logger.info(f"  Creating: {matched_data_path}")
        logger.info(f"  Creating: {unmatched_data_path}")
        logger.info(f"  Creating: {error_log_path}")
        matched_data_file = stack.enter_context(open(matched_data_path, "wb"))
        unmatched_data_file = stack.enter_context(open(unmatched_data_path, "wb"))
        error_log = stack.enter_context(open(error_log_path, "w", encoding="ascii"))

        progress_callback = None
        pbar = None
        if sys.stdout.isatty():
            try:
                from tqdm import tqdm

                total_size = sum(Path(csv_path).stat().st_size for csv_path in csv_paths)
                pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Processing {binary_name}", leave=False)
                # Ensure the bar is closed even if pass 1 throws.
                stack.callback(pbar.close)
                last_bytes = [0]

                def progress_wrapper(current_bytes):
                    delta = current_bytes - last_bytes[0]
                    last_bytes[0] = current_bytes
                    pbar.update(delta)

                progress_callback = progress_wrapper
            except ImportError:
                pass

        for match_data in lockstep_function_match(csv_paths, progress_callback):
            func_name = match_data["function_name"]
            rows = match_data["rows"]
            count = match_data["count"]

            if count >= 2:
                entry = process_matched_function_pass1(
                    func_name,
                    rows,
                    version_keys,
                    mapping_dict,
                    matched_data_file,
                    function_names_registry,
                    error_log=error_log,
                )
                if entry is not None:
                    matched_data_entries.append(entry)
                else:
                    entries = process_unmatched_function_pass1(
                        func_name,
                        rows,
                        version_keys,
                        mapping_dict,
                        unmatched_data_file,
                        function_names_registry,
                        error_log=error_log,
                    )
                    unmatched_data_entries.extend(entries)

            elif count == 1:
                entries = process_unmatched_function_pass1(
                    func_name,
                    rows,
                    version_keys,
                    mapping_dict,
                    unmatched_data_file,
                    function_names_registry,
                    error_log=error_log,
                )
                unmatched_data_entries.extend(entries)

        # Pass 1 done — finalize + emit the sidecar BEFORE pass 2 so
        # pass 2 can resolve every section-CSV function-name cell to
        # its 1-indexed line number.
        function_names_registry.finalize()
        sidecar_path = function_names_registry.write_sidecar(output_dir, binary_name)
        logger.info(f"  Wrote: {sidecar_path}")

        function_lookup = build_function_lookup_table(matched_data_entries, unmatched_data_entries)

        logger.info(f"  Creating: {warn_log_path}")
        warn_log = stack.enter_context(open(warn_log_path, "w", encoding="ascii"))

        matched_outputs = open_section_outputs(output_dir, prefix)
        stack.callback(matched_outputs.close)
        write_matched_sections_pass2(
            matched_data_entries,
            function_lookup,
            matched_outputs.sections_file,
            matched_outputs.index_file,
            warn_log,
            variants,
            function_names_registry,
            error_log=error_log,
        )

        unmatched_outputs = open_section_outputs(output_dir, unmatched_prefix)
        stack.callback(unmatched_outputs.close)
        write_unmatched_sections_pass2(
            unmatched_data_entries,
            function_lookup,
            unmatched_outputs.sections_file,
            unmatched_outputs.index_file,
            warn_log,
            variants,
            function_names_registry,
            error_log=error_log,
        )
