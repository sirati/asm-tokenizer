"""Per-binary ``_variants.bin`` cross-check.

Single concern: prove that the variant-axis token records the
memmap_builder emitted into ``<binary>_variants.bin`` decode (through
the unified vocab) back to the axis strings the per-variant CSVs imply.

Flow per variant:

  1. Reconstruct the canonical variant identity (canonical-4 axes +
     ``extra_metadata`` + filename) from the per-variant
     ``_output.csv`` via ``VariantInfo.from_csv``. The classmethod is
     the single source of truth for filename + sidecar parsing; the
     validator must not parallel-implement it.
  2. Build the expected ordered axis-string list via
     ``build_axis_strings`` — same helper the encoder used at write
     time, so a positional mismatch is impossible unless the bin was
     corrupted or the vocab was rewritten under it.
  3. Look up the variant's byte offset in the slim CSV's
     ``filename -> offset`` table.
  4. Memmap-slice the bin record at that offset (via ``read_record``)
     and resolve each ID through the unified vocab back to its
     registered string.
  5. Assert the recovered string list equals the expected list.

Missing ``_variants.bin`` / slim CSV / unknown-filename / decoder
error all surface as explicit validation-error strings — silent skip
would hide the very kind of drift this cross-check exists to catch.

The unified vocab is passed in by the caller (loaded ONCE per
validator entry via ``aligned_data.loader.unified_vocab_gate``), so a
v2 / missing dataset short-circuits at the validator boundary instead
of producing per-binary decode errors here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

from tokenizer.aligned_data.loader.variant_resolver import (
    load_variants_offset_to_filename,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.variant_info import VariantInfo
from tokenizer.variant_tokens.prefixes import build_axis_strings
from tokenizer.variant_tokens.record import read_record

if TYPE_CHECKING:  # pragma: no cover - import-cycle break
    from .validator import VersionInfo


def cross_check_variants_bin(
    versions: List["VersionInfo"],
    output_dir: Path,
    binary_name: str,
    vocab_manager: VocabularyManager,
) -> List[str]:
    """Return a list of cross-check error strings (empty on success).

    See module docstring for the per-variant procedure. Errors flow
    back through ``ValidationStats.errors`` so the CLI's summary block
    surfaces them alongside data-token mismatches without a separate
    reporter.
    """
    errors: List[str] = []
    bin_path = output_dir / f"{binary_name}_variants.bin"
    slim_csv_path = output_dir / f"{binary_name}_variants.csv"

    if not bin_path.exists():
        errors.append(
            f"_variants.bin missing for {binary_name} at {bin_path}; "
            "the memmap_builder did not emit the variant-axis bin"
        )
        return errors
    if not slim_csv_path.exists():
        errors.append(
            f"_variants.csv missing for {binary_name} at {slim_csv_path}; "
            "cannot resolve variant filenames to bin offsets"
        )
        return errors

    filename_to_offset = _invert_offset_to_filename(
        load_variants_offset_to_filename(slim_csv_path)
    )

    bin_mmap = np.memmap(str(bin_path), dtype=np.uint8, mode="r")
    try:
        for idx, version in enumerate(versions):
            errors.extend(
                _cross_check_one_variant(
                    idx, version, filename_to_offset, bin_mmap, vocab_manager
                )
            )
    finally:
        _close_memmap(bin_mmap)
    return errors


def _cross_check_one_variant(
    idx: int,
    version: "VersionInfo",
    filename_to_offset: Dict[str, int],
    bin_mmap: np.ndarray,
    vocab_manager: VocabularyManager,
) -> List[str]:
    """Per-variant cross-check body. Returns 0..N error strings."""
    try:
        variant_info = VariantInfo.from_csv(version.csv_path)
    except (ValueError, OSError) as exc:
        return [
            f"_variants.bin cross-check: version {idx} "
            f"({version.csv_path.name}) failed to reconstruct "
            f"VariantInfo: {exc}"
        ]

    expected_strings = build_axis_strings(variant_info)
    filename = variant_info.filename
    if filename not in filename_to_offset:
        return [
            f"_variants.bin cross-check: filename {filename!r} (from "
            f"{version.csv_path.name}) not present in the slim "
            f"_variants.csv offset table"
        ]

    offset = filename_to_offset[filename]
    try:
        record = read_record(bin_mmap, offset)
    except (AssertionError, ValueError, IndexError) as exc:
        return [
            f"_variants.bin cross-check: read_record at offset "
            f"0x{offset:x} for {filename!r} failed: {exc}"
        ]

    n_tokens = int(record[0])
    actual_strings = [
        vocab_manager.get_token_str(int(tid)) for tid in record[1:1 + n_tokens]
    ]
    if actual_strings != expected_strings:
        return [
            f"_variants.bin cross-check: variant {filename!r} at "
            f"offset 0x{offset:x} decoded to {actual_strings!r}, "
            f"expected {expected_strings!r}"
        ]
    return []


def _invert_offset_to_filename(offset_to_filename: Dict[int, str]) -> Dict[str, int]:
    """Build the reverse ``filename -> offset`` lookup table.

    The slim CSV's canonical direction is ``offset -> filename`` (the
    dataloader's hot path needs filename attached to a memmap slice).
    The validator works in the other direction (it has the filename from
    ``VariantInfo.from_csv`` and needs the offset), so invert once at
    entry instead of scanning the table per variant.
    """
    return {filename: offset for offset, filename in offset_to_filename.items()}


def _close_memmap(mmap_obj: Any) -> None:
    """Best-effort release of a numpy memmap's underlying mmap handle.

    Mirrors ``aligned_data.loader.session._close_memmap`` — relying on
    GC alone leaks the FD on long-lived workers.
    """
    inner = getattr(mmap_obj, "_mmap", None)
    if inner is not None:
        try:
            inner.close()
        except Exception:  # pragma: no cover - defensive
            pass
