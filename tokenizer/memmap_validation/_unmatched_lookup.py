"""Per-binary ``(func_name, vkey) -> unmatched_index`` lookup builder.

Single concern: walk the unmatched sections CSV (post-restructuring
5-cell layout ``[line_no_b64, variant_refs, called_b64,
inlining_data, indexer_hex]``), resolve the base64 line numbers
through the function-names sidecar, and assign each variant_ref a
running unmatched-arm index that matches the order
``unmatched_index.bin`` records them in.

Extracted from ``validator.validate_memmap_output`` so the
orchestrator stops carrying the prelude-routing + sidecar-resolution
mechanics for the unmatched arm. The pre-restructuring code used a
raw ``open(path) + csv.reader`` and worked by accident (the
``len(row) == 6`` filter happened to skip the 1-cell prelude row);
this module routes the open through
:func:`metadata_loader.open_sections_csv` so the prelude is consumed
before the reader sees the stream.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from ..aligned_data.csv_format import (
    parse_function_line_no,
    parse_variant_refs,
)
from ..aligned_data.loader.metadata_loader import open_sections_csv
from ..aligned_data.loader.variant_resolver import (
    load_variants_offset_to_filename,
)
from ..memmap_builder import VersionKey


def build_unmatched_index_lookup(
    unmatched_sections: Path,
    variants_sidecar: Path,
    version_keys: List[VersionKey],
    line_to_name: Dict[int, str],
) -> Tuple[Dict[tuple, int], List[str]]:
    """Build the per-binary ``(func_name, vkey) -> index_entry_idx`` map.

    The slim ``<binary>_variants.csv`` lists one ``(filename, offset)``
    row per variant in the same registration order as ``version_keys``
    (both derive from ``config.versions``), so the i-th entry by
    ascending offset matches ``version_keys[i]``.

    Returns ``(lookup, errors)``. ``errors`` carries sidecar-resolution
    misses (a row references a sidecar line that the function-names
    sidecar does not list); the caller folds them into ``stats.errors``
    so the validator's punch list surfaces them rather than crashing
    the loop.
    """
    lookup: Dict[tuple, int] = {}
    errors: List[str] = []

    offset_to_filename = load_variants_offset_to_filename(variants_sidecar)
    offset_to_vkey = {
        offset: version_keys[i]
        for i, offset in enumerate(sorted(offset_to_filename.keys()))
    }

    index_entry_idx = 0
    handle, _ = open_sections_csv(unmatched_sections)
    try:
        reader = csv.reader(handle)
        for row in reader:
            if not row or len(row) != 5:
                continue
            line_no = parse_function_line_no(row[0])
            func_name = line_to_name.get(line_no)
            if func_name is None:
                errors.append(
                    f"{unmatched_sections}: row references sidecar line "
                    f"{line_no} absent from function-names sidecar"
                )
                continue
            variant_refs_str = row[1]
            if variant_refs_str:
                for ref in parse_variant_refs(variant_refs_str):
                    vkey = offset_to_vkey[int(ref, 16)]
                    lookup[(func_name, vkey)] = index_entry_idx
                    index_entry_idx += 1
    finally:
        handle.close()
    return lookup, errors
