"""Per-binary ``(func_name, vkey) -> unmatched_index`` lookup builder.

Single concern: walk the unmatched sections CSV (current 6-cell layout
``[line_no_b64, variant_refs, called_line_nos, call_targets,
indexer_hex, duplicated]``), resolve the base64 line numbers through the
function-names sidecar, and assign each variant_ref a running
unmatched-arm index that matches the order ``unmatched_index.bin``
records them in. Only ``line_no_b64`` and ``variant_refs`` are consumed
here; the ``called_line_nos``, ``call_targets`` and ``duplicated`` cells
are not touched by this lookup builder, so the typed-cell rewrites and
the duplicated-marker column do not affect this code path.

The trailing ``duplicated`` cell is the section-level ``0``/``1`` marker
(authoritative copy lives in ``sections.bin``); this CSV column only
mirrors it for the human-readable debug catalog. The pre-marker layout
carried a 5th-and-final ``indexer_hex`` cell (5 cells total); the
even-older pre-restructuring layout instead carried a trailing 8-hex
``length`` cell (also 6 cells). The duplicated marker is always exactly
``0`` or ``1``, so a 6-cell row whose last cell is NOT ``0``/``1`` is
the legacy length-bearing layout and is rejected with a regenerate
pointer.

Extracted from ``validator.validate_memmap_output`` so the
orchestrator stops carrying the prelude-routing + sidecar-resolution
mechanics for the unmatched arm. The open routes through
:func:`metadata_loader.open_sections_csv` so the ``# format=N`` prelude
is consumed before the reader sees the stream.
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
            if not row:
                continue
            if len(row) == 6 and row[5] not in ("0", "1"):
                # 6 cells but the trailing cell is not the duplicated
                # marker → the even-older length-bearing layout.
                raise ValueError(
                    f"{unmatched_sections}: legacy 6-cell row encountered "
                    "(trailing cell is not the 0/1 duplicated marker, so it "
                    "predates the current layout); re-run memmap_builder on "
                    "the per-binary CSVs to regenerate"
                )
            if len(row) != 6:
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
