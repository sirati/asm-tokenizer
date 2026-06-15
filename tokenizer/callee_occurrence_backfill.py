"""Stamp the callee ``occurrence`` ordinal onto resolved local call-targets.

Concern
-------
A v2 per-binary CSV emits, for each function, a ``local_funcs`` metadata
list whose entries carry the callee's canonical ``name`` and resolved
entry ``addr`` (see ``constant_handler.emitters_v2._emit_local_func``).
When several distinct function bodies share one canonical name (per-TU
static initializers, anon-namespace collisions, LTO clones that diverge),
the build side groups them into same-FID sibling sections keyed by
``(name, occurrence)`` (``memmap_builder._pass2.group_unmatched_entries_by_function``).
A call into such a duplicated name is otherwise unresolvable on the build
side and gets stamped ``MISSING_VARIANT_INDEX``.

This module closes that gap on the PRODUCER side: given the finished CSV
and a map from callee entry address to the ``occurrence`` ordinal that
went into that body's CSV ``occurrence`` column, it injects an
``"occurrence"`` field onto every ``local_funcs`` entry whose ``addr`` is
a known duplicated-name body. The result lets the build side resolve the
call to the specific sibling instead of the missing sentinel.

Module boundary
---------------
This module owns ONE concern: the in-place occurrence stamp on a finished
v2 CSV. It knows nothing about tokenization, the deduper, the string /
range sidecars, or how the address->occurrence map was built. The caller
(the tokenize finalize step) builds the map (it already owns occurrence
assignment) and hands it across this boundary. The map's CONTENT encodes
the policy: only addresses belonging to a DUPLICATED canonical name are
present (a call into a non-duplicated name resolves by name alone and
needs no disambiguator), so an empty map means "nothing to disambiguate"
and the CSV is left byte-for-byte untouched.

Byte-determinism
----------------
A v2 record is always exactly one physical line: every cell is
newline-free by construction (base64 token blobs, ASCII-sanitised
canonical names, and a no-indent ``json.dumps`` metadata cell none of
which can contain ``0x0A``). Function rows that receive NO injection are
therefore streamed through VERBATIM as their original raw line -- no
CSV/JSON round-trip, so their bytes cannot drift. Only rows that actually
gain an ``occurrence`` are re-serialized, and their metadata cell is
re-dumped with the EXACT ``json.dumps`` kwargs
``main_loop._build_v2_metadata_json`` uses (``separators=(",", ":")``,
default ``ensure_ascii=True``, no ``sort_keys``) so an injected cell
differs from the original only by the added key. The optional
``version=`` prelude, the interleaved ``vocabulary`` rows, and the header
row are all passed through verbatim (classified via the shared detectors
in ``tokenizer.aligned_data.match``).
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import List, Mapping, Optional

from tokenizer.aligned_data.match import is_version_prelude_row, is_vocab_row

# Large functions carry ``tokens_base64`` cells well past the default
# 131072-byte csv field limit; mirror the raise the readers in
# ``aligned_data.match`` apply at module-load so this module's own
# ``csv.reader`` does not reject those rows.
csv.field_size_limit(sys.maxsize)

# The metadata JSON cell is the LAST column of a v2 function row (header:
# function_name, occurrence, tokens_base64, block_runlength_base64,
# instruction_runlength_base64, metadata).
_METADATA_COLUMN = -1
_LOCAL_FUNCS_KEY = "local_funcs"
_ADDR_KEY = "addr"
_OCCURRENCE_KEY = "occurrence"
_HEADER_FIRST_CELL = "function_name"


def _parse_one_record(raw_line: str) -> List[str]:
    """Parse a single physical CSV line into its field list.

    Safe because v2 records never span multiple physical lines (no cell
    can contain a newline -- see the module docstring).
    """
    return next(csv.reader(io.StringIO(raw_line)))


def _serialize_metadata(metadata: dict) -> str:
    """Re-dump a metadata cell with main_loop's exact ``json.dumps`` kwargs.

    MUST mirror ``main_loop._build_v2_metadata_json`` byte-for-byte so an
    injected cell differs from the original only by the added
    ``occurrence`` key.
    """
    return json.dumps(metadata, separators=(",", ":"))


def _inject_into_cell(
    metadata_cell: str, addr_to_occurrence: Mapping[int, int]
) -> Optional[str]:
    """Return a rewritten metadata cell, or ``None`` if nothing changed.

    Parses the cell, stamps ``occurrence`` onto each ``local_funcs`` entry
    whose ``addr`` is present in ``addr_to_occurrence`` (and therefore
    belongs to a duplicated canonical name), and re-serializes. Returns
    ``None`` when the cell has no ``local_funcs`` block or none of its
    entries match -- the caller then passes the original row through
    verbatim (no byte churn).
    """
    metadata = json.loads(metadata_cell)
    local_funcs = metadata.get(_LOCAL_FUNCS_KEY)
    if not local_funcs:
        return None

    changed = False
    for entry in local_funcs:
        addr_hex = entry.get(_ADDR_KEY)
        if addr_hex is None:
            continue
        try:
            addr = int(addr_hex, 16) if isinstance(addr_hex, str) else int(addr_hex)
        except (TypeError, ValueError):
            continue
        occurrence = addr_to_occurrence.get(addr)
        if occurrence is None:
            continue
        # Append the disambiguator (deterministic position after the
        # existing name/addr keys). Idempotent: a re-run overwrites with
        # the same value.
        entry[_OCCURRENCE_KEY] = occurrence
        changed = True

    if not changed:
        return None
    return _serialize_metadata(metadata)


def _patched_row_text(row: List[str], new_cell: str) -> str:
    """Serialize one function row with its metadata cell replaced.

    Uses ``csv.writer`` with the SAME dialect the producer used
    (``lineterminator='\\n'``, default QUOTE_MINIMAL) so the rewritten
    row's quoting matches what the producer would have emitted for the
    new cell.
    """
    patched = list(row)
    patched[_METADATA_COLUMN] = new_cell
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerow(patched)
    return buffer.getvalue()


def backfill_callee_occurrence(
    csv_path: Path, addr_to_occurrence: Mapping[int, int]
) -> None:
    """Stamp callee ``occurrence`` onto duplicated-name local call-targets.

    Args:
        csv_path: A finished v2 per-binary CSV (already flushed/closed).
        addr_to_occurrence: Map from callee entry address to its CSV
            ``occurrence`` column value, containing ONLY addresses whose
            canonical name is duplicated within this binary. Empty => no
            duplicated names => nothing to disambiguate.

    No-op (the file is not touched at all -- no temp file, no rename)
    when ``addr_to_occurrence`` is empty. Otherwise rewrites the CSV
    atomically: a sibling temp file in the same directory is filled, then
    ``os.replace``d over the original. A failure mid-rewrite leaves the
    original intact and propagates (a half-done disambiguation is a
    correctness defect, not something to swallow).
    """
    if not addr_to_occurrence:
        # Guardrail: common case (no duplicated canonical name) costs
        # zero -- no rewrite pass, no temp file.
        return

    csv_path = Path(csv_path)
    tmp_path = csv_path.with_name(csv_path.name + ".occbackfill.tmp")
    header_done = False

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as src, open(
            tmp_path, "w", newline="", encoding="utf-8"
        ) as dst:
            for raw_line in src:
                if not raw_line.strip():
                    # Defensive: preserve any stray blank line verbatim.
                    dst.write(raw_line)
                    continue

                row = _parse_one_record(raw_line)

                if not header_done:
                    # Optional ``version=`` prelude then the header row;
                    # both pass through verbatim. The header's first cell
                    # is ``function_name``; after it, data rows begin.
                    if is_version_prelude_row(row):
                        dst.write(raw_line)
                        continue
                    if row[0] == _HEADER_FIRST_CELL:
                        header_done = True
                        dst.write(raw_line)
                        continue
                    # No prelude/header seen yet but this isn't one => a
                    # malformed/headerless file; fall through and treat as
                    # data so we never silently drop content.
                    header_done = True

                if is_vocab_row(row):
                    # Interleaved vocabulary definition rows: verbatim.
                    dst.write(raw_line)
                    continue

                new_cell = _inject_into_cell(row[_METADATA_COLUMN], addr_to_occurrence)
                if new_cell is None:
                    # Untouched rows stream through as their original raw
                    # text -- no JSON/CSV round-trip, no byte drift.
                    dst.write(raw_line)
                else:
                    dst.write(_patched_row_text(row, new_cell))
        os.replace(tmp_path, csv_path)
    except BaseException:
        # Atomicity: leave the original intact, drop the partial temp,
        # and re-raise (failures must surface).
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
