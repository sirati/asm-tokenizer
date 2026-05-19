"""Pass-1 walkers + cross-pass lookup table.

Pass 1 streams the per-binary CSV rows, writes the binary records into
``_data.bin``, and collects the per-function metadata pass 2 needs to
emit section rows and index entries. Pass 2 lives in
:mod:`tokenizer.memmap_builder._pass2` and is re-exported here so the
``builder.py`` import surface is unchanged.

A per-version ``process_function_binary_data`` returning ``None`` is
the sole "skip this version" signal: the encoder caught an
:class:`IndexEntrySkip`, logged the reason into ``<binary>.error.log``,
and truncated the partial data write. Both pass-1 walkers honour the
signal — matched drops the whole function if no version survived;
unmatched simply omits the skipped version. Pass 2 then never sees
those entries, so no section row or index entry is emitted for them.

Both walkers also feed every function name they actually emit (the
header function plus its called-function references) into a shared
:class:`FunctionNamesRegistry`. The registry is finalized between
pass 1 and pass 2 so pass 2 can resolve each name to its 1-indexed
sidecar line number for the base64 indirection written into the
section CSVs.
"""

from typing import Dict, List

from .function_names import FunctionNamesRegistry
from .helpers import (
    get_called_functions_from_row,
    process_function_binary_data,
    should_skip_function_for_matched,
    should_skip_function_for_unmatched,
)
from ._pass2 import (  # re-export so builder.py's import stays one module
    group_unmatched_entries_by_function,
    write_matched_sections_pass2,
    write_unmatched_sections_pass2,
)

__all__ = (
    "build_function_lookup_table",
    "group_unmatched_entries_by_function",
    "process_matched_function_pass1",
    "process_unmatched_function_pass1",
    "write_matched_sections_pass2",
    "write_unmatched_sections_pass2",
)


def process_matched_function_pass1(
    func_name: str,
    rows: List,
    version_keys: List,
    mapping_dict: Dict,
    data_file,
    registry: FunctionNamesRegistry,
    *,
    error_log=None,
) -> dict:
    """Process a matched function in pass 1: write binary data and collect metadata.

    Any per-version ``process_function_binary_data`` call that returns
    ``None`` (encoder cap-overflow logged to ``error_log``) is omitted
    from the collected ``version_data``. If all versions were skipped
    the whole function is dropped (returns ``None``) so pass 2 emits
    neither a section row nor an index entry for it.

    ``registry`` is the shared :class:`FunctionNamesRegistry` populated
    in pass 1 so pass 2 can resolve every section-CSV function-name
    cell to its 1-indexed sidecar line number. Names are added only
    when the function ultimately survives all skip predicates and the
    encoder — names that never reach the section CSV would otherwise
    bloat the sidecar without ever being looked up.
    """
    if func_name.startswith(".L"):
        return None

    if should_skip_function_for_matched(rows):
        return None

    dedup_cache = {}
    all_called_by_vkey = {}

    for vkey, row in zip(version_keys, rows):
        if row is not None:
            all_called_by_vkey[vkey] = get_called_functions_from_row(row)

    unique_called = sorted(set(fn for called_list in all_called_by_vkey.values() for fn in called_list))

    version_data = []
    for vkey, row in zip(version_keys, rows):
        if row is None:
            continue

        called = all_called_by_vkey[vkey]
        mapping = mapping_dict.get(vkey)
        binary_data = process_function_binary_data(
            row,
            mapping,
            data_file,
            dedup_cache,
            func_name=func_name,
            error_log=error_log,
        )
        if binary_data is None:
            continue

        version_data.append(
            {
                "vkey": vkey,
                "called": called,
                "data_offset": binary_data.data_offset,
                "data_len": binary_data.data_len,
                "token_len": binary_data.token_len,
            }
        )

    if not version_data:
        return None

    unique_offsets = set(vdata["data_offset"] for vdata in version_data)
    if len(unique_offsets) == 1:
        return None

    # Function survived encoding; record the header name + every
    # called-name pass 2 will write into the section CSV. The registry
    # dedupes internally, so adding callees on every emit is cheap.
    registry.add(func_name)
    for called_name in unique_called:
        registry.add(called_name)

    return {
        "func_name": func_name,
        "unique_called": unique_called,
        "version_data": version_data,
    }


def process_unmatched_function_pass1(
    func_name: str,
    rows: List,
    version_keys: List,
    mapping_dict: Dict,
    data_file,
    registry: FunctionNamesRegistry,
    *,
    error_log=None,
) -> List[dict]:
    """Process an unmatched function in pass 1: write binary data and collect metadata.

    Entries whose ``process_function_binary_data`` returned ``None``
    (encoder cap-overflow logged to ``error_log``) are omitted from the
    returned list — pass 2 only emits section rows and index entries
    for the versions that survived encoding.

    ``registry`` is the shared :class:`FunctionNamesRegistry`. The
    header function name is recorded once per surviving version
    (the registry dedupes), and every called-function name a
    surviving version references is recorded so pass 2 can resolve
    the section-CSV cells back to 1-indexed sidecar line numbers.

    The encoder's contract is "return ``None`` on cap-overflow after
    truncating the partial write and logging via ``error_log``" — so
    no exception should normally escape ``process_function_binary_data``.
    Any exception that does escape is an IO error or a programmer bug
    in the encoder chain, and is left to propagate so it surfaces
    instead of being silently swallowed.
    """
    if func_name.startswith(".L"):
        return []

    unmatched_entries = []
    for vkey, row in zip(version_keys, rows):
        if row is None:
            continue

        if should_skip_function_for_unmatched(row):
            continue

        dedup_cache = {}
        called = get_called_functions_from_row(row)
        mapping = mapping_dict.get(vkey)

        binary_data = process_function_binary_data(
            row,
            mapping,
            data_file,
            dedup_cache,
            func_name=func_name,
            error_log=error_log,
        )
        if binary_data is None:
            continue

        unmatched_entries.append(
            {
                "func_name": func_name,
                "vkey": vkey,
                "data_offset": binary_data.data_offset,
                "data_len": binary_data.data_len,
                "token_len": binary_data.token_len,
                "called": set(called),
            }
        )
        registry.add(func_name)
        for called_name in called:
            registry.add(called_name)

    return unmatched_entries


def build_function_lookup_table(matched_data_entries: List[dict], unmatched_data_entries: List[dict]) -> dict:
    """Build lookup table: {(func_name, vkey): (offset, length, is_matched)}."""
    function_lookup = {}

    for entry in matched_data_entries:
        func_name = entry["func_name"]
        version_data = entry["version_data"]
        for vdata in version_data:
            vkey = vdata["vkey"]
            function_lookup[(func_name, vkey)] = (
                vdata["data_offset"],
                vdata["data_len"],
                1,
            )

    for entry in unmatched_data_entries:
        func_name = entry["func_name"]
        vkey = entry["vkey"]
        function_lookup[(func_name, vkey)] = (
            entry["data_offset"],
            entry["data_len"],
            0,
        )

    return function_lookup
