"""Pass-1 walkers + cross-pass lookup table.

Pass 1 streams the per-binary CSV rows (already parsed + hashed by the
per-CSV iterator in :mod:`tokenizer.aligned_data.parsed_record_iter`)
and writes binary records into the per-arm ``_data.bin`` via the global
content-addressed dedup helper (:mod:`tokenizer.memmap_builder._dedup`).
Pass 2 lives in :mod:`tokenizer.memmap_builder._pass2` and is re-exported
here so the ``builder.py`` import surface is unchanged.

Both walkers feed every function name they actually emit (header
function + called-function references) into a shared
:class:`FunctionNamesRegistry` so pass 2 can resolve each name to its
1-indexed sidecar line number for the base64 indirection written into
the section CSVs.

Encoder-skip handling: ``assemble_function_record`` returns ``None`` on
an :class:`IndexEntrySkip` cap overflow; the corresponding variant is
dropped from this function's pass-2 output. Matched drops the whole
function if no variant survived; unmatched simply omits the offending
variant.
"""

from typing import Dict, List, Optional

from tokenizer.aligned_data._writers import assemble_function_record
from tokenizer.aligned_data.parsed_record_iter import Matched, ParsedRecord, Unmatched

from ._dedup import ArmDedupState, dedup_and_write
from ._pass2 import (  # re-export so builder.py's import stays one module
    group_unmatched_entries_by_function,
    write_matched_sections_pass2,
    write_unmatched_sections_pass2,
)
from .function_names import FunctionNamesRegistry

__all__ = (
    "build_function_lookup_table",
    "group_unmatched_entries_by_function",
    "process_matched_function",
    "process_unmatched_function",
    "write_matched_sections_pass2",
    "write_unmatched_sections_pass2",
)


def process_matched_function(
    matched: Matched,
    version_keys: List,
    arm_state: ArmDedupState,
    registry: FunctionNamesRegistry,
    *,
    error_log=None,
) -> Optional[dict]:
    """Run matched pass-1 on one function; return entry dict or ``None``.

    Returns ``None`` on any of:

    * function name starts with ``.L`` (local label);
    * every variant's encoder raised ``IndexEntrySkip``;
    * every variant deduplicated to the same offset (no inter-variant
      signal — the function is dropped from matched and the caller
      re-routes it through :func:`process_unmatched_function`).

    On success the returned dict carries ``func_name``,
    ``unique_called`` (sorted union of callees across surviving
    variants), and ``version_data`` (per-variant offset/length plus
    the variant's local callee set). The shape matches what
    :func:`_pass2.write_matched_sections_pass2` consumes.
    """
    func_name = matched.func_name
    if func_name.startswith(".L"):
        return None

    unique_called = sorted(
        {pair for rec in matched.records.values() for pair in rec.called_funcs},
        key=lambda nt: (nt[0], nt[1].value),
    )

    version_data = []
    for variant_index, rec in matched.records.items():
        emit = _emit_record(rec, arm_state, func_name=func_name, error_log=error_log)
        if emit is None:
            continue
        offset, total = emit
        version_data.append(
            {
                "vkey": version_keys[variant_index],
                "called": set(rec.called_funcs),
                "data_offset": offset,
                "data_len": total,
                "token_len": int(rec.tokens.size),
            }
        )

    if not version_data:
        return None

    unique_offsets = {vdata["data_offset"] for vdata in version_data}
    if len(unique_offsets) == 1:
        return None

    extern_libraries = _union_extern_libraries(
        matched.records, func_name=func_name, error_log=error_log
    )

    registry.add(func_name)
    for called_name, _type in unique_called:
        registry.add(called_name)

    return {
        "func_name": func_name,
        "unique_called": unique_called,
        "extern_libraries": extern_libraries,
        "version_data": version_data,
    }


def process_unmatched_function(
    func_name: str,
    records: "Dict[int, ParsedRecord]",
    version_keys: List,
    arm_state: ArmDedupState,
    registry: FunctionNamesRegistry,
    *,
    error_log=None,
) -> List[dict]:
    """Run unmatched pass-1 on one function across its variants.

    Two entry conditions cover both calling patterns:

    * Iterator emitted :class:`Unmatched` (one variant) — caller passes
      ``records = {variant_index: record}``.
    * Matched failed in :func:`process_matched_function` and the caller
      falls back through this path — caller passes the full
      ``matched.records`` dict.

    Returns one entry dict per surviving variant. The shape matches what
    :func:`_pass2.write_unmatched_sections_pass2` consumes via
    :func:`group_unmatched_entries_by_function`.
    """
    if func_name.startswith(".L"):
        return []

    entries = []
    for variant_index, rec in records.items():
        emit = _emit_record(rec, arm_state, func_name=func_name, error_log=error_log)
        if emit is None:
            continue
        offset, total = emit
        entries.append(
            {
                "func_name": func_name,
                "vkey": version_keys[variant_index],
                "data_offset": offset,
                "data_len": total,
                "token_len": int(rec.tokens.size),
                "called": set(rec.called_funcs),
                "extern_libraries": dict(rec.extern_libraries),
            }
        )
        registry.add(func_name)
        for called_name, _type in rec.called_funcs:
            registry.add(called_name)

    return entries


def _union_extern_libraries(
    records: "Dict[int, ParsedRecord]",
    *,
    func_name: str,
    error_log,
) -> "dict[str, str]":
    """Union per-variant ``extern_libraries`` dicts for one function.

    Iterates variants in sorted ``variant_index`` order for determinism.
    First writer wins on conflict: if two variants report different
    libraries for the same EXTERN callee, log a single warning into
    ``error_log`` (if open) and keep the first variant's value. A repeat
    of the same library is a no-op.
    """
    merged: dict[str, str] = {}
    for variant_index in sorted(records):
        rec = records[variant_index]
        for name, library in rec.extern_libraries.items():
            existing = merged.get(name)
            if existing is None:
                merged[name] = library
            elif existing != library:
                if error_log is not None:
                    error_log.write(
                        f"WARN: function {func_name} extern library mismatch "
                        f"across variants: {dict(rec.extern_libraries)}\n"
                    )
    return merged


def _emit_record(
    rec: ParsedRecord,
    arm_state: ArmDedupState,
    *,
    func_name: str,
    error_log,
) -> "Optional[tuple[int, int]]":
    """Assemble + dedup + write one record. Return ``(offset, length)`` or ``None``.

    Encapsulates the encoder-skip handling for the two walker arms: a
    cap overflow logs into ``error_log`` and returns ``None`` so the
    caller drops the variant; on success the dedup helper's return is
    passed through.
    """
    record_bytes = assemble_function_record(
        rec.tokens,
        rec.block_runlength,
        rec.insn_runlength,
        func_name=func_name,
        error_log=error_log,
    )
    if record_bytes is None:
        return None
    return dedup_and_write(arm_state, record_bytes, rec.content_hash)


def build_function_lookup_table(
    matched_data_entries: List[dict],
    unmatched_data_entries: List[dict],
) -> dict:
    """Build lookup table: ``{(func_name, vkey): (offset, length, is_matched)}``."""
    function_lookup = {}

    for entry in matched_data_entries:
        func_name = entry["func_name"]
        for vdata in entry["version_data"]:
            function_lookup[(func_name, vdata["vkey"])] = (
                vdata["data_offset"],
                vdata["data_len"],
                1,
            )

    for entry in unmatched_data_entries:
        function_lookup[(entry["func_name"], entry["vkey"])] = (
            entry["data_offset"],
            entry["data_len"],
            0,
        )

    return function_lookup
