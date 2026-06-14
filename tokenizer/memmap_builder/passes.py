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

from typing import Dict, Iterable, List, Optional, Set, Tuple

from tokenizer.aligned_data._writers import assemble_function_record
from tokenizer.aligned_data.parsed_record_iter import Matched, ParsedRecord, Unmatched

from ._dedup import ArmDedupState, dedup_and_write
from ._extern_library_merge import merge_extern_libraries
from ._pass2 import (  # re-export so builder.py's import stays one module
    group_unmatched_entries_by_function,
    write_matched_sections_pass2,
    write_unmatched_sections_pass2,
)
from ._typed_called_union import category_grouped_first_seen_union
from .function_names import FunctionNamesRegistry

__all__ = (
    "build_function_lookup_table",
    "collect_sectioned_func_names",
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
    ``unique_called`` (category-grouped first-seen union of callees
    across variants — LOCAL → PLT → EXT blocks, intra-category
    encounter-ordered, anchored on the lowest-index variant that first
    introduced each callee; see
    :func:`_typed_called_union.category_grouped_first_seen_union`), and
    ``version_data`` (per-variant offset/length plus the variant's
    local callee set). The shape matches what
    :func:`_pass2.write_matched_sections_pass2` consumes.
    """
    func_name = matched.func_name
    if func_name.startswith(".L"):
        return None

    unique_called = category_grouped_first_seen_union(
        matched.records[variant_index].called_funcs
        for variant_index in sorted(matched.records)
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
                # Per-variant typed callee iterable in the parsed-record's
                # encoder allocation order (categories concatenated LOCAL
                # -> PLT -> EXT). Stored as a list so the downstream
                # per-variant inlining-data emit can walk it in encounter
                # order; the matched-arm table-level order comes from
                # ``unique_called`` separately (plan Decisions 20 + 21).
                "called": list(rec.called_funcs),
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

    extern_libraries = merge_extern_libraries(
        (matched.records[i].extern_libraries for i in sorted(matched.records)),
        func_name=func_name,
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
                # Per-stream occurrence ordinal of THIS body for its
                # canonical name. Unique names top out at 0; a DUPLICATED
                # name (per-TU static initializers, thunks) bumps it per
                # divergent body. The unmatched grouper keys on
                # ``(func_name, occurrence)`` so the kth distinct body's
                # per-stream variants fold into one section instead of all
                # same-name bodies collapsing into a single giant group.
                "occurrence": rec.occurrence,
                "vkey": version_keys[variant_index],
                "data_offset": offset,
                "data_len": total,
                "token_len": int(rec.tokens.size),
                # Per-variant typed callee iterable in the parsed-record's
                # encoder allocation order. Stored as a list so the
                # unmatched grouper's order-preserving union (and the
                # downstream per-variant inlining-data emit) walk it in
                # encounter order (plan Decisions 20 + 21).
                "called": list(rec.called_funcs),
                "extern_libraries": dict(rec.extern_libraries),
            }
        )
        registry.add(func_name)
        for called_name, _type in rec.called_funcs:
            registry.add(called_name)

    return entries


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

    The record's ``entry_idx`` is sourced from
    ``arm_state.n_entries_emitted`` — the next available encounter-order
    ordinal. Dedup hits do NOT advance the counter (the existing
    record's idx is the one that matters); ``_dedup.dedup_and_write``
    owns that bookkeeping on the actual-write branches.
    """
    record_bytes = assemble_function_record(
        rec.tokens,
        rec.block_runlength,
        rec.insn_runlength,
        entry_idx=arm_state.n_entries_emitted,
        func_name=func_name,
        error_log=error_log,
    )
    if record_bytes is None:
        return None
    return dedup_and_write(arm_state, record_bytes, rec.content_hash)


def build_function_lookup_table(
    matched_data_entries: "Iterable[dict]",
    unmatched_data_entries: "Iterable[dict]",
) -> dict:
    """Build lookup table: ``{(func_name, vkey): (offset, length, is_matched)}``.

    Both arguments are consumed by a single forward iteration each, so a
    streaming source (e.g. an :class:`~._entry_spool.EntrySpool`) is
    accepted in place of a materialised list — the table is the only
    whole-binary structure that survives, and it is far smaller than the
    nested entry payloads it indexes.
    """
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


def collect_sectioned_func_names(
    matched_data_entries: "Iterable[dict]",
    unmatched_data_entries: "Iterable[dict]",
) -> "Tuple[Set[str], Set[str]]":
    """Derive ``(matched_func_names, sectioned_func_names)`` from the arms.

    ``matched_func_names`` is every function name surviving the matched
    arm; ``sectioned_func_names`` is the union with the unmatched arm's
    names — the set of names whose section will land in
    ``<binary>_sections.bin``. Both are whole-binary cross-pass tables
    pass 2 needs before emitting any section, so they are derived here in
    one forward pass over each arm rather than via a list comprehension
    over a retained entry list (which would pin the whole corpus in RAM).
    """
    matched_func_names: "Set[str]" = {
        entry["func_name"] for entry in matched_data_entries
    }
    sectioned_func_names: "Set[str]" = set(matched_func_names)
    sectioned_func_names.update(
        entry["func_name"] for entry in unmatched_data_entries
    )
    return matched_func_names, sectioned_func_names
