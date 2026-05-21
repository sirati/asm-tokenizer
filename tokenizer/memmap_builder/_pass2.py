"""Pass-2 section + index emitters (matched + unmatched).

Pass 1 (``tokenizer.memmap_builder.passes``) writes ``_data.bin``
records and collects per-function metadata; pass 2 emits the section
CSVs, per-arm index files, and the per-binary ``<binary>_sections.bin``
catalog (the new BIN consumed by the v4-cutover dataloader).

Layout responsibilities:

* Matched header rows + unmatched first cell carry the base64 of the
  function's line number in ``<binary>_function_names.txt``; called-
  funcs cells are comma-joined base64 line numbers. Raw function names
  no longer appear in either section CSV.
* Matched variant rows carry a single ``indexer_hex`` cell (8 hex chars
  encoding ``offset >> 4`` into the variant's ``matched_data.bin``
  record). No per-variant entry in any index file. Record geometry is
  recovered from the self-describing record header, so the inline
  encoding no longer carries a length.
* ``matched_index.bin`` is the function-to-CSV-section locator
  (u40 ``csv_offset >> 2`` + u24 ``csv_section_length >> 2``, 8 bytes).
  Both quantities are 4-byte aligned: after every section's blank-row
  terminator the writer pads with raw ``\\n`` bytes until the next
  section start lands on a 4-byte boundary AND ≥ 1 fully empty line of
  separation is guaranteed.
* ``unmatched_index.bin`` is one entry per version's data-bin record
  (16-byte aligned offsets, 4-byte entries) and continues using
  ``write_index_entry``.
* Inlining-data cells keep LOCAL ``called_func_id`` indices.
* All ``csv.writer`` callsites in this module pin
  ``lineterminator='\\n'`` so byte counts (and therefore the
  matched-arm pad arithmetic) are deterministic.

Typed-callee contract (Phase 3 wiring): pass-1 emits
``unique_called`` as ``list[tuple[name, CallTargetType]]`` and per-
variant ``called`` as ``set[tuple[name, CallTargetType]]``. The CSV
cell shapes do not yet expose the type (Phase 4 widens the header's
called_line_nos cell), so the CSV path projects the typed tuples to a
name-only dedup; the BIN path consumes the typed tuples directly.
``extern_libraries`` (``dict[name, library]``) flows through the
matched entry dict and the unmatched per-variant entry dict so the
BIN's extern call_targets can stamp the right
``ExternProviderRegistry`` line number.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from ..aligned_data.call_target_type import CallTargetType
from ..aligned_data.csv_section_index import write_csv_section_index_entry
from ..aligned_data.extern_providers import ExternProviderRegistry
from ..aligned_data.inline_indexer import encode_inline_indexer
from ..aligned_data.io import (
    write_function_section_csv,
    write_index_entry,
    write_unmatched_section_csv,
)
from ..aligned_data.csv_format import (
    format_function_line_no,
    format_function_line_nos_csv,
)
from ..aligned_data.matched_sections_bin import (
    CallTargetSpec,
    PerCallEntry,
    SectionWriter,
)
from .function_names import FunctionNamesRegistry
from .variants import VariantRegistry, write_warn_log_entry

logger = logging.getLogger(__name__)

# Matched-section CSV alignment:
#
# * Each section must start at a byte offset that is a multiple of
#   ``_SECTION_ALIGN``.
# * Each section must be separated from the previous one by ≥ 1 fully
#   empty line, i.e. ≥ 2 ``\n`` bytes between the last content byte of
#   section N and the first byte of section N+1. ``csv.writer.writerow([])``
#   already emits one ``\n``; this module writes 1-``_SECTION_ALIGN``
#   additional raw ``\n`` bytes after it.
#
# The csv writer is pinned to ``lineterminator='\n'`` so the byte
# arithmetic below is platform-independent.
_SECTION_ALIGN: int = 4
_SECTION_LINE_TERMINATOR: str = "\n"


# Typed-callee aliases used by the BIN walker. Kept local to this
# module so the public function signatures stay backwards-compatible
# with the existing ``dict``-shaped pass-1 outputs.
TypedCallee = Tuple[str, CallTargetType]


def _pad_section_to_alignment(sections_file, content_offset: int) -> int:
    """Pad a just-closed matched section to ``_SECTION_ALIGN`` + ≥ 1 empty line.

    Caller invariant: ``writer.writerow([])`` has just emitted one
    ``\n``, so the current file position is at ``end0`` relative to
    ``content_offset``. This helper computes the smallest
    ``next_start >= end0 + 1`` that is a multiple of
    :data:`_SECTION_ALIGN` and writes ``next_start - end0`` raw ``\n``
    characters straight through the underlying text stream (bypassing
    the csv writer for these bytes -- they are inter-section structure,
    not row content). Returns the new section_end (== ``next_start``)
    relative to ``content_offset``.
    """
    end0 = sections_file.tell() - content_offset
    # Smallest 4-multiple strictly > end0. That single bump satisfies
    # BOTH constraints (>= end0 + 1 AND multiple of _SECTION_ALIGN), so
    # pad always lands in 1..._SECTION_ALIGN inclusive. Combined with
    # the two ``\n``s already on disk (last variant row's terminator +
    # ``writerow([])``'s blank-row terminator), the run of ``\n`` bytes
    # after the last non-``\n`` content byte spans
    # ``2 + pad`` chars → 3..(_SECTION_ALIGN + 2) inclusive.
    next_start = ((end0 // _SECTION_ALIGN) + 1) * _SECTION_ALIGN
    pad = next_start - end0
    sections_file.write(_SECTION_LINE_TERMINATOR * pad)
    return sections_file.tell() - content_offset


def _project_called_names_unique(
    typed_callees: "list[TypedCallee]",
) -> List[str]:
    """Project a typed callee list to a name-only ordered dedup.

    The Phase-3 CSV cell shape (matched-arm header's
    ``called_line_nos_b64`` and the unmatched-arm equivalent) does not
    yet carry the type tag, so the CSV writer needs a NAME-ONLY list to
    feed :func:`format_function_line_nos_csv`. Phase 4 will widen the
    cell with the type-character suffix and drop this projection.

    Order is preserved from the input (Phase 3.1 hands matched
    entries' ``unique_called`` already sorted by ``(name, type.value)``;
    unmatched-arm aggregation re-sorts the union); the first
    occurrence wins for the dedup.
    """
    seen: "set[str]" = set()
    out: List[str] = []
    for name, _type in typed_callees:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _build_matched_func_name_set(matched_data_entries: List[dict]) -> "set[str]":
    """Collect the set of function names that survived the matched arm.

    The BIN's call_target ``is_matched`` flag asks whether the CALLEE's
    section will land in the matched-arm catalog. The matched arm
    consults ``matched_data_entries`` (which has been filtered down by
    pass-1's drop rules — name-starts-with-.L, all-encoders-skipped,
    all-variants-deduped-to-the-same-offset); a callee that doesn't
    appear here will land in the unmatched-arm catalog instead.
    """
    return {entry["func_name"] for entry in matched_data_entries}


def _build_call_targets_spec(
    typed_unique_called: "list[TypedCallee]",
    extern_libraries: "dict[str, str]",
    matched_func_names: "set[str]",
    sectioned_func_names: "set[str]",
    registry: FunctionNamesRegistry,
    extern_providers: ExternProviderRegistry,
) -> "tuple[list[CallTargetSpec], dict[TypedCallee, int]]":
    """Assemble the per-section :class:`CallTargetSpec` list + an
    index map for per-call lookups.

    Single chokepoint for the typed-callee -> CallTargetSpec
    translation; the matched + unmatched arms differ only in HOW they
    arrive at ``typed_unique_called`` (per-entry on matched, union over
    a function group on unmatched). Both arms then need the same
    extern-library resolution + matched-set lookup, so collapsing it
    here keeps the BIN-emission flow uniform.

    The returned index map keys on the (name, type) tuple so per-call
    entries from each variant can resolve their ``called_idx`` against
    the typed call_target table (the BIN's array of CallTargetSpecs,
    one row per (function_name_ptr, type)).

    Phase-1 drop rules (caller starts with ``.L``, all encoders
    skipped, all variants deduped to the same offset) can leave a
    LOCAL/PLT callee referenced from another function's variant
    without that callee having a section in the BIN. The
    :class:`SectionWriter` resolves LOCAL/PLT call_targets via
    ``known_sections`` and opens a back-patch hole on miss; the hole
    will never fill, and finalize would trip. To avoid leaking holes
    for filtered callees, we re-classify a LOCAL/PLT callee whose
    function name is NOT in ``sectioned_func_names`` as ``EXTERN``
    with ``extern_provider_line_no=None`` (= the "library unknown"
    sentinel). The CSV path is unchanged — it records names verbatim
    and Phase 4 widens the cell to expose the type tag.
    """
    specs: List[CallTargetSpec] = []
    index_map: "dict[TypedCallee, int]" = {}
    for idx, (callee_name, callee_type) in enumerate(typed_unique_called):
        callee_fid = registry.line_no(callee_name)
        is_matched = callee_name in matched_func_names
        effective_type = callee_type
        extern_line: "int | None" = None
        if callee_type == CallTargetType.EXTERN:
            library = extern_libraries.get(callee_name)
            if library is not None:
                extern_line = extern_providers.add(library)
        elif callee_name not in sectioned_func_names:
            # LOCAL / PLT but no section was emitted for the callee —
            # demote to EXTERN with unknown-library so the BIN writer
            # writes the ``0`` sentinel rather than opening a forever-
            # unresolved header hole. is_matched stays False because
            # there is no matched-arm section to refer to.
            effective_type = CallTargetType.EXTERN
            is_matched = False
        specs.append(
            CallTargetSpec(
                function_name_ptr=callee_fid,
                type=effective_type,
                is_matched=is_matched,
                extern_provider_line_no=extern_line,
            )
        )
        index_map[(callee_name, callee_type)] = idx
    return specs, index_map


def _emit_variant_per_call_entries(
    section_writer: SectionWriter,
    variant_called: "set[TypedCallee]",
    unique_called_index_map: "dict[TypedCallee, int]",
    callee_vkey,
    registry: FunctionNamesRegistry,
    sectioned_func_names: "set[str]",
) -> None:
    """Translate one variant's ``called`` set into per-call BIN entries.

    The BIN's per-call entry shape is ``(called_idx,
    section_variant_index)``. ``section_variant_index`` is the index
    of the CALLER's variant inside the CALLEE's section's variant
    block list, looked up at emit time via
    ``known_section_variants[(callee_FID, callee_vkey)]``. The variant
    key here is the caller's own vkey — pass-1 keys ``function_lookup``
    on ``(callee_name, caller_vkey)`` and the callee section
    registers its variant slots under the same vkey, so the lookup
    resolves directly.

    Per-call entries are emitted ONLY for callees that will have a
    section in the BIN. Phase 1's codec test pins this behaviour: the
    section's call_target TABLE carries every callee (LOCAL/PLT/EXTERN),
    but per-call entries are skipped for EXTERN (no callee section
    exists) AND for any LOCAL/PLT whose function pass-1 dropped
    (``startswith('.L')``, all encoders skipped, or all-variants-
    dedup-collapsed). ``sectioned_func_names`` is the union of matched
    + unmatched func_names — the set of names whose section will be
    written. A miss skips the per-call entry; the call_target's
    ``is_matched`` flag still encodes the "this callee is unsectioned"
    fact at the table level.

    Iteration order is sorted by ``(name, type.value)`` so the BIN's
    per-call entries are deterministic across runs and matched/unmatched
    arms.
    """
    sorted_calls = sorted(
        variant_called, key=lambda nt: (nt[0], nt[1].value)
    )
    entries: List[PerCallEntry] = []
    for callee_name, callee_type in sorted_calls:
        if callee_type is CallTargetType.EXTERN:
            continue
        if callee_name not in sectioned_func_names:
            continue
        key = (callee_name, callee_type)
        called_idx = unique_called_index_map[key]
        callee_fid = registry.line_no(callee_name)
        entries.append(
            PerCallEntry(
                called_idx=called_idx,
                callee_function_name_ptr=callee_fid,
                callee_vkey=callee_vkey,
            )
        )
    section_writer.emit_per_call_entries(entries)


def write_matched_sections_pass2(
    matched_data_entries: List[dict],
    function_lookup: dict,
    sections_file,
    index_file,
    warn_log,
    variants: VariantRegistry,
    registry: FunctionNamesRegistry,
    section_writer: SectionWriter,
    extern_providers: ExternProviderRegistry,
    matched_func_names: "set[str]",
    sectioned_func_names: "set[str]",
    *,
    error_log=None,
):
    """Pass 2: matched sections CSV + pre-v1 ``matched_index.bin`` + BIN catalog.

    ``registry`` is the FINALISED ``FunctionNamesRegistry``; the caller
    (builder.py) must finalise + write the sidecar BEFORE pass 2.
    ``index_file`` is the pre-v1 layout matched-index handle (no v1
    16-byte prelude, no alignment shift). ``error_log`` is forwarded
    to the section-index writer so per-entry cap overflows are logged
    and the entry skipped.

    ``section_writer`` is the per-binary :class:`SectionWriter` shared
    with the unmatched arm; the matched arm emits its sections first
    (encounter order preserved), then the unmatched arm appends. The
    BIN's section-offset map (``known_sections``) accumulates across
    both arms, so a matched section's call_target referencing an
    unmatched callee is back-patched when the unmatched arm later
    emits that callee.

    ``extern_providers`` is the per-binary
    :class:`ExternProviderRegistry`; ``add(library)`` returns the
    1-indexed line number that the BIN's extern call_targets stamp
    into their ``function_section_ptr`` field.

    ``matched_func_names`` is the set of function names appearing in
    the matched arm's surviving entries — used to fill the
    ``is_matched`` flag on each call_target.

    The CSV-prelude bytes already written by the caller are excluded
    from every stored ``section_start`` by snapshotting the current
    file position once at entry -- the reader compensates by adding
    its ``content_offset`` back at load time.
    """
    content_offset = sections_file.tell()
    writer = csv.writer(sections_file, lineterminator=_SECTION_LINE_TERMINATOR)
    # (func_name, section_start, section_len) per function in encounter
    # order. The previous avg-len-bucket sort fed
    # ``select_random_function_by_length`` (now a NotImplementedError
    # stub), so the index no longer carries an avg_len column and the
    # writer no longer reorders entries: readers do not depend on entry
    # order matching CSV row order.
    pending_index_entries: List = []

    for entry in matched_data_entries:
        func_name = entry["func_name"]
        typed_unique_called: "list[TypedCallee]" = entry["unique_called"]
        extern_libraries: "dict[str, str]" = entry["extern_libraries"]
        version_data = entry["version_data"]

        # ----- CSV section header + per-variant inlining rows -----
        # The CSV path still consumes a name-only view (Phase 4 widens
        # the cell). The BIN path consumes the typed tuples directly.
        called_names_unique = _project_called_names_unique(typed_unique_called)
        called_name_to_csv_idx = {
            name: idx for idx, name in enumerate(called_names_unique)
        }

        section_start = sections_file.tell() - content_offset
        line_no_b64 = format_function_line_no(registry.line_no(func_name))
        called_line_nos_b64 = format_function_line_nos_csv(
            [registry.line_no(name) for name in called_names_unique]
        )
        writer.writerow([line_no_b64, called_line_nos_b64])

        # ----- BIN: open section + emit call_targets table -----
        function_name_ptr = registry.line_no(func_name)
        section_writer.begin_section(function_name_ptr)
        call_targets, unique_called_index_map = _build_call_targets_spec(
            typed_unique_called,
            extern_libraries,
            matched_func_names,
            sectioned_func_names,
            registry,
            extern_providers,
        )
        section_writer.emit_call_targets(call_targets)

        for vdata in version_data:
            vkey = vdata["vkey"]
            called: "set[TypedCallee]" = vdata["called"]
            data_offset = vdata["data_offset"]

            variant_ref = variants.ref(vkey)
            inlining_data = {}
            for (called_name, _type) in called:
                csv_idx = called_name_to_csv_idx[called_name]
                lookup_key = (called_name, vkey)
                if lookup_key in function_lookup:
                    func_offset, func_len, is_matched = function_lookup[lookup_key]
                    inlining_data[csv_idx] = (func_offset, func_len, is_matched)
                else:
                    write_warn_log_entry(warn_log, func_name, variant_ref, called_name)

            inlining_list = [
                [idx, start, length, is_matched]
                for idx, (start, length, is_matched) in sorted(inlining_data.items())
            ]

            indexer_hex = encode_inline_indexer(data_offset)
            write_function_section_csv(writer, variant_ref, inlining_list, indexer_hex)

            # ----- BIN: variant block -----
            section_writer.begin_variant(
                variant_ref_offset=variants.byte_offset(vkey),
                data_offset_shifted=data_offset >> 4,
            )
            _emit_variant_per_call_entries(
                section_writer,
                called,
                unique_called_index_map,
                callee_vkey=vkey,
                registry=registry,
                sectioned_func_names=sectioned_func_names,
            )
            section_writer.end_variant(vkey=vkey)

        writer.writerow([])
        section_end = _pad_section_to_alignment(sections_file, content_offset)
        pending_index_entries.append(
            (func_name, section_start, section_end - section_start)
        )

        section_writer.end_section()

    for func_name, section_start, section_len in pending_index_entries:
        write_csv_section_index_entry(
            index_file,
            csv_offset=section_start,
            csv_section_length=section_len,
            func_name=func_name,
            error_log=error_log,
        )


def group_unmatched_entries_by_function(
    unmatched_data_entries: List[dict],
) -> Dict[str, dict]:
    """Group unmatched entries by function name.

    Per-function aggregator: collects the vkeys in encounter order
    (their list position is the ``comp_set_id`` referenced from
    ``called_by_version`` and the inlining-data merge). The pass-2
    writer resolves vkeys to ``0x<hex>`` variant refs via the
    ``VariantRegistry``; this stage stays unaware of the on-disk
    encoding.

    Typed-callee carry-over: the ``called`` set under each grouped
    entry preserves the ``(name, CallTargetType)`` tuple shape from
    pass 1, and ``extern_libraries`` accumulates across the function
    group's per-variant entries. Same-name-different-library across
    variants on the SAME extern name is a builder bug; the first
    encountered library wins and a warning is logged at the BIN-
    emission site (this aggregator stays I/O-free).
    """
    unmatched_by_func: Dict[str, dict] = {}
    for entry in unmatched_data_entries:
        func_name = entry["func_name"]
        vkey = entry["vkey"]

        if func_name not in unmatched_by_func:
            unmatched_by_func[func_name] = {
                "all_called": set(),
                "version_data_list": [],
                "called_by_version": [],
                "vkeys": [],
                "extern_libraries": {},
            }

        group = unmatched_by_func[func_name]
        group["all_called"].update(entry["called"])
        group["version_data_list"].append(
            (entry["data_offset"], entry["data_len"], entry["token_len"])
        )
        comp_set_id = len(group["vkeys"])
        group["vkeys"].append(vkey)
        group["called_by_version"].append((comp_set_id, entry["called"]))
        for name, lib in entry["extern_libraries"].items():
            if name in group["extern_libraries"] and group["extern_libraries"][name] != lib:
                # Same extern name reports two different libraries
                # across variants of the SAME function. Builder bug —
                # the BIN walker logs the conflict; here we just keep
                # the first encountered so the in-memory state stays
                # deterministic. The conflict report is deferred to
                # the BIN emitter where we have an error_log handle.
                group.setdefault("_extern_library_conflicts", []).append(
                    (name, group["extern_libraries"][name], lib)
                )
                continue
            group["extern_libraries"][name] = lib

    return unmatched_by_func


def _build_unmatched_inlining_data_list(
    called_by_version,
    unique_called_list: List[str],
    vkeys: List,
    function_lookup: dict,
    warn_log,
    func_name: str,
    variants: VariantRegistry,
) -> List:
    """Resolve each unmatched call site into the on-disk inlining tuple.

    ``called_by_version`` carries the typed callee set per version
    (``set[(name, CallTargetType)]``); this helper projects to the
    name-only CSV index space (matching ``unique_called_list``, which
    is a name-only ordered dedup) for the inlining-cell ``called_func_id``
    field. Phase 4 will widen the inlining cell to carry the type tag
    and drop the projection.
    """
    inlining_data_list = []
    for comp_set_id, called_funcs in called_by_version:
        for (called_name, _type) in called_funcs:
            called_func_id = unique_called_list.index(called_name)
            lookup_key = (called_name, vkeys[comp_set_id])
            if lookup_key in function_lookup:
                func_offset, func_len, is_matched = function_lookup[lookup_key]
                inlining_data_list.append(
                    [called_func_id, comp_set_id, func_offset, func_len, is_matched]
                )
            else:
                write_warn_log_entry(
                    warn_log, func_name, variants.ref(vkeys[comp_set_id]), called_name
                )
    return inlining_data_list


def _format_unmatched_inlining_str(inlining_data_list: List) -> str:
    """Merge per-version tuples + format the on-disk inlining-data cell.

    ``called_func_id`` is a LOCAL index into the section's
    ``unique_called`` list (NOT a function name). Duplicate
    ``(called_func_id, offset, length, is_matched)`` tuples appearing
    across multiple versions collapse into one entry whose
    ``comp_set_id`` field underscore-joins the contributing version IDs.
    """
    grouped = defaultdict(list)
    for called_func_id, comp_set_id, offset, length, is_matched in inlining_data_list:
        grouped[(called_func_id, offset, length, is_matched)].append(comp_set_id)

    merged_entries = []
    for (called_func_id, offset, length, is_matched), comp_set_ids in sorted(grouped.items()):
        comp_set_str = "_".join(map(str, sorted(comp_set_ids)))
        merged_entries.append(
            f"{called_func_id}-{comp_set_str},{offset:x},{length:x},{is_matched}"
        )
    return ";".join(merged_entries)


def _log_extern_library_conflicts(
    func_name: str,
    conflicts: "list[tuple[str, str, str]]",
) -> None:
    """Surface same-name-different-library extern conflicts via the module logger.

    Builder bug: the same extern callee name should report the same
    provider library across every variant of the same unmatched
    function group. ``<binary>.error.log`` is reserved for structured
    cap-overflow rows (see ``ALLOWED_REASONS`` in
    :mod:`tokenizer.memmap_builder.error_log`); a library mismatch is
    a builder-bug signal, not a per-function skip event, so it routes
    through the same channel Phase 3.1's
    :func:`passes._union_extern_libraries` uses for the matched arm.

    .. note::
        Matched-arm union + warning lives in
        ``passes.py:_union_extern_libraries`` (Phase 3.1); the
        unmatched-arm union + warning lives in
        :func:`group_unmatched_entries_by_function` here. Two arms,
        two sources, but identical first-wins + warn semantics. A
        follow-up refactor could lift the union into a shared helper
        once both arms can agree on the iteration shape (the matched
        arm walks ``Dict[variant_index, ParsedRecord]``, the unmatched
        arm walks a flat sequence of per-variant dicts).
    """
    for name, kept, dropped in conflicts:
        logger.warning(
            "function %s extern library mismatch across variants: %s/%s -> kept %s, dropped %s",
            func_name,
            func_name,
            name,
            kept,
            dropped,
        )


def write_unmatched_sections_pass2(
    unmatched_data_entries: List[dict],
    function_lookup: dict,
    sections_file,
    index_file,
    warn_log,
    variants: VariantRegistry,
    registry: FunctionNamesRegistry,
    section_writer: SectionWriter,
    extern_providers: ExternProviderRegistry,
    matched_func_names: "set[str]",
    sectioned_func_names: "set[str]",
    *,
    error_log=None,
):
    """Pass 2: unmatched sections CSV + v1 ``unmatched_index.bin`` + BIN catalog.

    ``registry`` provides line numbers for the first-cell + called-funcs
    cell base64 indirection. Inlining-data cells keep LOCAL
    ``called_func_id`` indices unchanged.

    ``index_file`` is the v1 unmatched-index handle; one entry per
    version's data-bin record (16-byte aligned, 4-byte u32 ``offset >> 4``
    entries -- record geometry comes from the self-describing record
    header, no length / sentinel / overlong machinery). The matched-arm
    restructuring does NOT touch this path. ``error_log`` is forwarded
    so per-version cap overflows are logged and the entry skipped (no
    abort).

    ``section_writer`` + ``extern_providers`` + ``matched_func_names``
    are the per-binary BIN-catalog handles shared with the matched arm
    (see :func:`write_matched_sections_pass2` for the semantics).

    No CSV-byte alignment is required on this arm -- it has no
    section-locator index that would benefit from a shifted offset --
    so the section writer just emits rows back-to-back with the pinned
    ``\\n`` line terminator.
    """
    writer = csv.writer(sections_file, lineterminator=_SECTION_LINE_TERMINATOR)
    unmatched_by_func = group_unmatched_entries_by_function(unmatched_data_entries)

    for func_name, data in unmatched_by_func.items():
        all_called: "set[TypedCallee]" = data["all_called"]
        version_data_list = data["version_data_list"]
        called_by_version = data["called_by_version"]
        vkeys = data["vkeys"]
        extern_libraries: "dict[str, str]" = data["extern_libraries"]
        _log_extern_library_conflicts(
            func_name, data.get("_extern_library_conflicts", [])
        )

        if not version_data_list:
            continue

        # Typed sorted union of all callees seen across this function
        # group's variants — drives the BIN's call_target table.
        typed_unique_called: "list[TypedCallee]" = sorted(
            all_called, key=lambda nt: (nt[0], nt[1].value)
        )
        # Name-only projection for the CSV path.
        unique_called_list = _project_called_names_unique(typed_unique_called)
        first_offset = version_data_list[0][0]

        inlining_data_list = _build_unmatched_inlining_data_list(
            called_by_version,
            unique_called_list,
            vkeys,
            function_lookup,
            warn_log,
            func_name,
            variants,
        )

        variant_refs = [variants.ref(vkey) for vkey in vkeys]
        line_no_b64 = format_function_line_no(registry.line_no(func_name))
        called_line_nos_b64 = format_function_line_nos_csv(
            [registry.line_no(name) for name in unique_called_list]
        )
        inlining_data_str = _format_unmatched_inlining_str(inlining_data_list)
        indexer_hex = encode_inline_indexer(first_offset)

        write_unmatched_section_csv(
            writer,
            line_no_b64,
            variant_refs,
            called_line_nos_b64,
            inlining_data_str,
            indexer_hex,
        )

        for data_offset, _data_len, _token_len in version_data_list:
            write_index_entry(
                index_file,
                data_offset,
                func_name=func_name,
                error_log=error_log,
            )

        # ----- BIN: section header + call_targets + per-variant blocks -----
        function_name_ptr = registry.line_no(func_name)
        section_writer.begin_section(function_name_ptr)
        call_targets, unique_called_index_map = _build_call_targets_spec(
            typed_unique_called,
            extern_libraries,
            matched_func_names,
            sectioned_func_names,
            registry,
            extern_providers,
        )
        section_writer.emit_call_targets(call_targets)

        # Each grouped per-variant entry contributes one variant block.
        # Order: encounter order (same as the vkeys list).
        for (comp_set_id, called_set), vkey in zip(called_by_version, vkeys):
            # Recover the data_offset for THIS variant from
            # version_data_list (same index as comp_set_id by
            # construction in group_unmatched_entries_by_function).
            variant_data_offset, _data_len, _token_len = version_data_list[
                comp_set_id
            ]
            section_writer.begin_variant(
                variant_ref_offset=variants.byte_offset(vkey),
                data_offset_shifted=variant_data_offset >> 4,
            )
            _emit_variant_per_call_entries(
                section_writer,
                called_set,
                unique_called_index_map,
                callee_vkey=vkey,
                registry=registry,
                sectioned_func_names=sectioned_func_names,
            )
            section_writer.end_variant(vkey=vkey)

        section_writer.end_section()
