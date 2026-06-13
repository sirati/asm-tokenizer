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
* ``matched_index.bin`` is the function-to-section locator into
  ``<binary>_sections.bin`` (u40 ``bin_offset >> 2`` + u24
  ``bin_section_length >> 2``, 8 bytes). Sections are 4-byte aligned
  in the BIN (the :class:`SectionWriter` pads each section trailer);
  the same packing works for both CSV and BIN offsets.
* ``unmatched_index.bin`` is one entry per version's data-bin record
  (16-byte aligned offsets, 4-byte entries) and continues using
  ``write_index_entry``.
* Per-variant call-targets cells use LOCAL ``called_func_id`` indices
  into the section's typed ``unique_called`` list (the per-call-site
  type tag rides on the header's called-funcs cell, not the per-variant
  cell).
* All ``csv.writer`` callsites in this module pin
  ``lineterminator='\\n'`` so byte counts (and therefore the
  matched-arm pad arithmetic) are deterministic.

Typed-callee contract (Phase 3 + Phase 4.1 wiring): pass-1 emits
``unique_called`` as ``list[tuple[name, CallTargetType]]`` and per-
variant ``called`` as ``set[tuple[name, CallTargetType]]``. The CSV
header's called-funcs cell carries the typed form
``<base64_line_no>:<L|P|E>`` (Phase 4.1 widening), so the CSV path
consumes the same typed tuples the BIN path does — no name-only
projection step. The dedup index that maps a callee back to its CSV
column position keys on ``(name, type)`` to keep two same-name-
different-type entries distinct, matching the BIN's call_target
table layout.
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
    format_called_line_nos_typed,
    format_function_line_no,
)
from ..aligned_data.matched_sections_bin import (
    CallTargetSpec,
    MISSING_VARIANT_INDEX,
    PerCallEntry,
    SectionWriter,
)
from ._extern_library_merge import merge_extern_libraries
from ._typed_called_union import category_grouped_first_seen_union
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

    Output ordering: the emitted ``specs`` are stable-sorted by
    :attr:`CallTargetSpec.type` so the BIN's ``Section.call_targets[]``
    is concatenated LOCAL → PLT → EXTERN (the invariant asserted at
    :mod:`tokenizer.aligned_data.loader._session_helpers`). The upstream
    parsed-record contract already groups by *declared* type
    (see :func:`_typed_called_union.category_grouped_first_seen_union`);
    this seam additionally re-shuffles any LOCAL/PLT row demoted to
    EXTERN above into the EXTERN block so the *effective* type
    sequence is also non-decreasing. The stable sort preserves
    intra-category encounter order produced upstream.
    """
    # Pass 1: compute effective_type per declared callee. The
    # declared-type tuple is what per-variant ``called`` sets carry, so
    # ``index_map`` keys on it; the emitted spec carries the effective
    # type after the demotion remap.
    pending: List[Tuple[TypedCallee, CallTargetSpec]] = []
    for callee_name, callee_type in typed_unique_called:
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
        pending.append(
            (
                (callee_name, callee_type),
                CallTargetSpec(
                    function_name_ptr=callee_fid,
                    type=effective_type,
                    is_matched=is_matched,
                    extern_provider_line_no=extern_line,
                ),
            )
        )

    # Pass 2: stable-sort by effective type so LOCAL/PLT demoted rows
    # land in the EXTERN block. Within each effective-type block the
    # upstream encounter order is preserved (Python's ``sorted`` is
    # stable). Index map is rebuilt against the post-sort positions so
    # downstream per-call resolution stays consistent.
    pending.sort(key=lambda item: item[1].type)
    specs = [spec for _key, spec in pending]
    index_map: "dict[TypedCallee, int]" = {
        key: idx for idx, (key, _spec) in enumerate(pending)
    }
    return specs, index_map


def _emit_variant_per_call_entries(
    section_writer: SectionWriter,
    variant_called: "list[TypedCallee]",
    unique_called_index_map: "dict[TypedCallee, int]",
    call_targets: "list[CallTargetSpec]",
    callee_variant_ref_offset: int,
    registry: FunctionNamesRegistry,
    sectioned_func_names: "set[str]",
    duplicated_names: "set[str]",
) -> None:
    """Translate one variant's ``called`` set into per-call BIN entries.

    The BIN's per-call entry shape is ``(called_idx,
    section_variant_index)``. ``section_variant_index`` is the index
    of the CALLER's variant inside the CALLEE's section's variant
    block list, resolved at the CALLEE's
    :meth:`SectionWriter.end_section` by parsing the callee section's
    bytes and matching each pending hole's ``callee_vkey`` against
    the section's on-disk ``variant_ref_offset`` field. The values
    therefore live in the SAME space: each ``PerCallEntry.callee_vkey``
    here is the byte offset of the caller's own vkey in the
    per-binary ``_variants.bin`` sidecar — the very value the callee
    section will stamp into its matching variant header at
    :meth:`SectionWriter.begin_variant`.

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

    Emit order invariant: per_call_entries within a variant are in
    non-decreasing :class:`CallTargetType` order — a contiguous LOCAL
    block (type=0) followed by a contiguous PLT block (type=1). EXTERN
    (type=2) is skipped above so it never appears. The sort is stable,
    so within each category the original encoder-allocation order from
    ``variant_called`` (and from ``rec.called_funcs`` upstream) is
    preserved. This blocks of-category ordering lets the dataloader
    address per-category chunks of per_call_entries via cumsum on
    per-Category callee counts — no per-entry
    ``call_targets[called_idx].flags`` lookup needed. The
    list-position match between encoder-allocation order and the
    section's ``call_target`` table (plan Decisions 20 + 21) is
    preserved within each category block.
    """
    entries: List[PerCallEntry] = []
    for callee_name, callee_type in variant_called:
        if callee_type is CallTargetType.EXTERN:
            continue
        if callee_name not in sectioned_func_names:
            continue
        key = (callee_name, callee_type)
        called_idx = unique_called_index_map[key]
        callee_fid = registry.line_no(callee_name)
        # A call into a DUPLICATED callee is real (the edge is recorded
        # via ``called_idx``) but unresolvable: the canonical name maps to
        # several distinct functions, now living as sibling variants in
        # one unmatched section, and we cannot know WHICH one this site
        # targets. Stamp the legitimately-missing sentinel rather than
        # resolve against an arbitrary sibling's variant table.
        resolved = (
            MISSING_VARIANT_INDEX if callee_name in duplicated_names else None
        )
        entries.append(
            PerCallEntry(
                called_idx=called_idx,
                callee_function_name_ptr=callee_fid,
                callee_vkey=callee_variant_ref_offset,
                resolved_section_variant_index=resolved,
            )
        )
    # Stable sort by call_target category: LOCAL (0) block then PLT (1)
    # block. EXTERN is filtered out above. Python's ``sorted`` is stable,
    # so encounter order within each category is preserved.
    entries.sort(key=lambda entry: call_targets[entry.called_idx].type)
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
    duplicated_names: "set[str]",
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
    # (func_name, bin_offset, bin_section_length) per function in
    # encounter order. The index now locates BIN sections (Phase 4
    # cutover); the CSV is kept on disk as the debug-only catalog. The
    # previous avg-len-bucket sort fed ``select_random_function_by_length``
    # (now a NotImplementedError stub), so the index no longer carries
    # an avg_len column and the writer no longer reorders entries.
    pending_index_entries: List = []

    for entry in matched_data_entries:
        func_name = entry["func_name"]
        typed_unique_called: "list[TypedCallee]" = entry["unique_called"]
        extern_libraries: "dict[str, str]" = entry["extern_libraries"]
        version_data = entry["version_data"]

        # ----- CSV section header + per-variant call-targets rows -----
        # The CSV cell carries the typed form post-Phase 4.1; both the
        # header's called-funcs cell and the BIN's call_target table
        # consume the same ``(name, CallTargetType)`` tuples. The CSV
        # column index for each callee is the position in
        # ``typed_unique_called``; the BIN-side index map produced by
        # :func:`_build_call_targets_spec` is identical (same input,
        # same enumerate order), so we use it directly for both paths.
        line_no_b64 = format_function_line_no(registry.line_no(func_name))
        called_line_nos_typed = format_called_line_nos_typed(
            [
                (registry.line_no(name), call_type)
                for name, call_type in typed_unique_called
            ]
        )
        writer.writerow([line_no_b64, called_line_nos_typed])

        # ----- BIN: open section + emit call_targets table -----
        # ``n_variants`` reserves the per-variant jump-table slot count
        # at section start; the writer asserts in :meth:`end_section`
        # that exactly this many ``begin_variant``/``end_variant`` pairs
        # followed. ``version_data`` is the iterable that drives the
        # variant-emission loop below, so its length IS the count.
        function_name_ptr = registry.line_no(func_name)
        section_writer.begin_section(
            function_name_ptr=function_name_ptr,
            n_variants=len(version_data),
        )
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
            called: "list[TypedCallee]" = vdata["called"]
            data_offset = vdata["data_offset"]

            variant_ref = variants.ref(vkey)
            call_targets_per_variant = {}
            for (called_name, called_type) in called:
                csv_idx = unique_called_index_map[(called_name, called_type)]
                lookup_key = (called_name, vkey)
                if lookup_key in function_lookup:
                    func_offset, _func_len, is_matched = function_lookup[lookup_key]
                    call_targets_per_variant[csv_idx] = (func_offset, is_matched)
                else:
                    write_warn_log_entry(warn_log, func_name, variant_ref, called_name)

            call_targets_list = [
                [idx, start, is_matched]
                for idx, (start, is_matched) in sorted(call_targets_per_variant.items())
            ]

            indexer_hex = encode_inline_indexer(data_offset)
            write_function_section_csv(
                writer, variant_ref, call_targets_list, indexer_hex
            )

            # ----- BIN: variant block -----
            # ``variant_ref_offset`` is the byte offset of this vkey in
            # the per-binary ``_variants.bin`` sidecar; it is also the
            # value every caller's :class:`PerCallEntry.callee_vkey`
            # carries for THIS variant, since
            # :meth:`SectionWriter.end_section` matches holes against
            # this section's on-disk ``variant_ref_offset`` field.
            callee_variant_ref_offset = variants.byte_offset(vkey)
            section_writer.begin_variant(
                variant_ref_offset=callee_variant_ref_offset,
                data_offset_shifted=data_offset >> 4,
            )
            _emit_variant_per_call_entries(
                section_writer,
                called,
                unique_called_index_map,
                call_targets,
                callee_variant_ref_offset=callee_variant_ref_offset,
                registry=registry,
                sectioned_func_names=sectioned_func_names,
                duplicated_names=duplicated_names,
            )
            section_writer.end_variant(vkey=vkey)

        writer.writerow([])
        _pad_section_to_alignment(sections_file, content_offset)
        bin_offset, bin_section_length = section_writer.end_section()
        pending_index_entries.append(
            (func_name, bin_offset, bin_section_length)
        )

    for func_name, bin_offset, bin_section_length in pending_index_entries:
        write_csv_section_index_entry(
            index_file,
            bin_offset=bin_offset,
            bin_section_length=bin_section_length,
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

    ``all_called`` is a category-grouped first-seen union over every
    version's ``entry["called"]`` iterable (LOCAL → PLT → EXT blocks,
    intra-category encoder-allocation-ordered; see
    :func:`_typed_called_union.category_grouped_first_seen_union`). The
    union runs once at the post-collection finalize step, fed the
    per-version callee lists in encounter order; the resulting list
    drives the section's ``call_targets[]`` ordering.
    """
    unmatched_by_func: Dict[str, dict] = {}
    for entry in unmatched_data_entries:
        func_name = entry["func_name"]
        vkey = entry["vkey"]

        if func_name not in unmatched_by_func:
            unmatched_by_func[func_name] = {
                "version_data_list": [],
                "called_by_version": [],
                "vkeys": [],
                "_per_variant_extern_libraries": [],
            }

        group = unmatched_by_func[func_name]
        group["version_data_list"].append(
            (entry["data_offset"], entry["data_len"], entry["token_len"])
        )
        comp_set_id = len(group["vkeys"])
        group["vkeys"].append(vkey)
        group["called_by_version"].append((comp_set_id, entry["called"]))
        group["_per_variant_extern_libraries"].append(entry["extern_libraries"])

    # Single-source-of-truth merges (extern libraries + typed-callee
    # union). Per-variant ordering is whatever the caller fed us
    # (typically variant-index order).
    for func_name, group in unmatched_by_func.items():
        group["extern_libraries"] = merge_extern_libraries(
            group.pop("_per_variant_extern_libraries"),
            func_name=func_name,
        )
        group["all_called"] = category_grouped_first_seen_union(
            called for _comp_set_id, called in group["called_by_version"]
        )

    return unmatched_by_func


def _build_unmatched_call_targets_list(
    called_by_version,
    typed_unique_called_index: "dict[TypedCallee, int]",
    vkeys: List,
    function_lookup: dict,
    warn_log,
    func_name: str,
    variants: VariantRegistry,
) -> List:
    """Resolve each unmatched call site into the on-disk call-targets tuple.

    ``called_by_version`` carries the typed callee set per version
    (``set[(name, CallTargetType)]``); ``typed_unique_called_index``
    maps each ``(name, type)`` pair to its position in the section's
    typed call-target list (= the CSV column index). The cell shape
    matches the BIN call_target table — two same-name-different-type
    entries stay distinct via the typed lookup. ``hex_length`` is gone
    post-Phase 4.1; records in ``_unmatched_data.bin`` are
    self-describing.
    """
    call_targets_data_list = []
    for comp_set_id, called_funcs in called_by_version:
        for (called_name, called_type) in called_funcs:
            called_func_id = typed_unique_called_index[(called_name, called_type)]
            lookup_key = (called_name, vkeys[comp_set_id])
            if lookup_key in function_lookup:
                func_offset, _func_len, is_matched = function_lookup[lookup_key]
                call_targets_data_list.append(
                    [called_func_id, comp_set_id, func_offset, is_matched]
                )
            else:
                write_warn_log_entry(
                    warn_log, func_name, variants.ref(vkeys[comp_set_id]), called_name
                )
    return call_targets_data_list


def _format_unmatched_call_targets_str(call_targets_data_list: List) -> str:
    """Merge per-version tuples + format the on-disk call-targets cell.

    ``called_func_id`` is a LOCAL index into the section's typed
    ``unique_called`` list (NOT a function name; the per-call-site
    type tag lives in the section header's called-funcs cell). Duplicate
    ``(called_func_id, offset, is_matched)`` tuples appearing
    across multiple versions collapse into one entry whose
    ``comp_set_id`` field underscore-joins the contributing version IDs.
    """
    grouped = defaultdict(list)
    for called_func_id, comp_set_id, offset, is_matched in call_targets_data_list:
        grouped[(called_func_id, offset, is_matched)].append(comp_set_id)

    merged_entries = []
    for (called_func_id, offset, is_matched), comp_set_ids in sorted(grouped.items()):
        comp_set_str = "_".join(map(str, sorted(comp_set_ids)))
        merged_entries.append(
            f"{called_func_id}-{comp_set_str},{offset:x},{is_matched}"
        )
    return ";".join(merged_entries)


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
    duplicated_names: "set[str]",
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
        all_called: "list[TypedCallee]" = data["all_called"]
        version_data_list = data["version_data_list"]
        called_by_version = data["called_by_version"]
        vkeys = data["vkeys"]
        extern_libraries: "dict[str, str]" = data["extern_libraries"]

        if not version_data_list:
            continue

        # Section-level typed callee table: category-grouped first-seen
        # union (LOCAL → PLT → EXT) of every variant's
        # ``entry["called"]`` iterable. Drives BOTH the BIN's
        # call_target table AND the CSV cell shape (Phase 4.1: the CSV
        # cell carries the typed form, no name-only projection step).
        # The grouping is enforced upstream in
        # :func:`group_unmatched_entries_by_function`.
        typed_unique_called: "list[TypedCallee]" = list(all_called)
        typed_unique_called_index: "dict[TypedCallee, int]" = {
            nt: idx for idx, nt in enumerate(typed_unique_called)
        }
        first_offset = version_data_list[0][0]

        call_targets_data_list = _build_unmatched_call_targets_list(
            called_by_version,
            typed_unique_called_index,
            vkeys,
            function_lookup,
            warn_log,
            func_name,
            variants,
        )

        variant_refs = [variants.ref(vkey) for vkey in vkeys]
        line_no_b64 = format_function_line_no(registry.line_no(func_name))
        called_line_nos_typed = format_called_line_nos_typed(
            [
                (registry.line_no(name), call_type)
                for name, call_type in typed_unique_called
            ]
        )
        call_targets_str = _format_unmatched_call_targets_str(call_targets_data_list)
        indexer_hex = encode_inline_indexer(first_offset)

        write_unmatched_section_csv(
            writer,
            line_no_b64,
            variant_refs,
            called_line_nos_typed,
            call_targets_str,
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
        # ``vkeys`` and ``called_by_version`` are built in lock-step by
        # :func:`group_unmatched_entries_by_function` (one append each
        # per source entry), so ``len(vkeys)`` matches the variant-block
        # loop below — exactly what the writer's jump-table reservation
        # and ``end_section`` assertion expect.
        function_name_ptr = registry.line_no(func_name)
        section_writer.begin_section(
            function_name_ptr=function_name_ptr,
            n_variants=len(vkeys),
        )
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
            # See the matched-arm equivalent for why ``callee_vkey``
            # passes the byte-offset value (same value as the variant's
            # on-disk ``variant_ref_offset``).
            callee_variant_ref_offset = variants.byte_offset(vkey)
            section_writer.begin_variant(
                variant_ref_offset=callee_variant_ref_offset,
                data_offset_shifted=variant_data_offset >> 4,
            )
            _emit_variant_per_call_entries(
                section_writer,
                called_set,
                unique_called_index_map,
                call_targets,
                callee_variant_ref_offset=callee_variant_ref_offset,
                registry=registry,
                sectioned_func_names=sectioned_func_names,
                duplicated_names=duplicated_names,
            )
            section_writer.end_variant(vkey=vkey)

        section_writer.end_section()
