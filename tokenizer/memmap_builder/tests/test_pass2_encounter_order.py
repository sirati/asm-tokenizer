"""Pass-2 ``call_targets[]`` ordering tracks encoder allocation order
within each category and concatenates the categories LOCAL → PLT →
EXTERN.

Two complementary surfaces are pinned here:

1. Unmatched arm: ``group_unmatched_entries_by_function`` collects
   callees across every version's entry then produces a category-
   grouped first-seen union (LOCAL → PLT → EXTERN; intra-category
   encounter-ordered, stable across variants).
   ``write_unmatched_sections_pass2`` consumes it as
   ``list(all_called)``. The grouper-level invariant is the single
   source-of-truth for the unmatched typed call-target table.
2. Matched arm: end-to-end BIN readback through ``iter_sections_bin``
   confirms ``Section.call_targets[]`` matches the encoder-allocation
   order from ``ParsedRecord.called_funcs``.

Per plan Decisions 20 + 21: per-category sub-order matches CSV array
order; categories concatenate in LOCAL -> PLT -> EXT order.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.csv_format import write_csv_prelude
from tokenizer.aligned_data.extern_providers import ExternProviderRegistry
from tokenizer.aligned_data.matched_sections_bin import (
    SectionWriter,
    iter_sections_bin,
)
from tokenizer.aligned_data.parsed_record_iter import Matched, ParsedRecord
from tokenizer.memmap_builder._dedup import open_arm_dedup_state
from tokenizer.memmap_builder._pass2 import (
    group_unmatched_entries_by_function,
    write_matched_sections_pass2,
)
from tokenizer.memmap_builder.function_names import FunctionNamesRegistry
from tokenizer.memmap_builder.passes import process_matched_function

from ._fixtures import StubVariants as _StubVariants


@dataclass(frozen=True)
class _FakeVKey:
    label: str


def _make_unmatched_entry(
    func_name: str,
    *,
    vkey,
    called: "list[tuple[str, CallTargetType]]",
    data_offset: int,
    data_len: int = 16,
    token_len: int = 8,
    extern_libraries: "dict[str, str] | None" = None,
) -> dict:
    """Shape matches the dict written by `process_unmatched_function`."""
    return {
        "func_name": func_name,
        "vkey": vkey,
        "data_offset": data_offset,
        "data_len": data_len,
        "token_len": token_len,
        # `called` stays an ordered iterable (a list here) so the test can
        # control the per-version callee order; the aggregator just feeds
        # each callee through the insertion-order check.
        "called": list(called),
        "extern_libraries": extern_libraries or {},
    }


def test_group_unmatched_all_called_is_encounter_ordered():
    """Two versions of the same unmatched function. Version 0's callee
    order is ``[(gamma, LOCAL), (alpha, LOCAL), (beta, LOCAL)]`` (NOT
    alphabetical). Version 1 reuses ``alpha`` + ``beta`` in a different
    order and adds a novel ``delta``. The grouper's ``all_called`` union
    preserves version-0's order, then appends ``delta`` at the tail."""
    LOCAL = CallTargetType.LOCAL
    vkey0 = _FakeVKey("v0")
    vkey1 = _FakeVKey("v1")
    entries = [
        _make_unmatched_entry(
            "fn",
            vkey=vkey0,
            called=[("gamma", LOCAL), ("alpha", LOCAL), ("beta", LOCAL)],
            data_offset=0x00,
        ),
        _make_unmatched_entry(
            "fn",
            vkey=vkey1,
            called=[("beta", LOCAL), ("alpha", LOCAL), ("delta", LOCAL)],
            data_offset=0x20,
        ),
    ]

    grouped = group_unmatched_entries_by_function(entries)
    assert list(grouped["fn"]["all_called"]) == [
        ("gamma", LOCAL),
        ("alpha", LOCAL),
        ("beta", LOCAL),
        ("delta", LOCAL),
    ]


def test_group_unmatched_cross_category_is_local_plt_extern_grouped():
    """Mixed-category single-version callee list (LOCAL + PLT + EXTERN
    in a non-grouped order). The grouper enforces LOCAL → PLT → EXTERN
    grouping (``Section.call_targets[]`` invariant asserted at
    ``loader/_session_helpers.py``); intra-category encounter order is
    preserved via a stable sort."""
    LOCAL = CallTargetType.LOCAL
    PLT = CallTargetType.PLT
    EXTERN = CallTargetType.EXTERN
    vkey0 = _FakeVKey("v0")
    entries = [
        _make_unmatched_entry(
            "fn",
            vkey=vkey0,
            called=[
                ("a", LOCAL),
                ("b", PLT),
                ("c", EXTERN),
                ("d", LOCAL),
            ],
            data_offset=0x00,
            extern_libraries={"c": "libc.so"},
        ),
    ]

    grouped = group_unmatched_entries_by_function(entries)
    assert list(grouped["fn"]["all_called"]) == [
        ("a", LOCAL),
        ("d", LOCAL),
        ("b", PLT),
        ("c", EXTERN),
    ]


def test_group_unmatched_called_by_version_carries_per_version_typed_set():
    """Per-version ``called_by_version`` keeps the typed ``(name, type)``
    tuples it received, indexed by ``comp_set_id`` (the version's slot
    in ``vkeys``). The encounter-order property at the union level does
    NOT mutate per-version data — readers consuming the per-version
    list still see each version's own callee set."""
    LOCAL = CallTargetType.LOCAL
    PLT = CallTargetType.PLT
    vkey0 = _FakeVKey("v0")
    vkey1 = _FakeVKey("v1")
    entries = [
        _make_unmatched_entry(
            "fn",
            vkey=vkey0,
            called=[("alpha", LOCAL), ("beta", PLT)],
            data_offset=0x00,
        ),
        _make_unmatched_entry(
            "fn",
            vkey=vkey1,
            called=[("gamma", LOCAL)],
            data_offset=0x20,
        ),
    ]

    grouped = group_unmatched_entries_by_function(entries)
    by_version = grouped["fn"]["called_by_version"]
    assert by_version == [
        (0, [("alpha", LOCAL), ("beta", PLT)]),
        (1, [("gamma", LOCAL)]),
    ]


def _make_parsed_record(
    func_name: str,
    *,
    called_funcs: "list[tuple[str, CallTargetType]]",
    seed: int,
) -> ParsedRecord:
    """ParsedRecord with caller-specified callee order preserved verbatim.

    ``seed`` differentiates per-variant body bytes so the matched-arm
    dedup helper doesn't collapse the two variants into one
    ``data_offset`` (matched pass-1 drops a function whose surviving
    variants all share the same offset).
    """
    tokens = np.array([seed, seed + 1, seed + 2], dtype=np.uint16)
    block_runlength = np.array([seed + 3], dtype=np.uint16)
    insn_runlength = np.array([seed + 4], dtype=np.uint8)
    return ParsedRecord(
        func_name=func_name,
        insn_runlength=insn_runlength,
        block_runlength=block_runlength,
        tokens=tokens,
        called_funcs=list(called_funcs),
        extern_libraries={},
        content_hash=seed,
    )


def test_bin_call_targets_match_encoder_allocation_order(tmp_path: Path) -> None:
    """End-to-end BIN readback: pass-1 + pass-2 emit
    ``Section.call_targets[]`` in encoder-allocation order; ``iter_sections_bin``
    recovers that order verbatim.

    Caller's ``called_funcs`` is deliberately non-alphabetical
    (``[c, a, b]`` LOCAL) so a regression to alphabetical sorting at any
    layer (parser dedupe, pass-1 union, pass-2 typed_unique_called) would
    surface as a re-ordered ``Section.call_targets`` field on readback.
    Each callee gets its own matched section so call_target rows stay
    LOCAL (the demotion path to EXTERN-unknown does not trigger).
    """
    LOCAL = CallTargetType.LOCAL
    v0 = ("v", 0)
    v1 = ("v", 1)
    caller = "caller_fn"
    callees = ["c_callee", "a_callee", "b_callee"]
    typed_called = [(name, LOCAL) for name in callees]

    caller_matched = Matched(
        func_name=caller,
        records={
            0: _make_parsed_record(caller, called_funcs=typed_called, seed=1),
            1: _make_parsed_record(caller, called_funcs=typed_called, seed=20),
        },
    )
    callee_matcheds = [
        Matched(
            func_name=name,
            records={
                0: _make_parsed_record(name, called_funcs=[], seed=100 + i * 2),
                1: _make_parsed_record(name, called_funcs=[], seed=200 + i * 2),
            },
        )
        for i, name in enumerate(callees)
    ]

    data_bin = tmp_path / "demo_data.bin"
    sections_csv = tmp_path / "demo_sections.csv"
    matched_index = tmp_path / "demo_index.bin"
    sections_bin = tmp_path / "demo_sections.bin"

    registry = FunctionNamesRegistry()
    arm_state = open_arm_dedup_state(data_bin)
    matched_data_entries = []
    try:
        for m in callee_matcheds:
            ent = process_matched_function(m, [v0, v1], arm_state, registry)
            assert ent is not None
            matched_data_entries.append(ent)
        caller_entry = process_matched_function(
            caller_matched, [v0, v1], arm_state, registry
        )
        assert caller_entry is not None
        matched_data_entries.append(caller_entry)
    finally:
        arm_state.writer.finalize()

    # Pass-1 must hand the union to pass-2 in encounter order — the
    # call_targets table inherits this order verbatim.
    assert caller_entry["unique_called"] == typed_called

    registry.finalize()

    function_lookup: dict = {}
    for ent in matched_data_entries:
        for vdata in ent["version_data"]:
            function_lookup[(ent["func_name"], vdata["vkey"])] = (
                vdata["data_offset"],
                vdata["data_len"],
                1,
            )

    matched_func_names = {caller, *callees}
    sectioned_func_names = set(matched_func_names)
    extern_providers = ExternProviderRegistry()
    section_writer = SectionWriter(sections_bin)
    try:
        with open(sections_csv, "w", newline="", encoding="ascii") as sf, \
             open(matched_index, "wb") as idxf:
            write_csv_prelude(sf)
            write_matched_sections_pass2(
                matched_data_entries,
                function_lookup,
                sf,
                idxf,
                io.StringIO(),
                _StubVariants(),
                registry,
                section_writer,
                extern_providers,
                matched_func_names=matched_func_names,
                sectioned_func_names=sectioned_func_names,
            )
        section_writer.finalize()
    except BaseException:
        section_writer.close()
        raise

    sections_by_fid = {
        sec.function_name_ptr: sec for sec in iter_sections_bin(sections_bin)
    }
    caller_section = sections_by_fid[registry.line_no(caller)]
    fids_in_call_targets_order = [
        ct.function_name_ptr for ct in caller_section.call_targets
    ]
    expected_fids = [registry.line_no(name) for name in callees]
    assert fids_in_call_targets_order == expected_fids, (
        "Section.call_targets[] order regressed; expected encoder allocation "
        f"order {expected_fids!r} (callees {callees!r}) but BIN holds "
        f"{fids_in_call_targets_order!r}"
    )


def test_bin_call_targets_local_plt_extern_grouped(tmp_path: Path) -> None:
    """End-to-end BIN readback: a multi-category caller's
    ``Section.call_targets[].type`` sequence is non-decreasing (LOCAL →
    PLT → EXTERN), and intra-category encounter order is preserved.

    The invariant is asserted at
    ``tokenizer/aligned_data/loader/_session_helpers.py``; this
    test pins it on the actual BIN bytes a freshly-built section
    carries.

    Caller variants reference callees in a deliberately interleaved
    declared-type order across two variants. Every LOCAL/PLT callee
    has its own matched section so no demotion fires (the demotion
    path is exercised separately by
    :func:`test_build_call_targets_spec_demoted_rows_land_in_extern_block`).
    """
    LOCAL = CallTargetType.LOCAL
    PLT = CallTargetType.PLT
    EXTERN = CallTargetType.EXTERN
    v0 = ("v", 0)
    v1 = ("v", 1)
    caller = "caller_fn"
    local_a = "loc_a"
    local_b = "loc_b"
    plt_a = "plt_a"
    plt_b = "plt_b"
    extern_a = "memcpy"

    # Interleaved declared types across two variants. Without category
    # grouping the union would produce e.g. [loc_a, plt_a, memcpy, loc_b, plt_b].
    caller_called_v0 = [
        (local_a, LOCAL),
        (plt_a, PLT),
        (extern_a, EXTERN),
        (local_b, LOCAL),
    ]
    caller_called_v1 = [
        (extern_a, EXTERN),
        (plt_a, PLT),
        (plt_b, PLT),
    ]

    caller_matched = Matched(
        func_name=caller,
        records={
            0: _make_parsed_record(caller, called_funcs=caller_called_v0, seed=1),
            1: _make_parsed_record(caller, called_funcs=caller_called_v1, seed=42),
        },
    )

    # Sectioned callees (one per LOCAL/PLT name) so the demotion path
    # in ``_build_call_targets_spec`` does NOT trigger for any of them.
    sectioned_names = [local_a, local_b, plt_a, plt_b]
    callee_matcheds = [
        Matched(
            func_name=name,
            records={
                0: _make_parsed_record(name, called_funcs=[], seed=100 + i * 2),
                1: _make_parsed_record(name, called_funcs=[], seed=200 + i * 2),
            },
        )
        for i, name in enumerate(sectioned_names)
    ]

    data_bin = tmp_path / "demo_data.bin"
    sections_csv = tmp_path / "demo_sections.csv"
    matched_index = tmp_path / "demo_index.bin"
    sections_bin = tmp_path / "demo_sections.bin"

    registry = FunctionNamesRegistry()
    arm_state = open_arm_dedup_state(data_bin)
    matched_data_entries = []
    try:
        for m in callee_matcheds:
            ent = process_matched_function(m, [v0, v1], arm_state, registry)
            assert ent is not None
            matched_data_entries.append(ent)
        caller_entry = process_matched_function(
            caller_matched, [v0, v1], arm_state, registry
        )
        assert caller_entry is not None
        matched_data_entries.append(caller_entry)
    finally:
        arm_state.writer.finalize()

    # Pass-1 union must already be category-grouped LOCAL → PLT → EXTERN.
    declared_types_in_union = [ct for _name, ct in caller_entry["unique_called"]]
    assert declared_types_in_union == sorted(declared_types_in_union), (
        "Pass-1 ``unique_called`` must be category-grouped LOCAL → PLT → "
        f"EXTERN; got {caller_entry['unique_called']!r}"
    )

    registry.finalize()

    function_lookup: dict = {}
    for ent in matched_data_entries:
        for vdata in ent["version_data"]:
            function_lookup[(ent["func_name"], vdata["vkey"])] = (
                vdata["data_offset"],
                vdata["data_len"],
                1,
            )

    matched_func_names = {caller, *sectioned_names}
    sectioned_func_names = set(matched_func_names)
    extern_providers = ExternProviderRegistry()
    section_writer = SectionWriter(sections_bin)
    try:
        with open(sections_csv, "w", newline="", encoding="ascii") as sf, \
             open(matched_index, "wb") as idxf:
            write_csv_prelude(sf)
            write_matched_sections_pass2(
                matched_data_entries,
                function_lookup,
                sf,
                idxf,
                io.StringIO(),
                _StubVariants(),
                registry,
                section_writer,
                extern_providers,
                matched_func_names=matched_func_names,
                sectioned_func_names=sectioned_func_names,
            )
        section_writer.finalize()
    except BaseException:
        section_writer.close()
        raise

    sections_by_fid = {
        sec.function_name_ptr: sec for sec in iter_sections_bin(sections_bin)
    }
    caller_section = sections_by_fid[registry.line_no(caller)]
    on_disk_types = [ct.type for ct in caller_section.call_targets]
    assert on_disk_types == sorted(on_disk_types), (
        "Section.call_targets[].type must be non-decreasing (LOCAL → PLT → "
        f"EXTERN); got {on_disk_types!r}"
    )
    # Intra-category encounter order verification: ``local_a`` first
    # (v0 introduced LOCAL), then ``local_b`` (v0 novel LOCAL); ``plt_a``
    # first (v0 introduced PLT), then ``plt_b`` (v1 novel PLT);
    # ``memcpy`` is the lone EXTERN row.
    fids_per_type: "dict[CallTargetType, list[int]]" = {
        LOCAL: [], PLT: [], EXTERN: [],
    }
    for ct in caller_section.call_targets:
        fids_per_type[ct.type].append(ct.function_name_ptr)
    assert fids_per_type[LOCAL] == [
        registry.line_no(local_a),
        registry.line_no(local_b),
    ]
    assert fids_per_type[PLT] == [
        registry.line_no(plt_a),
        registry.line_no(plt_b),
    ]
    assert fids_per_type[EXTERN] == [registry.line_no(extern_a)]


def test_build_call_targets_spec_demoted_rows_land_in_extern_block() -> None:
    """``_build_call_targets_spec`` demotes a LOCAL/PLT callee whose
    name is absent from ``sectioned_func_names`` to EXTERN with the
    "library unknown" sentinel; the resulting row must land in the
    EXTERN block so the wire-order LOCAL → PLT → EXTERN invariant
    holds.

    Inputs are constructed already-category-grouped (per the upstream
    contract). The unsectioned LOCAL row would otherwise sit at the
    LOCAL block's position; ``_build_call_targets_spec``'s post-
    demotion stable sort moves it to the EXTERN block tail, after any
    upstream-declared-EXTERN rows whose position pre-sort was already
    in the EXTERN block.
    """
    LOCAL = CallTargetType.LOCAL
    PLT = CallTargetType.PLT
    EXTERN = CallTargetType.EXTERN

    sectioned_local = "loc_real"
    unsectioned_local = "loc_dropped"
    plt_real = "plt_real"
    declared_extern = "memcpy"

    typed_unique_called = [
        (sectioned_local, LOCAL),
        (unsectioned_local, LOCAL),  # will demote to EXTERN
        (plt_real, PLT),
        (declared_extern, EXTERN),
    ]
    extern_libraries = {declared_extern: "libc.so"}

    registry = FunctionNamesRegistry()
    for name, _t in typed_unique_called:
        registry.add(name)
    registry.finalize()

    sectioned_func_names = {sectioned_local, plt_real}
    matched_func_names = {sectioned_local, plt_real}
    extern_providers = ExternProviderRegistry()

    from tokenizer.memmap_builder._pass2 import _build_call_targets_spec

    specs, index_map = _build_call_targets_spec(
        typed_unique_called,
        extern_libraries,
        matched_func_names,
        sectioned_func_names,
        registry,
        extern_providers,
    )

    effective_types = [s.type for s in specs]
    assert effective_types == sorted(effective_types), (
        "Specs must be sorted by effective type (LOCAL → PLT → EXTERN); "
        f"got {effective_types!r}"
    )
    fids_per_type: "dict[CallTargetType, list[int]]" = {
        LOCAL: [], PLT: [], EXTERN: [],
    }
    for s in specs:
        fids_per_type[s.type].append(s.function_name_ptr)
    assert fids_per_type[LOCAL] == [registry.line_no(sectioned_local)]
    assert fids_per_type[PLT] == [registry.line_no(plt_real)]
    # EXTERN block: ``declared_extern`` (declared EXTERN, encounter-
    # ordered at the upstream EXTERN tail) stays in its declared-
    # encounter position; the stable sort places the demoted
    # ``unsectioned_local`` row immediately after, because its pre-sort
    # position was earlier (LOCAL block) and the sort key collision
    # places equal-key items in their pre-sort relative order.
    # Concretely: pre-sort positions of effective-EXTERN rows are
    # ``unsectioned_local`` at index 1 and ``declared_extern`` at index
    # 3; stable sort preserves that → ``unsectioned_local`` first, then
    # ``declared_extern``.
    assert fids_per_type[EXTERN] == [
        registry.line_no(unsectioned_local),
        registry.line_no(declared_extern),
    ], (
        "Stable sort must preserve pre-sort relative order for "
        "equal-effective-type rows"
    )
    # index_map must key on (name, DECLARED type) and resolve to the
    # POST-sort positions in ``specs`` so per_call_entry lookups stay
    # consistent.
    assert index_map[(unsectioned_local, LOCAL)] == 2  # EXTERN block, first
    assert index_map[(declared_extern, EXTERN)] == 3  # EXTERN block, second
    assert index_map[(sectioned_local, LOCAL)] == 0
    assert index_map[(plt_real, PLT)] == 1
