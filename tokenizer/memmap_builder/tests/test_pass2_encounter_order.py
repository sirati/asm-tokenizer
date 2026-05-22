"""Pass-2 ``call_targets[]`` ordering tracks encoder allocation order.

Two complementary surfaces are pinned here:

1. Unmatched arm: ``group_unmatched_entries_by_function`` collects
   callees across every version's entry into a per-function
   ``all_called`` insertion-ordered dict (first-seen wins), then
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


def test_group_unmatched_cross_category_order_preserved():
    """Mixed-category single-version callee list (LOCAL + PLT + EXTERN
    in a non-alphabetical order). The grouper preserves the order across
    category boundaries — per-category sub-order (LOCAL -> PLT -> EXT)
    is enforced upstream at the parsed-record layer, the aggregator
    stays category-agnostic."""
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
        ("b", PLT),
        ("c", EXTERN),
        ("d", LOCAL),
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
