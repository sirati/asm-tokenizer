"""Bug-fix sanity test: a callee name shared between PLT and EXTERN
categories must stay as two distinct call_targets through the full
builder pipeline AND the BIN reader.

The iterator-level boundary is pinned by
``test_parsed_record_iter.test_v2_same_name_in_plt_and_ext_stays_distinct``;
this file closes the post-iterator end-to-end path:

  ParsedRecord (typed (name, type) tuples)
   -> process_matched_function (pass 1; assembles ``unique_called``)
   -> write_matched_sections_pass2 (pass 2; drives SectionWriter)
   -> iter_sections_bin (BIN reader; recovers parsed CallTarget rows)

Pre-refactor the typed-callee dedup keyed on name alone, collapsing a
PLT stub ``foo`` and an extern body ``foo`` into one row. Post-refactor
both rows survive end-to-end and the reader returns two distinct
``CallTarget`` entries differing only in ``type``.
"""

from __future__ import annotations

import io
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
from tokenizer.memmap_builder.function_names import FunctionNamesRegistry
from tokenizer.memmap_builder.passes import (
    process_matched_function,
    write_matched_sections_pass2,
)


class _StubVariants:
    """Bare ``.ref(vkey)`` + ``.byte_offset(vkey)`` registry.

    Matches the shape ``write_matched_sections_pass2`` consumes; the
    deterministic counter keeps CSV cell + BIN field in sync. No
    unified-vocab dependency — the BIN's variant_ref slot is opaque to
    the call_target invariant under test.
    """

    def __init__(self) -> None:
        self._slots: dict = {}
        self._next = 0x10

    def _ensure(self, vkey) -> int:
        if vkey not in self._slots:
            self._slots[vkey] = self._next
            self._next += 0x10
        return self._slots[vkey]

    def ref(self, vkey) -> str:
        return f"{self._ensure(vkey):x}"

    def byte_offset(self, vkey) -> int:
        return self._ensure(vkey)


def _make_parsed_record(
    func_name: str,
    *,
    called_funcs: "list[tuple[str, CallTargetType]]",
    extern_libraries: "dict[str, str]",
    seed: int,
) -> ParsedRecord:
    """One ``ParsedRecord`` carrying typed call_target entries verbatim.

    Body bytes vary by ``seed`` so the dedup helper doesn't collapse
    the two variants into one ``data_offset`` (matched pass-1 drops the
    function when every variant dedup-collapses to the same offset).
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
        extern_libraries=dict(extern_libraries),
        content_hash=seed,  # distinct per variant -> no primary-map collision
    )


def test_same_name_in_plt_and_extern_yields_two_distinct_call_targets(
    tmp_path: Path,
) -> None:
    """Builder pipeline + BIN reader keep PLT ``foo`` and EXTERN ``foo``
    as two distinct call_target rows.

    Pre-refactor a name-only dedup collapsed both into one row. The
    post-refactor pipeline keys ``unique_called`` on ``(name, type)`` and
    the BIN's call_target table preserves both rows; ``iter_sections_bin``
    recovers them as two ``CallTarget`` entries sharing
    ``function_name_ptr`` but differing in ``type``.

    Both arms point at a sectioned callee so the
    :func:`_build_call_targets_spec` demotion path (LOCAL/PLT callee
    without a section -> EXTERN-unknown) does NOT fire; otherwise the
    PLT row would alias the EXTERN row's effective ``type`` and the
    test would conflate the bug under test with the demotion path
    (which is exercised independently by
    ``test_unsectioned_local_callee_demoted_to_extern_unknown``).
    """
    caller = "caller_fn"
    shared_callee = "foo"
    v0 = ("v", 0)
    v1 = ("v", 1)

    # Both PLT + EXTERN under the same name — this is the exact shape
    # ``_extract_called_funcs`` produces from a v2 metadata cell with
    # ``plt_funcs: [{"name": "foo"}]`` and ``ext_funcs: [{"name":
    # "foo"}]`` for the same function. We construct the parsed records
    # directly (skipping CSV parse) so the test pins the post-iterator
    # builder + reader chain, complementary to the iterator-boundary
    # test in ``test_parsed_record_iter``.
    typed_called = [
        (shared_callee, CallTargetType.PLT),
        (shared_callee, CallTargetType.EXTERN),
    ]
    extern_libraries = {shared_callee: "libc"}

    caller_matched = Matched(
        func_name=caller,
        records={
            0: _make_parsed_record(
                caller,
                called_funcs=typed_called,
                extern_libraries=extern_libraries,
                seed=1,
            ),
            1: _make_parsed_record(
                caller,
                called_funcs=typed_called,
                extern_libraries=extern_libraries,
                seed=20,
            ),
        },
    )
    # The shared callee gets its own matched section so the PLT row in
    # the caller stays as PLT (rather than being demoted to EXTERN-
    # unknown). Two distinct variants on the callee side too, so
    # process_matched_function doesn't drop it.
    callee_matched = Matched(
        func_name=shared_callee,
        records={
            0: _make_parsed_record(
                shared_callee,
                called_funcs=[],
                extern_libraries={},
                seed=100,
            ),
            1: _make_parsed_record(
                shared_callee,
                called_funcs=[],
                extern_libraries={},
                seed=200,
            ),
        },
    )

    # -----------------------------------------------------------------
    # Drive real pass 1 (process_matched_function) -> real pass 2
    # (write_matched_sections_pass2 + SectionWriter), then read the BIN
    # back via iter_sections_bin. No bespoke stubs for the writer side;
    # the dedup helper writes into a real ``_data.bin`` so the per-record
    # content-hash + dedup path stays exercised end-to-end.
    # -----------------------------------------------------------------
    data_bin = tmp_path / "demo_data.bin"
    sections_csv = tmp_path / "demo_sections.csv"
    matched_index = tmp_path / "demo_index.bin"
    sections_bin = tmp_path / "demo_sections.bin"

    registry = FunctionNamesRegistry()
    arm_state = open_arm_dedup_state(data_bin)
    try:
        callee_entry = process_matched_function(
            callee_matched,
            version_keys=[v0, v1],
            arm_state=arm_state,
            registry=registry,
        )
        caller_entry = process_matched_function(
            caller_matched,
            version_keys=[v0, v1],
            arm_state=arm_state,
            registry=registry,
        )
    finally:
        arm_state.writer.finalize()
    assert callee_entry is not None, "callee pass-1 should survive"
    assert caller_entry is not None, "caller pass-1 should survive"
    # Pass-1 must surface BOTH typed callees in ``unique_called``; a
    # regression to name-only dedup would collapse the list to one.
    assert caller_entry["unique_called"] == [
        (shared_callee, CallTargetType.PLT),
        (shared_callee, CallTargetType.EXTERN),
    ], (
        "pass-1 must preserve both (name, type) tuples; got: "
        f"{caller_entry['unique_called']!r}"
    )

    registry.finalize()

    matched_data_entries = [callee_entry, caller_entry]
    function_lookup: dict = {}
    for ent in matched_data_entries:
        for vdata in ent["version_data"]:
            function_lookup[(ent["func_name"], vdata["vkey"])] = (
                vdata["data_offset"],
                vdata["data_len"],
                1,
            )

    matched_func_names = {caller, shared_callee}
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

    # -----------------------------------------------------------------
    # Reader side: iter_sections_bin must recover TWO distinct
    # call_targets sharing function_name_ptr but differing in type.
    # -----------------------------------------------------------------
    sections = list(iter_sections_bin(sections_bin))
    # Encounter order: callee first, then caller.
    assert len(sections) == 2, f"two sections expected, got {len(sections)}"
    callee_section, caller_section = sections
    assert callee_section.function_name_ptr == registry.line_no(shared_callee)
    assert caller_section.function_name_ptr == registry.line_no(caller)

    callee_fid = registry.line_no(shared_callee)
    same_name_targets = [
        ct
        for ct in caller_section.call_targets
        if ct.function_name_ptr == callee_fid
    ]
    assert len(same_name_targets) == 2, (
        f"PLT + EXTERN ``{shared_callee}`` must stay distinct in the BIN; "
        f"got {len(same_name_targets)} row(s): {same_name_targets!r}"
    )
    types = {ct.type for ct in same_name_targets}
    assert types == {CallTargetType.PLT, CallTargetType.EXTERN}, (
        f"two call_targets but wrong types; got: {types!r}"
    )
    # The reader's _unpack_flags preserves the type discriminator on
    # each row independently — pin that explicitly so a future regression
    # that drops or aliases one of the bits surfaces here.
    by_type = {ct.type: ct for ct in same_name_targets}
    plt_target = by_type[CallTargetType.PLT]
    extern_target = by_type[CallTargetType.EXTERN]
    assert plt_target.function_name_ptr == callee_fid
    assert extern_target.function_name_ptr == callee_fid
    # PLT row resolves to the callee's section (LOCAL/PLT path through
    # SectionWriter's known_sections).
    assert plt_target.function_section_ptr == callee_section.section_offset
    # EXTERN row carries the providers line number for ``libc`` (the
    # extern-libraries dict was populated for the EXTERN entry). Not
    # the library-unknown ``0`` sentinel.
    assert extern_target.function_section_ptr > 0
    # ``function_section_ptr`` is the dispositive structural difference
    # between the two rows: PLT carries the BIN section offset of the
    # callee section, EXTERN carries the providers-sidecar line number.
    # Same name, distinct rows, distinct resolution semantics.
    assert plt_target.function_section_ptr != extern_target.function_section_ptr
    # Both rows reference a matched function (``foo`` IS in
    # matched_func_names); ``is_matched`` is a property of the callee
    # function, not of the call_target type, so both rows share it.
    assert plt_target.is_matched is True
    assert extern_target.is_matched is True
