"""Matched-arm pass-2 BIN catalog wire-format contract.

The Phase-3.2 wiring drives ``SectionWriter`` from
``write_matched_sections_pass2`` / ``write_unmatched_sections_pass2``
alongside the existing CSV write. The tests below pin the BIN-side
contract (extern provider sidecar registration, EXTERN call_targets
landing in the call_target table but NOT in per-call entries, matched
+ unmatched callees sharing one BIN catalog) so a future refactor
can't regress without flagging.

What we deliberately do NOT test here: CSV alignment / padding
(covered by ``test_pass2_alignment_blocker.py``); SectionWriter's
back-patch internals (covered by
``tokenizer/aligned_data/tests/test_matched_sections_bin.py``);
end-to-end ``build_memmap_files`` smoke (covered by
``test_builder_smoke.py``).
"""

from __future__ import annotations

import io
from pathlib import Path

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.csv_format import write_csv_prelude
from tokenizer.aligned_data.extern_providers import (
    ExternProviderRegistry,
    iter_extern_providers,
)
from tokenizer.aligned_data.index_format import write_index_prelude
from tokenizer.aligned_data.matched_sections_bin import (
    SectionWriter,
    iter_sections_bin,
)
from tokenizer.memmap_builder._pass2 import (
    write_matched_sections_pass2,
    write_unmatched_sections_pass2,
)
from tokenizer.memmap_builder.function_names import FunctionNamesRegistry


class _StubVariants:
    """Bare ``.ref(vkey)`` + ``.byte_offset(vkey)`` registry.

    The hex string and the integer come from the same deterministic
    per-vkey counter so the CSV cell and BIN field stay in sync. No
    unified-vocab dependency — keeps the test focused on the BIN
    contract.
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


def _make_registry(*names: str) -> FunctionNamesRegistry:
    reg = FunctionNamesRegistry()
    for n in names:
        reg.add(n)
    reg.finalize()
    return reg


def _drive_matched(
    tmp_path: Path,
    matched_data_entries,
    function_lookup,
    registry,
    *,
    matched_func_names,
    sectioned_func_names,
):
    """Run the matched-arm pass-2 emitter + finalize the BIN.

    Returns ``(extern_providers, list[Section])`` so callers can
    inspect both artefacts. The unmatched arm is run as a no-op pass
    (empty entries) so the BIN's combined-arm finalisation runs the
    same code path the production builder hits.
    """
    sections_csv = tmp_path / "demo_sections.csv"
    index_bin = tmp_path / "demo_index.bin"
    unmatched_csv = tmp_path / "demo_unmatched_sections.csv"
    unmatched_index = tmp_path / "demo_unmatched_index.bin"
    bin_path = tmp_path / "demo_sections.bin"

    variants = _StubVariants()
    extern_providers = ExternProviderRegistry()
    section_writer = SectionWriter(bin_path)
    try:
        with open(sections_csv, "w", newline="", encoding="ascii") as sf, \
             open(index_bin, "wb") as idxf:
            write_csv_prelude(sf)
            write_matched_sections_pass2(
                matched_data_entries,
                function_lookup,
                sf,
                idxf,
                io.StringIO(),
                variants,
                registry,
                section_writer,
                extern_providers,
                matched_func_names=matched_func_names,
                sectioned_func_names=sectioned_func_names,
            )
        with open(unmatched_csv, "w", newline="", encoding="ascii") as sf, \
             open(unmatched_index, "wb") as idxf:
            write_csv_prelude(sf)
            write_index_prelude(idxf)
            write_unmatched_sections_pass2(
                [],  # no unmatched data in these fixtures
                function_lookup,
                sf,
                idxf,
                io.StringIO(),
                variants,
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
    sidecar = extern_providers.write_sidecar(tmp_path, "demo")
    return extern_providers, list(iter_sections_bin(bin_path)), sidecar


def test_matched_section_local_callee_resolves_in_bin(tmp_path: Path) -> None:
    """A matched function calling another matched function: the BIN
    catalog records one section per function; the caller's call_target
    table references the callee by ``function_section_ptr`` resolved
    to the callee's section offset.

    Production invariant: a binary's variants are PER-BINARY-BUILD
    (a vkey identifies a build), so caller + callee inside the same
    variant share the same vkey. The per-call entry's
    ``section_variant_index`` resolves at the callee's section against
    ``(callee_FID, caller_vkey)`` — which the callee's own
    ``end_variant`` registered under the same caller_vkey key. The
    fixture mirrors this by using the SAME ``v0``/``v1`` vkeys on both
    sides.
    """
    caller = "caller_fn"
    callee = "callee_fn"
    registry = _make_registry(caller, callee)
    callee_typed = (callee, CallTargetType.LOCAL)
    v0 = ("v", 0)
    v1 = ("v", 1)

    entries = [
        {
            "func_name": callee,
            "unique_called": [],
            "extern_libraries": {},
            "version_data": [
                {
                    "vkey": v0,
                    "called": set(),
                    "data_offset": 0,
                    "data_len": 16,
                    "token_len": 4,
                },
                {
                    "vkey": v1,
                    "called": set(),
                    "data_offset": 16,
                    "data_len": 16,
                    "token_len": 4,
                },
            ],
        },
        {
            "func_name": caller,
            "unique_called": [callee_typed],
            "extern_libraries": {},
            "version_data": [
                {
                    "vkey": v0,
                    "called": {callee_typed},
                    "data_offset": 32,
                    "data_len": 16,
                    "token_len": 4,
                },
                {
                    "vkey": v1,
                    "called": {callee_typed},
                    "data_offset": 48,
                    "data_len": 16,
                    "token_len": 4,
                },
            ],
        },
    ]
    function_lookup = {
        (callee, v0): (0, 16, 1),
        (callee, v1): (16, 16, 1),
    }

    _, sections, _ = _drive_matched(
        tmp_path,
        entries,
        function_lookup,
        registry,
        matched_func_names={caller, callee},
        sectioned_func_names={caller, callee},
    )

    # Sections emerge in encounter order: callee first, then caller.
    assert len(sections) == 2
    callee_section, caller_section = sections
    assert callee_section.function_name_ptr == registry.line_no(callee)
    assert caller_section.function_name_ptr == registry.line_no(caller)
    # Caller's single call_target points at the callee's section.
    assert len(caller_section.call_targets) == 1
    ct = caller_section.call_targets[0]
    assert ct.function_name_ptr == registry.line_no(callee)
    assert ct.function_section_ptr == callee_section.section_offset
    assert ct.type is CallTargetType.LOCAL
    assert ct.is_matched is True
    # Per-call entries reference the callee — one per caller variant.
    # callee_section registered variant 0 for v0, variant 1 for v1; the
    # caller's variant 0 (vkey v0) resolves to callee variant_idx 0,
    # caller variant 1 (vkey v1) resolves to callee variant_idx 1.
    assert len(caller_section.variants) == 2
    expected_sv = [0, 1]
    for variant, expected_idx in zip(caller_section.variants, expected_sv):
        assert len(variant.per_call_entries) == 1
        called_idx, sv_idx = variant.per_call_entries[0]
        assert called_idx == 0  # only one call_target
        assert sv_idx == expected_idx
        assert sv_idx != 0xFFFF, (
            "section_variant_index leaked the unresolved-hole sentinel"
        )


def test_matched_section_extern_callee_lands_in_call_target_table_only(
    tmp_path: Path,
) -> None:
    """An EXTERN callee gets a call_target row in the section header
    (with ``function_section_ptr`` = extern providers line number),
    but per-call entries reference NO EXTERN call_targets — the BIN
    layout puts the extern-vs-section dispatch on the call_target's
    ``type`` flag, not on per-call resolution.
    """
    func = "caller_fn"
    extern_name = "puts"
    registry = _make_registry(func, extern_name)
    extern_typed = (extern_name, CallTargetType.EXTERN)

    entries = [
        {
            "func_name": func,
            "unique_called": [extern_typed],
            "extern_libraries": {extern_name: "libc"},
            "version_data": [
                {
                    "vkey": ("v", 0),
                    "called": {extern_typed},
                    "data_offset": 0,
                    "data_len": 16,
                    "token_len": 4,
                },
                {
                    "vkey": ("v", 1),
                    "called": {extern_typed},
                    "data_offset": 16,
                    "data_len": 16,
                    "token_len": 4,
                },
            ],
        },
    ]
    function_lookup = {
        # extern callee has no data record; the warn-log path absorbs
        # the miss when CSV emit reaches it. Not under test here.
    }

    extern_providers, sections, sidecar_path = _drive_matched(
        tmp_path,
        entries,
        function_lookup,
        registry,
        matched_func_names={func},
        sectioned_func_names={func},
    )

    assert len(sections) == 1
    section = sections[0]
    assert len(section.call_targets) == 1
    ct = section.call_targets[0]
    assert ct.function_name_ptr == registry.line_no(extern_name)
    assert ct.type is CallTargetType.EXTERN
    assert ct.is_matched is False
    # libc registered at line 1.
    assert ct.function_section_ptr == 1
    sidecar_rows = list(iter_extern_providers(sidecar_path))
    assert sidecar_rows == [(1, "libc")]

    # Per-call entries are EMPTY: the variant references only an
    # EXTERN call_target, and per-call entries skip EXTERN to keep
    # the section_variant_index field meaningful.
    for variant in section.variants:
        assert variant.per_call_entries == [], (
            "per-call entries must skip EXTERN call_targets"
        )


def test_extern_unknown_library_writes_zero_sentinel(tmp_path: Path) -> None:
    """An EXTERN callee whose library is missing from
    ``extern_libraries`` lands as ``function_section_ptr=0`` (the
    library-unknown sentinel).
    """
    func = "caller_fn"
    extern_name = "mystery"
    registry = _make_registry(func, extern_name)
    extern_typed = (extern_name, CallTargetType.EXTERN)

    entries = [
        {
            "func_name": func,
            "unique_called": [extern_typed],
            "extern_libraries": {},  # library unknown
            "version_data": [
                {
                    "vkey": ("v", 0),
                    "called": set(),
                    "data_offset": 0,
                    "data_len": 16,
                    "token_len": 4,
                },
                {
                    "vkey": ("v", 1),
                    "called": set(),
                    "data_offset": 16,
                    "data_len": 16,
                    "token_len": 4,
                },
            ],
        },
    ]

    extern_providers, sections, sidecar_path = _drive_matched(
        tmp_path,
        entries,
        {},
        registry,
        matched_func_names={func},
        sectioned_func_names={func},
    )

    section = sections[0]
    ct = section.call_targets[0]
    assert ct.type is CallTargetType.EXTERN
    assert ct.function_section_ptr == 0  # library-unknown sentinel.

    # Sidecar has no library rows (only the prelude).
    assert list(iter_extern_providers(sidecar_path)) == []


def test_unmatched_arm_emits_sections_into_combined_bin(tmp_path: Path) -> None:
    """Unmatched-arm pass-2 appends sections into the SAME BIN the
    matched arm wrote into. Section order: matched first, then
    unmatched (the production builder runs matched before unmatched).
    """
    matched_fn = "matched_caller"
    unmatched_fn = "unmatched_callee"
    registry = _make_registry(matched_fn, unmatched_fn)
    unmatched_typed = (unmatched_fn, CallTargetType.LOCAL)
    v0 = ("v", 0)

    matched_entries = [
        {
            "func_name": matched_fn,
            "unique_called": [unmatched_typed],
            "extern_libraries": {},
            "version_data": [
                {
                    "vkey": v0,
                    "called": {unmatched_typed},
                    "data_offset": 0,
                    "data_len": 16,
                    "token_len": 4,
                },
                {
                    "vkey": ("v", 1),
                    "called": {unmatched_typed},
                    "data_offset": 16,
                    "data_len": 16,
                    "token_len": 4,
                },
            ],
        },
    ]
    unmatched_entries = [
        {
            "func_name": unmatched_fn,
            "vkey": v0,
            "data_offset": 32,
            "data_len": 16,
            "token_len": 4,
            "called": set(),
            "extern_libraries": {},
        },
    ]
    function_lookup = {(unmatched_fn, v0): (32, 16, 0)}

    # The unmatched callee is referenced from matched_fn's variant v0
    # only; matched_fn's variant ("v", 1) ALSO references it via the
    # per-call lookup, so we need the unmatched section to have a
    # variant for ("v", 1) too. But unmatched_entries has only v0.
    # Override: have only v0 in matched_fn's variants to keep the
    # fixture self-consistent.
    matched_entries[0]["version_data"] = [matched_entries[0]["version_data"][0]]

    sections_csv = tmp_path / "demo_sections.csv"
    index_bin = tmp_path / "demo_index.bin"
    unmatched_csv = tmp_path / "demo_unmatched_sections.csv"
    unmatched_index = tmp_path / "demo_unmatched_index.bin"
    bin_path = tmp_path / "demo_sections.bin"

    variants = _StubVariants()
    extern_providers = ExternProviderRegistry()
    section_writer = SectionWriter(bin_path)
    sectioned = {matched_fn, unmatched_fn}
    try:
        with open(sections_csv, "w", newline="", encoding="ascii") as sf, \
             open(index_bin, "wb") as idxf:
            write_csv_prelude(sf)
            write_matched_sections_pass2(
                matched_entries, function_lookup, sf, idxf, io.StringIO(),
                variants, registry, section_writer, extern_providers,
                matched_func_names={matched_fn},
                sectioned_func_names=sectioned,
            )
        with open(unmatched_csv, "w", newline="", encoding="ascii") as sf, \
             open(unmatched_index, "wb") as idxf:
            write_csv_prelude(sf)
            write_index_prelude(idxf)
            write_unmatched_sections_pass2(
                unmatched_entries, function_lookup, sf, idxf, io.StringIO(),
                variants, registry, section_writer, extern_providers,
                matched_func_names={matched_fn},
                sectioned_func_names=sectioned,
            )
        section_writer.finalize()
    except BaseException:
        section_writer.close()
        raise

    sections = list(iter_sections_bin(bin_path))
    assert len(sections) == 2
    # Encounter order: matched arm first, then unmatched.
    matched_section, unmatched_section = sections
    assert matched_section.function_name_ptr == registry.line_no(matched_fn)
    assert unmatched_section.function_name_ptr == registry.line_no(unmatched_fn)
    # matched_fn's call_target points at unmatched_fn's section
    # (via the BIN's known_sections back-patch — matched section
    # forward-references unmatched section, resolved at unmatched
    # section's end_section).
    ct = matched_section.call_targets[0]
    assert ct.function_name_ptr == registry.line_no(unmatched_fn)
    assert ct.function_section_ptr == unmatched_section.section_offset
    assert ct.type is CallTargetType.LOCAL
    # is_matched=False because unmatched_fn is NOT in matched_func_names.
    assert ct.is_matched is False


def test_unsectioned_local_callee_demoted_to_extern_unknown(
    tmp_path: Path,
) -> None:
    """A LOCAL callee whose function pass-1 filtered out (no section)
    is demoted to ``EXTERN`` with the library-unknown sentinel so the
    writer doesn't leak a forever-unresolved header hole. Per-call
    entries skip the demoted callee.
    """
    func = "caller_fn"
    filtered_callee = ".Llocal_label"
    registry = _make_registry(func, filtered_callee)
    callee_typed = (filtered_callee, CallTargetType.LOCAL)

    entries = [
        {
            "func_name": func,
            "unique_called": [callee_typed],
            "extern_libraries": {},
            "version_data": [
                {
                    "vkey": ("v", 0),
                    "called": {callee_typed},
                    "data_offset": 0,
                    "data_len": 16,
                    "token_len": 4,
                },
                {
                    "vkey": ("v", 1),
                    "called": {callee_typed},
                    "data_offset": 16,
                    "data_len": 16,
                    "token_len": 4,
                },
            ],
        },
    ]

    _, sections, _ = _drive_matched(
        tmp_path,
        entries,
        {},
        registry,
        matched_func_names={func},
        sectioned_func_names={func},  # filtered_callee is NOT here
    )

    section = sections[0]
    assert len(section.call_targets) == 1
    ct = section.call_targets[0]
    assert ct.type is CallTargetType.EXTERN, (
        "unsectioned LOCAL callee must be demoted to EXTERN"
    )
    assert ct.function_section_ptr == 0  # library-unknown sentinel
    assert ct.is_matched is False
    # Per-call entries skip the demoted callee.
    for variant in section.variants:
        assert variant.per_call_entries == []
