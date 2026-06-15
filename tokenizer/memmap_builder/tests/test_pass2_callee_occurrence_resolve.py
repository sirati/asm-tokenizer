"""Callee-occurrence disambiguator — call resolves to the specific sibling.

A call into a DUPLICATED canonical name is, by default, stamped the
legitimately-missing sentinel (``0xFFFE``): the name maps to several
distinct sibling bodies and a bare ``(name, type)`` call edge cannot say
which one the site targets (pinned in
``test_pass2_duplicated_name_unmatched``).

When the producer DOES know — it injects the callee body's ``occurrence``
onto the caller's ``local_funcs`` metadata entry — the build side must
resolve the call to that SPECIFIC occurrence's sibling section's variant
index, NOT the missing sentinel and NOT an arbitrary sibling. These tests
pin that contract end-to-end through the real merge + pass-1 + pass-2 +
SectionWriter, reading the resolved ``section_variant_index`` back off the
emitted ``_sections.bin``.

GATED: these assert the occurrence-aware resolution that the
``SectionWriter`` (FID, occurrence)-aware back-patch performs and that
``_emit_variant_per_call_entries`` triggers by populating
``PerCallEntry.callee_occurrence``. They are RED until that field +
resolver land; they are the target contract for that change.

The three rows of the emit-decision matrix are pinned:
* dup callee + occurrence k present  -> resolves to sibling-k (this file);
* dup callee + occurrence ABSENT     -> 0xFFFE (this file + the sibling
  duplicated-name test);
* non-dup callee                     -> resolves by name as today (the
  existing matched/unmatched suites already pin this; re-pinned here as a
  control alongside the dup case).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.csv_format import write_csv_prelude
from tokenizer.aligned_data.extern_providers import ExternProviderRegistry
from tokenizer.aligned_data.matched_sections_bin import (
    MISSING_VARIANT_INDEX,
    SectionWriter,
    iter_sections_bin,
)
from tokenizer.aligned_data.index_format import write_index_prelude
from tokenizer.aligned_data.parsed_record_iter import (
    DuplicateNameClassifier,
    Matched,
    ParsedRecord,
    Unmatched,
    called_from_v2_metadata,
    lockstep_records,
)
from tokenizer.memmap_builder._dedup import (
    finalize_arm_dedup_state,
    open_arm_dedup_state,
)
from tokenizer.memmap_builder.function_names import FunctionNamesRegistry
from tokenizer.memmap_builder.passes import (
    build_function_lookup_table,
    process_matched_function,
    process_unmatched_function,
    write_matched_sections_pass2,
    write_unmatched_sections_pass2,
)

from ._fixtures import StubVariants as _StubVariants

_VKEYS = (("v", 0), ("v", 1))
_DUP = "__cxx_global_var_init"
_CALLER = "caller_fn"


def _record(
    func_name: str,
    occurrence: int,
    *,
    called: "list[tuple[str, CallTargetType]]",
    called_occurrences: "dict[str, int]",
    seed: int,
) -> ParsedRecord:
    return ParsedRecord(
        func_name=func_name,
        occurrence=occurrence,
        insn_runlength=np.array([seed + 4], dtype=np.uint16),
        block_runlength=np.array([seed + 3], dtype=np.uint16),
        tokens=np.array([seed, seed + 1, seed + 2], dtype=np.uint16),
        called_funcs=list(called),
        extern_libraries={},
        called_occurrences=dict(called_occurrences),
        content_hash=seed,
    )


def _stream(records: "list[ParsedRecord]"):
    return iter(records)


def _emit_arms(tmp_path: Path, streams):
    """Drive real merge + pass 1 over ``streams``."""
    classifier = DuplicateNameClassifier()
    matched_data_path = tmp_path / "demo_data.bin"
    unmatched_data_path = tmp_path / "demo_unmatched_data.bin"
    registry = FunctionNamesRegistry()
    matched_state = open_arm_dedup_state(matched_data_path)
    unmatched_state = open_arm_dedup_state(unmatched_data_path)

    matched_entries: list[dict] = []
    unmatched_entries: list[dict] = []
    error_log = io.StringIO()
    for item in lockstep_records(streams, classifier=classifier):
        if isinstance(item, Matched):
            entry = process_matched_function(
                item, list(_VKEYS), matched_state, registry, error_log=error_log
            )
            assert entry is not None
            matched_entries.append(entry)
        else:
            assert isinstance(item, Unmatched)
            unmatched_entries.extend(
                process_unmatched_function(
                    item.func_name,
                    {item.variant_index: item.record},
                    list(_VKEYS),
                    unmatched_state,
                    registry,
                    error_log=error_log,
                )
            )
    finalize_arm_dedup_state(matched_state)
    finalize_arm_dedup_state(unmatched_state)
    registry.finalize()
    registry.write_sidecar(tmp_path, "demo")
    return matched_entries, unmatched_entries, classifier.duplicated_names, registry


def _emit_sections_bin(
    tmp_path: Path, matched_entries, unmatched_entries, duplicated_names, registry
):
    function_lookup = build_function_lookup_table(matched_entries, unmatched_entries)
    matched_func_names = {e["func_name"] for e in matched_entries}
    sectioned_func_names = matched_func_names | {
        e["func_name"] for e in unmatched_entries
    }
    variants = _StubVariants()
    extern_providers = ExternProviderRegistry()
    bin_path = tmp_path / "demo_sections.bin"
    section_writer = SectionWriter(bin_path)
    try:
        with open(tmp_path / "m_sec.csv", "w", newline="", encoding="ascii") as sf, \
             open(tmp_path / "m_idx.bin", "wb") as idxf:
            write_csv_prelude(sf)
            write_matched_sections_pass2(
                matched_entries, function_lookup, sf, idxf, io.StringIO(),
                variants, registry, section_writer, extern_providers,
                matched_func_names, sectioned_func_names,
                duplicated_names=duplicated_names,
            )
        with open(tmp_path / "u_sec.csv", "w", newline="", encoding="ascii") as sf, \
             open(tmp_path / "u_idx.bin", "wb") as idxf:
            write_csv_prelude(sf)
            write_index_prelude(idxf)
            write_unmatched_sections_pass2(
                unmatched_entries, function_lookup, sf, idxf, io.StringIO(),
                variants, registry, section_writer, extern_providers,
                matched_func_names, sectioned_func_names,
                duplicated_names=duplicated_names,
            )
        section_writer.finalize()
    except BaseException:
        section_writer.close()
        raise
    return list(iter_sections_bin(bin_path)), registry


def _dup_call_sv_indices(sections, caller_fid, dup_fid):
    """Collect the resolved ``section_variant_index`` of every per-call edge
    from the caller section into the duplicated callee FID."""
    caller_sections = [s for s in sections if s.function_name_ptr == caller_fid]
    assert len(caller_sections) == 1
    caller = caller_sections[0]
    dup_called_idxs = {
        i
        for i, ct in enumerate(caller.call_targets)
        if ct.function_name_ptr == dup_fid
    }
    sv = []
    for variant in caller.variants:
        for called_idx, sv_idx in variant.per_call_entries:
            if called_idx in dup_called_idxs:
                sv.append(sv_idx)
    return caller, sv


def test_forward_ref_matched_caller_resolves_to_occurrence_k_sibling(
    tmp_path: Path,
) -> None:
    """THE headline forward-ref case: a MATCHED caller (emitted before the
    unmatched dup-name siblings) whose metadata says it calls ``_DUP@1``
    resolves, via the deferred (FID, occurrence) back-patch, to the
    occurrence-1 sibling section's variant — NOT 0xFFFE, NOT occurrence-0.

    The matched arm runs first, so the dup siblings do not exist when the
    caller closes; the resolution must therefore go through the deferred
    sibling-close back-patch keyed on (FID, occurrence).
    """
    dup_call = [(_DUP, CallTargetType.LOCAL)]
    occ_one = {_DUP: 1}
    stream0 = [
        _record(_DUP, 0, called=[], called_occurrences={}, seed=10),
        _record(_DUP, 1, called=[], called_occurrences={}, seed=20),
        _record(_CALLER, 0, called=dup_call, called_occurrences=occ_one, seed=30),
    ]
    stream1 = [
        _record(_DUP, 0, called=[], called_occurrences={}, seed=40),
        _record(_DUP, 1, called=[], called_occurrences={}, seed=50),
        _record(_CALLER, 0, called=dup_call, called_occurrences=occ_one, seed=60),
    ]
    matched_entries, unmatched_entries, duplicated_names, registry = _emit_arms(
        tmp_path, [_stream(stream0), _stream(stream1)]
    )
    assert duplicated_names == {_DUP}
    # The caller is unique in both streams -> matched arm (closes first).
    assert {e["func_name"] for e in matched_entries} == {_CALLER}

    sections, registry = _emit_sections_bin(
        tmp_path, matched_entries, unmatched_entries, duplicated_names, registry
    )
    dup_fid = registry.line_no(_DUP)
    caller_fid = registry.line_no(_CALLER)

    # The two occurrence sibling sections of _DUP, in occurrence order
    # (occurrence is the section's distinct-identity key; section emit
    # order follows it).
    dup_sections = [s for s in sections if s.function_name_ptr == dup_fid]
    assert len(dup_sections) == 2
    occ1_section = dup_sections[1]

    caller, sv_indices = _dup_call_sv_indices(sections, caller_fid, dup_fid)
    assert sv_indices, "caller's per-call edge into _DUP was dropped"

    # Every resolved edge must point at a REAL variant index inside the
    # occurrence-1 sibling (not 0xFFFE, and addressable in that section).
    for sv_idx in sv_indices:
        assert sv_idx != MISSING_VARIANT_INDEX, (
            "call into _DUP@1 was stamped MISSING despite a present occurrence"
        )
        assert 0 <= sv_idx < len(occ1_section.variants), (
            f"resolved section_variant_index={sv_idx} is out of range for the "
            f"occurrence-1 sibling ({len(occ1_section.variants)} variants)"
        )


def test_dup_call_without_occurrence_stays_missing(tmp_path: Path) -> None:
    """Matrix row 3: a call into a duplicated name with NO injected
    occurrence (unresolved/indirect) stays 0xFFFE — never resolved to a
    sibling."""
    dup_call = [(_DUP, CallTargetType.LOCAL)]
    stream0 = [
        _record(_DUP, 0, called=[], called_occurrences={}, seed=11),
        _record(_DUP, 1, called=[], called_occurrences={}, seed=21),
        _record(_CALLER, 0, called=dup_call, called_occurrences={}, seed=31),
    ]
    stream1 = [
        _record(_DUP, 0, called=[], called_occurrences={}, seed=41),
        _record(_DUP, 1, called=[], called_occurrences={}, seed=51),
        _record(_CALLER, 0, called=dup_call, called_occurrences={}, seed=61),
    ]
    matched_entries, unmatched_entries, duplicated_names, registry = _emit_arms(
        tmp_path, [_stream(stream0), _stream(stream1)]
    )
    sections, registry = _emit_sections_bin(
        tmp_path, matched_entries, unmatched_entries, duplicated_names, registry
    )
    dup_fid = registry.line_no(_DUP)
    caller_fid = registry.line_no(_CALLER)
    _caller, sv_indices = _dup_call_sv_indices(sections, caller_fid, dup_fid)
    assert sv_indices, "caller's per-call edge into _DUP was dropped"
    for sv_idx in sv_indices:
        assert sv_idx == MISSING_VARIANT_INDEX, (
            f"dup call with no occurrence got J={sv_idx:#06x}, expected MISSING"
        )


def test_one_caller_two_siblings_resolves_to_missing_not_a_sibling(
    tmp_path: Path,
) -> None:
    """Build-level never-wrong-sibling guard: a caller whose metadata
    targets BOTH siblings of ``_DUP`` (occurrence 0 AND occurrence 1)
    produces an AMBIGUOUS call edge — the extractor demotes it (drops the
    name from the resolvable map), so the build side stamps 0xFFFE, NEVER
    sibling-0 or sibling-1.

    Driven through the real :func:`called_from_v2_metadata` so the
    metadata-level conflict demotion is exercised end-to-end: two
    ``local_funcs`` entries with the same name + different addr + different
    occurrence collapse to one ``(_DUP, LOCAL)`` edge whose occurrence is
    unresolvable.
    """
    metadata_cell = (
        '{"local_funcs":['
        '{"name":"' + _DUP + '","addr":"0xaaa","occurrence":0},'
        '{"name":"' + _DUP + '","addr":"0xbbb","occurrence":1}'
        "]}"
    )
    called, _extern, called_occurrences = called_from_v2_metadata(metadata_cell)
    # The two sibling edges collapse to one (name, type); the conflicting
    # occurrence is demoted out of the resolvable map.
    assert called == [(_DUP, CallTargetType.LOCAL)]
    assert _DUP not in called_occurrences

    dup_call = list(called)
    stream0 = [
        _record(_DUP, 0, called=[], called_occurrences={}, seed=12),
        _record(_DUP, 1, called=[], called_occurrences={}, seed=22),
        _record(
            _CALLER, 0, called=dup_call,
            called_occurrences=called_occurrences, seed=32,
        ),
    ]
    stream1 = [
        _record(_DUP, 0, called=[], called_occurrences={}, seed=42),
        _record(_DUP, 1, called=[], called_occurrences={}, seed=52),
        _record(
            _CALLER, 0, called=dup_call,
            called_occurrences=called_occurrences, seed=62,
        ),
    ]
    matched_entries, unmatched_entries, duplicated_names, registry = _emit_arms(
        tmp_path, [_stream(stream0), _stream(stream1)]
    )
    sections, registry = _emit_sections_bin(
        tmp_path, matched_entries, unmatched_entries, duplicated_names, registry
    )
    dup_fid = registry.line_no(_DUP)
    caller_fid = registry.line_no(_CALLER)
    _caller, sv_indices = _dup_call_sv_indices(sections, caller_fid, dup_fid)
    assert sv_indices, "caller's per-call edge into _DUP was dropped"
    for sv_idx in sv_indices:
        assert sv_idx == MISSING_VARIANT_INDEX, (
            f"ambiguous two-sibling call got J={sv_idx:#06x}; must be MISSING, "
            f"never an arbitrary sibling"
        )


def test_non_duplicated_callee_resolves_by_name(tmp_path: Path) -> None:
    """Matrix row 1 control: a call into a NON-duplicated callee resolves
    by name as today — unaffected by the occurrence machinery.

    ``_CALLEE`` is unique in both streams (matched arm); the caller's edge
    into it resolves to a real variant index, NOT the missing sentinel,
    regardless of the occurrence feature.
    """
    callee = "ordinary_callee"
    caller = "ordinary_caller"
    callee_call = [(callee, CallTargetType.LOCAL)]
    stream0 = [
        _record(callee, 0, called=[], called_occurrences={}, seed=13),
        _record(caller, 0, called=callee_call, called_occurrences={}, seed=33),
    ]
    stream1 = [
        _record(callee, 0, called=[], called_occurrences={}, seed=43),
        _record(caller, 0, called=callee_call, called_occurrences={}, seed=63),
    ]
    matched_entries, unmatched_entries, duplicated_names, registry = _emit_arms(
        tmp_path, [_stream(stream0), _stream(stream1)]
    )
    assert duplicated_names == set()
    sections, registry = _emit_sections_bin(
        tmp_path, matched_entries, unmatched_entries, duplicated_names, registry
    )
    callee_fid = registry.line_no(callee)
    caller_fid = registry.line_no(caller)
    _caller, sv_indices = _dup_call_sv_indices(sections, caller_fid, callee_fid)
    assert sv_indices, "caller's per-call edge into the callee was dropped"
    for sv_idx in sv_indices:
        assert sv_idx != MISSING_VARIANT_INDEX, (
            "non-duplicated callee resolved to MISSING; name resolution "
            "should be unaffected by the occurrence machinery"
        )
