"""Duplicated-name call edges are unresolvable — uniformly MISSING.

A call into a DUPLICATED canonical name is stamped the legitimately-
missing sentinel (``0xFFFE``): the name maps to several distinct sibling
bodies and the memmap stores ONE shared ``function_section_ptr`` per call
edge across all of a caller's variants, so the format physically cannot
point variant-A at sibling-A and variant-B at sibling-B. The producer's
per-CSV ``occurrence`` ordinal is an emission-ORDER index, NOT a cross-
variant-stable body identity, so it cannot rescue the edge — variants
routinely disagree on which body is the kth same-name one (machine-
outliner clones even differ in COUNT across arch). Resolving via per-
variant occurrence (the a078d06 attempt) therefore either tripped the
``conflicting callee_occurrence`` guard or silently spliced the wrong
body. The standing rule restored here: a duplicated-name callee goes
MISSING uniformly, gating on the same ``duplicated_names`` predicate that
routes duplicated function DEFINITIONS to the unmatched arm.

These tests pin that contract end-to-end through the real merge + pass-1
+ pass-2 + SectionWriter, reading the resolved ``section_variant_index``
back off the emitted ``_sections.bin``:
* dup callee, occurrence injected   -> 0xFFFE (the occurrence is ignored);
* dup callee, occurrence ABSENT     -> 0xFFFE;
* dup callee, AMBIGUOUS (two siblings, extractor-demoted) -> 0xFFFE;
* non-dup callee                    -> resolves by name as today (control).
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


def test_dup_call_with_injected_occurrence_still_stays_missing(
    tmp_path: Path,
) -> None:
    """The standing rule holds EVEN when the producer injected an
    occurrence: a MATCHED caller whose metadata says it calls ``_DUP@1``
    is STILL stamped 0xFFFE for that edge in every variant — never
    resolved to the occurrence-1 sibling.

    The producer's per-CSV ``occurrence`` is an emission-ORDER ordinal,
    not a cross-variant-stable body identity, and the memmap stores ONE
    shared ``function_section_ptr`` per call edge across all caller
    variants. So a duplicated callee has no cross-variant target the
    format can address: the edge is recorded but unresolvable, regardless
    of any injected occurrence. (Pre-fix, a078d06 tried to resolve this to
    sibling-k via per-variant occurrence agreement; that mis-spliced
    wrong bodies when variants disagreed on the ordinal.)
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

    _caller, sv_indices = _dup_call_sv_indices(sections, caller_fid, dup_fid)
    assert sv_indices, "caller's per-call edge into _DUP was dropped"

    # Every edge into the duplicated callee is MISSING in EVERY variant,
    # the injected occurrence notwithstanding.
    for sv_idx in sv_indices:
        assert sv_idx == MISSING_VARIANT_INDEX, (
            f"call into _DUP@1 got J={sv_idx:#06x}; a duplicated callee has no "
            f"cross-variant target, so it must be MISSING regardless of the "
            f"injected occurrence"
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
