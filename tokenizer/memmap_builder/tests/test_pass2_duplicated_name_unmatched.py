"""Duplicated-name routing + call-side J contract (owner correctness fix).

A canonical function name can map to MULTIPLE distinct functions inside
one binary (per-TU static initializers, thunks, anon-namespace
collisions). The body-divergence deduper keeps each body and bumps the
per-binary ``occurrence`` column. Such a name is DUPLICATED and cannot be
matched — there is no single function for the matched arm to align
variant-against-variant.

The owner ruling, pinned here end-to-end:

1. **Definition side** — every row of a duplicated name routes to the
   UNMATCHED arm (occurrence 0 included), and the unmatched grouper keys
   on the DISTINCT identity ``(func_name, occurrence)`` so each divergent
   body becomes its OWN section. N occurrences ⇒ N distinct sections that
   SHARE the name's FID (same-FID siblings), each holding only that body's
   per-stream arch-variants — NOT one merged group of all bodies (the
   merge was both the catalog-corruption AND the pass-2 superlinearity
   root cause). Each duplicated-routed section carries the
   ``is_duplicated`` section-header marker; genuinely-single-variant
   unmatched functions (occurrence-0-only, one section) do not.

2. **Call side** — a matched caller's edge into a duplicated callee is
   RECORDED (the call_target row + per-call entry exist) but its
   ``section_variant_index`` is the legitimately-missing sentinel
   :data:`MISSING_VARIANT_INDEX` (``0xFFFE``): we cannot know WHICH of
   the same-named functions the site targets, so we honestly mark it
   unresolvable rather than resolve against an arbitrary sibling.

The driving merge is the real :func:`lockstep_records`; pass 1/2 are the
real builder walkers, so this is a faithful end-to-end pin.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.csv_format import write_csv_prelude
from tokenizer.aligned_data.extern_providers import ExternProviderRegistry
from tokenizer.aligned_data.index_format import (
    iter_index_entries,
    write_index_prelude,
)
from tokenizer.aligned_data.matched_sections_bin import (
    MISSING_VARIANT_INDEX,
    SectionWriter,
    iter_sections_bin,
)
from tokenizer.aligned_data.parsed_record_iter import (
    DuplicateNameClassifier,
    Matched,
    ParsedRecord,
    Unmatched,
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

# Two build variants (= two input CSV streams) keyed by index in the
# per-stream iterator list. VKeys are opaque hashables; the StubVariants
# registry maps each to a deterministic byte offset.
_VKEYS = (("v", 0), ("v", 1))

_DUP = "__cxx_global_var_init"
_CALLER = "caller_fn"


def _record(
    func_name: str,
    occurrence: int,
    *,
    called: "list[tuple[str, CallTargetType]]",
    seed: int,
) -> ParsedRecord:
    """A ParsedRecord with distinct body bytes per ``seed``.

    Distinct bodies keep the dedup helper from collapsing two variants
    to one ``data_offset`` (which would itself drop a matched function),
    so the routing under test is the only thing exercised.
    """
    return ParsedRecord(
        func_name=func_name,
        occurrence=occurrence,
        insn_runlength=np.array([seed + 4], dtype=np.uint16),
        block_runlength=np.array([seed + 3], dtype=np.uint16),
        tokens=np.array([seed, seed + 1, seed + 2], dtype=np.uint16),
        called_funcs=list(called),
        extern_libraries={},
        called_occurrences={},
        content_hash=seed,
    )


def _stream(records: "list[ParsedRecord]"):
    """Per-CSV iterator: rows sorted by (func_name, occurrence)."""
    return iter(records)


def _build_streams():
    """Two streams where ``_DUP`` is duplicated in BOTH and ``_CALLER``
    is unique in both and calls ``_DUP``.

    Sorted by (func_name, occurrence): ``__cxx_global_var_init`` (occ 0,
    occ 1) precedes ``caller_fn`` lexically.
    """
    dup_call = [(_DUP, CallTargetType.LOCAL)]
    stream0 = [
        _record(_DUP, 0, called=[], seed=10),
        _record(_DUP, 1, called=[], seed=20),
        _record(_CALLER, 0, called=dup_call, seed=30),
    ]
    stream1 = [
        _record(_DUP, 0, called=[], seed=40),
        _record(_DUP, 1, called=[], seed=50),
        _record(_CALLER, 0, called=dup_call, seed=60),
    ]
    return [_stream(stream0), _stream(stream1)]


def test_lockstep_routes_duplicated_name_to_unmatched_and_records_it() -> None:
    """The merge itself: a name duplicated in any stream never reaches
    the Matched arm and is recorded in the classifier."""
    classifier = DuplicateNameClassifier()
    items = list(lockstep_records(_build_streams(), classifier=classifier))

    by_name: dict[str, list] = {}
    for item in items:
        by_name.setdefault(item.func_name, []).append(item)

    # _CALLER is unique in each stream and present in both ⇒ one Matched.
    assert len(by_name[_CALLER]) == 1
    assert isinstance(by_name[_CALLER][0], Matched)

    # _DUP reaches occurrence 1 ⇒ every body (occ 0 + occ 1, both
    # streams = 4 rows) goes down the Unmatched arm.
    dup_items = by_name[_DUP]
    assert len(dup_items) == 4
    assert all(isinstance(it, Unmatched) for it in dup_items)
    assert {it.variant_index for it in dup_items} == {0, 1}

    # The single classification boundary recorded the duplicated name.
    assert classifier.duplicated_names == {_DUP}


def test_cross_stream_single_in_one_duplicated_in_other_is_unmatched() -> None:
    """Owner ruling: 'duplicated ⇒ unmatched no matter what'.

    A name that is occurrence-0-only in stream 0 but duplicated (occ 0 +
    occ 1) in stream 1 is DUPLICATED — the in-stream divergence in
    stream 1 means the canonical name no longer identifies a single
    function across the binary, so NO body (including stream 0's single
    occurrence) may take the matched arm.
    """
    name = "mixed_fn"
    stream0 = [_record(name, 0, called=[], seed=70)]  # single here
    stream1 = [  # duplicated here
        _record(name, 0, called=[], seed=80),
        _record(name, 1, called=[], seed=90),
    ]
    classifier = DuplicateNameClassifier()
    items = list(
        lockstep_records(
            [_stream(stream0), _stream(stream1)], classifier=classifier
        )
    )

    assert classifier.duplicated_names == {name}
    assert all(isinstance(it, Unmatched) for it in items)
    # All three bodies routed to unmatched (1 from stream 0, 2 from
    # stream 1) — none promoted to matched despite stream 0 being single.
    assert len(items) == 3
    assert sorted(it.variant_index for it in items) == [0, 1, 1]


def _emit_arms(tmp_path: Path):
    """Drive real pass 1 over the merged stream → matched/unmatched entry
    lists + the duplicated-name set + a finalised registry.
    """
    classifier = DuplicateNameClassifier()
    matched_data_path = tmp_path / "demo_data.bin"
    unmatched_data_path = tmp_path / "demo_unmatched_data.bin"
    registry = FunctionNamesRegistry()
    matched_state = open_arm_dedup_state(matched_data_path)
    unmatched_state = open_arm_dedup_state(unmatched_data_path)

    matched_entries: list[dict] = []
    unmatched_entries: list[dict] = []
    error_log = io.StringIO()
    for item in lockstep_records(_build_streams(), classifier=classifier):
        if isinstance(item, Matched):
            entry = process_matched_function(
                item, list(_VKEYS), matched_state, registry, error_log=error_log
            )
            assert entry is not None
            matched_entries.append(entry)
        else:
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
    return (
        matched_entries,
        unmatched_entries,
        classifier.duplicated_names,
        registry,
    )


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


def test_duplicated_name_end_to_end_distinct_sections_and_missing_call_j(
    tmp_path: Path,
) -> None:
    """Full pipeline: a duplicated callee becomes ONE DISTINCT unmatched
    section PER occurrence (each marked duplicated), and the caller's call
    edge into it is stamped 0xFFFE.

    Owner ruling: do NOT name-merge the same-name bodies into a single
    section. ``_DUP`` is duplicated (occurrence 0 + 1) in both streams, so
    the unmatched arm emits TWO distinct sections that SHARE the ``_DUP``
    FID — same-FID siblings keyed on ``(func_name, occurrence)``, each
    holding that distinct body's per-stream arch-variants. Both carry the
    ``is_duplicated`` marker; the genuinely-single-variant caller does not.
    """
    matched_entries, unmatched_entries, duplicated_names, registry = _emit_arms(
        tmp_path
    )

    # (b) The duplicated function landed in the UNMATCHED arm; the
    # matched arm carries only the unique caller.
    assert duplicated_names == {_DUP}
    assert {e["func_name"] for e in matched_entries} == {_CALLER}
    assert _DUP in {e["func_name"] for e in unmatched_entries}

    sections, registry = _emit_sections_bin(
        tmp_path, matched_entries, unmatched_entries, duplicated_names, registry
    )

    dup_fid = registry.line_no(_DUP)
    caller_fid = registry.line_no(_CALLER)

    # (a) The duplicated name is NOT merged: occurrence 0 and occurrence 1
    # are two DISTINCT sections sharing the _DUP FID (same-FID siblings),
    # each carrying that body's two per-stream arch-variants.
    dup_sections = [s for s in sections if s.function_name_ptr == dup_fid]
    assert len(dup_sections) == 2, (
        f"expected 2 distinct duplicated sections (occ 0 + occ 1), got "
        f"{len(dup_sections)}: name_ptrs={[s.function_name_ptr for s in sections]!r}"
    )
    for s in dup_sections:
        assert len(s.variants) == 2, (
            f"each distinct body keeps its 2 per-stream arch-variants, got "
            f"{len(s.variants)}"
        )

    # (c-section) Every duplicated-routed section carries the marker; the
    # genuinely-single-variant caller does not.
    assert all(s.is_duplicated for s in dup_sections)
    caller_sections = [s for s in sections if s.function_name_ptr == caller_fid]
    assert len(caller_sections) == 1
    caller_section = caller_sections[0]
    assert not caller_section.is_duplicated

    # The call edge into the duplicated callee is RECORDED in the
    # call_target table.
    dup_targets = [
        ct for ct in caller_section.call_targets if ct.function_name_ptr == dup_fid
    ]
    assert len(dup_targets) == 1

    # (c) Every per-call entry pointing at the duplicated callee carries
    # the legitimately-missing sentinel — recorded but unresolved.
    dup_called_idxs = {
        i
        for i, ct in enumerate(caller_section.call_targets)
        if ct.function_name_ptr == dup_fid
    }
    saw_edge = False
    for variant in caller_section.variants:
        for called_idx, sv_idx in variant.per_call_entries:
            if called_idx in dup_called_idxs:
                saw_edge = True
                assert sv_idx == MISSING_VARIANT_INDEX, (
                    f"call into duplicated callee got J={sv_idx:#06x}, "
                    f"expected MISSING_VARIANT_INDEX={MISSING_VARIANT_INDEX:#06x}"
                )
    assert saw_edge, "caller's per-call edge into the duplicated callee was dropped"

    # Loader cardinality invariant (``_build_record_to_section_idx``): the
    # unmatched index must carry exactly one record per section-variant
    # across ALL unmatched sections, in lockstep. Distinct sections must
    # not break this -- each occurrence's section writes its own index
    # entries before its section, so the totals still line up. Here the
    # only unmatched functions are the two duplicated sections (the caller
    # is matched), so the index count equals their combined variant count.
    total_section_variants = sum(len(s.variants) for s in dup_sections)
    n_index_entries = sum(1 for _ in iter_index_entries(tmp_path / "u_idx.bin"))
    assert n_index_entries == total_section_variants, (
        f"unmatched index has {n_index_entries} records but the duplicated "
        f"sections declare {total_section_variants} variants; the loader's "
        f"record-to-section mapping would raise"
    )


def _identical_body(func_name: str, occurrence: int) -> ParsedRecord:
    """Two distinct functions with BYTE-IDENTICAL bodies (same seed)."""
    return _record(func_name, occurrence, called=[], seed=777)


def test_byte_identical_distinct_bodies_share_one_data_bin_record(
    tmp_path: Path,
) -> None:
    """(b) Distinct duplicated sections whose bodies are byte-identical
    LINK via the existing content-dedup hashmap: they reference ONE shared
    ``_unmatched_data.bin`` record (its offset), they are NOT re-written.

    This is the only allowed non-linearity (content-hash dedup). The two
    occurrences remain DISTINCT sections (the routing/identity change),
    but their identical bodies do not duplicate bytes on disk.
    """
    classifier = DuplicateNameClassifier()
    unmatched_data_path = tmp_path / "demo_unmatched_data.bin"
    registry = FunctionNamesRegistry()
    unmatched_state = open_arm_dedup_state(unmatched_data_path)
    error_log = io.StringIO()

    name = "__static_init"
    # One stream, two occurrences of the same name with identical bodies.
    stream = [_identical_body(name, 0), _identical_body(name, 1)]
    unmatched_entries: list[dict] = []
    for item in lockstep_records([_stream(stream)], classifier=classifier):
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
    finalize_arm_dedup_state(unmatched_state)

    assert classifier.duplicated_names == {name}
    # Two DISTINCT entries (occurrence 0 + 1) ...
    occ = sorted(e["occurrence"] for e in unmatched_entries)
    assert occ == [0, 1]
    # ... but the identical bodies dedup to ONE shared data.bin offset.
    offsets = {e["data_offset"] for e in unmatched_entries}
    assert len(offsets) == 1, (
        f"byte-identical distinct bodies were re-written instead of linked "
        f"via content-dedup: offsets={offsets!r}"
    )


def test_genuinely_single_variant_unmatched_is_one_section(
    tmp_path: Path,
) -> None:
    """A unique name present in exactly one stream stays ONE unmatched
    section and is NOT marked duplicated — the distinct-identity grouping
    leaves single-variant functions untouched."""
    classifier = DuplicateNameClassifier()
    unmatched_data_path = tmp_path / "demo_unmatched_data.bin"
    registry = FunctionNamesRegistry()
    unmatched_state = open_arm_dedup_state(unmatched_data_path)
    error_log = io.StringIO()

    name = "lone_fn"
    stream = [_record(name, 0, called=[], seed=11)]
    unmatched_entries: list[dict] = []
    for item in lockstep_records([_stream(stream)], classifier=classifier):
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
    finalize_arm_dedup_state(unmatched_state)
    registry.finalize()
    registry.write_sidecar(tmp_path, "demo")

    assert classifier.duplicated_names == set()
    sections, registry = _emit_sections_bin(
        tmp_path, [], unmatched_entries, set(), registry
    )
    lone = [s for s in sections if s.function_name_ptr == registry.line_no(name)]
    assert len(lone) == 1
    assert len(lone[0].variants) == 1
    assert not lone[0].is_duplicated
