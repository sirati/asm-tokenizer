"""Duplicated-name routing + call-side J contract (owner correctness fix).

A canonical function name can map to MULTIPLE distinct functions inside
one binary (per-TU static initializers, thunks, anon-namespace
collisions). The body-divergence deduper keeps each body and bumps the
per-binary ``occurrence`` column. Such a name is DUPLICATED and cannot be
matched — there is no single function for the matched arm to align
variant-against-variant.

The owner ruling, pinned here end-to-end:

1. **Definition side** — every row of a duplicated name routes to the
   UNMATCHED arm (occurrence 0 included). The matched arm therefore
   never emits two sections that share a ``function_name_ptr``
   ("same-FID siblings"), which was the catalog-corruption root cause.

2. **Call side** — a matched caller's edge into a duplicated callee is
   RECORDED (the call_target row + per-call entry exist) but its
   ``section_variant_index`` is the legitimately-missing sentinel
   :data:`MISSING_VARIANT_INDEX` (``0xFFFE``): we cannot know WHICH of
   the same-named functions the site targets, so we honestly mark it
   unresolvable rather than resolve against an arbitrary sibling.

The driving merge is the real :func:`lockstep_records`; pass 1/2 are the
real builder walkers, so this is a faithful end-to-end pin. On the
occurrence-blind predecessor the merge yielded the duplicated name twice
down the MATCHED arm, producing two sections that shared one
``function_name_ptr`` — the assertions below fail loudly on that code.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.csv_format import write_csv_prelude
from tokenizer.aligned_data.extern_providers import ExternProviderRegistry
from tokenizer.aligned_data.index_format import write_index_prelude
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


def test_duplicated_name_end_to_end_no_siblings_and_missing_call_j(
    tmp_path: Path,
) -> None:
    """Full pipeline: duplicated callee → one unmatched section (no
    same-FID matched siblings) + caller's call edge stamped 0xFFFE."""
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

    # (a) No two sections share a function_name_ptr — same-FID siblings
    # are gone from the matched arm at the source.
    name_ptrs = [s.function_name_ptr for s in sections]
    assert len(name_ptrs) == len(set(name_ptrs)), (
        f"same-FID sibling sections present: {name_ptrs!r}"
    )

    dup_fid = registry.line_no(_DUP)
    caller_fid = registry.line_no(_CALLER)
    caller_sections = [s for s in sections if s.function_name_ptr == caller_fid]
    assert len(caller_sections) == 1
    caller_section = caller_sections[0]

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
