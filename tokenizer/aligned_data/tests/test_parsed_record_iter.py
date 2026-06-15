"""Call-target type extraction in the parsed-record iterator.

The v1/v2 metadata columns carry per-callee category info (local /
plt / extern) that the iterator must preserve as a typed
discriminator. Collapsing on name alone was the audit-confirmed
correctness bug: a PLT stub ``foo`` and an extern body ``foo`` from
the same caller are LEGITIMATELY DISTINCT call_targets and must round-
trip as two separate entries.

These tests pin the contract at the iterator's internal extraction
boundary so any future regression on that invariant fails loudly here
before propagating downstream.
"""

from __future__ import annotations

import json

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.parsed_record_iter import _extract_called_funcs


def _row_with_v2_metadata(metadata: dict) -> tuple[list[str], dict[str, int]]:
    row = ["fn", json.dumps(metadata)]
    column_index = {"func_name": 0, "metadata": 1}
    return row, column_index


def _row_with_v1_opaque_metadata(opaque_repr: str) -> tuple[list[str], dict[str, int]]:
    row = ["fn", opaque_repr]
    column_index = {"func_name": 0, "opaque_metadata": 1}
    return row, column_index


def test_v2_same_name_in_plt_and_ext_stays_distinct():
    """A name appearing in both ``plt_funcs`` and ``ext_funcs`` must
    yield two distinct typed entries, not collapse to one."""
    metadata = {
        "plt_funcs": [{"name": "foo"}],
        "ext_funcs": [{"name": "foo"}],
    }
    row, column_index = _row_with_v2_metadata(metadata)

    called, extern_libraries, _called_occurrences = _extract_called_funcs(row, column_index)

    assert called == [
        ("foo", CallTargetType.PLT),
        ("foo", CallTargetType.EXTERN),
    ]
    assert extern_libraries == {}


def test_v2_categories_emit_typed_tuples_per_category():
    """Every v2 metadata category maps to its own CallTargetType.

    Output ordering is encoder-allocation-order per category, with
    categories concatenated LOCAL -> PLT -> EXTERN (not alphabetical).
    """
    metadata = {
        "local_funcs": [{"name": "loc"}],
        "plt_funcs": [{"name": "stub"}],
        "ext_funcs": [{"name": "ext"}],
    }
    row, column_index = _row_with_v2_metadata(metadata)

    called, extern_libraries, _called_occurrences = _extract_called_funcs(row, column_index)

    assert called == [
        ("loc", CallTargetType.LOCAL),
        ("stub", CallTargetType.PLT),
        ("ext", CallTargetType.EXTERN),
    ]
    assert extern_libraries == {}


def test_v2_duplicate_within_same_category_deduplicates():
    """Repeats inside one category collapse on ``(name, type)``."""
    metadata = {
        "local_funcs": [{"name": "loc"}, {"name": "loc"}],
    }
    row, column_index = _row_with_v2_metadata(metadata)

    called, extern_libraries, _called_occurrences = _extract_called_funcs(row, column_index)

    assert called == [("loc", CallTargetType.LOCAL)]
    assert extern_libraries == {}


def test_v2_extern_library_threaded_through_extraction():
    """v2 ``ext_funcs`` entries with a ``library`` populate the second
    return value; entries without ``library`` (or ``library=None``) are
    absent from the dict; non-EXTERN categories never appear."""
    metadata = {
        "local_funcs": [{"name": "loc", "library": "libignored.so"}],
        "plt_funcs": [{"name": "stub", "library": "libplt.so"}],
        "ext_funcs": [
            {"name": "ext_known", "library": "libfoo.so"},
            {"name": "ext_none", "library": None},
            {"name": "ext_missing"},
        ],
    }
    row, column_index = _row_with_v2_metadata(metadata)

    called, extern_libraries, _called_occurrences = _extract_called_funcs(row, column_index)

    assert called == [
        ("loc", CallTargetType.LOCAL),
        ("stub", CallTargetType.PLT),
        ("ext_known", CallTargetType.EXTERN),
        ("ext_none", CallTargetType.EXTERN),
        ("ext_missing", CallTargetType.EXTERN),
    ]
    assert extern_libraries == {"ext_known": "libfoo.so"}


def test_v1_opaque_metadata_emits_only_local_entries():
    """v1 only ever recorded local_function callees; every emitted
    tuple must therefore carry CallTargetType.LOCAL."""
    opaque = repr(
        [
            (0, 0, "alpha", "local_function", "x"),
            (0, 0, "bravo", "local_function", "x"),
        ]
    )
    row, column_index = _row_with_v1_opaque_metadata(opaque)

    called, extern_libraries, _called_occurrences = _extract_called_funcs(row, column_index)

    assert called == [
        ("alpha", CallTargetType.LOCAL),
        ("bravo", CallTargetType.LOCAL),
    ]
    for _name, ttype in called:
        assert ttype is CallTargetType.LOCAL
    assert extern_libraries == {}


def test_v1_opaque_metadata_filters_non_local_entries():
    """v1 entries with a non-``local_function`` type field are dropped,
    matching the pre-refactor behaviour."""
    opaque = repr(
        [
            (0, 0, "kept", "local_function", "x"),
            (0, 0, "dropped", "external_function", "x"),
        ]
    )
    row, column_index = _row_with_v1_opaque_metadata(opaque)

    called, extern_libraries, _called_occurrences = _extract_called_funcs(row, column_index)

    assert called == [("kept", CallTargetType.LOCAL)]
    assert extern_libraries == {}


def test_v1_opaque_metadata_always_empty_library_dict():
    """v1 has no library info — the second return value is always {}."""
    opaque = repr([(0, 0, "alpha", "local_function", "x")])
    row, column_index = _row_with_v1_opaque_metadata(opaque)

    _called, extern_libraries, _called_occurrences = _extract_called_funcs(row, column_index)

    assert extern_libraries == {}


def test_v1_opaque_metadata_never_carries_occurrence():
    """v1 predates the occurrence disambiguator — third return is always {}."""
    opaque = repr([(0, 0, "alpha", "local_function", "x")])
    row, column_index = _row_with_v1_opaque_metadata(opaque)

    _called, _extern_libraries, called_occurrences = _extract_called_funcs(
        row, column_index
    )

    assert called_occurrences == {}


def test_v2_local_func_occurrence_harvested_by_name():
    """A ``local_funcs`` entry carrying the producer-injected ``occurrence``
    surfaces in the third return keyed by callee name; an entry without it
    is absent (no sibling disambiguation needed)."""
    metadata = {
        "local_funcs": [
            {"name": "dup", "addr": "0xaaa", "occurrence": 1},
            {"name": "uniq", "addr": "0xbbb"},
        ],
    }
    row, column_index = _row_with_v2_metadata(metadata)

    called, _extern_libraries, called_occurrences = _extract_called_funcs(
        row, column_index
    )

    assert called == [
        ("dup", CallTargetType.LOCAL),
        ("uniq", CallTargetType.LOCAL),
    ]
    assert called_occurrences == {"dup": 1}


def test_v2_occurrence_only_read_from_local_category():
    """``occurrence`` on a non-LOCAL entry is ignored — the disambiguator
    is meaningful only for in-binary local sibling bodies."""
    metadata = {
        "plt_funcs": [{"name": "stub", "occurrence": 2}],
        "ext_funcs": [{"name": "ext", "occurrence": 3}],
    }
    row, column_index = _row_with_v2_metadata(metadata)

    _called, _extern_libraries, called_occurrences = _extract_called_funcs(
        row, column_index
    )

    assert called_occurrences == {}


def test_v2_non_int_occurrence_is_ignored():
    """A bool (``isinstance(True, int)`` is True) or a non-int occurrence is
    not a valid ordinal — it must not be harvested."""
    metadata = {
        "local_funcs": [
            {"name": "boolish", "addr": "0xa", "occurrence": True},
            {"name": "stringish", "addr": "0xb", "occurrence": "1"},
        ],
    }
    row, column_index = _row_with_v2_metadata(metadata)

    _called, _extern_libraries, called_occurrences = _extract_called_funcs(
        row, column_index
    )

    assert called_occurrences == {}


def test_v2_same_name_conflicting_occurrence_is_dropped():
    """One caller targeting TWO sibling bodies of the same dup name emits two
    ``local_funcs`` entries with that name at DIFFERENT occurrences. The
    name dedupes to one ``(name, LOCAL)`` call_target, so the wire format
    cannot resolve the two edges to two siblings. The ambiguous name is
    excluded from the resolvable map (build side stamps the missing
    sentinel), never resolved to an arbitrary sibling."""
    metadata = {
        "local_funcs": [
            {"name": "F", "addr": "0xaaa", "occurrence": 0},
            {"name": "F", "addr": "0xbbb", "occurrence": 1},
        ],
    }
    row, column_index = _row_with_v2_metadata(metadata)

    called, _extern_libraries, called_occurrences = _extract_called_funcs(
        row, column_index
    )

    assert called == [("F", CallTargetType.LOCAL)]
    assert "F" not in called_occurrences
    assert called_occurrences == {}


def test_v2_same_name_same_occurrence_is_not_a_conflict():
    """Defensive: the same name repeated at the SAME occurrence is
    idempotent, not a conflict — it stays resolvable."""
    metadata = {
        "local_funcs": [
            {"name": "F", "addr": "0xaaa", "occurrence": 2},
            {"name": "F", "addr": "0xaaa", "occurrence": 2},
        ],
    }
    row, column_index = _row_with_v2_metadata(metadata)

    _called, _extern_libraries, called_occurrences = _extract_called_funcs(
        row, column_index
    )

    assert called_occurrences == {"F": 2}


def test_v2_conflict_stays_dropped_after_third_sighting():
    """Once a name is ambiguous it remains dropped even if a later entry
    repeats one of the conflicting values — never silently resolvable."""
    metadata = {
        "local_funcs": [
            {"name": "F", "addr": "0xaaa", "occurrence": 0},
            {"name": "F", "addr": "0xbbb", "occurrence": 1},
            {"name": "F", "addr": "0xccc", "occurrence": 0},
        ],
    }
    row, column_index = _row_with_v2_metadata(metadata)

    _called, _extern_libraries, called_occurrences = _extract_called_funcs(
        row, column_index
    )

    assert "F" not in called_occurrences
