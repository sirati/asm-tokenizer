"""Encoder-allocation order preservation in the parsed-record iterator.

The encoder allocates per-category identities in encounter order and
appends each allocation to ``metadata[Category]``; the CSV writer emits
those arrays in that order with no sort. Every downstream consumer of
``ParsedRecord.called_funcs`` operates positionally on the per-category
sub-list, so the parser MUST preserve that order instead of dedupe-via-
set + alphabetical sort (which destroyed it pre-refactor).

These tests pin:

* per-category encoder-allocation order survives the parser
  (no alphabetisation);
* categories concatenate in LOCAL -> PLT -> EXTERN order;
* a name appearing in two categories still produces two distinct
  ``(name, type)`` entries.
"""

from __future__ import annotations

import json

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.parsed_record_iter import _extract_called_funcs


def _row_with_v2_metadata(metadata: dict) -> tuple[list[str], dict[str, int]]:
    row = ["fn", json.dumps(metadata)]
    column_index = {"func_name": 0, "metadata": 1}
    return row, column_index


def test_v2_local_funcs_preserve_encounter_order_no_alphabetisation():
    """``local_funcs`` array order is the encoder's LOCAL identity index;
    it must survive the parser verbatim, not be alphabetised."""
    metadata = {
        "local_funcs": [{"name": "b"}, {"name": "a"}, {"name": "c"}],
    }
    row, column_index = _row_with_v2_metadata(metadata)

    called, extern_libraries = _extract_called_funcs(row, column_index)

    # NOT alphabetised; matches the CSV array's encoder-allocation order.
    assert called == [
        ("b", CallTargetType.LOCAL),
        ("a", CallTargetType.LOCAL),
        ("c", CallTargetType.LOCAL),
    ]
    assert extern_libraries == {}


def test_v2_cross_category_same_name_stays_distinct_and_local_precedes_plt():
    """A name shared between LOCAL and PLT yields two distinct entries,
    with LOCAL preceding PLT per the (LOCAL -> PLT -> EXTERN)
    concatenation order."""
    metadata = {
        "local_funcs": [{"name": "foo"}],
        "plt_funcs": [{"name": "foo"}],
    }
    row, column_index = _row_with_v2_metadata(metadata)

    called, extern_libraries = _extract_called_funcs(row, column_index)

    assert called == [
        ("foo", CallTargetType.LOCAL),
        ("foo", CallTargetType.PLT),
    ]
    assert extern_libraries == {}


def test_v2_all_three_categories_concat_local_plt_extern():
    """When all three categories carry entries, the concatenation order
    is LOCAL -> PLT -> EXTERN, and each category's sub-order matches the
    CSV array order."""
    metadata = {
        "local_funcs": [{"name": "loc2"}, {"name": "loc1"}],
        "plt_funcs": [{"name": "plt_b"}, {"name": "plt_a"}],
        "ext_funcs": [{"name": "ext_z"}, {"name": "ext_y"}],
    }
    row, column_index = _row_with_v2_metadata(metadata)

    called, extern_libraries = _extract_called_funcs(row, column_index)

    assert called == [
        ("loc2", CallTargetType.LOCAL),
        ("loc1", CallTargetType.LOCAL),
        ("plt_b", CallTargetType.PLT),
        ("plt_a", CallTargetType.PLT),
        ("ext_z", CallTargetType.EXTERN),
        ("ext_y", CallTargetType.EXTERN),
    ]
    assert extern_libraries == {}


def test_v2_within_category_dedupe_keeps_first_encounter():
    """Repeats within one category dedupe to the FIRST encounter
    position (not the last); cross-category remains distinct."""
    metadata = {
        "local_funcs": [
            {"name": "x"},
            {"name": "y"},
            {"name": "x"},  # alias of position 0; must not re-emit
        ],
        "plt_funcs": [{"name": "x"}],  # distinct typed entry
    }
    row, column_index = _row_with_v2_metadata(metadata)

    called, extern_libraries = _extract_called_funcs(row, column_index)

    assert called == [
        ("x", CallTargetType.LOCAL),
        ("y", CallTargetType.LOCAL),
        ("x", CallTargetType.PLT),
    ]
    assert extern_libraries == {}
