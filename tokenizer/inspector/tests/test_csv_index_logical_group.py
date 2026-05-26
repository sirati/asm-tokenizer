"""Tests for the logical-name grouping helpers inside
:class:`~tokenizer.inspector._render._ftl_backend._csv_index.CsvIndex`.

The grouping pass collapses canonical-name lockstep yields whose
``@thunk:<offset>`` suffix differs across ELF builds so the inspector
function list surfaces one entry per source-level symbol rather than
one per per-binary placeholder address.
"""

from __future__ import annotations

from typing import List

import numpy as np

from tokenizer.aligned_data.parsed_record_iter import (
    LockstepYield,
    Matched,
    ParsedRecord,
    Unmatched,
)
from tokenizer.inspector._render._ftl_backend._csv_index import (
    _build_logical_groups,
    _content_hash_for_group,
    _variant_lookup,
)


def _rec(name: str, content_hash: int = 0) -> ParsedRecord:
    """Build a minimal :class:`ParsedRecord` for grouping tests; only
    ``func_name`` + ``content_hash`` are read by the grouping pass."""
    return ParsedRecord(
        func_name=name,
        insn_runlength=np.zeros(0, dtype=np.uint8),
        block_runlength=np.zeros(0, dtype=np.uint16),
        tokens=np.zeros(0, dtype=np.uint16),
        called_funcs=[],
        extern_libraries={},
        content_hash=content_hash,
    )


def test_build_groups_collapses_thunk_offsets() -> None:
    """Three Unmatched yields with the same logical name (different
    ``@thunk:<offset>`` per binary) collapse to one group; an unrelated
    function keeps its own group."""
    records: List[LockstepYield] = [
        Unmatched(func_name="gzseek@thunk:1056296", record=_rec("gzseek@thunk:1056296", 11), variant_index=0),
        Unmatched(func_name="gzseek@thunk:1056324", record=_rec("gzseek@thunk:1056324", 22), variant_index=1),
        Unmatched(func_name="gzseek@thunk:1056400", record=_rec("gzseek@thunk:1056400", 33), variant_index=2),
        Unmatched(func_name="gzwrite", record=_rec("gzwrite", 44), variant_index=0),
    ]
    groups, logical_names = _build_logical_groups(records)
    assert logical_names == ["gzseek", "gzwrite"]
    assert groups == [[0, 1, 2], [3]]


def test_build_groups_preserves_first_occurrence_order() -> None:
    """Group ordering follows first sighting of each logical name --
    the inspector function list keeps its lockstep-driven order."""
    records: List[LockstepYield] = [
        Unmatched(func_name="b", record=_rec("b"), variant_index=0),
        Unmatched(func_name="a@thunk:1", record=_rec("a@thunk:1"), variant_index=0),
        Unmatched(func_name="a@thunk:2", record=_rec("a@thunk:2"), variant_index=1),
        Unmatched(func_name="c", record=_rec("c"), variant_index=0),
    ]
    _, logical_names = _build_logical_groups(records)
    assert logical_names == ["b", "a", "c"]


def test_build_groups_keeps_distinct_names_distinct() -> None:
    """Names that are not pre/post strip equivalent (e.g. ``gzseek`` vs
    ``gzseek64``) MUST stay in separate groups; the suffix-strip only
    peels off the per-binary thunk offset."""
    records: List[LockstepYield] = [
        Unmatched(func_name="gzseek", record=_rec("gzseek"), variant_index=0),
        Unmatched(func_name="gzseek64", record=_rec("gzseek64"), variant_index=0),
    ]
    groups, logical_names = _build_logical_groups(records)
    assert logical_names == ["gzseek", "gzseek64"]
    assert groups == [[0], [1]]


def test_build_groups_merges_local_and_thunk_under_same_logical() -> None:
    """Variant A defines ``foo`` locally; variant B has a PLT thunk to
    extern ``foo``. Both map to logical name ``foo`` and collapse into
    one inspector function (the user's stated requirement: one entry
    per function-id, not per address)."""
    records: List[LockstepYield] = [
        Unmatched(func_name="foo", record=_rec("foo"), variant_index=0),
        Unmatched(func_name="foo@thunk:9999", record=_rec("foo@thunk:9999"), variant_index=1),
    ]
    groups, logical_names = _build_logical_groups(records)
    assert logical_names == ["foo"]
    assert groups == [[0, 1]]


def test_variant_lookup_matched_returns_variant_record() -> None:
    """``_variant_lookup`` is the per-yield resolver: Matched returns
    the per-variant dict's entry; misses return ``None``."""
    r0 = _rec("foo", content_hash=1)
    r2 = _rec("foo", content_hash=3)
    record = Matched(func_name="foo", records={0: r0, 2: r2})
    assert _variant_lookup(record, 0) is r0
    assert _variant_lookup(record, 2) is r2
    assert _variant_lookup(record, 1) is None


def test_variant_lookup_unmatched_resolves_only_its_slot() -> None:
    """Unmatched answers only the slot it carries; any other slot is
    a miss (the group walker keeps probing other members)."""
    rec = _rec("foo")
    record = Unmatched(func_name="foo", record=rec, variant_index=3)
    assert _variant_lookup(record, 3) is rec
    assert _variant_lookup(record, 0) is None
    assert _variant_lookup(record, 2) is None


def test_content_hash_for_group_picks_first_member() -> None:
    """The first member's content hash anchors the group's stable key;
    deterministic given the lockstep yield order."""
    records: List[LockstepYield] = [
        Unmatched(func_name="foo@thunk:1", record=_rec("foo@thunk:1", 42), variant_index=0),
        Unmatched(func_name="foo@thunk:2", record=_rec("foo@thunk:2", 99), variant_index=1),
    ]
    groups, _ = _build_logical_groups(records)
    assert _content_hash_for_group(records, groups[0]) == 42
