"""Pass-2 unmatched-arm ``all_called`` union preserves encoder-allocation order.

The unmatched-arm aggregator ``group_unmatched_entries_by_function``
collects callees across every version's entry into a per-function
``all_called`` union. The order of that union is now insertion-order
(first-seen wins) instead of an alphabetical post-sort; the consuming
writer ``write_unmatched_sections_pass2`` materialises it as
``list(all_called)`` so the section's typed call-target table follows
the same encounter order the parsed-record layer carries (plan
Decisions 20 + 21).

These tests pin the new shape at the grouper level — the consumer's
``list(all_called)`` step is a thin projection of the grouper's
insertion-ordered dict, so a regression that re-introduces a post-union
``sorted(...)`` would surface as a grouper-level shape mismatch. The
end-to-end CSV/BIN ordering is exercised indirectly by the
section-writer + iter_sections_bin tests in the rest of the suite, which
all consume the same ``typed_unique_called`` list this aggregator feeds.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.memmap_builder._pass2 import group_unmatched_entries_by_function


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
