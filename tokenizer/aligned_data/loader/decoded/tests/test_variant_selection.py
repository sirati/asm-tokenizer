"""Equivalence + identity pins for ``called_by_in_selection``.

The production hotspot was rewritten from a per-call ``any``-over-genexpr
to a cached inverted-index intersection. This test pins the new output
byte-for-byte against a frozen reference copy of the OLD logic across a
spread of synthetic sections, plus a directly enumerated truth table.
"""

from __future__ import annotations

import itertools
import random

from tokenizer.aligned_data.matched_sections_bin import Section, VariantBlock
from tokenizer.aligned_data.loader.decoded._variant_selection import (
    called_by_in_selection,
)


def _old_called_by_in_selection(section, selection_v_idxs, called_idx):
    """Verbatim pre-rewrite implementation, kept as the oracle."""
    return frozenset(
        v
        for v in selection_v_idxs
        if any(
            ce[0] == called_idx
            for ce in section.variants[v].per_call_entries
        )
    )


def _variant(per_call_entries):
    return VariantBlock(
        variant_ref_offset=0,
        data_offset_shifted=0,
        per_call_entries=list(per_call_entries),
    )


def _section(variants):
    return Section(
        function_name_ptr=0,
        section_offset=0,
        call_targets=[],
        variants=list(variants),
    )


def _random_section(rng, n_variants, n_targets, max_entries):
    variants = []
    for _ in range(n_variants):
        n_entries = rng.randint(0, max_entries)
        entries = [
            (rng.randrange(n_targets), rng.randrange(8))
            for _ in range(n_entries)
        ]
        variants.append(_variant(entries))
    return _section(variants)


def test_matches_old_on_enumerated_small_cases():
    # Exhaustively enumerate tiny sections: every variant either does or
    # does not call each of two targets, over all selection subsets.
    targets = [0, 1]
    for call_pattern in itertools.product([(), (0,), (1,), (0, 1)], repeat=3):
        section = _section(
            _variant([(t, 0) for t in pat]) for pat in call_pattern
        )
        all_v = range(len(section.variants))
        for r in range(len(call_pattern) + 1):
            for selection in itertools.combinations(all_v, r):
                sel = frozenset(selection)
                for called_idx in targets + [2]:  # 2 = uncalled target
                    new = called_by_in_selection(section, sel, called_idx)
                    old = _old_called_by_in_selection(section, sel, called_idx)
                    assert new == old
                    assert isinstance(new, frozenset)


def test_matches_old_on_random_spread():
    rng = random.Random(0xA5C11)
    for _ in range(200):
        section = _random_section(
            rng,
            n_variants=rng.randint(1, 12),
            n_targets=rng.randint(1, 6),
            max_entries=rng.randint(0, 10),
        )
        n_v = len(section.variants)
        # random selection subset (including out-of-spread indices is not
        # possible here since selection comes from valid v_idxs)
        sel = frozenset(
            v for v in range(n_v) if rng.random() < 0.5
        )
        for called_idx in range(8):
            new = called_by_in_selection(section, sel, called_idx)
            old = _old_called_by_in_selection(section, sel, called_idx)
            assert new == old


def test_empty_section_and_empty_selection():
    empty = _section([])
    assert called_by_in_selection(empty, frozenset(), 0) == frozenset()
    assert called_by_in_selection(empty, frozenset({0, 1}), 0) == frozenset()

    section = _section([_variant([(3, 0)]), _variant([(3, 1), (4, 0)])])
    assert called_by_in_selection(section, frozenset(), 3) == frozenset()


def test_repeated_query_same_section_is_consistent():
    # The cache must not change results across repeated queries.
    section = _section(
        [_variant([(1, 0), (2, 0)]), _variant([(2, 1)]), _variant([])]
    )
    sel = frozenset({0, 1, 2})
    first = [called_by_in_selection(section, sel, k) for k in range(4)]
    second = [called_by_in_selection(section, sel, k) for k in range(4)]
    assert first == second
    assert first[1] == frozenset({0})
    assert first[2] == frozenset({0, 1})
    assert first[0] == frozenset()
