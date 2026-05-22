"""Shared category-grouped first-seen union of per-variant typed callees.

Both pass-1 producers — the matched arm in :mod:`passes`
(:func:`process_matched_function`) and the unmatched arm in
:mod:`_pass2` (:func:`group_unmatched_entries_by_function`) — collapse
the per-variant ``ParsedRecord.called_funcs`` lists of one function
group into a single section-level ``typed_unique_called`` list. The
list feeds :func:`_pass2._build_call_targets_spec`, which emits one
``CallTargetSpec`` per entry into the BIN's ``Section.call_targets[]``
table.

The loader-side docstring at
:mod:`tokenizer.aligned_data.loader._session_splice` asserts that
``Section.call_targets[]`` is encounter-ordered within each
:class:`CallTargetType` and concatenated LOCAL → PLT → EXT. The
per-row :attr:`ParsedRecord.called_funcs` already satisfies the
invariant (see
:func:`tokenizer.aligned_data.parsed_record_iter._called_from_v2_metadata`),
but an inline first-seen union across variants interleaves categories
when later variants contribute a novel callee whose category sits
ahead of an already-seen category. Centralising the union here lets
the invariant be enforced once, at the seam where the section-level
list is produced.

Policy: first-seen wins on duplicate ``(name, type)`` pairs across
variants; the surviving entries are then re-grouped by
:class:`CallTargetType` value (LOCAL=0, PLT=1, EXTERN=2) via a stable
sort, so intra-category encoder-allocation order is preserved.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from ..aligned_data.call_target_type import CallTargetType


TypedCallee = Tuple[str, CallTargetType]


def category_grouped_first_seen_union(
    per_variant_called: "Iterable[Iterable[TypedCallee]]",
) -> "List[TypedCallee]":
    """Return the LOCAL → PLT → EXT-grouped first-seen union.

    Iterates ``per_variant_called`` once; for each typed callee tuple,
    record it the first time it is seen and skip on subsequent sights
    (matched + unmatched arms both want the per-section list to carry
    each ``(name, type)`` pair at most once). The accumulator preserves
    encounter order across variants; a final stable sort by
    :class:`CallTargetType` value re-groups the list into LOCAL → PLT
    → EXT blocks while keeping intra-category encounter order (Python's
    ``sorted`` is stable).

    The intra-category order matters: per the encoder-allocation-order
    contract on :attr:`ParsedRecord.called_funcs`, the K-th name in
    category C in the merged list is the function whose encoder-
    allocated identity for C is K (anchored on the first variant that
    introduced it; later variants only contribute novel names at the
    intra-category tail).
    """
    seen: "set[TypedCallee]" = set()
    encounter_ordered: "List[TypedCallee]" = []
    for variant_called in per_variant_called:
        for typed_callee in variant_called:
            if typed_callee not in seen:
                seen.add(typed_callee)
                encounter_ordered.append(typed_callee)
    return sorted(encounter_ordered, key=lambda nt: nt[1])
