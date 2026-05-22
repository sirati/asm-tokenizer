"""Variant-selection helpers for the splice walker.

Single concern: translate between the three flavors of variant index
the walker juggles --

* vkey == ``variant_ref_offset`` (the FID-keyed cross-section variant
  identity).
* in-section ``v_idx`` (position in ``section.variants``).
* ``section_variant_index`` carried in a ``per_call_entry`` (the
  callee section's v_idx for that call site).

Used by ``tokenizer.aligned_data.loader.decoded.splice``. Not part of
the public splice API.
"""

from __future__ import annotations

from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX


def selection_v_idxs_in_section(
    section, selection_vkeys: frozenset
) -> frozenset:
    """Translate a vkey selection to the v_idxs present in ``section``.

    Each ``section.variants[v]`` carries a ``variant_ref_offset`` (vkey);
    the returned frozenset is the subset of v_idxs whose variant_ref_
    offset is in ``selection_vkeys``. Used by the inlining-equivalence
    check and by the per_call_entries lookup for callee variant choice.
    """
    return frozenset(
        v_idx
        for v_idx, variant in enumerate(section.variants)
        if variant.variant_ref_offset in selection_vkeys
    )


def called_by_in_selection(
    section, selection_v_idxs: frozenset, called_idx: int
) -> frozenset:
    """Subset of ``selection_v_idxs`` whose variant called ``called_idx``.

    A variant ``v`` "calls" ``called_idx`` iff some entry in
    ``section.variants[v].per_call_entries`` has its first field equal
    to ``called_idx``.
    """
    return frozenset(
        v
        for v in selection_v_idxs
        if any(
            ce[0] == called_idx for ce in section.variants[v].per_call_entries
        )
    )


def lookup_callee_variant_for(
    variant, called_idx: int
) -> "int | None":
    """Return ``section_variant_index`` for the first per-call entry
    whose ``called_idx`` matches; ``None`` if this variant didn't call
    ``called_idx``.
    """
    for ce_called_idx, sv_idx in variant.per_call_entries:
        if ce_called_idx == called_idx:
            return sv_idx
    return None


def _usable(J: "int | None") -> bool:
    """A J is usable iff it exists and isn't the missing-vkey sentinel.

    ``MISSING_VARIANT_INDEX`` (0xFFFE) is stamped by the BIN writer when
    a per-call entry records "this call existed" but the callee's
    variant set doesn't include the caller's vkey (cross-arm vkey
    mismatch after pass-1 drop rules). Such a J cannot resolve to a
    real callee variant, so the walker must treat the slot the same as
    a missing/extern callee -- leave the call-site tokens, do not
    splice a body.
    """
    return J is not None and J != MISSING_VARIANT_INDEX


def choose_callee_variant(
    section,
    primary_variant_idx: int,
    called_by: frozenset,
    called_idx: int,
) -> "int | None":
    """Pick the callee variant index ``J`` to pass to the decode callback.

    Three-level fallback chain (canonical plan extension; design
    decision approved by orchestrator under "Phase 2.W ambiguity",
    reasoning summary inlined below):

    1. Primary's ``per_call_entries`` for ``called_idx`` → J. Applies
       regardless of flag. The legacy ``version=0`` hardcode is
       replaced with data-driven J: e.g. J=2 if primary called K with
       J=2 instead of always J=0.
    2. Else: the LOWEST v_idx in ``called_by`` (v_idxs whose vkey ∈
       ``initial_selection_vkeys`` AND who called ``called_idx``) →
       that v's per_call_entry → J. Deterministic without relying on
       set iteration order. Reachable only when primary didn't call
       ``called_idx``. Under flag ON, the D5 skip guarantees
       ``called_by`` is non-empty when K isn't skipped; under flag OFF
       it may still be empty, triggering level 3.
    3. Else (flag OFF only): the LOWEST v_idx among ALL variants in
       the section that called ``called_idx``. By construction of
       ``section.call_targets`` (union of every variant's
       per_call_entries), at least one variant called K → this level
       is total for non-sentinel J.

    Each level skips sentinel-J entries (``MISSING_VARIANT_INDEX``);
    returns ``None`` if no variant in the section offers a usable J.
    The walker treats ``None`` as not-spliceable (same as a missing
    callee), per the BIN format spec for vkey-mismatch slots.
    """
    primary_J = lookup_callee_variant_for(
        section.variants[primary_variant_idx], called_idx
    )
    if _usable(primary_J):
        return primary_J
    for v in sorted(called_by):
        J = lookup_callee_variant_for(section.variants[v], called_idx)
        if _usable(J):
            return J
    # Final fallback: scan ALL variants in the section for the lowest
    # v_idx that called this target with a non-sentinel J.
    for v_idx in range(len(section.variants)):
        J = lookup_callee_variant_for(section.variants[v_idx], called_idx)
        if _usable(J):
            return J
    return None


def narrow_selection_vkeys(
    section,
    callee_section,
    called_by: frozenset,
    called_idx: int,
) -> frozenset:
    """Compute the narrowed vkey set to pass into the callee recursion.

    For each ``v`` in ``called_by``, look up its ``per_call_entry`` for
    ``called_idx`` to get the callee variant index ``J_v``, then
    translate ``J_v`` through the callee section to the callee
    variant's vkey (``callee_section.variants[J_v].variant_ref_offset``).
    The set of those vkeys is the new selection at the callee level.
    """
    new_vkeys: set = set()
    for v in called_by:
        J_v = lookup_callee_variant_for(section.variants[v], called_idx)
        if not _usable(J_v):
            # ``called_by`` includes v iff v called the target, but the
            # per_call_entry may carry MISSING_VARIANT_INDEX (vkey
            # mismatch) -- such a v contributes no faithful callee vkey.
            continue
        new_vkeys.add(callee_section.variants[J_v].variant_ref_offset)
    return frozenset(new_vkeys)
