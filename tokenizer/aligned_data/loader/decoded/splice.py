"""Depth-capped DFS splicer: caller body + (compacted) callee bodies.

Sole concern of this module: walk one root function's call-target graph,
decode each visited callee, concatenate the resulting per-function
arrays verbatim across the splice tree, and run a single per-Category
identity-space compaction pass at the top level so the final
``DecodedFunction`` carries dense ``[0, K)`` uint16 identities.

The splicer is **pure on its inputs**: callee decode + presence are
delegated to ``decode_callee_to_staging`` and ``is_callee_present``
callbacks supplied by the caller. The Phase-4 wiring closes those
callbacks over a ``BinarySession``; this module's tests close them over
hand-built stubs.

Algorithm (per plan ``## Algorithm changes`` + ``## Locked-in decisions``
items 8, 9, 13, 19, 22, 26, 28, 30, 31):

* Cycle key = ``(arm, section_offset)`` (decision 13). Initialised with
  the root's own key, so a callee that recurses back into the root is
  caught on the first level.
* Depth budget decrements once per recursion step. At ``depth == 0`` or
  on an empty ``call_targets`` list, the walker returns the staging
  unchanged.
* No per-Category running-max rebase: identity arrays are concatenated
  verbatim across the splice (FID-keyed staging arrays carry the
  callee's globally-unique FID, so the same callee gets the same value
  everywhere; per-function-counter categories carry per-source values
  that compaction densifies).
* Top-level compaction (:func:`_compact_ids`, plan Decision 28) maps
  the per-Category value space to a dense ``[0, K)`` range,
  first-occurrence wins. Aliases collapse to the same compacted id;
  sentinel positions map to the reserved u16 sentinel ``0xFFFF``.
* DAG-active-path visited semantics: a callee FID is added to
  ``visited`` before recursion and removed after. A callee reachable
  through two separate branches gets spliced once per visit; only an
  *active* call chain back to itself blocks the recursion.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from tokenizer.tokens import Category

from .decoded_function import DecodedFunction
from .extract import _StagingDecoded


# Public uint16 sentinel that consumers see in the post-compaction
# :class:`DecodedFunction` identity arrays. Decision 7 + 31.
IDENTITY_SENTINEL = np.uint16(0xFFFF)
_IDENTITY_MAX_NON_SENTINEL = 0xFFFE


# ---------------------------------------------------------------------------
# Compaction helper -- plan Decision 28 + 30 + 31
# ---------------------------------------------------------------------------


def _input_sentinel_for(dtype: np.dtype) -> int:
    """Return the staging-side sentinel value for ``dtype``.

    The FID-keyed staging arrays are ``uint32`` and use ``0xFFFFFFFF``
    as the sentinel; the per-function-counter categories stay
    ``uint16`` with sentinel ``0xFFFF``. Compaction folds either input
    sentinel to the public uint16 sentinel in its output.
    """
    if dtype == np.uint32:
        return 0xFFFFFFFF
    if dtype == np.uint16:
        return 0xFFFF
    raise ValueError(
        f"unsupported staging identity dtype {dtype!r}; "
        "expected uint16 or uint32"
    )


def _compact_ids(ids: np.ndarray, *, offset: int = 0) -> np.ndarray:
    """Compact a per-Category staging identity array to dense uint16 ids.

    Plan Decision 28 + 31. The mapping is first-occurrence wins:
    position ``i`` gets the same compacted id as the FIRST position
    holding ``ids[i]``. Sentinel positions (value equal to the dtype's
    staging sentinel from :func:`_input_sentinel_for`) map to the
    public uint16 sentinel ``0xFFFF`` in the output. Non-sentinel
    unique values get compact ids in ``[offset, offset + n_unique)``;
    a non-zero ``offset`` lets a consumer reserve a low-id range
    (default ``0``: ids start at zero).

    The algorithm runs in a single Python pass over the array followed
    by two vectorised numpy passes; for typical post-splice array
    sizes (a few hundred per Category per function) the per-occurrence
    overhead is negligible. A pure-numpy alternative via
    ``np.unique(return_inverse=True)`` exists but sorts the values,
    losing the first-occurrence-wins property; the explicit dict-based
    pass keeps that invariant.

    The output dtype is always ``uint16``; ``n_unique`` is asserted
    against the u16 ceiling so a regression that floods the splice with
    > 65534 distinct identities surfaces loudly rather than silently
    aliasing into the sentinel.
    """
    N = ids.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.uint16)

    input_sentinel = _input_sentinel_for(ids.dtype)
    is_sentinel = ids == input_sentinel
    mask = ~is_sentinel.copy()                 # True at unique-candidate positions
    remap_lookup = np.zeros(N, dtype=np.uint32)
    seen: "dict[int, int]" = {}

    for i in range(N):
        if is_sentinel[i]:
            continue
        v = int(ids[i])
        first_j = seen.get(v)
        if first_j is None:
            seen[v] = i
        else:
            mask[i] = False
            remap_lookup[i] = first_j

    n_unique = int(mask.sum())
    if n_unique > _IDENTITY_MAX_NON_SENTINEL:
        raise AssertionError(
            f"compacted id space {n_unique} exceeds u16 - 1 "
            f"({_IDENTITY_MAX_NON_SENTINEL}); a single spliced view "
            "carries more distinct identities than the public u16 "
            "identity domain can hold"
        )

    remap_lookup[mask] = np.arange(offset, offset + n_unique, dtype=np.uint32)
    alias_positions = (~mask) & (~is_sentinel)
    if alias_positions.any():
        # Resolve aliases to their first-occurrence's compact id. One
        # vectorised double-indexing step: remap_lookup[alias] currently
        # holds the first-occurrence position, indexing again resolves
        # it to the first-occurrence's compact id.
        remap_lookup[alias_positions] = remap_lookup[remap_lookup[alias_positions]]

    # Sentinel positions take the reserved sentinel in the OUTPUT u16
    # space; compaction never lets a sentinel become a non-sentinel and
    # vice versa.
    remap_lookup[is_sentinel] = int(IDENTITY_SENTINEL)
    return remap_lookup.astype(np.uint16)


# ---------------------------------------------------------------------------
# Concatenation helper
# ---------------------------------------------------------------------------


def _concat_staging(
    root: _StagingDecoded, *callee_pieces: _StagingDecoded
) -> _StagingDecoded:
    """Concatenate the root's staging with every callee piece (in order).

    Identity arrays are concatenated VERBATIM per Category -- no rebase,
    no running-max arithmetic. Compaction (top-level) collapses
    duplicates afterwards.

    ``func_name`` + ``metadata`` propagate from the root only -- the
    spliced view is conceptually one function from the consumer's
    perspective. Number side-arrays concatenate in stream order
    (the per-function ``_decode_to_staging`` already emits them in
    stream-position order).
    """
    all_pieces: Tuple[_StagingDecoded, ...] = (root, *callee_pieces)
    real_tokens = np.concatenate([p.real_tokens for p in all_pieces])
    identities = {
        c: np.concatenate([p.identities[c] for p in all_pieces])
        for c in Category
    }
    numbers_significant = np.concatenate(
        [p.numbers_significant for p in all_pieces]
    )
    numbers_sign_exponent = np.concatenate(
        [p.numbers_sign_exponent for p in all_pieces]
    )
    return _StagingDecoded(
        real_tokens=real_tokens,
        identities=identities,
        numbers_significant=numbers_significant,
        numbers_sign_exponent=numbers_sign_exponent,
        func_name=root.func_name,
        metadata=root.metadata,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def splice_with_callees(
    *,
    root_staging: _StagingDecoded,
    root_arm: str,
    root_section,
    root_section_offset: int,
    decode_callee_to_staging: Callable[[int, str], Tuple[_StagingDecoded, object]],
    is_callee_present: Callable[[int, str], bool],
    max_depth: int,
) -> DecodedFunction:
    """Depth-capped DFS splice with top-level per-Category compaction.

    Args:
        root_staging: Already-decoded :class:`_StagingDecoded` for the
            root caller. Its ``real_tokens`` ALWAYS appear unchanged at
            the head of the spliced output. FID-keyed identity arrays
            in the staging carry the resolved callee FIDs (the session
            wiring resolves these via the caller's Section before
            calling here); per-function-counter category arrays carry
            per-source values that compaction densifies.
        root_arm: ``"matched"`` or ``"unmatched"`` -- the arm the root
            was loaded from. Propagated into the cycle key alongside
            section offsets (plan Decision 13).
        root_section: Section object describing the root's
            ``call_targets``. Only the ``call_targets`` attribute is
            read; each call_target must expose ``function_section_ptr``
            (callee section offset).
        root_section_offset: Section offset of the root itself. Seeded
            into the visited set so a callee that recurses back into
            the root is caught on the first level.
        decode_callee_to_staging: ``(callee_section_offset, arm) ->
            (_StagingDecoded, callee_section)`` callback. Loads +
            FID-resolved-decodes the callee and returns its own parsed
            section so the walker can recurse on the callee's
            call_targets. Phase 4 wires this through
            :class:`BinarySession`; tests inject a stub.
        is_callee_present: ``(callee_section_offset, arm) -> bool``.
            Returns ``True`` iff the callee was emitted in the requested
            arm and will resolve via ``decode_callee_to_staging``. Externs
            and missing sections return ``False`` -- their call-site
            tokens stay in the caller's stream but their bodies are NOT
            spliced.
        max_depth: Recursion budget. ``0`` returns the (compacted) root
            staging unchanged. Each level of nested callee consumes one
            budget unit.

    Returns:
        A :class:`DecodedFunction` whose arrays are the concatenation of
        the root and every successfully-spliced callee subtree (DFS
        call-target order), with per-Category identities compacted to
        dense ``[0, K)`` uint16 values. The same callee referenced from
        multiple sites in the splice tree shares one compacted id.
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0; got {max_depth}")

    visited: set[Tuple[str, int]] = {(root_arm, root_section_offset)}
    spliced = _decode_then_splice(
        staging=root_staging,
        section=root_section,
        arm=root_arm,
        depth=max_depth,
        visited=visited,
        decode_callee_to_staging=decode_callee_to_staging,
        is_callee_present=is_callee_present,
    )

    # ---- Top-level per-Category compaction (plan Decision 30) ----
    final_identities = {
        c: _compact_ids(spliced.identities[c]) for c in Category
    }

    return DecodedFunction(
        real_tokens=spliced.real_tokens,
        identities=final_identities,
        numbers_significant=spliced.numbers_significant,
        numbers_sign_exponent=spliced.numbers_sign_exponent,
        func_name=spliced.func_name,
        metadata=spliced.metadata,
    )


# ---------------------------------------------------------------------------
# Inner recursion
# ---------------------------------------------------------------------------


def _decode_then_splice(
    *,
    staging: _StagingDecoded,
    section,
    arm: str,
    depth: int,
    visited: set,
    decode_callee_to_staging: Callable[[int, str], Tuple[_StagingDecoded, object]],
    is_callee_present: Callable[[int, str], bool],
) -> _StagingDecoded:
    """Recursive worker. See :func:`splice_with_callees` for the contract.

    Returns a :class:`_StagingDecoded` whose per-Category identity
    arrays are the verbatim concatenation of the current node's staging
    + each spliced callee subtree's staging. No per-Category rebase --
    compaction at the top level handles the identity-domain density.
    """
    # Leaf conditions: no more recursion budget, or no callees to expand.
    if depth == 0 or len(section.call_targets) == 0:
        return staging

    callee_pieces: list[_StagingDecoded] = []
    for ct in section.call_targets:
        callee_offset = ct.function_section_ptr
        cycle_key = (arm, callee_offset)

        if cycle_key in visited:
            # Active-path cycle: skip splice, leave call-site tokens
            # in the caller's stream.
            continue
        if not is_callee_present(callee_offset, arm):
            # Extern / missing section: same treatment as cycle --
            # call-site tokens stay, body NOT spliced.
            continue

        visited.add(cycle_key)
        try:
            callee_staging, callee_section = decode_callee_to_staging(
                callee_offset, arm
            )
            callee_subtree = _decode_then_splice(
                staging=callee_staging,
                section=callee_section,
                arm=arm,
                depth=depth - 1,
                visited=visited,
                decode_callee_to_staging=decode_callee_to_staging,
                is_callee_present=is_callee_present,
            )
        finally:
            # DAG-active-path semantics: a callee reachable via a
            # different branch at the same depth must be allowed to
            # splice again.
            visited.discard(cycle_key)

        callee_pieces.append(callee_subtree)

    return _concat_staging(staging, *callee_pieces)
