"""Depth-capped DFS splicer: caller body + (rebased) callee bodies.

Sole concern of this module: walk one root function's call-target graph,
decode each visited callee, and concatenate the resulting
``DecodedFunction`` arrays after rebasing per-``Category`` identity arrays
so the spliced stream's identities stay collision-free.

The splicer is **pure on its inputs**: callee loading + decoding is
delegated to ``decode_callee`` and ``is_callee_present`` callbacks
supplied by the caller. The Phase-4 wiring closes those callbacks over a
``BinarySession``; this module's tests close them over hand-built stubs.

Algorithm (per plan ## Algorithm — splice_with_callees +
## Locked-in decisions items 7, 8, 9, 13):

* Cycle key = ``(arm, section_offset)`` (decision 13). Initialised with the
  root's own key, so a callee that recurses back into the root is caught.
* Depth budget decrements once per recursion step. At ``depth == 0`` or on
  an empty ``call_targets`` list, the function returns its own decoded body
  unchanged — the caller's call-site real-tokens stay in the stream, but
  no callee bodies are spliced in beyond that point.
* Per-``Category`` running max excludes the ``0xFFFF`` sentinel; the next
  callee's per-category offset is ``running_max[c] + 1`` (or ``0`` if the
  caller has emitted no non-sentinel identity for that category).
* Rebase is a vectorised ``np.where`` that keeps the sentinel sticky and
  clips any non-sentinel value that overflows ``0xFFFE`` to the sentinel.
* DAG-active-path visited semantics: a callee FID is added to ``visited``
  before recursion and removed after. A callee reachable through two
  separate branches gets spliced once per visit; only an *active* call
  chain back to itself blocks the recursion. (See plan pseudo-code
  ``visited.add(...) / visited.discard(...)``.)
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from tokenizer.tokens import Category

from .decoded_function import DecodedFunction


# Sentinel value (also enforced by ``DecodedFunction.__post_init__`` /
# ``Category``-array invariants). Centralised here so the rebase + max
# helpers share a single source of truth.
IDENTITY_SENTINEL = np.uint16(0xFFFF)
_IDENTITY_MAX_NON_SENTINEL = 0xFFFE


# ---------------------------------------------------------------------------
# Identity-array helpers
# ---------------------------------------------------------------------------


def _max_non_sentinel(arr: np.ndarray) -> int:
    """Return the largest non-sentinel value in ``arr`` as a Python ``int``.

    Returns ``-1`` if ``arr`` is empty or every entry is the sentinel.
    The ``-1`` return is sentinel-of-a-sentinel: the caller uses it to
    decide whether to apply ``running_max + 1`` (we have a non-sentinel
    baseline) or ``0`` (we don't, start fresh).
    """
    if arr.size == 0:
        return -1
    non_sentinel = arr[arr != IDENTITY_SENTINEL]
    if non_sentinel.size == 0:
        return -1
    return int(non_sentinel.max())


def _rebase_identity_array(arr: np.ndarray, offset: int) -> np.ndarray:
    """Vectorised sentinel-sticky add (u16 clip).

    * ``arr[i] == 0xFFFF`` stays ``0xFFFF`` regardless of ``offset``.
    * ``arr[i] + offset > 0xFFFE`` clips to ``0xFFFF`` (overflow flagged
      as unresolved, per plan ## Locked-in decisions item 7).
    * Otherwise the value is ``arr[i] + offset`` cast back to ``uint16``.

    ``offset`` is a non-negative Python ``int``; the function widens to
    ``uint32`` for the add so values up to ``0xFFFE + offset`` never wrap
    silently inside numpy.
    """
    if offset < 0:
        raise ValueError(f"identity rebase offset must be >= 0; got {offset}")
    if arr.size == 0:
        # Preserve dtype + emptiness for the concat downstream.
        return arr.astype(np.uint16, copy=False)

    wide = arr.astype(np.uint32) + np.uint32(offset)
    overflowed = wide > _IDENTITY_MAX_NON_SENTINEL
    is_sentinel = arr == IDENTITY_SENTINEL
    return np.where(
        is_sentinel | overflowed,
        IDENTITY_SENTINEL,
        wide.astype(np.uint16),
    ).astype(np.uint16, copy=False)


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------


def _concat_decoded(
    root: DecodedFunction, *callee_pieces: DecodedFunction
) -> DecodedFunction:
    """Concatenate the root's arrays with every callee piece (in order).

    ``func_name`` + ``metadata`` propagate from the root only — the spliced
    view is conceptually one function from the consumer's perspective.
    """
    all_pieces: Tuple[DecodedFunction, ...] = (root, *callee_pieces)
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
    return DecodedFunction(
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
    root_decoded: DecodedFunction,
    root_arm: str,
    root_section,
    root_section_offset: int,
    decode_callee: Callable[[int, str], Tuple[DecodedFunction, object]],
    is_callee_present: Callable[[int, str], bool],
    max_depth: int,
) -> DecodedFunction:
    """Depth-capped DFS splice with per-``Category`` identity rebase.

    Args:
        root_decoded: Already-decoded ``DecodedFunction`` for the root
            (the caller). Its ``real_tokens`` ALWAYS appear unchanged at
            the head of the spliced output.
        root_arm: ``"matched"`` or ``"unmatched"`` — the arm the root was
            loaded from. Propagated into the cycle key alongside section
            offsets (plan decision 13).
        root_section: Section object describing the root's call_targets.
            Only the ``call_targets`` attribute is read; each call_target
            must expose ``function_section_ptr`` (callee section offset)
            and ``is_matched``.
        root_section_offset: Section offset of the root itself. Seeded into
            the visited set so a callee that recurses back into the root
            is caught on the first level.
        decode_callee: ``(callee_section_offset, arm) -> (DecodedFunction,
            callee_section)`` callback. Loads + decodes the callee and
            returns its own parsed section so the walker can recurse on
            the callee's call_targets. Phase 4 wires this through
            ``BinarySession``; tests inject a stub.
        is_callee_present: ``(callee_section_offset, arm) -> bool``.
            Returns ``True`` iff the callee was emitted in the requested
            arm and will resolve via ``decode_callee``. Externs and
            missing sections return ``False`` — their call-site tokens
            stay in the caller's stream but their bodies are NOT spliced.
        max_depth: Recursion budget. ``0`` returns ``root_decoded``
            unchanged. Each level of nested callee consumes one budget
            unit; at ``depth == 0`` the inner walker stops descending
            but keeps the caller's call-site tokens in the stream.

    Returns:
        A new ``DecodedFunction`` whose arrays are the concatenation of
        the root and every successfully-spliced callee subtree (DFS
        section-order, per plan ## Locked-in decisions item 9). The
        identity arrays of each callee subtree are rebased per
        ``Category`` against the running max accumulated by the caller +
        any preceding callee subtree at the same level.
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0; got {max_depth}")

    visited: set[Tuple[str, int]] = {(root_arm, root_section_offset)}
    return _decode_then_splice(
        decoded=root_decoded,
        section=root_section,
        arm=root_arm,
        depth=max_depth,
        visited=visited,
        decode_callee=decode_callee,
        is_callee_present=is_callee_present,
    )


# ---------------------------------------------------------------------------
# Inner recursion
# ---------------------------------------------------------------------------


def _decode_then_splice(
    *,
    decoded: DecodedFunction,
    section,
    arm: str,
    depth: int,
    visited: set,
    decode_callee: Callable[[int, str], Tuple[DecodedFunction, object]],
    is_callee_present: Callable[[int, str], bool],
) -> DecodedFunction:
    """Recursive worker. See ``splice_with_callees`` for the contract.

    ``section`` is the already-parsed Section object whose
    ``call_targets`` we walk. ``decoded`` is the matching pre-decoded
    body.
    """
    # Leaf conditions: no more recursion budget, or no callees to expand.
    if depth == 0 or len(section.call_targets) == 0:
        return decoded

    # Per-category running-max identity (excludes the 0xFFFF sentinel
    # per plan decision 7).
    running_max = {
        c: _max_non_sentinel(decoded.identities[c]) for c in Category
    }

    callee_pieces: list[DecodedFunction] = []
    for ct in section.call_targets:
        callee_offset = ct.function_section_ptr
        cycle_key = (arm, callee_offset)

        if cycle_key in visited:
            # Active-path cycle: skip splice, leave call-site tokens
            # in the caller's stream.
            continue
        if not is_callee_present(callee_offset, arm):
            # Extern / missing section: same treatment as cycle —
            # call-site tokens stay, body NOT spliced.
            continue

        visited.add(cycle_key)
        try:
            callee_decoded, callee_section = decode_callee(callee_offset, arm)
            callee_subtree = _decode_then_splice(
                decoded=callee_decoded,
                section=callee_section,
                arm=arm,
                depth=depth - 1,
                visited=visited,
                decode_callee=decode_callee,
                is_callee_present=is_callee_present,
            )
        finally:
            # DAG-active-path semantics: a callee reachable via a different
            # branch at the same depth must be allowed to splice again.
            visited.discard(cycle_key)

        # Per-category rebase against the running max accumulated so far.
        rebased_identities = {}
        for c in Category:
            base = running_max[c]
            offset = base + 1 if base >= 0 else 0
            rebased = _rebase_identity_array(
                callee_subtree.identities[c], offset
            )
            rebased_identities[c] = rebased
            # Update running max with the rebased values (sentinels
            # excluded). A subtree that emits only sentinels leaves the
            # running max unchanged; that's correct — the next callee
            # offsets off the same baseline.
            subtree_max = _max_non_sentinel(rebased)
            if subtree_max > base:
                running_max[c] = subtree_max

        callee_pieces.append(
            DecodedFunction(
                real_tokens=callee_subtree.real_tokens,
                identities=rebased_identities,
                numbers_significant=callee_subtree.numbers_significant,
                numbers_sign_exponent=callee_subtree.numbers_sign_exponent,
                func_name=callee_subtree.func_name,
                metadata=callee_subtree.metadata,
            )
        )

    return _concat_decoded(decoded, *callee_pieces)
