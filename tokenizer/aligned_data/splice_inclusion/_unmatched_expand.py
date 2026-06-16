"""Unmatched-outline inlining: surface matched callees behind unmatched edges.

Single concern (opt-in semantic transform, gated by the caller): given a
splice level's resolved call edges -- each tagged matched / unmatched by
the BIN's per-call-target ``is_matched`` flag (the callee's arm) -- replace
every UNMATCHED edge with the MATCHED edges reachable BEHIND it, recursing
through unmatched->unmatched chains up to a depth cap (default 3) with
cycle guarding. MATCHED edges pass through unchanged. The returned matched
edge set is what the caller feeds into the shared :class:`OnceOnlyInclusion`
outline detector instead of the raw level.

WHY this is its own concern, owned here and not in either loader: an
unmatched call target is most likely a compiler-OUTLINED function, whose
body we DO carry (the edge resolves in-arm) but whose presence skews the
columnwise-ALL "reached by every variant => prune" outline heuristic --
the heuristic wants to see the MATCHED callees an outline ultimately
reaches, not the outline shell. Both decode loaders (the batch-decode
callee walk and the vector-batch inclusion BFS) apply the SAME transform
to their per-level edge set before driving the decider, so the transform
must live in ONE module with a loader-agnostic API; duplicating it would
let the two loaders' semantics drift (and the cross-loader byte-identity
gate would break).

The boundary sentence (design-first rule):

  *Given a level's resolved edges (each carrying its decider mask row, a
  callee identity dedup key, an is-matched bit, and an opaque
  loader-native payload) plus a callback that resolves ONE node's direct
  child edges and a depth cap, return the matched edges -- direct matched
  plus matched edges surfaced behind unmatched edges within <= cap
  unmatched->unmatched levels -- preserving mask row and a deterministic
  order, never re-surfacing a node already on the current expansion path
  (cycle guard) or already surfaced (dedup).*

The transform knows NOTHING about sections, catalogs, variants, bodies, or
tokens: ``Edge.payload`` is opaque and ``resolve_children`` is the only
means to walk deeper. The loader supplies both. This keeps the module a
pure cap-bounded graph expansion that either loader can drive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

__all__ = ["Edge", "expand_unmatched_edges"]


@dataclass(frozen=True)
class Edge:
    """One resolved call edge at a splice level (loader-agnostic).

    ``mask_row`` is the :class:`OnceOnlyInclusion` mask row (the decider
    variant / sampled slot) the edge belongs to -- preserved verbatim
    through the expansion so a matched callee surfaced behind an
    unmatched edge is attributed to the SAME row the unmatched edge was.

    ``dedup_key`` is the callee's once-only identity (the same key the
    caller feeds :meth:`OnceOnlyInclusion.step_level` as
    ``callee_function_id`` -- the callee section offset / index). The
    expansion uses it ONLY for cycle / dedup bookkeeping within one level;
    the decider still applies its own cross-level once-only rule.

    ``is_matched`` is the BIN per-call-target flag: ``True`` when the
    callee resides in the matched arm. ``False`` (unmatched) edges are the
    ones recursed BEHIND; ``True`` edges pass through.

    ``payload`` is an opaque loader-native record (e.g. the batch-decode
    ``ResolvedCalleeMeta`` or the vector-batch ``(node, sec, type)``
    tuple) the caller maps back to its own emission / descent after the
    expansion. The transform never inspects it -- it only routes it.
    """

    mask_row: int
    dedup_key: int
    is_matched: bool
    payload: object


def expand_unmatched_edges(
    edges: Sequence[Edge],
    resolve_children: Callable[[Edge], Sequence[Edge]],
    *,
    max_unmatched_depth: int = 3,
) -> List[Edge]:
    """Surface matched edges behind unmatched edges (cap-bounded, guarded).

    Parameters
    ----------
    edges:
        The level's resolved edges, in the caller's emission order. Each
        carries its ``mask_row`` (preserved), ``dedup_key`` (cycle / dedup
        identity), ``is_matched`` bit, and opaque ``payload``.
    resolve_children:
        Given ONE :class:`Edge` (an unmatched callee), return that
        callee's OWN direct child edges -- exactly the per-node resolution
        the caller already runs to build a level. The returned children
        carry the SAME ``mask_row`` as the unmatched parent (the caller's
        resolver stamps it), so a deep matched callee stays attributed to
        the originating row. May return an empty sequence (a leaf / fully
        gated-out outline).
    max_unmatched_depth:
        Cap on consecutive unmatched->unmatched recursion levels (>= 0).
        ``0`` surfaces nothing behind unmatched edges (they are simply
        dropped, since an unmatched edge is never itself fed to outline
        detection); the default ``3`` follows three levels of nested
        outlines. The cap counts UNMATCHED hops only -- a matched edge
        found at any depth terminates that branch (it is surfaced, not
        recursed).

    Returns
    -------
    list[Edge]
        The MATCHED edges to feed outline detection: every direct matched
        edge of ``edges`` (in input order), then -- per unmatched edge, in
        input order -- the matched edges surfaced behind it (each unmatched
        node's matched children, recursing unmatched children up to the
        cap), de-duplicated within this call on
        ``(mask_row, dedup_key)`` so one row never surfaces the same
        callee twice. Order is deterministic (input order at every level,
        depth-first per unmatched edge) so both loaders produce identical
        sequences.

    The expansion is per-row independent: a ``(mask_row, dedup_key)`` seen
    on one row does not suppress the same callee on another row. Cycle
    guarding is on the (mask_row, dedup_key) pair along the CURRENT
    expansion path AND the already-surfaced set, so an unmatched->unmatched
    cycle terminates at the cap or the revisit, whichever is first.
    """
    if max_unmatched_depth < 0:
        raise ValueError(
            f"max_unmatched_depth must be >= 0; got {max_unmatched_depth}"
        )

    out: List[Edge] = []
    # Per-row de-dup of surfaced matched edges: a row never emits the same
    # callee identity twice (the decider would once-only it anyway, but
    # de-duping here keeps the fed set minimal and the output deterministic
    # regardless of how many unmatched paths reach the same matched node).
    surfaced: set = set()

    def _emit_matched(edge: Edge) -> None:
        key = (edge.mask_row, edge.dedup_key)
        if key in surfaced:
            return
        surfaced.add(key)
        out.append(edge)

    def _recurse(edge: Edge, depth: int, path: frozenset) -> None:
        """Surface matched edges behind one unmatched ``edge``.

        ``path`` is the set of ``(mask_row, dedup_key)`` unmatched nodes on
        the current recursion branch -- the cycle guard. ``depth`` is the
        number of unmatched hops taken so far for this branch.
        """
        if depth >= max_unmatched_depth:
            return
        for child in resolve_children(edge):
            if child.is_matched:
                _emit_matched(child)
                continue
            ckey = (child.mask_row, child.dedup_key)
            if ckey in path:
                # Unmatched->unmatched cycle on this branch: stop.
                continue
            _recurse(child, depth + 1, path | {ckey})

    for edge in edges:
        if edge.is_matched:
            _emit_matched(edge)
        else:
            _recurse(edge, 0, frozenset({(edge.mask_row, edge.dedup_key)}))

    return out
