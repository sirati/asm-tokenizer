"""Per-batch sizing helper -- derive ``(num_variants_per_section,
context_len)`` straight from the loader's own primitives.

Single concern of this module: given a session + a
``list[SectionPointerSpec]``, ask Stage 1's :func:`resolve_section_pointers`
to enumerate every variant of every requested section and reduce the
result to the smallest ``(num_variants_per_section, context_len)`` pair
that lets :func:`batch_decode` hold every variant's full token stream
without mid-cut.

What this module owns (the boundary):

* Input: a :class:`BinarySession` + the request's
  :class:`SectionPointerSpec` list.
* Output: a :class:`SizingSpec` -- ``num_variants_per_section`` =
  ``max(len(section.variants))`` across the request, ``context_len`` =
  longest-variant body + headroom.

What this module does NOT own:

* Variant sampling -- the resolver enumerates every variant under a
  cap-bypass bound (``selection_size == n_variants`` returns
  ``range(n_variants)`` without touching the rng -- see
  :func:`tokenizer.aligned_data.loader._session_helpers._select_variant_indices`).
* :func:`batch_decode` invocation -- callers pin the explicit ints we
  return on the pipeline entry; the entry signature stays int-only.

The helper exposes both a session-level public entry
(:func:`compute_auto_sizes`) and a pure-data reducer
(:func:`auto_size_from_resolved`) so callers that already hold the
resolved list -- e.g. Stage 1's wiring -- can skip the second
session round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np

from ._resolve_pointers import ResolvedSection, resolve_section_pointers
from ._types import SectionPointerSpec

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from ..session import BinarySession


__all__ = [
    "CONTEXT_LEN_HEADROOM",
    "SizingSpec",
    "auto_size_from_resolved",
    "compute_auto_sizes",
]


# Phase-1 headroom (in tokens) added on top of the longest variant's
# raw body length so :func:`batch_decode`'s ``context_len`` covers the
# per-row variant-axis prefix + per-call prepend slot + multi-chunk
# promotion expansion without mid-cut. A Phase-2 follow-up (plan
# decision #26 + plan section "compute_auto_sizes headroom
# correctness") replaces this constant with Stage 1's precise
# ``predicted_full_length`` -- summing per-call-target expansion under
# Stage 2's ``expand_tokens`` and adding the per-variant
# ``variant_tokens`` prefix. Until that lands, ``+64`` is the
# conservative safety budget every auto-sized caller agrees on.
# TODO(plan inspector-render-backends.md decision #26): switch to the
# Stage 1 precise predictor.
CONTEXT_LEN_HEADROOM: int = 64


@dataclass(frozen=True)
class SizingSpec:
    """Typed return for :func:`compute_auto_sizes`.

    Attributes:
        num_variants_per_section: The max ``len(section.variants)``
            across the resolved request -- the upper bound on the
            variant axis. Pass verbatim as
            :func:`batch_decode`'s ``num_variants_per_section``.
        context_len: The longest per-variant full-token-stream length
            (``len(fd.tokens) + fd.variant_tokens.shape[0]``) across
            every resolved variant, plus :data:`CONTEXT_LEN_HEADROOM`.
            Pass verbatim as :func:`batch_decode`'s ``context_len``.
    """

    num_variants_per_section: int
    context_len: int


def compute_auto_sizes(
    session: "BinarySession",
    section_pointers: List[SectionPointerSpec],
) -> SizingSpec:
    """Derive the smallest ``(num_variants_per_section, context_len)``
    pair that lets :func:`batch_decode` hold every variant of every
    pointer in ``section_pointers`` without mid-cut.

    Walks the same per-arm load helpers as Stage 1 by delegating to
    :func:`resolve_section_pointers`; the cap-bypass bound
    (``num_variants_per_section`` >= every section's variant count)
    makes :func:`_select_variant_indices` return ``range(n_variants)``
    without touching the rng, so every variant lands on
    :attr:`ResolvedSection.function_data_per_sampled_variant` and the
    rng we pass through is never consumed.

    Args:
        session: The :class:`BinarySession` whose per-arm loaders supply
            the sections + function bodies.
        section_pointers: The pointer list the caller intends to pass
            to :func:`batch_decode`. The same list MUST be passed to
            both calls so the size we return covers the same input.

    Returns:
        A :class:`SizingSpec` whose two ints feed
        :func:`batch_decode` verbatim.

    Raises:
        ValueError: Bubbles up from :func:`resolve_section_pointers`
            on an unknown :class:`SectionKind` member or empty input.
    """
    # ``np.iinfo(np.int64).max`` exceeds every plausible section's
    # variant count, so ``_select_variant_indices`` clamps to the
    # section's own ``n_variants`` and returns the trivial
    # ``range(n_variants)`` -- no rng draws happen, the rng we pass
    # through is just an API stub.
    cap_bypass = int(np.iinfo(np.int64).max)
    resolved = resolve_section_pointers(
        session,
        section_pointers,
        num_variants_per_section=cap_bypass,
        rng=np.random.default_rng(0),
    )
    return auto_size_from_resolved(resolved)


def auto_size_from_resolved(
    resolved: List[ResolvedSection],
) -> SizingSpec:
    """Reduce a resolved-section list to a :class:`SizingSpec`.

    Pure-data helper -- no session, no IO. Used by
    :func:`compute_auto_sizes` after a round-trip through
    :func:`resolve_section_pointers`, but also reusable by any caller
    that already holds the resolved list (e.g. Stage 1's wiring).

    Args:
        resolved: One :class:`ResolvedSection` per section pointer; the
            ``section.variants`` list supplies the variant-count axis
            and ``function_data_per_sampled_variant`` supplies the
            per-variant token-length axis.

    Returns:
        A :class:`SizingSpec`. With an empty ``resolved`` both fields
        are ``0`` + :data:`CONTEXT_LEN_HEADROOM` respectively -- the
        caller is responsible for treating an empty pointer list as
        a no-op upstream.
    """
    num_variants_per_section = 0
    longest_variant_tokens = 0
    for rs in resolved:
        num_variants_per_section = max(
            num_variants_per_section, len(rs.section.variants)
        )
        for fd in rs.function_data_per_sampled_variant:
            full_len = int(len(fd.tokens)) + int(fd.variant_tokens.shape[0])
            if full_len > longest_variant_tokens:
                longest_variant_tokens = full_len
    return SizingSpec(
        num_variants_per_section=num_variants_per_section,
        context_len=longest_variant_tokens + CONTEXT_LEN_HEADROOM,
    )
