"""Per-section variant-index selection strategy (the null-object seam).

Single concern of this module: given a section's variant count plus an
RNG, produce the ORDERED variant-index list the resolver loads. Nothing
here knows about sessions, decode, sampling pools, or batches -- it is the
narrow strategy boundary :func:`resolve_section_pointers` resolves once
per section pointer.

The boundary is a :class:`VariantSelection` protocol with exactly one
method, :meth:`VariantSelection.select`. Two implementations satisfy it:

* :class:`CountThenRNGSelection` -- the historical count-then-RNG rule.
  It delegates VERBATIM to
  :func:`.._session_helpers._select_variant_indices` (the single source
  of truth for the without-replacement sampling rule); it does NOT
  re-implement that rule.
* :class:`ExplicitIndicesSelection` -- a caller-pinned deterministic
  index list (the validation path). The RNG argument is intentionally
  ignored; the indices are returned as-is after a bounds check.

The resolver collapses the two via the null-object default::

    selection = pointer.variant_selection or CountThenRNGSelection(...)
    sampled = selection.select(n_variants=..., rng=...)

so there is NO ``if explicit: ... else: count ...`` branch and neither
path acquires knowledge of the other. The count path
(``variant_selection is None``) hits :class:`CountThenRNGSelection` with
the identical ``(n_variants, max_variants, rng)`` arguments the resolver
passed before, so it stays byte-identical to the pre-seam behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from .._session_helpers import _select_variant_indices


__all__ = [
    "VariantSelection",
    "CountThenRNGSelection",
    "ExplicitIndicesSelection",
]


@runtime_checkable
class VariantSelection(Protocol):
    """Strategy: section variant-count + RNG -> ordered variant indices.

    The single method every selection implements. Returns an ``int64``
    ndarray of ordered variant indices whose length is ``<= n_variants``
    and whose every element is a valid index into the section's variant
    list (``0 <= idx < n_variants``).
    """

    def select(
        self, *, n_variants: int, rng: Optional[np.random.Generator]
    ) -> np.ndarray:
        """Return the ordered variant indices for one section."""
        ...


@dataclass(frozen=True)
class CountThenRNGSelection:
    """The historical count-then-RNG variant selection.

    ``select`` delegates VERBATIM to
    :func:`.._session_helpers._select_variant_indices` with
    ``max_variants=self.max_variants`` -- the single source of truth for
    the ``min(max_variants, n_variants)`` clamp + sort-on-subset rule.
    This is the null-object DEFAULT the resolver substitutes when a
    pointer carries no explicit selection, so the count path stays
    byte-identical to the pre-seam call.
    """

    max_variants: int

    def select(
        self, *, n_variants: int, rng: Optional[np.random.Generator]
    ) -> np.ndarray:
        return _select_variant_indices(
            n_variants=n_variants, max_variants=self.max_variants, rng=rng
        )


@dataclass(frozen=True)
class ExplicitIndicesSelection:
    """A caller-pinned, deterministic variant-index list.

    The validation path's selection: the indices are decided upstream
    (the deterministic shuffle+chunk kernel) and ride inside the section
    pointer, so ``select`` returns them as-is. The ``rng`` argument is
    intentionally UNUSED -- this selection is fully deterministic and
    must not consume the shared RNG stream.

    ``select`` raises :class:`ValueError` when any pinned index falls
    outside ``[0, n_variants)``; an out-of-band index is the stale-sidecar
    symptom (an index list built against a different section variant count)
    and is surfaced loudly rather than silently mis-loading a body.
    """

    indices: Tuple[int, ...]

    def select(
        self, *, n_variants: int, rng: Optional[np.random.Generator]
    ) -> np.ndarray:
        arr = np.asarray(self.indices, dtype=np.int64)
        if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= n_variants):
            raise ValueError(
                "ExplicitIndicesSelection index out of range for a section "
                f"with {n_variants} variants: {tuple(self.indices)!r} "
                "(stale-sidecar guard)"
            )
        return arr
