"""Spec-list boundary + resolution helpers for the collection layer.

Single concern: the pure ``IndexSpec``-list arithmetic the collection's
public surface needs -- normalising the ``specs`` XOR ``(reduction,
depth)`` boundary into one canonical list, deriving the stable display
order, and resolving the per-call ``spec=`` selector against the
configured specs. No I/O, no sampling, no filesystem awareness.

The collection binds N ``(reduction, depth)`` :class:`IndexSpec` pairs
(see :class:`.._collection.IndexedMemmapCollection`). Every place that
must turn a caller-facing spec selector into a concrete configured spec
routes through :func:`resolve_spec` so the single-spec back-compat rule
and the multi-spec ambiguity error live in exactly one place.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .._types import IndexSpec, LengthReduction


__all__ = [
    "spec_tag",
    "normalize_specs",
    "sorted_specs",
    "resolve_spec",
]


def spec_tag(spec: IndexSpec) -> str:
    """Canonical human/log tag for one spec: ``<mode>_d<NNN>``.

    Mirrors the ``.idx`` filename's ``_<mode>_d<NNN>`` infix so error
    messages and log records name a spec the same way its file is
    named on disk.
    """
    return f"{spec.reduction.filename_tag()}_d{spec.depth:03d}"


def normalize_specs(
    specs: Optional[Sequence[IndexSpec]],
    reduction: Optional[LengthReduction],
    depth: Optional[int],
) -> List[IndexSpec]:
    """Resolve the ``specs`` XOR ``(reduction, depth)`` boundary to a list.

    Exactly ONE of (``specs`` non-empty) or (``reduction`` AND ``depth``
    both given) is accepted; anything else raises :class:`ValueError`.
    The single-spec convenience form ``reduction=..., depth=...`` becomes
    ``[IndexSpec(reduction, depth)]`` so every caller past this boundary
    handles a uniform N-spec list. Duplicate specs raise.
    """
    specs_given = specs is not None and len(specs) > 0
    convenience_given = reduction is not None and depth is not None

    if specs_given and (reduction is not None or depth is not None):
        raise ValueError(
            "IndexedMemmapCollection.discover: pass EITHER specs=... OR "
            "reduction=... depth=..., not both",
        )
    if specs_given:
        resolved = list(specs)
    elif convenience_given:
        resolved = [IndexSpec(reduction=reduction, depth=depth)]
    else:
        raise ValueError(
            "IndexedMemmapCollection.discover: provide specs=[IndexSpec, "
            "...] or the single-spec convenience reduction=... depth=...",
        )

    seen: List[IndexSpec] = []
    for spec in resolved:
        if spec in seen:
            raise ValueError(
                "IndexedMemmapCollection.discover: duplicate spec "
                f"{spec_tag(spec)!r} in specs",
            )
        seen.append(spec)
    return seen


def sorted_specs(specs: Sequence[IndexSpec]) -> List[IndexSpec]:
    """Specs in stable display order: by ``(filename_tag, depth)``."""
    return sorted(specs, key=IndexSpec.sort_key)


def resolve_spec(
    spec: Optional[IndexSpec],
    configured: Sequence[IndexSpec],
) -> IndexSpec:
    """Resolve a per-call ``spec=`` selector against the configured specs.

    * ``spec is None`` with exactly one configured spec -> that spec
      (full back-compat for single-spec callers).
    * ``spec is None`` with several configured -> :class:`ValueError`
      naming the configured specs (the caller must disambiguate).
    * ``spec`` given but not configured -> :class:`ValueError`.
    """
    if spec is None:
        if len(configured) == 1:
            return configured[0]
        tags = ", ".join(spec_tag(s) for s in sorted_specs(configured))
        raise ValueError(
            "IndexedMemmapCollection: spec=None is ambiguous; this "
            f"collection is configured for several specs ({tags}); pass "
            "spec=IndexSpec(...) to select one",
        )
    if spec not in configured:
        tags = ", ".join(spec_tag(s) for s in sorted_specs(configured))
        raise ValueError(
            "IndexedMemmapCollection: spec "
            f"{spec_tag(spec)!r} is not configured for this collection "
            f"(configured: {tags})",
        )
    return spec
