"""Filter configuration + pure filter pass + result sum-type.

Single concern: pure-data types for the variant filter surface (the
"f" hotkey) + the function that drops disabled-value variants from a
sibling set. NO Textual imports, NO discovery logic, NO dialog widget
code -- those live in sibling modules of this subpackage.

The filter is an axis-keyed Mapping ``axis -> frozenset[str]`` carrying
the set of axis values the user has DISABLED for that axis. A value
present in the disabled set hides every variant whose axis evaluates to
that value. Multi-axis filters intersect (a variant is hidden iff ANY
axis-value is disabled).

:class:`FilterConfig` equality is value-based (frozen dataclass +
``Mapping[AxisDescriptor, frozenset[str]]``) -- two configs built from
the same descriptors + the same disabled sets compare equal so the
dispatcher can short-circuit "user re-accepted the same filter" with
the same idiom :mod:`._order` uses for :class:`OrderConfig`.

The empty / no-filter form is :func:`FilterConfig.empty` which compares
equal to ``None`` semantically (no axis disabled) but stays a typed
value the apply-pass can short-circuit on without ``is None`` branching
in callers that thread the config through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence, Union

from tokenizer.inspector._render._protocol import RenderedVariant
from tokenizer.inspector._tree_model import VariantNode

from .._order import AxisDescriptor, extract_axis_value


__all__ = [
    "FilterConfig",
    "FilterAccepted",
    "FilterCancelled",
    "FilterResult",
    "apply_filter",
    "missing_value_token",
    "MISSING_VALUE_TOKEN",
]


# ---------------------------------------------------------------------------
# Missing-value token: the synthetic axis value used for "axis missing on
# this variant" in both the dialog row + the disabled-set semantics.
# Mirrors :mod:`._order._grouping`'s ``_MISSING_VALUE_LABEL`` so the same
# token is shown in the dialog and stored in the disabled set when the
# user filters out missing-value variants.
# ---------------------------------------------------------------------------

MISSING_VALUE_TOKEN = "?"


def missing_value_token() -> str:
    """The synthetic axis value the filter uses for missing-axis variants.

    Pure accessor (so callers don't reach into the module constant).
    """
    return MISSING_VALUE_TOKEN


# ---------------------------------------------------------------------------
# Result sum type (mirrors :mod:`._order._axes` for dispatcher symmetry)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterAccepted:
    """Modal dismiss with a fresh :class:`FilterConfig`."""

    config: "FilterConfig"


@dataclass(frozen=True)
class FilterCancelled:
    """Modal dismiss with no change (Esc)."""


FilterResult = Union[FilterAccepted, FilterCancelled]
"""Sum type returned via :meth:`textual.screen.ModalScreen.dismiss`.

Pattern-matched by type at the dispatcher callback (same idiom as
:class:`OrderResult`)."""


# ---------------------------------------------------------------------------
# FilterConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterConfig:
    """User-disabled axis-value sets.

    ``disabled`` maps each :class:`AxisDescriptor` whose dialog row had
    at least one value UNCHECKED to the frozenset of those unchecked
    values. Axes with every value checked (the "show all" baseline)
    are absent from the mapping -- the apply pass only iterates over
    the present keys.

    The dataclass is frozen + the mapping is captured as an
    ``immutables``-style :class:`tuple` of items (sorted by axis label
    for stable equality + hashing). Two configs built from the same
    descriptors + disabled-set contents compare equal.
    """

    # Tuple of ``(axis, frozenset[value])`` pairs, sorted by ``axis.label``.
    # Storing a tuple instead of a Mapping is what lets the dataclass be
    # frozen+hashable (Mapping is not hashable).
    _items: tuple[tuple[AxisDescriptor, frozenset[str]], ...] = field(default=())

    @classmethod
    def empty(cls) -> "FilterConfig":
        """The no-axis-disabled baseline.

        Equal-by-value to any other :func:`FilterConfig.empty` result; the
        dispatcher short-circuits the rebuild when the new config compares
        equal to the active one (same pattern :class:`OrderConfig` uses).
        """
        return cls(_items=())

    @classmethod
    def build(
        cls,
        disabled: Mapping[AxisDescriptor, Iterable[str]],
    ) -> "FilterConfig":
        """Construct from an ``axis -> disabled-values`` mapping.

        Empty disabled sets are silently dropped from the canonical
        form so equality with :func:`empty` holds when no axis actually
        disables anything. Sorted by ``axis.label`` for deterministic
        equality.
        """
        items: list[tuple[AxisDescriptor, frozenset[str]]] = []
        for axis, values in disabled.items():
            frozen = frozenset(values)
            if not frozen:
                continue
            items.append((axis, frozen))
        items.sort(key=lambda pair: pair[0].label)
        return cls(_items=tuple(items))

    def is_empty(self) -> bool:
        """No axis disables anything -- the filter pass is a passthrough."""
        return not self._items

    @property
    def disabled(self) -> Mapping[AxisDescriptor, frozenset[str]]:
        """Public read-only view of the disabled-by-axis mapping."""
        return dict(self._items)

    def disabled_for(self, axis: AxisDescriptor) -> frozenset[str]:
        """Frozenset of disabled values for ``axis`` (empty when absent)."""
        for ax, vals in self._items:
            if ax == axis:
                return vals
        return frozenset()

    def axes(self) -> tuple[AxisDescriptor, ...]:
        """Tuple of axes that disable at least one value."""
        return tuple(ax for ax, _ in self._items)


# ---------------------------------------------------------------------------
# Pure filter pass
# ---------------------------------------------------------------------------


def apply_filter(
    variants: Sequence[VariantNode],
    rendered_by_variant: Mapping[int, RenderedVariant],
    config: Optional[FilterConfig],
) -> list[VariantNode]:
    """Drop variants whose axis values are disabled by ``config``.

    Pure function: same input + same config = same output. Empty /
    ``None`` config short-circuits to ``list(variants)`` so callers
    that always thread the config can keep one code path.

    A variant is KEPT iff every axis in ``config._items`` evaluates to
    a value NOT in that axis's disabled set. Missing axis values
    (``extract_axis_value`` returned ``None``) compare against
    :data:`MISSING_VALUE_TOKEN` so a user who explicitly unchecks the
    ``"?"`` bucket also hides variants with no value on that axis.
    """
    if config is None or config.is_empty():
        return list(variants)

    out: list[VariantNode] = []
    for v in variants:
        rv = rendered_by_variant.get(v.variant_idx)
        if rv is None:
            # No rendered data side-table available -- defensive
            # passthrough so a stale variant doesn't vanish silently.
            out.append(v)
            continue
        if _variant_passes(rv, config):
            out.append(v)
    return out


def _variant_passes(rv: RenderedVariant, config: FilterConfig) -> bool:
    """Does the variant survive every axis-disabled check in ``config``?"""
    for axis, disabled_values in config._items:
        raw = extract_axis_value(axis, rv)
        value = MISSING_VALUE_TOKEN if raw is None else str(raw)
        if value in disabled_values:
            return False
    return True
