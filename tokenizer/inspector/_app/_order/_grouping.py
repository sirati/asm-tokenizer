"""Variant grouping pass + ``VariantGroupNode`` dataclass.

Single concern: take a flat list of :class:`VariantNode`s + an
:class:`OrderConfig` and produce a recursively-partitioned tree of
:class:`VariantGroupNode` (one wrapper per grouping-axis level) with
:class:`VariantNode` leaves at the bottom. The pass is a function at
the ``_app/_order/`` boundary -- :class:`FunctionNode.expand` stays
UNCHANGED (cluster #6 W4-AMENDED); the dispatcher in
:mod:`tokenizer.inspector._app._application` runs the pass after
expand returns.

Natural-alphanumeric sort is delegated to
:func:`natsort.natsort_keygen` (cluster #15 W4-AMENDED) -- the
``"clang10"`` vs ``"clang9"`` ordering people expect when scanning
variants drops out for free. The ``None``-valued "?" bucket is forced
to the end of each sibling group by prefixing its sort key with a
character that natsort sorts after every printable string.

:class:`VariantGroupNode` lives HERE, NOT in
:mod:`tokenizer.inspector._tree_model` (cluster #6): the model tree
deliberately stays grouping-agnostic so a future reordering / regroup
never has to touch :class:`FunctionNode.expand`'s contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Sequence, Union

import natsort

from tokenizer.arch.family_display import PLATFORM_FAMILY_DISPLAY
from tokenizer.arch_translation import arch_to_platform
from tokenizer.inspector._render._protocol import RenderedVariant
from tokenizer.inspector._tree_model import VariantNode

from tokenizer.variant_tokens.prefixes import ARCH_PREFIX, POSITIONAL_PREFIXES

from ._axes import (
    AxisDescriptor,
    AxisKind,
    BITWIDTH_AXIS_KEY,
    OrderConfig,
    extract_axis_value,
)


__all__ = [
    "VariantGroupNode",
    "format_grouping_label",
    "group_variants",
    "sort_variants_flat",
]


# Force-last bucket label for missing axis values. The renderer emits
# this as the visible bucket name; the sort path uses the prefix
# constant below so the bucket sinks to the bottom of every sibling
# group regardless of the natsort table.
_MISSING_VALUE_LABEL = "?"

# Sort-key prefix that ranks AFTER every printable string in natsort.
# ``￿`` is the highest BMP code point (just below the surrogate
# range); used as the first natsort token for the ``None``-bucket
# group so missing-value buckets sink to the bottom.
_FORCE_LAST_PREFIX = "￿"

# Pre-built natsort key generator: caches the regex compilation once
# per process so the per-call cost is just dict lookups.
_natural_sort_key = natsort.natsort_keygen()


# ---------------------------------------------------------------------------
# Group node dataclass
# ---------------------------------------------------------------------------


@dataclass
class VariantGroupNode:
    """One grouping wrapper produced by the :func:`group_variants` pass.

    Carries the typed :class:`AxisDescriptor` directly (no duplicate
    ``axis_label``/``axis_key`` fields; the descriptor IS the source of
    truth -- cluster #6 W4-AMENDED). ``axis_value`` is the common axis
    value all children share (already ``str`` -- missing values
    stringified to :data:`_MISSING_VALUE_LABEL`). ``children`` is the
    nested :class:`VariantGroupNode` (further-grouped) or
    :class:`VariantNode` (leaf) list.

    ``rendered_by_variant`` carries the parent dispatcher's
    ``variant_idx -> RenderedVariant`` lookup so an auto-expand walk
    after a regroup can read EXTRA_META values WITHOUT re-fetching the
    backend's variants (cluster #23 / B-L5 H3). Mapped once at the App
    boundary + threaded through the entire group tree by reference.

    ``can_expand`` is True by construction (a wrapper always shows its
    children); ``is_failed`` always False (the wrapper itself never
    decodes anything). ``remembered_scroll_x`` enables the same per-row
    horizontal scroll memory the leaf nodes carry.
    """

    axis: AxisDescriptor
    axis_value: str
    children: List[Union["VariantGroupNode", VariantNode]]
    rendered_by_variant: Mapping[int, RenderedVariant]
    is_failed: bool = False
    can_expand: bool = field(default=True, init=False)
    remembered_scroll_x: int = field(default=0, init=False)

    def expand(self) -> List[Any]:
        """Surface the pre-grouped child list verbatim.

        The grouping pass already built the tree; expand is a noop
        traversal that hands the children to the UI dispatcher (same
        contract every other tree-model node exposes).
        """
        return list(self.children)


def format_grouping_label(axis: AxisDescriptor, value: str) -> str:
    """Default ``"<axis.label>: <value>"`` format for a group row.

    Single helper next to the dataclass (W3-8 W4-AMENDED) so per-kind
    overrides land via a ``match axis.kind`` here on the day a UX
    iteration needs them.
    """
    return f"{axis.label}: {value}"


# ---------------------------------------------------------------------------
# Grouping pass
# ---------------------------------------------------------------------------


def sort_variants_flat(
    variants: Sequence[VariantNode],
) -> List[VariantNode]:
    """Sort variants in-canonical-axes-order using the same natsort key
    that :func:`group_variants` uses, but without any grouping pass.

    Used at the App boundary when no :class:`OrderConfig` is active so
    the user-visible variant siblings still come out naturally sorted
    (``v9`` before ``v10``; ``arm32`` before ``arm64``; ``x64`` before
    ``x86``; ...). Sole source of truth for "default natural variant
    sort" — :mod:`tokenizer.inspector._label` no longer re-implements
    this with a hand-rolled regex (cluster M-2 / M1 audit findings).

    Reads :attr:`VariantNode.label_axes` directly (no need for the
    full :class:`RenderedVariant` side-table since :data:`POSITIONAL_PREFIXES`
    fully determines the default sort).
    """
    if not variants:
        return []

    def _key(v: VariantNode) -> tuple:
        return tuple(
            _natsort_key_with_missing_last(v.label_axes.get(prefix))
            for prefix in POSITIONAL_PREFIXES
        )

    return sorted(variants, key=_key)


def group_variants(
    variants: Sequence[VariantNode],
    rendered_by_variant: Mapping[int, RenderedVariant],
    config: OrderConfig,
) -> List[Union[VariantGroupNode, VariantNode]]:
    """Apply the :class:`OrderConfig`'s grouping + sort to ``variants``.

    Returns either a flat list of :class:`VariantNode`s (when no
    grouping axes are configured -- only sort applies) OR a nested
    :class:`VariantGroupNode` tree (one level per grouping axis, in the
    ``OrderConfig.ordered_axes`` order). The pass is pure: same input
    + same config = byte-identical output.

    ``rendered_by_variant`` maps ``variant_idx -> RenderedVariant`` for
    every variant in ``variants``; the pass keys axis-value extraction
    on this Mapping (the model :class:`VariantNode` carries only
    ``label_axes`` not the full :class:`RenderedVariant`, so EXTRA_META
    extraction needs the side-table).

    Algorithm:
    1. Compute each variant's sort key by chaining
       :func:`extract_axis_value` over ``config.ordered_axes`` (NOT just
       the grouping subset -- the order axes drive intra-group sort).
    2. Partition the sorted variants by the grouping subset, in
       ``ordered_axes`` order; non-grouping axes contribute to sort
       only.
    """
    if not variants:
        return []

    grouping_in_order = [
        axis for axis in config.ordered_axes if axis in config.grouping_axes
    ]

    # Cross-axis interaction context: when BOTH the arch axis AND the
    # bitwidth axis are configured for grouping, the arch axis must
    # surface the family-display name (``arm`` / ``mips`` / ``x86`` /
    # ...) instead of the raw bitness-bearing arch (``arm32``/``arm64``
    # /...) so the two-level group tree reads as
    # ``arch: arm`` -> ``32/64: {32,64}`` instead of duplicating the
    # bitness information at both levels. The bitwidth axis still
    # surfaces ``32`` / ``64`` unchanged. Computed once per
    # ``group_variants`` call (the answer is config-level, not
    # variant-level).
    collapse_arch_to_family = _arch_family_collapse_active(config)

    # Pre-compute the full sort tuple per variant. The tuple covers
    # every ordered axis; missing values sink via the force-last prefix.
    decorated = [
        (
            _variant_sort_tuple(
                v, rendered_by_variant, config.ordered_axes, collapse_arch_to_family
            ),
            v,
        )
        for v in variants
    ]
    decorated.sort(key=lambda item: item[0])
    sorted_variants = [v for _, v in decorated]

    if not grouping_in_order:
        return list(sorted_variants)

    return _partition_recursive(
        sorted_variants,
        rendered_by_variant,
        grouping_in_order,
        collapse_arch_to_family,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _arch_family_collapse_active(config: OrderConfig) -> bool:
    """``True`` when BOTH the arch positional axis AND the bitwidth
    axis are in :attr:`OrderConfig.grouping_axes`.

    Sole gate for the family-display collapse applied by
    :func:`_axis_value_for_grouping`. Pure-config decision; no
    per-variant input.
    """
    has_arch = any(
        axis.kind is AxisKind.POSITIONAL and axis.key == ARCH_PREFIX
        for axis in config.grouping_axes
    )
    has_bitwidth = any(
        axis.kind is AxisKind.BITWIDTH and axis.key == BITWIDTH_AXIS_KEY
        for axis in config.grouping_axes
    )
    return has_arch and has_bitwidth


def _axis_value_for_grouping(
    axis: AxisDescriptor,
    rv: RenderedVariant,
    collapse_arch_to_family: bool,
) -> str | None:
    """Same as :func:`extract_axis_value` plus the arch-to-family
    collapse when ``collapse_arch_to_family`` is on.

    Single chokepoint for the cross-axis interaction so both the sort
    tuple and the partition pass see identical values (without this,
    the partition walk's consecutive-same-value chunking would split
    a single family into multiple bucket runs whenever raw-arch sort
    order disagrees with family contiguity — e.g. ``i686`` would
    sort before ``arm32`` and break the ``x86`` family bucket).
    """
    raw = extract_axis_value(axis, rv)
    if (
        not collapse_arch_to_family
        or raw is None
        or axis.kind is not AxisKind.POSITIONAL
        or axis.key != ARCH_PREFIX
    ):
        return raw
    try:
        platform = arch_to_platform(raw)
    except ValueError:
        return raw
    return PLATFORM_FAMILY_DISPLAY.get(platform, raw)


def _variant_sort_tuple(
    variant: VariantNode,
    rendered_by_variant: Mapping[int, RenderedVariant],
    ordered_axes: Sequence[AxisDescriptor],
    collapse_arch_to_family: bool,
) -> tuple:
    """Tuple of natsort keys, one entry per axis (in ``ordered_axes``).

    Missing values get the force-last prefix so they sink to the
    bottom of every sibling group.
    """
    rv = rendered_by_variant.get(variant.variant_idx)
    parts: list[Any] = []
    for axis in ordered_axes:
        value = (
            None
            if rv is None
            else _axis_value_for_grouping(axis, rv, collapse_arch_to_family)
        )
        parts.append(_natsort_key_with_missing_last(value))
    return tuple(parts)


def _natsort_key_with_missing_last(value: object) -> Any:
    """``None`` -> force-last natsort key; everything else natsort
    keys as-is."""
    if value is None:
        return _natural_sort_key(_FORCE_LAST_PREFIX)
    return _natural_sort_key(str(value))


def _partition_recursive(
    sorted_variants: Sequence[VariantNode],
    rendered_by_variant: Mapping[int, RenderedVariant],
    remaining_grouping: Sequence[AxisDescriptor],
    collapse_arch_to_family: bool,
) -> List[Union[VariantGroupNode, VariantNode]]:
    """Recursive partition by ``remaining_grouping[0]``, descend on the rest.

    ``sorted_variants`` is already in the final sort order (per
    :func:`group_variants`); this pass walks once + chunks consecutive
    same-value runs into one group each.
    """
    if not remaining_grouping:
        # Bottom of the recursion: variants surface as-is at this
        # level. Cast to the wider variant-or-group union type for the
        # caller's consumption.
        return list(sorted_variants)

    head = remaining_grouping[0]
    rest = remaining_grouping[1:]

    groups: list[VariantGroupNode] = []
    current_bucket_label: str | None = None
    current_children: list[VariantNode] = []

    def _flush() -> None:
        nonlocal current_children, current_bucket_label
        if not current_children or current_bucket_label is None:
            return
        nested = _partition_recursive(
            current_children, rendered_by_variant, rest, collapse_arch_to_family
        )
        groups.append(
            VariantGroupNode(
                axis=head,
                axis_value=current_bucket_label,
                children=list(nested),
                rendered_by_variant=rendered_by_variant,
            )
        )
        current_children = []
        current_bucket_label = None

    for variant in sorted_variants:
        rv = rendered_by_variant.get(variant.variant_idx)
        raw_value = (
            None
            if rv is None
            else _axis_value_for_grouping(head, rv, collapse_arch_to_family)
        )
        label = _MISSING_VALUE_LABEL if raw_value is None else str(raw_value)
        if label != current_bucket_label:
            _flush()
            current_bucket_label = label
        current_children.append(variant)

    _flush()
    return list(groups)
