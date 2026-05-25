"""Axis model + canonical-5 builder + result sum-type for the inspector's
Order modal.

Single concern: pure data types + module-level value extraction for the
variant-axis sort/group surface. NO Textual imports, NO grouping logic,
NO dialog widget code -- those live in sibling modules.

The four pieces this module owns:

* :class:`AxisKind` enum -- the discriminator for the three axis
  flavours (positional axes, BITWIDTH derived from arch, EXTRA_META
  per-binary keys).
* :class:`AxisDescriptor` frozen dataclass -- pure data (kind + key +
  label). NO Callable closure (cluster #10 W4-AMENDED): closure
  identity would break :class:`OrderConfig` equality on re-press.
  Value extraction goes through the module-level
  :func:`extract_axis_value` dispatcher.
* :class:`OrderConfig` frozen dataclass -- the accepted ordering +
  per-axis grouping toggle. Two configs constructed from the same
  axes (separate factory calls) compare equal (pure data + sets).
* :class:`OrderResult` typed sum (:class:`OrderAccepted` /
  :class:`OrderCancelled`) -- the modal-dismiss payload. Pattern-
  matched by type at the dispatcher callback (cluster #14, B-L1 H4).

Canonical-5 default axes (from
:data:`tokenizer.variant_tokens.prefixes.POSITIONAL_PREFIXES` plus
BITWIDTH): arch / 32_64 / compiler / version / opt. The BITWIDTH axis
is derived from ``arch`` via
:data:`tokenizer.arch.bitwidth.PLATFORM_BITWIDTH`; unknown arch
collapses to ``None`` (renders as the ``"?"`` bucket).

Extra-meta discovery is lazy across the currently-loaded
:class:`RenderedVariant`s (W3-20): callers pass the variants they have
seen so far + this module returns the sorted, deduplicated key list.
The dialog seeds one EXTRA_META :class:`AxisDescriptor` per key on
each open.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Tuple, Union

from tokenizer.arch.bitwidth import PLATFORM_BITWIDTH
from tokenizer.arch_translation import arch_to_platform
from tokenizer.inspector._render._protocol import RenderedVariant
from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
)


__all__ = [
    "AxisKind",
    "AxisDescriptor",
    "OrderConfig",
    "OrderAccepted",
    "OrderCancelled",
    "OrderResult",
    "extract_axis_value",
    "build_canonical_axes",
    "build_extra_meta_axis",
    "discover_extra_meta_keys",
    "BITWIDTH_AXIS_KEY",
]


# ---------------------------------------------------------------------------
# Discriminator + descriptor + config
# ---------------------------------------------------------------------------


class AxisKind(Enum):
    """Three axis flavours the Order modal exposes.

    :attr:`POSITIONAL`: one of the canonical-4 positional axes
    (``arch:`` / ``comp:`` / ``cver:`` / ``opt:``); ``key`` matches the
    prefix declared in
    :data:`tokenizer.variant_tokens.prefixes.POSITIONAL_PREFIXES`, value
    read off :attr:`RenderedVariant.label_axes`.

    :attr:`BITWIDTH`: derived "32 / 64" axis. Value read from
    :data:`tokenizer.arch.bitwidth.PLATFORM_BITWIDTH` keyed by
    ``arch_to_platform(label_axes[ARCH_PREFIX])``; unknown arch yields
    ``None``.

    :attr:`EXTRA_META`: one per per-binary metadata residue key
    (sidecar fields, build-flag groups, hardening / sanitizer
    settings, ...); ``key`` matches the dict key in
    :attr:`RenderedVariant.extra_metadata`.
    """

    POSITIONAL = "positional"
    BITWIDTH = "bitwidth"
    EXTRA_META = "extra_meta"


# Sentinel "key" for the single BITWIDTH axis. The axis is not a
# positional prefix nor a metadata residue key; pinning the synthetic
# key as a module-level constant keeps it out of the
# stringly-typed-discriminator anti-pattern.
BITWIDTH_AXIS_KEY = "__bitwidth__"


@dataclass(frozen=True)
class AxisDescriptor:
    """Pure-data axis descriptor.

    Three fields: the typed :class:`AxisKind`, the descriptor key
    (positional prefix / :data:`BITWIDTH_AXIS_KEY` / metadata key), and
    the human-readable label for the dialog row + grouping header.

    Two descriptors built from the same ``(kind, key, label)`` triple
    compare equal -- the dataclass is the closed identity, no
    Callable closure (cluster #10 W4-AMENDED). Value extraction is
    routed through the module-level :func:`extract_axis_value`
    dispatcher.
    """

    kind: AxisKind
    key: str
    label: str


@dataclass(frozen=True)
class OrderConfig:
    """The accepted ordering + per-axis grouping toggle.

    ``ordered_axes`` is the user-arranged tuple of descriptors (top-
    to-bottom in the dialog). ``grouping_axes`` is the subset of
    ``ordered_axes`` the user checked for grouping; element identity is
    the :class:`AxisDescriptor`'s pure-data value equality (two
    instances built from the same triple compare equal -- the
    short-circuit on first re-press works).

    Equality is value-based (frozenset + tuple of frozen dataclasses)
    -- the dispatcher uses this to avoid rebuilding the tree when the
    user re-opens the dialog + accepts the same config.
    """

    ordered_axes: Tuple[AxisDescriptor, ...]
    grouping_axes: frozenset[AxisDescriptor]


# ---------------------------------------------------------------------------
# Order result sum type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderAccepted:
    """Modal dismiss with a fresh :class:`OrderConfig`."""

    config: OrderConfig


@dataclass(frozen=True)
class OrderCancelled:
    """Modal dismiss with no change (Esc)."""


OrderResult = Union[OrderAccepted, OrderCancelled]
"""Sum type returned via :meth:`textual.screen.ModalScreen.dismiss`.

Pattern-matched by type at the dispatcher callback (cluster #14, B-L1
H4); using ``Optional[OrderConfig]`` would conflate "user cancelled"
with "user accepted an empty config".
"""


# ---------------------------------------------------------------------------
# Canonical-5 axis builders + extra-meta discovery
# ---------------------------------------------------------------------------


# Per-positional-axis human label (no trailing colon). Single source of
# truth for the dialog row + grouping header; the
# :data:`POSITIONAL_PREFIXES` order is what's authoritative across the
# codebase, so an axis addition there must be matched here.
_POSITIONAL_AXIS_LABELS: Mapping[str, str] = {
    ARCH_PREFIX: "arch",
    COMP_PREFIX: "compiler",
    CVER_PREFIX: "version",
    OPT_PREFIX: "opt",
}

# Canonical-5 default ordering: arch, 32_64, compiler, version, opt.
# arch comes first so BITWIDTH (derived from arch) sits adjacent.
_CANONICAL_ORDER: Tuple[Tuple[AxisKind, str, str], ...] = (
    (AxisKind.POSITIONAL, ARCH_PREFIX, _POSITIONAL_AXIS_LABELS[ARCH_PREFIX]),
    (AxisKind.BITWIDTH, BITWIDTH_AXIS_KEY, "32/64"),
    (AxisKind.POSITIONAL, COMP_PREFIX, _POSITIONAL_AXIS_LABELS[COMP_PREFIX]),
    (AxisKind.POSITIONAL, CVER_PREFIX, _POSITIONAL_AXIS_LABELS[CVER_PREFIX]),
    (AxisKind.POSITIONAL, OPT_PREFIX, _POSITIONAL_AXIS_LABELS[OPT_PREFIX]),
)


def build_canonical_axes() -> Tuple[AxisDescriptor, ...]:
    """Construct the canonical-5 default axis tuple.

    Pure data; idempotent. Two calls produce equal tuples (the frozen
    descriptors compare by value). Callers prepend / append the
    discovered EXTRA_META axes to extend the dialog row set.
    """
    return tuple(
        AxisDescriptor(kind=kind, key=key, label=label)
        for kind, key, label in _CANONICAL_ORDER
    )


def build_extra_meta_axis(key: str, label: Optional[str] = None) -> AxisDescriptor:
    """One EXTRA_META :class:`AxisDescriptor` for a given metadata key.

    ``label`` defaults to ``key`` (raw metadata key shown in the
    dialog); callers override when the key has a known prettier name.
    """
    return AxisDescriptor(
        kind=AxisKind.EXTRA_META, key=key, label=key if label is None else label
    )


def discover_extra_meta_keys(variants: Iterable[RenderedVariant]) -> Tuple[str, ...]:
    """Sorted, deduplicated EXTRA_META key list across given variants.

    Discovery is lazy across the currently-loaded variants (W3-20). A
    new function expansion can surface additional EXTRA_META keys on
    the next ``o``-press; this returns whatever is visible right now.

    Accepts any iterable whose elements expose ``.extra_metadata`` as
    a string mapping -- the inspector calls this with the active
    backend's :class:`RenderedVariant` sequence.
    """
    seen: set[str] = set()
    for rv in variants:
        for k in rv.extra_metadata:
            seen.add(k)
    return tuple(sorted(seen))


# ---------------------------------------------------------------------------
# Module-level value extraction (no Callable closure -- cluster #10)
# ---------------------------------------------------------------------------


def extract_axis_value(
    axis: AxisDescriptor, rv: RenderedVariant
) -> Optional[str]:
    """Dispatch axis-value extraction off :attr:`AxisDescriptor.kind`.

    Returns ``None`` for missing values (positional axis absent,
    BITWIDTH for unknown arch, EXTRA_META key absent) -- the grouping
    pass renders ``None`` as the ``"?"`` bucket.

    Pure function; no caching (cheap dict lookups). Re-entrant.
    """
    if axis.kind is AxisKind.POSITIONAL:
        return _coerce_axis_value(rv.label_axes.get(axis.key))
    if axis.kind is AxisKind.BITWIDTH:
        arch = rv.label_axes.get(ARCH_PREFIX)
        if arch is None:
            return None
        try:
            platform = arch_to_platform(str(arch))
        except ValueError:
            return None
        return PLATFORM_BITWIDTH.get(platform)
    if axis.kind is AxisKind.EXTRA_META:
        value = rv.extra_metadata.get(axis.key)
        if value is None or value == "":
            return None
        return str(value)
    # AxisKind is closed; any miss is a contract drift.
    raise AssertionError(f"unhandled AxisKind: {axis.kind!r}")


def _coerce_axis_value(value: object) -> Optional[str]:
    """``None`` passes through; everything else stringifies once."""
    if value is None:
        return None
    return str(value)
