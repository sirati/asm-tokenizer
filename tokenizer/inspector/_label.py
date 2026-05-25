"""Pure label-assembly helpers for the inspector's tree rows.

Single concern: turn already-in-memory data (FID integers, sidecar
``line_to_name`` map, :class:`FunctionData.metadata`, block objects)
into the strings shown in TUI tree rows. NO Textual imports, NO
filesystem I/O, NO data fetching — every input is supplied by the
caller (the tree-model or render layer).

Row layouts (mirrors plan ``polished-greeting-moler.md`` § D4):

* Function row: ``local function <name>`` (or ``function ?`` when the
  FID has no name entry — extern / unresolvable).
* Variant row: positional axes only, space-joined as
  ``<arch> <comp> v<cver> -<opt>``. Construction reads the four keys
  declared by :data:`tokenizer.variant_tokens.prefixes.POSITIONAL_PREFIXES`
  off ``FunctionData.metadata``.
* Block row: ``Block: <i>   <first N chars of to_asm_like()>`` — preview
  truncation only; the UI layer owns the ``>>`` marker.
* Inline call row: ``<kind> function <K>: <name>[@<provider>]``
  with ``kind`` routed off the typed :class:`CallTargetType` (LOCAL /
  PLT / EXTERN) — no string compares. The token marks a function
  *reference* (the LOCAL_FUNC / PLT_FUNC / EXT_FUNC identity-band
  token); the actual ``call`` opcode is a separate instruction-rep
  token that renders on its own asm line, so no ``call `` prefix here.
* Inline jump row: ``jump block: <target_block_idx>``.

The variant-axis-key mapping below is the loader's metadata-dict shape
(``arch`` / ``compiler`` / ``compilerversion`` / ``opt`` — see
``tokenizer/aligned_data/loader/_session_parsers.py`` and
``tokenizer/aligned_data/loader/variant_resolver.py``), keyed by the
canonical prefix tuple so a future axis addition in
``POSITIONAL_PREFIXES`` flags the missing mapping at import time via
the assert below — same tripwire discipline as the prefixes module.
"""

from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence, Tuple, Union

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
    POSITIONAL_PREFIXES,
)


__all__ = [
    "aligned_variant_labels",
    "variant_label",
    "variant_label_from_axes",
    "variant_natural_sort_key",
    "function_label",
    "block_preview",
    "resolve_function_name_for_fid",
    "inline_call_label",
    "inline_jump_label",
]


# Missing-name placeholder. One symbol shared across function rows
# (extern / unresolvable FID) and inline-call rows (callee FID not in
# the sidecar) so the visual treatment stays consistent.
_MISSING_NAME = "?"

# Mapping from the canonical positional-axis prefix (single source of
# truth in :mod:`tokenizer.variant_tokens.prefixes`) to the
# ``FunctionData.metadata`` key holding that axis's value. The
# loader's metadata-dict shape predates the prefix table; this table
# is the read-side bridge. Renderer-only — encoder uses
# ``build_axis_strings`` for the on-wire form.
_AXIS_PREFIX_TO_METADATA_KEY: dict[str, str] = {
    ARCH_PREFIX: "arch",
    COMP_PREFIX: "compiler",
    CVER_PREFIX: "compilerversion",
    OPT_PREFIX: "opt",
}

# Per-axis prefix character rendered on the human-readable variant
# label (``v`` for compiler version, ``-`` for opt). Empty string for
# arch + compiler (the value alone is the label fragment). Routed
# through the same canonical prefix keys so a future axis addition
# fails the import-time assert below alongside the metadata-key map.
_AXIS_PREFIX_TO_LABEL_PREFIX: dict[str, str] = {
    ARCH_PREFIX: "",
    COMP_PREFIX: "",
    CVER_PREFIX: "v",
    OPT_PREFIX: "-",
}

# Tripwire: a new positional axis must extend both maps in lock-step
# with :data:`POSITIONAL_PREFIXES`; an import-time failure is louder
# than a silent ``?`` for the new axis.
assert set(_AXIS_PREFIX_TO_METADATA_KEY) == set(POSITIONAL_PREFIXES)
assert set(_AXIS_PREFIX_TO_LABEL_PREFIX) == set(POSITIONAL_PREFIXES)


# Per-:class:`CallTargetType` rendered word for the inline-call row.
# Dict dispatch (not if/elif) so adding a category is a one-line
# extension; routed off the typed enum so no string compares cross
# this boundary.
_CALL_TARGET_TYPE_TO_LABEL: dict[CallTargetType, str] = {
    CallTargetType.LOCAL: "local",
    CallTargetType.PLT: "plt",
    CallTargetType.EXTERN: "ext",
}

# Tripwire on enum completeness — adding a new CallTargetType variant
# must extend the renderer map.
assert set(_CALL_TARGET_TYPE_TO_LABEL) == set(CallTargetType)


def _axis_value_strings(
    label_axes: Mapping[str, Optional[str]],
) -> list[str]:
    """Per-axis rendered fragment list in :data:`POSITIONAL_PREFIXES` order.

    Each entry is ``f"{label_prefix}{value_or_missing}"`` (e.g. ``"x86"``,
    ``"clang"``, ``"v8.0"``, ``"-O3"``). Sole producer of the per-axis
    string shape — shared by :func:`variant_label_from_axes` (joins with
    a single space) and :func:`aligned_variant_labels` (column-pads
    before joining). Missing axes render as :data:`_MISSING_NAME`
    (``"?"``) with their per-axis prefix preserved (``"v?"`` / ``"-?"``).
    """
    parts: list[str] = []
    for prefix in POSITIONAL_PREFIXES:
        value = label_axes.get(prefix)
        value_str = _MISSING_NAME if value is None else str(value)
        parts.append(f"{_AXIS_PREFIX_TO_LABEL_PREFIX[prefix]}{value_str}")
    return parts


def variant_label_from_axes(
    label_axes: Mapping[str, Optional[str]],
) -> str:
    """Assemble the variant row label from a typed axis Mapping.

    ``label_axes`` is the prefix-keyed Mapping carried by the
    rendered-variant typed value (callers pre-flatten over
    :data:`POSITIONAL_PREFIXES`). Reading the Mapping in that canonical
    order yields a stable axis ordering: space-joined as
    ``<arch> <comp> v<cver> -<opt>``. Missing axes render as
    :data:`_MISSING_NAME` (``"?"``).
    """
    return " ".join(_axis_value_strings(label_axes))


# Natural-sort key: split a string into alternating (text, int) parts
# so ``v10`` sorts AFTER ``v9`` (vs lexicographic where ``v10`` < ``v9``).
# Digit sub-strings convert to ``int``; the leading/trailing/separator
# text stays as-is. Empty-string placeholder where a row's axis value is
# missing — sorts first under the missing-axes-first convention.
_NATSORT_SPLIT = re.compile(r"(\d+)")


def _natural_sort_key(value: Optional[str]) -> Tuple[Union[int, str], ...]:
    """Return a natsort key tuple for one axis value.

    Splits ``value`` on digit-runs; digit substrings become ``int``,
    text substrings stay as ``str`` lower-cased (case-insensitive sort
    so ``Clang`` and ``clang`` co-sort). ``None`` (missing axis) sorts
    first via a sentinel empty tuple.
    """
    if value is None:
        return ()
    parts: list[Union[int, str]] = []
    for part in _NATSORT_SPLIT.split(value):
        if not part:
            continue
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part.lower())
    return tuple(parts)


def variant_natural_sort_key(
    label_axes: Mapping[str, Optional[str]],
) -> Tuple[Tuple[Union[int, str], ...], ...]:
    """Multi-axis natural-sort key for a variant's :attr:`label_axes`.

    Single concern: produce the comparable key for sorting a variant
    sibling set in the canonical :data:`POSITIONAL_PREFIXES` order so
    ``v10`` sorts AFTER ``v9``. The tuple shape preserves Python's
    lexicographic tuple compare while individual axis values use
    :func:`_natural_sort_key` for the digit-aware ordering. Missing
    axes sort first (empty inner tuple).
    """
    return tuple(
        _natural_sort_key(label_axes.get(prefix))
        for prefix in POSITIONAL_PREFIXES
    )


def aligned_variant_labels(
    label_axes_list: Sequence[Mapping[str, Optional[str]]],
) -> tuple[str, ...]:
    """Return aligned variant labels in lockstep order.

    Single concern: column alignment given the FULL sibling set. Each
    axis's column width is the maximum rendered-fragment width across
    the sibling set; each row's per-axis fragment is left-aligned to
    that width and joined with a single space (preserving the existing
    convention from :func:`variant_label_from_axes`). The trailing
    axis column (``-opt``) is NOT padded with trailing spaces — pad
    only fills the gap before the next column.

    The per-axis label-prefix mapping + missing-axis rendering stay
    owned by :func:`variant_label_from_axes` via the shared
    :func:`_axis_value_strings` helper; this function adds the
    sibling-set-aware column policy on top.

    Empty input returns an empty tuple. Single-element input still
    returns aligned strings (trivially: the row IS its own max).
    """
    rows: list[list[str]] = [
        _axis_value_strings(axes) for axes in label_axes_list
    ]
    if not rows:
        return ()
    n_axes = len(POSITIONAL_PREFIXES)
    # Per-axis max width across the sibling set; each row is guaranteed
    # to have ``n_axes`` fragments by :func:`_axis_value_strings`. The
    # trailing axis (``-opt``) intentionally uses its own per-row length
    # as the "max" — :py:meth:`str.ljust` becomes a no-op there, so the
    # last column has no trailing whitespace without a special-case
    # branch inside the per-row loop.
    column_widths: list[int] = [
        max(len(row[axis_idx]) for row in rows) for axis_idx in range(n_axes - 1)
    ]
    return tuple(
        " ".join(
            [row[i].ljust(column_widths[i]) for i in range(n_axes - 1)]
            + [row[n_axes - 1]]
        )
        for row in rows
    )


def variant_label(function_data) -> str:
    """Assemble the positional-axis variant label.

    Reads ``function_data.metadata`` keyed by the four positional axes
    declared in :data:`POSITIONAL_PREFIXES` and returns a
    space-joined string like ``"x86 clang v8.0 -O3"``. Missing axes
    render as ``"?"`` (use case: unmatched section synthesised before
    a resolver row was attached).

    Thin wrapper around :func:`variant_label_from_axes`: bridges the
    loader's metadata-key shape to the canonical prefix-keyed Mapping
    via :data:`_AXIS_PREFIX_TO_METADATA_KEY`.
    """
    metadata = function_data.metadata
    label_axes: dict[str, Optional[str]] = {
        prefix: metadata.get(_AXIS_PREFIX_TO_METADATA_KEY[prefix])
        for prefix in POSITIONAL_PREFIXES
    }
    return variant_label_from_axes(label_axes)


def function_label(name: str | None) -> str:
    """Format the top-level function row.

    Returns ``"local function <name>"`` for a resolved name, or
    ``"function ?"`` when ``name`` is ``None`` — meaning the FID had
    no sidecar entry (extern / unresolvable). Per plan D3 only
    matched (local) functions appear at the top level, so the
    "function ?" path is defensive: a corrupt sidecar shouldn't crash
    the tree, just degrade gracefully.
    """
    if name is None:
        return f"function {_MISSING_NAME}"
    return f"local function {name}"


def block_preview(block, max_chars: int = 80) -> str:
    """First ``max_chars`` chars of ``block.to_asm_like()``.

    Raw truncation only — returns the prefix without any trailing
    marker. The UI layer (the horizontal-scroll widget) decides
    whether to append ``>>`` based on viewport state, so this helper
    stays renderer-agnostic.
    """
    asm = block.to_asm_like()
    return asm[:max_chars]


def resolve_function_name_for_fid(
    fid: int,
    line_to_name: Mapping[int, str],
) -> str | None:
    """Map an FID to its function name via the sidecar.

    Returns ``None`` when the FID isn't in ``line_to_name`` — typically
    means the call target is an extern or otherwise not represented in
    this binary's function-names sidecar.
    """
    return line_to_name.get(fid)


def inline_call_label(
    kind: CallTargetType,
    counter_id: int,
    callee_name: str | None,
    provider: str | None = None,
) -> str:
    """Render the inline-call row label.

    NOTE: The function name (``inline_call_label`` / ``InlineCallEntry``
    / ``InlineCallNode``) is historical — semantically these rows mark
    a function *reference* token (LOCAL_FUNC / PLT_FUNC / EXT_FUNC in
    the IDENTITY band), NOT a ``call`` opcode. The actual ``call``
    asm instruction is a separate instruction-rep token (``>=16``)
    that renders on its own asm line within the block. Renaming the
    symbols is a wider refactor and out of scope here; the rendered
    label text below has been corrected to drop the misleading
    ``"call "`` prefix.

    Per plan D4 (corrected):

    * ``CallTargetType.LOCAL`` -> ``"local function <K>: <name>"``
    * ``CallTargetType.PLT``   -> ``"plt function <K>: <name>"``
    * ``CallTargetType.EXTERN`` -> ``"ext function <K>: <name>@<provider>"``

    Missing ``callee_name`` renders as ``"?"`` (the FID resolved to
    nothing). For EXTERN, ``provider`` is the library / sidecar name
    appended after ``@``; a ``None`` provider falls back to
    ``"?"`` so the visual ``@``-suffix shape is preserved.

    ``kind`` is the typed :class:`CallTargetType` enum — no raw string
    routing. Dispatch on the per-enum word goes through
    :data:`_CALL_TARGET_TYPE_TO_LABEL`.
    """
    word = _CALL_TARGET_TYPE_TO_LABEL[kind]
    name = _MISSING_NAME if callee_name is None else callee_name
    base = f"{word} function {counter_id}: {name}"
    if kind is CallTargetType.EXTERN:
        provider_str = _MISSING_NAME if provider is None else provider
        return f"{base}@{provider_str}"
    return base


def inline_jump_label(target_block_idx: int) -> str:
    """``jump block: <target_block_idx>`` — sibling jump within a variant."""
    return f"jump block: {target_block_idx}"
