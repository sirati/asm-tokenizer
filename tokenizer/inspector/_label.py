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
* Inline call row: ``call <kind> function <K>: <name>[@<provider>]``
  with ``kind`` routed off the typed :class:`CallTargetType` (LOCAL /
  PLT / EXTERN) — no string compares.
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

from typing import Mapping

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
    POSITIONAL_PREFIXES,
)


__all__ = [
    "variant_label",
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


def variant_label(function_data) -> str:
    """Assemble the positional-axis variant label.

    Reads ``function_data.metadata`` keyed by the four positional axes
    declared in :data:`POSITIONAL_PREFIXES` and returns a
    space-joined string like ``"x86 clang v8.0 -O3"``. Missing axes
    render as ``"?"`` (use case: unmatched section synthesised before
    a resolver row was attached).
    """
    metadata = function_data.metadata
    parts: list[str] = []
    for prefix in POSITIONAL_PREFIXES:
        metadata_key = _AXIS_PREFIX_TO_METADATA_KEY[prefix]
        value = metadata.get(metadata_key)
        value_str = _MISSING_NAME if value is None else str(value)
        parts.append(f"{_AXIS_PREFIX_TO_LABEL_PREFIX[prefix]}{value_str}")
    return " ".join(parts)


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

    Per plan D4:

    * ``CallTargetType.LOCAL`` -> ``"call local function <K>: <name>"``
    * ``CallTargetType.PLT``   -> ``"call plt function <K>: <name>"``
    * ``CallTargetType.EXTERN`` -> ``"call ext function <K>: <name>@<provider>"``

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
    base = f"call {word} function {counter_id}: {name}"
    if kind is CallTargetType.EXTERN:
        provider_str = _MISSING_NAME if provider is None else provider
        return f"{base}@{provider_str}"
    return base


def inline_jump_label(target_block_idx: int) -> str:
    """``jump block: <target_block_idx>`` — sibling jump within a variant."""
    return f"jump block: {target_block_idx}"
