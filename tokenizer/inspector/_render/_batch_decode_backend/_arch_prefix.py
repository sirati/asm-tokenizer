"""Arch-prefix elision for BatchDecode token text.

Single concern: produce an ordered tuple of vocab-string prefixes for
a given arch label, and strip the first matching prefix from a raw
token-text atom.

BatchDecode-only. The FTL backend already strips the arch prefix in
:meth:`PlatformTokenInner.to_asm_like` (which returns ``self.token``
without the platform prefix); BatchDecode reaches the vocab via
``VocabularyManager.get_token_str`` which returns the registered name
WITH the per-ISA / family / unified prefix attached (e.g.
``x64_mov`` or ``unified_x86_mov``). To match the FTL display, the
BatchDecode emitter strips the active arch's prefix before handing the
text to :class:`AsmLine`.

Prefix ordering (most-specific first): per-ISA name (``x64_``), then
family prefix (``x_``), then unified-promoted family name
(``unified_x86_``). The first ``startswith`` match wins; a token whose
vocab name does NOT start with any of these prefixes passes through
unchanged (architecture-neutral names like ``v2:HEX``).

Plan reference: ``inspector-followup.md`` §A.2 (W4-amended).
"""

from __future__ import annotations

from tokenizer.arch import PLATFORM_FAMILY, PLATFORM_UNIFIED
from tokenizer.arch_translation import arch_to_platform


__all__ = ["arch_prefix_tuple", "strip_arch_prefix"]


# Module-load tripwire: every PLATFORM_FAMILY value must have a
# matching PLATFORM_UNIFIED entry. Without that, building the
# unified-promoted prefix below would silently emit garbage for a
# family that lacks a unified name. Wave-4 audit H-7 baked this into
# R1b to prevent the "fabricated vocab string" failure mode at the
# arch-prefix layer.
_FAMILY_VALUES = set(PLATFORM_FAMILY.values())
_UNIFIED_KEYS = set(PLATFORM_UNIFIED)
assert _FAMILY_VALUES == _UNIFIED_KEYS, (
    "PLATFORM_FAMILY values must align with PLATFORM_UNIFIED keys; "
    f"family-only: {_FAMILY_VALUES - _UNIFIED_KEYS}, "
    f"unified-only: {_UNIFIED_KEYS - _FAMILY_VALUES}"
)


def arch_prefix_tuple(arch_label: str) -> tuple[str, ...]:
    """Ordered vocab-string prefix tuple for ``arch_label``.

    ``arch_label`` is a sidecar-form arch string (``x86_64``,
    ``aarch64``, ``armv7l-hf`` ...). It is canonicalised to a
    :data:`tokenizer.arch.Platform` literal via
    :func:`arch_to_platform`, then the prefixes are emitted in
    most-specific-first order so the first ``startswith`` match in
    :func:`strip_arch_prefix` picks the tightest binding:

    1. Per-ISA prefix: ``<platform>_`` (e.g. ``x64_``, ``arm64_``).
    2. Family prefix: ``<family>_`` (e.g. ``x_``, ``arm_``). Skipped
       when the family name equals the platform name (e.g. ``mips32``
       collapses to family ``mips`` but ``mips32_`` is already the
       per-ISA prefix and there's no separate ``mips_`` family prefix
       to elide for the ``mips32`` literal).
    3. Unified-promoted prefix: ``<unified>_`` (e.g. ``unified_x86_``,
       ``unified_arm_``).

    Returns an empty tuple if ``arch_label`` is the empty string --
    convenient for backends that haven't plumbed the arch through yet
    (no stripping happens; tokens pass through unchanged).
    """
    if not arch_label:
        return ()
    platform = arch_to_platform(arch_label)
    family = PLATFORM_FAMILY[platform]
    unified = PLATFORM_UNIFIED[family]
    prefixes: list[str] = [f"{platform}_"]
    if family != platform:
        prefixes.append(f"{family}_")
    prefixes.append(f"{unified}_")
    return tuple(prefixes)


def strip_arch_prefix(token_str: str, prefixes: tuple[str, ...]) -> str:
    """Strip the first matching prefix from ``token_str``.

    Pure function. Iterates ``prefixes`` in order; the first one whose
    ``token_str.startswith`` matches has its length sliced off. A token
    whose vocab name does NOT start with any of the prefixes passes
    through unchanged -- the prefix tuple is not exhaustive (e.g.
    architecture-neutral names like ``v2:HEX`` or the IDENTITY-band
    placeholders never carry an arch prefix).

    Empty ``prefixes`` is a fast-path no-op for backends that haven't
    plumbed the arch through yet.
    """
    for prefix in prefixes:
        if token_str.startswith(prefix):
            return token_str[len(prefix):]
    return token_str
