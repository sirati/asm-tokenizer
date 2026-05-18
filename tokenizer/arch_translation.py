"""Translate sidecar ``arch`` strings to the tokenizer's ``Platform``.

This module hosts **two** translation functions with distinct concerns
that must not be conflated by callers:

- :func:`arch_to_platform` — **ISA / disassembler dispatch.**
  Bitness-only collapse. Maps verbose distro-style arch names
  (``x86_64``, ``aarch64``, ``armv7l-hf``, ``mips64el`` ...) onto the
  tokenizer's compact ``Platform`` literal type
  (``x86``, ``x64``, ``arm32``, ``arm64``, ``mips32``, ``mips64``,
  ``ppc32``, ``ppc64``, ``riscv32``, ``riscv64``).
  ABI/float/endianness sub-variants intentionally collapse: the enum
  is bitness-only and the backend (Ghidra / angr) recovers
  endianness/float-mode from the ELF header. Unknown input is a hard
  error (``ValueError``) so a never-before-seen arch fails loudly at
  the worker boundary instead of silently picking a wrong Platform.

- :func:`arch_to_variant_arch` — **variant-vocab identity.**
  ABI/sub-arch preserving. Only the family-equivalent aliases collapse
  (``x86_64``/``amd64``/``x64`` → ``x64``; ``aarch64``/``arm64`` →
  ``arm64``); everything else is identity (``armv7l``, ``armv7l-hf``,
  ``armv6l``, ``mipsel``, ``mips64el``, ``ppc64le``, ``riscv64``, …).
  Unknown input passes through unchanged — the variant vocab is
  corpus-driven and must never block a run on an unfamiliar arch.

Sidecar callers cannot rely on the tokenizer's filename auto-detect
(the binary inside the tarball is named ``hello`` or ``busybox``,
carrying no platform prefix), so the worker handler must compute the
``Platform`` value from the variant's ``arch`` field and pass it in
explicitly. :func:`arch_to_platform` is the single source of truth
for that translation; every arch string the dataset emits MUST
resolve to a member of ``Platform`` there.

Endianness suffixes (``el`` for little-endian, ``eb`` / bare for
big-endian) intentionally collapse onto the same ``Platform`` value
for :func:`arch_to_platform`. So ``mipsel`` and ``mips`` both →
``mips32``; ``mips64el`` and ``mips64`` both → ``mips64``; same for
``ppc*``.

ARM variants (``armv7l``, ``armv7l-hf``, ``armv6l``, ``armhf`` ...)
all map to ``arm32`` under :func:`arch_to_platform` for the same
reason. Under :func:`arch_to_variant_arch` these stay **distinct**
because the variant vocab models ABI/float-mode identity.
"""

from __future__ import annotations

from tokenizer.arch import Platform

# Sidecar arch string → tokenizer ``Platform`` literal.
#
# Sourced from the actual dataset (``walk_dataset(src/dataset)``) plus
# the tokenizer's own legacy 4-axis filename platform names (which are
# already valid ``Platform`` values, so they map identity). Every key
# is a string that has been observed as a sidecar ``arch`` value or as
# a legacy filename platform prefix; new strings must be added here
# explicitly — ``arch_to_platform`` raises on miss instead of guessing.
_ARCH_TO_PLATFORM: dict[str, Platform] = {
    # x86 family
    "x86": "x86",
    "i686": "x86",
    "i386": "x86",
    "x64": "x64",
    "x86_64": "x64",
    "amd64": "x64",
    # ARM family (32-bit ABI/float variants all → arm32; backend
    # recovers endianness/float-mode from the ELF header)
    "arm32": "arm32",
    "armv7l": "arm32",
    "armv7l-hf": "arm32",
    "armv6l": "arm32",
    "armhf": "arm32",
    "arm": "arm32",
    "arm64": "arm64",
    "aarch64": "arm64",
    # MIPS family (endianness suffix ``el`` collapses; bitness stays)
    "mips32": "mips32",
    "mips": "mips32",
    "mipsel": "mips32",
    "mips64": "mips64",
    "mips64el": "mips64",
    # PowerPC family
    "ppc32": "ppc32",
    "ppc": "ppc32",
    "ppc64": "ppc64",
    "ppc64le": "ppc64",
    # RISC-V family
    "riscv32": "riscv32",
    "riscv64": "riscv64",
}


def arch_to_platform(arch: str) -> Platform:
    """Translate a sidecar ``arch`` string to the tokenizer's
    ``Platform`` literal.

    Raises ``ValueError`` on unknown input. The error message lists the
    accepted set so a sidecar with a never-before-seen arch fails
    loudly at the worker boundary rather than silently picking a wrong
    Platform downstream.
    """
    try:
        return _ARCH_TO_PLATFORM[arch]
    except KeyError:
        accepted = ", ".join(sorted(_ARCH_TO_PLATFORM))
        raise ValueError(
            f"unknown sidecar arch {arch!r}; accepted: {accepted}"
        ) from None


# Variant-vocab arch aliases: ONLY the family-equivalent renames
# collapse here (x86_64/amd64/x64 -> x64; aarch64/arm64 -> arm64).
# Every other arch passes through identity so the variant vocab
# preserves ABI/sub-arch identity (armv7l vs armv7l-hf vs armv6l
# remain distinct tokens; ppc64le keeps its endianness suffix; etc.).
#
# Unlike ``_ARCH_TO_PLATFORM`` this map is intentionally NOT
# exhaustive: it is a sparse alias table consulted by
# ``arch_to_variant_arch``, which falls back to identity for any
# unknown input. The variant vocab is corpus-driven and must never
# block a run on an arch we haven't catalogued yet.
_VARIANT_ARCH_ALIASES: dict[str, str] = {
    "x86_64": "x64",
    "amd64": "x64",
    "x64": "x64",
    "aarch64": "arm64",
    "arm64": "arm64",
}


def arch_to_variant_arch(arch: str) -> str:
    """Canonicalise an ``arch`` string for the **variant vocab**.

    This is the ABI/sub-arch-preserving sibling of
    :func:`arch_to_platform`. Use this when constructing variant-axis
    tokens (``arch:<canonical>``), NOT when dispatching the
    disassembler.

    Behaviour:

    - ``x86_64`` / ``amd64`` / ``x64`` → ``x64``
    - ``aarch64`` / ``arm64`` → ``arm64``
    - Everything else (``armv7l``, ``armv7l-hf``, ``armv6l``,
      ``mipsel``, ``mips64el``, ``ppc64le``, ``riscv32``,
      ``riscv64``, …) passes through identity.
    - Unknown input passes through identity. There is **no**
      ``ValueError``: the variant vocab is corpus-driven and must
      never block a run on an arch the alias table hasn't catalogued.

    Contract contrast with :func:`arch_to_platform`:

    >>> arch_to_platform("armv7l-hf")
    'arm32'
    >>> arch_to_variant_arch("armv7l-hf")
    'armv7l-hf'

    The first collapses ABI/float-mode hints because the disassembler
    only needs bitness; the second preserves them because the variant
    vocab models per-binary identity.
    """
    return _VARIANT_ARCH_ALIASES.get(arch, arch)


def all_known_arch_strings() -> tuple[str, ...]:
    """Every arch alias understood by the translator, sorted.

    Used by the dispatcher to seed a permissive ``--platform`` allowlist
    that admits both canonical ``Platform`` literals and the sidecar
    arch strings that filename-based discovery walks see in
    ``<arch>-<compiler>-...`` outputs.
    """
    return tuple(sorted(_ARCH_TO_PLATFORM))
