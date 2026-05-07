"""Translate sidecar ``arch`` strings to the tokenizer's ``Platform``.

Single concern: map the verbose, distro-style architecture names that
appear in JSON sidecars (``x86_64``, ``aarch64``, ``armv7l-hf``,
``mips64el``, ``mipsel``, ``i686``, ``ppc32``, ``ppc64``, ``riscv64``,
...) onto the tokenizer's compact ``Platform`` literal type
(``x86``, ``x64``, ``arm32``, ``arm64``, ``mips32``, ``mips64``,
``ppc32``, ``ppc64``, ``riscv32``, ``riscv64``).

Sidecar callers cannot rely on the tokenizer's filename auto-detect
(the binary inside the tarball is named ``hello`` or ``busybox``,
carrying no platform prefix), so the worker handler must compute the
``Platform`` value from the variant's ``arch`` field and pass it in
explicitly. This module is the single source of truth for that
translation; every arch string the dataset emits MUST resolve to a
member of ``Platform`` here, and unknown arches surface as
``ValueError`` at the boundary instead of silently falling through to
a wrong platform.

Endianness suffixes (``el`` for little-endian, ``eb`` / bare for
big-endian) intentionally collapse onto the same ``Platform`` value:
the tokenizer's ``Platform`` enum is bitness-only, and endianness is
recovered downstream by the disassembly backend (Ghidra / angr) from
the ELF header. So ``mipsel`` and ``mips`` both → ``mips32``;
``mips64el`` and ``mips64`` both → ``mips64``; same for ``ppc*``.

ARM variants (``armv7l``, ``armv7l-hf``, ``armv6l``, ``armhf`` ...)
all map to ``arm32`` for the same reason: ABI/float-mode hints belong
to the binary, not the ISA-token namespace.
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
