"""Per-``Platform`` family-display name lookup.

Single concern: map every :data:`tokenizer.arch.Platform` literal onto a
human-readable family name used by inspector UX surfaces that need to
collapse a bitness-bearing arch (``arm32`` / ``arm64`` /
``mips32`` / ``mips64`` / ``x86`` / ``x64`` / ...) down to a single
family bucket (``arm`` / ``mips`` / ``x86`` / ...).

Distinct from :data:`tokenizer.arch.PLATFORM_FAMILY` (the vocab-merging
family prefix used by the unified vocab writer). That map collapses
``x86`` and ``x64`` to ``"x"`` — short enough to read inside a
token-name prefix but not the right surface for a user-facing
grouping header. The display map collapses them to ``"x86"`` instead
(the conventional shorthand for "x86-family"), matching the user-
visible bitwidth-collapsed groups people expect when sorting binaries
by ISA.

A module-load tripwire asserts the dict covers every member of
:data:`Platform` so a new platform literal must land here before the
``Platform`` change can import.
"""

from __future__ import annotations

from typing import Mapping, get_args

from tokenizer.arch import Platform


__all__ = ["PLATFORM_FAMILY_DISPLAY"]


# Human-readable family name for every ``Platform`` literal. ``x86``/
# ``x64`` collapse to ``"x86"`` (the conventional shorthand) rather
# than ``"x"`` so the grouping header stays readable; all other
# families drop the ``32`` / ``64`` suffix verbatim.
PLATFORM_FAMILY_DISPLAY: Mapping[Platform, str] = {
    "x86": "x86", "x64": "x86",
    "arm32": "arm", "arm64": "arm",
    "mips32": "mips", "mips64": "mips",
    "ppc32": "ppc", "ppc64": "ppc",
    "riscv32": "riscv", "riscv64": "riscv",
}


# Module-load tripwire: every ``Platform`` literal MUST appear in the
# map. If ``Platform`` gains a new entry without a display-family
# mapping the import fails immediately — no silent fall-through to the
# raw arch string downstream.
assert set(PLATFORM_FAMILY_DISPLAY) == set(get_args(Platform)), (
    "PLATFORM_FAMILY_DISPLAY must cover every Platform literal; "
    f"missing={set(get_args(Platform)) - set(PLATFORM_FAMILY_DISPLAY)!r}, "
    f"extra={set(PLATFORM_FAMILY_DISPLAY) - set(get_args(Platform))!r}"
)
