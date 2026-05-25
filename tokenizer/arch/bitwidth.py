"""Per-``Platform`` bitwidth lookup.

Single concern: map every :data:`tokenizer.arch.Platform` literal onto
its bitwidth suffix (``"32"`` / ``"64"``) so callers that need a clean
"32 or 64" answer (notably the inspector's BITWIDTH axis grouping)
have a single source of truth.

Most platform names already carry the bitness in their suffix
(``arm32``/``arm64``/``mips64``/...); ``x86`` does not, because
:func:`tokenizer.arch_translation.arch_to_platform` returns the bare
literal ``"x86"`` for 32-bit x86 (``i686``, ``i386``, ``x86``). The
explicit ``x86 -> "32"`` mapping is the reason this module exists
rather than a ``.endswith(("32","64"))`` one-liner at the call site.

A module-load tripwire asserts the dict covers every member of
:data:`Platform` so a new platform literal must land here before the
``Platform`` change can import.
"""

from __future__ import annotations

from typing import Mapping, get_args

from tokenizer.arch import Platform


__all__ = ["PLATFORM_BITWIDTH"]


# Explicit bitwidth for every ``Platform`` literal. The 32 / 64 suffix
# is recoverable from the literal for all entries except ``x86`` (which
# means 32-bit x86 — ``x64`` is the 64-bit member). The map is total
# over ``Platform`` by construction; the module-load assert below
# pins coverage.
PLATFORM_BITWIDTH: Mapping[Platform, str] = {
    "x86": "32", "x64": "64",
    "arm32": "32", "arm64": "64",
    "mips32": "32", "mips64": "64",
    "ppc32": "32", "ppc64": "64",
    "riscv32": "32", "riscv64": "64",
}


# Module-load tripwire: every ``Platform`` literal MUST appear in the
# map. If ``Platform`` gains a new entry without a bitwidth mapping the
# import fails immediately — no silent ``None``-bucket downstream.
assert set(PLATFORM_BITWIDTH) == set(get_args(Platform)), (
    "PLATFORM_BITWIDTH must cover every Platform literal; "
    f"missing={set(get_args(Platform)) - set(PLATFORM_BITWIDTH)!r}, "
    f"extra={set(PLATFORM_BITWIDTH) - set(get_args(Platform))!r}"
)
