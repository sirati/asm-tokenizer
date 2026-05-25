"""Unit tests for :data:`tokenizer.arch.bitwidth.PLATFORM_BITWIDTH`.

Covers:

* the module-load tripwire (coverage of every :data:`Platform` literal),
* the explicit ``x86 -> "32"`` mapping (the reason this module exists
  — :func:`tokenizer.arch_translation.arch_to_platform` returns the
  bare literal ``"x86"`` for 32-bit x86 without any bitness suffix),
* the per-literal correctness of every entry.
"""

from __future__ import annotations

from typing import get_args

from tokenizer.arch import Platform
from tokenizer.arch.bitwidth import PLATFORM_BITWIDTH


def test_bitwidth_covers_every_platform_literal():
    """Every member of :data:`Platform` MUST have a bitwidth mapping.

    The module-load assertion already enforces this; a duplicate check
    here documents the contract for readers and protects against a
    future refactor that might silently swap the assert for a softer
    runtime check."""
    assert set(PLATFORM_BITWIDTH) == set(get_args(Platform))


def test_x86_maps_to_32_explicitly():
    """The motivating special case: ``x86`` is 32-bit x86 (``x64`` is
    the 64-bit member). Without this explicit entry the inspector's
    BITWIDTH axis would silently leave 32-bit x86 binaries in the
    ``?`` bucket (cluster W3-6)."""
    assert PLATFORM_BITWIDTH["x86"] == "32"
    assert PLATFORM_BITWIDTH["x64"] == "64"


def test_explicit_suffix_platforms_map_correctly():
    """The remaining platform literals carry a numeric suffix; the
    map's value MUST agree with that suffix. Catches a copy-paste
    typo (e.g. ``arm64 -> "32"``) at test time."""
    expected: dict[Platform, str] = {
        "arm32": "32", "arm64": "64",
        "mips32": "32", "mips64": "64",
        "ppc32": "32", "ppc64": "64",
        "riscv32": "32", "riscv64": "64",
    }
    for platform, bitwidth in expected.items():
        assert PLATFORM_BITWIDTH[platform] == bitwidth, (
            f"PLATFORM_BITWIDTH[{platform!r}] = "
            f"{PLATFORM_BITWIDTH[platform]!r}, expected {bitwidth!r}"
        )


def test_bitwidth_values_are_only_32_or_64():
    """The map's value range is the closed set ``{"32", "64"}`` — any
    other string would silently break the inspector's grouping bucket
    labels (the axis surfaces these values verbatim as group titles)."""
    assert set(PLATFORM_BITWIDTH.values()) == {"32", "64"}
