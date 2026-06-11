"""Unit tests for fullname axis parsing.

Covers the documented program-name derivation (part after the first
underscore, including ``__hash`` suffixes), the dashed-ISA case
(``armv7l-hf``), the dashed-comp_version "parse from both ends" rule,
family/bitness derivation, and the unparseable-name degradation contract
(NULL axes, never a crash).
"""

from __future__ import annotations

from scripts.collect_stats.axes import parse_axes


def test_simple_fullname_axes() -> None:
    a = parse_axes("x64-clang-3.5-O0_minigzipsh")
    assert a.parsed
    assert a.program == "minigzipsh"
    assert a.isa_exact == "x64"
    assert a.isa_family == "x86"  # x86 family covers x64
    assert a.bitness == 64
    assert a.comp == "clang"
    assert a.comp_version == "3.5"
    assert a.optim_level == "O0"


def test_arm32_family_and_bitness() -> None:
    a = parse_axes("arm32-gcc-4.8-O2_clambc")
    assert a.isa_exact == "arm32"
    assert a.isa_family == "arm"
    assert a.bitness == 32
    assert a.comp == "gcc"
    assert a.comp_version == "4.8"
    assert a.optim_level == "O2"


def test_dashed_isa_armv7l_hf() -> None:
    """``armv7l-hf`` is a single ISA token that itself contains a dash —
    the longest-prefix match must claim it rather than splitting at the
    first dash."""
    a = parse_axes("armv7l-hf-clang-10.0.1-Oz_hello__15f3f338")
    assert a.parsed
    assert a.isa_exact == "armv7l-hf"
    assert a.isa_family == "arm"
    assert a.bitness == 32
    assert a.comp == "clang"
    assert a.comp_version == "10.0.1"
    assert a.optim_level == "Oz"
    # Program retains the double-underscore + hash suffix verbatim.
    assert a.program == "hello__15f3f338"


def test_underscore_in_isa_x86_64() -> None:
    """``x86_64`` contains an underscore — the ISA must be matched before
    the program is split, or ``x86_64`` would be severed into ``x86`` +
    ``64-...`` and mis-parse.  Program is split from the post-ISA tail."""
    a = parse_axes("x86_64-clang-11.1.0-O2_busybox__f0d42c45")
    assert a.parsed
    assert a.isa_exact == "x86_64"
    assert a.isa_family == "x86"
    assert a.bitness == 64
    assert a.comp == "clang"
    assert a.comp_version == "11.1.0"
    assert a.optim_level == "O2"
    assert a.program == "busybox__f0d42c45"


def test_dashed_comp_version() -> None:
    """A compiler version that contains a dash survives because the
    middle is reassembled from the dash fields between comp and optim."""
    a = parse_axes("x64-gcc-7-20210101-O3_prog")
    assert a.isa_exact == "x64"
    assert a.comp == "gcc"
    assert a.comp_version == "7-20210101"
    assert a.optim_level == "O3"
    assert a.program == "prog"


def test_program_only_no_dashes_in_remainder_is_unparseable() -> None:
    """A prefix with a known ISA but fewer than 3 comp/optim fields can't
    be split into comp/version/optim — axes go NULL, program is kept."""
    a = parse_axes("x64-clang_prog")
    assert not a.parsed
    assert a.isa_exact is None
    assert a.comp is None
    assert a.program == "prog"


def test_unknown_isa_is_unparseable() -> None:
    a = parse_axes("weirdname-noisa_thing")
    assert not a.parsed
    assert a.isa_exact is None
    assert a.isa_family is None
    assert a.bitness is None
    assert a.comp is None
    assert a.comp_version is None
    assert a.optim_level is None
    # Program is still recovered from the first-underscore split.
    assert a.program == "thing"


def test_no_underscore_has_no_program() -> None:
    a = parse_axes("x64-clang-3.5-O0")
    # No underscore ⇒ no program; the prefix is the whole string and the
    # remainder still parses into the axis tuple.
    assert a.program is None
    assert a.isa_exact == "x64"
    assert a.optim_level == "O0"


def test_mips64el_exact_vs_family() -> None:
    a = parse_axes("mips64el-gcc-9-Os_busybox")
    assert a.isa_exact == "mips64el"
    assert a.isa_family == "mips"
    assert a.bitness == 64
    assert a.optim_level == "Os"
