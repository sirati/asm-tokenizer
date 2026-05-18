"""Tests for ``tokenizer.variant_tokens.prefixes``.

Covers axis-string construction and the alias-collapse contract. If
sibling 1C has not landed in this worktree the alias collapse falls
back to identity (see ``prefixes.py`` import block); the
arch-collapse test below checks the placeholder behaves identically
to identity so 1A merges cleanly against any 1C merge order.
"""

from __future__ import annotations

from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
    build_arch_token,
    build_axis_strings,
    build_comp_token,
    build_cver_token,
    build_metadata_tokens,
    build_opt_token,
    arch_to_variant_arch,
)
from ._fakes import FakeVersionInfo


def test_individual_axis_builders():
    assert build_arch_token("x86_64").startswith(ARCH_PREFIX)
    assert build_comp_token("gcc") == f"{COMP_PREFIX}gcc"
    assert build_cver_token("gcc", "13.2.0") == f"{CVER_PREFIX}gcc:13.2.0"
    # opt strips a single leading dash but is otherwise literal.
    assert build_opt_token("-O2") == f"{OPT_PREFIX}O2"
    assert build_opt_token("O2") == f"{OPT_PREFIX}O2"
    assert build_opt_token("-Os") == f"{OPT_PREFIX}Os"


def test_cver_namespaces_under_compiler():
    """Same version string under two compilers must differ — the
    point of the cver namespacing is to prevent collisions."""
    gcc = build_cver_token("gcc", "13.2.0")
    clang = build_cver_token("clang", "13.2.0")
    assert gcc != clang


def test_arch_alias_collapse_or_identity():
    """If 1C has merged, ``arch_to_variant_arch`` collapses x86_64
    family to ``x64`` and aarch64 family to ``arm64``. If 1C has not
    merged, the placeholder is identity. Both behaviors are
    deterministic — assert that calling twice returns the same
    result, and that any collapse is consistent across the input
    family (x86_64 and amd64 land on the same alias)."""
    a = arch_to_variant_arch("x86_64")
    b = arch_to_variant_arch("amd64")
    # Either both collapse to x64 (post-1C) or both stay distinct
    # (placeholder identity). The only forbidden state is "one
    # collapses, the other doesn't" — which would mean a half-merged
    # alias table.
    if a != "x86_64":  # 1C has landed
        assert a == "x64"
        assert b == "x64"
    else:  # placeholder identity
        assert a == "x86_64"
        assert b == "amd64"


def test_axis_strings_layout_no_metadata():
    """Plan §"Token-string prefixes" — exactly 4 positional axes when
    extra_metadata is empty."""
    vi = FakeVersionInfo(extra_metadata={})
    strings = build_axis_strings(vi)
    assert len(strings) == 4
    assert strings[0].startswith(ARCH_PREFIX)
    assert strings[1].startswith(COMP_PREFIX)
    assert strings[2].startswith(CVER_PREFIX)
    assert strings[3].startswith(OPT_PREFIX)


def test_axis_strings_metadata_sorted_alpha_by_key():
    """Metadata tail order: keys alphabetical."""
    vi = FakeVersionInfo(
        extra_metadata={"sanitizer": "address", "hardening": "full"}
    )
    strings = build_axis_strings(vi)
    # Tail starts at index 4.
    tail = strings[4:]
    assert tail == ["hardening:full", "sanitizer:address"]


def test_axis_strings_multi_valued_key_sorted_within_key():
    """A multi-valued key emits one token per value, value-order
    sorted via ``sorted(str(v) for v in values)`` per plan."""
    vi = FakeVersionInfo(
        extra_metadata={"hardening": ["full", "fortify"]}
    )
    strings = build_axis_strings(vi)
    # "fortify" < "full" alphabetically.
    assert strings[4:] == ["hardening:fortify", "hardening:full"]


def test_axis_strings_mixed_type_values_str_coerced():
    """The ``sorted(str(v) for v in values)`` coercion protects
    against ``TypeError`` on lists like ``[1, "a", True]`` — matches
    the ``encode_flags`` precedent at ``variants.py:64``."""
    vi = FakeVersionInfo(
        extra_metadata={"mix": [1, "a", True]}
    )
    strings = build_axis_strings(vi)
    # str(True) == "True", str(1) == "1"; sorted: "1","True","a".
    assert strings[4:] == ["mix:1", "mix:True", "mix:a"]


def test_build_metadata_tokens_grouped_by_key():
    grouped = build_metadata_tokens(
        {"b": "2", "a": ["x", "y"]}
    )
    assert grouped == [
        ("a", ["a:x", "a:y"]),
        ("b", ["b:2"]),
    ]


def test_axis_strings_accepts_variant_info_field_name():
    """``variant_info.VariantInfo`` uses ``compiler_version`` (with
    underscore); the builder shape uses ``compilerversion``. The
    function must accept either via getattr fallback."""

    class VariantInfoShape:
        arch = "x86_64"
        compiler = "gcc"
        compiler_version = "13.2.0"
        opt = "-O2"
        extra_metadata: dict = {}

    strings = build_axis_strings(VariantInfoShape())
    assert any(s.startswith(CVER_PREFIX) for s in strings)
    # Version made it through.
    assert any(s.endswith("13.2.0") for s in strings)


def test_axis_strings_determinism_across_runs():
    """Same input → same output, byte-identical. Critical for
    reproducible vocab IDs across runs."""
    vi = FakeVersionInfo(
        extra_metadata={"hardening": ["full", "fortify"], "sanitizer": "address"}
    )
    a = build_axis_strings(vi)
    b = build_axis_strings(vi)
    assert a == b
