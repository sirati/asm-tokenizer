"""Tests for the BatchDecode arch-prefix elision helpers.

Pins:

* :func:`arch_prefix_tuple` returns a most-specific-first ordered
  tuple containing the per-ISA prefix, the family prefix (when
  distinct), and the unified-promoted prefix.
* :func:`strip_arch_prefix` slices off the first matching prefix and
  passes non-matching tokens through unchanged.
* The module-load tripwire (``PLATFORM_FAMILY`` values align with
  ``PLATFORM_UNIFIED`` keys) cannot regress without the import
  failing.

Plan reference: ``inspector-followup.md`` §A.2 (W4-amended).
"""

from __future__ import annotations

import pytest

from tokenizer.arch import PLATFORM_FAMILY, PLATFORM_UNIFIED
from tokenizer.inspector._render._batch_decode_backend._arch_prefix import (
    arch_prefix_tuple,
    strip_arch_prefix,
)


# Sample arch labels across all five families; the sidecar form goes
# through ``arch_to_platform`` so both canonical (``x64``) and verbose
# (``x86_64``) inputs are covered.
@pytest.mark.parametrize(
    "arch_label,platform_prefix,family_prefix,unified_prefix",
    [
        ("x64", "x64_", "x_", "unified_x86_"),
        ("x86_64", "x64_", "x_", "unified_x86_"),
        ("x86", "x86_", "x_", "unified_x86_"),
        ("aarch64", "arm64_", "arm_", "unified_arm_"),
        ("armv7l", "arm32_", "arm_", "unified_arm_"),
        ("mips64", "mips64_", "mips_", "unified_mips_"),
        ("ppc64le", "ppc64_", "ppc_", "unified_ppc_"),
        ("riscv64", "riscv64_", "riscv_", "unified_riscv_"),
    ],
)
def test_arch_prefix_tuple_orders_most_specific_first(
    arch_label: str,
    platform_prefix: str,
    family_prefix: str,
    unified_prefix: str,
) -> None:
    prefixes = arch_prefix_tuple(arch_label)
    assert prefixes[0] == platform_prefix
    assert prefixes[-1] == unified_prefix
    assert family_prefix in prefixes
    # First entry is the per-ISA prefix; the unified prefix is last
    # so the most-specific match wins in ``strip_arch_prefix``.
    assert prefixes.index(platform_prefix) < prefixes.index(unified_prefix)
    assert prefixes.index(family_prefix) < prefixes.index(unified_prefix)


def test_arch_prefix_tuple_empty_label_yields_empty_tuple() -> None:
    """Backends that haven't plumbed the arch through pass ``""``; the
    helper MUST return an empty tuple so the strip becomes a no-op."""
    assert arch_prefix_tuple("") == ()


def test_arch_prefix_tuple_unknown_label_yields_empty_tuple() -> None:
    """Unmatched-arm :class:`FunctionData.metadata` carries the literal
    sentinel ``"unknown"`` (per
    ``_session_parsers.build_unmatched_function_data``); the helper
    MUST return an empty tuple so the cross-arm navigation to an
    unmatched callee does not blow up with ``ValueError`` from
    :func:`arch_to_platform`."""
    assert arch_prefix_tuple("unknown") == ()


@pytest.mark.parametrize(
    "arch_label,raw,expected",
    [
        ("x64", "x64_mov", "mov"),
        ("x64", "x_addss", "addss"),
        ("x64", "unified_x86_pause", "pause"),
        ("aarch64", "arm64_ldr", "ldr"),
        ("aarch64", "arm_b_eq", "b_eq"),
        ("aarch64", "unified_arm_nop", "nop"),
        # Non-matching token passes through unchanged: an architecture-
        # neutral name like ``v2:HEX`` or a MEM display char.
        ("x64", "v2:deadbeef", "v2:deadbeef"),
        ("x64", "[", "["),
        ("x64", "", ""),
    ],
)
def test_strip_arch_prefix_strips_first_match(
    arch_label: str,
    raw: str,
    expected: str,
) -> None:
    prefixes = arch_prefix_tuple(arch_label)
    assert strip_arch_prefix(raw, prefixes) == expected


def test_strip_arch_prefix_empty_prefixes_is_noop() -> None:
    """Empty prefix tuple means "no arch context" -- every token passes
    through verbatim."""
    for raw in ("x64_mov", "unified_arm_ldr", "v2:42", ""):
        assert strip_arch_prefix(raw, ()) == raw


def test_strip_arch_prefix_picks_most_specific_match() -> None:
    """``x64_mov`` would also match the ``x_`` family prefix on the
    suffix ``"64_mov"``, but ``startswith("x64_")`` wins first per the
    ordered tuple. The result keeps the bare mnemonic ``mov``, NOT
    ``64_mov``."""
    prefixes = arch_prefix_tuple("x64")
    assert strip_arch_prefix("x64_mov", prefixes) == "mov"


def test_platform_family_unified_alignment_tripwire() -> None:
    """The module-load tripwire asserts ``set(PLATFORM_FAMILY.values())
    == set(PLATFORM_UNIFIED)``. Pinning the invariant as a test keeps
    the failure mode auditable when the source-of-truth maps grow."""
    assert set(PLATFORM_FAMILY.values()) == set(PLATFORM_UNIFIED)
