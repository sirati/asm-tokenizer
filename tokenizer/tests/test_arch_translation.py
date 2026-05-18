"""Tests for ``tokenizer.arch_translation``.

Covers the contract split between :func:`arch_to_platform` (bitness-only
ISA dispatch; ``ValueError`` on unknown) and
:func:`arch_to_variant_arch` (ABI/sub-arch preserving variant-vocab
identity; identity passthrough on unknown).
"""

from __future__ import annotations

import pytest

from tokenizer.arch_translation import (
    arch_to_platform,
    arch_to_variant_arch,
)


class TestArchToVariantArchAliases:
    """The five aliases that DO collapse for the variant vocab."""

    @pytest.mark.parametrize(
        "raw",
        ["x86_64", "amd64", "x64"],
    )
    def test_x64_family_collapses(self, raw: str) -> None:
        assert arch_to_variant_arch(raw) == "x64"

    @pytest.mark.parametrize(
        "raw",
        ["aarch64", "arm64"],
    )
    def test_arm64_family_collapses(self, raw: str) -> None:
        assert arch_to_variant_arch(raw) == "arm64"


class TestArchToVariantArchIdentity:
    """Everything outside the alias table passes through unchanged."""

    @pytest.mark.parametrize(
        "raw",
        [
            "armv7l",
            "armv7l-hf",
            "armv6l",
            "armhf",
            "arm32",
            "arm",
            "mips",
            "mipsel",
            "mips32",
            "mips64",
            "mips64el",
            "ppc",
            "ppc32",
            "ppc64",
            "ppc64le",
            "riscv32",
            "riscv64",
            "x86",
            "i686",
            "i386",
        ],
    )
    def test_known_non_aliased_passes_through(self, raw: str) -> None:
        assert arch_to_variant_arch(raw) == raw

    def test_arm32_variants_stay_distinct(self) -> None:
        # The whole point of the variant vocab: ABI/float-mode hints
        # are preserved, NOT collapsed into one "arm32" bucket like
        # arch_to_platform does. These three must survive as
        # independent tokens.
        outputs = {
            arch_to_variant_arch("armv7l"),
            arch_to_variant_arch("armv7l-hf"),
            arch_to_variant_arch("armv6l"),
        }
        assert outputs == {"armv7l", "armv7l-hf", "armv6l"}

    def test_unknown_input_identity_no_exception(self) -> None:
        # Corpus-driven contract: never block a run on an unfamiliar
        # arch. No ValueError, no warning — just identity.
        assert arch_to_variant_arch("foobar123") == "foobar123"

    def test_empty_string_identity_no_exception(self) -> None:
        assert arch_to_variant_arch("") == ""

    def test_unknown_input_does_not_raise(self) -> None:
        # Explicit contract guard mirroring arch_to_platform's
        # ValueError discipline (where the two functions diverge).
        try:
            arch_to_variant_arch("totally-made-up-arch-9000")
        except Exception as exc:  # pragma: no cover - test fails on path
            pytest.fail(
                f"arch_to_variant_arch raised on unknown input: {exc!r}"
            )


class TestContractContrast:
    """Same input, two functions, two different answers — by design."""

    def test_armv7l_hf_diverges(self) -> None:
        # arch_to_platform: bitness-only -> arm32
        # arch_to_variant_arch: ABI-preserving -> armv7l-hf
        assert arch_to_platform("armv7l-hf") == "arm32"
        assert arch_to_variant_arch("armv7l-hf") == "armv7l-hf"
        assert arch_to_platform("armv7l-hf") != arch_to_variant_arch(
            "armv7l-hf"
        )

    def test_x86_64_converges(self) -> None:
        # Same input, same output here — both collapse the x86_64
        # alias — but for different reasons (bitness vs vocab rename).
        assert arch_to_platform("x86_64") == "x64"
        assert arch_to_variant_arch("x86_64") == "x64"

    def test_mips64el_diverges(self) -> None:
        # arch_to_platform: endianness collapses -> mips64
        # arch_to_variant_arch: endianness preserved -> mips64el
        assert arch_to_platform("mips64el") == "mips64"
        assert arch_to_variant_arch("mips64el") == "mips64el"

    def test_unknown_input_diverges_on_failure_mode(self) -> None:
        # arch_to_platform raises on unknown; arch_to_variant_arch
        # returns identity. This documents the asymmetric failure
        # contract between the two.
        with pytest.raises(ValueError):
            arch_to_platform("foobar123")
        assert arch_to_variant_arch("foobar123") == "foobar123"
