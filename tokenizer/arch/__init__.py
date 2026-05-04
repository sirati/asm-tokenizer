from typing import Literal

from tokenizer.arch.provider import ArchitectureProvider

# Per-ISA platform names. Each name carries an explicit bitness suffix
# so the family-prefix used by cross-bitness vocab merging
# (`PLATFORM_FAMILY`) can stay distinct from any single ISA name —
# without that, e.g. a unified `mips_addu` token (family-merged across
# 32+64) would collide visually with the old "mips means 32-bit MIPS"
# convention. Earlier revs called 32-bit MIPS just `mips` and 32-bit
# PPC just `ppc`; both renamed here to `mips32` / `ppc32` to match the
# explicit-bitness pattern arm32/arm64/riscv32/riscv64 already use AND
# to free `mips` / `ppc` as family-merge prefixes.
Platform = Literal[
    "x86", "x64",
    "arm32", "arm64",
    "mips32", "mips64",
    "ppc32", "ppc64",
    "riscv32", "riscv64",
]


# Family prefix used in the unified vocab when family-merging is active
# (the only mode the unifier writes today). Each prefix is distinct
# from every member ISA name so a token like `mips_addu` in the unified
# vocab unambiguously means "family-merged across mips32+mips64", and
# `mips32_addu` means "single-ISA, only seen in mips32 binaries".
PLATFORM_FAMILY: dict[str, str] = {
    "x86": "x", "x64": "x",
    "arm32": "arm", "arm64": "arm",
    "mips32": "mips", "mips64": "mips",
    "ppc32": "ppc", "ppc64": "ppc",
    "riscv32": "riscv", "riscv64": "riscv",
}


def get_provider(platform: Platform, backend: str = "angr") -> ArchitectureProvider:
    if platform in ("x86", "x64"):
        if backend == "ghidra":
            from tokenizer.arch.x86.ghidra.provider import X86GhidraProvider

            return X86GhidraProvider(platform)
        from tokenizer.arch.x86.angr.provider import X86AngrProvider

        return X86AngrProvider(platform)
    elif platform in ("arm32", "arm64"):
        from tokenizer.arch.arm32.provider import ARM32Provider

        return ARM32Provider(platform)
    elif platform in ("mips32", "mips64"):
        from tokenizer.arch.mips.provider import MIPSProvider

        return MIPSProvider(platform)
    elif platform in ("ppc32", "ppc64"):
        from tokenizer.arch.ppc.provider import PPCProvider

        return PPCProvider(platform)
    elif platform in ("riscv32", "riscv64"):
        from tokenizer.arch.riscv.provider import RISCVProvider

        return RISCVProvider(platform)
    else:
        raise ValueError(f"Unsupported platform: {platform}")
