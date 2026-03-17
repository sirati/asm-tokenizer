from typing import Literal

from tokenizer.arch.provider import ArchitectureProvider

Platform = Literal["x86", "x64", "arm32", "arm64", "mips", "mips64", "ppc", "ppc64", "riscv32", "riscv64"]


def get_provider(platform: Platform) -> ArchitectureProvider:
    if platform in ("x86", "x64"):
        from tokenizer.arch.x86.provider import X86Provider

        return X86Provider(platform)
    elif platform in ("arm32", "arm64"):
        from tokenizer.arch.arm32.provider import ARM32Provider

        return ARM32Provider(platform)
    elif platform in ("mips", "mips64"):
        from tokenizer.arch.mips.provider import MIPSProvider

        return MIPSProvider(platform)
    elif platform in ("ppc", "ppc64"):
        from tokenizer.arch.ppc.provider import PPCProvider

        return PPCProvider(platform)
    elif platform in ("riscv32", "riscv64"):
        from tokenizer.arch.riscv.provider import RISCVProvider

        return RISCVProvider(platform)
    else:
        raise ValueError(f"Unsupported platform: {platform}")
