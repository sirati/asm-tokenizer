from typing import Literal

from tokenizer.arch.provider import ArchitectureProvider


def get_provider(platform: Literal["x86", "x64", "arm32", "arm64"]) -> ArchitectureProvider:
    if platform in ("x86", "x64"):
        from tokenizer.arch.x86.provider import X86Provider

        return X86Provider(platform)
    elif platform == "arm32":
        from tokenizer.arch.arm32.provider import ARM32Provider

        return ARM32Provider()
    else:
        raise ValueError(f"Unsupported platform: {platform}")
