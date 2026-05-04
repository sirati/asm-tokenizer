from typing import Literal

from tokenizer.arch import Platform as _CanonicalPlatform

# Vocab-unifier-local Platform alias: the canonical tokenizer arch list
# plus the synthetic "unified" sentinel used for already-unified vocab
# files. Pre-existing "arm" / missing-mips32-ppc-riscv versions of this
# Literal were stale relative to `tokenizer.arch.Platform`; they made
# the loader's `startswith`-based filename detection mismatch on every
# arch the canonical list calls out (e.g. arm64 -> "arm", mips32 ->
# unmatched assertion).
Platform = _CanonicalPlatform | Literal["unified"]
