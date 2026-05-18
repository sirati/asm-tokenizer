"""Test doubles used across the variant_tokens unit-test suite.

These are intentionally tiny: the production code under test only
calls ``vocab.get_token_id(s)``, ``vocab.get_token_str(id)``, and
reads ``extra_metadata`` / canonical-4 fields off the version-info
shape. A real ``VocabularyManager`` would pull in the token-class
dispatch machinery (Batch 1B, not yet merged in this worktree).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


class FakeVocab:
    """Minimal ``get_token_id`` / ``get_token_str`` stand-in.

    Tracks two-way string <-> id mappings just like
    ``VocabularyManager`` does, but skips token-class registration,
    digit reservation, and platform tracking — none of which the
    encoder/decoder needs. ``register(token)`` is idempotent and
    assigns sequential IDs starting at ``base_id``.
    """

    def __init__(self, base_id: int = 256) -> None:
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._next_id = base_id

    def register(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        token_id = self._next_id
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        self._next_id += 1
        return token_id

    def register_at(self, token: str, token_id: int) -> int:
        """Force a specific ID — used by the uint16-overflow test."""
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        if token_id >= self._next_id:
            self._next_id = token_id + 1
        return token_id

    def get_token_id(self, token: str) -> int:
        return self._token_to_id.get(token, -1)

    def get_token_str(self, token_id: int) -> str:
        return self._id_to_token.get(token_id, "")


@dataclass
class FakeVersionInfo:
    """Duck-typed ``BinaryVersionInfo`` for encoder tests.

    Field names match ``tokenizer.memmap_builder.builder.BinaryVersionInfo``
    (``compilerversion``, not ``compiler_version``). The encoder also
    accepts the ``variant_info.VariantInfo`` shape via ``getattr``
    fallback — covered in a dedicated test below.
    """

    arch: str = "x86_64"
    compiler: str = "gcc"
    compilerversion: str = "13.2.0"
    opt: str = "-O2"
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


def make_vocab_with(version_info: Any, base_id: int = 256) -> FakeVocab:
    """Register every axis string for ``version_info`` into a fresh
    fake vocab — convenience for the round-trip tests."""
    from tokenizer.variant_tokens.prefixes import build_axis_strings

    vocab = FakeVocab(base_id=base_id)
    for token in build_axis_strings(version_info):
        vocab.register(token)
    return vocab
