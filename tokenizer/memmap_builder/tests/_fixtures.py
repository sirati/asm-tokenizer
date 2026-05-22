"""Shared fixtures for memmap_builder tests.

Lives in the test tree (not under ``conftest.py``) so the helpers are
explicit imports — same convention the aligned_data loader tests use
for their corpus-builder fixtures.
"""

from __future__ import annotations


class StubVariants:
    """Bare ``.ref(vkey)`` + ``.byte_offset(vkey)`` registry.

    The hex string and the integer come from the same deterministic
    per-vkey counter so the CSV cell and BIN field stay in sync. No
    unified-vocab dependency — keeps the test focused on the BIN
    contract.

    Single source of truth for tests that exercise the
    :mod:`tokenizer.memmap_builder._pass2` walkers without standing up
    a full :class:`VariantRegistry`.
    """

    def __init__(self) -> None:
        self._slots: dict = {}
        self._next = 0x10

    def _ensure(self, vkey) -> int:
        if vkey not in self._slots:
            self._slots[vkey] = self._next
            self._next += 0x10
        return self._slots[vkey]

    def ref(self, vkey) -> str:
        return f"{self._ensure(vkey):x}"

    def byte_offset(self, vkey) -> int:
        return self._ensure(vkey)
