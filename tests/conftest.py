"""Pytest configuration for the owned-view conformance suite.

Session-scoped fixtures open each provider once on a known x86_64 fixture
binary and share the CFG across all tests. ARM-specific tests open their
own arm32 provider on demand via the ``arm32_*_provider`` fixtures (also
session-scoped) so the build_cfg cost is paid once per arch per provider.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer.disasm import DisassemblyProvider, get_disassembly_provider


X64_FIXTURE = Path("/home/sirati/devel/python/asm-tokenizer/src/zlib/x64-gcc-7-Os_minigzip")
ARM32_FIXTURE = Path("/home/sirati/devel/python/asm-tokenizer/src/zlib/arm32-gcc-7-Os_minigzip")


def _build_provider(backend: str, binary: Path) -> DisassemblyProvider:
    provider = get_disassembly_provider(backend, binary)
    provider.build_cfg()
    return provider


# ---- x64 (default) provider fixtures --------------------------------------
@pytest.fixture(scope="session")
def ghidra_provider() -> DisassemblyProvider:
    provider = _build_provider("ghidra", X64_FIXTURE)
    yield provider
    provider.close()


@pytest.fixture(scope="session")
def angr_provider() -> DisassemblyProvider:
    provider = _build_provider("angr", X64_FIXTURE)
    yield provider
    provider.close()


# ---- arm32 provider fixtures (for ARM-only tests) -------------------------
@pytest.fixture(scope="session")
def arm32_ghidra_provider() -> DisassemblyProvider:
    provider = _build_provider("ghidra", ARM32_FIXTURE)
    yield provider
    provider.close()


@pytest.fixture(scope="session")
def arm32_angr_provider() -> DisassemblyProvider:
    provider = _build_provider("angr", ARM32_FIXTURE)
    yield provider
    provider.close()


# ---- Parametrize helpers --------------------------------------------------
# Each test selects its own provider fixture via indirect parametrization;
# ``provider_name`` is the id and dispatches to the matching session fixture.
@pytest.fixture
def provider(request) -> DisassemblyProvider:
    """Resolve the ``provider_name`` parametrize id to a session fixture."""
    name = request.param
    if name == "ghidra":
        return request.getfixturevalue("ghidra_provider")
    if name == "angr":
        return request.getfixturevalue("angr_provider")
    raise ValueError(f"Unknown provider name: {name}")


@pytest.fixture
def arm32_provider(request) -> DisassemblyProvider:
    name = request.param
    if name == "ghidra":
        return request.getfixturevalue("arm32_ghidra_provider")
    if name == "angr":
        return request.getfixturevalue("arm32_angr_provider")
    raise ValueError(f"Unknown provider name: {name}")
