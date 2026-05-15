"""Pytest configuration for the test suite.

Two concerns merged here:

1. Import-path bootstrap. The asm-tokenizer codebase is a flat package
   tree (``tokenizer/``, ``shared/``, ``dynrunner/``, ...) at the repo
   root without an installable ``pyproject.toml``. ``python -m tokenizer``
   works from the repo root because the cwd is auto-added to
   ``sys.path``; pytest invocations from any other cwd need explicit
   injection.

2. Session-scoped provider fixtures for the conformance suite. Each
   provider is opened once on a known fixture binary (x64 for default,
   arm32 for ARM-specific tests) and shared across all tests in a
   session.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
