"""Test-harness probe for disassembler-provider availability.

Single concern: determine once whether the angr disassembly backend can
be imported in this environment, and expose the pytest skip surface that
angr-only tests + future cross-provider tests build on.

Ghidra is the production default; angr is best-effort and may be absent
(e.g. its transitive ``networkx`` dependency is missing under ``nix
develop``). ``find_spec("angr")`` alone is insufficient: angr itself
resolves -- it is the transitive dependency that fails -- so the guarded
import is the only reliable probe, and this module is the ONE place it
lives.
"""

from __future__ import annotations

import importlib

import pytest

try:  # the guarded probe -- catches the transitive networkx ModuleNotFoundError
    importlib.import_module("tokenizer.disasm.angr_provider")
    HAS_ANGR = True
except ImportError:
    HAS_ANGR = False


requires_angr = pytest.mark.skipif(
    not HAS_ANGR,
    reason="angr disassembler backend unavailable (e.g. networkx missing in this env)",
)

# Sanctioned mechanism for FUTURE cross-provider tests (the "two
# variants" idiom): parametrize over both providers, with the angr leg
# automatically skipped where the backend is unavailable.
PROVIDER_PARAMS = ["ghidra", pytest.param("angr", marks=requires_angr)]


@pytest.fixture(params=PROVIDER_PARAMS)
def provider(request: pytest.FixtureRequest) -> str:
    """Yield each available provider name in turn for cross-provider tests."""
    return request.param
