"""Pytest configuration for the test suite.

Three concerns:

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

3. Custom markers (``slow``) registered up-front so ``pytest -m "slow"``
   selection doesn't trip ``PytestUnknownMarkWarning``. ``slow`` flags
   multi-second tests (e.g. the J.2 multi-binary Ghidra E2E smoke).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from tokenizer.disasm import DisassemblyProvider, get_disassembly_provider


X64_FIXTURE = Path("/home/sirati/devel/python/asm-tokenizer/src/zlib/x64-gcc-7-Os_minigzip")
ARM32_FIXTURE = Path("/home/sirati/devel/python/asm-tokenizer/src/zlib/arm32-gcc-7-Os_minigzip")


def pytest_configure(config):
    """Register custom markers so ``-m slow`` selection doesn't trip
    PytestUnknownMarkWarning. ``slow`` flags multi-second tests (e.g.
    the J.2 multi-binary Ghidra E2E smoke).
    """
    config.addinivalue_line(
        "markers",
        "slow: tests that take more than a few seconds (typically because "
        "they shell out to Ghidra / tokenize a whole binary).",
    )


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


# ---- Standalone-tokenizer subprocess factory ------------------------------
@pytest.fixture
def fresh_ghidra_csv(tmp_path: Path) -> Callable[[str, Path], Path]:
    """Factory that invokes the standalone ``python -m tokenizer --backend
    ghidra --batch ...`` pipeline on a single binary and returns the path
    to the freshly emitted CSV under the test's ``tmp_path``.

    Tests that pin post-refactor wire shapes (rather than the cached
    ``/tmp/asm_smoke/out/`` snapshot, which can pre-date the refactor)
    use this fixture to guarantee they exercise current producer output.
    The cost is a Ghidra spin-up per call (~30-120s); callers should
    carry the ``slow`` marker.
    """
    def _make(binary_name: str, source_root: Path) -> Path:
        queue = tmp_path / "queue.txt"
        queue.write_text(f"{binary_name}\n")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        cmd = [
            sys.executable,
            "-m",
            "tokenizer",
            "--backend", "ghidra",
            "--batch", str(queue),
            "--source", str(source_root),
            "--output", str(out_dir),
            "--platform", "auto",
        ]
        env = dict(os.environ)
        env["ASM_TOKENIZER_FORMAT_VERSION"] = "2"

        proc = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, (
            f"tokenize failed (rc={proc.returncode}) for {binary_name}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        csv_path = out_dir / f"{binary_name}_output.csv"
        assert csv_path.is_file(), f"CSV not emitted at {csv_path}"
        return csv_path

    return _make
