"""Pytest sys.path bootstrap for the tokenizer-package tests.

The asm-tokenizer codebase is a flat package tree (``tokenizer/``,
``shared/``, ``dynrunner/``, ...) at the repo root without an
installable ``pyproject.toml``. ``python -m pytest`` from any cwd
other than the repo root needs explicit injection.

Kept minimal (no provider fixtures, no disassembler imports) so the
fast unit tests under ``tokenizer/tests/`` stay fast — the heavy
provider fixtures live in the top-level ``tests/conftest.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
