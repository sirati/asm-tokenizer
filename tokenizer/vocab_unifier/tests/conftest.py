"""Local pytest bootstrap so this test file can run from any cwd.

Mirrors the bootstrap in ``tokenizer/variant_tokens/tests/conftest.py``
— asm-tokenizer's flat package tree has no installable
``pyproject.toml``, so pytest needs the repo root injected into
``sys.path`` explicitly when these tests run in isolation (e.g.
``pytest tokenizer/vocab_unifier/tests/`` from a subagent worktree).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
