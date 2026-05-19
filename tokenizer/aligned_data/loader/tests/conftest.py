"""Local pytest bootstrap so this test file runs from any cwd.

The asm-tokenizer flat tree has no installable ``pyproject.toml`` —
pytest needs the repo root injected into ``sys.path`` explicitly when
these tests run in isolation (mirrors the bootstrap in other test
subpackages such as ``tokenizer/variant_tokens/tests/conftest.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
