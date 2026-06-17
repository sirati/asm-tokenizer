"""Import-order regression guard for the realized_lengths / sorted_index /
binary_discovery / loader.vector_batch package cycle.

These packages cross-import:

    realized_lengths._catalog -> sorted_index._prepass (matched-arm reuse)
    sorted_index._collection  -> loader.vector_batch + binary_discovery
    loader.vector_batch       -> realized_lengths (RealizedGeometryReader)
    binary_discovery          -> realized_lengths (ARMS / GEOMETRY_ARMS)

If the realized_lengths -> sorted_index edge is a module-level (eager) import,
importing the *tail* package first leaves it half-initialized and the cycle
raises ``ImportError``. The failure is ORDER-DEPENDENT: a single test process
that imports one of these first masks it for the rest, so each order must run
in a FRESH interpreter.

History: the same class of latent cycle was fixed once for the decode-engine
edge (lazy-import in ``_sampler._engine``) but never guarded, and recurred
(realized_lengths._catalog -> sorted_index, surfaced when build-side discovery
imported realized_lengths first). This test pins every tail-first order.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Every module that, imported FIRST in a clean interpreter, would expose an
# eager edge of the cycle. Add to this list, do not collapse into one process.
_TAIL_FIRST_MODULES = [
    "tokenizer.aligned_data.realized_lengths",
    "tokenizer.aligned_data.realized_lengths._format",
    "tokenizer.aligned_data.realized_lengths._geometry_format",
    "tokenizer.aligned_data.realized_lengths._catalog",
    "tokenizer.aligned_data.binary_discovery",
    "tokenizer.aligned_data.sorted_index._prepass",
    "tokenizer.aligned_data.sorted_index._collection._discovery",
    "tokenizer.aligned_data.sorted_index",
    "tokenizer.aligned_data.loader.vector_batch.session_handles",
    "tokenizer.aligned_data.loader.vector_batch",
]


@pytest.mark.parametrize("module", _TAIL_FIRST_MODULES)
def test_module_imports_first_without_cycle(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"importing {module!r} first triggered a circular import "
        f"(an eager cross-package edge regressed):\n{result.stderr}"
    )
