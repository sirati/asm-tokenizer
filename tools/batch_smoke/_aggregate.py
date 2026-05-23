"""Corpus-wide rollup + git-tip helper for the batch-decode smoke.

Two single-concern helpers consumed by the driver layer:

* :func:`aggregate` -- sum per-binary shape counts into one corpus-wide
  block.
* :func:`git_tip`   -- short-circuiting ``git rev-parse HEAD`` for the
  results-JSON header.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict


def aggregate(per_binary: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Sum per-binary shape counts into one corpus-wide block.

    The aggregate intentionally drops the per-binary ``tokens_shape``
    + ``wall_seconds``; corpus-wide they're a ``total_tokens`` scalar
    and the outer ``wall_seconds`` covers the run as a whole.
    """
    batch_size = 0
    total_tokens = 0
    total_identity_chunks = 0
    total_number_chunks = 0
    section_count = 0
    for block in per_binary.values():
        batch_size += int(block["batch_size"])
        rows, cols = block["tokens_shape"]
        total_tokens += int(rows) * int(cols)
        total_identity_chunks += int(block["total_identity_chunks"])
        total_number_chunks += int(block["total_number_chunks"])
        section_count += int(block["section_count"])
    return {
        "batch_size": batch_size,
        "total_tokens": total_tokens,
        "total_identity_chunks": total_identity_chunks,
        "total_number_chunks": total_number_chunks,
        "section_count": section_count,
    }


def git_tip(cwd: Path) -> str:
    """Return ``git rev-parse HEAD`` or empty string when git is absent.

    The empty string keeps the JSON shape stable when the script runs
    outside a git checkout (mirrors the legacy splice-smoke helper).
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except FileNotFoundError:
        pass
    return ""
