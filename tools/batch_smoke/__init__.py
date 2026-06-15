"""Internal helpers for :mod:`tools.run_batch_smoke`.

Single concern per submodule:

* :mod:`._metrics`   -- per-session :func:`batch_decode` invocation and
  shape capture.
* :mod:`._aggregate` -- corpus-wide rollup + git-tip helper.

The per-binary memmap-directory scan + selection
(:func:`discover_binaries` / :func:`filter_binaries`) was relocated to
:mod:`tokenizer.aligned_data.binary_discovery` so it ships in the
production image (``tools`` is not packaged); it is re-exported here for
the smoke driver's import surface.

The CLI surface (argparse, JSON writing) lives in
:mod:`tools.run_batch_smoke` so the public driver stays a single file.
"""

from tokenizer.aligned_data.binary_discovery import (
    discover_binaries,
    filter_binaries,
)

from ._aggregate import aggregate, git_tip
from ._metrics import collect_session_metrics, matched_section_pointers

__all__ = [
    "aggregate",
    "collect_session_metrics",
    "discover_binaries",
    "filter_binaries",
    "git_tip",
    "matched_section_pointers",
]
