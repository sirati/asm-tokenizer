"""Batch-vectorized v2 dataloader pipeline (4 stages).

Public re-exports per the plan's "Module layout" table:

- :data:`BatchDecodeResult` — the user-facing flat-tensor result of
  :func:`batch_decode`.
- :func:`batch_decode` — end-to-end batch decode entry point.
- :data:`VariantPadding` — runtime enum controlling how short sections pad
  into the linear batch layout.
- :data:`SectionPointerSpec` — typed ``(arm, idx)`` section pointer input.

The dataclass / enum definitions live in ``_types`` (owned by the Phase-0b
subagent). Best-effort re-export below: if ``_types`` is not yet on disk
during isolated Phase-0c validation, the module imports cleanly but exposes
only the entry-point stub. Once ``_types`` lands in the parent merge, the
re-exports below resolve and ``__all__`` lists the full public surface.
"""

from __future__ import annotations

from ._entry import batch_decode

__all__ = ["batch_decode"]

try:
    from ._types import (  # noqa: F401  — re-export
        BatchDecodeResult,
        SectionPointerSpec,
        VariantPadding,
    )
except ImportError:  # pragma: no cover — Phase-0c isolated validation only
    pass
else:
    __all__ += ["BatchDecodeResult", "SectionPointerSpec", "VariantPadding"]
