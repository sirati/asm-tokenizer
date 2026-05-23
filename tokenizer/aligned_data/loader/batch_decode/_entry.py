"""Top-level ``batch_decode`` entry point — wires the four stages into the
end-to-end public API.

This module is the Phase-4 wiring point: once stages 1..4 are implemented in
their respective stub modules, the body of :func:`batch_decode` becomes a
straightforward composition (``walk_sections`` → ``predict_lengths`` →
``build_bulk_bytes`` → ``assemble_batch``). Until then, the body raises
``NotImplementedError`` — the signature already locks in the public API
surface that downstream consumers (training loop, smoke tests) will call.

Default values match the plan's D5 + D6:
  - ``variant_padding=VariantPadding.PAD_NULL`` — short sections pad with
    all-null-content rows (recommended default).
  - ``inlined_equivalent_call_targets_only=False``, ``include_fid_sidecar=False``,
    ``keep_intermediate=False`` — minimal output by default.

Runtime-default import note: the ``VariantPadding.PAD_NULL`` default below
needs the enum class at import time. ``_types`` is owned by a sibling Phase-0b
subagent; during Phase-0c isolated validation that file may not exist yet, in
which case the import below falls back to a ``None`` sentinel and the default
becomes ``None``. The Phase-4 implementation either inlines the real default
after ``_types`` is unconditionally present (post-merge), or detects the
sentinel and substitutes ``VariantPadding.PAD_NULL`` internally. Either way,
the body raises ``NotImplementedError`` until Phase 4 wires it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import numpy as np

    from ..session import BinarySession
    from ._types import (
        BatchDecodeResult,
        SectionPointerSpec,
        VariantPadding,
    )

# Best-effort runtime import for the default value. See module docstring for
# the rationale — Phase 0c does not depend on Phase 0b's ``_types.py`` to
# import cleanly in isolation.
try:
    from ._types import VariantPadding as _VariantPaddingRuntime  # type: ignore[no-redef]

    _DEFAULT_VARIANT_PADDING: object = _VariantPaddingRuntime.PAD_NULL
except ImportError:  # pragma: no cover — only hit during Phase-0c isolated validation
    _DEFAULT_VARIANT_PADDING = None


def batch_decode(
    session: "BinarySession",
    section_pointers: "List[SectionPointerSpec]",
    *,
    num_variants_per_section: int,
    context_len: int,
    max_depth: int,
    variant_padding: "VariantPadding" = _DEFAULT_VARIANT_PADDING,  # type: ignore[assignment]
    inlined_equivalent_call_targets_only: bool = False,
    include_fid_sidecar: bool = False,
    keep_intermediate: bool = False,
    rng: "Optional[np.random.Generator]" = None,
) -> "BatchDecodeResult":
    """End-to-end batch decode: stage 1 → 2 → 3 → 4. See ``batch_decode_plan.md``."""

    raise NotImplementedError(
        "Phase 4 wires this once all stage stubs are filled in."
    )
