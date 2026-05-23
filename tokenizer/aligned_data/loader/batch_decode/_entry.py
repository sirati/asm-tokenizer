"""Top-level ``batch_decode`` entry point -- wires the four stages into the
end-to-end public API.

Once each stage's stub module is implemented, the body of :func:`batch_decode`
becomes a straightforward composition (``walk_sections`` -> ``predict_lengths``
-> ``build_bulk_bytes`` -> ``assemble_batch``). Until then, the body raises
``NotImplementedError`` -- the signature already locks in the public API
surface that downstream consumers (training loop, smoke tests) will call.

Default values match the plan's D5 + D6:
  - ``variant_padding=VariantPadding.PAD_NULL`` -- short sections pad with
    all-null-content rows (recommended default).
  - ``inlined_equivalent_call_targets_only=False``, ``include_fid_sidecar=False``,
    ``keep_intermediate=False`` -- minimal output by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from ._types import BatchDecodeResult, SectionPointerSpec, VariantPadding

if TYPE_CHECKING:
    import numpy as np

    from ..session import BinarySession


def batch_decode(
    session: "BinarySession",
    section_pointers: List[SectionPointerSpec],
    *,
    num_variants_per_section: int,
    context_len: int,
    max_depth: int,
    variant_padding: VariantPadding = VariantPadding.PAD_NULL,
    inlined_equivalent_call_targets_only: bool = False,
    include_fid_sidecar: bool = False,
    keep_intermediate: bool = False,
    rng: "Optional[np.random.Generator]" = None,
) -> BatchDecodeResult:
    """End-to-end batch decode: stage 1 -> 2 -> 3 -> 4. See ``batch_decode_plan.md``."""

    raise NotImplementedError(
        "Stage implementations not yet wired together."
    )
