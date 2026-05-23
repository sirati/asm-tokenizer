"""Top-level ``batch_decode`` entry point -- wires the four stages into the
end-to-end public API.

Composition: :func:`walk_sections` (stage 1) -> :func:`predict_lengths`
(stage 2) -> :func:`build_bulk_bytes` (stage 3) -> :func:`assemble_batch`
(stage 4). Each stage owns ONE concern; this module owns the linear
threading + the public-API surface (default values, RNG defaulting).

Default values match the plan's D5 + D6:

* ``variant_padding=VariantPadding.PAD_NULL`` -- short sections pad with
  all-null-content rows (recommended default).
* ``inlined_equivalent_call_targets_only=False``,
  ``include_fid_sidecar=False``, ``keep_intermediate=False`` -- minimal
  output by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ._assemble import assemble_batch
from ._bulk_bytes import build_bulk_bytes
from ._length_predict import predict_lengths
from ._section_walk import walk_sections
from ._types import BatchDecodeResult, SectionPointerSpec, VariantPadding

if TYPE_CHECKING:
    from ..session import BinarySession


__all__ = ["batch_decode"]


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
    """End-to-end batch decode: stage 1 -> 2 -> 3 -> 4.

    See ``batch_decode_plan.md`` for the full pipeline design; this
    function is purely the linear composition of the four single-concern
    stage entry points.

    Parameters
    ----------
    session:
        :class:`BinarySession` whose per-arm loaders supply the
        section + function bodies. The session is the boundary across
        which raw-data access happens; all four stages read THROUGH it.
    section_pointers:
        List of :class:`SectionPointerSpec` -- one ``(arm, idx)`` per
        section to include in the batch.
    num_variants_per_section:
        Variant-sampling count per section. Combined with
        ``variant_padding`` to compute the batch's ``batch_size`` at
        stage 1.
    context_len:
        Per-row token budget (column count of the output tensor).
    max_depth:
        Stage-1 splice-tree DFS depth cap (inlining-depth bound).
    variant_padding:
        :class:`VariantPadding` policy for the variant axis (plan D6 +
        ALG-10).
    inlined_equivalent_call_targets_only:
        When True, stage 1's callee walk includes only call targets
        whose ``is_matched`` flag is True (i.e. only inlining-equivalent
        callees). Default False -- include every call target.
    include_fid_sidecar:
        When True, stage 4 builds the optional
        ``(fid_sidecar, fid_row_offsets)`` pair (plan D5).
    keep_intermediate:
        When True, the finalised :class:`Stage3Batch` is carried on the
        result's :attr:`BatchDecodeResult.intermediate` field.
    rng:
        :class:`numpy.random.Generator` for stage 1's variant sampling.
        Defaults to a fresh :func:`numpy.random.default_rng` -- i.e.
        non-reproducible. Pass an explicit generator for deterministic
        sampling.
    """
    if rng is None:
        rng = np.random.default_rng()

    stage1 = walk_sections(
        session,
        section_pointers,
        num_variants_per_section=num_variants_per_section,
        max_depth=max_depth,
        variant_padding=variant_padding,
        inlined_equivalent_call_targets_only=inlined_equivalent_call_targets_only,
        rng=rng,
    )
    stage2 = predict_lengths(stage1, context_len=context_len)
    stage3 = build_bulk_bytes(stage2)
    return assemble_batch(
        stage3,
        context_len=context_len,
        include_fid_sidecar=include_fid_sidecar,
        keep_intermediate=keep_intermediate,
    )
