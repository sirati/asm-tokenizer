"""Stage 1 stub — section pointer resolution + RNG variant sampling + recursive
DFS callee discovery + per-variant raw-data load.

This module is the Phase-1 entry point of the batch-decode pipeline. Its job:

1. Build ``batch_idx_to_section_variant`` per ALG-10's policy table — this
   determines which ``(section_idx, variant_idx)`` pairs populate each row of
   the final output, including any padding rows.
2. Resolve every supplied section pointer through ``BinarySession`` and sample
   variant indices via ``_select_variant_indices``.
3. Per sampled variant, load the root function's ``FunctionData`` and build an
   ``InlineDecodeState`` (``format_version=1`` — mandatory per the unified-vocab
   contract).
4. DFS into ``call_targets_section`` up to ``max_depth``; load each reachable
   inlined callee, recording its ``encounter_category`` (LOCAL_FUNC for root +
   LOCAL-inlined; PLT_FUNC for PLT-inlined; EXT_FUNC is never inlined). DAG
   semantics — visited set keyed on ``(arm, section_offset)``, popped on
   backtrack.
5. Apply ``inlined_equivalent_call_targets_only`` filter when ``True``: skip
   callees where ALL or NONE of the parent's variants called this target (only
   inline when SOME but not ALL did).

Body is intentionally a ``NotImplementedError`` — Phase 1 subagents fill this
in. See ``batch_decode_plan.md`` section ``## Stages — algorithm sketch`` ‒
``Stage 1: section walk + raw-data load`` for the full algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import numpy as np

    from ..session import BinarySession
    from ._types import (
        SectionPointerSpec,
        Stage1Batch,
        VariantPadding,
    )


def walk_sections(
    session: "BinarySession",
    section_pointers: "List[SectionPointerSpec]",
    *,
    num_variants_per_section: int,
    max_depth: int,
    variant_padding: "VariantPadding",
    inlined_equivalent_call_targets_only: bool,
    rng: "np.random.Generator",
) -> "Stage1Batch":
    """Stage 1: section pointer resolution + RNG variant sampling + DFS callee
    discovery + per-variant raw-data load.

    Produces ``Stage1Batch`` with the 4-level hierarchy fully populated and
    ``batch_idx_to_section_variant`` computed per the ``VariantPadding`` policy
    (ALG-10).

    Per the plan:
      - Top-level mapping per ALG-10 → ``u32[batch_size, 2]`` with sentinel
        ``(UINT32_MAX, UINT32_MAX)`` for padding rows.
      - Per section pointer (level 2): resolve via the session's matched /
        unmatched loaders, sample variant indices via
        ``_select_variant_indices``.
      - Per sampled variant (level 3): load the root function +
        ``InlineDecodeState(format_version=1)``; root becomes level-4 index 0
        in this variant's ``call_targets`` list.
      - Per recursive call target (level 4): DFS up to ``max_depth``; record
        ``parent_call_target_index`` + ``encounter_category`` per call_target.
    """

    raise NotImplementedError(
        "Phase 1 — see batch_decode_plan.md '## Stages — algorithm sketch' "
        "Stage 1 + ALG-10."
    )
