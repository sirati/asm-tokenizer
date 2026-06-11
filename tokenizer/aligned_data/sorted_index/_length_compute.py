"""Per-section depth-N length compute for the matched-arm sorted index.

Single concern: turn a binary session + a list of
:class:`LengthReduction` modes into one ``u32[num_matched_sections]``
per mode, where each entry is the reduced key length for the matched
section at that index.

Two pieces live here:

* :func:`_count_variants_per_section` (plan ALG-7) -- pre-pass over
  ``<binary>_sections.bin`` that counts variants per matched section.
  Required upstream of the Stage 1+2 walk so the walker can pre-filter
  0-variant sections (``_select_variant_indices`` raises on
  ``n_variants <= 0`` -- plan audit C1).

* :func:`compute_reduced_lengths` (plan ALG-1) -- the multi-mode shared
  Stage 1 + Stage 2 walk. Chunks the populated sections, stages every
  chunk's Stage 1 walk onto ONE shared
  :class:`BucketedRunLengthCollector`, then flushes once before
  finalising + running ``predict_lengths`` per chunk under
  :attr:`VariantPadding.RAGGED` + ``num_variants_per_section =
  LARGE_VARIANT_CAP``. Collapses each section's surviving-token
  counts into the per-mode result arrays via
  :meth:`LengthReduction.reduce`. The single-collector lifecycle
  amortises every call_target row's ``run_lengths`` across the entire
  sorted-index build (not just one CHUNK_SIZE-sized chunk).

Boundary contract (the design-first sentence):

  *Given a session + the pre-counted variant counts + a depth + a list
  of reductions, produce one u32 length array per reduction with the
  same length as the matched-arm and the reduced key length for each
  section index.  No I/O.  No CLI parsing.  Stamps 0 for 0-variant
  sections; raises :class:`AssertionError` if Stage 2's cutoff fires
  (under :data:`LARGE_CONTEXT_LEN` this should be impossible -- a fire
  means the chosen ``LARGE_CONTEXT_LEN`` is insufficient for the
  corpus and the index would under-report lengths silently).*
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._length_predict import (
    predict_lengths,
)
from tokenizer.aligned_data.loader.batch_decode._section_walk import (
    PendingStage1Batch,
    finalise_pending_stage1,
    walk_sections,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.decoded._bucketed_run_lengths import (
    BucketedRunLengthCollector,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.session import BinarySession

from ._dedup import PLAIN, DuplicateHandling
from ._gating import VariantGate
from ._prepass import SectionVariantInfo, read_section_variant_info
from ._types import IndexSpec, LengthReduction


__all__ = [
    "CHUNK_SIZE",
    "LARGE_VARIANT_CAP",
    "LARGE_CONTEXT_LEN",
    "_count_variants_per_section",
    "compute_reduced_lengths",
]


#: Number of populated matched sections walked through Stage 1+2 per
#: chunk. Sized to keep peak working-set inside ~200 MB on a typical
#: corpus (plan ALG-1 memory bound: chunk_size * E[variants] *
#: E[expanded_len] * 2 bytes + tree overhead). Smaller chunks pay more
#: Stage 1 setup; larger chunks risk OOM on pathological variants.
CHUNK_SIZE = 64

#: ``num_variants_per_section`` sentinel that asks
#: :func:`_select_variant_indices` to return EVERY variant in encounter
#: order (no sampling, no RNG consumption). Bounded by the BIN's u16
#: variant-count slot (<= 65534) so int32_max is safely above every
#: real corpus.
LARGE_VARIANT_CAP = int(np.iinfo(np.int32).max)

#: Stage 2 cutoff budget chosen so no plausible corpus variant
#: triggers ``cut_call_target_index < len(call_targets)``. The compute
#: ASSERTS no cut fires (plan D-2.2): a fire means the sorted index
#: would under-report a section's true length silently, so the build
#: refuses to proceed.
LARGE_CONTEXT_LEN = 2**30


def _count_variants_per_section(
    base_path: Path,
    binary_name: str,
) -> np.ndarray:
    """Return ``u32[num_matched_sections]`` of per-section variant counts.

    Reads ``<binary>_sections.bin`` via :func:`iter_sections_bin`.
    Index ``i`` corresponds to the i-th MATCHED section in iteration
    order, which equals the index space used by
    :meth:`BinarySession.load_matched` (both are sequential over
    ``matched_index.bin``'s ``bin_starts``).

    .. note::

       The unmatched arm shares the same BIN file -- its sections are
       written immediately after the matched region. The count of
       matched sections is recovered from ``<binary>_matched_index.bin``
       (the matched-arm locator); this function returns ONLY the
       matched-region variant counts.

       :func:`iter_sections_bin` / :func:`parse_section_bin` walk the
       FULL section structure (header + jump-table + call_targets +
       variants region) because section size is variable -- there is
       NO header-only fast-path to skip past a section without
       parsing its body. This pre-pass is therefore O(num_sections *
       small_constant); fast in absolute terms (sections.bin is small,
       megabytes per binary) but not O(num_sections * O(1)).

    This is the counts-only view of :func:`read_section_variant_info`;
    callers that also need the per-variant data-bin pointers (the
    duplicate / minimum-variant feature) consume the richer
    :class:`SectionVariantInfo` directly.
    """
    return read_section_variant_info(base_path, binary_name).counts


def _variant_lengths_at_depth(stage2_variant, depth: int) -> int:
    """Sum the surviving-token counts of one variant's call_targets at depth.

    A call_target belongs to the depth-``depth`` expansion iff its DFS
    ``path_depth`` is ``<= depth`` (the depth-cap makes the depth-``k``
    tree an exact prefix of every deeper walk -- see
    :attr:`Stage1CallTarget.path_depth`). The variant's depth-``depth``
    length is the sum of the surviving-token counts over exactly those
    call_targets. Under :data:`LARGE_CONTEXT_LEN` no cutoff fires, so
    each call_target's ``surviving_token_count`` equals its full length.
    """
    return int(
        sum(
            st2_ct.surviving_token_count
            for st2_ct in stage2_variant.call_targets
            if st2_ct.stage1.path_depth <= depth
        )
    )


def compute_reduced_lengths(
    session: BinarySession,
    *,
    section_info: SectionVariantInfo,
    depths: List[int],
    reductions: List[LengthReduction],
    gate: VariantGate = VariantGate(),
    duplicate_handling: DuplicateHandling = PLAIN,
) -> Dict[IndexSpec, np.ndarray]:
    """Per-(mode, depth) reduced per-section length, from ONE Stage 1+2 walk.

    Returns ``{IndexSpec(reduction, depth) -> u32[num_sections]}``. Each
    output array's ``[i]`` is the reduced key length for matched section
    ``i`` at that ``(reduction, depth)``. The single walk runs at
    ``max(depths)``; every shallower depth is recovered as the exact
    prefix of call_targets whose ``path_depth <= depth`` (no extra
    walks).

    0-variant sections (``section_info.counts[i] == 0``) are STAMPED
    DIRECTLY with 0 in every output array and excluded from the Stage
    1+2 walk -- :func:`_select_variant_indices` raises on
    ``n_variants <= 0`` (plan audit C1).

    Sections failing the top-level minimum-variant ``gate`` are likewise
    stamped 0 across every output -- the same length-0 representation a
    0-variant section takes (a length-0 bucket is never drawn at a real
    training target length, so this excludes the section without
    touching the wire format / filename / reader). The gate inspects
    only top-level (depth-0) variant counts, so a gated-out section is
    excluded uniformly across every ``(reduction, depth)``.

    The walk runs ``num_variants_per_section = LARGE_VARIANT_CAP``
    under :attr:`VariantPadding.RAGGED` so every real variant of every
    populated section gets exactly one Stage 2 row. ``context_len =
    LARGE_CONTEXT_LEN`` guarantees no Stage 2 cutoff fires; a fired
    cutoff raises :class:`AssertionError` (plan D-2.2 -- silent
    under-reporting would be worse than a hard fail at build time).

    Parameters
    ----------
    session
        Open :class:`BinarySession` for the binary; the session's
        memmap handles must be live for the call's duration (Stage 1
        loads variant bodies through it).
    section_info
        :class:`SectionVariantInfo` from :func:`read_section_variant_info`
        -- the per-section top-level variant counts + data-bin pointers.
        ``counts.size`` is the number of matched sections.
    depths
        Splice depths to materialise. One output array per
        ``(reduction, depth)`` pair; the walk's ``max_depth`` is
        ``max(depths)``. Must be non-empty with every entry ``>= 0``.
    reductions
        The :class:`LengthReduction` modes to compute. The walk
        executes ONCE for all (mode, depth) pairs (the cost-amortising
        property named in the plan).
    gate
        Top-level minimum-variant emission gate. Defaults to the
        disabled gate (every section emitted).
    duplicate_handling
        Top-level duplicate strategy. Defaults to :data:`PLAIN` (no
        dedup; byte-identical to the pre-feature reduction).

    Returns
    -------
    Dict[IndexSpec, np.ndarray]
        One ``u32[num_sections]`` array per ``(reduction, depth)`` pair.

    Raises
    ------
    AssertionError
        If Stage 2's cutoff fires on any variant (per plan D-2.2).
        The message identifies the offending section + variant index
        and points at :data:`LARGE_CONTEXT_LEN`.
    ValueError
        If ``depths`` is empty or carries a negative entry.
    """
    if not depths:
        raise ValueError("depths must be a non-empty list")
    if any(d < 0 for d in depths):
        raise ValueError(f"depths must all be >= 0; got {depths!r}")

    counts = section_info.counts
    num_sections = int(counts.size)
    max_depth = max(depths)

    # Preallocate per-(mode, depth) result arrays; defaults are 0 so
    # 0-variant AND gated-out sections need no further write.
    specs = [
        IndexSpec(reduction=red, depth=d) for red in reductions for d in depths
    ]
    results: Dict[IndexSpec, np.ndarray] = {
        spec: np.zeros(num_sections, dtype=np.uint32) for spec in specs
    }

    # Pre-filter 0-variant sections (plan audit C1 fix) AND gated-out
    # sections: both stay at the zero default and are excluded from the
    # walk. The gate reads ONLY top-level counts (depth-independent).
    emitted = np.fromiter(
        (
            counts[i] > 0
            and gate.passes(
                n_total=int(counts[i]),
                n_unique=section_info.unique_count(i),
            )
            for i in range(num_sections)
        ),
        dtype=bool,
        count=num_sections,
    )
    populated_idx = np.nonzero(emitted)[0]
    if populated_idx.size == 0:
        # Nothing emitted; every result stays zero.
        return results

    # Deterministic but ZERO EFFECT: under LARGE_VARIANT_CAP every
    # variant is selected without invoking the RNG (plan D3).
    rng = np.random.default_rng(0)

    # One-collector-per-build contract: every chunk's Stage 1 walk
    # stages onto the same collector; one flush amortises every
    # call_target row's ``run_lengths`` across the WHOLE sorted-index
    # build (not just one CHUNK_SIZE-sized chunk).
    collector = BucketedRunLengthCollector()
    chunk_pendings: List[Tuple[np.ndarray, PendingStage1Batch]] = []
    for chunk_start in range(0, populated_idx.size, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, populated_idx.size)
        chunk_idxs = populated_idx[chunk_start:chunk_end]
        section_pointers = [
            SectionPointerSpec(arm=SectionKind.MATCHED, idx=int(i))
            for i in chunk_idxs
        ]
        chunk_pendings.append((
            chunk_idxs,
            walk_sections(
                session,
                section_pointers,
                num_variants_per_section=LARGE_VARIANT_CAP,
                max_depth=max_depth,
                variant_padding=VariantPadding.RAGGED,
                inlined_equivalent_call_targets_only=False,
                rng=rng,
                collector=collector,
            ),
        ))

    # ONE flush -- one pow2-bucketed 2D run_lengths dispatch per
    # bucket across every chunk's call_target rows.
    runlen_results = collector.flush()

    for chunk_idxs, pending in chunk_pendings:
        stage1 = finalise_pending_stage1(pending, runlen_results)
        stage2 = predict_lengths(stage1, context_len=LARGE_CONTEXT_LEN)

        # ``stage2.sections`` is parallel to ``stage1.sections`` which
        # is parallel to ``section_pointers`` (same order).
        for chunk_offset, stage2_section in enumerate(stage2.sections):
            global_idx = int(chunk_idxs[chunk_offset])
            n_variants = len(stage2_section.variants)
            if n_variants == 0:
                # Belt-and-braces. The pre-filter above already excluded
                # this index; keep the zero default.
                continue

            # Plan D-2.2: assert no cutoff fired. Under
            # LARGE_CONTEXT_LEN this SHOULD never trigger; if it does,
            # the chosen LARGE_CONTEXT_LEN is insufficient for the
            # corpus and the sorted index would silently under-report
            # the section's true length.
            for stage2_variant in stage2_section.variants:
                if stage2_variant.cut_call_target_index != len(
                    stage2_variant.call_targets
                ):
                    raise AssertionError(
                        f"sorted-index length compute: variant cut fired "
                        f"at section_idx={global_idx} "
                        f"variant_idx={stage2_variant.stage1.variant_idx}; "
                        f"raise LARGE_CONTEXT_LEN beyond {LARGE_CONTEXT_LEN}"
                    )

            # Top-level data-bin pointers for this section, parallel to
            # ``stage2_section.variants`` (RAGGED selects every variant
            # in on-disk order, matching the pre-pass's pointer order).
            data_pointers = section_info.data_pointers[global_idx]

            for depth in depths:
                variant_lengths = np.fromiter(
                    (
                        _variant_lengths_at_depth(stage2_variant, depth)
                        for stage2_variant in stage2_section.variants
                    ),
                    dtype=np.uint32,
                    count=n_variants,
                )
                for red in reductions:
                    results[IndexSpec(reduction=red, depth=depth)][
                        global_idx
                    ] = np.uint32(
                        duplicate_handling.reduce_section(
                            red,
                            lengths=variant_lengths,
                            data_pointers=data_pointers,
                        )
                    )

    return results
