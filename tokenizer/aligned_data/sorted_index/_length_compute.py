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
  Stage 1 + Stage 2 walk. Chunks the populated sections, runs ONE
  ``walk_sections`` + ``predict_lengths`` per chunk under
  :attr:`VariantPadding.RAGGED` + ``num_variants_per_section =
  LARGE_VARIANT_CAP``, then collapses each section's surviving-token
  counts into the per-mode result arrays via
  :meth:`LengthReduction.reduce`.

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
from typing import Dict, List

import numpy as np

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.loader.batch_decode._length_predict import (
    predict_lengths,
)
from tokenizer.aligned_data.loader.batch_decode._section_walk import (
    walk_sections,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.matched_sections_bin import iter_sections_bin

from ._types import LengthReduction


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
    """
    base_path = Path(base_path)
    matched_index = base_path / f"{binary_name}_index.bin"
    pair = read_csv_section_index_arrays(matched_index)
    if pair is None:
        # No matched arm. Return empty array; downstream consumers see
        # zero-length matched index and stamp nothing.
        return np.zeros(0, dtype=np.uint32)
    matched_bin_starts, _matched_bin_lengths = pair
    num_matched = len(matched_bin_starts)
    if num_matched == 0:
        return np.zeros(0, dtype=np.uint32)

    sections_path = base_path / f"{binary_name}_sections.bin"
    counts = np.zeros(num_matched, dtype=np.uint32)
    for i, section in enumerate(iter_sections_bin(sections_path)):
        if i >= num_matched:
            break
        counts[i] = len(section.variants)
    return counts


def compute_reduced_lengths(
    session: BinarySession,
    *,
    num_sections: int,
    section_variant_counts: np.ndarray,
    depth: int,
    reductions: List[LengthReduction],
) -> Dict[LengthReduction, np.ndarray]:
    """Per-mode reduced per-section length, computed in ONE Stage 1+2 walk.

    Returns ``{reduction -> u32[num_sections]}``. Each output array's
    ``[i]`` is the reduced key length for matched section ``i``.

    0-variant sections (``section_variant_counts[i] == 0``) are STAMPED
    DIRECTLY with 0 in every output array and excluded from the Stage
    1+2 walk -- :func:`_select_variant_indices` raises on
    ``n_variants <= 0`` (plan audit C1).

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
    num_sections
        Number of MATCHED sections in the binary. Must equal
        ``section_variant_counts.size``.
    section_variant_counts
        ``u32[num_sections]`` per-section variant count, as produced
        by :func:`_count_variants_per_section`.
    depth
        Maximum splice depth fed to :func:`walk_sections` (the index's
        ``_d<NNN>`` filename tag).
    reductions
        The :class:`LengthReduction` modes to compute. The walk
        executes ONCE for all modes (the cost-amortising property
        named in the plan); each mode's result array is collapsed from
        the same per-section variant-length vector.

    Returns
    -------
    Dict[LengthReduction, np.ndarray]
        One ``u32[num_sections]`` array per requested reduction. Key
        identity matches the input ``reductions`` list -- callers may
        index by their own :class:`LengthReduction` instances.

    Raises
    ------
    AssertionError
        If Stage 2's cutoff fires on any variant (per plan D-2.2).
        The message identifies the offending section + variant index
        and points at :data:`LARGE_CONTEXT_LEN`.
    """
    if section_variant_counts.shape != (num_sections,):
        raise ValueError(
            f"section_variant_counts.shape={section_variant_counts.shape!r} "
            f"does not match num_sections={num_sections}"
        )

    # Preallocate per-mode result arrays; defaults are 0 so 0-variant
    # sections need no further write.
    results: Dict[LengthReduction, np.ndarray] = {
        red: np.zeros(num_sections, dtype=np.uint32) for red in reductions
    }

    # Pre-filter 0-variant sections (plan audit C1 fix).
    populated_idx = np.nonzero(section_variant_counts > 0)[0]
    if populated_idx.size == 0:
        # Whole binary is 0-variant; every result stays zero.
        return results

    # Deterministic but ZERO EFFECT: under LARGE_VARIANT_CAP every
    # variant is selected without invoking the RNG (plan D3).
    rng = np.random.default_rng(0)

    for chunk_start in range(0, populated_idx.size, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, populated_idx.size)
        chunk_idxs = populated_idx[chunk_start:chunk_end]
        section_pointers = [
            SectionPointerSpec(arm=SectionKind.MATCHED, idx=int(i))
            for i in chunk_idxs
        ]
        stage1 = walk_sections(
            session,
            section_pointers,
            num_variants_per_section=LARGE_VARIANT_CAP,
            max_depth=depth,
            variant_padding=VariantPadding.RAGGED,
            inlined_equivalent_call_targets_only=False,
            rng=rng,
        )
        stage2 = predict_lengths(stage1, context_len=LARGE_CONTEXT_LEN)

        # ``stage2.sections`` is parallel to ``stage1.sections`` which
        # is parallel to ``section_pointers`` (same order).
        for chunk_offset, stage2_section in enumerate(stage2.sections):
            global_idx = int(chunk_idxs[chunk_offset])
            n_variants = len(stage2_section.variants)
            if n_variants == 0:
                # Belt-and-braces. The 0-variant pre-filter above
                # already excluded this index; keep the zero default.
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

            variant_lengths = np.fromiter(
                (
                    stage2_variant.total_surviving_token_count
                    for stage2_variant in stage2_section.variants
                ),
                dtype=np.uint32,
                count=n_variants,
            )
            for red in reductions:
                results[red][global_idx] = np.uint32(red.reduce(variant_lengths))

    return results
