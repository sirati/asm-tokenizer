"""Per-section depth-N length compute for the matched-arm sorted index.

Single concern: turn the pre-passed section catalog + the ``_data.bin``
byte array into one ``u32[num_matched_sections]`` per ``(reduction,
depth)`` pair, where each entry is the reduced key length for the
matched section at that index.

The heavy lifting is fully vectorized and walk-free:

* :func:`.._graph_lengths.compute_node_lengths` -- per-(section,
  variant) spliced lengths straight from the catalog's splice graph +
  per-unique-record contributing lengths (no token bodies are decoded,
  no Stage 1+2 batches are materialised; peak working-set is a few
  small per-variant columns regardless of corpus size);
* :meth:`DuplicateHandling.reduce_segmented` -- per-section reduction
  of the node lengths across all sections in one call per
  ``(reduction, depth)`` pair.

Boundary contract (the design-first sentence):

  *Given the pre-passed catalog + the data bytes + depths + reductions,
  produce one u32 length array per (reduction, depth) with the reduced
  key length per matched section. No file I/O. No CLI parsing. Stamps
  0 for 0-variant and gated-out sections; raises AssertionError if any
  length reaches the legacy walk's cutoff budget (plan D-2.2 -- the
  index must never silently under-report).*
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ._dedup import PLAIN, DuplicateHandling
from ._gating import VariantGate
from ._graph_lengths import LARGE_CONTEXT_LEN, compute_node_lengths
from ._prepass import SectionVariantInfo
from ._types import IndexSpec, LengthReduction


__all__ = [
    "LARGE_CONTEXT_LEN",
    "compute_reduced_lengths",
]


def compute_reduced_lengths(
    section_info: SectionVariantInfo,
    data_u8: np.ndarray,
    *,
    depths: List[int],
    reductions: List[LengthReduction],
    gate: VariantGate = VariantGate(),
    duplicate_handling: DuplicateHandling = PLAIN,
) -> Dict[IndexSpec, np.ndarray]:
    """Per-(mode, depth) reduced per-section length, walk-free.

    Returns ``{IndexSpec(reduction, depth) -> u32[num_sections]}``;
    each output's ``[i]`` is the reduced key length for matched section
    ``i``. All requested depths come from ONE graph traversal at
    ``max(depths)`` (every shallower depth is an exact prefix of the
    depth recurrence), and every ``(reduction, depth)`` reduction is a
    single segmented numpy call.

    0-variant sections and sections failing the top-level
    minimum-variant ``gate`` are stamped 0 in every output -- the
    length-0 representation is never drawn at a real training target
    length, so gating needs no wire-format change. The gate inspects
    only top-level variant counts (depth-independent).

    Parameters
    ----------
    section_info
        Catalog pre-pass (:func:`._prepass.read_section_variant_info`).
    data_u8
        The matched arm's ``_data.bin`` as a uint8 array; a read-only
        ``np.memmap`` keeps the build's resident set bounded on
        multi-GB corpora (only header bytes + token regions are paged
        in, in bounded chunks).
    depths
        Splice depths to materialise; non-empty, every entry >= 0.
    reductions
        :class:`LengthReduction` modes; one output per (mode, depth).
    gate
        Top-level minimum-variant emission gate (default: disabled).
    duplicate_handling
        Top-level duplicate strategy (default: :data:`PLAIN`).

    Raises
    ------
    AssertionError
        If any spliced length reaches :data:`LARGE_CONTEXT_LEN` (the
        legacy walk's cutoff would have fired -- plan D-2.2).
    ValueError
        If ``depths`` is empty or carries a negative entry.
    """
    if not depths:
        raise ValueError("depths must be a non-empty list")
    if any(d < 0 for d in depths):
        raise ValueError(f"depths must all be >= 0; got {depths!r}")

    cols = section_info.cols
    num_sections = int(cols.n_variants.size)
    specs = [
        IndexSpec(reduction=red, depth=d) for red in reductions for d in depths
    ]
    results: Dict[IndexSpec, np.ndarray] = {
        spec: np.zeros(num_sections, dtype=np.uint32) for spec in specs
    }
    if num_sections == 0 or int(cols.var_n_calls.size) == 0:
        return results

    # 0-variant pre-filter + top-level gate, both vectorized. Gated-out
    # sections keep the zero default in EVERY output; they still
    # participate as splice CALLEES of emitted sections (the walk never
    # gated callees either).
    emitted = (cols.n_variants > 0) & gate.passes_batch(
        n_total=cols.n_variants,
        n_unique=section_info.unique_counts(),
    )

    node_lengths = compute_node_lengths(
        cols, section_info.section_offsets, data_u8, depths
    )

    for depth, lengths in node_lengths.items():
        for red in reductions:
            per_section = duplicate_handling.reduce_segmented(
                red,
                lengths=lengths,
                data_pointers=cols.var_data_offset_shifted,
                seg_offsets=cols.var_offsets,
            )
            per_section[~emitted] = 0
            # < LARGE_CONTEXT_LEN (asserted upstream) so the u32 cast
            # is lossless.
            results[IndexSpec(reduction=red, depth=depth)] = (
                per_section.astype(np.uint32)
            )

    return results
