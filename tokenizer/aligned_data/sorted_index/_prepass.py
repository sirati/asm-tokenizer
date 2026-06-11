"""Matched-section variant pre-pass (plan ALG-7, extended).

Single concern: ONE read of ``<binary>_sections.bin`` that recovers,
per matched section, the data needed to (a) pre-filter the Stage 1+2
walk and (b) drive the top-level duplicate / minimum-variant logic --
the per-section variant count AND the per-section variant data-bin
pointers (``data_offset_shifted``). Both come from the same parsed
:class:`VariantBlock` records, so reading them together avoids a second
pass over the catalog (no parallel cache: the pointers ARE the parsed
bytes, surfaced once for the downstream consumers that the no-reparse
rule otherwise forces to re-walk).

Boundary contract (the design-first sentence):

  *Given the memmap directory + a binary name, return a
  :class:`SectionVariantInfo` carrying the per-section top-level
  variant counts and the per-section variant ``data_offset_shifted``
  vectors -- the single source the 0-variant pre-filter, the
  duplicate-aware reduction grouping, and the minimum-variant gate all
  read from.*

``data_offset_shifted`` equality is the "same content" relation: two
variants of one section sharing a ``data_offset_shifted`` point at the
same ``_data.bin`` record (identical body tokens), which is exactly the
top-level duplicate relation the feature defines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.matched_sections_bin import iter_sections_bin


__all__ = ["SectionVariantInfo", "read_section_variant_info"]


@dataclass(frozen=True)
class SectionVariantInfo:
    """Per matched-section top-level variant metadata from one BIN pass.

    Parallel-indexed by matched-section index (the index space
    :meth:`BinarySession.load_matched` uses). ``counts[i]`` is the
    top-level variant count of section ``i``; ``data_pointers[i]`` is
    the ``u32`` vector of that section's variant ``data_offset_shifted``
    values, in on-disk variant order (length ``counts[i]``).
    """

    counts: np.ndarray
    """``u32[num_matched_sections]`` -- top-level variant count per
    section."""

    data_pointers: List[np.ndarray]
    """Per-section ``u32`` ``data_offset_shifted`` vectors;
    ``len(data_pointers) == counts.size`` and
    ``data_pointers[i].size == counts[i]``."""

    def unique_count(self, section_idx: int) -> int:
        """Distinct ``data_offset_shifted`` values in section ``section_idx``.

        The count of top-level duplicate-GROUPS -- i.e. the number of
        UNIQUE variants the ``--min-variants-unique`` gate measures.
        """
        return int(np.unique(self.data_pointers[section_idx]).size)


def _empty() -> SectionVariantInfo:
    return SectionVariantInfo(
        counts=np.zeros(0, dtype=np.uint32),
        data_pointers=[],
    )


def read_section_variant_info(
    base_path: Path,
    binary_name: str,
) -> SectionVariantInfo:
    """Read per-matched-section variant counts + data pointers in one pass.

    Reads ``<binary>_sections.bin`` via :func:`iter_sections_bin`,
    bounded to the matched region recovered from
    ``<binary>_index.bin`` (the matched-arm locator). Index ``i``
    corresponds to the i-th MATCHED section in iteration order, matching
    :meth:`BinarySession.load_matched`'s index space.

    Returns an empty :class:`SectionVariantInfo` when the binary has no
    matched arm.
    """
    base_path = Path(base_path)
    pair = read_csv_section_index_arrays(base_path / f"{binary_name}_index.bin")
    if pair is None:
        return _empty()
    matched_bin_starts, _matched_bin_lengths = pair
    num_matched = len(matched_bin_starts)
    if num_matched == 0:
        return _empty()

    sections_path = base_path / f"{binary_name}_sections.bin"
    counts = np.zeros(num_matched, dtype=np.uint32)
    data_pointers: List[np.ndarray] = []
    for i, section in enumerate(iter_sections_bin(sections_path)):
        if i >= num_matched:
            break
        counts[i] = len(section.variants)
        data_pointers.append(
            np.fromiter(
                (v.data_offset_shifted for v in section.variants),
                dtype=np.uint32,
                count=len(section.variants),
            )
        )
    return SectionVariantInfo(counts=counts, data_pointers=data_pointers)
