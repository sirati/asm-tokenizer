"""Matched-section catalog pre-pass for the sorted-index build.

Single concern: ONE columnar read of the matched region of
``<binary>_sections.bin`` (bounded by the ``<binary>_index.bin``
locator), surfacing everything the build consumes downstream -- the
per-section variant counts (0-variant pre-filter + minimum-variant
gate), the per-variant data-bin pointers (duplicate grouping + own
lengths), the call-target tables and per-call entries (the splice
graph) -- as flat numpy columns via
:class:`~tokenizer.aligned_data.matched_sections_columnar.
ColumnarSections`.

Boundary contract (the design-first sentence):

  *Given the memmap directory + a binary name, return a
  :class:`SectionVariantInfo` carrying the columnar matched-section
  catalog + the locator offsets -- the single parsed source every
  downstream sorted-index stage (gating, dedup, graph lengths) reads
  from, so nothing re-parses the BIN.*
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.matched_sections_columnar import (
    ColumnarSections,
    parse_sections_columnar,
)


__all__ = ["SectionVariantInfo", "read_section_variant_info"]


@dataclass(frozen=True)
class SectionVariantInfo:
    """Columnar matched-section catalog + locator offsets, one BIN pass.

    ``cols`` is indexed by matched-section position (the index space
    :meth:`BinarySession.load_matched` uses); ``section_offsets`` is
    the locator's parallel byte-offset column (the values call-target
    ``function_section_ptr`` fields point at).
    """

    cols: ColumnarSections
    section_offsets: np.ndarray
    """``int64[num_matched_sections]`` -- BIN byte offset per section."""

    @property
    def counts(self) -> np.ndarray:
        """``i64[num_matched_sections]`` top-level variant counts."""
        return self.cols.n_variants

    def unique_counts(self) -> np.ndarray:
        """Distinct ``data_offset_shifted`` count per section.

        The number of top-level duplicate-GROUPS -- what the
        ``--min-variants-unique`` gate measures.
        """
        cols = self.cols
        n_sections = cols.n_variants.size
        total = cols.var_data_offset_shifted.size
        if total == 0:
            return np.zeros(n_sections, dtype=np.int64)
        seg = np.repeat(
            np.arange(n_sections, dtype=np.int64), cols.n_variants
        )
        order = np.lexsort((cols.var_data_offset_shifted, seg))
        sp = cols.var_data_offset_shifted[order]
        ss = seg[order]
        first = np.ones(total, dtype=bool)
        first[1:] = (sp[1:] != sp[:-1]) | (ss[1:] != ss[:-1])
        return np.bincount(ss[first], minlength=n_sections)

    def unique_count(self, section_idx: int) -> int:
        """Scalar convenience over :meth:`unique_counts`."""
        lo = int(self.cols.var_offsets[section_idx])
        hi = int(self.cols.var_offsets[section_idx + 1])
        return int(
            np.unique(self.cols.var_data_offset_shifted[lo:hi]).size
        )


def _empty() -> SectionVariantInfo:
    return SectionVariantInfo(
        cols=parse_sections_columnar(
            np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.int64)
        ),
        section_offsets=np.zeros(0, dtype=np.int64),
    )


def read_section_variant_info(
    base_path: Path,
    binary_name: str,
) -> SectionVariantInfo:
    """One columnar read of the matched region of ``sections.bin``.

    Bounded to the matched region recovered from ``<binary>_index.bin``
    (the matched-arm locator); index ``i`` corresponds to the i-th
    MATCHED section, matching :meth:`BinarySession.load_matched`'s
    index space. Returns an empty :class:`SectionVariantInfo` when the
    binary has no matched arm.
    """
    base_path = Path(base_path)
    pair = read_csv_section_index_arrays(base_path / f"{binary_name}_index.bin")
    if pair is None:
        return _empty()
    starts, lengths = pair
    if len(starts) == 0:
        return _empty()
    blob = np.fromfile(
        base_path / f"{binary_name}_sections.bin", dtype=np.uint8
    )
    starts = np.asarray(starts, dtype=np.int64)
    return SectionVariantInfo(
        cols=parse_sections_columnar(blob, starts, lengths),
        section_offsets=starts,
    )
