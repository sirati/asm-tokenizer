"""Read-side catalog<->geometry cross-axis stale-sidecar guard.

Single concern: at the handle-open boundary, assert that an arm's
realized-GEOMETRY sidecar (the ``RealizedGeometryReader`` axes) is
consistent with THAT arm's columnar ``sections.bin`` catalog -- i.e. it
carries exactly one node per catalog variant over exactly the catalog's
sections. This is the READ-side twin of the build-side contract in
:mod:`...sorted_index._builder` (which raises ``ValueError`` when the
realized-length sidecar's ``body_lengths.size`` does not match the
catalog pre-pass's variant count); the geometry reader's own internal
guard only checks its CSR is self-consistent (``csr[-1] ==
body_lengths.size``), never against the catalog it must align with.

A ``_realized.bin`` that is STALE relative to a rebuilt (longer)
``_sections.bin`` passes every existing guard, then the geometry prepass
gathers ``body_axis[node]`` with a flat catalog NODE index that exceeds
the stale axis length -> a bare ``IndexError`` deep in
:func:`..._geometry._flatten_emission`. Detecting the mismatch HERE, at
open, turns that into a loud, actionable error naming the binary, arm,
and both counts.

Boundary contract (the design-first sentence):

  *Given an arm's columnar catalog + its realized-geometry reader (both
  already opened), raise a generator-pointing :class:`ValueError` iff the
  geometry's per-node axis length or section count disagrees with the
  catalog's eager variant total / section count -- a no-op on any
  consistent build.*

The two catalog quantities the guard reads -- ``var_offsets[-1]`` (total
variants) and ``n_variants.size`` (section count) -- are seeded EAGERLY
on both the eager :class:`...matched_sections_columnar.ColumnarSections`
and its lazy twin's section-level skeleton, so the guard needs no heavy
per-section fill.
"""

from __future__ import annotations

from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.realized_lengths import RealizedGeometryReader


__all__ = ["check_geometry_matches_catalog"]


def check_geometry_matches_catalog(
    cols: ColumnarSections,
    geometry: RealizedGeometryReader,
    *,
    binary_name: str,
    arm_name: str,
) -> None:
    """Raise if ``geometry`` is stale relative to ``cols`` for this arm.

    Cross-checks the two axes the geometry prepass indexes by catalog
    position:

    * the per-node body/id/value axis length
      (``geometry.body_lengths.size``) must equal the catalog's total
      variant count (``cols.var_offsets[-1]``); and
    * the geometry's section count (``geometry.n_sections``) must equal
      the catalog's section count (``cols.n_variants.size``).

    A mismatch means the ``_realized.bin`` / ``_realized_index.bin`` pair
    is stale relative to a rebuilt ``_sections.bin`` (or vice versa); the
    raised :class:`ValueError` names the binary, the arm, and both counts
    and points at the generator -- the SAME contract style the build side
    enforces in :mod:`...sorted_index._builder`. A no-op (zero output
    change) on any consistent build.
    """
    catalog_variants = int(cols.var_offsets[-1])
    geometry_variants = int(geometry.body_lengths.size)
    if geometry_variants != catalog_variants:
        raise ValueError(
            f"stale realized-geometry sidecar for {binary_name!r} "
            f"({arm_name} arm): catalog has {catalog_variants} variants "
            f"but the sidecar carries {geometry_variants} -- re-run the "
            f"realized-geometry generator"
        )

    catalog_sections = int(cols.n_variants.size)
    geometry_sections = int(geometry.n_sections)
    if geometry_sections != catalog_sections:
        raise ValueError(
            f"stale realized-geometry sidecar for {binary_name!r} "
            f"({arm_name} arm): catalog has {catalog_sections} sections "
            f"but the sidecar carries {geometry_sections} -- re-run the "
            f"realized-geometry generator"
        )
