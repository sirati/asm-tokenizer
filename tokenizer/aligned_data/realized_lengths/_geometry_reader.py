"""Lazy mmap reader for the realized-GEOMETRY sidecars.

Single concern: zero-copy addressing into one arm's
``(_realized.bin, _realized_index.bin)`` pair. The reader opens the pair
read-only (preludes validated), holds the three axis memmap views + the
shared CSR view, and answers per-section / per-(section, variant)
queries by slicing each axis via the CSR jump table -- no materialised
wrapper lists, no caching layer over the already-addressable bytes
(house rules).

Boundary contract (the design-first sentence):

  *Given an arm's geometry + index sidecar paths, expose the raw u32
  body-length / id-count / value-count axes + CSR offsets for vectorized
  consumers, plus zero-copy per-section slices and scalar (section,
  variant) triple lookups -- reading the bytes directly, never copying
  them into Python state.*

A discovery helper (:func:`realized_geometry_present`) lets future
consumers fail with a clear "run the realized-geometry generator first"
message instead of a bare ``FileNotFoundError``; no consumer is wired
here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from ._geometry_format import (
    GEOMETRY_MATCHED_ARM,
    RealizedGeometryArm,
    read_geometry_pair,
)


__all__ = [
    "RealizedGeometryReader",
    "realized_geometry_present",
    "require_realized_geometry",
]


class RealizedGeometryReader:
    """Lazy mmap view over one arm's realized-geometry sidecar pair.

    Construct via :meth:`open` (the arm-keyed entry point) so the matched
    / unmatched filename pair is never hand-rolled by callers. The three
    axis memmap views + the shared CSR view stay live for the reader's
    lifetime; :meth:`close` releases them deterministically (the views
    are also released on GC).
    """

    def __init__(
        self,
        body_lengths: np.ndarray,
        id_counts: np.ndarray,
        value_counts: np.ndarray,
        csr: np.ndarray,
    ) -> None:
        # ``read_geometry_pair`` already prelude-validated both files and
        # split the three parallel axes of equal length.
        if csr.size == 0:
            raise ValueError(
                "realized-geometry index has no CSR entries; a valid "
                "(possibly empty) index always carries at least the "
                "single 0 terminator -- re-run the realized-geometry "
                "generator to regenerate"
            )
        if int(csr[-1]) != int(body_lengths.size):
            raise ValueError(
                f"realized-geometry CSR terminator {int(csr[-1])} does not "
                f"match the per-axis element count {int(body_lengths.size)}; "
                f"the sidecar pair is inconsistent -- re-run the "
                f"realized-geometry generator to regenerate"
            )
        self._body_lengths = body_lengths
        self._id_counts = id_counts
        self._value_counts = value_counts
        self._csr = csr

    @classmethod
    def open(
        cls,
        base_path: Path,
        binary_name: str,
        arm: RealizedGeometryArm = GEOMETRY_MATCHED_ARM,
    ) -> "RealizedGeometryReader":
        """Open ``arm``'s sidecar pair for ``binary_name`` under ``base_path``.

        Raises :class:`FileNotFoundError` (named to point at the
        generator) when either sidecar is absent; prefer
        :func:`require_realized_geometry` for the friendly pre-flight
        check.
        """
        base_path = Path(base_path)
        geometry_path = arm.geometry_path(base_path, binary_name)
        index_path = arm.index_path(base_path, binary_name)
        require_realized_geometry(base_path, binary_name, arm)
        body, ids, values, csr = read_geometry_pair(geometry_path, index_path)
        return cls(body, ids, values, csr)

    # ------------------------------------------------------------------
    # Vectorized access
    # ------------------------------------------------------------------
    @property
    def body_lengths(self) -> np.ndarray:
        """The raw ``u32`` body-length axis (one per (section, variant))."""
        return self._body_lengths

    @property
    def id_counts(self) -> np.ndarray:
        """The raw ``u32`` identity-carrier-count axis."""
        return self._id_counts

    @property
    def value_counts(self) -> np.ndarray:
        """The raw ``u32`` numeric-chunk-count axis."""
        return self._value_counts

    @property
    def csr_offsets(self) -> np.ndarray:
        """The ``n_sections + 1`` CSR element-offset jump table."""
        return self._csr

    @property
    def n_sections(self) -> int:
        return max(0, int(self._csr.size) - 1)

    # ------------------------------------------------------------------
    # Per-section / scalar access
    # ------------------------------------------------------------------
    def _section_bounds(self, section_idx: int) -> Tuple[int, int]:
        if section_idx < 0 or section_idx >= self.n_sections:
            raise IndexError(
                f"section_idx {section_idx} out of range "
                f"[0, {self.n_sections})"
            )
        return int(self._csr[section_idx]), int(self._csr[section_idx + 1])

    def per_section(
        self, section_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Zero-copy ``(body, id, value)`` views of section ``section_idx``.

        Returns three parallel slices of the axis memmaps (no copy); a
        0-variant section yields three empty views. Raises
        :class:`IndexError` for an out-of-range section.
        """
        lo, hi = self._section_bounds(section_idx)
        return (
            self._body_lengths[lo:hi],
            self._id_counts[lo:hi],
            self._value_counts[lo:hi],
        )

    def geometry(self, section_idx: int, variant_idx: int) -> Tuple[int, int, int]:
        """Scalar ``(body_len, id_count, value_count)`` of ``(section, variant)``.

        Raises :class:`IndexError` for an out-of-range section or
        variant.
        """
        lo, hi = self._section_bounds(section_idx)
        n_variants = hi - lo
        if variant_idx < 0 or variant_idx >= n_variants:
            raise IndexError(
                f"variant_idx {variant_idx} out of range "
                f"[0, {n_variants}) for section {section_idx}"
            )
        row = lo + variant_idx
        return (
            int(self._body_lengths[row]),
            int(self._id_counts[row]),
            int(self._value_counts[row]),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release every memmap handle deterministically."""
        for arr in (
            self._body_lengths,
            self._id_counts,
            self._value_counts,
            self._csr,
        ):
            mmap = getattr(arr, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "RealizedGeometryReader":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def realized_geometry_present(
    base_path: Path,
    binary_name: str,
    arm: RealizedGeometryArm = GEOMETRY_MATCHED_ARM,
) -> bool:
    """True iff both of ``arm``'s geometry sidecar files exist for ``binary_name``."""
    base_path = Path(base_path)
    return (
        arm.geometry_path(base_path, binary_name).exists()
        and arm.index_path(base_path, binary_name).exists()
    )


def require_realized_geometry(
    base_path: Path,
    binary_name: str,
    arm: RealizedGeometryArm = GEOMETRY_MATCHED_ARM,
) -> None:
    """Raise a generator-pointing error if ``arm``'s geometry sidecars are absent.

    The single chokepoint future consumers call before opening a reader,
    so the "run the realized-geometry generator first" guidance lives in
    one place instead of every call site.
    """
    if realized_geometry_present(base_path, binary_name, arm):
        return
    raise FileNotFoundError(
        f"realized-geometry sidecars for {binary_name!r} ({arm.name} arm) are "
        f"missing under {base_path}; run the realized-geometry generator "
        f"first: python -m tokenizer.aligned_data.realized_lengths "
        f"--input-dir {base_path}"
    )
