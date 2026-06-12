"""Lazy mmap reader for the realized-length sidecars.

Single concern: zero-copy addressing into one arm's
``(_lengths.bin, _lengths_index.bin)`` pair. The reader opens the pair
read-only (preludes validated), holds the two memmap views, and answers
per-section / per-(section, variant) queries by slicing the body via the
CSR jump table -- no materialised wrapper lists, no caching layer over
the already-addressable bytes (house rules).

Boundary contract (the design-first sentence):

  *Given an arm's lengths + index sidecar paths, expose the raw u32
  lengths array + CSR offsets for vectorized consumers, plus zero-copy
  per-section slices and scalar (section, variant) lookups -- reading
  the bytes directly, never copying them into Python state.*

A discovery helper (:func:`realized_lengths_present`) lets future
consumers fail with a clear "run the realized-lengths generator first"
message instead of a bare ``FileNotFoundError``; no consumer is wired
here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._format import (
    MATCHED_ARM,
    RealizedLengthsArm,
    read_lengths_pair,
)


__all__ = [
    "RealizedLengths",
    "realized_lengths_present",
    "require_realized_lengths",
]


class RealizedLengths:
    """Lazy mmap view over one arm's realized-length sidecar pair.

    Construct via :meth:`open` (the arm-keyed entry point) so the matched
    / unmatched filename pair is never hand-rolled by callers. The two
    memmap views stay live for the reader's lifetime; :meth:`close`
    releases them deterministically (the views are also released on GC).
    """

    def __init__(self, lengths: np.ndarray, csr: np.ndarray) -> None:
        # ``read_lengths_pair`` already prelude-validated both files.
        if csr.size == 0:
            raise ValueError(
                "realized-lengths index has no CSR entries; a valid "
                "(possibly empty) index always carries at least the "
                "single 0 terminator -- re-run the realized-lengths "
                "generator to regenerate"
            )
        if int(csr[-1]) != int(lengths.size):
            raise ValueError(
                f"realized-lengths CSR terminator {int(csr[-1])} does not "
                f"match the body length count {int(lengths.size)}; the "
                f"sidecar pair is inconsistent -- re-run the "
                f"realized-lengths generator to regenerate"
            )
        self._lengths = lengths
        self._csr = csr

    @classmethod
    def open(
        cls,
        base_path: Path,
        binary_name: str,
        arm: RealizedLengthsArm = MATCHED_ARM,
    ) -> "RealizedLengths":
        """Open ``arm``'s sidecar pair for ``binary_name`` under ``base_path``.

        Raises :class:`FileNotFoundError` (named to point at the
        generator) when either sidecar is absent; prefer
        :func:`require_realized_lengths` for the friendly pre-flight
        check.
        """
        base_path = Path(base_path)
        lengths_path = arm.lengths_path(base_path, binary_name)
        index_path = arm.index_path(base_path, binary_name)
        require_realized_lengths(base_path, binary_name, arm)
        lengths, csr = read_lengths_pair(lengths_path, index_path)
        return cls(lengths, csr)

    # ------------------------------------------------------------------
    # Vectorized access
    # ------------------------------------------------------------------
    @property
    def lengths(self) -> np.ndarray:
        """The raw ``u32`` body: one realized length per (section, variant)."""
        return self._lengths

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
    def per_section(self, section_idx: int) -> np.ndarray:
        """Zero-copy view of section ``section_idx``'s variant lengths.

        Returns a slice of the body memmap (no copy); a 0-variant
        section yields an empty view. Raises :class:`IndexError` for an
        out-of-range section.
        """
        if section_idx < 0 or section_idx >= self.n_sections:
            raise IndexError(
                f"section_idx {section_idx} out of range "
                f"[0, {self.n_sections})"
            )
        lo = int(self._csr[section_idx])
        hi = int(self._csr[section_idx + 1])
        return self._lengths[lo:hi]

    def length(self, section_idx: int, variant_idx: int) -> int:
        """Scalar realized length of ``(section_idx, variant_idx)``.

        Raises :class:`IndexError` for an out-of-range section or
        variant.
        """
        section = self.per_section(section_idx)
        if variant_idx < 0 or variant_idx >= section.size:
            raise IndexError(
                f"variant_idx {variant_idx} out of range "
                f"[0, {section.size}) for section {section_idx}"
            )
        return int(section[variant_idx])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release both memmap handles deterministically."""
        for arr in (self._lengths, self._csr):
            mmap = getattr(arr, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "RealizedLengths":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def realized_lengths_present(
    base_path: Path,
    binary_name: str,
    arm: RealizedLengthsArm = MATCHED_ARM,
) -> bool:
    """True iff both of ``arm``'s sidecar files exist for ``binary_name``."""
    base_path = Path(base_path)
    return (
        arm.lengths_path(base_path, binary_name).exists()
        and arm.index_path(base_path, binary_name).exists()
    )


def require_realized_lengths(
    base_path: Path,
    binary_name: str,
    arm: RealizedLengthsArm = MATCHED_ARM,
) -> None:
    """Raise a generator-pointing error if ``arm``'s sidecars are absent.

    The single chokepoint future consumers call before opening a reader,
    so the "run the realized-lengths generator first" guidance lives in
    one place instead of every call site.
    """
    if realized_lengths_present(base_path, binary_name, arm):
        return
    raise FileNotFoundError(
        f"realized-length sidecars for {binary_name!r} ({arm.name} arm) are "
        f"missing under {base_path}; run the realized-lengths generator "
        f"first: python -m tokenizer.aligned_data.realized_lengths "
        f"--input-dir {base_path}"
    )
