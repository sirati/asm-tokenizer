"""Open the body-free + body geometry handles for the vectorized path.

Single concern: bundle, for one binary's MATCHED arm, the handles the
geometry prepass (plan C1) + the fused scatter (plan C2) consume -- the
columnar ``sections.bin`` catalog + its ``section_offsets``, the RLG3
realized-geometry reader, the ``_variants.bin`` prefix bytes, and the
``_data.bin`` body bytes -- opened through the SAME readers the index
build uses, never a bespoke BIN parse.

The handles are all lazy / mmap views (house rules); :meth:`close`
releases the geometry reader + the memmaps deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.realized_lengths import RealizedGeometryReader
from tokenizer.aligned_data.realized_lengths._geometry_format import (
    GEOMETRY_MATCHED_ARM,
)
from tokenizer.aligned_data.sorted_index._prepass import (
    read_section_variant_info,
)


__all__ = ["VectorBatchHandles", "open_vector_batch_handles"]


@dataclass(frozen=True)
class VectorBatchHandles:
    """The opened geometry + body handles for one binary (matched arm).

    ``cols`` / ``section_offsets`` index by MATCHED section position (the
    ``BinarySession.load_matched`` index space, parallel to the RLG3
    axes). ``geometry`` is the RLG3 reader; ``variants_u8`` /
    ``data_u8`` are read-only uint8 memmap views.
    """

    cols: ColumnarSections
    section_offsets: np.ndarray
    geometry: RealizedGeometryReader
    variants_u8: np.ndarray
    data_u8: np.ndarray

    def close(self) -> None:
        """Release the geometry reader + the memmap views."""
        self.geometry.close()
        for arr in (self.variants_u8, self.data_u8, self.section_offsets):
            mmap = getattr(arr, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "VectorBatchHandles":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def open_vector_batch_handles(
    base_path: Path, binary_name: str
) -> VectorBatchHandles:
    """Open the matched-arm geometry + body handles for ``binary_name``.

    Parameters
    ----------
    base_path / binary_name:
        The memmap directory + binary stem (the same keys the index
        build + :class:`BinarySession` use).

    Returns
    -------
    VectorBatchHandles
        The columnar catalog + section offsets, the RLG3 reader, and the
        ``_variants.bin`` / ``_data.bin`` uint8 memmaps.
    """
    base_path = Path(base_path)
    info = read_section_variant_info(base_path, binary_name)
    geometry = RealizedGeometryReader.open(
        base_path, binary_name, GEOMETRY_MATCHED_ARM
    )
    # An ABSENT ``_variants.bin`` is valid: the session's variant resolver
    # treats it as "no variant-prefix records" (empty ``variant_tokens``).
    # Hand the prefix readers an empty buffer so they mirror that exactly
    # (see ``_prefix`` / ``_prefix_values`` empty-buffer handling).
    variants_path = base_path / f"{binary_name}_variants.bin"
    variants_u8 = (
        np.memmap(variants_path, dtype=np.uint8, mode="r")
        if variants_path.exists()
        else np.zeros(0, dtype=np.uint8)
    )
    data_u8 = np.memmap(
        base_path / f"{binary_name}_data.bin", dtype=np.uint8, mode="r"
    )
    return VectorBatchHandles(
        cols=info.cols,
        section_offsets=np.asarray(info.section_offsets, dtype=np.int64),
        geometry=geometry,
        variants_u8=variants_u8,
        data_u8=data_u8,
    )
