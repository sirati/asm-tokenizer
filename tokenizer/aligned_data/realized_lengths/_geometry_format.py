"""On-disk format for the realized-GEOMETRY sidecars.

Single concern: the byte layout of the per-binary realized-geometry
sidecar pair (one pair per arm) and the arm-keyed filename suffixes that
name them. The 16-byte file-level preludes (magics ``RLG3`` / ``RGIX``)
are owned by :mod:`tokenizer.aligned_data.memmap_format` (the single
source of truth for every memmap magic + version); this module layers
the three-block body + shared CSR geometry on top and owns nothing about
the prelude bytes beyond importing the wrappers.

The geometry sidecar is a SUPERSET of the realized-length sidecar (see
:mod:`._format`): instead of one ``body_len`` per (section, variant) it
stores the full ``bulk_contributing_geometry`` triple
``(body_len, id_count, value_count)``. Per arm (matched / unmatched) two
files form a pair:

* ``<binary><geometry_suffix>`` -- ``RLG3`` prelude + THREE contiguous
  ``u32`` blocks each of length ``N = Σ variants``, section-major,
  variants in catalog order within each section, in axis order
  (body_len, id_count, value_count). The three blocks are parallel: row
  ``i`` of each block describes the same (section, variant).
* ``<binary><index_suffix>`` -- ``RGIX`` prelude + ``n_sections + 1``
  ``u32`` CSR entries. ``entry[s]`` is the ELEMENT offset (not byte
  offset) of section ``s``'s first variant in EACH block; the ``+1``
  terminator carries ``N`` (one block's length). ONE CSR serves all
  three blocks because they are parallel -- a section's variant count is
  ``entry[s + 1] - entry[s]`` regardless of axis.

Dtype choice: every value (the three body axes and the CSR jump-table
entries) is ``u32`` -- identical to the realized-length sidecar's
choice, for the same reasons (per-record contributing counts and CSR
element offsets are both ``< 2**32`` for any single binary). All
little-endian (numpy native on the target).

The two arms reuse this exact layout against separate data files; the
only thing that differs is the filename suffix, threaded as a typed
:class:`RealizedGeometryArm` so no caller hand-rolls the matched /
unmatched string pair. The geometry suffixes are DISTINCT from the
length suffixes (``_realized*.bin`` vs ``_lengths*.bin``) so the two
sidecar families coexist without collision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from tokenizer.aligned_data.memmap_format import (
    REALIZED_GEOMETRY_BIN_PRELUDE_SIZE,
    REALIZED_GEOMETRY_INDEX_BIN_PRELUDE_SIZE,
    assert_realized_geometry_index_prelude,
    assert_realized_geometry_prelude,
    encode_realized_geometry_index_prelude,
    encode_realized_geometry_prelude,
)

from ._catalog import SectionRegion
from ._format import CSR_DTYPE, LENGTH_DTYPE, MAX_REALIZED_LENGTH


__all__ = [
    "GEOMETRY_DTYPE",
    "MAX_REALIZED_VALUE",
    "N_GEOMETRY_AXES",
    "RealizedGeometryArm",
    "GEOMETRY_MATCHED_ARM",
    "GEOMETRY_UNMATCHED_ARM",
    "GEOMETRY_ARMS",
    "write_geometry_pair",
    "read_geometry_pair",
]


#: Body element dtype: one count per (section, variant) per axis. Shared
#: with the realized-length body dtype -- the geometry body is just three
#: of those blocks (axis-major).
GEOMETRY_DTYPE = np.dtype(LENGTH_DTYPE)

#: Number of parallel ``u32`` blocks in the geometry body, in stored
#: order: (body_len, id_count, value_count).
N_GEOMETRY_AXES: int = 3

#: Largest value any geometry axis can carry. Identical to the
#: realized-length cap (``0xFFFFFFFE``): ``0xFFFFFFFF`` is RESERVED as the
#: ``HashMapU64U32`` vectorized-miss sentinel used by the shared dedup
#: (see ``_compute._U32_MISS``), so a stored count of ``0xFFFFFFFF`` would
#: be indistinguishable from a hashmap miss. The generator hard-errors
#: (never clamps) on overflow of ANY axis.
MAX_REALIZED_VALUE: int = MAX_REALIZED_LENGTH


@dataclass(frozen=True)
class RealizedGeometryArm:
    """One arm's geometry-sidecar filename suffixes + its companion data suffix.

    ``data_suffix`` is the per-binary ``_data.bin`` reference the
    generator dedups against for THIS arm; ``geometry_suffix`` /
    ``index_suffix`` name the two sidecars it writes. Threaded as one
    typed object so the generator and reader never re-pick the
    matched-vs-unmatched string pair inline (mirrors
    :class:`._format.RealizedLengthsArm`).
    """

    name: str
    region: SectionRegion
    geometry_suffix: str
    index_suffix: str
    data_suffix: str

    def geometry_path(self, base_path: Path, binary_name: str) -> Path:
        return Path(base_path) / f"{binary_name}{self.geometry_suffix}"

    def index_path(self, base_path: Path, binary_name: str) -> Path:
        return Path(base_path) / f"{binary_name}{self.index_suffix}"

    def data_path(self, base_path: Path, binary_name: str) -> Path:
        return Path(base_path) / f"{binary_name}{self.data_suffix}"


GEOMETRY_MATCHED_ARM = RealizedGeometryArm(
    name="matched",
    region=SectionRegion.MATCHED,
    geometry_suffix="_realized.bin",
    index_suffix="_realized_index.bin",
    data_suffix="_data.bin",
)

GEOMETRY_UNMATCHED_ARM = RealizedGeometryArm(
    name="unmatched",
    region=SectionRegion.UNMATCHED,
    geometry_suffix="_unmatched_realized.bin",
    index_suffix="_unmatched_realized_index.bin",
    data_suffix="_unmatched_data.bin",
)

#: Both arms in catalog emission order (matched region precedes the
#: unmatched region in ``_sections.bin``).
GEOMETRY_ARMS: Tuple[RealizedGeometryArm, ...] = (
    GEOMETRY_MATCHED_ARM,
    GEOMETRY_UNMATCHED_ARM,
)


def write_geometry_pair(
    geometry_path: Path,
    index_path: Path,
    *,
    body_lengths: np.ndarray,
    id_counts: np.ndarray,
    value_counts: np.ndarray,
    csr_offsets: np.ndarray,
) -> None:
    """Write one arm's ``(realized.bin, realized_index.bin)`` pair.

    The three axes are flat per-variant blocks (section-major), each of
    the SAME length ``N``; ``csr_offsets`` is the ``n_sections + 1`` CSR
    element-offset table whose last entry equals ``N``. Each block is
    cast to the on-disk dtype and written contiguously in axis order
    (body_len, id_count, value_count); the caller must have already
    range-checked every axis (the generator hard-asserts u32 fit before
    calling here).
    """
    blocks = tuple(
        np.ascontiguousarray(axis, dtype=GEOMETRY_DTYPE)
        for axis in (body_lengths, id_counts, value_counts)
    )
    sizes = {int(block.size) for block in blocks}
    if len(sizes) != 1:
        raise ValueError(
            f"geometry axes must be parallel; got block sizes "
            f"{[int(b.size) for b in blocks]}"
        )
    (n,) = sizes
    csr = np.ascontiguousarray(csr_offsets, dtype=CSR_DTYPE)
    if csr.size == 0 or int(csr[-1]) != n:
        raise ValueError(
            f"CSR terminator {int(csr[-1]) if csr.size else '<empty>'} must "
            f"equal the per-axis element count {n}"
        )
    with open(geometry_path, "wb") as fh:
        fh.write(encode_realized_geometry_prelude())
        for block in blocks:
            fh.write(block.tobytes())
    with open(index_path, "wb") as fh:
        fh.write(encode_realized_geometry_index_prelude())
        fh.write(csr.tobytes())


def read_geometry_pair(
    geometry_path: Path, index_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Open one arm's pair read-only; return ``(body, id, value, csr)`` views.

    The three axis arrays are zero-copy ``np.memmap`` views over their
    respective ``u32`` block of the geometry file (the 16-byte prelude
    plus the preceding blocks are skipped via ``offset=``); ``csr`` is a
    zero-copy view over the index file's body. Preludes are validated
    (magic + version) before the views are constructed; a wrong magic /
    version raises :class:`ValueError`. Missing files raise
    :class:`FileNotFoundError` -- callers wanting a friendlier "run the
    generator first" message gate on existence via the discovery helper
    before calling here.
    """
    geometry_path = Path(geometry_path)
    index_path = Path(index_path)
    _assert_prelude(geometry_path, assert_realized_geometry_prelude)
    _assert_prelude(index_path, assert_realized_geometry_index_prelude)

    total_elems = _body_element_count(
        geometry_path, REALIZED_GEOMETRY_BIN_PRELUDE_SIZE, GEOMETRY_DTYPE
    )
    if total_elems % N_GEOMETRY_AXES != 0:
        raise ValueError(
            f"{geometry_path}: geometry body of {total_elems} elements is "
            f"not divisible by {N_GEOMETRY_AXES} axes; re-run the "
            f"realized-geometry generator to regenerate"
        )
    n = total_elems // N_GEOMETRY_AXES
    n_csr = _body_element_count(
        index_path, REALIZED_GEOMETRY_INDEX_BIN_PRELUDE_SIZE, CSR_DTYPE
    )
    axes = tuple(
        np.memmap(
            str(geometry_path),
            dtype=GEOMETRY_DTYPE,
            mode="r",
            offset=REALIZED_GEOMETRY_BIN_PRELUDE_SIZE
            + axis_idx * n * GEOMETRY_DTYPE.itemsize,
            shape=(n,),
        )
        for axis_idx in range(N_GEOMETRY_AXES)
    )
    csr = np.memmap(
        str(index_path),
        dtype=CSR_DTYPE,
        mode="r",
        offset=REALIZED_GEOMETRY_INDEX_BIN_PRELUDE_SIZE,
        shape=(n_csr,),
    )
    return (*axes, csr)


def _assert_prelude(path: Path, assert_fn) -> None:
    """Read + validate the 16-byte prelude of ``path`` via ``assert_fn``."""
    with open(path, "rb") as fh:
        assert_fn(fh.read(16), path=str(path))


def _body_element_count(path: Path, prelude_size: int, dtype: np.dtype) -> int:
    """Number of ``dtype`` elements in ``path``'s body region.

    Raises :class:`ValueError` if the body region is not a whole number
    of elements (a truncation / corruption signature) -- the message
    names the regenerate path.
    """
    filesize = path.stat().st_size
    body_bytes = filesize - prelude_size
    if body_bytes < 0 or body_bytes % dtype.itemsize != 0:
        raise ValueError(
            f"{path}: body region of {body_bytes} bytes is not a whole "
            f"number of {dtype.itemsize}-byte elements; re-run the "
            f"realized-geometry generator to regenerate"
        )
    return body_bytes // dtype.itemsize
