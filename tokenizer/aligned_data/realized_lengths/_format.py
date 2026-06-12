"""On-disk format for the realized-token-length sidecars.

Single concern: the byte layout of the four per-binary sidecar files
the realized-lengths pass emits, and the arm-keyed filename suffixes
that name them. The 16-byte file-level preludes (magics ``RLEN`` /
``RLIX``) are owned by :mod:`tokenizer.aligned_data.memmap_format` (the
single source of truth for every memmap magic + version); this module
layers the body / CSR-jump-table geometry on top and owns nothing about
the prelude bytes beyond importing the wrappers.

Per arm (matched / unmatched) two files form a pair:

* ``<binary><lengths_suffix>`` -- ``RLEN`` prelude + a flat ``u32``
  body: one realized record-body length per (matched section, variant),
  section-major, variants in catalog order within each section.
* ``<binary><index_suffix>`` -- ``RLIX`` prelude + ``n_sections + 1``
  ``u32`` CSR entries. ``entry[s]`` is the ELEMENT offset (not byte
  offset) of section ``s``'s first variant length in the body; the
  ``+1`` terminator carries the total length count so each section's
  variant count is ``entry[s + 1] - entry[s]`` with no separate count
  field.

Dtype choice (documented per the owner mandate): both the body lengths
and the CSR jump-table entries are ``u32``. The lengths are per-record
realized body counts -- the same field the legacy ``_index.bin``
stores as ``u32`` and that the generator hard-asserts never exceeds the
u32 range. The CSR entries are ELEMENT offsets bounded by the binary's
total variant count (one entry per (section, variant)), which is itself
``< 2**32`` for any single binary; ``u32`` is sufficient and keeps the
jump table half the size of a ``u64`` table. Both are little-endian
(numpy native on the target).

The two arms reuse this exact layout against separate data files; the
only thing that differs is the filename suffix, threaded as a typed
:class:`RealizedLengthsArm` so no caller hand-rolls the matched /
unmatched string pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from tokenizer.aligned_data.memmap_format import (
    REALIZED_LENGTHS_BIN_PRELUDE_SIZE,
    REALIZED_LENGTHS_INDEX_BIN_PRELUDE_SIZE,
    assert_realized_lengths_index_prelude,
    assert_realized_lengths_prelude,
    encode_realized_lengths_index_prelude,
    encode_realized_lengths_prelude,
)


__all__ = [
    "LENGTH_DTYPE",
    "CSR_DTYPE",
    "MAX_REALIZED_LENGTH",
    "RealizedLengthsArm",
    "MATCHED_ARM",
    "UNMATCHED_ARM",
    "ARMS",
    "write_lengths_pair",
    "read_lengths_pair",
]


#: Body element dtype: one realized record-body length per variant.
LENGTH_DTYPE = np.dtype("<u4")

#: CSR jump-table element dtype: per-section element offsets into the body.
CSR_DTYPE = np.dtype("<u4")

#: Largest realized length the ``u32`` body can carry. The generator
#: hard-errors (never clamps) on overflow.
MAX_REALIZED_LENGTH: int = (1 << 32) - 1


@dataclass(frozen=True)
class RealizedLengthsArm:
    """One arm's sidecar filename suffixes + its companion data suffix.

    ``data_suffix`` / ``catalog`` are the per-binary ``_data.bin`` /
    section-catalog references the generator dedups against for THIS
    arm; ``lengths_suffix`` / ``index_suffix`` name the two sidecars it
    writes. Threaded as one typed object so the generator and reader
    never re-pick the matched-vs-unmatched string pair inline.
    """

    name: str
    lengths_suffix: str
    index_suffix: str
    data_suffix: str

    def lengths_path(self, base_path: Path, binary_name: str) -> Path:
        return Path(base_path) / f"{binary_name}{self.lengths_suffix}"

    def index_path(self, base_path: Path, binary_name: str) -> Path:
        return Path(base_path) / f"{binary_name}{self.index_suffix}"

    def data_path(self, base_path: Path, binary_name: str) -> Path:
        return Path(base_path) / f"{binary_name}{self.data_suffix}"


MATCHED_ARM = RealizedLengthsArm(
    name="matched",
    lengths_suffix="_lengths.bin",
    index_suffix="_lengths_index.bin",
    data_suffix="_data.bin",
)

UNMATCHED_ARM = RealizedLengthsArm(
    name="unmatched",
    lengths_suffix="_unmatched_lengths.bin",
    index_suffix="_unmatched_lengths_index.bin",
    data_suffix="_unmatched_data.bin",
)

#: Both arms in catalog emission order (matched region precedes the
#: unmatched region in ``_sections.bin``).
ARMS: Tuple[RealizedLengthsArm, ...] = (MATCHED_ARM, UNMATCHED_ARM)


def write_lengths_pair(
    lengths_path: Path,
    index_path: Path,
    *,
    lengths: np.ndarray,
    csr_offsets: np.ndarray,
) -> None:
    """Write one arm's ``(lengths.bin, lengths_index.bin)`` pair.

    ``lengths`` is the flat per-variant body (section-major); ``csr_offsets``
    is the ``n_sections + 1`` CSR element-offset table whose last entry
    equals ``lengths.size``. Both are cast to the on-disk dtypes; the
    caller must have already range-checked the lengths (the generator
    hard-asserts u32 fit before calling here).
    """
    body = np.ascontiguousarray(lengths, dtype=LENGTH_DTYPE)
    csr = np.ascontiguousarray(csr_offsets, dtype=CSR_DTYPE)
    if csr.size == 0 or int(csr[-1]) != int(body.size):
        raise ValueError(
            f"CSR terminator {int(csr[-1]) if csr.size else '<empty>'} must "
            f"equal the body length count {int(body.size)}"
        )
    with open(lengths_path, "wb") as fh:
        fh.write(encode_realized_lengths_prelude())
        fh.write(body.tobytes())
    with open(index_path, "wb") as fh:
        fh.write(encode_realized_lengths_index_prelude())
        fh.write(csr.tobytes())


def read_lengths_pair(
    lengths_path: Path, index_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Open one arm's pair read-only; return ``(lengths_view, csr_view)``.

    Both returned arrays are zero-copy ``np.memmap`` views over the body
    region of their file (the 16-byte prelude is skipped via ``offset=``).
    Preludes are validated (magic + version) before the views are
    constructed; a wrong magic / version raises :class:`ValueError`.
    Missing files raise :class:`FileNotFoundError` -- callers wanting a
    friendlier "run the generator first" message gate on existence via
    the discovery helper before calling here.
    """
    lengths_path = Path(lengths_path)
    index_path = Path(index_path)
    _assert_prelude(lengths_path, assert_realized_lengths_prelude)
    _assert_prelude(index_path, assert_realized_lengths_index_prelude)

    n_lengths = _body_element_count(
        lengths_path, REALIZED_LENGTHS_BIN_PRELUDE_SIZE, LENGTH_DTYPE
    )
    n_csr = _body_element_count(
        index_path, REALIZED_LENGTHS_INDEX_BIN_PRELUDE_SIZE, CSR_DTYPE
    )
    lengths = np.memmap(
        str(lengths_path),
        dtype=LENGTH_DTYPE,
        mode="r",
        offset=REALIZED_LENGTHS_BIN_PRELUDE_SIZE,
        shape=(n_lengths,),
    )
    csr = np.memmap(
        str(index_path),
        dtype=CSR_DTYPE,
        mode="r",
        offset=REALIZED_LENGTHS_INDEX_BIN_PRELUDE_SIZE,
        shape=(n_csr,),
    )
    return lengths, csr


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
            f"realized-lengths generator to regenerate"
        )
    return body_bytes // dtype.itemsize
