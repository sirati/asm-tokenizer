"""Per-binary realized-length sidecar generator.

Single concern: for one binary, emit the four realized-length sidecars
(``_lengths.bin`` + ``_lengths_index.bin`` per arm) by gluing the
per-arm catalog read, the dedup-aware length compute, and the format
writer -- owning none of those three concerns' internals.

Boundary contract (the design-first sentence):

  *Given the memmap directory + a binary name, write each arm's
  per-variant realized-length body + per-section CSR jump table -- one
  catalog read, one dedup-aware compute, one writer call per arm.*

Sequencing: this pass runs BEFORE the sorted-index build (which will
later consume these sidecars instead of recomputing lengths). It reads
exactly the section catalog + the arm's ``_data.bin`` and writes only
the four sidecars; no ``BinarySession``, no sorted-index involvement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from tokenizer.aligned_data.memmap_format import (
    DATA_BIN_PRELUDE_SIZE,
    assert_data_bin_prelude,
)

from ._catalog import ArmCatalog, read_arm_catalog
from ._compute import realized_lengths_for_offsets
from ._format import ARMS, RealizedLengthsArm, write_lengths_pair


__all__ = ["generate_realized_lengths"]


def generate_realized_lengths(
    base_path: Path, binary_name: str
) -> Dict[str, List[Path]]:
    """Write all four realized-length sidecars for one binary.

    Returns ``{arm.name -> [lengths_path, index_path]}`` for every arm
    written (an empty arm still writes a valid header-only pair so the
    reader and downstream consumers see a uniform, complete file set).
    """
    base_path = Path(base_path)
    written: Dict[str, List[Path]] = {}
    for arm in ARMS:
        catalog = read_arm_catalog(base_path, binary_name, arm)
        lengths = _compute_arm_lengths(base_path, binary_name, arm, catalog)
        lengths_path = arm.lengths_path(base_path, binary_name)
        index_path = arm.index_path(base_path, binary_name)
        write_lengths_pair(
            lengths_path,
            index_path,
            lengths=lengths,
            csr_offsets=catalog.var_offsets,
        )
        written[arm.name] = [lengths_path, index_path]
    return written


def _compute_arm_lengths(
    base_path: Path,
    binary_name: str,
    arm: RealizedLengthsArm,
    catalog: ArmCatalog,
) -> np.ndarray:
    """Per-variant ``u32`` realized lengths for one arm.

    Memmaps the arm's ``_data.bin`` read-only (prelude-validated) and
    runs the dedup-aware compute over the catalog's record offsets. An
    empty arm (no variants) returns an empty array without opening the
    data file -- the body is empty and the CSR jump table degenerates to
    a single 0 terminator the writer still emits.
    """
    if catalog.record_offsets.size == 0:
        return np.zeros(0, dtype=np.uint32)

    data_path = arm.data_path(base_path, binary_name)
    data_u8 = np.memmap(str(data_path), dtype=np.uint8, mode="r")
    try:
        assert_data_bin_prelude(
            bytes(data_u8[:DATA_BIN_PRELUDE_SIZE]), path=str(data_path)
        )
        return realized_lengths_for_offsets(data_u8, catalog.record_offsets)
    finally:
        # np.memmap owns an mmap handle; close it deterministically
        # rather than waiting on GC (the CLI loops over many binaries).
        if data_u8._mmap is not None:  # pragma: no branch
            data_u8._mmap.close()
