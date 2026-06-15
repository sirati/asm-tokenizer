"""Per-binary realized-GEOMETRY sidecar generator.

Single concern: for one binary, emit the realized-geometry sidecar pair
per arm (``_realized.bin`` + ``_realized_index.bin``) by gluing the
per-region catalog read, the dedup-aware geometry compute, and the
geometry-format writer -- owning none of those three concerns'
internals. Sits ALONGSIDE the realized-length generator; the
``_lengths.bin`` family it never touches.

Boundary contract (the design-first sentence):

  *Given the memmap directory + a binary name, write each arm's
  per-variant geometry triple body + per-section CSR jump table -- one
  catalog read, one dedup-aware compute, one writer call per arm.*
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from tokenizer.aligned_data.memmap_format import (
    DATA_BIN_PRELUDE_SIZE,
    assert_data_bin_prelude,
)

from ._catalog import ArmCatalog, read_region_catalog
from ._compute import RealizedGeometry, realized_geometry_for_offsets
from ._geometry_format import (
    GEOMETRY_ARMS,
    RealizedGeometryArm,
    write_geometry_pair,
)


__all__ = ["generate_realized_geometry"]


def generate_realized_geometry(
    base_path: Path, binary_name: str
) -> Dict[str, List[Path]]:
    """Write both realized-geometry sidecar pairs for one binary.

    Returns ``{arm.name -> [geometry_path, index_path]}`` for every arm
    written (an empty arm still writes a valid header-only pair so the
    reader and downstream consumers see a uniform, complete file set).
    """
    base_path = Path(base_path)
    written: Dict[str, List[Path]] = {}
    for arm in GEOMETRY_ARMS:
        catalog = read_region_catalog(base_path, binary_name, arm.region)
        geometry = _compute_arm_geometry(base_path, binary_name, arm, catalog)
        geometry_path = arm.geometry_path(base_path, binary_name)
        index_path = arm.index_path(base_path, binary_name)
        write_geometry_pair(
            geometry_path,
            index_path,
            body_lengths=geometry.body_len,
            id_counts=geometry.id_count,
            value_counts=geometry.value_count,
            csr_offsets=catalog.var_offsets,
        )
        written[arm.name] = [geometry_path, index_path]
    return written


def _compute_arm_geometry(
    base_path: Path,
    binary_name: str,
    arm: RealizedGeometryArm,
    catalog: ArmCatalog,
) -> RealizedGeometry:
    """Per-variant ``u32`` geometry triple for one arm.

    Memmaps the arm's ``_data.bin`` read-only (prelude-validated) and
    runs the dedup-aware geometry compute over the catalog's record
    offsets. An empty arm (no variants) returns empty triple columns
    without opening the data file -- the body is empty and the CSR jump
    table degenerates to a single 0 terminator the writer still emits.
    """
    if catalog.record_offsets.size == 0:
        empty = np.zeros(0, dtype=np.uint32)
        return RealizedGeometry(empty, empty.copy(), empty.copy())

    data_path = arm.data_path(base_path, binary_name)
    data_u8 = np.memmap(str(data_path), dtype=np.uint8, mode="r")
    try:
        assert_data_bin_prelude(
            bytes(data_u8[:DATA_BIN_PRELUDE_SIZE]), path=str(data_path)
        )
        return realized_geometry_for_offsets(data_u8, catalog.record_offsets)
    finally:
        # np.memmap owns an mmap handle; close it deterministically
        # rather than waiting on GC (the CLI loops over many binaries).
        if data_u8._mmap is not None:  # pragma: no branch
            data_u8._mmap.close()
