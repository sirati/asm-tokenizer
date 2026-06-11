"""Per-binary sorted-index build entry (plan §"Module layout").

Single concern: glue the catalog pre-pass + the walk-free length
compute + wire encode into one per-binary call, and offer a thin
file-writing wrapper that stamps the canonical
``<binary>_sorted_<mode>_d<depth>.idx`` filenames.

Boundary contract (the design-first sentence):

  *Given the memmap directory + a binary name + reductions + depths,
  read the catalog once, memmap the data bin read-only, compute every
  (reduction, depth) length array via
  :func:`compute_reduced_lengths`, and return one wire-encoded blob
  per pair. The file-writing wrapper layers filename construction on
  top -- it owns no compute logic.*

No :class:`BinarySession` involvement: the build reads exactly three
sidecars (``_index.bin`` locator, ``_sections.bin`` catalog,
``_data.bin`` record headers + token regions) and none of the
session's metadata machinery. No CLI parsing here; no string-typed
modes. Callers wanting CLI / multi-binary fan-out go through
:mod:`.__main__`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tokenizer.aligned_data.memmap_format import (
    DATA_BIN_PRELUDE_SIZE,
    assert_data_bin_prelude,
)

from ._dedup import PLAIN, DuplicateHandling
from ._gating import VariantGate
from ._length_compute import compute_reduced_lengths
from ._prepass import read_section_variant_info
from ._types import IndexSpec, LengthReduction
from ._wire import encode_sorted_index


__all__ = ["build_sorted_index_bytes", "write_sorted_index_files"]


def build_sorted_index_bytes(
    base_path: Path,
    binary_name: str,
    *,
    reductions: List[LengthReduction],
    depths: List[int],
    gate: VariantGate = VariantGate(),
    duplicate_handling: DuplicateHandling = PLAIN,
) -> Dict[IndexSpec, bytes]:
    """Build per-(mode, depth) sorted-index bytes for one binary.

    Runs the columnar pre-pass (:func:`read_section_variant_info`),
    memmaps ``<binary>_data.bin`` read-only (prelude-validated), and
    computes EVERY requested ``(reduction, depth)`` from one graph
    traversal via :func:`compute_reduced_lengths` (plan §D8: the
    heavy work is not repeated per reduction or per depth). Each
    resulting ``u32[num_sections]`` array is wire-encoded via
    :func:`encode_sorted_index`.

    Parameters
    ----------
    base_path
        Memmap directory containing the ``<binary>_*`` sidecars.
    binary_name
        The binary's ``<binary>`` prefix.
    reductions / depths
        The modes / splice depths to compute. Either list empty ->
        empty dict (nothing is opened).
    gate / duplicate_handling
        Top-level minimum-variant gate + duplicate strategy.

    Returns
    -------
    Dict[IndexSpec, bytes]
        ``{IndexSpec(reduction, depth) -> wire bytes}``.
    """
    if not reductions or not depths:
        return {}

    base_path = Path(base_path)
    section_info = read_section_variant_info(base_path, binary_name)
    if section_info.counts.size == 0:
        # No matched arm: every output is the canonical empty index.
        return {
            IndexSpec(reduction=red, depth=d): encode_sorted_index(
                np.zeros(0, dtype=np.uint32)
            )
            for red in reductions
            for d in depths
        }

    data_path = base_path / f"{binary_name}_data.bin"
    data_u8 = np.memmap(str(data_path), dtype=np.uint8, mode="r")
    try:
        assert_data_bin_prelude(
            bytes(data_u8[:DATA_BIN_PRELUDE_SIZE]), path=str(data_path)
        )
        per_spec_lengths = compute_reduced_lengths(
            section_info,
            data_u8,
            depths=depths,
            reductions=reductions,
            gate=gate,
            duplicate_handling=duplicate_handling,
        )
    finally:
        # np.memmap owns an mmap handle; close it deterministically
        # rather than waiting on GC (the CLI loops over many binaries).
        if data_u8._mmap is not None:  # pragma: no branch
            data_u8._mmap.close()

    return {
        spec: encode_sorted_index(lengths)
        for spec, lengths in per_spec_lengths.items()
    }


def write_sorted_index_files(
    memmap_dir: Path,
    binary_name: str,
    *,
    reductions: List[LengthReduction],
    depths: List[int],
    gate: VariantGate = VariantGate(),
    duplicate_handling: DuplicateHandling = PLAIN,
    output_dir: Optional[Path] = None,
) -> Dict[IndexSpec, Path]:
    """Build per-(mode, depth) bytes and write canonical filenames.

    The canonical filename grammar (plan §D5, regex-locked in
    :mod:`._reader`)::

        <binary>_sorted_<mode>_d<depth>.idx

    where ``<mode>`` is :meth:`LengthReduction.filename_tag` and
    ``<depth>`` is zero-padded to three digits. The gating + duplicate
    parameters affect file CONTENT only; the filename scheme is
    unchanged.

    Parameters mirror :func:`build_sorted_index_bytes`;
    ``output_dir`` defaults to ``memmap_dir`` (the conventional
    placement next to the other per-binary sidecars) and is created if
    missing. Returns ``{IndexSpec -> written path}``.
    """
    if not reductions or not depths:
        return {}

    target_dir = Path(output_dir) if output_dir is not None else Path(memmap_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    per_spec_bytes = build_sorted_index_bytes(
        Path(memmap_dir),
        binary_name,
        reductions=reductions,
        depths=depths,
        gate=gate,
        duplicate_handling=duplicate_handling,
    )

    written: Dict[IndexSpec, Path] = {}
    for spec, blob in per_spec_bytes.items():
        filename = (
            f"{binary_name}_sorted_{spec.reduction.filename_tag()}"
            f"_d{spec.depth:03d}.idx"
        )
        path = target_dir / filename
        path.write_bytes(blob)
        written[spec] = path
    return written
