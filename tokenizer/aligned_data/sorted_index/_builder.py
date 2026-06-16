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

No :class:`BinarySession` involvement: the build reads the
``_index.bin`` locator + ``_sections.bin`` catalog for the splice
geometry and the matched-arm realized-length sidecar
(:mod:`tokenizer.aligned_data.realized_lengths`) for the per-variant
body lengths -- it never pages in ``_data.bin``, because the sidecar
(generated as the Phase-4a pass that runs BEFORE this build) already
carries every record's contributing body length. The sidecar is a HARD
precondition: an absent one fails loudly via
:func:`require_realized_lengths`, never a silent ``_data.bin`` recompute.
No CLI parsing here; no string-typed modes. Callers wanting CLI /
multi-binary fan-out go through :mod:`.__main__`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Import the reader + arm selector from the realized_lengths SUBMODULES
# (not the package ``__init__``): the package init pulls in the generator,
# whose catalog read imports ``sorted_index._prepass`` and so re-enters
# THIS package -- a circular import. The reader path
# (``_reader`` -> ``_format`` -> ``memmap_format``) is cycle-free.
from tokenizer.aligned_data.realized_lengths._format import MATCHED_ARM
from tokenizer.aligned_data.realized_lengths._reader import RealizedLengths

from ._dedup import PLAIN, DuplicateHandling
from ._gating import VariantGate
from ._length_compute import compute_reduced_lengths
from ._prepass import (
    index_locator_path,
    read_section_variant_info,
    sections_bin_path,
)
from ._types import IndexSpec, LengthReduction
from ._wire import encode_sorted_index


__all__ = [
    "build_sorted_index_bytes",
    "write_sorted_index_files",
    "sorted_index_input_paths",
]


def sorted_index_input_paths(
    memmap_dir: Path, binary_name: str
) -> List[Path]:
    """Every per-binary input file the build READS from ``memmap_dir``.

    The build's input footprint, owned where the build lives: the
    matched-section catalog + locator the pre-pass memmaps
    (:func:`sections_bin_path` / :func:`index_locator_path`) and the
    matched-arm realized-length sidecar pair the body-length compute
    consumes (``MATCHED_ARM`` lengths + CSR index). ``_data.bin`` is
    deliberately ABSENT -- the realized-length sidecar exists precisely
    so the build never pages it in.

    Returned in a fixed order; paths are the canonical on-disk locations
    whether or not they currently exist (a binary with no matched arm
    legitimately lacks some). A consumer staging these to node-local
    scratch joins through here instead of re-deriving the suffixes the
    pre-pass + realized-length format own, so the build's read set has a
    single source of truth.
    """
    base = Path(memmap_dir)
    return [
        index_locator_path(base, binary_name),
        sections_bin_path(base, binary_name),
        MATCHED_ARM.lengths_path(base, binary_name),
        MATCHED_ARM.index_path(base, binary_name),
    ]


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
    opens the matched-arm realized-length sidecar (the per-variant body
    lengths -- a HARD precondition; an absent sidecar raises a
    generator-pointing :class:`FileNotFoundError`, never a silent
    ``_data.bin`` recompute), and computes EVERY requested
    ``(reduction, depth)`` from one graph traversal via
    :func:`compute_reduced_lengths` (plan §D8: the heavy work is not
    repeated per reduction or per depth). Each resulting
    ``u32[num_sections]`` array is wire-encoded via
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

    # Matched-arm body lengths come from the realized-length sidecar
    # (Phase-4a), never a fresh ``_data.bin`` geometry decode. The reader
    # pre-flights existence via ``require_realized_lengths`` (a missing
    # sidecar raises a generator-pointing FileNotFoundError); its flat
    # ``lengths`` body is section-major in the SAME ``var_offsets`` order
    # as the pre-pass, so it aligns element-for-element with the catalog.
    with RealizedLengths.open(base_path, binary_name, MATCHED_ARM) as rlen:
        body_lengths = np.asarray(rlen.lengths, dtype=np.int64)
        total_vars = int(section_info.cols.var_n_calls.size)
        if body_lengths.size != total_vars:
            raise ValueError(
                f"matched-arm realized-length sidecar for {binary_name!r} "
                f"carries {body_lengths.size} body lengths but the catalog "
                f"pre-pass has {total_vars} variants; the sidecar is stale "
                f"-- re-run the realized-lengths generator"
            )
        per_spec_lengths = compute_reduced_lengths(
            section_info,
            body_lengths,
            depths=depths,
            reductions=reductions,
            gate=gate,
            duplicate_handling=duplicate_handling,
        )

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
