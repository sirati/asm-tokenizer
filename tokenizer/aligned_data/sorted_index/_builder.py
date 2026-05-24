"""Per-binary sorted-index build entry (plan §"Module layout").

Single concern: glue the already-shipped pre-pass + length compute +
wire encode into one per-binary call, and offer a thin file-writing
wrapper that opens a :class:`BinarySession` and stamps the canonical
``<binary>_sorted_<mode>_d<depth>.idx`` filenames.

Boundary contract (the design-first sentence):

  *Given an open :class:`BinarySession` + the memmap directory + a
  binary name + a list of reductions + a depth, run ONE shared Stage
  1+2 walk per chunk via :func:`compute_reduced_lengths` and return
  one wire-encoded blob per requested reduction.  The file-writing
  wrapper layers session lifecycle + filename construction on top --
  it owns no compute logic.*

No CLI parsing here; no string-typed modes; no batch-decode imports
beyond the typed parameters already consumed by
:mod:`._length_compute`.  Callers wanting CLI / multi-binary fan-out
go through :mod:`.__main__`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.session import BinarySession

from ._length_compute import (
    _count_variants_per_section,
    compute_reduced_lengths,
)
from ._types import LengthReduction
from ._wire import encode_sorted_index


__all__ = ["build_sorted_index_bytes", "write_sorted_index_files"]


def build_sorted_index_bytes(
    session: BinarySession,
    base_path: Path,
    binary_name: str,
    *,
    reductions: List[LengthReduction],
    depth: int,
) -> Dict[LengthReduction, bytes]:
    """Build per-mode sorted-index bytes for one binary.

    Runs the cheap pre-pass (``_count_variants_per_section``) to recover
    the matched-arm variant counts, then performs ONE Stage 1+2 walk
    per chunk across ALL requested reductions via
    :func:`compute_reduced_lengths` (the cost-amortising property named
    in plan §D8 -- ``compute_reduced_lengths`` is NOT called once per
    reduction).  Each per-mode ``u32[num_sections]`` length array is
    then wire-encoded via :func:`encode_sorted_index`.

    Parameters
    ----------
    session
        Open :class:`BinarySession` for ``binary_name``.  Must remain
        live for the duration of the call -- Stage 1 loads variant
        bodies through it.
    base_path
        Memmap directory containing ``<binary>_sections.bin`` /
        ``<binary>_index.bin`` etc.  The pre-pass reads through this
        path; the session was opened from a :class:`BinaryDataset`
        rooted at the same directory.
    binary_name
        The binary's name (the ``<binary>`` prefix on the per-binary
        sidecars).
    reductions
        :class:`LengthReduction` modes to compute.  Empty list is
        permitted and returns an empty dict (no walk runs).
    depth
        Maximum splice depth fed to the Stage 1+2 walk.  Encoded
        in the output filename as ``_d<depth:03d>`` by
        :func:`write_sorted_index_files`.

    Returns
    -------
    Dict[LengthReduction, bytes]
        ``{reduction -> bytes}``.  Key identity matches the input
        ``reductions`` list -- callers may index by their own
        :class:`LengthReduction` instances.  Each value is the wire-
        encoded sorted index per :mod:`._wire` (LE u32 throughout).
    """
    if not reductions:
        return {}

    # Pre-pass (plan ALG-7): per-section variant counts.  Required for
    # the 0-variant pre-filter inside ``compute_reduced_lengths``.
    section_variant_counts = _count_variants_per_section(base_path, binary_name)
    num_sections = int(section_variant_counts.size)

    # ONE shared Stage 1+2 walk for all reductions (plan §D8).
    per_mode_lengths = compute_reduced_lengths(
        session,
        num_sections=num_sections,
        section_variant_counts=section_variant_counts,
        depth=depth,
        reductions=reductions,
    )

    # Per-mode wire encode.  Each call is independent.
    return {
        reduction: encode_sorted_index(per_mode_lengths[reduction])
        for reduction in reductions
    }


def write_sorted_index_files(
    memmap_dir: Path,
    binary_name: str,
    *,
    reductions: List[LengthReduction],
    depth: int,
    output_dir: Optional[Path] = None,
) -> Dict[LengthReduction, Path]:
    """Open a session, build per-mode bytes, write canonical filenames.

    The canonical filename grammar (plan §D5, regex-locked in
    :mod:`._reader`)::

        <binary>_sorted_<mode>_d<depth>.idx

    where ``<mode>`` is :meth:`LengthReduction.filename_tag` (``"max"``
    or ``"p<NN>"`` with zero-padded percentile) and ``<depth>`` is
    zero-padded to three digits so files lexsort by depth.

    Parameters
    ----------
    memmap_dir
        Per-binary memmap directory -- the source of the session's
        sidecar files (sections.bin / index.bin / etc.).
    binary_name
        The binary's ``<binary>`` prefix.
    reductions
        :class:`LengthReduction` modes to write.  Empty list returns an
        empty dict (no session is opened).
    depth
        Splice depth.  Forwarded to :func:`build_sorted_index_bytes`
        and encoded in the filename.
    output_dir
        Directory to write the ``.idx`` files into.  Defaults to
        ``memmap_dir`` (the conventional placement next to the other
        per-binary sidecars).  Created if it does not exist.

    Returns
    -------
    Dict[LengthReduction, Path]
        ``{reduction -> path}`` for every written file.
    """
    if not reductions:
        return {}

    target_dir = Path(output_dir) if output_dir is not None else Path(memmap_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset = BinaryDataset(memmap_dir, binary_name, vocab_manager=None)
    with dataset.open_session() as session:
        per_mode_bytes = build_sorted_index_bytes(
            session,
            Path(memmap_dir),
            binary_name,
            reductions=reductions,
            depth=depth,
        )

    written: Dict[LengthReduction, Path] = {}
    for reduction, blob in per_mode_bytes.items():
        filename = (
            f"{binary_name}_sorted_{reduction.filename_tag()}"
            f"_d{depth:03d}.idx"
        )
        path = target_dir / filename
        path.write_bytes(blob)
        written[reduction] = path
    return written
