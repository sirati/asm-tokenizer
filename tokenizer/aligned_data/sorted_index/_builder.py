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

from ._dedup import PLAIN, DuplicateHandling
from ._gating import VariantGate
from ._length_compute import compute_reduced_lengths
from ._prepass import read_section_variant_info
from ._types import IndexSpec, LengthReduction
from ._wire import encode_sorted_index


__all__ = ["build_sorted_index_bytes", "write_sorted_index_files"]


def build_sorted_index_bytes(
    session: BinarySession,
    base_path: Path,
    binary_name: str,
    *,
    reductions: List[LengthReduction],
    depths: List[int],
    gate: VariantGate = VariantGate(),
    duplicate_handling: DuplicateHandling = PLAIN,
) -> Dict[IndexSpec, bytes]:
    """Build per-(mode, depth) sorted-index bytes for one binary.

    Runs the cheap pre-pass (:func:`read_section_variant_info`) to
    recover the matched-arm variant counts + data-bin pointers, then
    performs ONE Stage 1+2 walk per chunk -- at ``max(depths)`` -- across
    ALL requested reductions AND depths via
    :func:`compute_reduced_lengths` (the cost-amortising property named
    in plan §D8: the walk is NOT repeated per reduction or per depth).
    Each per-(mode, depth) ``u32[num_sections]`` length array is then
    wire-encoded via :func:`encode_sorted_index`.

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
    depths
        Splice depths to materialise (one output per ``(reduction,
        depth)`` pair).  Empty list returns an empty dict (no walk).
    gate
        Top-level minimum-variant emission gate (defaults to disabled).
    duplicate_handling
        Top-level duplicate strategy (defaults to :data:`PLAIN`).

    Returns
    -------
    Dict[IndexSpec, bytes]
        ``{IndexSpec(reduction, depth) -> bytes}``.  Each value is the
        wire-encoded sorted index per :mod:`._wire` (LE u32 throughout).
    """
    if not reductions or not depths:
        return {}

    # Pre-pass (plan ALG-7): per-section variant counts + data-bin
    # pointers.  Counts drive the 0-variant pre-filter; pointers drive
    # the duplicate / minimum-variant feature.  ONE read of sections.bin.
    section_info = read_section_variant_info(base_path, binary_name)

    # ONE shared Stage 1+2 walk for all (mode, depth) pairs (plan §D8).
    per_spec_lengths = compute_reduced_lengths(
        session,
        section_info=section_info,
        depths=depths,
        reductions=reductions,
        gate=gate,
        duplicate_handling=duplicate_handling,
    )

    # Per-(mode, depth) wire encode.  Each call is independent.
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
    """Open a session, build per-(mode, depth) bytes, write filenames.

    The canonical filename grammar (plan §D5, regex-locked in
    :mod:`._reader`)::

        <binary>_sorted_<mode>_d<depth>.idx

    where ``<mode>`` is :meth:`LengthReduction.filename_tag` (``"max"``
    or ``"p<NN>"`` with zero-padded percentile) and ``<depth>`` is
    zero-padded to three digits so files lexsort by depth. The gating +
    duplicate parameters affect file CONTENT only; the filename scheme
    is unchanged (a consumer reading the ``.idx`` does not need to know
    which gate / duplicate policy produced it).

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
    depths
        Splice depths to write (one file per ``(reduction, depth)``
        pair).  Empty list returns an empty dict (no session opened).
    gate
        Top-level minimum-variant emission gate (defaults to disabled).
    duplicate_handling
        Top-level duplicate strategy (defaults to :data:`PLAIN`).
    output_dir
        Directory to write the ``.idx`` files into.  Defaults to
        ``memmap_dir`` (the conventional placement next to the other
        per-binary sidecars).  Created if it does not exist.

    Returns
    -------
    Dict[IndexSpec, Path]
        ``{IndexSpec(reduction, depth) -> path}`` for every written file.
    """
    if not reductions or not depths:
        return {}

    target_dir = Path(output_dir) if output_dir is not None else Path(memmap_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset = BinaryDataset(memmap_dir, binary_name, vocab_manager=None)
    with dataset.open_session() as session:
        per_spec_bytes = build_sorted_index_bytes(
            session,
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
