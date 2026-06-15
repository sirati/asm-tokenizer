"""CLI entry for ``python -m tokenizer.aligned_data.sorted_index``.

Single concern: parse argv, discover binaries in a memmap directory,
and call :func:`write_sorted_index_files` once per binary with the
parsed (typed) reductions.  All text -> typed conversion happens at
the argparse boundary so the downstream pipeline is fully typed.

CLI surface (plan §D8 + plan §"Module layout"):

    python -m tokenizer.aligned_data.sorted_index \\
        --input-dir <memmap_dir> \\
        --mode max [--mode pNN ...] \\
        --depth 3 \\
        [--output-dir <dir>] \\
        [--only <binary>[,<binary>...]] \\
        [--max-binaries N]

Multiple ``--mode`` flags accumulate -- one shared walk per binary
produces one ``.idx`` file per requested reduction (plan §D8 cost-
amortisation).  ``--output-dir`` defaults to ``--input-dir`` (the
conventional sidecar-adjacent placement).

PRECONDITION: each binary's matched-arm realized-length sidecar must
already exist (run ``python -m
tokenizer.aligned_data.realized_lengths --input-dir <memmap_dir>``
first -- the Phase-4a pass). The build consumes those body lengths
instead of re-decoding ``_data.bin`` geometry; an absent sidecar raises
a generator-pointing :class:`FileNotFoundError`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from tools.batch_smoke._discovery import discover_binaries, filter_binaries

from ._builder import write_sorted_index_files
from ._dedup import DEDUP_BY_DATA_POINTER, PLAIN
from ._gating import VariantGate
from ._modes import parse_reduction
from ._types import LengthReduction


__all__ = ["main"]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser.

    Modes pass through :func:`parse_reduction` at the boundary via
    ``type=parse_reduction`` so ``args.mode`` is already
    ``List[LengthReduction]`` by the time ``main`` reads it.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tokenizer.aligned_data.sorted_index",
        description=(
            "Build per-binary sorted-index files for the matched-arm "
            "depth-N length sampler. Runs ONE shared Stage 1+2 walk per "
            "binary across every requested reduction. PRECONDITION: each "
            "binary's matched-arm realized-length sidecar must already "
            "exist (run `python -m tokenizer.aligned_data.realized_lengths "
            "--input-dir <dir>` first); this build consumes those body "
            "lengths instead of re-decoding _data.bin."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Memmap directory containing the per-binary sidecars.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        type=parse_reduction,
        required=True,
        metavar="MODE",
        help=(
            "Reduction mode (repeatable). Accepts 'max' or 'p<NN>' "
            "with 1 <= NN <= 99. 'p100' canonicalises to 'max'."
        ),
    )
    parser.add_argument(
        "--depth",
        action="append",
        type=int,
        required=True,
        dest="depth",
        metavar="DEPTH",
        help=(
            "Splice depth (repeatable). One shared Stage 1+2 walk runs "
            "at max(depths); every shallower depth is recovered as an "
            "exact prefix, producing one .idx per (mode, depth) pair."
        ),
    )
    parser.add_argument(
        "--min-variants",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Emit a section only if it has at least N top-level "
            "variants (duplicates included). 0 (default) disables."
        ),
    )
    parser.add_argument(
        "--min-variants-unique",
        type=int,
        default=0,
        metavar="M",
        help=(
            "Emit a section only if at least M of its top-level "
            "variants are UNIQUE (distinct data-bin pointer). Composes "
            "with --min-variants (requires M <= N). 0 (default) "
            "disables."
        ),
    )
    parser.add_argument(
        "--adjust-for-duplicates",
        action="store_true",
        help=(
            "Collapse top-level variants sharing a data-bin pointer "
            "into one item before reduction (PERCENTILE uses the group "
            "average, MAX the group max). Affects file content only, "
            "not the filename."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write .idx files into. Defaults to "
            "--input-dir."
        ),
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Comma-separated allow-list of binary names. Applied "
            "before --max-binaries."
        ),
    )
    parser.add_argument(
        "--max-binaries",
        type=int,
        default=None,
        help=(
            "Cap on number of binaries to process (after --only). "
            "Useful for smoke runs."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry. ``argv`` defaults to :data:`sys.argv[1:]`.

    Returns the process exit code (0 on success). Each produced file
    is announced on stdout, one path per line, so callers (and the
    smoke harness) can pipe the output into a follow-on tool.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    reductions: List[LengthReduction] = list(args.mode)
    depths: List[int] = list(args.depth)

    # VariantGate's __post_init__ validates the minimums (M <= N when
    # both set, non-negative); surface a bad combination as a CLI error.
    try:
        gate = VariantGate(
            min_variants=args.min_variants,
            min_variants_unique=args.min_variants_unique,
        )
    except ValueError as exc:
        parser.error(str(exc))
    duplicate_handling = (
        DEDUP_BY_DATA_POINTER if args.adjust_for_duplicates else PLAIN
    )

    discovered = discover_binaries(args.input_dir)
    selected = filter_binaries(
        discovered,
        only=args.only,
        max_binaries=args.max_binaries,
    )

    for binary_name in selected:
        written = write_sorted_index_files(
            args.input_dir,
            binary_name,
            reductions=reductions,
            depths=depths,
            gate=gate,
            duplicate_handling=duplicate_handling,
            output_dir=args.output_dir,
        )
        for path in written.values():
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
