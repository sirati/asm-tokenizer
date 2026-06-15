"""CLI entry for ``python -m tokenizer.aligned_data.realized_lengths``.

Single concern: parse argv, discover binaries in a memmap directory, and
call :func:`generate_realized_lengths` once per binary. Mirrors the
``sorted_index`` CLI's discovery (``discover_binaries`` /
``filter_binaries``) so the two passes select the same binary set from
the same directory.

CLI surface::

    python -m tokenizer.aligned_data.realized_lengths \\
        --input-dir <memmap_dir> \\
        [--only <binary>[,<binary>...]] \\
        [--max-binaries N]

Each written sidecar path is announced on stdout, one per line, so the
output pipes into a follow-on tool exactly like the sorted_index CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from tools.batch_smoke._discovery import discover_binaries, filter_binaries

from ._generate import generate_realized_lengths
from ._geometry_generate import generate_realized_geometry


__all__ = ["main"]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tokenizer.aligned_data.realized_lengths",
        description=(
            "Generate per-binary realized-token sidecars for every binary "
            "in a memmap directory: the realized-length pair (_lengths.bin "
            "+ _lengths_index.bin) AND the realized-geometry superset pair "
            "(_realized.bin + _realized_index.bin), per arm. Runs BEFORE the "
            "sorted-index build, which later consumes these sidecars."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Memmap directory containing the per-binary sidecars.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Comma-separated allow-list of binary names. Applied before "
            "--max-binaries."
        ),
    )
    parser.add_argument(
        "--max-binaries",
        type=int,
        default=None,
        help="Cap on number of binaries to process (after --only).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry. ``argv`` defaults to :data:`sys.argv[1:]`.

    Returns the process exit code (0 on success). Each produced sidecar
    path is printed, one per line.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    discovered = discover_binaries(args.input_dir)
    selected = filter_binaries(
        discovered,
        only=args.only,
        max_binaries=args.max_binaries,
    )

    for binary_name in selected:
        for generate in (generate_realized_lengths, generate_realized_geometry):
            written = generate(args.input_dir, binary_name)
            for paths in written.values():
                for path in paths:
                    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
