"""CLI orchestration for corpus-statistics collection.

Single concern: wire discovery → axis parsing → vocab counting →
raw-size resolution → SQLite persistence, and print the run summary.
This is the only module that imports the others; each crossed boundary
is a typed dataclass.

Run from the repo root inside ``nix develop``::

    python -m scripts.collect_stats --out-root out --db corpus_stats.db \\
        --binaries-root ~/Downloads/Dataset-1 --rebuild --examples

``--binaries-root`` is repeatable and optional; without it ``raw_size``
is NULL for every binary.  ``--rebuild`` deletes an existing ``--db``
first; otherwise an existing DB is an error (so a stats run never
silently appends to stale tables).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

# The repo root must be importable so ``tokenizer.*`` resolves when this
# package is run as ``python -m scripts.collect_stats`` or directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.collect_stats import queries  # noqa: E402
from scripts.collect_stats.axes import parse_axes  # noqa: E402
from scripts.collect_stats.db import (  # noqa: E402
    BinaryRow,
    Phase1Row,
    Phase3Row,
    StatsDB,
)
from scripts.collect_stats.discovery import (  # noqa: E402
    discover_binaries,
    discover_phase3,
)
from scripts.collect_stats.raw_index import RawResolver  # noqa: E402
from scripts.collect_stats.vocab import count_vocab  # noqa: E402


@dataclass
class _Summary:
    """End-of-run tallies printed after the database is built."""

    binaries_seen: int = 0
    files_recorded: int = 0
    vocab_parsed: int = 0
    raw_resolved: int = 0
    raw_missing: int = 0
    unparseable: int = 0
    programs: int = 0

    def render(self) -> str:
        return (
            "Run summary:\n"
            f"  binaries seen   : {self.binaries_seen}\n"
            f"  files recorded  : {self.files_recorded}\n"
            f"  vocab parsed    : {self.vocab_parsed}\n"
            f"  raw resolved    : {self.raw_resolved}\n"
            f"  raw missing     : {self.raw_missing}\n"
            f"  unparseable axes: {self.unparseable}\n"
            f"  phase-3 programs: {self.programs}"
        )


def _build_binary_row(
    binary, resolver: RawResolver, summary: _Summary
) -> tuple[BinaryRow, list]:
    """Map a discovered :class:`BinaryDir` to its DB row + phase-1 files,
    updating the summary tallies along the way."""
    axes = parse_axes(binary.fullname)
    if not axes.parsed:
        summary.unparseable += 1
        print(f"WARNING: unparseable fullname axes: {binary.fullname!r}", file=sys.stderr)

    vocab_size = count_vocab(binary.output_csv)
    if vocab_size is not None:
        summary.vocab_parsed += 1

    # Raw binaries on disk are named by their FULLNAME
    # (``arm32-clang-3.5-O0_minigzip``), not by the parsed program
    # (``minigzip``); resolve by fullname so the size lookup hits.
    raw_path = resolver.resolve(binary.fullname)
    raw_size = raw_path.stat().st_size if raw_path is not None else None
    if raw_size is not None:
        summary.raw_resolved += 1
    else:
        summary.raw_missing += 1

    row = BinaryRow(
        fullname=binary.fullname,
        program=axes.program,
        package=binary.package,
        isa_exact=axes.isa_exact,
        isa_family=axes.isa_family,
        bitness=axes.bitness,
        comp=axes.comp,
        comp_version=axes.comp_version,
        optim_level=axes.optim_level,
        raw_size=raw_size,
        vocab_size=vocab_size,
    )
    return row, binary.files


def _populate(db: StatsDB, out_root: Path, resolver: RawResolver) -> _Summary:
    """Discover, transform, and persist every binary and phase-3 program."""
    summary = _Summary()

    for binary in discover_binaries(out_root):
        summary.binaries_seen += 1
        row, phase1_files = _build_binary_row(binary, resolver, summary)
        binary_id = db.insert_binary(row)
        db.insert_phase1_files(
            [Phase1Row(binary_id, f.kind, f.size_bytes) for f in phase1_files]
        )
        summary.files_recorded += len(phase1_files)

    for program in discover_phase3(out_root):
        summary.programs += 1
        db.insert_phase3_files(
            [Phase3Row(program.program, f.kind, f.size_bytes) for f in program.files]
        )
        summary.files_recorded += len(program.files)

    db.commit()
    return summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="collect_stats",
        description="Build an SQLite DB of corpus size statistics from an out/ tree.",
    )
    parser.add_argument(
        "--out-root", type=Path, required=True, help="Path to the phase-1/3 out/ tree."
    )
    parser.add_argument(
        "--db", type=Path, required=True, help="SQLite database file to write."
    )
    parser.add_argument(
        "--binaries-root",
        type=Path,
        action="append",
        default=[],
        help="Root to search for raw binaries by exact filename (repeatable).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete an existing --db before building.",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Print the example ratio queries and their results after building.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.db.exists():
        if not args.rebuild:
            print(
                f"ERROR: {args.db} already exists; pass --rebuild to overwrite.",
                file=sys.stderr,
            )
            return 1
        args.db.unlink()

    resolver = RawResolver(args.binaries_root)
    with StatsDB(args.db) as db:
        db.create_schema()
        summary = _populate(db, args.out_root, resolver)
        print(summary.render())

        if args.examples:
            with sqlite3.connect(args.db) as conn:
                queries.run_examples(conn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
