"""Phase 3 (memmap_builder) end-to-end driver against the local LMU smoke corpus.

Walks ``~/lmu_smoke_corpus/{nmap,dataset/hello}/<variant>/*_output.csv``,
groups variants by binary name (``pkg``), and calls
:func:`tokenizer.memmap_builder.build_memmap_files` once per binary
group. Intended for use under ``tools/measure_python_ram.sh`` so the
per-invocation peak + avg RAM lands in a JSON line.

Output dir defaults to ``$STAGE3_OUTPUT_DIR`` env var, falling back to a
fresh ``tempfile.mkdtemp`` so consecutive runs don't collide. Override
via ``--output-dir`` if you want a known location.

Variants whose ``_meta.json`` is missing or whose ``.mapping.b64c`` is
absent are skipped with a stderr note — those would fail the builder's
unified-vocab gate anyway.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from tokenizer.memmap_builder.builder import BinaryVersionInfo, build_memmap_files
from tokenizer.variant_info import VariantInfo


def _discover_variants(corpus_root: Path) -> list[Path]:
    """Return every ``*_output.csv`` under nmap/ and dataset/hello/."""
    return sorted(
        list((corpus_root / "nmap").rglob("*_output.csv"))
        + list((corpus_root / "dataset" / "hello").rglob("*_output.csv"))
    )


def _build_version(csv_path: Path) -> BinaryVersionInfo | None:
    """One BinaryVersionInfo per CSV via ``VariantInfo.from_csv``.

    Returns ``None`` if the matching ``.mapping.b64c`` doesn't exist
    (vocab_unifier hasn't been run for this variant) — without it the
    builder would either skip the variant or fail loudly later, so we
    drop it up front and log the omission.
    """
    base = csv_path.name.removesuffix("_output.csv")
    mapping_path = csv_path.with_name(base + "_output.mapping.b64c")
    if not mapping_path.exists():
        return None
    try:
        info = VariantInfo.from_csv(csv_path)
    except Exception as exc:
        print(f"  skip {csv_path}: VariantInfo.from_csv failed: {exc}", file=sys.stderr)
        return None
    return BinaryVersionInfo(
        path=csv_path,
        mapping_path=mapping_path,
        arch=info.arch,
        compiler=info.compiler,
        compilerversion=info.compiler_version,
        opt=info.opt,
        pkg=info.pkg,
        variant_id=info.variant_id,
        extra_metadata=info.extra_metadata,
        filename=info.filename,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path.home() / "lmu_smoke_corpus",
        help="Root containing nmap/ + dataset/hello/ + unified_vocab.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write per-binary build_memmap output (default: fresh tempdir).",
    )
    parser.add_argument(
        "--unified-vocab",
        type=Path,
        default=None,
        help="Override unified_vocab.csv path (default: <corpus-root>/unified_vocab.csv).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated binary-name filter (e.g. 'hello,ncat'); default = all.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Python logging level (DEBUG/INFO/WARNING/ERROR). Default WARNING.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    corpus_root = args.corpus_root.expanduser().resolve()
    unified_vocab = args.unified_vocab or (corpus_root / "unified_vocab.csv")
    if not unified_vocab.exists():
        print(f"unified_vocab.csv not found at {unified_vocab}", file=sys.stderr)
        return 2

    if args.output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="stage3_bench_"))
    else:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"corpus_root  : {corpus_root}", file=sys.stderr)
    print(f"unified_vocab: {unified_vocab}", file=sys.stderr)
    print(f"output_dir   : {output_dir}", file=sys.stderr)

    csv_paths = _discover_variants(corpus_root)
    print(f"discovered   : {len(csv_paths)} _output.csv files", file=sys.stderr)

    groups: dict[str, list[BinaryVersionInfo]] = defaultdict(list)
    skipped = 0
    for csv_path in csv_paths:
        bvi = _build_version(csv_path)
        if bvi is None:
            skipped += 1
            continue
        groups[bvi.pkg].append(bvi)
    print(
        f"grouped      : {len(groups)} binaries, "
        f"{sum(len(v) for v in groups.values())} variants "
        f"({skipped} skipped — no mapping or unparseable axes)",
        file=sys.stderr,
    )

    if args.only:
        keep = {x.strip() for x in args.only.split(",") if x.strip()}
        groups = {k: v for k, v in groups.items() if k in keep}
        print(f"--only filter: kept {list(groups)}", file=sys.stderr)

    totals = {"binaries": 0, "variants": 0, "wall_s": 0.0}
    for binary_name, versions in sorted(groups.items()):
        t0 = time.monotonic()
        build_memmap_files(versions, output_dir, binary_name, unified_vocab)
        dt = time.monotonic() - t0
        totals["binaries"] += 1
        totals["variants"] += len(versions)
        totals["wall_s"] += dt
        print(
            f"  {binary_name:<18} variants={len(versions):>4}  wall={dt:6.2f}s",
            file=sys.stderr,
        )

    print(
        f"summary: binaries={totals['binaries']} variants={totals['variants']} "
        f"wall_s={totals['wall_s']:.2f} output={output_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
