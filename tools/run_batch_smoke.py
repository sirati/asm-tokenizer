"""End-to-end :func:`batch_decode` smoke driver against a real memmap dir.

Walks every per-binary memmap dir produced by ``run_stage3.py`` (or the
production ``memmap_builder``) and exercises
:func:`tokenizer.aligned_data.loader.batch_decode.batch_decode` over each
binary's matched arm, capping the number of sections per binary so RAM /
wall time stay bounded on large corpora. Captures per-binary + aggregate
shape metrics and writes a versioned baseline JSON.

The per-concern helpers (discovery, per-session metric capture,
aggregation, git-tip) live in :mod:`tools.batch_smoke`; this file owns
the CLI surface + JSON output composition.

Output schema (``tools/batch_smoke_results.json`` by default)::

    {
      "schema_version": 1,
      "timestamp": "ISO-8601",
      "tip": "<git HEAD sha>",
      "memmap_dir": "<path>",
      "config": {
          "num_variants_per_section": <int>,
          "context_len": <int>,
          "max_depth": <int>,
          "variant_padding": "<enum-value>",
          "max_functions_per_binary": <int>,
          "seed": <int>
      },
      "per_binary": {
          "<binary_name>": {
              "batch_size": <int>,
              "tokens_shape": [<rows>, <cols>],
              "total_identity_chunks": <int>,
              "total_number_chunks": <int>,
              "section_count": <int>,
              "wall_seconds": <float>
          },
          ...
      },
      "aggregate": {
          "batch_size": <int>,
          "total_tokens": <int>,
          "total_identity_chunks": <int>,
          "total_number_chunks": <int>,
          "section_count": <int>
      },
      "wall_seconds": <float>
    }

CLI::

    python tools/run_batch_smoke.py \\
        --memmap-dir /tmp/stage3_phase5_smoke \\
        --output tools/batch_smoke_results.json \\
        --max-functions-per-binary 32 \\
        --num-variants-per-section 3 \\
        --context-len 512 \\
        --max-depth 2 \\
        --variant-padding pad_null \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from tokenizer.aligned_data.loader.aligned_data_loader import AlignedDataLoader
from tokenizer.aligned_data.loader.batch_decode import VariantPadding

from tools.batch_smoke import (
    aggregate,
    collect_session_metrics,
    discover_binaries,
    filter_binaries,
    git_tip,
)


_VARIANT_PADDING_BY_FLAG = {p.value: p for p in VariantPadding}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memmap-dir",
        type=Path,
        required=True,
        help="Per-binary memmap directory produced by run_stage3.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "batch_smoke_results.json",
        help="Where to write the per-binary + aggregate JSON results.",
    )
    parser.add_argument(
        "--unified-vocab",
        type=Path,
        default=None,
        help=(
            "Override unified_vocab.csv path (default: "
            "<memmap-dir>/unified_vocab.csv)."
        ),
    )
    parser.add_argument(
        "--max-binaries",
        type=int,
        default=None,
        help="Cap on binaries processed (default: unlimited).",
    )
    parser.add_argument(
        "--max-functions-per-binary",
        type=int,
        default=32,
        help="Cap on matched sections exercised per binary (default 32).",
    )
    parser.add_argument(
        "--num-variants-per-section",
        type=int,
        default=3,
        help="Variant-axis count passed to batch_decode (default 3).",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=512,
        help="Per-row token budget passed to batch_decode (default 512).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Stage-1 inlining-depth bound for batch_decode (default 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the variant-sampling RNG (default 42).",
    )
    parser.add_argument(
        "--variant-padding",
        choices=sorted(_VARIANT_PADDING_BY_FLAG.keys()),
        default=VariantPadding.PAD_NULL.value,
        help="Variant-axis padding policy (default pad_null).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated binary-name filter (e.g. 'hello,ncat').",
    )
    return parser


def _run_one_binary(
    loader: AlignedDataLoader,
    binary_name: str,
    *,
    num_variants_per_section: int,
    context_len: int,
    max_depth: int,
    variant_padding: VariantPadding,
    max_functions_per_binary: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Open a session on ``binary_name`` and collect its metrics.

    Threading through ``AlignedDataLoader`` rather than constructing
    ``BinaryDataset`` directly so the unified-vocab gate fires once at
    loader construction and every per-binary session shares the same
    validated vocab manager.
    """
    dataset = loader.datasets[binary_name]
    matched_count = int(getattr(dataset, "matched_count", 0))
    with dataset.open_session() as sess:
        return collect_session_metrics(
            sess,
            matched_count,
            num_variants_per_section=num_variants_per_section,
            context_len=context_len,
            max_depth=max_depth,
            variant_padding=variant_padding,
            max_functions_per_binary=max_functions_per_binary,
            rng=rng,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    memmap_dir = args.memmap_dir.expanduser().resolve()
    if not memmap_dir.is_dir():
        print(f"memmap dir not found: {memmap_dir}", file=sys.stderr)
        return 2

    variant_padding = _VARIANT_PADDING_BY_FLAG[args.variant_padding]

    binary_names = filter_binaries(
        discover_binaries(memmap_dir),
        only=args.only,
        max_binaries=args.max_binaries,
    )
    if not binary_names:
        print(f"no binaries discovered under {memmap_dir}", file=sys.stderr)
        return 2

    unified_vocab = (
        args.unified_vocab.expanduser().resolve()
        if args.unified_vocab is not None
        else memmap_dir / "unified_vocab.csv"
    )

    print(
        f"memmap_dir   : {memmap_dir}\n"
        f"unified_vocab: {unified_vocab}\n"
        f"binaries     : {binary_names}\n"
        f"max_per_bin  : {args.max_functions_per_binary}\n"
        f"context_len  : {args.context_len}",
        file=sys.stderr,
    )

    t_start = time.monotonic()
    loader = AlignedDataLoader(
        base_path=memmap_dir,
        binary_names=binary_names,
        unified_vocab_path=unified_vocab,
    )

    # Single Generator shared across binaries; deterministic across runs
    # for a given ``--seed`` because the per-binary order is the sorted
    # discovery order (plus the deterministic ``--only`` filter).
    rng = np.random.default_rng(args.seed)

    per_binary: Dict[str, Dict[str, Any]] = {}
    for binary_name in binary_names:
        block = _run_one_binary(
            loader,
            binary_name,
            num_variants_per_section=args.num_variants_per_section,
            context_len=args.context_len,
            max_depth=args.max_depth,
            variant_padding=variant_padding,
            max_functions_per_binary=args.max_functions_per_binary,
            rng=rng,
        )
        per_binary[binary_name] = block
        print(
            f"  {binary_name:<24}  "
            f"batch={block['batch_size']:>5}  "
            f"sections={block['section_count']:>5}  "
            f"idents={block['total_identity_chunks']:>7}  "
            f"nums={block['total_number_chunks']:>7}  "
            f"wall={block['wall_seconds']:6.3f}s",
            file=sys.stderr,
        )

    wall_seconds = time.monotonic() - t_start
    repo_root = Path(__file__).resolve().parent.parent
    results = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tip": git_tip(repo_root),
        "memmap_dir": str(memmap_dir),
        "config": {
            "num_variants_per_section": args.num_variants_per_section,
            "context_len": args.context_len,
            "max_depth": args.max_depth,
            "variant_padding": variant_padding.value,
            "max_functions_per_binary": args.max_functions_per_binary,
            "seed": args.seed,
        },
        "per_binary": per_binary,
        "aggregate": aggregate(per_binary),
        "wall_seconds": round(wall_seconds, 3),
    }

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
