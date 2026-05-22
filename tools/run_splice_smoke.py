"""Phase 4 splice smoke driver against a pre-built memmap directory.

Walks every per-binary memmap dir produced by ``run_stage3.py`` (or the
production memmap_builder) and exercises
:py:meth:`tokenizer.aligned_data.loader.session.BinarySession.splice_with_callees`
at one or more ``max_depth`` values per matched function. Captures
per-binary + aggregate metrics and writes a versioned baseline JSON.

Output schema (``tools/splice_smoke_results.json``):

    {
      "schema_version": 1,
      "timestamp": "ISO-8601",
      "tip": "<git HEAD sha>",
      "memmap_dir": "<path>",
      "max_functions_per_binary": <int>,
      "per_binary": {
          "<binary_name>": {"depth_0": {...}, "depth_1": {...}, ...},
          ...
      },
      "aggregate": {"depth_0": {...}, ...},
      "ram_mb": {"peak": <float|null>, "avg": <float|null>},
      "wall_seconds": <float>
    }

Each ``depth_N`` block carries::

    {
      "functions_spliced": <int>,
      "total_real_tokens": <int>,
      "mean_length_ratio_vs_depth_0": <float>,
      "sentinel_count": <int>
    }

``mean_length_ratio_vs_depth_0`` is the per-function mean of
``len(spliced.real_tokens) / len(depth_0.real_tokens)`` across every
function that was spliced at BOTH depth 0 and depth N. For ``N == 0``
the ratio is trivially 1.0 (per function), so the mean is also 1.0.

The RAM block is populated only when this script is invoked under
``tools/measure_python_ram.sh -o <file>``: the wrapper writes a JSON
line carrying ``peak_bytes`` / ``avg_bytes`` which the caller can
manually fold into the smoke results. When run bare, both fields are
``null``.

CLI::

    python tools/run_splice_smoke.py \
        --memmap-dir /tmp/stage3_phase5_smoke \
        --vocab ~/lmu_smoke_corpus/unified_vocab.csv \
        --depths 0,1,3 \
        --max-functions-per-binary 20 \
        --results-json tools/splice_smoke_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from tokenizer.aligned_data.loader.aligned_data_loader import AlignedDataLoader
from tokenizer.tokens import Category


_IDENTITY_SENTINEL = 0xFFFF


# ---------------------------------------------------------------------------
# Per-binary discovery
# ---------------------------------------------------------------------------


def _discover_binaries(memmap_dir: Path) -> List[str]:
    """Return the sorted set of binary names present in ``memmap_dir``.

    A binary is recognised by the presence of ``<name>_index.bin`` (the
    matched-arm index sidecar emitted by the memmap builder).  Names
    are returned in deterministic alphabetical order so multiple runs
    against the same directory produce the same per-binary key order
    in the results JSON.
    """
    suffix = "_index.bin"
    names: List[str] = []
    for entry in memmap_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        # Skip unmatched arm's sidecar (``<name>_unmatched_index.bin``);
        # the matched arm is the entry point and pulls the unmatched
        # arm in via the same BinaryDataset.
        if stem.endswith("_unmatched"):
            continue
        names.append(stem)
    return sorted(names)


# ---------------------------------------------------------------------------
# Per-function splice + metric collection
# ---------------------------------------------------------------------------


def _count_sentinels(identities: Dict[Category, np.ndarray]) -> int:
    """Total count of ``0xFFFF`` sentinel values across every category."""
    total = 0
    for arr in identities.values():
        if arr.size:
            total += int((arr == _IDENTITY_SENTINEL).sum())
    return total


def _max_non_sentinel_per_category(
    identities: Dict[Category, np.ndarray],
) -> Dict[str, int]:
    """Largest non-sentinel compacted id per Category (``-1`` if empty
    or all-sentinel).

    Plan Decision 28 + the smoke baseline: after compaction the identity
    space is bounded by the count of UNIQUE identities in the spliced
    view, not by depth x per-function counter. This histogram lets the
    smoke baseline assert the bound stays far short of the u16 ceiling
    on real corpora.
    """
    per_category: Dict[str, int] = {}
    for category in Category:
        arr = identities.get(category)
        if arr is None or arr.size == 0:
            per_category[category.name] = -1
            continue
        non_sentinel = arr[arr != _IDENTITY_SENTINEL]
        if non_sentinel.size == 0:
            per_category[category.name] = -1
        else:
            per_category[category.name] = int(non_sentinel.max())
    return per_category


def _splice_one_binary(
    loader: AlignedDataLoader,
    binary_name: str,
    depths: Tuple[int, ...],
    max_functions: int,
) -> Dict[str, Dict[str, float]]:
    """Splice the first ``max_functions`` matched functions at every depth.

    Returns the per-depth metrics block for this binary.  Per-depth
    counters are keyed ``"depth_<N>"`` to match the on-disk JSON
    schema.  ``max_functions`` is a hard cap — the script never tries
    to splice more than this many functions per binary even if the
    arm has more, so RAM + wall time stay bounded on large corpora.
    """
    dataset = loader.datasets[binary_name]
    matched_count = int(getattr(dataset, "matched_count", 0))
    if matched_count == 0:
        return {f"depth_{d}": _empty_block() for d in depths}

    take = min(matched_count, max_functions)

    # Per-depth accumulators.  Length ratios are stored as Python lists
    # so the mean is computed once at the end.
    per_depth_functions: Dict[int, int] = {d: 0 for d in depths}
    per_depth_real_tokens: Dict[int, int] = {d: 0 for d in depths}
    per_depth_sentinels: Dict[int, int] = {d: 0 for d in depths}
    per_depth_ratios: Dict[int, List[float]] = {d: [] for d in depths}
    # Per-depth per-Category maximum compacted id observed across every
    # spliced function in this binary. ``-1`` baseline lets a function
    # that never emitted a non-sentinel id for a given Category leave
    # the running max untouched (the bound is the WORST case across
    # the sampled functions).
    per_depth_max_id: Dict[int, Dict[str, int]] = {
        d: {category.name: -1 for category in Category} for d in depths
    }

    with dataset.open_session() as sess:
        for func_idx in range(take):
            # depth=0 first so its real_token length is the ratio
            # denominator for the deeper depths.
            depth_0_len: Optional[int] = None
            for depth in depths:
                # ``splice_with_callees`` now returns ``list[DecodedFunction]``
                # of length ``min(max_variants, len(section.variants))``;
                # the smoke runner pins ``max_variants=1`` to keep the
                # baseline shape stable, so we unwrap the single stream
                # for per-function metrics.
                #
                # Deterministic per-(binary, func_idx) seed so the
                # baseline JSON is reproducible across runs even though
                # ``max_variants=1`` now SAMPLES one variant (legacy
                # ``version=0`` was a fixed pick). The seed deliberately
                # does NOT include ``depth`` -- a fresh ``default_rng``
                # built from the same SeedSequence is statistically
                # equivalent across depth iterations, so every depth
                # for a given (binary, func_idx) samples the SAME root
                # variant. That keeps the length-ratio vs depth-0
                # meaningful (same numerator + denominator function).
                # ``binary_name`` is folded in via its UTF-8 byte hash
                # (built-in ``hash()`` is process-randomized;
                # ``int.from_bytes`` is stable).
                name_seed = int.from_bytes(
                    binary_name.encode("utf-8"), "little"
                ) & 0xFFFFFFFF
                rng = np.random.default_rng(
                    np.random.SeedSequence([name_seed, func_idx])
                )
                spliced_streams = sess.splice_with_callees(
                    func_idx,
                    arm="matched",
                    max_depth=depth,
                    max_variants=1,
                    rng=rng,
                )
                spliced = spliced_streams[0]
                n_real = int(spliced.real_tokens.shape[0])
                per_depth_functions[depth] += 1
                per_depth_real_tokens[depth] += n_real
                per_depth_sentinels[depth] += _count_sentinels(
                    spliced.identities
                )
                per_function_max = _max_non_sentinel_per_category(
                    spliced.identities
                )
                for cat_name, max_id in per_function_max.items():
                    if max_id > per_depth_max_id[depth][cat_name]:
                        per_depth_max_id[depth][cat_name] = max_id
                if depth == 0:
                    depth_0_len = n_real
                    per_depth_ratios[depth].append(1.0)
                else:
                    if depth_0_len is None or depth_0_len == 0:
                        # depth_0 must be in ``depths`` and run first; if
                        # the user dropped depth 0 from the CLI list,
                        # fall back to ``len(spliced) / max(1, n_real)``
                        # against itself (ratio == 1.0).  The flag is the
                        # missing-depth-0 case; document in result.
                        per_depth_ratios[depth].append(float("nan"))
                    else:
                        per_depth_ratios[depth].append(n_real / depth_0_len)

    return {
        f"depth_{d}": {
            "functions_spliced": per_depth_functions[d],
            "total_real_tokens": per_depth_real_tokens[d],
            "mean_length_ratio_vs_depth_0": _safe_mean(per_depth_ratios[d]),
            "sentinel_count": per_depth_sentinels[d],
            "max_compacted_id_per_category": per_depth_max_id[d],
        }
        for d in depths
    }


def _safe_mean(values: List[float]) -> float:
    """Mean over finite values; ``0.0`` on an empty / all-NaN list.

    Ignoring NaNs (which only arise when the caller dropped depth 0
    from ``--depths``) keeps the ratio's "vs depth-0" semantics honest
    rather than emitting a misleading number.
    """
    finite = [v for v in values if v == v]  # filters NaN
    if not finite:
        return 0.0
    return sum(finite) / len(finite)


def _empty_block() -> Dict[str, float]:
    return {
        "functions_spliced": 0,
        "total_real_tokens": 0,
        "mean_length_ratio_vs_depth_0": 0.0,
        "sentinel_count": 0,
        "max_compacted_id_per_category": {
            category.name: -1 for category in Category
        },
    }


# ---------------------------------------------------------------------------
# Aggregation across binaries
# ---------------------------------------------------------------------------


def _aggregate(
    per_binary: Dict[str, Dict[str, Dict[str, float]]],
    depths: Tuple[int, ...],
) -> Dict[str, Dict[str, float]]:
    """Sum per-binary blocks into one corpus-wide block per depth.

    ``functions_spliced``, ``total_real_tokens``, ``sentinel_count``
    sum directly.  ``mean_length_ratio_vs_depth_0`` is the function-
    count weighted mean of the per-binary means (so an all-binary
    average isn't dominated by the smallest binary's outlier).
    """
    out: Dict[str, Dict[str, float]] = {}
    for d in depths:
        key = f"depth_{d}"
        functions = 0
        real_tokens = 0
        sentinels = 0
        ratio_weighted_sum = 0.0
        # Corpus-wide max-compacted-id per Category: take the max of
        # the per-binary maxima (so the bound is the worst case across
        # the full smoke set, not the average).
        per_category_max: Dict[str, int] = {
            category.name: -1 for category in Category
        }
        for block in per_binary.values():
            entry = block[key]
            functions += entry["functions_spliced"]
            real_tokens += entry["total_real_tokens"]
            sentinels += entry["sentinel_count"]
            ratio_weighted_sum += (
                entry["mean_length_ratio_vs_depth_0"]
                * entry["functions_spliced"]
            )
            for cat_name, max_id in entry.get(
                "max_compacted_id_per_category", {}
            ).items():
                if max_id > per_category_max[cat_name]:
                    per_category_max[cat_name] = max_id
        out[key] = {
            "functions_spliced": functions,
            "total_real_tokens": real_tokens,
            "mean_length_ratio_vs_depth_0": (
                ratio_weighted_sum / functions if functions else 0.0
            ),
            "sentinel_count": sentinels,
            "max_compacted_id_per_category": per_category_max,
        }
    return out


# ---------------------------------------------------------------------------
# Git tip helper -- short-circuits when ``git`` is unavailable
# ---------------------------------------------------------------------------


def _git_tip(cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except FileNotFoundError:
        pass
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memmap-dir",
        type=Path,
        default=Path("/tmp/stage3_phase5_smoke"),
        help="Per-binary memmap directory produced by run_stage3.py.",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        required=True,
        help="Path to the unified_vocab.csv used by the memmap builder.",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="0,1,3",
        help="Comma-separated max_depth values to splice at (default: 0,1,3).",
    )
    parser.add_argument(
        "--max-functions-per-binary",
        type=int,
        default=20,
        help="Cap on matched functions exercised per binary (default 20).",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help=(
            "If given, write the JSON results here AND to stdout. "
            "Default: tools/splice_smoke_results.json relative to this "
            "script."
        ),
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated binary-name filter (e.g. 'hello,ncat').",
    )
    args = parser.parse_args()

    memmap_dir = args.memmap_dir.expanduser().resolve()
    vocab_path = args.vocab.expanduser().resolve()
    if not memmap_dir.is_dir():
        print(f"memmap dir not found: {memmap_dir}", file=sys.stderr)
        return 2
    if not vocab_path.exists():
        print(f"unified vocab not found: {vocab_path}", file=sys.stderr)
        return 2

    depths: Tuple[int, ...] = tuple(
        int(x.strip()) for x in args.depths.split(",") if x.strip()
    )
    if not depths:
        print("--depths must list at least one integer", file=sys.stderr)
        return 2

    binary_names = _discover_binaries(memmap_dir)
    if args.only:
        keep = {x.strip() for x in args.only.split(",") if x.strip()}
        binary_names = [n for n in binary_names if n in keep]
    if not binary_names:
        print(f"no binaries discovered under {memmap_dir}", file=sys.stderr)
        return 2

    print(
        f"memmap_dir   : {memmap_dir}\n"
        f"vocab        : {vocab_path}\n"
        f"binaries     : {binary_names}\n"
        f"depths       : {list(depths)}\n"
        f"max_per_bin  : {args.max_functions_per_binary}",
        file=sys.stderr,
    )

    t_start = time.monotonic()

    loader = AlignedDataLoader(
        base_path=memmap_dir,
        binary_names=binary_names,
        unified_vocab_path=vocab_path,
    )

    per_binary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for binary_name in binary_names:
        t0 = time.monotonic()
        per_binary[binary_name] = _splice_one_binary(
            loader,
            binary_name,
            depths,
            args.max_functions_per_binary,
        )
        dt = time.monotonic() - t0
        summary = ", ".join(
            f"d{d}={per_binary[binary_name][f'depth_{d}']['functions_spliced']}"
            f"/{per_binary[binary_name][f'depth_{d}']['total_real_tokens']}rt"
            for d in depths
        )
        print(f"  {binary_name:<18}  {summary}  wall={dt:6.2f}s", file=sys.stderr)

    aggregate = _aggregate(per_binary, depths)
    wall_seconds = time.monotonic() - t_start

    repo_root = Path(__file__).resolve().parent.parent
    results = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tip": _git_tip(repo_root),
        "memmap_dir": str(memmap_dir),
        "max_functions_per_binary": args.max_functions_per_binary,
        "depths": list(depths),
        "per_binary": per_binary,
        "aggregate": aggregate,
        "ram_mb": {"peak": None, "avg": None},
        "wall_seconds": round(wall_seconds, 3),
    }

    results_path = args.results_json or (repo_root / "tools" / "splice_smoke_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"wrote {results_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
