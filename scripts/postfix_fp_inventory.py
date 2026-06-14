#!/usr/bin/env python3
"""Read-only inventory of the postfix-FP hallucination across the corpus.

Scans every ``*_output.csv`` under a root (default
``out/``) with the CONFIRMED discriminator (raw uint16 stream only, no
decoder) and records, per CSV, how many BROKEN postfix-FP floats it carries.
The discriminator + vocab parsing are imported from
``fix_postfix_fp_csv`` so there is a SINGLE source of truth for "what is a
broken float"; this module only counts and aggregates.

NO file is modified. Emits a machine-readable JSON inventory and prints a
per-package summary.

Output JSON shape::

    {
      "root": "<scanned root>",
      "totals": {
        "n_csvs_scanned": int,
        "n_affected_csvs": int,
        "n_affected_packages": int,
        "n_broken_floats": int,
        "affected_packages": [str, ...]
      },
      "per_package": {pkg: {"n_csvs": int, "n_affected_csvs": int,
                            "n_broken_floats": int}},
      "affected": [
        {"csv_path", "package", "variant", "n_broken_floats",
         "n_functions_with_broken", "float_type_ids"},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from tokenizer.compact_base64_utils import base64_to_ndarray_vec

# Single source of truth for the discriminator + vocab parse (validated).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fix_postfix_fp_csv import (  # noqa: E402
    PREAMBLE,
    TOKENS_COL,
    VOCAB_MARKER,
    VocabSnapshot,
    find_broken_float_positions,
)

OUTPUT_SUFFIX = "_output.csv"


def _variant_and_package(csv_path: Path, root: Path) -> tuple[str, str]:
    """Package = first path component under root; variant = file stem minus
    the ``_output`` suffix (e.g. ``x64-gcc-9-O0_z3``)."""
    rel = csv_path.relative_to(root)
    package = rel.parts[0] if len(rel.parts) > 1 else root.name
    variant = csv_path.name[: -len(OUTPUT_SUFFIX)]
    return variant, package


def scan_csv(csv_path: Path) -> tuple[int, int, list[int]]:
    """Return (n_broken_floats, n_functions_with_broken, sorted float_type_ids).

    Resolves the six float-type ids via THIS CSV's LAST vocab snapshot, then
    counts broken postfix floats across all function rows. Read-only.
    """
    last_vocab_row = None
    func_rows: list[str] = []
    with open(csv_path, newline="") as fh:
        preamble = fh.readline()
        if preamble.strip() != PREAMBLE:
            raise ValueError(f"{csv_path}: unexpected preamble {preamble!r}")
        reader = csv.reader(fh)
        header = next(reader)
        for r in reader:
            if not r:
                continue
            if r[0] == VOCAB_MARKER:
                last_vocab_row = r
            else:
                func_rows.append(r[TOKENS_COL])
    if last_vocab_row is None:
        raise ValueError(f"{csv_path}: no vocabulary snapshot row")

    snap = VocabSnapshot.parse(last_vocab_row)
    float_ids = snap.float_type_ids()
    if not float_ids:
        return 0, 0, []

    n_broken = 0
    n_funcs = 0
    for tok_b64 in func_rows:
        ids = base64_to_ndarray_vec(tok_b64).astype(np.int64)
        positions = find_broken_float_positions(ids, float_ids)
        if positions:
            n_broken += len(positions)
            n_funcs += 1
    return n_broken, n_funcs, sorted(float_ids)


def build_inventory(root: Path) -> dict:
    csvs = sorted(root.rglob(f"*{OUTPUT_SUFFIX}"))
    per_package: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_csvs": 0, "n_affected_csvs": 0, "n_broken_floats": 0}
    )
    affected: list[dict] = []
    n_scanned = 0
    errors: list[dict] = []

    for csv_path in csvs:
        variant, package = _variant_and_package(csv_path, root)
        per_package[package]["n_csvs"] += 1
        n_scanned += 1
        try:
            n_broken, n_funcs, float_ids = scan_csv(csv_path)
        except Exception as exc:  # never abort the sweep on one bad file
            errors.append({"csv_path": str(csv_path), "error": repr(exc)})
            continue
        if n_broken:
            per_package[package]["n_affected_csvs"] += 1
            per_package[package]["n_broken_floats"] += n_broken
            affected.append(
                {
                    "csv_path": str(csv_path),
                    "package": package,
                    "variant": variant,
                    "n_broken_floats": n_broken,
                    "n_functions_with_broken": n_funcs,
                    "float_type_ids": float_ids,
                }
            )

    affected_packages = sorted(
        p for p, s in per_package.items() if s["n_affected_csvs"]
    )
    totals = {
        "n_csvs_scanned": n_scanned,
        "n_affected_csvs": len(affected),
        "n_affected_packages": len(affected_packages),
        "n_broken_floats": sum(a["n_broken_floats"] for a in affected),
        "affected_packages": affected_packages,
    }
    return {
        "root": str(root),
        "totals": totals,
        "per_package": dict(sorted(per_package.items())),
        "affected": sorted(affected, key=lambda a: a["csv_path"]),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("out"),
        help="root dir to scan recursively for *_output.csv (default: out)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("/tmp/postfix_fp_inventory.json"),
        help="where to write the JSON inventory",
    )
    args = parser.parse_args(argv)

    csv.field_size_limit(1 << 30)

    inv = build_inventory(args.root.resolve())
    args.json_out.write_text(json.dumps(inv, indent=2))

    t = inv["totals"]
    print(f"scanned {t['n_csvs_scanned']} CSVs under {inv['root']}")
    print(
        f"AFFECTED: {t['n_affected_csvs']} CSVs across "
        f"{t['n_affected_packages']} packages; "
        f"{t['n_broken_floats']} broken postfix floats total"
    )
    print(f"affected packages: {t['affected_packages']}")
    print("per-package (csvs / affected / broken):")
    for pkg, s in inv["per_package"].items():
        if s["n_affected_csvs"]:
            print(
                f"  {pkg}: {s['n_csvs']} csvs, {s['n_affected_csvs']} affected, "
                f"{s['n_broken_floats']} broken"
            )
    if inv["errors"]:
        print(f"WARNING: {len(inv['errors'])} CSVs errored during scan:")
        for e in inv["errors"][:10]:
            print(f"  {e['csv_path']}: {e['error']}")
    print(f"JSON inventory -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
