# size_stats

Two standalone bash reporters that measure how the tokenizer pipeline's output
sizes relate to the raw input binaries. Both take `<SRC_ROOT> <OUT_ROOT>` (and
an optional report-CSV path) and run wherever invoked — they operate on plain
directory paths, so "apply to LMU" just means copying them to the gateway and
pointing them at the NFS trees.

* `SRC_ROOT` — the raw-binary root (`gen-binary-dataset/out/dataset`), holding
  `<pkg>/<variant>/<binname>` ELFs with sibling `<variant>.json` metadata.
* `OUT_ROOT` — the tokenizer `out/` dir (`tokenizer-<dataset>/out`).

Shared file-measurement primitives live in `_lib.sh` (a single `find`-with-%s
walk, byte humanizer); each reporter owns its own join + summary.

## `phase1_size_stats.sh` — per-binary: raw ELF vs its phase-1 `_output.csv`

For every phase-1 `*_output.csv` it pairs the output with the raw binary it came
from and emits `(raw_bytes, output_csv_bytes, ratio)` per binary. The join is
exact: the output anchor's parent directory relative to `OUT_ROOT`
(`<pkg>/<variant>/<binname>`) is identical to the raw binary's path relative to
`SRC_ROOT` — no fragile parsing of the canonical
`<arch>-<comp>-<ver>-<opt>_<pkg>__<hash>` base name (which never appears on the
source side). `.json` sidecars and the phase-1 sidecars
(`_meta.json`, `_strings.bin`, `_function_ranges.txt`) are not counted; only the
`_output.csv` artifact is.

Report CSV columns: `binary,raw_bytes,output_csv_bytes,ratio_out_over_raw`.

## `memmap_size_stats.sh` — aggregate: all memmap artifacts vs sum of raw binaries

Sums every file under `OUT_ROOT/build_memmap/` **except** the `*sections.csv`
mirrors (`<prog>_sections.csv` and `<prog>_unmatched_sections.csv` — the verbose
CSV mirrors of the packed `*_sections.bin` catalogs) and reports that total
against the sum of all raw binary bytes. The ratio is corpus-level on purpose: a
`build_memmap/<program>/` dir is keyed by *binary name* (merging every variant,
hash-disambiguated on collision, e.g. `<hash>_pzstd`), so there is no clean 1:1
raw denominator per program. Per-program memmap byte totals are still listed for
inspection.

Report CSV columns: `program,memmap_bytes,sections_csv_excluded_bytes`.

## Applied: `gen-binary-dataset` → `tokenizer-gen-binary-dataset` (LMU, 2026-06-23)

Run on the LMU gateway against
`/home/k/kruppb/BIG/slurm/{gen-binary-dataset/out/dataset, tokenizer-gen-binary-dataset/out}`.
Captured outputs in `reports/`.

Phase-1 (per-binary):

| metric | value |
|---|---|
| raw binaries on disk | 167,248 |
| phase-1 outputs paired | 83,971 (outputs with no matching source: 0) |
| sources without an output | 83,277 |
| raw total (paired) | 53.57 GiB |
| `_output.csv` total | 110.71 GiB |
| overall output/raw | **2.07×** |
| per-file ratio | mean 2.14×, min 0.0006×, max 4.88× |

Memmap (aggregate, after the phase-3 build completed):

| metric | value |
|---|---|
| programs (binaries) | 133 |
| memmap total (counted) | 80.07 GiB |
| `*sections.csv` excluded | 3.32 GiB |
| raw binary total (all 167,248) | 81.95 GiB |
| memmap / all-raw | **0.98×** |
| memmap / tokenized-raw (53.57 GiB) | **1.49×** |

Note: roughly half the raw binaries lie outside the tokenized set, so the memmap
(built from the tokenized subset) is 0.98× of the *whole* raw corpus but 1.49× of
the *tokenized* subset that actually fed it.

## Thesis write-up

`reports/dataset_size_stats.typ` is a self-contained, compilable Typst section
documenting these results (corpus, method, phase-1 and phase-3 tables,
discussion) for inclusion in the thesis via `#include`. Compile with
`typst compile reports/dataset_size_stats.typ`.
