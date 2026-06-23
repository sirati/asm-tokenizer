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

Optional env var `SIZE_STATS_PRUNE` (space-separated top-level source subdir
names) excludes subtrees from the raw-binary walk — e.g. non-tokenized subset
splits like BinaryCorp's `small_train`/`small_test`.

## `phase1_size_stats.sh` — per-binary: raw ELF vs its phase-1 `_output.csv`

Reports two figures. **Aggregate** (layout-agnostic): the sum of all
`*_output.csv` bytes over the sum of all raw binary bytes — needs no pairing and
is exact when coverage is complete. **Per-binary** (structural join): pairs each
output with its raw binary to recover the inflation distribution and a
coverage-robust paired ratio.

The join tries two source relpaths per output anchor, using whichever names an
existing raw binary: (1) the anchor's parent dir relative to `OUT_ROOT` — the
per-binary-subdir layout (`<pkg>/<variant>/<binname>/<base>_output.csv` ↔
`<pkg>/<variant>/<binname>`); (2) the anchor with `_output.csv` stripped — the
flat layout (`<pkg>/<base>_output.csv` ↔ `<pkg>/<base>`). Both are membership
tests against the raw set, so the right one wins with no per-dataset branching.
Layouts that mangle the name between source and output (e.g. BinaryCorp's
truncated build hash) don't pair structurally — rely on the aggregate there.
Only `_output.csv` is counted; `.json`/`_meta.json`/`_strings.bin`/
`_function_ranges.txt` sidecars are excluded.

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

## Applied: three corpora on LMU (2026-06-23)

Run on the LMU gateway. Roots:

| corpus | SRC_ROOT | OUT_ROOT | extra env |
|---|---|---|---|
| gen-binary-dataset | `BIG/slurm/gen-binary-dataset/out/dataset` | `BIG/slurm/tokenizer-gen-binary-dataset/out` | — |
| Dataset-1 | `~/corpus-v2` | `~/slurm/tokenizer/out` | memmap: `SIZE_STATS_PROG_STRIP='^.*_'` |
| BinaryCorp | `~/binarycorps` | `~/slurm/tokenizer-binarycorps/out` | `SIZE_STATS_PRUNE='small_train small_test'`; memmap `SIZE_STATS_PROG_STRIP='-O[0-9s]+-[0-9a-f]+$'` |

Captured outputs in `reports/` (`<corpus>_phase1_*`, `<corpus>_memmap_*`).

Phase-1 — token stream vs. the raw binaries that were tokenized:

| corpus | raw (tok.) | output | inflation | per-binary median |
|---|---|---|---|---|
| gen-binary-dataset | 53.57 GiB | 110.71 GiB | **2.07×** | 2.27× |
| Dataset-1 | 12.08 GiB | 29.66 GiB | **2.46×** | 1.98× |
| BinaryCorp | 24.16 GiB | 33.52 GiB | **1.39×** | — (no structural pairing; ~100% coverage so aggregate is exact) |

Phase-3 — `build_memmap` artefacts vs. the raw binaries present in those maps:

| corpus | programs (built) | memmap | sections.csv excl. | raw (in maps) | map / raw |
|---|---|---|---|---|---|
| gen-binary-dataset | 133 / 133 | 80.07 GiB | 3.32 GiB | 53.57 GiB | **1.49×** |
| Dataset-1 | 28 / 28 | 17.14 GiB | 1.14 GiB | 12.11 GiB | **1.42×** |
| BinaryCorp | 821 / 9,684 | 13.24 GiB | 702.15 MiB | 12.25 GiB | **1.08×** |

Notes: a memmap group merges every opt/build variant of a binary, so program
count ≪ binary count. The raw denominator is scoped to binaries whose program
has a built memmap. **BinaryCorp's phase-3 build is only 8.5% complete (821 of
9,684 programs)** — its 1.08× is the ratio over the built subset; rerun when the
build finishes.

## Thesis write-up

`reports/dataset_size_stats.typ` is a self-contained, compilable Typst section
documenting these results across all three corpora (corpora overview, method,
phase-1 and phase-3 comparison tables, discussion) for inclusion in the thesis
via `#include`. Compile with `typst compile reports/dataset_size_stats.typ`.
