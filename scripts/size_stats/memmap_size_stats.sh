#!/usr/bin/env bash
# Aggregate size stats: phase-3 memmap artifacts vs the sum of the raw binaries.
#
# Single concern: sum the memmap artifact bytes (every file under
# `build_memmap/` EXCEPT the human-readable `*sections.csv` mirrors) and report
# that total against the sum of all raw binary bytes -- a corpus-level
# compaction ratio.
#
# Why aggregate (not per-binary like the phase-1 reporter): a `build_memmap/
# <program>/` directory is keyed by *binary name* and merges every variant of
# that binary (and name collisions are hash-disambiguated, e.g.
# `<hash>_pzstd`), so there is no clean 1:1 raw-binary denominator per program.
# The well-defined comparison is "all memmap bytes" vs "the sum of the binary
# files", which is what the task asks for. Per-program memmap byte totals are
# still listed in the CSV for inspection; the ratio is corpus-level.
#
# Exclusion rule: `*sections.csv` (both `<prog>_sections.csv` and
# `<prog>_unmatched_sections.csv`) are the verbose CSV mirrors of the packed
# `*_sections.bin` catalogs; excluded per the task spec ("ignoring sections.csv
# but not .bin"). Everything else -- *.bin, *.idx, *.txt, *.log, _variants.csv,
# unified_vocab.csv -- is counted.
#
# Usage: memmap_size_stats.sh <SRC_ROOT> <OUT_ROOT> [REPORT_CSV]
#   SRC_ROOT    raw-binary root (the `dataset/` dir holding <pkg>/<variant>/<binname>)
#   OUT_ROOT    tokenizer `out/` dir holding the `build_memmap/` subtree
#   REPORT_CSV  optional path for the per-program CSV (default: stdout-only summary)

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

if [[ $# -lt 2 || $# -gt 3 ]]; then
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
fi

SRC_ROOT="${1%/}"
OUT_ROOT="${2%/}"
REPORT_CSV="${3:-}"
require_dir "$SRC_ROOT" SRC_ROOT
require_dir "$OUT_ROOT" OUT_ROOT

MEMMAP_ROOT="$OUT_ROOT/build_memmap"
require_dir "$MEMMAP_ROOT" "OUT_ROOT/build_memmap"

awk -F'\t' -v report="$REPORT_CSV" '
function hr(b,   u, i, x) {
    split("B KiB MiB GiB TiB PiB", u, " ")
    i=1; x=b+0
    while (x>=1024 && i<6) { x/=1024; i++ }
    return sprintf((i==1 ? "%d %s" : "%.2f %s"), x, u[i])
}
# Pass 1: raw binaries -- the aggregate denominator. Skip the .json sidecars
# (metadata, not binaries). All other files under SRC_ROOT are raw binaries.
FNR==NR {
    if ($2 ~ /\.json$/) next
    raw_tot += $1
    raw_count++
    next
}
# Pass 2: memmap artifacts. relpath = <program>/<file>.
{
    n = split($2, p, "/")
    prog = p[1]
    file = p[n]
    seen[prog] = 1
    if (file ~ /sections\.csv$/) { excl[prog] += $1; excl_tot += $1; next }
    mem[prog] += $1
    mem_tot += $1
}
END {
    for (prog in seen) {
        prog_count++
        if (report != "")
            printf "%s,%d,%d\n", prog, mem[prog], excl[prog] > report
    }
    printf "=== phase-3 memmap size stats (build_memmap artifacts vs sum of raw binaries) ===\n"
    printf "programs (binaries):    %d\n", prog_count
    printf "raw binary files:       %d\n", raw_count
    printf "memmap total (counted): %s (%d B)\n", hr(mem_tot), mem_tot
    printf "sections.csv excluded:  %s (%d B)\n", hr(excl_tot), excl_tot
    printf "raw binary total:       %s (%d B)\n", hr(raw_tot), raw_tot
    if (raw_tot > 0)
        printf "overall memmap/raw:     %.4fx\n", mem_tot / raw_tot
    if (report != "")
        printf "per-program CSV:        %s\n", report
}
' <(list_file_sizes "$SRC_ROOT") <(list_file_sizes "$MEMMAP_ROOT")

# awk wrote the rows; prepend the header line.
if [[ -n "$REPORT_CSV" && -f "$REPORT_CSV" ]]; then
    { echo "program,memmap_bytes,sections_csv_excluded_bytes"; cat "$REPORT_CSV"; } > "${REPORT_CSV}.tmp"
    mv "${REPORT_CSV}.tmp" "$REPORT_CSV"
fi
