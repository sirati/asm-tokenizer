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
# Env:
#   SIZE_STATS_PRUNE       space-separated top-level source subdir names to
#                          exclude from the raw-binary walk (subset splits etc.)
#   SIZE_STATS_PROG_STRIP  ERE removed from each raw binary's basename to derive
#                          its program name (e.g. '-O[0-9s]+-[0-9a-f]+$' for the
#                          opt/hash variant suffix). When set, the raw
#                          denominator is scoped to ONLY programs that have a
#                          built memmap, and memmap coverage is reported -- use
#                          this when the phase-3 build is partial so the ratio
#                          compares against the binaries actually in the memmaps,
#                          not the whole corpus.

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

awk -F'\t' -v report="$REPORT_CSV" -v strip="${SIZE_STATS_PROG_STRIP:-}" '
function hr(b,   u, i, x) {
    split("B KiB MiB GiB TiB PiB", u, " ")
    i=1; x=b+0
    while (x>=1024 && i<6) { x/=1024; i++ }
    return sprintf((i==1 ? "%d %s" : "%.2f %s"), x, u[i])
}
# Pass 1: raw binaries. Skip .json sidecars (metadata, not binaries). When a
# program-strip rule is given, derive each binary`s program name (basename minus
# the strip pattern, e.g. the opt/hash variant suffix) and accumulate raw bytes
# per program -- so the denominator can be scoped to only the programs that have
# a built memmap (the existing memmap files), rather than the whole corpus.
FNR==NR {
    if ($2 ~ /\.json$/) next
    rawall_tot += $1
    raw_count++
    if (strip != "") {
        nseg = split($2, sp, "/")
        prog = sp[nseg]
        sub(strip, "", prog)
        rawbyprog[prog] += $1
        if (!(prog in srcprog)) { srcprog[prog] = 1; src_prog_count++ }
    }
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
        if (strip != "") built_raw += rawbyprog[prog]
        if (report != "")
            printf "%s,%d,%d\n", prog, mem[prog], excl[prog] > report
    }
    raw_used = (strip != "" ? built_raw : rawall_tot)
    printf "=== phase-3 memmap size stats (build_memmap artifacts vs raw binaries) ===\n"
    printf "built programs:         %d\n", prog_count
    printf "raw binary files:       %d\n", raw_count
    printf "memmap total (counted): %s (%d B)\n", hr(mem_tot), mem_tot
    printf "sections.csv excluded:  %s (%d B)\n", hr(excl_tot), excl_tot
    if (strip != "") {
        printf "source programs total:  %d\n", src_prog_count
        printf "memmap coverage:        %d/%d programs (%.1f%%)\n", prog_count, src_prog_count, (src_prog_count>0 ? 100.0*prog_count/src_prog_count : 0)
        printf "raw (built programs):   %s (%d B)\n", hr(built_raw), built_raw
        printf "raw (all binaries):     %s (%d B)\n", hr(rawall_tot), rawall_tot
    } else {
        printf "raw binary total:       %s (%d B)\n", hr(rawall_tot), rawall_tot
    }
    if (raw_used > 0)
        printf "memmap/raw (built):     %.4fx\n", mem_tot / raw_used
    if (report != "")
        printf "per-program CSV:        %s\n", report
}
' <(list_file_sizes "$SRC_ROOT" ${SIZE_STATS_PRUNE:-}) <(list_file_sizes "$MEMMAP_ROOT")

# awk wrote the rows; prepend the header line.
if [[ -n "$REPORT_CSV" && -f "$REPORT_CSV" ]]; then
    { echo "program,memmap_bytes,sections_csv_excluded_bytes"; cat "$REPORT_CSV"; } > "${REPORT_CSV}.tmp"
    mv "${REPORT_CSV}.tmp" "$REPORT_CSV"
fi
