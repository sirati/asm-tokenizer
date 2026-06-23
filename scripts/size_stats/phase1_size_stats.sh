#!/usr/bin/env bash
# Per-binary size stats: raw input binary vs its phase-1 tokenizer output.
#
# Single concern: for every phase-1 `*_output.csv` artifact, pair it with the
# raw binary it was produced from and report (raw_bytes, output_bytes, ratio).
#
# Two figures are always reported:
#
#  * Aggregate (layout-agnostic): the sum of all `*_output.csv` bytes over the
#    sum of all raw binary bytes. This needs no pairing and is exact whenever
#    coverage is complete (every raw binary tokenised).
#  * Per-binary (structural join): each output is paired with its raw binary so
#    the per-file inflation distribution and the matched-only ratio (the true
#    inflation factor when coverage is partial) can be reported.
#
# The structural join tries two candidate source relpaths for each
# `<...>_output.csv` anchor, in order, and uses whichever names an existing raw
# binary:
#   1. the anchor's *parent directory* relative to OUT_ROOT  -- per-binary-subdir
#      layout, OUT_ROOT/<pkg>/<variant>/<binname>/<base>_output.csv paired with
#      SRC_ROOT/<pkg>/<variant>/<binname>;
#   2. the anchor path with the `_output.csv` suffix stripped -- flat layout,
#      OUT_ROOT/<pkg>/<base>_output.csv paired with SRC_ROOT/<pkg>/<base>.
# Both are membership tests against the actual raw-binary set, so the right one
# wins without per-dataset branching. Layouts whose output filename is mangled
# away from the source name (no structural correspondence) pair as zero; rely on
# the aggregate figure there.
#
# Usage: phase1_size_stats.sh <SRC_ROOT> <OUT_ROOT> [REPORT_CSV]
#   SRC_ROOT    raw-binary root holding the per-package binaries
#   OUT_ROOT    tokenizer `out/` dir holding per-package phase-1 outputs
#   REPORT_CSV  optional path for the per-binary CSV (default: stdout-only summary)
# Env:
#   SIZE_STATS_PRUNE  space-separated top-level source subdir names to exclude
#                     from the raw-binary walk (e.g. subset splits not tokenised)

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

# Phase-1 anchors only: prune the phase-3 / aux subtrees that carry no
# per-binary `_output.csv` (build_memmap, unify_vocab, dot-dirs like
# .publish-tmp), then keep the canonical output anchor.
phase1_anchors() {
    find "$OUT_ROOT" -mindepth 1 \
        \( -name '.*' -o -name build_memmap -o -name unify_vocab \) -prune -o \
        -type f -name '*_output.csv' -printf '%s\t%P\n'
}

awk -F'\t' -v report="$REPORT_CSV" '
function hr(b,   u, i, x) {
    split("B KiB MiB GiB TiB PiB", u, " ")
    i=1; x=b+0
    while (x>=1024 && i<6) { x/=1024; i++ }
    return sprintf((i==1 ? "%d %s" : "%.2f %s"), x, u[i])
}
# Pass 1: raw binaries. Key = relpath under SRC_ROOT. Skip the .json sidecars
# (they sit one level above the ELFs and are metadata, not binaries).
FNR==NR {
    if ($2 ~ /\.json$/) next
    src[$2] = $1
    src_count++
    rawall_tot += $1
    next
}
# Pass 2: phase-1 output anchors. Pair via parent-dir or de-suffixed relpath.
{
    rel = $2
    csv = $1
    csvall_tot += csv          # aggregate over ALL outputs (no pairing needed)
    nall++
    n = split(rel, p, "/")
    parent = p[1]
    for (i = 2; i < n; i++) parent = parent "/" p[i]
    desfx = rel
    sub(/_output\.csv$/, "", desfx)
    key = (parent in src) ? parent : ((desfx in src) ? desfx : "")
    if (key != "") {
        raw = src[key]
        matched++
        raw_tot += raw
        csv_tot += csv
        ratio = (raw > 0 ? csv / raw : 0)
        if (matched == 1 || ratio < min_r) min_r = ratio
        if (ratio > max_r) max_r = ratio
        ratio_sum += ratio
        if (report != "")
            printf "%s,%d,%d,%.6f\n", key, raw, csv, ratio > report
        delete src[key]   # so leftovers in src[] are the untokenized binaries
    } else {
        orphan++          # output with no matching source binary
    }
}
END {
    untok = src_count - matched
    printf "=== phase-1 size stats (raw binary vs _output.csv) ===\n"
    printf "raw binaries (SRC):     %d\n", src_count
    printf "phase-1 output anchors: %d\n", nall
    printf "--- aggregate (all outputs vs all raw binaries) ---\n"
    printf "raw total (all):        %s (%d B)\n", hr(rawall_tot), rawall_tot
    printf "output total (all):     %s (%d B)\n", hr(csvall_tot), csvall_tot
    if (rawall_tot > 0)
        printf "aggregate output/raw:   %.4fx\n", csvall_tot / rawall_tot
    printf "--- per-binary (structural pairing) ---\n"
    printf "outputs paired:         %d\n", matched
    printf "outputs w/o source:     %d\n", orphan
    printf "sources w/o output:     %d\n", untok
    printf "raw total (paired):     %s (%d B)\n", hr(raw_tot), raw_tot
    printf "output total (paired):  %s (%d B)\n", hr(csv_tot), csv_tot
    if (raw_tot > 0)
        printf "paired output/raw:      %.4fx\n", csv_tot / raw_tot
    if (matched > 0)
        printf "per-file ratio:         mean=%.4fx min=%.4fx max=%.4fx\n", ratio_sum / matched, min_r, max_r
    if (report != "")
        printf "per-binary CSV:         %s\n", report
}
' <(list_file_sizes "$SRC_ROOT" ${SIZE_STATS_PRUNE:-}) <(phase1_anchors)

# awk wrote the rows; prepend the header line.
if [[ -n "$REPORT_CSV" && -f "$REPORT_CSV" ]]; then
    { echo "binary,raw_bytes,output_csv_bytes,ratio_out_over_raw"; cat "$REPORT_CSV"; } > "${REPORT_CSV}.tmp"
    mv "${REPORT_CSV}.tmp" "$REPORT_CSV"
fi
