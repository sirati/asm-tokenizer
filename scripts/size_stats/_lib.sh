#!/usr/bin/env bash
# Shared size-accounting primitives for the size_stats tools.
#
# Single concern: turning a directory tree into "(size, relpath)" rows and
# rendering byte totals for humans. The two reporters
# (phase1_size_stats.sh, memmap_size_stats.sh) own their own join + summary
# logic; everything they have in common about *measuring files* lives here so
# neither reporter re-implements the find/stat primitive or the byte formatter.
#
# Sourced, never executed. Callers `source "$(dirname "$0")/_lib.sh"`.

set -euo pipefail

# require_dir <path> <label> -- abort with a clear message if not a directory.
require_dir() {
    local path="$1" label="$2"
    if [[ ! -d "$path" ]]; then
        printf 'error: %s is not a directory: %s\n' "$label" "$path" >&2
        exit 2
    fi
}

# list_file_sizes <root> [prune_name ...]
# Emit one "<size_bytes>\t<relpath-under-root>" row per regular file under
# <root>, pruning any directory whose basename matches a given prune_name or
# starts with a dot. relpath uses find's %P (root-relative), which both
# reporters key their joins on. A single find walk -> no per-file stat().
list_file_sizes() {
    local root="$1"; shift
    local prune_expr=( '(' -name '.*' )
    local name
    for name in "$@"; do
        prune_expr+=( -o -name "$name" )
    done
    prune_expr+=( ')' )
    find "$root" -mindepth 1 "${prune_expr[@]}" -prune -o \
        -type f -printf '%s\t%P\n'
}

# hr <bytes> -- render a byte count as a human-readable string (awk, no numfmt
# dependency). Used only for the stdout summaries; the CSV rows stay raw bytes.
hr() {
    awk -v b="$1" 'BEGIN{
        split("B KiB MiB GiB TiB PiB", u, " ")
        i=1; x=b+0
        while (x>=1024 && i<6){ x/=1024; i++ }
        printf (i==1 ? "%d %s" : "%.2f %s"), x, u[i]
    }'
}
