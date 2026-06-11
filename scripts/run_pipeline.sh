#!/usr/bin/env bash
# Full local pipeline (tokenize -> unify-vocab -> build-memmap) over $SRC into
# $OUT, parametrized by an optional space-separated $EXCLUDE list of subfolders.
# Empty $EXCLUDE => include everything (z3 too). Launch plainly (detached):
#   nohup setsid env EXCLUDE= CORES=1 ./scripts/run_pipeline.sh > "$LOG" 2>&1 < /dev/null & disown
#
# --memprofile needs a delegated cgroup-v2 hierarchy to read per-task
# memory.current, so the tokenize python is wrapped in its OWN
# `systemd-run --user --scope -p Delegate=yes` (the scope's direct child must be
# `nix develop`, NOT an intervening bash — otherwise the bash sits in the scope
# root and the framework's `workers/` subtree_control write hits EACCES via
# cgroup-v2's no-internal-processes rule). unify-vocab/build-memmap run plain:
# no --memprofile => no nested-cgroup setup => no scope needed.
#
# Tokenize also runs --always-restart-worker (fresh Ghidra/JVM per binary so the
# JVM's never-returned committed heap can't ratchet RSS up across a heavy
# sequence like z3). At --cores 1 there's no big+big concurrency so culls are
# not expected, but the phase stays self-healing: between passes it drops
# cull-partials (an _output.csv with no _meta.json) and re-runs tokenize
# --skip-existing until a pass reports zero failed tasks (bounded by
# $MAX_TOK_PASSES). A pass that produces no completion marker (startup crash)
# aborts the pipeline loudly rather than advancing over a partial corpus.
set -euo pipefail
cd /home/sirati/devel/python/asm-tokenizer

OUT=${OUT:-out}
SRC=${SRC:-src}
CORES=${CORES:-1}
EXCLUDE=${EXCLUDE:-}                 # space-separated subfolders; empty = include all
MAX_TOK_PASSES=${MAX_TOK_PASSES:-5}
LOG_DIR=${LOG_DIR:-/tmp/regen_out_logs}
mkdir -p "$LOG_DIR"

excl_args=()
for x in $EXCLUDE; do excl_args+=(--exclude-subfolder "$x"); done

remove_partials() {
  # A cull/crash partial = _output.csv present but _meta.json absent
  # (_meta.json is the reliable completeness marker; 0-byte consts/strings
  # are normal). Only safe to call when no tokenizer is running.
  local removed=0 base
  while IFS= read -r csv; do
    base="${csv%_output.csv}"
    if [ ! -f "${base}_meta.json" ]; then
      rm -f -- "${base}_consts.txt" "${base}_function_ranges.txt" \
               "${base}_output.csv" "${base}_strings.bin" \
               "${base}_meta.json" "${base}_output.mapping.b64c"
      removed=$((removed + 1))
    fi
  done < <(find "$OUT" -name '*_output.csv')
  echo "  remove_partials: dropped $removed"
}

disk_watchdog() {
  # Hard safety against ENOSPC: analyzing huge MIPS z3 binaries can balloon
  # Ghidra/angr *transient* scratch in /tmp (same partition) to hundreds of GB
  # and fill the disk, which wedges the worker and kills the run (observed
  # 2026-06-03: 592GB consumed -> ENOSPC). If free space drops below the floor,
  # SIGKILL the active tokenizer worker (the disk hog) so its scratch is
  # reclaimed and the framework respawns/retries; the binary fails out rather
  # than taking the whole machine's disk down. Never fires for normal binaries.
  local floor_kb=$((200 * 1024 * 1024))   # 200 GB
  while sleep 30; do
    local avail
    avail=$(df -k --output=avail "$OUT" 2>/dev/null | tail -1 | tr -dc '0-9')
    [ -n "$avail" ] || continue
    if [ "$avail" -lt "$floor_kb" ]; then
      echo "$(date -Is) WATCHDOG: free $((avail/1024/1024))GB < 200GB floor — SIGKILL tokenizer worker(s) to prevent ENOSPC" >&2
      pkill -KILL -f 'tokenizer --dynamic_queue' 2>/dev/null || true
      sleep 90
    fi
  done
}

echo "=== $(date -Is) PIPELINE start  src=$SRC out=$OUT cores=$CORES exclude='${EXCLUDE:-<none>}' ==="
disk_watchdog & WD_PID=$!
trap 'kill "$WD_PID" 2>/dev/null || true' EXIT
echo "    disk watchdog armed (pid $WD_PID, 200GB floor)"

for pass in $(seq 1 "$MAX_TOK_PASSES"); do
  echo "=== $(date -Is) tokenize pass $pass/$MAX_TOK_PASSES ==="
  remove_partials
  passlog="$LOG_DIR/tok_pass_${pass}.log"
  set +e
  systemd-run --user --scope -p Delegate=yes --quiet \
    nix develop --command python -m dynrunner \
      --task tokenize --source "$SRC" --output "$OUT" \
      "${excl_args[@]}" --cores "$CORES" --skip-existing \
      --always-restart-worker --memprofile 2>&1 | tee "$passlog"
  set -e
  if ! grep -qE 'processing complete|Completed: [0-9]+/[0-9]+' "$passlog"; then
    echo "FATAL: tokenize pass $pass produced no completion marker (startup crash?) — aborting" >&2
    exit 1
  fi
  failed=$(grep -oE 'Failed tasks: [0-9]+' "$passlog" | grep -oE '[0-9]+' | head -1)
  echo "=== $(date -Is) tokenize pass $pass done: failed='${failed:-0}' ==="
  [ "${failed:-0}" = "0" ] && { echo "tokenize converged on pass $pass"; break; }
done
remove_partials

echo "=== $(date -Is) unify-vocab ==="
nix develop --command python -m dynrunner \
  --task unify-vocab --source "$OUT" --output "$OUT" "${excl_args[@]}" --cores "$CORES"

echo "=== $(date -Is) build-memmap ==="
nix develop --command python -m dynrunner \
  --task build-memmap --source "$OUT" --output "$OUT" "${excl_args[@]}" \
  --unified-vocab "$OUT/unified_vocab.csv" --cores "$CORES"

echo "RUN_PIPELINE_DONE $(date -Is)"
