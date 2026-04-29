#!/usr/bin/env bash
# Multi-secondary test harness for asm-tokenizer's dynrunner integration.
#
# Wraps each test mode in a single shared systemd-user scope with cgroup
# limits set to 1/4 host RAM and 1/2 host cores. Inside the scope we
# also pass --cores / --max-memory so dynamic_runner's scheduler agrees
# with the kernel's cgroup budget (the runner reads from /proc/meminfo,
# not the cgroup).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT_DIR="$REPO_ROOT/src/zlib"
TEST_ROOT=$(mktemp -d /tmp/asm-multi-XXXXXX)
LOG_ROOT="$TEST_ROOT/logs"
mkdir -p "$LOG_ROOT"

# Resolve nix store path for podman so the inner shell doesn't have to
# re-evaluate `nixpkgs#podman` every time a secondary is spawned. The
# package has multiple outputs (binary + manpages); pick the one that
# actually contains `bin/podman`.
PODMAN_BIN=""
for p in $(nix build --no-link --print-out-paths nixpkgs#podman); do
  if [[ -x "$p/bin/podman" ]]; then
    PODMAN_BIN="$p/bin"
    break
  fi
done
if [[ -z "$PODMAN_BIN" ]]; then
  echo "could not locate podman binary in nixpkgs#podman outputs" >&2
  exit 1
fi

# Shared resources across the whole test:
#   - 1/4 of 96 GiB = 24 GiB
#   - 1/2 of 32 cores = 16 cores
LIMIT_MEM="24G"
LIMIT_CPU_QUOTA="1600%"
RUNNER_MAX_MEMORY="24G"
RUNNER_CORES="16"

# Same input set, same name regex, same compiler/platform filters.
# 6 binaries (4 zlib gcc-5 -O3 across x86/x64/arm32/arm64 plus the
# clang mips64). The clang one is in a different (compiler, version)
# slot so the run sees genuine spread.
BINARY_FILTERS=(--name-regex "minigzipsh" --platform x86 x64 arm32 arm64 mips32 mips64 --compiler gcc)

declare -A MODE_DESCRIPTIONS=(
  [single-process]="--multi-computer single-process --jobs 2 (in-process distributed primary + 2 in-memory secondaries)"
  [local-subprocess]="--multi-computer local --jobs 2 (network primary spawning 2 subprocess secondaries via QUIC)"
)

log() { printf '%s | %s\n' "$(date +%H:%M:%S)" "$*"; }

run_mode() {
  local mode="$1"
  shift
  local mode_args=("$@")
  local out_dir="$TEST_ROOT/out-$mode"
  mkdir -p "$out_dir"
  local log_file="$LOG_ROOT/$mode.log"

  log "== mode: $mode =="
  log "   ${MODE_DESCRIPTIONS[$mode]:-}"
  log "   output : $out_dir"
  log "   log    : $log_file"

  local rc
  cd "$REPO_ROOT"
  nix develop --command python -m dynrunner --task tokenize \
    --raw-logs \
    --source "$INPUT_DIR" \
    --output "$out_dir" \
    "${BINARY_FILTERS[@]}" \
    --cores "$RUNNER_CORES" \
    --max-memory "$RUNNER_MAX_MEMORY" \
    "${mode_args[@]}" \
    >"$log_file" 2>&1
  rc=$?
  log "   exit: $rc"
  return $rc
}

summarize_mode() {
  local mode="$1"
  local log_file="$LOG_ROOT/$mode.log"
  local out_dir="$TEST_ROOT/out-$mode"

  local completed
  completed=$(grep -E '^(P\|)?Completed: ' "$log_file" | tail -1 | awk -F'[: /]+' '{print $(NF-1)}')
  local errored
  errored=$(grep -E 'Errored: |^P\|Failed: ' "$log_file" | tail -1 | awk '{print $NF}')
  local csv_count
  csv_count=$(find "$out_dir" -maxdepth 1 -name '*_output.csv' 2>/dev/null | wc -l)
  local sec_count
  sec_count=$(grep -cE 'secondary[ -]?finished' "$log_file" 2>/dev/null || echo 0)

  printf '   %-18s completed=%s errored=%s csv-outputs=%s secondaries-finished=%s\n' \
    "$mode" "${completed:-?}" "${errored:-?}" "$csv_count" "$sec_count"
}

main() {
  log "## test root: $TEST_ROOT"
  log "## cgroup    : MemoryMax=$LIMIT_MEM, CPUQuota=$LIMIT_CPU_QUOTA"
  log "## runner    : --cores $RUNNER_CORES --max-memory $RUNNER_MAX_MEMORY"
  log "## input     : $INPUT_DIR"
  log "## host res  : $(grep MemTotal /proc/meminfo | awk '{print $2 " " $3}'), $(nproc) cores"
  log

  local rc_overall=0

  for mode in single-process local-subprocess; do
    case "$mode" in
      single-process)
        run_mode "$mode" --multi-computer single-process --jobs 2 || rc_overall=$?
        ;;
      local-subprocess)
        run_mode "$mode" --multi-computer local --jobs 2 || rc_overall=$?
        ;;
    esac
  done

  # Container mode: load the image into podman, run the same task set
  # inside a single container (cgroup-limited via --memory / --cpus, on
  # top of the outer systemd-user scope). This verifies the docker
  # image build, the entrypoint, and that PYTHONPATH wires the wheel
  # for `python -m dynrunner` inside the container.
  local podman_log="$LOG_ROOT/podman-container.log"
  local podman_out="$TEST_ROOT/out-podman-container"
  mkdir -p "$podman_out"
  log "== mode: podman-container =="
  log "   single-container: dynrunner --task tokenize inside the dockerImage"
  log "   output : $podman_out"
  log "   log    : $podman_log"
  cd "$REPO_ROOT"
  PATH="$PODMAN_BIN:$PATH" podman load -i result >>"$podman_log" 2>&1 || true
  PATH="$PODMAN_BIN:$PATH" podman run --rm \
    --memory="$LIMIT_MEM" --cpus=16 \
    -v "$INPUT_DIR:/app/src-network:ro" \
    -v "$podman_out:/app/output" \
    localhost/asm-tokenizer:latest dynrunner --task tokenize \
    --raw-logs --source /app/src-network --output /app/output \
    "${BINARY_FILTERS[@]}" \
    --max-memory "$RUNNER_MAX_MEMORY" --cores "$RUNNER_CORES" \
    >>"$podman_log" 2>&1
  local podman_rc=$?
  log "   exit: $podman_rc"
  [[ $podman_rc -ne 0 ]] && rc_overall=$podman_rc

  # Multi-secondary podman mode: each secondary in its own container,
  # primary on host, all sharing the outer systemd-user scope's
  # cgroup via --cgroup-parent. See podman_orchestrator.py for the
  # bind-mount story (host paths replicated inside the container) and
  # the /.dockerenv marker that flips _dispatch_secondary into
  # in_docker mode (so outputs land in /app/out-tmp instead of an
  # ephemeral tmpdir).
  local pmulti_log="$LOG_ROOT/podman-multi-secondary.log"
  local pmulti_root="$TEST_ROOT/podman-multi-secondary"
  mkdir -p "$pmulti_root"
  log "== mode: podman-multi-secondary =="
  log "   each secondary in its own container; primary on host"
  log "   output : $pmulti_root"
  log "   log    : $pmulti_log"
  cd "$REPO_ROOT"
  PATH="$PODMAN_BIN:$PATH" \
    nix develop --command python "$REPO_ROOT/test/multi_secondary/podman_orchestrator.py" \
      --raw-logs --num-secondaries 3 \
      --input-dir "$INPUT_DIR" \
      --output-root "$pmulti_root" \
      >>"$pmulti_log" 2>&1
  local pmulti_rc=$?
  log "   exit: $pmulti_rc"
  [[ $pmulti_rc -ne 0 ]] && rc_overall=$pmulti_rc

  log
  log "## summary"
  for mode in single-process local-subprocess; do
    summarize_mode "$mode"
  done
  local podman_csv
  podman_csv=$(find "$podman_out" -maxdepth 1 -name '*_output.csv' 2>/dev/null | wc -l)
  local podman_completed
  podman_completed=$(grep -E 'processing complete' "$podman_log" | tail -1 | grep -oE 'completed=[0-9]+' | cut -d= -f2)
  printf '   %-18s completed=%s csv-outputs=%s container=podman\n' \
    "podman-container" "${podman_completed:-?}" "$podman_csv"

  local pmulti_completed
  pmulti_completed=$(grep -E '^.* INFO completed=' "$pmulti_log" | tail -1 | grep -oE 'completed=[0-9]+' | cut -d= -f2)
  local pmulti_failed
  pmulti_failed=$(grep -E '^.* INFO completed=' "$pmulti_log" | tail -1 | grep -oE 'failed=[0-9]+' | cut -d= -f2)
  local pmulti_csv_total
  pmulti_csv_total=$(find "$pmulti_root" -name '*_output.csv' 2>/dev/null | wc -l)
  local pmulti_sec_with_csv
  pmulti_sec_with_csv=$(find "$pmulti_root" -mindepth 2 -name '*_output.csv' 2>/dev/null | awk -F/ '{print $(NF-1)}' | sort -u | wc -l)
  printf '   %-18s completed=%s failed=%s csv-outputs=%s containers-with-output=%s\n' \
    "podman-multi" "${pmulti_completed:-?}" "${pmulti_failed:-?}" \
    "$pmulti_csv_total" "$pmulti_sec_with_csv"

  log
  log "## artifacts"
  log "   logs: $LOG_ROOT"
  log "   outs: $TEST_ROOT/out-*"
  log "   exit: $rc_overall"
  return $rc_overall
}

main "$@"
