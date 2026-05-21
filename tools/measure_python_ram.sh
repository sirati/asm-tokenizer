#!/usr/bin/env bash
# Measure peak + sampled-avg RAM of an arbitrary python invocation via cgroup v2.
#
# Usage:
#   tools/measure_python_ram.sh [-o FILE | --output FILE] [python args...]
#
# Examples:
#   tools/measure_python_ram.sh -m pytest tests/test_foo.py        # JSON → stdout
#   tools/measure_python_ram.sh tools/run_stage3.py                # JSON → stdout
#   tools/measure_python_ram.sh -o /tmp/r.jsonl tools/run_stage3.py  # appended
#
# Output: one JSON line (appended) with ``peak_bytes`` (cgroup-kernel-
# tracked, exact), ``avg_bytes`` (20 Hz sampled mean of memory.current),
# ``samples``, ``wall_s``, ``exit``.
#
# Mechanism: wraps ``python "$@"`` in a transient ``systemd-run --user``
# service unit with ``MemoryAccounting=yes``.  Reads ``memory.peak`` from
# the cgroup BEFORE the unit exits — no ``RemainAfterExit``, no lingering
# systemd state.  The wrapped process's exit code is preserved.
#
# Requires: systemd (user instance) + cgroup v2 with ``memory.peak``.
#
# Self-recursive: the outer driver launches systemd, the inner copy (gated
# by ``_PYTHON_RAM_INNER=1``) actually runs python and reads the cgroup.
#
# Origin: adapted from ml-project's tools/measure_pytest_ram.sh
# (worktrees/bump-vram-rebased) by generalising the inner invocation
# from ``python -m pytest`` to ``python``.
set -u

if [ "${_PYTHON_RAM_INNER:-0}" = "1" ]; then
    # Inner — runs INSIDE the systemd unit.
    CG_PATH="/sys/fs/cgroup$(awk -F: '$1==0{print $3}' /proc/self/cgroup)"
    SAMPLE_FILE=$(mktemp)
    trap 'rm -f "$SAMPLE_FILE"' EXIT

    # 20 Hz sampler — sharpens avg; peak is kernel-tracked regardless of cadence.
    (while [ -e "$SAMPLE_FILE" ]; do
       cat "$CG_PATH/memory.current" >> "$SAMPLE_FILE" 2>/dev/null || break
       sleep 0.05
     done) &
    SAMPLER_PID=$!

    START=$(date +%s.%N)
    python "$@"
    PYTEST_EXIT=$?
    END=$(date +%s.%N)

    kill "$SAMPLER_PID" 2>/dev/null
    wait "$SAMPLER_PID" 2>/dev/null || true

    PEAK=$(cat "$CG_PATH/memory.peak")
    AVG=$(awk '{sum+=$1; n++} END {if(n>0) print int(sum/n); else print 0}' "$SAMPLE_FILE")
    NSAMPLES=$(wc -l < "$SAMPLE_FILE")
    WALL=$(awk -v s="$START" -v e="$END" 'BEGIN {printf "%.2f", e-s}')

    printf '{"args": %q, "peak_bytes": %s, "avg_bytes": %s, "samples": %d, "wall_s": %s, "exit": %d}\n' \
           "$*" "$PEAK" "$AVG" "$NSAMPLES" "$WALL" "$PYTEST_EXIT" \
           >> "${RESULTS_FILE:-/dev/stdout}"
    exit $PYTEST_EXIT
fi

# Outer — parse our own option, then launch the systemd unit calling
# THIS SAME SCRIPT with _PYTEST_RAM_INNER=1.  Inheriting PATH +
# PYTHONPATH + the current working directory is required under nix-
# shell; without --setenv=PATH the inner runs against the system's
# empty PATH and python is not found.
#
# We consume ``-o`` / ``--output`` ourselves; everything else is
# forwarded verbatim to pytest.  Stop parsing at the first non-option
# token (test path or pytest flag) or at an explicit ``--``.
RESULTS_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        -o|--output)
            RESULTS_FILE="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

# Without -o, capture to a temp file and print to the outer stdout
# after the unit exits.  We cannot use ``/dev/stdout`` as the inner's
# RESULTS_FILE because the systemd unit's stdout is redirected to
# ``$LOG`` (which carries pytest's chatter); the structured JSON would
# end up mixed in with pytest output on stderr.
PRINT_AFTER_RUN=0
if [ -z "$RESULTS_FILE" ]; then
    RESULTS_FILE=$(mktemp)
    PRINT_AFTER_RUN=1
fi

UNIT_NAME="python-ram-$(date +%s%N)"
LOG=$(mktemp)
SCRIPT_REALPATH=$(realpath "$0")
BASH_REALPATH=$(realpath "$(command -v bash)")

systemd-run --user --quiet --wait \
    --unit="$UNIT_NAME" \
    --working-directory="$PWD" \
    -p MemoryAccounting=yes \
    -p StandardOutput=file:"$LOG" \
    -p StandardError=append:"$LOG" \
    --setenv=PATH="$PATH" \
    --setenv=PYTHONPATH="${PYTHONPATH:-}" \
    --setenv=RESULTS_FILE="$RESULTS_FILE" \
    --setenv=_PYTHON_RAM_INNER=1 \
    -- "$BASH_REALPATH" "$SCRIPT_REALPATH" "$@"
EXIT=$?

# Surface pytest's stdout/stderr to the outer stderr for debugging.
cat "$LOG" >&2
rm -f "$LOG"

if [ "$PRINT_AFTER_RUN" = "1" ]; then
    cat "$RESULTS_FILE"
    rm -f "$RESULTS_FILE"
fi
exit $EXIT
