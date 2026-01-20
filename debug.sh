#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <timeout_seconds>s <tail_lines> [python args...]" >&2
  exit 1
fi

TIMEOUT="$1"
TAIL_LINES="$2"
shift 2

# Enforce the "<number>s" format (e.g. 10s, 15s)
if [[ ! "$TIMEOUT" =~ ^[0-9]+s$ ]]; then
  echo "Error: timeout must be specified as seconds with 's' suffix (e.g. 10s, 15s)" >&2
  exit 1
fi

# timeout --foreground -k "$TIMEOUT" "$TIMEOUT" \
#   python -m dynamic_batch --raw-logs --debugs --debug --simulate-errors 40 --platform x86 --compiler clang "$@" 2>&1 | tail -n "$TAIL_LINES"

# TAIL is disabled for now - its better that way - do not add a tail yourself

timeout --foreground -k "$TIMEOUT" "$TIMEOUT" \
  python -m dynamic_batch --raw-logs --debugs --debug --simulate-errors 40 --platform x86 --compiler clang "$@" 2>&1
