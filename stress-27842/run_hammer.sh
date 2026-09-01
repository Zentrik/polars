#!/usr/bin/env bash
# Cycle the executor-hammer through modes/thread-counts. Each invocation is a
# fresh process (executor thread count is fixed per process), so a panic in one
# config is captured and the campaign continues to the next.
set -u
BIN="$1"; LOG="$2"; ROUND_SECS="${3:-360}"
export RUST_BACKTRACE=full
ulimit -c unlimited 2>/dev/null || true

run() {
  local mode="$1" threads="$2" secs="$3"
  echo "=== $(date +%H:%M:%S) START mode=$mode threads=$threads secs=$secs ===" >>"$LOG"
  "$BIN" --mode "$mode" --threads "$threads" --secs "$secs" >>"$LOG" 2>&1
  local rc=$?
  echo "=== $(date +%H:%M:%S) END   mode=$mode threads=$threads rc=$rc ===" >>"$LOG"
  if [ $rc -ne 0 ]; then
    echo "!!!! NONZERO EXIT rc=$rc mode=$mode threads=$threads (see backtrace above) !!!!" >>"$LOG"
  fi
}

# Order: highest-signal configs first. scope_churn = cancel-vs-steal race;
# fanout = buffer grow/shrink churn; both at heavy oversubscription.
run scope_churn 256 "$ROUND_SECS"
run fanout      256 "$ROUND_SECS"
run mixed       128 "$ROUND_SECS"
run fanout      128 "$ROUND_SECS"
run scope_churn 64  "$ROUND_SECS"
run yield_storm 256 "$ROUND_SECS"
run foreign     64  "$ROUND_SECS"
run mixed       256 "$ROUND_SECS"
echo "=== $(date +%H:%M:%S) HAMMER CAMPAIGN COMPLETE ===" >>"$LOG"
