#!/usr/bin/env bash
# Build + run the executor-hammer under ThreadSanitizer to catch data races in
# the async executor *before* they corrupt the deque (rather than waiting for
# the corruption to manifest). Requires nightly + rust-src (-Zbuild-std).
#
# NB: crossbeam-epoch's reclamation and SeqCst-fence patterns can produce TSan
# reports that are not true bugs; triage any hits against known crossbeam noise
# and focus on races that touch the polars async_executor code itself.
set -eu
HAMMER_DIR="$(cd "$(dirname "$0")/executor-hammer" && pwd)"
TARGET="${CARGO_TARGET_DIR:?set CARGO_TARGET_DIR}"
LOG="${1:-/tmp/tsan.log}"

cd "$HAMMER_DIR"
export RUSTFLAGS="-Zsanitizer=thread -Cdebuginfo=2"
export RUST_BACKTRACE=1
# TSan hates the SIGABRT-on-report default racing with our threads; keep halt.
export TSAN_OPTIONS="halt_on_error=1 second_deadlock_stack=1 history_size=4"

echo "building TSan hammer ..." | tee "$LOG"
# Use the repo-pinned nightly (rust-toolchain.toml) which has rust-src; do NOT
# pass +nightly, which selects the generic nightly channel that lacks it.
cargo build -Z build-std \
  --target x86_64-unknown-linux-gnu --release 2>&1 | tail -5 | tee -a "$LOG"

BIN="$TARGET/x86_64-unknown-linux-gnu/release/executor-hammer"
echo "running TSan hammer (shorter, TSan is ~10-20x slower) ..." | tee -a "$LOG"
for spec in "scope_churn 32 90" "fanout 32 90" "mixed 16 90"; do
  set -- $spec
  echo "=== TSAN mode=$1 threads=$2 secs=$3 ===" | tee -a "$LOG"
  "$BIN" --mode "$1" --threads "$2" --secs "$3" 2>&1 | tee -a "$LOG" || {
    echo "!!!! TSan reported (or hammer exited nonzero) for mode=$1 !!!!" | tee -a "$LOG"
  }
done
echo "=== TSAN CAMPAIGN COMPLETE ===" | tee -a "$LOG"
