#!/usr/bin/env bash
# Run several configs sequentially, one GPU job at a time.
#
# The lab box has one card, and CLAUDE.md's rule is absolute: never two GPU jobs
# at once, because it corrupts the timing and thermal profile of whatever was
# already running. So this waits for any in-flight run to exit before starting,
# and never overlaps its own runs either.
#
# Usage: tmux new -d -s queue 'scripts/run_queue.sh configs/a.yaml configs/b.yaml'
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MARKER=${MARKER:-/tmp/queue-done}
rm -f "$MARKER"

# Wait for a run started outside this script (matches `python -m flab.sequential`).
while pgrep -f "flab\.sequential" >/dev/null 2>&1; do
  echo "== $(date -Is) waiting for the in-flight run to finish =="
  sleep 30
done

rc_all=0
for cfg in "$@"; do
  echo "== $(date -Is) starting $cfg =="
  scripts/run_sequential.sh "$cfg"
  rc=$?
  echo "== $(date -Is) finished $cfg rc=$rc =="
  [ "$rc" -ne 0 ] && rc_all=$rc
  # A failure in one config does not stop the queue: the remaining runs are
  # independent, and losing four good runs to one bad one helps nobody. The
  # per-run marker records each outcome.
done

# Marker means *finished*, not *worked* — same convention as everywhere else.
printf 'worst_rc=%s configs=%s finished=%s\n' "$rc_all" "$#" "$(date -Is)" > "$MARKER"
echo "== $(date -Is) queue done, worst rc=$rc_all =="
exit "$rc_all"
