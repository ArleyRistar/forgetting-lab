#!/usr/bin/env bash
# Supervise a sequential run with bounded auto-retry (phase-1a task 5).
#
# Spec 6.1a asks for crash-resume "from day one": at 5-10 h/week of human
# attention, a 3 a.m. OOM must cost minutes, not a calendar day. The harness
# handles *where* to resume; this handles *getting restarted at all*.
#
# Usage: tmux new -As seq 'scripts/run_sequential.sh configs/dev-3stage.yaml'
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

CONFIG=${1:?usage: run_sequential.sh <config.yaml> [run-dir]}
RUN_NAME=$(python3 -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$CONFIG")
RUN_DIR=${2:-outputs/runs/$RUN_NAME}
MAX_RETRIES=${MAX_RETRIES:-3}
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

# Stamp logs per *invocation*, not just per attempt. `attempt` resets to 0 every
# time the supervisor starts, so plain attempt-N.log meant a restart silently
# overwrote the log of the crash that caused it — losing the one file you most
# want in a post-mortem. Found during the 2026-08-08 shakedown resume test.
STAMP=$(date +%Y%m%d-%H%M%S)

attempt=0
rc=1
while [ "$attempt" -le "$MAX_RETRIES" ]; do
  log="$LOG_DIR/$STAMP-attempt-$attempt.log"
  echo "== attempt $attempt/$MAX_RETRIES at $(date -Is) -> $log =="
  uv run python -m flab.sequential --config "$CONFIG" --run-dir "$RUN_DIR" > "$log" 2>&1
  rc=$?
  ln -sfn "$(basename "$log")" "$LOG_DIR/latest.log"

  if [ "$rc" -eq 0 ]; then
    echo "== finished cleanly on attempt $attempt at $(date -Is) =="
    break
  fi

  # An OOM is not retryable at the same batch size. Retrying it three times
  # just burns an hour and heat-soaks the chassis for whatever runs next, and
  # the chassis takes 10+ minutes to recover (LAB-NOTES). Stop and say so.
  if grep -qE "CUDA out of memory|torch.OutOfMemoryError|CUDA error: out of memory" "$log"; then
    echo "!! OOM on attempt $attempt - NOT retrying; the batch size must change first."
    echo "!! Last lines:"; tail -5 "$log"
    break
  fi

  # A config mismatch is a human error, not a transient one; retrying cannot
  # fix it and the message explains what to do.
  if grep -q "ConfigMismatch" "$log"; then
    echo "!! Config does not match this run directory - NOT retrying."
    grep -A4 ConfigMismatch "$log" | head -8
    break
  fi

  echo "!! attempt $attempt failed with rc=$rc; retrying"
  tail -3 "$log"
  attempt=$((attempt + 1))
done

# The marker means *finished*, not *worked* - hence recording rc in it rather
# than only writing it on success. A watcher must be able to tell the two apart
# without parsing logs. Appended, not overwritten, so a restart's outcome does
# not erase the record of the crash before it.
printf 'rc=%s attempts=%s finished=%s log=%s\n' \
  "$rc" "$attempt" "$(date -Is)" "$STAMP" >> "$RUN_DIR/SUPERVISOR-DONE"
echo "== supervisor done rc=$rc after $((attempt + 1)) attempt(s) =="
exit "$rc"
