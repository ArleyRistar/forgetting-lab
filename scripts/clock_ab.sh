#!/usr/bin/env bash
# Clock-cap A/B — LAB-NOTES open item 2.
#
# Hypothesis: a LOWER SM clock cap raises *average* throughput by preventing the
# boost -> overheat -> hard-throttle oscillation seen during bring-up.
#
# Design notes:
#  - 150 steps/arm, not the 50 the open item proposed: steady state takes 10+
#    minutes on this chassis, so 50 steps would measure mostly the cold regime.
#  - Cooldown to <=45 C between arms so arm 2 does not start heat-soaked.
#  - Uses mem_probe.py because it is the only full-fine-tune entrypoint with
#    eval and checkpointing disabled; train.py would insert an eval at step 100
#    and pollute the step timing. Full FT is also the heavier thermal load and
#    is what phase 1c actually runs.
set -uo pipefail
cd "$HOME/forgetting-lab" || exit 1
STEPS=${STEPS:-150}

cooldown() {
  local t deadline=$((SECONDS + 900))
  while [ $SECONDS -lt $deadline ]; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    [ "$t" -le 45 ] && break
    sleep 20
  done
  echo "== cooled to ${t}C =="
}

run_arm() {
  local mhz=$1 tag=$2
  echo "== arm $tag: capping at ${mhz} MHz at $(date -Is) =="
  sudo nvidia-smi -lgc "$mhz" >/dev/null 2>&1 || echo "!! failed to set ${mhz}"
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader
  : > "/tmp/ab-$tag.csv"
  ( while true; do
      nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu \
        --format=csv,noheader,nounits >> "/tmp/ab-$tag.csv"
      sleep 10
    done ) & local sampler=$!
  uv run scripts/mem_probe.py --max-steps "$STEPS" --dtype bfloat16 --optim adamw_torch \
    > "/tmp/ab-$tag.log" 2>&1
  kill $sampler 2>/dev/null
  echo "== arm $tag done at $(date -Is) =="
}

cooldown
run_arm 1200 a1200
cooldown
run_arm 1000 a1000

# restore the standing cap
sudo nvidia-smi -lgc 1200 >/dev/null 2>&1
echo "== restored 1200 MHz cap =="
echo DONE > /tmp/ab-done
