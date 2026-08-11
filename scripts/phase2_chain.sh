#!/usr/bin/env bash
# Phase-2 chain: ONE process per (arm, pair, seed), predictors+prune after each.
#
# Memory grows across model loads inside a single process: a 56-run invocation
# OOM'd at run ~21 even with per-arm micro-batch and expandable_segments, while a
# 28-run one completed. Rather than chase the leak, each triple gets a fresh
# process (7 model loads) and the OS reclaims everything on exit.
#
# Never uses pgrep: an earlier chain waited on `pgrep -f scripts/phase2.py`, which
# matched a lingering shell of the agent's own tooling and hung for ~4h.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/arley/forgetting-lab
log() { echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p outputs/phase2
rm -f outputs/phase2/CHAIN-FAILED outputs/phase2/CHAIN-COMPLETE
# truncate: an append-mode log let a stale OOM traceback be reported
# as the cause of a failure that was actually a missing checkpoint.
: > outputs/phase2-chain.log

for seed in 1 2; do
  for arm in ternary float; do
    for pair in conflict disjoint; do
      label="$arm/$pair/s$seed"
      log "START $label"
      if ! uv run scripts/phase2.py --arms "$arm" --pairs "$pair" --seeds "$seed" \
           >> outputs/phase2-chain.log 2>&1; then
        log "FAIL $label — chain stops"; touch outputs/phase2/CHAIN-FAILED; exit 1
      fi
      if ! uv run scripts/phase2_flips.py --prune >> outputs/phase2-chain.log 2>&1; then
        log "FAIL predictors after $label"; touch outputs/phase2/CHAIN-FAILED; exit 1
      fi
      log "OK    $label  ($(python3 -c "import json;print(len(json.load(open('outputs/phase2/predictors.json'))))" 2>/dev/null) predictors, $(df -h /home | tail -1 | awk '{print $4}') free)"
    done
  done
done
log "complete"
touch outputs/phase2/CHAIN-COMPLETE
